"""Optional bounded least-squares stage after completed powered CMA refinement.

Accept only the terminal solver iterate after fresh all-ray/curvature/volume
checks. A rejected trial never replaces its CMA parent.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import least_squares
from . import powered_four_mirror as pm
from .search import write_json
from .four_mirror_audit import dense_pupil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--out',required=True)
    p.add_argument('--max-nfev',type=int,default=40);a=p.parse_args()
    original=json.loads(Path(a.source).read_text());g=np.asarray(original.get('genes',original['best']['genes']))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for module in [Path(__file__),Path(pm.__file__)]:shutil.copyfile(module,out/(module.stem+'_snapshot.py'))
    config=vars(a);write_json(out/'configuration.json',config)
    pupil=dense_pupil(4,24);train=dict(fields=5,waves=7,pupil_xy=pupil)
    before=pm.evaluate(g,**train);calls=0;started=time.monotonic()
    # Ensure least-squares cannot buy imaging quality by greatly growing package.
    volume_cap=before['volume_cm3']*1.01
    def fun(x):
        nonlocal calls
        residual=pm.residuals(x,**train)
        volume=pm.geometry_metrics(x)['volume_cm3']
        residual=np.r_[residual,.25*max(volume/volume_cap-1,0)]
        calls+=1
        if calls%200==0:
            write_json(out/'checkpoint.json',dict(residual_calls=calls,elapsed_seconds=time.monotonic()-started,sum_squares=float(residual@residual)))
        return residual
    initial=fun(g)
    opt=least_squares(fun,np.clip(g,-1+1e-10,1-1e-10),bounds=(-1,1),max_nfev=a.max_nfev,
                      method='trf',jac='2-point',diff_step=1e-5,x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)
    final=opt.x
    pupil=dense_pupil(7,48);audit_args=dict(fields=9,waves=11,pupil_xy=pupil)
    old=pm.evaluate(g,**audit_args);new=pm.evaluate(final,**audit_args)
    layout=pm.decode(final);curvature=pm.curvature_metrics(layout,radial_samples=65,azimuth_samples=360)
    accepted=bool(new['throughput']==1 and new['sampled_min_gap_mm']>=2 and
                  new['rms_spot_mm']<old['rms_spot_mm'] and new['volume_cm3']<=old['volume_cm3']*1.02 and
                  curvature['sign_violation']<=1e-10 and curvature['max_slope']<=.7 and curvature['max_abs_sag_mm']<=4)
    result=dict(configuration=config,source=a.source,genes=final,parent_genes=g,accepted=accepted,
                before_training=before,before_audit=old,audit=new,curvature=curvature,geometry=pm.geometry_metrics(final),layout=layout,
                residual_calls=calls,scipy_nfev=opt.nfev,scipy_njev=opt.njev,solver_message=opt.message,
                initial_sum_squares=float(initial@initial),final_sum_squares=float(2*opt.cost),elapsed_seconds=time.monotonic()-started,
                limitations=['Geometric imaging merit, not diffraction PSF or WFE optimization.','Independent roots and mechanical envelopes require separate validation.','Only terminal solver iterate considered.'])
    write_json(out/'result.json',result)
    print(json.dumps(dict(accepted=accepted,residual_calls=calls,before_rms_um=old['rms_spot_mm']*1000,after_rms_um=new['rms_spot_mm']*1000,volume_cm3=new['volume_cm3'])),flush=True)

if __name__=='__main__':main()
