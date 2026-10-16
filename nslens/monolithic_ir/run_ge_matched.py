"""Durable three-seed CMA + least-squares run of the published Ge specifications."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from .search import _write

TRAIN=dict(fields=(5,3),waves=3,pupils=5)
AUDIT=dict(fields=7,waves=7,pupils=9)

def worker(job):
    from . import search, model_ge_matched
    from . import polish
    seed,root,base,full,nfev=job
    root=Path(root); search.model=model_ge_matched; search.TRAIN=TRAIN; search.AUDIT=AUDIT
    polish.TRAIN=TRAIN; polish.AUDIT=AUDIT
    search.run(root/'search'/f'seed{seed}',seed=seed,base_evals=base,freeform_evals=full)
    source=root/'search'/f'seed{seed}'/'result.json'
    polish.run(source,root/'polish'/f'seed{seed}',max_nfev=nfev)
    return str(root/'polish'/f'seed{seed}'/'result.json')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--base-evals',type=int,default=1200);p.add_argument('--freeform-evals',type=int,default=3600)
    p.add_argument('--max-nfev',type=int,default=150);p.add_argument('--seeds',type=int,nargs='+',default=[401,402,403])
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):_write(root/'status.json',dict(stage=stage,updated_unix=time.time(),**extra))
    config=dict(material='Ge',f_number=1.4,pupil_diameter_mm=8.4,inferred_focal_length_mm=11.76,
        field_deg=[37,25],band_um=[8,12],internal_geometric_chief_path_mm=58,
        surface_classes=['rotational_asphere','XY_freeform','rotational_asphere','rotational_asphere'],
        variables=[17,35],training=TRAIN,audit=AUDIT,seeds=a.seeds,base_evals=a.base_evals,
        freeform_evals=a.freeform_evals,least_squares_max_nfev=a.max_nfev,
        source='https://www.nature.com/articles/s41377-021-00538-1',
        limitations=['Not the unpublished original prescription or a reproduction of the Fermat solver.',
         'Ge index uses Cauchy interpolation of paper endpoints 4.0067 at 8 um and 4.0029 at 12 um, not its undisclosed catalogue.',
         '15 rectangular training fields; exact paper field placement is not known.',
         '3 aspheres use local radial polynomial sag through sixth order; geometry family and parameter limits are our assumptions.',
         '80 x 32 mm auxiliary body footprint is a computational enclosure, not a published envelope.',
         '17 um is only a residual scaling/reference box, not a detector specification for this paper.',
         'Geometric spots are not a diffraction-limited or MTF validation.'])
    _write(root/'configuration.json',config)
    try:
        status('CMA_then_polishing',seeds=a.seeds)
        with ProcessPoolExecutor(max_workers=min(a.workers,len(a.seeds))) as pool:
            results=list(pool.map(worker,[(seed,str(root),a.base_evals,a.freeform_evals,a.max_nfev) for seed in a.seeds]))
        winner=min(results,key=lambda f:json.loads(Path(f).read_text())['training']['loss'])
        data=json.loads(Path(winner).read_text());_write(root/'selected.json',data)
        seed=data['seed'];trajectory=[dict(stage='initial',genes=[0.]*35,evaluations=0)]
        trajectory+=json.loads((root/'search'/f'seed{seed}'/'trajectory.json').read_text())
        trajectory += [dict(row,stage='least_squares') for row in json.loads((Path(winner).parent/'trajectory.json').read_text())]
        _write(root/'trajectory.json',trajectory)
        status('auditing',selected_source=winner)
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.audit_ge_matched','--source',winner,'--out',str(root/'audit.json')],check=True)
        status('rendering',selected_source=winner)
        subprocess.run([sys.executable,'-m','nslens.monolithic_ir.render_ge_matched','--source',winner,'--out',str(root/'visualizations'),'--trajectory',str(root/'trajectory.json')],check=True)
        status('complete',selected_source=winner,training=data['training'],audit_file=str(root/'audit.json'),
               note='Inspect audit gates; completion does not imply success or superiority over the published method.')
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
