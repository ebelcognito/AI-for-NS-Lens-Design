"""Bounded geometric least-squares refinement after independent CMA discovery.

This is a separate hybrid arm with a stricter merit function, not extra CMA results.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import least_squares
from . import model
from .search import serial, write_json


def sampled_volume(genes, aperture_radii=None):
    """Cheap fixed-grid optical-surface AABB, not mechanical package volume."""
    m=model.decode(genes, aperture_radii=aperture_radii)
    grid=np.linspace(-1,1,9); x,y=np.meshgrid(grid,grid); keep=x*x+y*y<=1
    points=[np.array([[0,-5,0],[0,5,0],m['image']])]
    for surface in m['surfaces']:
        xx=x[keep]*surface['aperture']; yy=y[keep]*surface['aperture']
        zz=model.sag(xx,yy,surface['coeff'])[0]
        points.append(np.column_stack([xx,yy,zz])@surface['basis'].T+surface['vertex'])
    p=np.concatenate(points)
    return float(np.prod(np.max(p,axis=0)-np.min(p,axis=0))/1000)


def run_one(config):
    source=Path(config['source']); original=json.loads(source.read_text())
    seed=original['seed']; out=Path(config['out'])/f'seed_{seed}'; out.mkdir()
    genes=np.asarray(original['genes']); before=model.evaluate(genes,fields=config['fields'],waves=config['waves'],pupils=config['pupils'],aperture_radii=config['aperture_radii'])
    cent=np.asarray(before['centroids_mm'])
    dispersion_sign=float(np.sign(np.mean(cent[:,-1,0]-cent[:,0,0]))) or 1.
    slit_sign=float(np.sign(np.mean(cent[-1,:,1]-cent[0,:,1]))) or 1.
    calls=0; best=None; started=time.monotonic(); history=[]
    pupil_xy=None
    if config['rim_points']:
        u=np.linspace(-1,1,config['pupils']); xx,yy=np.meshgrid(u,u); keep=xx*xx+yy*yy<1-1e-10
        angle=np.arange(config['rim_points'])*2*np.pi/config['rim_points']
        pupil_xy=np.vstack([np.column_stack([xx[keep],yy[keep]]),np.column_stack([np.cos(angle),np.sin(angle)])])
    before=model.evaluate(genes,fields=config['fields'],waves=config['waves'],pupils=config['pupils'],pupil_xy=pupil_xy,aperture_radii=config['aperture_radii'])
    def fun(g):
        nonlocal calls,best
        r=model.residuals(g,fields=config['fields'],waves=config['waves'],pupils=config['pupils'],
                          dispersion_sign=dispersion_sign,slit_sign=slit_sign, mapping_weight=config['mapping_weight'], pupil_xy=pupil_xy,aperture_radii=config['aperture_radii'],curvature_weight=config['curvature_weight'])
        if config['volume_weight']:
            r=np.r_[r,config['volume_weight']*max(sampled_volume(g,config['aperture_radii'])/config['volume_target']-1,0)]
        calls+=1; score=float(r@r)
        if best is None or score<best['residual_sum_squares']:
            best=dict(genes=np.array(g).copy(),residual_sum_squares=score,calls=calls)
        if calls%200==0:
            row=dict(calls=calls,elapsed_seconds=time.monotonic()-started,best_sum_squares=best['residual_sum_squares'])
            history.append(row); write_json(out/'checkpoint.json',dict(seed=seed,**{**best,**row}))
            print(json.dumps(dict(seed=seed,**row)),flush=True)
        return r
    initial=fun(genes)
    opt=least_squares(fun,np.clip(genes,-1+1e-10,1-1e-10),bounds=(-1,1),
                      method='trf',jac='2-point',diff_step=1e-5,x_scale='jac',
                      ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=config['max_nfev'])
    # Accept solver final iterate, never a numerical Jacobian probe.
    final=opt.x
    audit_args=dict(fields=config['audit_fields'],waves=config['audit_waves'],pupils=config['audit_pupils'],aperture_radii=config['aperture_radii'])
    result=dict(seed=seed,source=str(source),configuration=config,genes=final,decoded=model.decode(final,aperture_radii=config['aperture_radii']),
                residual_calls=calls,scipy_nfev=opt.nfev,scipy_njev=opt.njev,solver_status=int(opt.status),solver_message=str(opt.message),
                initial_sum_squares=float(initial@initial),final_sum_squares=float(2*opt.cost),
                training_pupil_xy=pupil_xy,elapsed_seconds=time.monotonic()-started,dispersion_sign=dispersion_sign,slit_sign=slit_sign,
                before_training=before,training=model.evaluate(final,fields=config['fields'],waves=config['waves'],pupils=config['pupils'],pupil_xy=pupil_xy,aperture_radii=config['aperture_radii']),
                before_audit=model.evaluate(genes,**audit_args),audit=model.evaluate(final,**audit_args),
                geometry=model.geometry_metrics(final,aperture_radii=config['aperture_radii']),initial_geometry=model.geometry_metrics(genes,aperture_radii=config['aperture_radii']),history=history)
    write_json(out/'result.json',result)
    print(json.dumps(serial(dict(seed=seed,complete=True,residual_calls=calls,audit=result['audit'],geometry=result['geometry']))),flush=True)
    return serial(result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[102,103])
    p.add_argument('--max-nfev',type=int,default=120)
    p.add_argument('--fields',type=int,default=3)
    p.add_argument('--waves',type=int,default=5)
    p.add_argument('--pupils',type=int,default=7)
    p.add_argument('--audit-fields',type=int,default=8)
    p.add_argument('--audit-waves',type=int,default=8)
    p.add_argument('--audit-pupils',type=int,default=10)
    p.add_argument('--volume-target',type=float,default=100.)
    p.add_argument('--volume-weight',type=float,default=.1)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--mapping-weight',type=float,default=1.)
    p.add_argument('--rim-points',type=int,default=0)
    p.add_argument('--aperture-radii',type=float,nargs=3,default=None)
    p.add_argument('--curvature-weight',type=float,default=0.)
    a=p.parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'configuration.json',vars(a))
    shutil.copyfile(Path(model.__file__),out/'model_snapshot.py')
    shutil.copyfile(Path(__file__),out/'polish_snapshot.py')
    configs=[dict(vars(a),source=str(Path(a.source)/f'seed_{seed}'/'result.json')) for seed in a.seeds]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=min(a.workers,len(configs))) as pool:
        results=list(pool.map(run_one,configs))
    write_json(out/'summary.json',dict(configuration=vars(a),wall_seconds=time.monotonic()-started,
        interpretation='Hybrid geometric least-squares after CMA. Different stricter merit; volume is a soft sampled optical-surface AABB penalty, not a certified package constraint.',
        runs=[{k:r[k] for k in ['seed','residual_calls','scipy_nfev','solver_status','initial_sum_squares','final_sum_squares','elapsed_seconds','before_audit','audit','geometry']} for r in results]))

if __name__=='__main__':
    main()
