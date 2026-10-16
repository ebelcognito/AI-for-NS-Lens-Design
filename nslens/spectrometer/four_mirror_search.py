"""Reproducible fold-only packaging search; existing imaging prescription is fixed.

Envelopes include sampled finite optical surfaces, the slit and full detector.
Clearance is a sampled diagnostic, not a manufacturing clearance certificate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
from . import model
from .search import write_json, serial


def surface_cloud(surface, samples=11):
    u=np.linspace(-1,1,samples); x,y=np.meshgrid(u,u); keep=x*x+y*y<=1+1e-12
    angle=np.linspace(0,2*np.pi,96,endpoint=False)
    x=np.r_[x[keep],np.cos(angle)]*surface['aperture']
    y=np.r_[y[keep],np.sin(angle)]*surface['aperture']
    z=model.sag(x,y,surface['coeff'])[0]
    return np.column_stack((x,y,z))@surface['basis'].T+surface['vertex']


def package_cloud(layout, samples=11):
    clouds=[surface_cloud(s,samples) for s in layout['surfaces']]
    bounds=np.asarray(layout['detector_bounds'])
    uv=np.array([[x,y] for x in bounds[:,0] for y in bounds[:,1]])
    corners=layout['image']+uv[:,0,None]*layout['image_x']+uv[:,1,None]*layout['image_y']
    slit=np.asarray(layout.get('slit_endpoints',[[0,-5,0],[0,5,0]]))
    return np.concatenate(clouds+[corners,slit])


def package_metrics(layout, angles=(0,0,0), samples=11):
    points=package_cloud(layout,samples)
    rotation=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
    q=points@rotation.T
    low=q.min(0); high=q.max(0); bbox=high-low
    return dict(bbox_mm=bbox,volume_cm3=float(np.prod(bbox)/1000),
                rotation_xyz_deg=np.asarray(angles),rotation_matrix=rotation,
                low_mm=low,high_mm=high,
                envelope_definition='Sampled full circular sag surfaces (including 96 rim points each), 10 mm slit, full detector rectangle; excludes substrates and mounts.')


def optimized_package(layout, seed=405, iterations=45):
    points=package_cloud(layout)
    def objective(a):
        q=points@Rotation.from_euler('xyz',a,degrees=True).as_matrix().T
        return float(np.prod(np.ptp(q,axis=0))/1000)
    opt=differential_evolution(objective,[(-90,90)]*3,seed=seed,maxiter=iterations,popsize=8,tol=1e-7,polish=False)
    local=minimize(objective,opt.x,method='Powell',bounds=[(-90,90)]*3,options={'maxiter':80,'maxfev':3000,'xtol':1e-7,'ftol':1e-9})
    a=local.x if local.fun<opt.fun else opt.x
    metric=package_metrics(layout,a)
    metric['orientation_evaluations']=int(opt.nfev+local.nfev)
    return metric


def path_screen(layout, max_rays=55):
    """Trace sampled known-path segments against every unintended finite surface."""
    paths=np.asarray(layout['nominal_paths'])
    if paths.shape[1]>max_rays:
        paths=paths[:,np.unique(np.linspace(0,paths.shape[1]-1,max_rays).astype(int))]
    blocked=np.zeros(paths.shape[1],bool)
    # For each path segment, destination surface is segment index except detector.
    for j in range(len(paths)-1):
        o=paths[j]; length=np.linalg.norm(paths[j+1]-o,axis=1)
        d=(paths[j+1]-o)/np.maximum(length[:,None],1e-30)
        for k,s in enumerate(layout['surfaces']):
            if k==j or k==j-1: continue
            _,_,distance,valid=model.intersect(o,d,s)
            blocked |= valid&(distance<length-1e-5)&(distance>1e-5)
    return float(blocked.mean())


def trace_summary(tr):
    good=tr['good']; xy=tr['xy']; n=good.sum(2)
    cent=np.sum(np.where(good[...,None],xy,0),axis=2)/np.maximum(n[...,None],1)
    distance=np.sum((xy-cent[:,:,None,:])**2,axis=-1)
    rms=np.sqrt(np.sum(np.where(good,distance,0),axis=2)/np.maximum(n,1))
    return dict(throughput=float(good.mean()),min_throughput=float((n/good.shape[2]).min()),
                obstruction_fraction=float(tr['obstructed'].mean()),rms_spot_mm=float(np.sqrt(np.mean(rms*rms))),
                max_spot_mm=float(rms.max()),centroids_mm=cent)


def run(args):
    from . import four_mirror as fm
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); baseline=fm.load_baseline(args.source,fields=5,waves=7,pupils=9)
    base_layout={**baseline['model'],'detector_bounds':baseline['detector_bounds']}
    baseline_geom=optimized_package(base_layout,args.seed)
    write_json(out/'configuration.json',vars(args))
    write_json(out/'baseline.json',dict(layout=base_layout,geometry=baseline_geom,training=trace_summary(baseline['trace'])))
    bounds=[(.33,.67),(60,120),(0,360),(.33,.68),(60,120),(0,360)]+[(-90,90)]*3
    calls=0;history=[];archive=[];stage='discovery'
    def objective(x):
        nonlocal calls,archive
        calls+=1
        layout=fm.build_layout(x[:6],baseline)
        package=package_metrics(layout,x[6:])
        screen=fm.screen_layout(layout,max_rays=31)
        obstruction=screen['obstruction_fraction']
        surface_geometry=fm.geometry_metrics(layout,samples=7)
        gap=min(surface_geometry['min_surface_gap_mm'],surface_geometry['min_detector_gap_mm'])
        score=package['volume_cm3']+1500*obstruction+100*max(2-gap,0)**2+1000*(not layout['nominal_segment_valid'])
        row=dict(params=np.array(x[:6]),rotation_xyz_deg=np.array(x[6:]),geometry=package,
                 sampled_gap_mm=gap,screen_obstruction_fraction=obstruction,
                 nominal_segment_valid=layout['nominal_segment_valid'],score=float(score),stage=stage,call=calls)
        if not archive or score<archive[-1]['score'] or len(archive)<32:
            # Retain separated fold solutions; orientation copies add little value.
            near=next((i for i,a in enumerate(archive) if np.linalg.norm((a['params']-x[:6])/np.array([.34,60,360,.35,60,360]))<.025),None)
            if near is None:archive.append(row)
            elif score<archive[near]['score']:archive[near]=row
            archive.sort(key=lambda a:a['score']);archive=archive[:32]
        if calls%200==0:
            progress=dict(calls=calls,elapsed_seconds=time.monotonic()-started,stage=stage,best=archive[0])
            history.append(progress);write_json(out/'checkpoint.json',progress)
            print(json.dumps(serial(dict(calls=calls,stage=stage,best_score=archive[0]['score'],best_volume_cm3=archive[0]['geometry']['volume_cm3'],elapsed_seconds=progress['elapsed_seconds']))),flush=True)
        return score
    # Planar initial candidates explicitly seed both handedness choices.
    initial=[]
    for a1 in (0,180):
        for a2 in (0,180):
            for t1 in (70,90,110):
                for t2 in (70,90,110):
                    x=np.array([.5,t1,a1,.5,t2,a2,*baseline_geom['rotation_xyz_deg']])
                    objective(x);initial.append(x)
    rng=np.random.default_rng(args.seed)
    initial.extend(rng.uniform(np.array(bounds)[:,0],np.array(bounds)[:,1],size=(72-len(initial),9)))
    opt=differential_evolution(objective,bounds,init=np.array(initial),seed=args.seed,maxiter=args.generations,tol=0,polish=False,updating='immediate')
    stage='local_refinement';discovery=list(archive)
    refinements=[]
    for row in discovery[:args.refine]:
        start=np.r_[row['params'],row['rotation_xyz_deg']]
        local=minimize(objective,start,method='Powell',bounds=bounds,options={'maxfev':args.refine_calls,'maxiter':30,'xtol':1e-4,'ftol':1e-6})
        refinements.append(dict(start_score=row['score'],final_score=float(local.fun),nfev=int(local.nfev),success=bool(local.success),message=str(local.message)))
    # Independently optimize orientation for finalists using the baseline procedure.
    audited=[]
    for row in archive[:args.finalists]:
        layout=fm.build_layout(row['params'],baseline)
        row={**row,'geometry':optimized_package(layout,args.seed),'surface_geometry':fm.geometry_metrics(layout,samples=25)}
        tr=fm.trace(layout,baseline,fields=8,waves=8,pupils=10)
        row['audit']=trace_summary(tr)
        row['passes_sampled_checks']=bool(row['audit']['min_throughput']>=1-1e-12 and min(row['surface_geometry']['min_surface_gap_mm'],row['surface_geometry']['min_detector_gap_mm'])>=2 and row['nominal_segment_valid'])
        row['reduction_vs_fair_baseline']=1-row['geometry']['volume_cm3']/baseline_geom['volume_cm3']
        audited.append(row)
    audited.sort(key=lambda r:(not r['passes_sampled_checks'],r['geometry']['volume_cm3']))
    best=audited[0];layout=fm.build_layout(best['params'],baseline)
    result=dict(configuration=vars(args),baseline=dict(geometry=baseline_geom,training=trace_summary(baseline['trace'])),
                best=best,layout={k:v for k,v in layout.items() if k not in ('nominal_paths','fold_fractions')},
                ranked_candidates=audited,discovery_archive=discovery,refinements=refinements,
                calls=calls,elapsed_seconds=time.monotonic()-started,history=history,
                limitations=['Imaging prescription frozen; ideal flat folds preserve imaging but can introduce clipping or obstruction.',
                            'Sampled surface-pair gaps are upper bounds, not clearance certificates.',
                            'No substrates, mounts, or detector body in this optical envelope.',
                            'Full detector rectangle follows baseline ray footprint plus 1 mm margin per edge.'])
    write_json(out/'result.json',result)
    np.savez_compressed(out/'best_paths.npz',paths=layout['nominal_paths'])
    print(json.dumps(serial(dict(complete=True,calls=calls,elapsed_seconds=result['elapsed_seconds'],best=best,baseline_geometry=baseline_geom))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='experiments/spectrometer_final_v1/seed_102/result.json')
    p.add_argument('--out',default='experiments/spectrometer_four_mirror_v1')
    p.add_argument('--seed',type=int,default=405)
    p.add_argument('--generations',type=int,default=25)
    p.add_argument('--refine',type=int,default=3)
    p.add_argument('--refine-calls',type=int,default=650)
    p.add_argument('--finalists',type=int,default=8)
    run(p.parse_args())

if __name__=='__main__':main()
