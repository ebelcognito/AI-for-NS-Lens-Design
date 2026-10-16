"""Population-only full-covariance CMA discovery for a two-flat-fold control.

No Powell, gradients, least squares, or local polishing. Existing powered optics
remain frozen. Feasibility scaffolds come only from unrefined random discovery.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from nslens.search.cma_me import CMAES
from . import four_mirror as fm
from .four_mirror_search import package_metrics, trace_summary
from .search import serial, write_json

BOUNDS=np.array([[.2,.85],[45,135],[0,360],[.2,.85],[45,135],[0,360],[-90,90],[-90,90],[-90,90]])
PARAMETERS=['fraction1','turn1_deg','azimuth1_deg','fraction2','turn2_deg','azimuth2_deg','box_x_deg','box_y_deg','box_z_deg']

def decode(x): return BOUNDS[:,0]+(np.asarray(x)+1)*.5*np.ptp(BOUNDS,axis=1)
def encode(x): return 2*(np.asarray(x)-BOUNDS[:,0])/np.ptp(BOUNDS,axis=1)-1

def retain(archive,row,capacity=32):
    near=next((i for i,a in enumerate(archive) if np.linalg.norm(a['genes'][:6]-row['genes'][:6])<.045),None)
    if near is None:archive.append(row)
    elif row['score']<archive[near]['score']:archive[near]=row
    archive.sort(key=lambda a:a['score']);del archive[capacity:]

def evaluate(x,baseline):
    physical=decode(x);layout=fm.build_layout(physical[:6],baseline)
    screen=fm.screen_layout(layout,max_rays=49)
    geometry=package_metrics(layout,physical[6:])
    surfaces=fm.geometry_metrics(layout,samples=13)
    gap=min(surfaces['min_surface_gap_mm'],surfaces['min_detector_gap_mm'])
    penalty=1500*screen['obstruction_fraction']+100*max(2-gap,0)**2+1000*(not layout['nominal_segment_valid'])
    return dict(genes=np.asarray(x).copy(),params=physical[:6],geometry=geometry,
                score=float(geometry['volume_cm3']+penalty),screen_obstruction_fraction=screen['obstruction_fraction'],
                sampled_gap_mm=gap,nominal_segment_valid=layout['nominal_segment_valid'])

def run_seed(config):
    started=time.monotonic(); seed=config['seed'];out=Path(config['out'])/f'seed_{seed}';out.mkdir()
    baseline=fm.load_baseline(config['source'],fields=5,waves=7,pupils=9)
    rng=np.random.default_rng(seed);archive=[];history=[];restarts=[];evaluations=0;stale=0
    scaffolds=[]
    for params in config['scaffolds']:
        x=encode([*params,0,0,0]);row=evaluate(x,baseline);row['origin']='unrefined_random_scaffold'
        scaffolds.append(row)
    emitter=None;best=None
    with (out/'populations.jsonl').open('w') as poplog,(out/'cma_states.jsonl').open('w') as statelog:
        while evaluations+config['popsize']<=config['budget']:
            if emitter is None or emitter.restart_needed or stale>=config['restart_generations']:
                if not restarts:
                    mean=rng.uniform(-.8,.8,9);sigma=.32;reason='independent_uniform_random_start'
                elif len(restarts)==1:
                    source=scaffolds[int(rng.integers(len(scaffolds)))];mean=source['genes'].copy();sigma=.20;reason='unrefined_random_scaffold_restart'
                elif archive and rng.random()<.8:
                    source=archive[int(rng.integers(min(6,len(archive))))];mean=source['genes'].copy();sigma=.16;reason='retained_candidate_restart'
                else:
                    mean=rng.uniform(-.8,.8,9);sigma=.30;reason='independent_uniform_random_restart'
                emitter_seed=int(rng.integers(2**31));emitter=CMAES(mean,sigma,popsize=config['popsize'],seed=emitter_seed,bounds=(-np.ones(9),np.ones(9)),boundary_handling='reflect')
                stale=0;restarts.append(dict(evaluations=evaluations,reason=reason,initial_mean=mean,sigma=sigma,emitter_seed=emitter_seed))
            samples=emitter.ask();previous=best['score'] if best else np.inf
            rows=[]
            for sample in samples:
                row=evaluate(sample,baseline);row['origin']='cma_population';row['evaluation']=evaluations+len(rows)+1
                rows.append(row);retain(archive,row)
                if best is None or row['score']<best['score']:best=row
            emitter.tell(np.argsort([r['score'] for r in rows]));evaluations+=len(rows)
            stale=stale+1 if best['score']>=previous-1e-9 else 0
            # Force broad retained-candidate restarts even under tiny improvements.
            if emitter.gen>=50:stale=config['restart_generations']
            record=dict(evaluations=evaluations,best_score=best['score'],best_volume_cm3=best['geometry']['volume_cm3'],
                        best_obstruction=best['screen_obstruction_fraction'],restarts=len(restarts),sigma=emitter.sigma,
                        generation=emitter.gen,elapsed_seconds=time.monotonic()-started)
            history.append(record)
            population=dict(evaluations=evaluations,genes=samples,scores=[r['score'] for r in rows],
                            volume_cm3=[r['geometry']['volume_cm3'] for r in rows],obstruction=[r['screen_obstruction_fraction'] for r in rows],
                            sampled_gap_mm=[r['sampled_gap_mm'] for r in rows])
            poplog.write(json.dumps(serial(population))+'\n');poplog.flush()
            state=dict(evaluations=evaluations,restart=len(restarts),state=emitter.state_snapshot())
            statelog.write(json.dumps(serial(state))+'\n');statelog.flush()
            write_json(out/'checkpoint.json',dict(seed=seed,**record,best=best,state=state))
            if evaluations%(10*config['popsize'])==0:print(json.dumps(serial(dict(seed=seed,**record))),flush=True)
    audited=[]
    for candidate in archive[:config['finalists']]:
        layout=fm.build_layout(candidate['params'],baseline)
        tr=fm.trace(layout,baseline,fields=8,waves=8,pupils=10)
        sg=fm.geometry_metrics(layout,samples=25);audit=trace_summary(tr)
        candidate={**candidate,'surface_geometry':sg,'audit':audit,
                   'passes_sampled_checks':bool(audit['min_throughput']>=1-1e-12 and min(sg['min_surface_gap_mm'],sg['min_detector_gap_mm'])>=2 and candidate['nominal_segment_valid'])}
        audited.append(candidate)
    audited.sort(key=lambda r:(not r['passes_sampled_checks'],r['geometry']['volume_cm3']))
    result=dict(seed=seed,configuration=config,evaluations=evaluations,scaffold_evaluations=len(scaffolds),finalist_audit_evaluations=len(audited),
                best=audited[0],ranked_candidates=audited,history=history,restarts=restarts,elapsed_seconds=time.monotonic()-started,
                scaffold_checks=scaffolds,algorithm='full_covariance_cma_es',local_refinement=False)
    write_json(out/'result.json',result)
    print(json.dumps(serial(dict(seed=seed,complete=True,evaluations=evaluations,best_volume_cm3=result['best']['geometry']['volume_cm3'],passes=result['best']['passes_sampled_checks']))),flush=True)
    return serial(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='experiments/spectrometer_four_mirror_cma_v1')
    p.add_argument('--source',default=fm.DEFAULT_BASELINE)
    p.add_argument('--seeds',type=int,nargs='+',default=[501,502,503])
    p.add_argument('--budget',type=int,default=3600);p.add_argument('--popsize',type=int,default=24)
    p.add_argument('--restart-generations',type=int,default=16)
    p.add_argument('--finalists',type=int,default=8);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fallback=json.loads(Path('experiments/spectrometer_four_mirror_fallback_v1/result.json').read_text())
    eligible=[r for r in fallback['verified'] if min(r['geometry']['min_surface_gap_mm'],r['geometry']['min_detector_gap_mm'])>=2 and np.all(np.asarray(r['params'])>=BOUNDS[:6,0]) and np.all(np.asarray(r['params'])<=BOUNDS[:6,1])]
    scaffolds=[r['params'] for r in eligible[:4]]
    if not scaffolds:raise RuntimeError('No permitted unrefined scaffold in specified bounds')
    baseline_saved=json.loads(Path('experiments/spectrometer_four_mirror_v1/baseline.json').read_text())
    write_json(out/'baseline.json',baseline_saved)
    config={**vars(args),'bounds':BOUNDS,'parameter_names':PARAMETERS,'scaffolds':scaffolds,
            'scaffold_source':'experiments/spectrometer_four_mirror_fallback_v1/result.json: verified unrefined random candidates',
            'bounds_rationale':'Broadens feasibility bounds to include independently sampled clear folds; geometry frozen, azimuth full circle; includes three package orientation angles.',
            'baseline_geometry':baseline_saved['geometry'],'local_refinement':False}
    write_json(out/'configuration.json',config)
    for name in ['four_mirror_cma.py','four_mirror.py','four_mirror_search.py','model.py']:
        shutil.copyfile(Path(__file__).parent/name,out/(name[:-3]+'_snapshot.py'))
    shutil.copyfile(Path('nslens/search/cma_me.py'),out/'cma_me_snapshot.py')
    jobs=[{**config,'seed':seed} for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run_seed,jobs))
    best_seed=min(results,key=lambda r:(not r['best']['passes_sampled_checks'],r['best']['geometry']['volume_cm3']))
    baseline=fm.load_baseline(args.source,fields=5,waves=7,pupils=9);layout=fm.build_layout(best_seed['best']['params'],baseline)
    best=best_seed['best'];best['reduction_vs_fair_baseline']=1-best['geometry']['volume_cm3']/baseline_saved['geometry']['volume_cm3']
    summary=dict(configuration=config,algorithm='full_covariance_cma_es',local_refinement=False,evaluations=sum(r['evaluations'] for r in results),
                 scaffold_evaluations=sum(r['scaffold_evaluations'] for r in results),audit_evaluations=sum(r['finalist_audit_evaluations'] for r in results),
                 elapsed_seconds=time.monotonic()-started,best_seed=best_seed['seed'],best=best,baseline={'geometry':baseline_saved['geometry']},
                 layout={k:v for k,v in layout.items() if k not in ('nominal_paths','fold_fractions')},
                 seeds=[dict(seed=r['seed'],best=r['best'],evaluations=r['evaluations'],restarts=r['restarts']) for r in results],
                 limitations=['Flat-fold control with fixed imaging prescription; no added powered mirror degrees of freedom.',
                             'Sampled gaps are not certified clearance bounds; envelopes exclude substrates and mounts.',
                             'Candidate orientation is a CMA parameter; no post-CMA local orientation or optical refinement.'])
    write_json(out/'result.json',summary);np.savez_compressed(out/'best_paths.npz',paths=layout['nominal_paths'])
    print(json.dumps(serial(dict(complete=True,evaluations=summary['evaluations'],best_seed=summary['best_seed'],best=best))),flush=True)

if __name__=='__main__':main()
