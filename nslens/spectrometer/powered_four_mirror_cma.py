"""Staged population CMA discovery for four powered mirrors and a grating.

Coarse stage evolves geometry and quadratic powers. Fine stage releases all
freeform terms. No local/gradient solver is invoked by this module.
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
from .search import serial,write_json


def retain(archive,row,capacity=24):
    near=next((i for i,a in enumerate(archive) if np.linalg.norm(a['genes']-row['genes'])/np.sqrt(len(row['genes']))<.012),None)
    if near is None:archive.append(row)
    elif row['metrics']['loss']<archive[near]['metrics']['loss']:archive[near]=row
    archive.sort(key=lambda a:a['metrics']['loss']);del archive[capacity:]


def run_seed(config):
    from . import powered_four_mirror as pm
    seed=config['seed'];rng=np.random.default_rng(seed);started=time.monotonic();out=Path(config['out'])/f'seed_{seed}';out.mkdir()
    history=[];stages=[];total=0;best=None;initial=[]
    scaffold=np.asarray(config['initial_genes'],float)
    train=dict(fields=config['fields'],waves=config['waves'],pupils=config['pupils'])
    if config.get('rim_points',0):
        u=np.linspace(-1,1,config['pupils']);xx,yy=np.meshgrid(u,u);keep=xx*xx+yy*yy<1-1e-12
        a=np.arange(config['rim_points'])*2*np.pi/config['rim_points']
        train['pupil_xy']=np.vstack([np.column_stack([xx[keep],yy[keep]]),np.column_stack([np.cos(a),np.sin(a)])])
    initial_metrics=pm.evaluate(scaffold,**train)
    best=dict(genes=scaffold.copy(),metrics=initial_metrics,evaluation=1,stage='inherited_seed')
    total=1
    write_json(out/'inherited_seed.json',best)
    with (out/'populations.jsonl').open('w') as populations,(out/'cma_states.jsonl').open('w') as states:
        for stage,dim,budget in [('base',pm.N_BASE,config['base_evals']),('freeform',pm.N_TOTAL,config['freeform_evals'])]:
            archive=[];stage_count=0;stale=0;emitter=None;restarts=[];population_best=np.inf
            if best is not None:retain(archive,best)
            while stage_count+config['popsize']<=budget:
                if emitter is None or emitter.restart_needed or stale>=config['restart_generations']:
                    if not restarts:
                        parent=(best['genes'] if best is not None else scaffold)[:dim].copy()
                        sigma=config['base_sigma'] if stage=='base' else config['fine_sigma']
                        if stage=='base':parent=np.clip(parent+rng.normal(0,config['initial_perturbation'],dim),-.999,.999)
                        reason='perturbed_inherited_scaffold' if stage=='base' else 'coarse_stage_incumbent'
                    elif archive and rng.random()<.85:
                        parent=archive[int(rng.integers(min(4,len(archive))))]['genes'][:dim].copy()
                        sigma=config['base_sigma']*.7 if stage=='base' else config['fine_sigma']*.7;reason='retained_candidate_restart'
                    else:
                        parent=np.clip(scaffold[:dim]+rng.normal(0,.15,dim),-.999,.999);sigma=.06;reason='randomized_scaffold_restart'
                    cseed=int(rng.integers(2**31));emitter=CMAES(parent,sigma,popsize=config['popsize'],seed=cseed,bounds=(-np.ones(dim),np.ones(dim)),boundary_handling='reflect')
                    if config.get('sensitivity_steps') is not None:
                        steps=np.asarray(config['sensitivity_steps'][:dim],float)*(1. if stage=='base' else .5)
                        emitter.C=np.diag((steps/sigma)**2);emitter._update_eigen()
                    stale=0;restarts.append(dict(stage_evaluations=stage_count,reason=reason,mean=parent,sigma=sigma,emitter_seed=cseed,initial_covariance=emitter.C.copy()))
                samples=emitter.ask();previous=population_best;rows=[]
                for sample in samples:
                    genes=np.zeros(pm.N_TOTAL);genes[:dim]=sample
                    metrics=pm.evaluate(genes,**train)
                    if not np.isfinite(metrics['loss']):raise RuntimeError('Nonfinite optical objective')
                    row=dict(genes=genes,metrics=metrics,evaluation=total+len(rows)+1,stage=stage)
                    rows.append(row);retain(archive,row)
                    if len(initial)<config['popsize']:initial.append(row)
                    if best is None or metrics['loss']<best['metrics']['loss']:best=row
                emitter.tell(np.argsort([r['metrics']['loss'] for r in rows]));stage_count+=len(rows);total+=len(rows)
                population_best=min(population_best,min(r['metrics']['loss'] for r in rows))
                stale=stale+1 if population_best>=previous-1e-12 else 0
                rec=dict(stage=stage,evaluations=total,stage_evaluations=stage_count,restarts=len(restarts),sigma=emitter.sigma,population_best_loss=population_best,
                         best=best['metrics'],elapsed_seconds=time.monotonic()-started)
                history.append(rec)
                populations.write(json.dumps(serial(dict(stage=stage,evaluations=total,genes=samples,metrics=[r['metrics'] for r in rows])))+'\n');populations.flush()
                states.write(json.dumps(serial(dict(stage=stage,evaluations=total,restart=len(restarts),state=emitter.state_snapshot())))+'\n');states.flush()
                write_json(out/'checkpoint.json',dict(seed=seed,**rec,best_candidate=best,state=emitter.state_snapshot()))
                if len(history)%10==0:print(json.dumps(serial(dict(seed=seed,**rec))),flush=True)
            stages.append(dict(stage=stage,dimensions=dim,budget=budget,evaluations=stage_count,restarts=restarts,best=best,archive=archive))
    audit_sampling=dict(fields=config['audit_fields'],waves=config['audit_waves'],pupils=config['audit_pupils'])
    candidates=[]
    for row in stages[-1]['archive'][:config['finalists']]:
        candidates.append({**row,'audit':pm.evaluate(row['genes'],**audit_sampling),'geometry':pm.geometry_metrics(row['genes'])})
    result=dict(seed=seed,configuration=config,evaluations=total,stages=stages,history=history,initial_population=initial,
                best=best,inherited_seed_evaluations=1,population_evaluations=total-1,audit=pm.evaluate(best['genes'],**audit_sampling),geometry=pm.geometry_metrics(best['genes']),
                layout=pm.decode(best['genes']),ranked_candidates=candidates,elapsed_seconds=time.monotonic()-started,
                algorithm='staged_full_covariance_cma_es',local_refinement=False)
    write_json(out/'result.json',result)
    print(json.dumps(serial(dict(seed=seed,complete=True,evaluations=total,audit=result['audit']))),flush=True)
    return serial(result)


def main():
    from . import powered_four_mirror as pm
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='experiments/spectrometer_powered_four_mirror_cma_v1')
    p.add_argument('--seeds',type=int,nargs='+',default=[601,602,603]);p.add_argument('--initial-genes',required=True)
    p.add_argument('--base-evals',type=int,default=1200);p.add_argument('--freeform-evals',type=int,default=2400)
    p.add_argument('--popsize',type=int,default=24);p.add_argument('--restart-generations',type=int,default=18)
    p.add_argument('--base-sigma',type=float,default=.025);p.add_argument('--fine-sigma',type=float,default=.015)
    p.add_argument('--initial-perturbation',type=float,default=.005)
    p.add_argument('--sensitivity-covariance',default=None)
    p.add_argument('--rim-points',type=int,default=0)
    p.add_argument('--fields',type=int,default=3);p.add_argument('--waves',type=int,default=5);p.add_argument('--pupils',type=int,default=5)
    p.add_argument('--audit-fields',type=int,default=8);p.add_argument('--audit-waves',type=int,default=8);p.add_argument('--audit-pupils',type=int,default=10)
    p.add_argument('--finalists',type=int,default=4);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds):p.error('Seeds must be unique')
    if args.popsize<4 or min(args.base_evals,args.freeform_evals)<args.popsize:p.error('Each stage requires a full population')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    initial=json.loads(Path(args.initial_genes).read_text());initial=initial['genes'] if isinstance(initial,dict) else initial
    if len(initial)!=pm.N_TOTAL or not np.isfinite(initial).all() or np.any(np.abs(initial)>1):raise ValueError('Initial genes must be finite normalized N_TOTAL vector')
    shutil.copyfile(args.initial_genes,out/'initial_genes_source.json')
    config={**vars(args),'initial_genes':initial,'algorithm':'staged_full_covariance_cma_es','local_refinement':False}
    if args.sensitivity_covariance:
        sensitivity=json.loads(Path(args.sensitivity_covariance).read_text())
        steps=[float(np.clip(.0005/max(r['detector_xy_sensitivity_mm_per_normalized_unit'],1e-12),1e-6,.05)) for r in sensitivity['rows']]
        if len(steps)!=pm.N_TOTAL or min(steps)<=0:raise ValueError('Invalid sensitivity covariance scales')
        config['sensitivity_steps']=steps
        config['sensitivity_stage_factors']={'base':1.,'freeform':.5}
        write_json(out/'sensitivity_preconditioner.json',dict(source=args.sensitivity_covariance,diagnostic=sensitivity,target_per_axis_xy_mm=.0005,normalized_steps=steps,stage_factors=config['sensitivity_stage_factors']))
    write_json(out/'configuration.json',config)
    for name in ['powered_four_mirror_cma.py','powered_four_mirror.py','four_mirror.py','model.py']:
        shutil.copyfile(Path(__file__).parent/name,out/(name[:-3]+'_snapshot.py'))
    shutil.copyfile(Path('nslens/search/cma_me.py'),out/'cma_me_snapshot.py')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(run_seed,[{**config,'seed':seed} for seed in args.seeds]))
    best=min(results,key=lambda r:r['best']['metrics']['loss'])
    summary=dict(configuration=config,evaluations=sum(r['evaluations'] for r in results),best_seed=best['seed'],
                 genes=best['best']['genes'],best=best['best'],audit=best['audit'],geometry=best['geometry'],layout=best['layout'],
                 seed_results=[dict(seed=r['seed'],best=r['best'],audit=r['audit'],geometry=r['geometry']) for r in results],
                 elapsed_seconds=time.monotonic()-started,local_refinement=False)
    write_json(out/'result.json',summary)
    print(json.dumps(serial(dict(complete=True,evaluations=summary['evaluations'],best_seed=best['seed'],audit=best['audit']))),flush=True)

if __name__=='__main__':main()
