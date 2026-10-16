"""An ancestry-split numerical optical adapter: predict then rank refined moves.

Development pilot, not a global search or pretrained-Qwen experiment.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from nslens.analysis.adapter_refinement import evaluate_state, refine_state
from nslens.analysis.optical_refinement import audit
from nslens.search.cma_me import CMAES

TASKS = {
    'field8': dict(focal=75., aperture=35., half_field_deg=8.),
    'aperture42': dict(focal=75., aperture=42., half_field_deg=6.),
    'focal90': dict(focal=90., aperture=35., half_field_deg=8.),
}
SOURCE = Path('experiments/mirror_count_geometry_20261005')

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item()))

def features(state, row, task):
    # Missing rays stay explicit; signed mapping residuals preserve field structure.
    good=np.asarray(row['ray_good'], bool)
    target=row['parity']*task['focal']*np.tan(np.deg2rad(row['ray_field_deg']))
    residual=np.where(good, np.asarray(row['ray_image_mm'])-target, 0.)
    route=np.asarray(row['ray_route'])
    return np.r_[state, [task['focal']/75,task['aperture']/35,task['half_field_deg']/8],
                 np.clip(residual/10,-10,10), good.astype(float),
                 (route[:,:3]==np.arange(3)).astype(float).ravel(),
                 np.log1p(row['loss']),float(row['crossing'])]

def target(row):
    return [np.log1p(row['loss']), row['worst_support'],
            np.log1p(row['worst_mapping_rms_mm'] if row['worst_mapping_rms_mm'] is not None else 100.)]

def generate(job):
    donor,name,budget,count,out=job
    task=TASKS[name]
    src=json.loads((SOURCE/f'cross_count_{3100+donor}.json').read_text())
    initial=np.r_[src['coarse']['3']['genes'], np.zeros(4)]
    warm=refine_state(initial,task,seed=7000+donor,budget=budget)
    parent=np.asarray(warm['state'])
    rng=np.random.default_rng(8000+donor)
    candidates=[]
    for k in range(count):
        state=parent.copy()
        if k:
            block=np.arange(13) if k%2 else np.arange(6,13)
            direction=rng.normal(size=len(block)); direction/=np.linalg.norm(direction)
            state[block]=np.clip(state[block]+(.025 if k%3 else .08)*direction,-1,1)
        obs=evaluate_state(state,task)
        result=refine_state(state,task,seed=9000+donor,budget=budget)
        candidates.append(dict(state=state,features=features(state,obs,task),observation=obs,
                               refined=result,target=target(result['coarse'])))
    row=dict(donor=donor,task=name,parent=parent,warm=warm,candidates=candidates)
    save(Path(out)/f'data_{name}_{donor}.json',row)
    return row

def cma_control(parent,task,budget,seed):
    best=evaluate_state(parent,task); state=np.asarray(parent).copy();calls=1
    cma=CMAES(state,.03,popsize=8,seed=seed,bounds=(-np.ones(13),np.ones(13)),boundary_handling='reflect')
    while calls<budget:
        pop=cma.ask(); rows=[evaluate_state(x,task) for x in pop[:min(8,budget-calls)]]
        calls+=len(rows)
        if len(rows)==8:cma.tell(np.argsort([r['loss'] for r in rows]))
        for x,r in zip(pop,rows):
            if r['loss']<best['loss']:best=r;state=x.copy()
    return dict(coarse=best,state=state,coarse_calls=calls)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--budget',type=int,default=33);p.add_argument('--candidates',type=int,default=8)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    protocol=dict(train_donors=list(range(6)),test_donors=[6,7],train_tasks=['field8','aperture42'],
        unseen_task='focal90',budget=args.budget,candidates=args.candidates,
        model='ExtraTreesRegressor 128 trees min_samples_leaf=3 random_state=42; fixed before test',
        targets=['log1p refined merit','refined support','log1p refined worst-field mapping RMS'],
        selection='minimum predicted refined log merit; support and mapping predictions diagnostics only',
        online_calls=args.candidates+args.budget,
        controls='random/initial-merit selectors observe identical pool then refine one; full-state local CMA gets identical total calls',
        costs='Common donor warmup separate. Counterfactual test labels are evaluation overhead, never training or selection. Offline data generation is charged separately.',
        limits='Eight pre-existing donor families, two held out; unseen task is extrapolation, not broad generalization. Local proposal pool; no global basin discovery or Qwen claim.',
        source_hashes={str(SOURCE/f'cross_count_{3100+d}.json'):hashlib.sha256((SOURCE/f'cross_count_{3100+d}.json').read_bytes()).hexdigest() for d in range(8)})
    paths=[Path(__file__),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/analysis/optical_refinement.py'),Path('nslens/analysis/variable_mirror.py'),Path('nslens/search/cma_me.py')]
    protocol['code_hashes']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    for x in paths:(out/x.name).write_bytes(x.read_bytes())
    save(out/'protocol.json',protocol)
    jobs=[(d,t,args.budget,args.candidates,str(out)) for d in range(8) for t in TASKS if d>=6 or t!='focal90']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        data=[]
        for row in pool.map(generate,jobs):
            data.append(row);print(f"generated donor={row['donor']} task={row['task']}",flush=True)
    train=[c for row in data if row['donor']<6 for c in row['candidates']]
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,random_state=42,n_jobs=1)
    model.fit([c['features'] for c in train],[c['target'] for c in train])
    import joblib
    joblib.dump(model,out/'numeric_adapter.joblib')
    results=[]
    for row in data:
        if row['donor']<6:continue
        cs=row['candidates'];pred=model.predict([c['features'] for c in cs]);truth=np.array([c['target'] for c in cs])
        choices={'learned':int(np.argmin(pred[:,0])), 'initial_loss':int(np.argmin([c['observation']['loss'] for c in cs])),
                 'random':int(np.random.default_rng(10000+row['donor']).integers(len(cs)))}
        for arm,idx in list(choices.items())+[('cma',None)]:
            r=cs[idx]['refined'] if idx is not None else cma_control(row['parent'],TASKS[row['task']],args.budget+args.candidates,11000+row['donor'])
            raw_loss=r['coarse']['loss']
            if r['coarse']['loss']>row['warm']['coarse']['loss']:r=row['warm']
            state=np.asarray(r['state']);dense=audit(state[:9],3,2*state[9:],TASKS[row['task']])
            results.append(dict(donor=row['donor'],task=row['task'],arm=arm,index=idx,
                loss=r['coarse']['loss'],raw_loss=raw_loss,audit=dense,state=state,online_calls=args.budget+args.candidates,
                pool_regret=float(r['coarse']['loss']-min(c['refined']['coarse']['loss'] for c in cs))))
        save(out/f'predictions_{row["task"]}_{row["donor"]}.json',dict(prediction=pred,truth=truth,choices=choices))
    summary={arm:dict(passes=sum(r['audit']['threshold_reached'] for r in results if r['arm']==arm),
        runs=sum(r['arm']==arm for r in results),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in results if r['arm']==arm]))) for arm in ['learned','initial_loss','random','cma']}
    costs=dict(training_label_calls=12*args.candidates*(1+args.budget),training_warmup_calls=12*args.budget,
        test_counterfactual_calls=6*args.candidates*(1+args.budget),test_warmup_calls=6*args.budget,
        cma_test_calls=6*(args.budget+args.candidates),dense_audits=len(results))
    save(out/'results.json',results);save(out/'summary.json',dict(arms=summary,costs=costs,seconds=time.monotonic()-started))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
