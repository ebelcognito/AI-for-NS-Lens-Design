"""Fit archive adapters with conservative source-family exclusion; evaluate saved pools."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nslens.analysis.mirror_adapter_pilot import save, TASKS, cma_control
from nslens.analysis.optical_refinement import audit


def families(row):
    found=set()
    for o in row['occurrences']:
        name=Path(o['source']).name
        # Canonical run filenames, not optimizer repeat seeds or summary rows.
        m=re.search(r'(?:^|_)(310[0-7]|410[0-7])\.json$',name)
        if m:found.add(int(m.group(1))%100)
        m=re.search(r'__(\d)__r\d+(?:__\w+)?\.json$',name)
        if m:found.add(int(m.group(1)))
        m=re.search(r'__(?:reuse_only|optical_residual|direction_random|restart)__(\d)\.json$',name)
        if m:found.add(int(m.group(1)))
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--pilot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.archive)
    if not (root/'manifest.json').exists():raise ValueError('Archive labeling must complete before fitting')
    rows=[json.loads(p.read_text()) for p in sorted((root/'labels').glob('*.json'))]
    # Exclude any exact physical state appearing in a held-out family, across tasks.
    key=lambda r:tuple(r['state'])
    heldkeys={key(r) for r in rows if families(r)&{6,7}}
    train=[r for r in rows if families(r) and families(r)<=set(range(6)) and key(r) not in heldkeys]
    X=np.array([r['features'] for r in train]);Y=np.array([r['target'] for r in train]);initial=np.log1p([r['observation']['loss'] for r in train])
    residual=Y.copy();residual[:,0]-=initial
    protocol=dict(train_ids=[r['id'] for r in train],train_records=len(train),excluded=len(rows)-len(train),
        grouping='Canonical source seed blocks 3100/4100+donor and hard/freeform donor filenames; all occurrences unioned. Unresolved records excluded. Exact held-out states excluded across specifications.',
        evaluation='Same six development pools as wiring pilot: family holdout retained but results already inspected; not a fresh confirmatory test.',
        models='Fixed ExtraTrees128/minleaf3 and StandardScaler+MLP64,32/max_iter200/seed42. Predict residual log-merit, support, log mapping.',
        limitations='Bounded local candidate pools, only two test families. Archived refinement repeats are not independent layouts. 33-call potential labels, not optimal attainable quality.')
    protocol['archive_pool_evaluation']='Group held-out archive records by donor and specification; observe all candidates, refine chosen for33 calls; CMA starts best observed candidate with33 further calls after pool_size screening calls. Every arm keeps best observed parent. Saved archive generation cost inherited, not online. No test labels used in selection.'
    save(out/'protocol.json',protocol)
    models={'archive_trees':ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,random_state=42,n_jobs=1),
            'archive_mlp':make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),max_iter=200,random_state=42,early_stopping=False))}
    results=[]
    for name,model in models.items():
        model.fit(X,residual);joblib.dump(model,out/f'{name}.joblib')
        for path in sorted(Path(a.pilot).glob('data_*.json')):
            row=json.loads(path.read_text())
            if row['donor']<6:continue
            cs=row['candidates'];pred=model.predict([c['features'] for c in cs]);pred[:,0]+=np.log1p([c['observation']['loss'] for c in cs])
            idx=int(np.argmin(pred[:,0]));chosen=cs[idx]['refined'];raw=chosen['coarse']['loss']
            if raw>row['warm']['coarse']['loss']:chosen=row['warm']
            state=np.asarray(chosen['state']);dense=audit(state[:9],3,2*state[9:],TASKS[row['task']])
            results.append(dict(arm=name,donor=row['donor'],task=row['task'],index=idx,predictions=pred,loss=chosen['coarse']['loss'],raw_loss=raw,audit=dense))
    baseline=json.loads((Path(a.pilot)/'results.json').read_text());combined=baseline+results
    summary={name:dict(runs=sum(r['arm']==name for r in combined),passes=sum(r['audit']['threshold_reached'] for r in combined if r['arm']==name),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in combined if r['arm']==name]))) for name in sorted({r['arm'] for r in combined})}
    save(out/'results.json',results);save(out/'summary.json',summary)
    archive_results=[]
    pools={}
    for r in rows:
        fs=families(r)
        if len(fs)==1 and fs&{6,7}:pools.setdefault((next(iter(fs)),tuple(sorted(r['task'].items()))),[]).append(r)
    for (donor,task_items),cs in pools.items():
        task=dict(task_items);base_idx=int(np.argmin([r['observation']['loss'] for r in cs]));parent=cs[base_idx]
        choices={'initial_loss':base_idx,'random':int(np.random.default_rng(15000+donor).integers(len(cs)))}
        for name,model in models.items():
            pred=model.predict([r['features'] for r in cs])[:,0]+np.log1p([r['observation']['loss'] for r in cs])
            choices[name]=int(np.argmin(pred))
        for arm,idx in list(choices.items())+[('cma',None)]:
            chosen=cs[idx]['refined'] if idx is not None else cma_control(parent['state'],task,33,16000+donor)
            state=chosen['state'];coarse=chosen['coarse']
            if coarse['loss']>parent['observation']['loss']:state=parent['state'];coarse=parent['observation']
            state=np.asarray(state);dense=audit(state[:9],3,2*state[9:],task)
            archive_results.append(dict(donor=donor,task=task,arm=arm,pool_size=len(cs),online_calls=len(cs)+33,loss=coarse['loss'],audit=dense,selected_id=cs[idx]['id'] if idx is not None else None))
    save(out/'archive_pool_results.json',archive_results)
    archive_summary={name:dict(runs=sum(r['arm']==name for r in archive_results),passes=sum(r['audit']['threshold_reached'] for r in archive_results if r['arm']==name),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in archive_results if r['arm']==name]))) for name in sorted({r['arm'] for r in archive_results})}
    save(out/'archive_pool_summary.json',archive_summary)
    print(json.dumps(dict(archive_pools=archive_summary),indent=2))
    print(json.dumps(dict(train_records=len(train),summary=summary),indent=2))

if __name__=='__main__':main()
