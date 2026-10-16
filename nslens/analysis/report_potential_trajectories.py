"""Equal-search-budget probe-and-continue versus static candidate allocation."""
import argparse,json,hashlib,itertools,time,shutil
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from nslens.analysis.mirror_residual_experiment import save
from nslens.analysis.mirror_spec_experiment import audit
OLD=Path('experiments/optical_potential_20261005')
METHODS=['fitness_spread','quality_spread','pilot_fitness','pilot_quality','learned_trajectory']
def quality(row):
 return (row['worst_mapping_rms_mm'] if row['worst_mapping_rms_mm'] is not None else 1000)+1000*(1-row['worst_support'])+30000*row.get('crossing',False)
def feature(row,trial,pilot):
 # Only current and prefix observations; no endpoints/audits as predictors.
 states=[trial['checkpoints'][str(t)]['coarse'] for t in [8,pilot]]
 initial=row['parent']['raw'] if 'raw' in row['parent'] else row['parent'];ff=list(row['features']['cheap'])
 for st in states:
  ff += [np.log1p(st['loss']),np.log1p(quality(st)),st['worst_support'],np.log1p(initial['loss'])-np.log1p(st['loss'])]
  for f in st['fields']:ff += [f['rms_spot_mm'] if f['rms_spot_mm'] is not None else 100, f['centroid_error_mm'] if f['centroid_error_mm'] is not None else 100,f['support']]
 return ff

def main():
 p=argparse.ArgumentParser();p.add_argument('--trajectories',required=True);p.add_argument('--out',required=True);p.add_argument('--frozen-models');a=p.parse_args();root=Path(a.trajectories);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(p.read_text()) for p in sorted(root.glob('*__*__*__*.json'))]
 # Original success labels are used ONLY for fitting training-donor models.
 old={p.name:json.loads(p.read_text()) for p in OLD.glob('*__*__*__*.json')}
 for r in rows:
  j=r['job'];r['_old']=old[f'{j["taskname"]}__{j["arm"]}__{j["n"]}__{j["seed"]}.json']
 save(out/'protocol.json',dict(trajectory_source=str(root),pilots=[16,64],budget='For m initial-failing candidates: total B=256+(m-1)*pilot. Probers spend m*pilot then256-pilot on one candidate. Static methods spend256 per candidate in ranked order and the exact remainder on the next. Same total coarse trace budget per task/donor/replicate. One final dense audit each.',selection='Static ranks by initial coarse merit or inherited initial dense quality. Probe chooses afterpilot by coarse merit, coarse quality or model. Return one among allocated endpoints by common coarse worst-field quality score; dense audit never selects it.',model='ExtraTrees160,minleaf8,maxfeatures0.8,seed612; trained on three original trajectories per configuration, excluding heldout donor, optionally excluding combined_shift task. Features initialcheap+coarse states8 andpilot. Target final256-call1mm success; no future endpoint as feature.',frozen_models=a.frozen_models,scope=('Fresh optimizer seeds with previously frozen models; same candidate population.' if a.frozen_models else 'Offline replay of existing seeds/outcomes, not new independent optimizer trials. Acquisition replay cost recorded separately; inherited labels and candidate-generation costs not free. All initial-failing candidates already have initial dense audits. Equal coarse calls and one final dense audit; not equal wall time across mirror counts.'),methods=METHODS))
 frozen=joblib.load(a.frozen_models) if a.frozen_models else None
 models={};choices=[]
 for mode in ['donor_holdout','new_task_and_donor']:
  for pilot in [16,64]:
   for seed in sorted(set(r['job']['seed'] for r in rows)):
    train=[r for r in rows if r['job']['seed']!=seed and (mode!='new_task_and_donor' or r['job']['taskname']!='combined_shift')];test=[r for r in rows if r['job']['seed']==seed and (mode!='new_task_and_donor' or r['job']['taskname']=='combined_shift')]
    X=[];Y=[]
    for r in train:
     for rep,t in enumerate(r['trials']):X.append(feature(r,t,pilot));Y.append(r['_old']['trials'][rep]['checkpoints']['256']['success'])
    key=f'{mode}_{pilot}_{seed}'
    if frozen is None:
     model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=8,max_features=.8,random_state=612,n_jobs=1);model.fit(X,Y);models[key]=dict(model=model,training_ids=[r['job'] for r in train])
    else:
     models[key]=frozen[key];model=models[key]['model']
     assert all(j['seed']!=seed for j in models[key]['training_ids'])
     if mode=='new_task_and_donor':assert all(j['taskname']!='combined_shift' for j in models[key]['training_ids'])
    for task in sorted(set(r['job']['taskname'] for r in test)):
     pool=[r for r in test if r['job']['taskname']==task];m=len(pool);total=256+(m-1)*pilot
     for rep in range(3):
      trial=[r['trials'][rep] for r in pool];states=[t['checkpoints'][str(pilot)]['coarse'] for t in trial];pred=model.predict([feature(r,t,pilot) for r,t in zip(pool,trial)]);rng=np.random.default_rng(seed*10+rep);ties=rng.random(m)
      rawparent=[r['parent']['raw'] if 'raw' in r['parent'] else r['parent'] for r in pool]
      ranks={'fitness_spread':np.array([r['loss'] for r in rawparent]),'quality_spread':np.array([quality(r['initial_audit']) for r in pool]),'pilot_fitness':np.array([r['loss'] for r in states]),'pilot_quality':np.array([quality(r) for r in states]),'learned_trajectory':-pred}
      for method in METHODS:
       order=np.lexsort((ties,ranks[method]));alloc=np.zeros(m,dtype=int)
       if method.endswith('_spread'):
        left=total
        for i in order:
         alloc[i]=min(256,left);left-=alloc[i]
         if left==0:break
        assert left==0
       else:alloc[:]=pilot;alloc[order[0]]=256
       assert sum(alloc)==total
       available=[(i,trial[i]['checkpoints'][str(int(alloc[i]))]['coarse']) for i in range(m) if alloc[i]>0]
       chosen,final=min(available,key=lambda t:(quality(t[1]),ties[t[0]]));job=pool[chosen]['job'];key=hashlib.sha256(json.dumps([job['task'],job['n'],final['genes']],sort_keys=True).encode()).hexdigest()
       choices.append(dict(mode=mode,pilot=pilot,seed=seed,task=task,repeat=rep,method=method,pool_size=m,budget=total,alloc=alloc.tolist(),chosen_job=job,chosen_budget=int(alloc[chosen]),coarse=final,audit_key=key))
 joblib.dump(models,out/'fold_models.joblib');save(out/'choices_before_audit.json',choices)
 # Audit cache only AFTER all choices frozen. Existing64/256 audits reused if exact.
 cache={}
 for r in rows:
  for t in r['_old']['trials']:
   for b in ['64','256']:
    c=t['checkpoints'][b];j=r['job'];key=hashlib.sha256(json.dumps([j['task'],j['n'],c['coarse']['genes']],sort_keys=True).encode()).hexdigest();cache[key]=c['audit']
 inherited=set(cache);newaudits=0
 for c in choices:
  if c['audit_key'] not in cache:cache[c['audit_key']]=audit(c['coarse']['genes'],c['chosen_job']['n'],c['chosen_job']['task']);newaudits+=1
  c['audit']=cache[c['audit_key']];c['success']=c['audit']['imaging_pass']['1.0']
 save(out/'policy_outcomes.json',choices)
 summary={}
 for mode in ['donor_holdout','new_task_and_donor']:
  summary[mode]={}
  for pilot in [16,64]:
   rr=[r for r in choices if r['mode']==mode and r['pilot']==pilot];methods={}
   for method in METHODS:
    ss=[r for r in rr if r['method']==method];methods[method]=dict(successes=sum(r['success'] for r in ss),trials=len(ss),rate=float(np.mean([r['success'] for r in ss])),mean_budget=float(np.mean([r['budget'] for r in ss])))
   comparisons={}
   for method in ['pilot_fitness','pilot_quality','learned_trajectory']:
    for control in ['fitness_spread','quality_spread']:
     ds=[]
     for seed in sorted(set(r['seed'] for r in rr)):
      aa=[r['success'] for r in rr if r['seed']==seed and r['method']==method];bb=[r['success'] for r in rr if r['seed']==seed and r['method']==control];ds.append(float(np.mean(aa)-np.mean(bb)))
     rng=np.random.default_rng(211);bs=np.mean(rng.choice(ds,(10000,len(ds)),replace=True),axis=1);comparisons[method+'_vs_'+control]=dict(delta=float(np.mean(ds)),bootstrap95=np.quantile(bs,[.025,.975]).tolist(),winning_donors=sum(d>0 for d in ds),donor_blocks=len(ds))
   summary[mode][str(pilot)]=dict(methods=methods,comparisons=comparisons)
 save(out/'summary.json',summary);save(out/'verification.json',dict(configurations=len(rows),policy_choices=len(choices),new_dense_audits=newaudits,inherited_cached_audits=len(inherited),equal_allocation_budgets_verified=True,one_final_audit_per_policy=True))
 shutil.copy(__file__,out/Path(__file__).name);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
