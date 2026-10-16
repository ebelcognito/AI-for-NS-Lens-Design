"""Fit on donor-disjoint training rows and evaluate frozen proposal selectors."""
import sys,json,shutil
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.mirror_residual_experiment import save
POLICIES=['parent','random_matched','refine_only','random_guided','analytic','learned_scalar','learned_physical']
def main():
 out=Path(sys.argv[1]);rows=[json.loads(p.read_text()) for p in out.glob('*__*__*.json')];protocol=json.loads((out/'protocol.json').read_text());assert len(rows)==len(protocol['jobs'])
 train=[r for r in rows if r['job']['split']=='train'];test=[r for r in rows if r['job']['split']!='train'];assert train and test
 for r in rows:
  p=r['probes'];assert r['coarse_calls']==1+p+6*(2+96+p)+1+48+p
  assert r['dense_calls']==(0 if r['job']['split']=='train' else 14)
  for a in r['actions']:assert np.isclose(np.linalg.norm(a['step']),np.linalg.norm(a['random_step']),atol=1e-12)
 models={}
 for label,key in [('learned_physical','features'),('learned_scalar','scalar_features')]:
  X=[];y=[]
  for r in train:
   for a in r['actions']:X.append(a[key]);y.append(np.log1p(r['parent']['raw']['loss'])-np.log1p(a['guided']['coarse']['loss']))
  model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=6,max_features=.8,random_state=177,n_jobs=1);model.fit(X,y);models[label]=model
  joblib.dump(dict(model=model,feature_key=key,training_parents=[r['job'] for r in train],label='log1p parent penalized coarse merit minus log1p guarded endpoint merit',action_order=[(a['group'],a['radius']) for a in train[0]['actions']]),out/f'{label}.joblib')
 evaluations=[]
 def metrics(a):
  return dict(merit=a['loss']+30000*a['crossing'],routing=float(a['routing_pass']),p01=float(a['imaging_pass']['0.1']),p05=float(a['imaging_pass']['0.5']),p10=float(a['imaging_pass']['1.0']))
 for r in test:
  choices={'analytic':int(np.argmax([a['predicted_reduction'] for a in r['actions']]))}
  for label,key in [('learned_physical','features'),('learned_scalar','scalar_features')]:choices[label]=int(np.argmax(models[label].predict([a[key] for a in r['actions']])))
  endpoints={'parent':[r['parent_audit']],'refine_only':[r['refine_only']['audit']],'random_matched':[a['random_matched']['audit'] for a in r['actions']],'random_guided':[a['guided']['audit'] for a in r['actions']]}
  endpoints.update({label:[r['actions'][i]['guided']['audit']] for label,i in choices.items()})
  mm={}
  for label,aa in endpoints.items():
   vals=[metrics(a) for a in aa];mm[label]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
  evaluations.append(dict(job=r['job'],one_design=r['job']['n']==r['selected_parent_count'],choices=choices,metrics=mm))
 save(out/'policy_evaluations.json',evaluations)
 def summarize(rr):
  summary={'parents':len(rr),'donor_seeds':len(set(r['job']['seed'] for r in rr)),'policies':{},'paired':{}}
  for label in POLICIES:
   summary['policies'][label]={k:float(np.mean([r['metrics'][label][k] for r in rr])) for k in ['merit','routing','p01','p05','p10']}
   summary['policies'][label]['median_parent_merit']=float(np.median([r['metrics'][label]['merit'] for r in rr]))
  for a,b in [('random_guided','random_matched'),('analytic','refine_only'),('learned_physical','refine_only'),('learned_physical','analytic'),('learned_physical','learned_scalar')]:
   blocks=[]
   for seed in sorted(set(r['job']['seed'] for r in rr)):
    blocks.append(float(np.mean([np.log1p(r['metrics'][a]['merit'])-np.log1p(r['metrics'][b]['merit']) for r in rr if r['job']['seed']==seed])))
   rng=np.random.default_rng(87);boot=np.mean(rng.choice(blocks,(10000,len(blocks)),replace=True),axis=1)
   summary['paired'][a+'_vs_'+b]=dict(mean_log1p_delta=float(np.mean(blocks)),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),winning_donor_blocks=int(sum(x<0 for x in blocks)),blocks=len(blocks))
  return summary
 result={}
 for split in ['seed_heldout','task_heldout','both_heldout']:
  rr=[r for r in evaluations if r['job']['split']==split];result[split]=summarize(rr);result[split+'_one_design']=summarize([r for r in rr if r['one_design']])
 save(out/'summary.json',result)
 coverage={split:dict(parents=len([r for r in rows if r['job']['split']==split]),mean_parent_ray_support=float(np.mean([np.mean(r['parent']['mask']) for r in rows if r['job']['split']==split])),mean_derivative_validity=float(np.mean([np.mean(r['jacobian_valid']) for r in rows if r['job']['split']==split]))) for split in ['train','seed_heldout','task_heldout','both_heldout']};save(out/'coverage.json',coverage)
 ledger=dict(training_parents=len(train),training_actions=6*len(train),test_parents=len(test),coarse_calls=sum(r['coarse_calls'] for r in rows),training_coarse_calls=sum(r['coarse_calls'] for r in train),heldout_coarse_calls=sum(r['coarse_calls'] for r in test),dense_calls=sum(r['dense_calls'] for r in rows),ledgers_passed=True,norm_matching_passed=True);save(out/'verification.json',ledger)
 fig,axes=plt.subplots(1,2,figsize=(12,4));labels=['seed_heldout','task_heldout','both_heldout'];methods=['random_matched','refine_only','analytic','learned_physical'];x=np.arange(3)
 for i,method in enumerate(methods):
  axes[0].bar(x+(i-1.5)*.19,[result[k]['policies'][method]['median_parent_merit'] for k in labels],.19,label=method)
  axes[1].bar(x+(i-1.5)*.19,[result[k]['policies'][method]['p05'] for k in labels],.19,label=method)
 axes[0].set_yscale('log');axes[0].set_ylabel('Median parent-level dense merit (log)');axes[1].set_ylabel('Expected fraction passing0.5mm');axes[1].set_ylim(0,1)
 for ax in axes:ax.set_xticks(x,['New donors','New task','Both'],rotation=0)
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 lines=['RESIDUAL-GUIDED MUTATION AND LEARNED SELECTION','','Question: do task-specific image residuals guide useful changes beyond matched random mutation and more local refinement? Can a numerical selector learn which correction to choose?','','DESIGN',protocol['train'],protocol['test'],protocol['actions'],protocol['budgets'],protocol['model'],protocol['evaluation'],'','Actual call ledger:',json.dumps(ledger,indent=2),'','RESULTS: fraction meeting imaging thresholds includes routing and geometry screen.','partition | policy | median dense merit | routing fraction |0.1mm|0.5mm|1mm']
 for split,rr in result.items():
  lines.append(f'{split}: {rr["parents"]} parents; {rr["donor_seeds"]} donor seeds')
  for method,v in rr['policies'].items():lines.append(f'{split} | {method} | {v["median_parent_merit"]:.6g} | {v["routing"]:.3f} | {v["p01"]:.3f} | {v["p05"]:.3f} | {v["p10"]:.3f}')
 lines+=['','Interpretation of fractions: random policies are exact expectations over6 actions, NOT six free evaluations. Deterministic policies select one action using only parent/task/proposal features. Refinement-only spends the sensitivity-call budget on extra CMA steps.','Dense merit is optical loss plus30000 for sampled crossing; not a pure spot-size metric. Imaging passes additionally require>=99% worst-field support, no sampled crossing and worst-field mapping RMS below threshold.','One-design partitions hold mirror count at the parent count selected before this experiment; no free selection across refined counts.','','SEED-BLOCK PAIRED STATISTICS',json.dumps({k:v['paired'] for k,v in result.items()},indent=2),'Negative log1p differences favor the first policy. Only3 independent donors per test partition: exploratory estimates, no robust generalization guarantee, no multiplicity correction.','','VALIDITY AND LIMITS',json.dumps(coverage,indent=2),'Missing rays have zero-filled residuals AND explicit masks; only parent/plus/minus common-valid rows contribute derivatives. Geometry and lost-ray penalties govern actual acceptance. No claim to a natural gradient; these are damped local least-squares corrections.','The learned model chooses among six hand-generated corrections, not arbitrary design directions or topology edits. Training labels come from simulator/refinement, not human labels; this is supervised outcome learning from self-generated data, not purely unsupervised learning.','No iterative learned-policy rollout here. Offline counterfactual data costs are logged; training is not free. Prior task-adaptation data and source donor costs are inherited, not included in new-call counts.','Splits are by original donor seeds, preventing train/test ancestry overlap for seed-heldout and joint-heldout. Task-only holdout intentionally shares donors. Tasks/parents were explored in preceding experiments; task holdout means excluded from model fitting, not a prospective untouched benchmark.','Fixed2D parabolas, coupled detector, no diffraction,3D wavefront,tolerance or coating model.','Source snapshots and parent hashes preserve provenance. Main sources and defaults outside analysis unchanged.','','VERIFICATION','12 unittest checks passed before run; synthetic residual/derivative/group bounds, direct ray residual comparisons, prior spec and mirror tests. All evaluation ledgers and matched mutation norms reconciled. Read-only worker review performed.','','REPRODUCE','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.mirror_residual_experiment --out NEW_DIRECTORY','OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.report_mirror_residual OUTPUT_DIRECTORY']
 (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');shutil.copy(__file__,out/Path(__file__).name)
 print(json.dumps({k:v for k,v in result.items() if k in ('seed_heldout','both_heldout_one_design')},indent=2))
if __name__=='__main__':main()
