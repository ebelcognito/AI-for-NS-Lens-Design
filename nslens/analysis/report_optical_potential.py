"""Grouped out-of-fold potential forecasts and limited-refinement selection."""
import sys,json,itertools,shutil
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.mirror_residual_experiment import save
METHODS=['random','current_fitness','current_quality','learned_scalar','learned_optical','learned_jacobian']
def main():
 out=Path(sys.argv[1]);allrows=[json.loads(p.read_text()) for p in sorted(out.glob('*__*__*__*.json'))];protocol=json.loads((out/'protocol.json').read_text());assert len(allrows)==len(protocol['jobs'])
 rows=[r for r in allrows if not r['initial_audit']['imaging_pass']['1.0']]
 for r in allrows:assert r['coarse_calls']==1+6*r['job']['n']+768 and r['dense_calls']==6
 predictions={};selections=[];forecasts=[];models={}
 for mode in ['donor_holdout','new_task_and_donor']:
  for budget in ['64','256']:
   for seed in sorted(set(r['job']['seed'] for r in rows)):
    train=[r for r in rows if r['job']['seed']!=seed and (mode!='new_task_and_donor' or r['job']['taskname']!='combined_shift')]
    test=[r for r in rows if r['job']['seed']==seed and (mode!='new_task_and_donor' or r['job']['taskname']=='combined_shift')]
    target=np.array([np.mean([t['checkpoints'][budget]['success'] for t in r['trials']]) for r in train]);scores={}
    for method,key in [('learned_scalar','scalar'),('learned_optical','cheap'),('learned_jacobian','physical')]:
     model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=5,max_features=.8,random_state=503,n_jobs=1);model.fit([r['features'][key] for r in train],target);scores[method]=model.predict([r['features'][key] for r in test]);models[f'{mode}_{budget}_{seed}_{method}']=dict(model=model,feature_key=key,train_ids=[r['job'] for r in train])
     for i,r in enumerate(test):
      yy=[float(t['checkpoints'][budget]['success']) for t in r['trials']];forecasts.append(dict(mode=mode,budget=budget,seed=seed,task=r['job']['taskname'],method=method,prediction=float(scores[method][i]),observed=float(np.mean(yy)),brier=float(np.mean((scores[method][i]-np.array(yy))**2))))
    for task in sorted(set(r['job']['taskname'] for r in test)):
     indices=[i for i,r in enumerate(test) if r['job']['taskname']==task];pool=[test[i] for i in indices];Y=np.array([[t['checkpoints'][budget]['success'] for t in r['trials']] for r in pool],float)
     rng=np.random.default_rng(991+seed);ties=rng.random(len(pool))
     ranking={'current_fitness':np.array([r['parent']['raw']['loss'] for r in pool]),'current_quality':np.array([(r['initial_audit']['worst_mapping_rms_mm'] if r['initial_audit']['worst_mapping_rms_mm'] is not None else 1000)+1000*(1-r['initial_audit']['worst_support'])+30000*r['initial_audit']['crossing'] for r in pool])}
     ranking.update({m:-scores[m][indices] for m in scores})
     for k in [1,2]:
      if len(pool)<k:continue
      for method in METHODS:
       if method=='random':choices=list(itertools.combinations(range(len(pool)),k))
       else:choices=[tuple(np.lexsort((ties,ranking[method]))[:k])]
       expected=float(np.mean([np.mean(np.sum(Y[list(c)],axis=0)) for c in choices]));anypass=float(np.mean([np.mean(np.max(Y[list(c)],axis=0)) for c in choices]))
       selected=[] if method=='random' else [pool[j]['job'] for j in choices[0]]
       selections.append(dict(mode=mode,budget=int(budget),seed=seed,task=task,k=k,method=method,candidates=len(pool),expected_successes=expected,probability_any=anypass,selected=selected,refinement_cost=int(budget)*k,extra_screen_cost=sum(r['probe_calls'] for r in pool) if method=='learned_jacobian' else 0))
 save(out/'selections.json',selections);save(out/'forecasts.json',forecasts);joblib.dump(models,out/'fold_models.joblib')
 summary={}
 for mode in ['donor_holdout','new_task_and_donor']:
  summary[mode]={}
  for budget in [64,256]:
   for k in [1,2]:
    key=f'B{budget}_top{k}';data=[r for r in selections if r['mode']==mode and r['budget']==budget and r['k']==k];group={}
    for method in METHODS:
     ss=[r for r in data if r['method']==method];group[method]=dict(pools=len(ss),expected_successes=float(np.mean([r['expected_successes'] for r in ss])),probability_any=float(np.mean([r['probability_any'] for r in ss])),mean_extra_screen_calls=float(np.mean([r['extra_screen_cost'] for r in ss])))
    paired={}
    for other in ['current_fitness','current_quality','learned_scalar']:
     differences=[]
     for seed in sorted(set(r['seed'] for r in data)):
      a=[r['probability_any'] for r in data if r['seed']==seed and r['method']=='learned_optical'];b=[r['probability_any'] for r in data if r['seed']==seed and r['method']==other];differences.append(np.mean(a)-np.mean(b))
     rng=np.random.default_rng(177);boot=np.mean(rng.choice(differences,(10000,len(differences)),replace=True),axis=1);paired[other]=dict(delta_probability_any=float(np.mean(differences)),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),winning_donor_blocks=int(sum(x>0 for x in differences)),donor_blocks=len(differences))
    summary[mode][key]=dict(methods=group,optical_vs_baselines=paired)
 calibration={mode:{str(b):{m:dict(brier=float(np.mean([r['brier'] for r in forecasts if r['mode']==mode and r['budget']==str(b) and r['method']==m])),mean_predicted=float(np.mean([r['prediction'] for r in forecasts if r['mode']==mode and r['budget']==str(b) and r['method']==m])),mean_observed=float(np.mean([r['observed'] for r in forecasts if r['mode']==mode and r['budget']==str(b) and r['method']==m]))) for m in ['learned_scalar','learned_optical','learned_jacobian']} for b in [64,256]} for mode in ['donor_holdout','new_task_and_donor']}
 save(out/'summary.json',summary);save(out/'calibration.json',calibration)
 distribution={str(b):{str(w):sum(sum(t['checkpoints'][str(b)]['success'] for t in r['trials'])==w for r in rows) for w in range(4)} for b in [64,256]}
 ledger=dict(configurations=len(allrows),initially_failing=len(rows),already_passing=len(allrows)-len(rows),coarse_calls=sum(r['coarse_calls'] for r in allrows),dense_calls=sum(r['dense_calls'] for r in allrows),probe_calls=sum(r['probe_calls'] for r in allrows),distribution=distribution,ledgers_passed=True);save(out/'verification.json',ledger)
 fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(2)
 for i,m in enumerate(METHODS):
  for ax,mode in zip(axes,['donor_holdout','new_task_and_donor']):ax.bar(x+(i-2.5)*.13,[summary[mode][f'B256_top{k}']['methods'][m]['probability_any'] for k in [1,2]],.13,label=m)
 for ax,title in zip(axes,['New donors: all5tasks','New donors and excluded task']):ax.set_xticks(x,['Refine1','Refine2']);ax.set_ylim(0,1);ax.set_ylabel('Observed probability of at least one1mm pass');ax.set_title(title)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 lines=['THRESHOLD-BASED OPTICAL POTENTIAL','','Definition: probability the fixed refinement procedure returns a design with <=1mm worst-field mapping RMS, >=99% support and no sampled crossings, at budget64 or256. Three independent refinements per configuration; uncertainty in each empirical probability is substantial.','No incumbent-relative target. Primary selection benchmark only uses configurations that failed the initial dense1mm audit. Initial audit status available equally to all methods; this is a measured-current-quality candidate pool, not a free screening claim.','','DATA',json.dumps(ledger,indent=2),'','SELECTION RESULTS','mode | budget | k | selector | P(at least one pass) | expected passes | extra screening calls']
 for mode,part in summary.items():
  for key,s in part.items():
   for method,v in s['methods'].items():lines.append(f'{mode} | {key} | {method} | {v["probability_any"]:.4f} | {v["expected_successes"]:.4f} | {v["mean_extra_screen_calls"]:.1f}')
 lines+=['','METHODS','current_fitness ranks current coarse scalar loss. current_quality ranks initial dense worst-field mapping error plus support/crossing penalties. learned_scalar sees coarse loss and task/count context. learned_optical adds27-ray residual patterns/support. learned_jacobian adds finite-difference correctability features and pays extra probe calls on every candidate.','Random is the exact average over all equally likely candidate subsets; selected designs are not replaced using endpoint information. Each repeat-index supplies one independent refinement per selected candidate; top2 event measured from those saved outcomes, not from oracle labels.','Within each heldout donor/task pool there are up to8 candidate configurations (2ancestral strategies x4counts). Initially passing configurations are excluded from both training and selection. Donor-block CV holds out every task/count/strategy with that seed. The stricter mode also excludes combined_shift from all model fitting.','No hyperparameters tuned on these outcomes. Grouped cross-validation is exploratory on a previously studied population, not prospective external confirmation. Models saved with training IDs.','','PAIRED SELECTION UNCERTAINTY',json.dumps({m:{k:v['optical_vs_baselines'] for k,v in p.items()} for m,p in summary.items()},indent=2),'Positive differences favor learned optical selection. Six original donor blocks, bootstrap intervals conditional on fitted CV models; repeated task/count rows are not independent. No multiplicity adjustment.','','COST AND LIMITS','Cheap optical vs scalar uses the same parent ray trace and same downstream refinement budget. Jacobian selection has extra candidate-screening cost and is NOT an equal-total-cost superiority test. All offline outcome/training collection is included in new-call ledger; prior parent generation and dense audits inherited.','Probabilities are conditional on CMA sigma0.03,mean-loss merit,64/256 calls. They are not global optimum bounds or intrinsic topology scores. Target criterion is worst-field error while refiner optimizes legacy mean-error merit; this mismatch may affect attainment.','Different mirror counts/task-adapted parents from only6source donor blocks; no3D,diffraction,coating,packaging or tolerance model. Initial-passing exclusion is retrospective using an existing measured audit, not a feature inferred for free.','All ledger assertions passed. Smoke separate. Source snapshots and parent hashes retained.','','REPRODUCE','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.optical_potential --out NEW_DIRECTORY','OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.report_optical_potential OUTPUT_DIRECTORY']
 (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');shutil.copy(__file__,out/Path(__file__).name);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
