"""Paired, norm- and evaluation-budget-controlled optical mutation experiment.

Numeric action selection is trained on disjoint parent ancestries. Dense outcomes
are audit-only. All probe, child, and refinement calls are recorded separately.
"""
import argparse,json,time,hashlib,shutil
import joblib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import binomtest
from nslens.search.cma_me import CMAES
from nslens.analysis.refinement_sampling_study import save,serial,reflect
from nslens.analysis import compensated_mirror as mirror,compensated_tir as tir

OLD=Path('experiments/refinement_sampling_20261005')

def parent_jobs(out,limit=None):
 jobs=[]
 for split,seeds,folder in [('train',range(601,625),'mirror_confirmation'),('test',range(1301,1325),'mirror_racing_confirmation')]:
  for seed in list(seeds)[:limit]:
   source=OLD/folder/'mirror_search'/f'retained_{seed}.json';x=json.loads(source.read_text())['genes']
   jobs.append(dict(backend='mirror',corners=0,split=split,seed=seed,genes=x,source=str(source),out=str(out)))
 for split,seeds,folder in [('train',range(1201,1213),'tir_confirmation'),('test',range(1501,1513),'tir_racing')]:
  for k in [4,5]:
   for seed in list(seeds)[:limit]:
    source=OLD/folder/f'base_k{k}_seed{seed}_portfolio'/'best_candidate.json';x=json.loads(source.read_text())['genes']
    jobs.append(dict(backend='tir',corners=k,split=split,seed=seed,genes=x,source=str(source),out=str(out)))
 return jobs

def refine(observe,x,first,calls,seed,active):
 c=CMAES(x[:active],.015,popsize=8,seed=seed,bounds=(-np.ones(active),np.ones(active)),boundary_handling='reflect')
 best=first;bx=x.copy();used=0;at48=None
 while used<calls:
  if c.restart_needed:c=CMAES(bx[:active],.01,popsize=8,seed=seed+used,bounds=(-np.ones(active),np.ones(active)),boundary_handling='reflect')
  xx=c.ask();yy=[]
  for v in xx:
   if used>=calls:break
   g=x.copy();g[:active]=v;o=observe(g);used+=1;yy.append(o['loss'])
   if o['loss']<best['loss']:best=o;bx=g.copy()
   if used==48:at48=(bx.copy(),best)
  if len(yy)==8:c.tell(np.argsort(yy))
 if at48 is None:at48=(bx.copy(),best)
 return at48,(bx,best)

def geometry_ok(x):
 from nslens.three_mirror.pilot import decode
 from nslens.three_mirror.audit import crossing
 m=decode(x);shapes=[]
 for v,t,n,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
  u=np.linspace(-w,w,513);shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*n)
 return not any(crossing(shapes[i],shapes[j]) for i in range(3) for j in range(i))

def run_parent(job):
 begun=time.monotonic();backend=job['backend'];module=mirror if backend=='mirror' else tir;x=np.array(job['genes']);d=len(x);active=9 if backend=='mirror' else d
 ledger=dict(coarse_attempts=0,coarse_physical=0,coarse_source_rays=0,dense_attempts=0,dense_physical=0,dense_source_rays=0)
 def obs(v,dense=False):
  o=module.observe(v,dense=dense) if backend=='mirror' else module.observe(v,corners=job['corners'],dense=dense,spot_weight=1.)
  prefix='dense' if dense else 'coarse';rays=(11*41 if dense else 27) if backend=='mirror' else o['raw'].get('source_rays',0)
  ledger[prefix+'_attempts']+=1;ledger[prefix+'_physical']+=int(rays>0);ledger[prefix+'_source_rays']+=rays
  return o
 parent=obs(x);before=ledger['coarse_attempts'];J=module.build_jacobian(x,parent,obs);probe_calls=ledger['coarse_attempts']-before
 rng=np.random.default_rng(700000+job['seed']+10000*job['corners']);actions=[]
 groups=[np.arange(6,9),np.arange(0,3),np.arange(3,6)] if backend=='mirror' else [np.arange(5+2*job['corners'],d),np.r_[4,np.arange(5,5+job['corners'])],np.r_[2,3,np.arange(5+job['corners'],5+2*job['corners'])]]
 for family,group in enumerate(groups):
  for radius in [.025,.075]:
   raw=np.zeros(d);raw[group]=rng.normal(size=len(group));raw*=radius/max(np.linalg.norm(raw),1e-30);raw=reflect(x+raw)-x
   raw[active:]=0.
   comp=module.compensated_delta(x,raw,J)
   # Same actual mutation length and bounds; no preservation win from shrinking alone.
   common=min(np.linalg.norm(raw),np.linalg.norm(comp));raw*=common/max(np.linalg.norm(raw),1e-30);comp*=common/max(np.linalg.norm(comp),1e-30)
   assert np.allclose(np.linalg.norm(raw),np.linalg.norm(comp),atol=1e-12)
   assert np.max(abs(x+raw))<=1+1e-10 and np.max(abs(x+comp))<=1+1e-10
   onehot=np.eye(3)[family];alignment=float(raw@comp/max(common*common,1e-30))
   features=np.r_[parent['loss'],parent['features'],onehot,radius,common,np.linalg.norm(J@raw[:active]),np.linalg.norm(J@comp[:active]),alignment]
   endpoints={};immediate={};local_seed=800000+job['seed']*20+len(actions)
   for mode,step in [('raw',raw),('compensated',comp)]:
    child=obs(x+step);immediate[mode]=dict(loss=child['loss'],valid=child['valid'],protected_drift=float(np.linalg.norm(child['protected']-parent['protected'])) if child['valid'] and parent['valid'] else None,feasible=child['raw'].get('feasible',child['raw'].get('worst_support',0)>=.99))
    short,long=refine(obs,x+step,child,48+(probe_calls if mode=='raw' else 0),local_seed,active)
    for tag,(g,o) in ([('raw48',short),('raw_matched',long)] if mode=='raw' else [('compensated',long)]):
     # Common incumbent guard: no method is credited for discarding a better parent.
     if parent['loss']<o['loss']:g=x.copy();o=parent
     end=dict(genes=g,loss=o['loss'],valid=o['valid'],raw=o['raw'])
     if job['split']=='test':
      end['audit']=obs(g,True)
      if backend=='mirror':end['geometry_ok']=geometry_ok(g)
     endpoints[tag]=end
   actions.append(dict(family=family,radius=radius,actual_norm=common,raw_delta=raw,compensated_delta=comp,features=features,immediate=immediate,endpoints=endpoints))
 row=dict(job=job,parent=parent,J=J,probe_calls=probe_calls,ledger=ledger,actions=actions,seconds=time.monotonic()-begun)
 name=f"{backend}_{job['corners']}_{job['split']}_{job['seed']}";save(Path(job['out'])/(name+'.json'),row);return row

def audit_loss(action,tag):return action['endpoints'][tag]['audit']['loss']
def audit_ok(action,tag,backend):
 r=action['endpoints'][tag]['audit']['raw'];return bool(r.get('feasible',False)) if backend=='tir' else bool(r['worst_support']>=.99 and action['endpoints'][tag].get('geometry_ok',False))

def summarize(out):
 rows=[json.loads(p.read_text()) for p in sorted(Path(out).glob('*_*_*.json')) if p.name.startswith(('mirror_','tir_'))]
 report={};rng=np.random.default_rng(63)
 for backend in ['mirror','tir']:
  train=[r for r in rows if r['job']['backend']==backend and r['job']['split']=='train'];test=[r for r in rows if r['job']['backend']==backend and r['job']['split']=='test']
  if not train or not test:continue
  X=[];y=[]
  for r in train:
   for a in r['actions']:X.append(a['features']);y.append(r['parent']['loss']-a['endpoints']['compensated']['loss'])
  X=np.array(X);models={}
  for label,cols in [('learned_physical',np.arange(X.shape[1])),('learned_scalar',np.r_[0,np.arange(X.shape[1]-8,X.shape[1]-3)])]:
   f=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=5,max_features=.8,random_state=93,n_jobs=1);f.fit(X[:,cols],y);models[label]=(f,cols)
  joblib.dump(dict(models=models,training_parents=[r['job'] for r in train],feature_count=X.shape[1],target='parent coarse loss minus guarded48-call refined loss'),Path(out)/(backend+'_selector.joblib'))
  results=[]
  for r in test:
   aa=r['actions'];xx=np.array([a['features'] for a in aa]);choices={k:int(np.argmax(f.predict(xx[:,cols]))) for k,(f,cols) in models.items()}
   scores={};valid={}
   for tag in ['raw48','raw_matched','compensated']:
    scores['random_'+tag]=float(np.mean([audit_loss(a,tag) for a in aa]));valid['random_'+tag]=float(np.mean([audit_ok(a,tag,backend) for a in aa]))
   for k,i in choices.items():scores[k]=audit_loss(aa[i],'compensated');valid[k]=float(audit_ok(aa[i],'compensated',backend))
   for mode,tag in [('raw','raw_matched'),('compensated','compensated')]:
    i=int(np.argmin([a['immediate'][mode]['loss'] for a in aa]));key='batch_'+mode;scores[key]=audit_loss(aa[i],tag);valid[key]=float(audit_ok(aa[i],tag,backend))
   # Selection oracle uses COARSE labels, never dense audit, and is only a headroom diagnostic.
   i=int(np.argmin([a['endpoints']['compensated']['loss'] for a in aa]));scores['coarse_oracle']=audit_loss(aa[i],'compensated');valid['coarse_oracle']=float(audit_ok(aa[i],'compensated',backend))
   results.append(dict(seed=r['job']['seed'],corners=r['job']['corners'],scores=scores,valid=valid,choices=choices))
  paired={}
  for left,right in [('random_compensated','random_raw48'),('random_compensated','random_raw_matched'),('learned_physical','random_compensated'),('learned_physical','learned_scalar'),('batch_compensated','batch_raw')]:
   seeds=sorted(set(r['seed'] for r in results));diff=np.array([np.mean([r['scores'][left]-r['scores'][right] for r in results if r['seed']==s]) for s in seeds]);wins=int(sum(diff<0));losses=int(sum(diff>0));boot=np.mean(rng.choice(diff,(10000,len(diff)),replace=True),axis=1)
   paired[left+'_vs_'+right]=dict(mean_loss_delta=float(np.mean(diff)),bootstrap95_seed_blocks=np.quantile(boot,[.025,.975]),block_wins=wins,block_losses=losses,sign_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.)
  drift={}
  for mode in ['raw','compensated']:
   vals=[a['immediate'][mode]['protected_drift'] for r in test for a in r['actions'] if a['immediate'][mode]['protected_drift'] is not None];drift[mode]=dict(median=float(np.median(vals)),valid_measured=len(vals))
  report[backend]=dict(train_parents=len(train),test_parents=len(test),methods={k:dict(mean_dense_loss=float(np.mean([r['scores'][k] for r in results])),median_parent_expected_loss=float(np.median([r['scores'][k] for r in results])),mean_audit_acceptance=float(np.mean([r['valid'][k] for r in results]))) for k in results[0]['scores']},paired=paired,protected_drift=drift,parent_results=results,training_coarse_attempts=sum(r['ledger']['coarse_attempts'] for r in train),test_coarse_attempts=sum(r['ledger']['coarse_attempts'] for r in test),limits='Fixed existing2D models and parent cohorts; offline one-decision policy evaluation, not a learned multi-step search. All dense calls excluded from selection. Random policy shown as exact expected performance over6 predeclared actions, not6 free evaluations.')
 save(Path(out)/'analysis.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);p.add_argument('--backend',choices=['all','mirror','tir'],default='all');p.add_argument('--analyze-only',action='store_true');p.add_argument('--include-smoke-parents',action='store_true');a=p.parse_args();out=Path(a.out)
 if a.analyze_only:print(serial(summarize(out)));return
 out.mkdir(parents=True,exist_ok=False);jobs=parent_jobs(out,a.limit);jobs=[j for j in jobs if (a.backend=='all' or j['backend']==a.backend) and (a.include_smoke_parents or j['split']=='train' or j['seed'] not in [1301,1501])]
 protocol=dict(jobs=jobs,actions='3 physical parameter groups ×2 radii .025/.075; same norm for raw/compensated.',compensation='damped nullspace of protected observables; J central probes h=.003; fixed parent Jacobian reused in dataset, cost charged per decision for cost-matched comparison.',refinement='48 calls compensated;48 and48+probe_calls checkpoints raw; CMA sigma .015/pop8;mirror aperture frozen.',learning='Smoke heldout parents1301/1501 excluded from main evaluation. ExtraTrees160trees/minleaf5; parent ancestry-disjoint training. Physical versus action-only model; no heldout hyperparameter selection. Final labels coarse; heldout outcomes dense audited.',task='TIR routing gates unchanged; spot_weight1 adds finite-bundle spot quality. Mirror original mapping loss. No wavefront/diffraction model.',sources={})
 for mod in [Path(__file__),Path(mirror.__file__),Path(tir.__file__)]:
  protocol['sources'][str(mod)]=hashlib.sha256(mod.read_bytes()).hexdigest();shutil.copyfile(mod,out/mod.name)
 save(out/'protocol.json',protocol)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for r in pool.map(run_parent,jobs):print(r['job']['backend'],r['job']['corners'],r['job']['split'],r['job']['seed'],round(r['seconds'],2),r['ledger'],flush=True)
 print(serial(summarize(out)),flush=True)
if __name__=='__main__':main()
