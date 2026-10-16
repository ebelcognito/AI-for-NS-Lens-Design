"""Learn residual-correction choice with disjoint donor and task validation."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.search.cma_me import CMAES
from nslens.analysis.mirror_spec_experiment import crosses,audit
from nslens.analysis import mirror_residual_moves as moves
SOURCE=Path('experiments/mirror_specs_20261005')
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 return x
def save(p,x):p.write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n')
def bounded(x,step):
 scale=1.
 for a,d in zip(x,step):
  if d>0:scale=min(scale,(1-a)/d)
  elif d<0:scale=min(scale,(-1-a)/d)
 return step*max(0,scale)
def refine(evaluate,x,first,budget,seed):
 c=CMAES(x,.03,popsize=8,seed=seed,bounds=(-np.ones(len(x)),np.ones(len(x))),boundary_handling='reflect');best=first;used=0
 while used<budget:
  pop=c.ask();rows=[]
  for v in pop[:min(8,budget-used)]:
   row=evaluate(v);rows.append(row);used+=1
   if row['loss']<best['loss']:best=row
  if len(rows)==8:c.tell(np.argsort([r['loss'] for r in rows]))
 return best

def run(job):
 started=time.monotonic();source=json.loads(Path(job['source']).read_text());n=job['n'];task=source['task'];x=np.array(source['coarse'][str(n)]['genes']);calls=0;dense=0;parity=None
 def evaluate(g):
  nonlocal calls
  ob=moves.observe(g,n,task,parity=parity);calls+=1;bad=crosses(g,n,33);ob['raw']['optical_loss']=ob['raw']['loss'];ob['raw']['loss']+=30000*bad;ob['raw']['crossing']=bad
  return ob
 parent=evaluate(x);parity=parent['parity'];before=calls;J,valid=moves.jacobian(x,parent,evaluate);probes=calls-before;assert probes==2*len(x)
 heldout=job['split']!='train'
 def endpoint(row):
  nonlocal dense
  if row['loss']>parent['raw']['loss']:row=parent['raw']
  output={'coarse':row}
  if heldout:output['audit']=audit(row['genes'],n,task);dense+=1
  return output
 def raw_eval(g):return evaluate(g)['raw']
 actions=[];groups=moves.action_groups(n);group_names=['curves','lengths','all']
 for gi,group in enumerate(group_names):
  indices=np.asarray(groups[group],int)
  for radius in [.03,.1]:
   aid=len(actions);rng=np.random.default_rng(910000+job['seed']*100+n*10+aid)
   step=moves.propose(x,parent['residual'],J,indices,radius)
   raw=np.zeros(len(x));raw[indices]=rng.normal(size=len(indices));raw*=np.linalg.norm(step)/max(np.linalg.norm(raw),1e-30);raw=bounded(x,raw)
   common=min(np.linalg.norm(step),np.linalg.norm(raw));step*=common/max(np.linalg.norm(step),1e-30);raw*=common/max(np.linalg.norm(raw),1e-30)
   assert np.isclose(np.linalg.norm(step),np.linalg.norm(raw),atol=1e-12)
   pred=float(np.sum(parent['residual']**2)-np.sum((parent['residual']+J@step)**2))
   basefeatures=np.r_[np.log1p(parent['raw']['loss']),n/6,task['focal']/50,task['aperture']/27.78,task['half_field_deg']/4,np.eye(3)[gi],radius,common]
   features=np.r_[basefeatures,parent['features'],pred,np.linalg.norm(J@step),np.mean(valid),np.linalg.norm(J),float(np.mean(parent['mask']))]
   seed=job['seed']*1000+n*10+aid
   guided_start=raw_eval(x+step);guided=refine(raw_eval,x+step,guided_start,48,seed)
   random_start=raw_eval(x+raw);random=refine(raw_eval,x+raw,random_start,48+probes,seed)
   actions.append(dict(group=group,radius=radius,step=step,random_step=raw,features=features,scalar_features=basefeatures,predicted_reduction=pred,guided_initial_loss=guided_start['loss'],random_initial_loss=random_start['loss'],guided=endpoint(guided),random_matched=endpoint(random)))
 # Same total per-decision tracing budget but all sensitivity calls allocated to CMA.
 only_first=raw_eval(x);only=refine(raw_eval,x,only_first,48+probes,job['seed']*1000+n*10)
 only=endpoint(only)
 parent_audit=audit(x,n,task) if heldout else None;dense+=int(heldout)
 expected=1+probes+6*(2+96+probes)+1+48+probes;assert calls==expected,(calls,expected)
 row=dict(job=job,task=task,parent=parent,selected_parent_count=source['selected_n'],J=J,jacobian_valid=valid,actions=actions,refine_only=only,parent_audit=parent_audit,coarse_calls=calls,dense_calls=dense,probes=probes,seconds=time.monotonic()-started)
 save(Path(job['out'])/f'{job["taskname"]}__{n}__{job["seed"]}.json',row)
 return {k:row[k] for k in ['coarse_calls','dense_calls','seconds']}|{'task':job['taskname'],'seed':job['seed'],'n':n,'split':job['split']}

def jobs(out):
 result=[]
 for path in sorted(SOURCE.glob('*__source_reuse__*.json')):
  source=json.loads(path.read_text());seed=source['seed'];taskname=source['taskname'];newseed=seed>=4103;newtask=taskname=='combined_shift'
  split='train' if not newseed and not newtask else ('both_heldout' if newseed and newtask else 'task_heldout' if newtask else 'seed_heldout')
  for n in range(3,7):result.append(dict(source=str(path),source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,taskname=taskname,n=n,split=split,out=str(out)))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);jj=jobs(out)
 if a.smoke:jj=[next(j for j in jj if j['split']=='both_heldout' and j['n']==3)]
 save(out/'protocol.json',dict(jobs=jj,train='Donor seeds4100..4102, all tasks except combined_shift. 48 parent configs; no dense training labels.',test='Seed-heldout4103..4105 on4 familiar tasks; combined task heldout on3 known and3 new donors. Strict joint holdout12 parents, only3 independent donors. Counts within a donor are correlated.',actions='Damped least-squares residual correction restricted to curvature, length, or all parameters, each at capped normalized radius0.03 or0.1; random matched actual norm in same group. No topology edit in this experiment.',residual='27 per-ray signed image-mapping errors normalized by max(target image halfheight,1mm); parent parity frozen for derivatives; invalid ray rows masked. Parent and all probes include geometry penalties.',budgets='One standalone guided decision:2d probes+1 child+48 CMA calls. Random and refine-only:1 child+48+2d CMA calls. Parent shared. Offline dataset computes all6 counterfactual actions, reuses one Jacobian; actual traces logged. Training collection and inherited parent/source costs are not free.',model='ExtraTrees160,min_samples_leaf6,max_features0.8,seed177. Predict log1p(parent merit)-log1p(guarded refined merit). Physical vs scalar task/action features. No test tuning or dense labels. Analytic selector uses largest linear predicted residual reduction.',evaluation='Dense15x101 plus513 surface crossing screen only on heldout endpoints. Select actions without endpoint audits. One-design subset uses parent count selected in prior study by coarse loss before this experiment; no free selection among refined counts.',limits='Offline one-step selection test, not closed-loop learned search. Fixed2D parabolas, inherited tasks and post-adaptation parents from previous study. Small donor split; joint holdout exploratory. Guarded endpoints keep parent if coarse loss worsens.'))
 for path in [Path(__file__),Path('nslens/analysis/mirror_residual_moves.py'),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(path,out/path.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(run,jj):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
