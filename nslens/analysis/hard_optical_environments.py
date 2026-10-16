"""Equal-budget harder 2D optical environments with held-away feasibility witnesses."""
import argparse, hashlib, json, shutil, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from nslens.analysis.variable_mirror import decode, trace
from nslens.analysis.mirror_count_experiment import initial
from nslens.analysis.mirror_spec_experiment import audit, crosses
from nslens.analysis.mirror_residual_experiment import save
from nslens.search.cma_me import CMAES
from nslens.analysis import mirror_residual_moves as moves
SOURCE=Path('experiments/mirror_count_geometry_20261005')
TASKS={
 'combined':dict(focal=75.,aperture=35.,half_field_deg=6.),
 'field8':dict(focal=75.,aperture=35.,half_field_deg=8.),
 'aperture42':dict(focal=75.,aperture=42.,half_field_deg=6.),
 'field8_aperture42':dict(focal=75.,aperture=42.,half_field_deg=8.),
 'compact160':dict(focal=75.,aperture=35.,half_field_deg=6.),
 'compact120':dict(focal=75.,aperture=35.,half_field_deg=6.),
}
CAPS={'compact160':160.,'compact120':120.}
ARMS=['restart','reuse_only','reuse_random','optical_residual']

def envelope(g,n,task):
 """Exact coordinate extrema of finite parabolas plus required detector segment."""
 m=decode(g,n);pts=[]
 for v,t,norm,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
  us=[-w,w]
  for axis in range(2):
   den=2*c*norm[axis]
   if abs(den)>1e-15:
    u=-t[axis]/den
    if -w<=u<=w:us.append(u)
  u=np.array(us);pts.extend(v+u[:,None]*t+(c*u*u)[:,None]*norm)
 half=task['focal']*np.tan(np.deg2rad(task['half_field_deg']))+1.
 pts.extend([m['image']-half*m['image_tangent'],m['image']+half*m['image_tangent']])
 return np.ptp(np.array(pts),axis=0)

def decorate(row,g,n,task,cap):
 row['crossing']=crosses(g,n,33);row['envelope_mm']=envelope(g,n,task).tolist()
 excess=np.maximum(np.array(row['envelope_mm'])/(cap or 1e12)-1,0)
 row['optical_loss']=row['loss'];row['loss']+=30000*row['crossing']+10000*float(excess@excess)
 return row

def dense(g,n,task,cap):
 r=audit(g,n,task);r['envelope_mm']=envelope(g,n,task).tolist();r['package_pass']=cap is None or max(r['envelope_mm'])<=cap+1e-9
 r['success']=bool(r['imaging_pass']['1.0'] and r['package_pass']);return r

def refine_budget(evaluate,x,first,budget,seed,sigma):
 c=CMAES(x,sigma,popsize=8,seed=seed,bounds=(-np.ones(len(x)),np.ones(len(x))),boundary_handling='reflect');best=first;used=0
 while used<budget:
  pop=c.ask();rows=[evaluate(g) for g in pop[:min(8,budget-used)]];used+=len(rows)
  for row in rows:
   if row['loss']<best['loss']:best=row
  if len(rows)==8:c.tell(np.argsort([r['loss'] for r in rows]))
 return best

def random_direction(x,group,norm,rng):
 """Random inward-biased feasible direction with exact prescribed norm."""
 step=np.zeros_like(x)
 if norm<=1e-15:return step
 z=rng.normal(size=len(group));xx=x[group]
 z=np.where(xx+norm>1,-np.abs(z),z)
 z=np.where(xx-norm<-1,np.abs(z),z)
 step[group]=z*(norm/np.linalg.norm(z))
 assert np.all(x+step>=-1-1e-12) and np.all(x+step<=1+1e-12)
 assert np.isclose(np.linalg.norm(step),norm,atol=1e-12)
 return step

def run(job):
 start=time.monotonic();task=TASKS[job['task']];cap=CAPS.get(job['task']);seed=job['seed'];arm=job['arm'];calls=0;pool={};history=[];repeat=job.get('repeat',0);seed_offset=repeat*10000000;mutations=[]
 source_path=SOURCE/f'cross_count_{3100+seed}.json';source=json.loads(source_path.read_text())
 def evaluate(g,n):
  nonlocal calls
  calls+=1;return decorate(trace(g,n,**task),g,n,task,cap)
 for rnd in range(8):
  n=3+rnd%4;rng=np.random.default_rng(810000+seed_offset+seed*100+rnd);before=calls
  if arm=='restart':x=initial(n,rng)
  else:
   x=np.array(pool[n]['genes'] if n in pool else source['coarse'][str(n)]['genes'])
   if arm=='reuse_random' and n in pool:x=np.clip(x+rng.normal(0,.08,len(x)),-1,1)
  parent=evaluate(x,n);first_loss=parent['loss'];best=parent
  if arm in ('optical_residual','direction_random'):
   # One additional path observation is explicitly charged; controls receive it as refinement budget.
   parity=parent['parity']
   def observe(g):
    nonlocal calls
    ob=moves.observe(g,n,task,parity);calls+=1;ob['raw']=decorate(ob['raw'],g,n,task,cap)
    scale=max(task['focal']*np.tan(np.deg2rad(task['half_field_deg'])),1.)
    extra=100*np.maximum(envelope(g,n,task)/(cap or 1e12)-1,0)/scale
    ob['residual']=np.r_[ob['residual'],extra];ob['mask']=np.r_[ob['mask'],[True,True]]
    return ob
   ob=observe(x);J,valid=moves.jacobian(x,ob,observe)
   # Only rays valid under every finite-difference perturbation influence this local model.
   mask=np.all(valid,axis=1);JJ=J.copy();rr=ob['residual'].copy();JJ[~mask]=0;rr[~mask]=0
   for group in [np.arange(2*n,3*n),np.arange(3*n)]:
    for radius in [.03,.1]:
     step=moves.propose(x,rr,JJ,group,radius)
     if arm=='direction_random':step=random_direction(x,group,np.linalg.norm(step),rng)
     candidate=evaluate(x+step,n)
     mutations.append(dict(round=rnd,n=n,group='curves' if len(group)==n else 'all',radius=radius,step_norm=float(np.linalg.norm(step)),parent_loss=parent['loss'],child_loss=candidate['loss']))
     if candidate['loss']<best['loss']:best=candidate
   x=np.array(best['genes'])
  remaining=257-(calls-before)
  best=refine_budget(lambda g:evaluate(g,n),x,best,remaining,820000+seed_offset+seed*100+rnd,.12 if arm=='restart' else .03)
  assert calls-before==257
  if n not in pool or best['loss']<pool[n]['loss']:pool[n]=best
  history.append(dict(round=rnd,n=n,initial_loss=first_loss,final_loss=best['loss'],incumbent_loss=pool[n]['loss'],calls=calls))
 chosen=min(pool,key=lambda n:pool[n]['loss']);final=dense(pool[chosen]['genes'],chosen,task,cap)
 result=dict(job=job,task=task,cap_mm=cap,coarse_calls=calls,dense_calls=1,selected_n=chosen,selected_coarse=pool[chosen],audit=final,coarse=pool,history=history,mutations=mutations,seconds=time.monotonic()-start,source=str(source_path),source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest())
 suffix=f'__r{repeat}' if repeat else ''
 save(Path(job['out'])/f'{job["task"]}__{arm}__{seed}{suffix}.json',result)
 return dict(task=job['task'],arm=arm,seed=seed,success=final['success'],seconds=result['seconds'])

def references(out):
 # Existing endpoints are never passed to run(). All qualifying old combined-task endpoints are checked.
 candidates=[]
 for p in sorted(Path('experiments/optical_potential_20261005').glob('combined_shift__*.json')):
  r=json.loads(p.read_text())
  for i,t in enumerate(r['trials']):
   c=t['checkpoints']['256']
   if c['success']:candidates.append(dict(source=str(p),trial=i,n=r['job']['n'],genes=c['coarse']['genes']))
 result={}
 for name,task in TASKS.items():
  checks=[dict(candidate=c,audit=dense(c['genes'],c['n'],task,CAPS.get(name))) for c in candidates]
  passing=[x for x in checks if x['audit']['success']]
  result[name]=dict(known_feasible=bool(passing),checked=len(checks),witness=min(passing,key=lambda x:x['audit']['worst_mapping_rms_mm']) if passing else None,checks=checks)
 save(out/'feasibility_references.json',result)
 return result

def summarize(out):
 rows=[json.loads(p.read_text()) for p in out.glob('*__*__*.json')];summary={}
 for name in TASKS:
  summary[name]={}
  for arm in ARMS:
   rr=[r for r in rows if r['job']['task']==name and r['job']['arm']==arm]
   if not rr:continue
   summary[name][arm]=dict(passes=sum(r['audit']['success'] for r in rr),runs=len(rr),routing_passes=sum(r['audit']['routing_pass'] for r in rr),package_passes=sum(r['audit']['package_pass'] for r in rr),median_error_mm=float(np.median([r['audit']['worst_mapping_rms_mm'] if r['audit']['worst_mapping_rms_mm'] is not None else 100 for r in rr])),coarse_calls_per_run=2056)
 save(out/'summary.json',summary);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=6);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 assert 1<=a.seeds<=8
 jobs=[dict(task=t,arm=arm,seed=s,out=str(out)) for t in TASKS for arm in ARMS for s in range(a.seeds)]
 if a.smoke:jobs=[dict(task='compact120',arm=arm,seed=0,out=str(out)) for arm in ARMS]
 save(out/'protocol.json',dict(tasks=TASKS,caps_mm=CAPS,arms=ARMS,jobs=jobs,budget='8 rounds, count3,4,5,6 twice;257 coarse calls/round=2056 total, including all derivative probes and proposal checks. One selected endpoint dense audit. Parent count selection by coarse merit only.',initialization='Restart uses fresh geometry. Other arms share old source-control donors (not combined-task reference solutions). First visits transfer each count; repeat visits reuse incumbent. Source development costs are inherited and not counted as online; restart does not receive this prior.',residual='27 normalized ray errors plus2 enclosure overflow residuals; central differences, common valid rays only, damped least squares in curvature or all genes, radii .03/.1. Best of4 proposals by coarse merit, then CMA. No learned policy or test fitting.',objective='Legacy optical trace loss plus30000 sampled crossing penalty plus10000*squared relative enclosure overflow. All arms use identical objective. CMA sigma.12 for fresh restarts (existing broad-search setting),.03 for inherited local refinement;pop8. No topology insertion; equal allocation across3–6 mirrors.',quality='<=1mm worst-field mapping RMS,>=99% worst-field support,no sampled crossings,plus each envelope dimension<=cap where present. Dense15fields x101rays;513sample crossing check.',enclosure='Exact axis-aligned extrema of finite parabolic mirrors plus detector target field span with1mm margin at each end. Enclosure translation allowed; axes fixed to incoming on-axis ray. No mounts, thickness, clearance, finite detector clipping,3D or diffraction model.',references='Existing successful combined-task endpoints checked separately; never available to search. A failed reference check means feasibility unknown, not impossible.',scope='Exploratory difficulty sweep on six familiar donor families. No hyperparameter tuning by environment. Source-controlled comparisons isolate residual proposal value; fresh-restart comparison also measures transfer advantage.'))
 for path in [Path(__file__),Path('nslens/analysis/variable_mirror.py'),Path('nslens/analysis/mirror_residual_moves.py')]:shutil.copy(path,out/path.name)
 references(out)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(run,jobs):print(json.dumps(r),flush=True)
 print(json.dumps(summarize(out)),flush=True)
if __name__=='__main__':main()
