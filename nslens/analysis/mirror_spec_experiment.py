"""Task-held-out optical specification transfer; no learned policy or test tuning."""
import argparse,json,time,shutil,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.search.cma_me import CMAES
from nslens.analysis.variable_mirror import trace,decode,propose
from nslens.analysis.mirror_count_experiment import initial
from nslens.three_mirror.audit import crossing
TASKS={
 'source_control':dict(focal=50.,aperture=27.78,half_field_deg=4.),
 'wide_field':dict(focal=50.,aperture=27.78,half_field_deg=6.),
 'large_aperture':dict(focal=50.,aperture=35.,half_field_deg=4.),
 'short_mapping':dict(focal=35.,aperture=27.78,half_field_deg=4.),
 'combined_shift':dict(focal=75.,aperture=35.,half_field_deg=6.),
}
ARMS=['restart','same_count','cross_count','source_reuse']
SOURCE=Path('experiments/mirror_count_geometry_20261005')
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def crosses(g,n,samples):
 m=decode(g,n);shapes=[]
 for v,t,norm,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
  u=np.linspace(-w,w,samples);shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*norm)
 return any(crossing(shapes[i],shapes[j]) for i in range(n) for j in range(i))
def audit(g,n,task):
 row=trace(g,n,fields=15,pupils=101,**task);bad=crosses(g,n,513);row['crossing']=bad;row['routing_pass']=row['worst_support']>=.99 and not bad
 error=row['worst_mapping_rms_mm'];row['imaging_pass']={str(t):bool(row['routing_pass'] and error is not None and error<=t) for t in [.1,.5,1.]}
 return row

def run(job):
 seed,taskname,arm,out,rounds,gens=job;task=TASKS[taskname];started=time.monotonic();calls=0;pool={};history=[];source_checks={}
 source_seed=3100+(seed-4100);source=json.loads((SOURCE/f'cross_count_{source_seed}.json').read_text())
 def evaluate(g,n):
  nonlocal calls
  row=trace(g,n,**task);calls+=1;bad=crosses(g,n,33);row['optical_loss']=row['loss'];row['crossing']=bad;row['loss']+=30000*bad;return row
 for r in range(rounds):
  n=3+r%4;rng=np.random.default_rng(seed*1000+r);g=initial(n,rng);origin='restart'
  if arm in ('same_count','source_reuse') and n in pool:
   g=np.clip(np.array(pool[n]['genes'])+rng.normal(0,.08,3*n),-1,1);origin='same_count'
  elif arm=='source_reuse':g=np.array(source['coarse'][str(n)]['genes']);origin='source'
  if arm=='cross_count':
   available=[k for k in pool if abs(k-n)==1]
   if available:
    k=min(available,key=lambda k:pool[k]['loss']);g=propose(pool[k]['genes'],k,n,rng);origin=f'count_{k}'
  best=evaluate(g,n)
  if origin=='source':source_checks[str(n)]=best.copy()
  first_loss=best['loss'];c=CMAES(g,.12,popsize=12,seed=seed*1000+r,bounds=(-np.ones(3*n),np.ones(3*n)),boundary_handling='reflect')
  for generation in range(gens):
   pop=c.ask();rows=[evaluate(x,n) for x in pop];c.tell(np.argsort([a['loss'] for a in rows]));child=min(rows,key=lambda a:a['loss'])
   if child['loss']<best['loss']:best=child
  if n not in pool or best['loss']<pool[n]['loss']:pool[n]=best
  history.append(dict(round=r,n=n,origin=origin,initial_loss=first_loss,refined_loss=best['loss'],incumbent_loss=pool[n]['loss'],calls=calls))
 selected=min(pool,key=lambda n:pool[n]['loss']);audits={str(n):audit(row['genes'],n,task) for n,row in pool.items()}
 # Source-only audit is diagnostic only; never influences proposals or selection.
 source_audits={str(n):audit(row['genes'],int(n),task) for n,row in source_checks.items()}
 result=dict(seed=seed,taskname=taskname,task=task,arm=arm,source_seed=source_seed,coarse_calls=calls,geometry_checks=calls,dense_calls=len(audits)+len(source_audits),seconds=time.monotonic()-started,history=history,coarse=pool,audits=audits,selected_n=selected,selected_audit=audits[str(selected)],source_initial=source_checks,source_audits=source_audits)
 save(Path(out)/f'{taskname}__{arm}__{seed}.json',result);return dict(task=taskname,arm=arm,seed=seed,seconds=round(result['seconds'],1))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=6);p.add_argument('--rounds',type=int,default=8);p.add_argument('--gens',type=int,default=16);p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS));a=p.parse_args()
 if a.seeds>8:raise ValueError('Only8 independent source donor runs available')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 sources=[SOURCE/f'cross_count_{3100+i}.json' for i in range(a.seeds)]
 save(out/'protocol.json',dict(tasks={k:TASKS[k] for k in a.tasks},arms=ARMS,seeds=list(range(4100,4100+a.seeds)),rounds=a.rounds,generations=a.gens,population=12,count_schedule='3,4,5,6 repeated; same across arms',source_task=TASKS['source_control'],source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},source_cost=sum(json.loads(p.read_text())['coarse_calls'] for p in sources),transfer='Exact source-count incumbent from matching independent donor on first visit; then same-count reuse. No fitting to test specifications, donor selection, or dense-audit feedback. No learned policy. Source-control shares source task and is not a held-out specification.',objective='Original unnormalized mm mapping merit + loss/route penalties +30000 sampled crossing penalty. Compare within tasks; do not average raw loss across tasks.',success='Posthoc15fields x101rays; no sampled crossing at513samples/surface, >=99% worst-field support; additionally worst field mapping RMS<=0.1,0.5,1.0mm. Diagnostic thresholds, not diffraction specifications.',limits='2D fixed parabolic shape family and mirror halfwidth30mm; detector coupled to last ray; no package envelope or coating/tolerance/3D model. Tasks share parameterization. Identical count allocation; no learned budget allocation.',cost='Online rounds*(1+12*gens) traces per run in every arm; geometry check every trace. Source development costs recorded separately, not free; diagnostics include additional source-only dense audits in source_reuse.'))
 for path in [Path(__file__),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(path,out/path.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[(seed,task,arm,str(out),a.rounds,a.gens) for task in a.tasks for seed in range(4100,4100+a.seeds) for arm in ARMS]
  for result in pool.map(run,jobs):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
