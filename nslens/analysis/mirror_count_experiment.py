"""Equal-budget exploratory 3--6 mirror architecture search."""
import argparse,json,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.search.cma_me import CMAES
from nslens.analysis.variable_mirror import trace,propose,decode
from nslens.three_mirror.audit import crossing

def save(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def initial(n,rng):
 angles=np.array([-150,180,-180,180,-180,180][:n])+rng.uniform(-1,1,n)*np.array([28]+[58]*(n-1))
 angles=(angles+180)%360-180
 return np.r_[angles/180,rng.uniform(-.8,.8,n),rng.uniform(-.8,.8,n)]
def run(job):
 seed,arm,out,rounds,gens,geometry_penalty=job; start=time.monotonic();pool={};history=[];calls=0
 def evaluate(x,n):
  row=trace(x,n)
  if geometry_penalty:
   m=decode(x,n);shapes=[]
   for v,t,norm,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
    u=np.linspace(-w,w,33);shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*norm)
   bad=any(crossing(shapes[i],shapes[j]) for i in range(n) for j in range(i))
   row['optical_loss']=row['loss'];row['sampled_crossing']=bad;row['loss']+=30000*bad
  return row
 for r in range(rounds):
  n=3+r%4;rng=np.random.default_rng(seed*1000+r);x=initial(n,rng);origin='restart'
  if arm=='same_count' and n in pool:
   x=np.clip(np.array(pool[n]['genes'])+rng.normal(0,.08,3*n),-1,1);origin=f'mutate_{n}'
  if arm=='cross_count':
   options=[k for k in pool if abs(k-n)==1]
   if options:
    k=min(options,key=lambda k:pool[k]['loss']);x=propose(np.array(pool[k]['genes']),k,n,rng);origin=f'transfer_{k}'
  best=evaluate(x,n);calls+=1
  c=CMAES(x,.12,popsize=12,seed=seed*1000+r,bounds=(-np.ones(3*n),np.ones(3*n)),boundary_handling='reflect')
  for generation in range(gens):
   pop=c.ask();rows=[evaluate(v,n) for v in pop];calls+=12;c.tell(np.argsort([v['loss'] for v in rows]))
   candidate=min(rows,key=lambda v:v['loss'])
   if candidate['loss']<best['loss']:best=candidate
  if n not in pool or best['loss']<pool[n]['loss']:pool[n]=best
  history.append(dict(round=r,n=n,origin=origin,child_refined_loss=best['loss'],incumbent_loss=pool[n]['loss'],calls=calls))
 audits={str(n):trace(v['genes'],n,fields=11,pupils=81) for n,v in pool.items()}
 selected=min(pool,key=lambda n:pool[n]['loss'])
 row=dict(seed=seed,arm=arm,coarse_calls=calls,dense_calls=len(audits),geometry_checks=calls if geometry_penalty else 0,seconds=time.monotonic()-start,history=history,coarse=pool,audits=audits,selected_n=selected,selected_audit=audits[str(selected)])
 save(Path(out)/f'{arm}_{seed}.json',row);return row

def summarize(out):
 rows=[json.loads(p.read_text()) for p in out.glob('*_*.json') if p.name.split('_')[-1].split('.')[0].isdigit()]
 summary={}
 for arm in ['restart','same_count','cross_count']:
  rr=[r for r in rows if r['arm']==arm]
  summary[arm]={'runs':len(rr),'coarse_calls':sum(r['coarse_calls'] for r in rr),'by_count':{str(n):{'median_dense_loss':float(np.median([r['audits'][str(n)]['loss'] for r in rr])),'mean_worst_support':float(np.mean([r['audits'][str(n)]['worst_support'] for r in rr])),'full_support_runs':sum(r['audits'][str(n)]['worst_support']>=.99 for r in rr)} for n in range(3,7)},'selected_median_dense_loss':float(np.median([r['selected_audit']['loss'] for r in rr]))}
 save(out/'summary.json',summary);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--geometry-penalty',action='store_true');p.add_argument('--seeds',type=int,default=8);p.add_argument('--seed-start',type=int,default=2700);p.add_argument('--rounds',type=int,default=12);p.add_argument('--gens',type=int,default=16);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',dict(geometry_penalty=a.geometry_penalty,geometry_rule="If enabled: +30000 for any proper crossing at33 samples/surface on every search evaluation; posthoc audit513 samples. Geometry checks are extra compute shared by all arms.",seeds=list(range(a.seed_start,a.seed_start+a.seeds)),rounds=a.rounds,generations=a.gens,population=12,arms=['restart','same_count','cross_count'],count_schedule='3,4,5,6 repeated; same allocation in every arm',task='2D ideal parabolic mirrors, 50mm target mapping, 27.78mm entrance aperture, +/-4deg fields, fixed30mm mirror halfwidth',scope='Fresh starts, no prior solved parents. Cross-count uses adjacent count incumbent; same-count mutation controls reuse. Broad +/-180deg turn genes. Detector follows final nominal ray with variable final distance. No complexity penalty; report each count separately. No package envelope or coating loss; no 3D/diffraction/clearance guarantee.',budget='Each round 1 initial trace +12*gens CMA evaluations. Dense11x81 audit only after search; never used for selection. Each count retains best coarse design; overall count selected by coarse merit.',loss='Image mapping MSE clipped100mm plus10000 lost fraction plus1000 normalized expected-route progress deficit; all entrance rays in denominator.'))
 for path in [Path(__file__),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(path,out/path.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(run,[(s,arm,str(out),a.rounds,a.gens,a.geometry_penalty) for s in range(a.seed_start,a.seed_start+a.seeds) for arm in ['restart','same_count','cross_count']]):print(r['seed'],r['arm'],round(r['seconds'],1),flush=True)
 print(json.dumps(summarize(out)),flush=True)
if __name__=='__main__':main()
