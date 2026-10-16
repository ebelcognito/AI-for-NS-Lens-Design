"""Measured TIR response versus independent bounded CMA refinement outcomes."""
import argparse,time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from nslens.discovery.polygon import PolygonCodec
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.portfolio import constraint_rank
from nslens.search.cma_me import CMAES
from nslens.analysis.refinement_sampling_study import save

def merit(r):
 if r['status']!='measured': return 10.
 return float(max(r['violations'].values(),default=0)+.1*sum(r['violations'].values())-.01*r['quality'])
def vec(r,keys):
 if r['status']!='measured': return np.full(len(keys)+1,2.)
 return np.r_[[r['violations'][k] for k in keys],.1*(1-r['quality'])]

def run(arg):
 corners,seed,out=arg; c=PolygonCodec(corners=corners); b=RoutingBrief(); rng=np.random.default_rng(seed); start=time.monotonic(); count=0; proposals=0
 def call(x,dense=False):
  nonlocal count,proposals
  z=evaluate(c,x,replace(b,rays_per_field=25,fields_deg=(-.5,-.25,0,.25,.5)) if dense else b,geometry_samples=1024 if dense else 256)
  count+=int(z['source_rays']>0); proposals+=1; return z
 # Keep the best of eight identical-cost measured starts; no access to other units.
 initial=[]
 while len(initial)<8 and proposals<64:
  x=rng.uniform(-.85,.85,c.gene_size); r=call(x)
  if r['status']=='measured':initial.append((x,r))
 x,r=max(initial,key=lambda q:constraint_rank(q[1])); keys=sorted(r['violations']); rv=vec(r,keys); cols=[]; probes=[]
 for j in range(c.gene_size):
  xp=x.copy();xm=x.copy();xp[j]+=.02;xm[j]-=.02;rp=call(xp);rm=call(xm);cols.append((vec(rp,keys)-vec(rm,keys))/.04);probes.extend([rp,rm])
 J=np.array(cols).T; fit=lsq_linear(J,-rv,bounds=(np.maximum(-.08,-1-x),np.minimum(.08,1-x)),max_iter=60)
 corrected=call(x+fit.x); pred=float(np.linalg.norm(rv+J@fit.x)**2)
 diag_calls=count; ends=[]
 for rep in range(2):
  cma=CMAES(x,.15,popsize=10,seed=seed*10+rep,bounds=(-np.ones(c.gene_size),np.ones(c.gene_size)),boundary_handling='reflect'); best=r; hist=[]; before=count; prop_before=proposals
  while count-before<160 and proposals-prop_before<640:
   xs=cma.ask();rr=[]
   for v in xs:
    if count-before>=160:break
    z=call(v);rr.append(z)
    if constraint_rank(z)>constraint_rank(best): best=z
    hist.append(dict(call=count-before,merit=merit(best),quality=best['quality'],feasible=best['feasible']))
   if len(rr)<10:break
   cma.tell(np.array(sorted(range(10),key=lambda i:constraint_rank(rr[i]),reverse=True)))
   if cma.restart_needed:break
  ends.append(dict(best=best,history=hist,physical_calls=count-before,proposals=proposals-prop_before))
 dense=[]
 for e in ends:dense.append(call(e['best']['genes'],True))
 row=dict(corners=corners,seed=seed,initial=r,predicted_residual=pred,initial_merit=merit(r),corrected_merit=merit(corrected),probe_route_stability=float(np.mean([p['route']==r['route'] for p in probes])),probe_merit_std=float(np.std([merit(p) for p in probes])),diagnostic_physical_calls=diag_calls,physical_calls=count,proposals=proposals,endpoints=ends,audits=dense,seconds=time.monotonic()-start)
 save(Path(out)/f'diagnostic_{corners}_{seed}.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',dict(seeds=list(range(1101,1117)),corners=[4,5],refinement_calls=160,replicates=2,diagnostic_step=.02,correction_radius=.08,notes='Prospective TIR response test. Same base brief; independent initial units, no learned global proxy. Predict second replicate from first 60-call progress. All evaluations and invalid geometry proposals counted. Dense audits do not feed selection.'))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(k,s,str(out)) for s in range(1101,1117) for k in [4,5]]):rows.append(r);print(r['corners'],r['seed'],r['seconds'],flush=True)
 report={}
 for k in [4,5]:
  rr=[r for r in rows if r['corners']==k];y=[merit(r['endpoints'][1]['best']) for r in rr];scores={}
  for key in ['initial_merit','predicted_residual','corrected_merit','probe_route_stability','probe_merit_std','early60']:
   x=[next((h['merit'] for h in reversed(r['endpoints'][0]['history']) if h['call']<=60),r['initial_merit']) if key=='early60' else r[key] for r in rr]; stat=spearmanr(x,y);scores[key]=dict(spearman=float(stat.statistic),p=float(stat.pvalue))
  report[k]=dict(n=len(rr),scores=scores,second_rep_dense_success=sum(r['audits'][1]['feasible'] for r in rr),physical_calls=sum(r['physical_calls'] for r in rr))
 save(out/'summary.json',report);print(report,flush=True)
if __name__=='__main__':main()
