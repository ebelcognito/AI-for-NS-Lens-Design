"""Frozen 4->2->1 refinement racing versus current TIR portfolio."""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import qmc
from nslens.discovery import benchmark
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.polygon import PolygonCodec
from nslens.discovery.portfolio import constraint_rank,select_diverse
from nslens.search.cma_me import CMAES
from nslens.analysis.refinement_sampling_study import save

def run_race(job):
 if job['arm']=='portfolio':return benchmark.run_job(job)
 out=Path(job['out']);out.mkdir(parents=True,exist_ok=False);c=PolygonCodec(**job['codec']);b=RoutingBrief(**job['brief']);rng=np.random.default_rng(job['seed']);start=time.monotonic()
 calls=proposals=0;feasible=[];best=None;counts={};phases=[]; max_calls=job['search_calls'];max_proposals=4*max_calls
 with (out/'candidates.jsonl').open('w') as stream:
  def call(x):
   nonlocal calls,proposals,best
   r=evaluate(c,x,b);r['evaluation']=proposals;proposals+=1;calls+=int(r['source_rays']>0);r['cumulative_search_rays']=calls*len(b.rays());counts[r['status']]=counts.get(r['status'],0)+1
   if best is None or constraint_rank(r)>constraint_rank(best):best=r
   if r['feasible']:feasible.append(r)
   stream.write(json.dumps(r,allow_nan=False)+'\n');return r
  initial=2*qmc.Sobol(c.gene_size,scramble=True,seed=job['seed']).random_base2(int(np.ceil(np.log2(max_proposals+64))))-1
  rr=[call(x) for x in initial[:min(40,max_calls)]]
  parents=select_diverse([r for r in rr if r['status']=='measured'],4)
  if not parents:parents=sorted(rr,key=constraint_rank,reverse=True)[:4]
  def make(x):return CMAES(np.asarray(x),.15,popsize=10,seed=int(rng.integers(0,2**30)),bounds=(-np.ones(c.gene_size),np.ones(c.gene_size)),boundary_handling='reflect')
  states=[dict(cma=make(r['genes']),best=r,generations=0) for r in parents]
  def advance(s,n):
   for _ in range(n):
    if calls>=max_calls or proposals>=max_proposals:return
    if s['cma'].restart_needed:s['cma']=make(s['best']['genes'])
    xs=s['cma'].ask();rs=[]
    for x in xs:
     if calls>=max_calls or proposals>=max_proposals:break
     r=call(x);rs.append(r)
     if constraint_rank(r)>constraint_rank(s['best']):s['best']=r
    if len(rs)==10:s['cma'].tell(np.array(sorted(range(10),key=lambda i:constraint_rank(rs[i]),reverse=True)));s['generations']+=1
  for s in states:advance(s,4)
  phases.append(dict(stage='four_starts',calls=calls,proposals=proposals))
  states=sorted(states,key=lambda s:constraint_rank(s['best']),reverse=True)[:2]
  for s in states:advance(s,5)
  phases.append(dict(stage='two_starts',calls=calls,proposals=proposals))
  winner=max(states,key=lambda s:constraint_rank(s['best']))
  while calls<max_calls and proposals<max_proposals:advance(winner,1)
 fields=sorted(b.fields_deg);ab=replace(b,rays_per_field=25,fields_deg=tuple(sorted(set(fields+[(a+b)/2 for a,b in zip(fields,fields[1:])]))))
 candidates=benchmark.selected_candidates(feasible,8);audits=[dict(coarse_evaluation=r['evaluation'],result=evaluate(c,r['genes'],ab,geometry_samples=1024)) for r in candidates];accepted=[r['result'] for r in audits if r['result']['feasible']]
 row=dict(unit=job['unit'],arm='race',seed=job['seed'],corners=c.corners,brief_id='base',config=job,proposals=proposals,search_calls=calls,search_source_rays=calls*len(b.rays()),audit_source_rays=sum(r['result']['source_rays'] for r in audits),reserved_source_rays=max_calls*len(b.rays())+8*len(ab.rays()),counts=counts,audited_success=bool(accepted),audited_designs=len(accepted),audited_routes=len({r['route'] for r in accepted}),best_audited_quality=max((r['quality'] for r in accepted),default=0.),wall_seconds=time.monotonic()-start,stop_reason='call_budget' if calls>=max_calls else 'proposal_budget',phases=phases)
 save(out/'archive.json',dict(candidates=candidates,audits=audits));save(out/'best_candidate.json',best);save(out/'summary.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--seeds',type=int,nargs='+',default=list(range(1501,1513)));a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 jobs=[dict(unit=f'base_k{k}_seed{s}',arm=arm,seed=s,codec=asdict(PolygonCodec(k)),brief=asdict(RoutingBrief()),brief_id='base',policy=dict(benchmark.POLICIES['portfolio']),search_calls=400,popsize=10,audit_rays=25,audit_limit=8,out=str(out/f'base_k{k}_seed{s}_{arm}')) for k in [4,5] for s in a.seeds for arm in ['portfolio','race']]
 save(out/'protocol.json',dict(stage='prospective_cross_domain_transfer',jobs=jobs,schedule='40 Sobol proposals; 4 diverse measured parents get4 generations each; best2 get5 additional generations; continue best until400 physical calls. CMA state preserved. Parents ranked by best constraint_rank. No tuning from these seeds.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for r in pool.map(run_race,jobs):rows.append(r);print(r['unit'],r['arm'],r['audited_success'],r['wall_seconds'],flush=True)
 save(out/'summary.json',rows)
if __name__=='__main__':main()
