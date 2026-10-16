"""Residual-targeted mirror insertion against equal-budget structural controls."""
import json,argparse,time,shutil,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.analysis import mirror_residual_moves as moves
from nslens.analysis.variable_mirror import propose
from nslens.analysis.mirror_spec_experiment import crosses,audit
from nslens.analysis.mirror_residual_experiment import refine,save
SOURCE=Path('experiments/mirror_specs_20261005')
def insert(x,n,leg,turn,folded=False):
 angles=np.array(x[:n])*180;lengths=65+40*np.array(x[n:2*n]);curves=np.array(x[2*n:]);old=lengths[leg];lengths[leg]=max(old/2,25)
 if folded:angles[leg+1:]*=-1
 angles=np.insert(angles,leg+1,turn);lengths=np.insert(lengths,leg+1,max(old/2,25));curves=np.insert(curves,leg+1,0.)
 return np.clip(np.r_[angles/180,(lengths-65)/40,curves],-1,1)
def run(job):
 path,n,out,folded=job;source=json.loads(Path(path).read_text());task=source['task'];seed=source['seed'];x=np.array(source['coarse'][str(n)]['genes']);newn=n+1;legs=[i for i in range(n-1) if not folded or 65+40*x[n+i]>=50];K=2*len(legs);assert K>0;total=8*K+96;calls=0;start=time.monotonic()
 def ob(g,count,parity=None):
  nonlocal calls
  r=moves.observe(g,count,task,parity=parity);calls+=1;bad=crosses(g,count,33);r['raw']['crossing']=bad;r['raw']['loss']+=30000*bad;return r
 candidates=[];flat=[]
 for leg in legs:
  for turn in ([-90.,90.] if folded else [-8.,8.]):
   g=insert(x,n,leg,turn,folded=folded);parent=ob(g,newn);flat.append(parent['raw']);J=np.zeros((27,3*newn));valid=[]
   # New mirror and its immediately adjacent old mirrors redistribute power.
   group=np.array([2*newn+leg,2*newn+leg+1,2*newn+leg+2])
   for j in group:
    plus=g.copy();minus=g.copy();plus[j]=min(1,g[j]+.003);minus[j]=max(-1,g[j]-.003)
    a=ob(plus,newn,parent['parity']);b=ob(minus,newn,parent['parity']);mask=parent['mask']&a['mask']&b['mask'];J[mask,j]=(a['residual'][mask]-b['residual'][mask])/(plus[j]-minus[j]);valid.append(float(np.mean(mask)))
   step=moves.propose(g,parent['residual'],J,group,.12);corrected=ob(g+step,newn)
   best=min([parent['raw'],corrected['raw']],key=lambda r:r['loss']);candidates.append(dict(leg=leg,turn=turn,initial=parent['raw'],corrected=corrected['raw'],selected=best,step_norm=float(np.linalg.norm(step)),valid_fraction=float(np.mean(valid))))
 assert calls==8*K;results={};cma_seed=seed*10+n
 def finish(first,count,budget):
  return refine(lambda g:ob(g,count)['raw'],np.array(first['genes']),first,budget,cma_seed)
 chosen=min(candidates,key=lambda r:r['selected']['loss']);guided=finish(chosen['selected'],newn,96);assert calls==total
 results['targeted']=dict(coarse=guided,online_calls=total,screen_calls=8*K,refinement_calls=96,audit=audit(guided['genes'],newn,task))
 # Flat candidate traces already exist in dataset, but standalone arm pays K.
 before=calls;flatbest=min(flat,key=lambda r:r['loss']);end=finish(flatbest,newn,total-K)
 results['flat_batch']=dict(coarse=end,online_calls=total,screen_calls=K,refinement_calls=total-K,audit=audit(end['genes'],newn,task),reused_dataset_traces=K)
 assert calls-before==total-K
 before=calls;rng=np.random.default_rng(seed*10+n);legacy=np.array(flat[int(rng.integers(K))]['genes']) if folded else propose(x,n,newn,rng);first=ob(legacy,newn)['raw'];end=finish(first,newn,total-1)
 results['legacy_one']=dict(coarse=end,online_calls=total,screen_calls=1,refinement_calls=total-1,audit=audit(end['genes'],newn,task));assert calls-before==total
 before=calls;first=ob(x,n)['raw'];end=finish(first,n,total-1)
 results['keep_count']=dict(coarse=end,online_calls=total,screen_calls=1,refinement_calls=total-1,audit=audit(end['genes'],n,task));assert calls-before==total
 parent_audit=audit(x,n,task)
 # Added-mirror-only metrics and deployment guard are separate. Guard costs one
 # common parent observation, inherited from prior study, not a free new search.
 for r in results.values():
  r['guarded_parent']=first['loss']<r['coarse']['loss']
  r['guarded_audit']=parent_audit if r['guarded_parent'] else r['audit']
 row=dict(source=str(path),source_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest(),taskname=source['taskname'],task=task,seed=seed,parent_n=n,parent_selected_n=source['selected_n'],candidates=candidates,results=results,parent_audit=parent_audit,coarse_calls=calls,dense_calls=5,standalone_budget=total,seconds=time.monotonic()-start)
 assert calls==4*total-K
 save(Path(out)/f'{source["taskname"]}__{n}__{seed}.json',row);return dict(task=source['taskname'],n=n,seed=seed,calls=calls,seconds=row['seconds'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--folded',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 jobs=[(str(path),n,str(out),a.folded) for path in sorted(SOURCE.glob('*__source_reuse__*.json')) for n in [3,4,5]]
 if a.smoke:jobs=jobs[:1]
 save(out/'protocol.json',dict(jobs=jobs,folded=a.folded,folded_override='If folded: use only legs>=50mm, split exactly, insert+/-90deg flat fold, reverse signs of all downstream turns. This reflects downstream optics across fold plane; curvatures retained. Ideal path geometry is preserved until finite aperture/new obstruction interferes. K=2*eligible legs. legacy_one selects one random folded candidate instead of legacy small-turn insertion. All other costs same.',method='For every intermirror leg, insert flat mirror at +/-8deg turn and split distance with25mm minimum. Trace, measure6 finite differences for new and adjacent curvatures, apply damped residual correction capped0.12 norm, trace correction. Choose best actual coarse candidate, then96 CMA evaluations.',controls='Same flat candidate batch with sensitivity/correction budget spent on more CMA; original random flat insertion with remaining budget spent on CMA; keeping count with full refinement budget. All same per-parent standalone budget8*K+96, K=2*(n-1).',cost='Actual dataset reuses flat batch traces and logs4*budget-K coarse calls. Dense audits5 per parent. Standalone controls still charged K. Prior adapted parent/source costs inherited; not a fresh-start total-compute test.',task='All5 previous specs,6 donors,3->4/4->5/5->6. No learned policy. Targeted method responds to inserted design residuals, not a proven architecture-capacity bound. Local curvature redistribution is approximate power adjustment, not exact conserved-power insertion.',evaluation='Dense15x101,513point surface crossing screen. Added designs and coarse guarded parent fallback reported separately. One-design subset uses previously selected parent count, fixed before this test.',limits='2D ideal parabolas. Step at grazing inserted mirror may change route discontinuously. No3D,diffraction,coating,tolerance,packaging constraints. Group comparisons paired by donor; task/count repeats correlated.'))
 if a.folded:
  protocol=json.loads((out/'protocol.json').read_text());protocol['method']='For each intermirror gap >=50mm: split exactly in half, insert a flat mirror with+/-90deg turn, negate downstream turns to reflect downstream optics. Trace, measure6 finite differences for new/adjacent curvatures, apply damped residual correction capped0.12, trace correction; select by actual coarse merit and refine96 calls.';protocol['controls']='Same folded candidate batch with correction budget spent on extra CMA; one uniformly random folded candidate plus remaining CMA; keeping parent count with full budget. All standalone budgets8*K+96 where K=2*eligible gaps.';save(out/'protocol.json',protocol)
 for path in [Path(__file__),Path('nslens/analysis/mirror_residual_moves.py'),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(path,out/path.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(run,jobs):print(json.dumps(row),flush=True)
if __name__=='__main__':main()
