"""Fixed-quality potential benchmark with repeated refinement and grouped validation."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.analysis import mirror_residual_moves as moves
from nslens.analysis.mirror_spec_experiment import crosses,audit
from nslens.analysis.mirror_residual_experiment import save
from nslens.search.cma_me import CMAES
SOURCE=Path('experiments/mirror_specs_20261005')
def jobs(out):
 jobs=[]
 for arm in ['same_count','source_reuse']:
  for path in sorted(SOURCE.glob(f'*__{arm}__*.json')):
   r=json.loads(path.read_text())
   for n in range(3,7):jobs.append(dict(source=str(path),source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),taskname=r['taskname'],task=r['task'],arm=arm,seed=r['seed'],n=n,out=str(out)))
 return jobs

def run(job):
 start=time.monotonic();source=json.loads(Path(job['source']).read_text());n=job['n'];task=job['task'];x=np.array(source['coarse'][str(n)]['genes']);calls=0;dense=0
 def observe(g,parity=None):
  nonlocal calls
  a=moves.observe(g,n,task,parity);calls+=1;a['raw']['crossing']=crosses(g,n,33);a['raw']['loss']+=30000*a['raw']['crossing'];return a
 parent=observe(x);J,valid=moves.jacobian(x,parent,lambda g:observe(g,parent['parity']));probe_calls=calls-1
 residual=parent['residual'];sv=np.linalg.svd(J,compute_uv=False);step=moves.propose(x,residual,J,np.arange(len(x)),.1);pred=residual+J@step
 fit=np.linalg.lstsq(J,-residual,rcond=1e-5)[0];irreducible=residual+J@fit
 context=[n/6,task['focal']/50,task['aperture']/27.78,task['half_field_deg']/4]
 scalar=np.r_[np.log1p(parent['raw']['loss']),context]
 cheap=np.r_[scalar,parent['features']]
 physical=np.r_[cheap,np.mean(valid),np.linalg.norm(residual),np.linalg.norm(pred),np.linalg.norm(irreducible),np.linalg.norm(fit),np.linalg.norm(step),np.linalg.norm(J.T@residual),np.sum(sv>max(sv[0]*1e-5,1e-10))/len(x),np.log1p(sv[0]),np.log1p(sv[-1])]
 trials=[]
 for repeat in range(3):
  # Unique task/arm/count seed, repeated trials independent of feature probes.
  local_seed=int(hashlib.sha256(f'{job["taskname"]}/{job["arm"]}/{job["seed"]}/{n}/{repeat}'.encode()).hexdigest()[:8],16)
  c=CMAES(x,.03,popsize=8,seed=local_seed,bounds=(-np.ones(len(x)),np.ones(len(x))),boundary_handling='reflect');best=parent['raw'];checkpoints={}
  for generation in range(32):
   pop=c.ask();rows=[observe(g)['raw'] for g in pop];c.tell(np.argsort([r['loss'] for r in rows]));candidate=min(rows,key=lambda r:r['loss'])
   if candidate['loss']<best['loss']:best=candidate
   used=(generation+1)*8
   if used in [64,256]:
    checked=audit(best['genes'],n,task);dense+=1;checkpoints[str(used)]=dict(coarse=best,audit=checked,success=checked['imaging_pass']['1.0'])
  trials.append(dict(seed=local_seed,checkpoints=checkpoints))
 assert calls==1+6*n+3*256
 record=dict(job=job,parent=parent,initial_audit=source['audits'][str(n)],features=dict(scalar=scalar,cheap=cheap,physical=physical),J=J,valid=valid,trials=trials,probe_calls=probe_calls,coarse_calls=calls,dense_calls=dense,seconds=time.monotonic()-start)
 save(Path(job['out'])/f'{job["taskname"]}__{job["arm"]}__{n}__{job["seed"]}.json',record)
 return dict(task=job['taskname'],arm=job['arm'],seed=job['seed'],n=n,seconds=record['seconds'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);jj=jobs(out)
 if a.smoke:jj=jj[:1]
 save(out/'protocol.json',dict(jobs=jj,quality='Returned design: <=1mm worst-field RMS mapping error, >=99% worst-field ray support, no sampled mirror crossings. Dense15x101 audit and513surface samples.',potential='Probability a fixed CMA refinement procedure returns a passing design at budget64 or256 coarse evaluations, conditional on starting design and task. Three independent optimizer repeats per parent, empirical fractions not exact probabilities.',refiner='CMA pop8,sigma0.03,bounded reflection. Coarse legacy mean mapping merit+lost-ray/route penalties+30000 sampled crossing penalty. Parent incumbent retained by coarse merit. Only coarse incumbent audited at checkpoints; no dense feedback.',population='240 task-adapted configurations: two prior search strategies x5tasks x6donor seed blocks x4mirror counts. Exclude initial dense-pass configurations in primary needs-improvement ranking benchmark. Their initial audits are inherited data available equally to all methods; future screening cost not free.',features='Scalar merit/context baseline; cheap27ray residual mean/std/RMS/support byfield plus task; expensive finite-difference correctability features (2d extra calls). No endpoint information enters features.',validation='Leave-one-donor-seed-out grouped CV across all tasks (6folds), additionally combined-task excluded entirely with each heldout donor. Rank candidates within task/donor pools; select top1/top2 under equal downstream refinement budgets. Fit ExtraTrees160,minleaf5,maxfeatures0.8,seed503 to per-parent empirical success fraction. No hyperparameter tuning.',cost='Actual calls per config1+2d+3*256 plus6dense audits. Initial dense audit/prior candidate-generation costs inherited. Jacobian method has extra screening cost; primary cheap-vs-scalar equal observation cost, expensive method reported separately. Outcome collection for all candidates is offline and not free.',limits='2D parabolas, variable3–6mirrors, task-adapted starting points, only6independent donors. Samples correlated across tasks/counts/strategies. Potentials conditional on optimizer/budget, not architecture bounds. Repeated trials from identical unperturbed parents; no alternative geometry generation in this test.'))
 for f in [Path(__file__),Path('nslens/analysis/mirror_residual_moves.py'),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(f,out/f.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(run,jj):print(json.dumps(r),flush=True)
if __name__=='__main__':main()
