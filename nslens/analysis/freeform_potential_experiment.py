"""Test the user-selected layout with a richer final-surface refinement family."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import lsq_linear
from nslens.search.cma_me import CMAES
from nslens.three_mirror.audit import crossing
from nslens.analysis import final_freeform as ff
from nslens.analysis.mirror_residual_experiment import save
SOURCE=Path('experiments/hard_confirmation_20261006/field8__optical_residual__1__r1.json')
ARMS=['last_parabolic','last_freeform','joint_parabolic','joint_freeform']

from nslens.analysis.optical_refinement import geometry_crossing, audit, slope_fit

def run(job):
 start=time.monotonic();source=json.loads(SOURCE.read_text());base=np.array(source['selected_coarse']['genes']);n=source['selected_n'];task=source['task'];arm=job['arm'];free=arm.endswith('freeform');joint=arm.startswith('joint');calls=0
 indices=np.arange(3*n) if joint else np.array([3*n-1]);x=np.r_[base[indices],np.zeros(4)] if free else base[indices].copy()
 def unpack(v):
  g=base.copy();g[indices]=v[:len(indices)];a=2*np.array(v[-4:]) if free else np.zeros(4);return g,a
 def evaluate(v):
  nonlocal calls
  g,a=unpack(v);r=ff.trace(g,n,a,fields=5,pupils=21,**task);calls+=1;r['crossing']=geometry_crossing(g,n,a,65);r['loss']+=30000*r['crossing'];r['parameters']=np.array(v).tolist();return r
 best=evaluate(x);first=best
 correction,compat=slope_fit(base,n,task,free);calls+=1 # slope constraints are one additional105-ray trace.
 for fraction in [1.,.5,.25]:
  candidate=x.copy();candidate[np.flatnonzero(indices==3*n-1)[0]]+=fraction*correction[0]
  if free:candidate[-4:]=fraction*correction[1:]/2
  r=evaluate(candidate)
  if r['loss']<best['loss']:best=r
 fitted=best;current=np.array(best['parameters']);cma=CMAES(current,.03,popsize=8,seed=job['seed'],bounds=(-np.ones(len(current)),np.ones(len(current))),boundary_handling='reflect')
 checkpoints={};search_calls=0
 while search_calls<1024:
  pop=cma.ask();rr=[evaluate(v) for v in pop];search_calls+=8;cma.tell(np.argsort([r['loss'] for r in rr]));b=min(rr,key=lambda r:r['loss'])
  if b['loss']<best['loss']:best=b
  if search_calls in [256,1024]:
   g,a=unpack(best['parameters']);checkpoints[str(search_calls)]=dict(coarse=best,audit=audit(g,n,a,task))
 assert calls==1029
 result=dict(job=job,source=str(SOURCE),source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),n=n,task=task,initial=first,fitted=fitted,slope_compatibility=compat,checkpoints=checkpoints,coarse_calls=calls,dense_calls=2,seconds=time.monotonic()-start)
 save(Path(job['out'])/f'{arm}__{job["seed"]}.json',result);return dict(arm=arm,seed=job['seed'],error=checkpoints['1024']['audit']['worst_mapping_rms_mm'],seconds=result['seconds'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 jobs=[dict(arm=arm,seed=91000000+s,out=str(out)) for arm in ARMS for s in range(3)]
 if a.smoke:jobs=[dict(arm='last_freeform',seed=91000000,out=str(out))]
 save(out/'protocol.json',dict(jobs=jobs,source=str(SOURCE),source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),purpose='Evaluate recoverable imaging quality of the user-selected five-mirror layout, not classify it by current image error. Selected single-case experiment, not a learned/population potential predictor.',arms=ARMS,freeform='2D final-mirror sag c*u² + a3*(u/30)^3 +a4*(u/30)^4+a5*(u/30)^5+a6*(u/30)^6. Each extra coefficient bounded±2mm, initially0. Curvature remains bounded by original gene range. No piston or tilt duplicates.',variables='last_* changes only last mirror curvature and optional4shape coefficients, all vertices/other surfaces/detector fixed. joint_* changes all original3n genes and optional4shape coefficients; detector follows original coupled geometry.',initialization='Every arm uses1baseline trace+1frozen-hit desired-slope fit trace+3candidate traces (fractions1,.5,.25) before1024CMA traces, same budget1029. Parabolic fits curvature only; freeform fits curvature+4higher terms. Best incumbent always retained. CMA sigma.03,pop8,3seeds; checkpoints256 and1024.',objective='5fields x21pupils training grid; original mean mapping MSE+lost-ray and route-progress penalties plus30000sampled crossing penalty. Current spot size is not used to reject a layout. Mapping quality evaluated after refinement.',audit='15fields x101pupils,513surface samples for crossings; no audit feedback to optimizer or incumbent selection. 1mm reported as previous quality reference only, not layout potential label.',physics='Exact polynomial roots for last-mirror intersections and analytic derivatives for reflection; nearest-surface routing retained. Zero coefficients reproduce original model. Finite30mm halfwidth. No full3D freeform, fabrication/tolerance/curvature regularization, or diffraction claim.',cost_note='Equal ray-evaluation count and rays/grid, not equal wallclock: polynomial intersections cost more. Source development costs inherited, not free.'))
 for path in [Path(__file__),Path('nslens/analysis/final_freeform.py'),Path('nslens/analysis/variable_mirror.py')]:shutil.copy(path,out/path.name)
 source=json.loads(SOURCE.read_text());save(out/'initial_dense.json',audit(source['selected_coarse']['genes'],source['selected_n'],np.zeros(4),source['task']))
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(run,jobs):print(json.dumps(r),flush=True)
 save(out/'completion.json',dict(runs=len(jobs),coarse_calls=1029*len(jobs),dense_calls=2*len(jobs)+1))
if __name__=='__main__':main()
