"""Rerank cached discovery pools after a common, fully budgeted refinement stage."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nslens.analysis.optical_refinement import refine_design,audit,REFINERS,DEFAULT_BUDGET
from nslens.analysis.mirror_residual_experiment import save
SOURCE=Path('experiments/hard_confirmation_20261006')

def run(job):
 start=time.monotonic();path=Path(job['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==job['source_hash'];source=json.loads(path.read_text());j=source['job'];task=source['task'];refined={}
 for n in range(3,7):
  # Same independent refiner seed across sampling methods and refinement families.
  seed=120000000+j['repeat']*10000+j['seed']*100+n
  refined[str(n)]=refine_design(source['coarse'][str(n)]['genes'],n,task,seed,refiner=job['refiner'])
 chosen=min(range(3,7),key=lambda n:refined[str(n)]['coarse']['loss']);best=refined[str(chosen)]['coarse']
 # No audit of unselected counts; final-design choice is already frozen.
 final=audit(best['genes'],chosen,best['freeform_coefficients_mm'],task);final['success']=final['threshold_reached'];final['package_pass']=True
 extra=sum(r['coarse_calls'] for r in refined.values());assert extra==1044
 result=dict(job=job,source_job=j,task=task,refinements=refined,selected_n=chosen,selected_coarse=best,audit=final,source_initial_audit=source['audit'],source_original_selected_n=source['selected_n'],additional_coarse_calls=extra,total_search_coarse_calls=source['coarse_calls']+extra,dense_calls=1,seconds=time.monotonic()-start)
 save(Path(job['out'])/f'{path.stem}__{job["refiner"]}.json',result)
 return dict(task=j['task'],arm=j['arm'],donor=j['seed'],repeat=j['repeat'],refiner=job['refiner'],success=final['success'],error=final['worst_mapping_rms_mm'],seconds=result['seconds'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 files=sorted(SOURCE.glob('*__*__*__r*.json'))
 if a.smoke:files=[SOURCE/'field8__optical_residual__1__r1.json']
 jobs=[dict(source=str(path),source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),refiner=refiner,out=str(out)) for path in files for refiner in REFINERS]
 old=json.loads((SOURCE/'protocol.json').read_text())
 save(out/'protocol.json',dict(jobs=jobs,source=str(SOURCE),refiners=REFINERS,default_refiner='last_freeform',tasks=old['tasks'],arms=old['arms'],donors=old['donors'],repeats=old['repeats'],scope='Cached first-stage search trajectories from previous confirmation, not newly independent discovery runs. All four3–6mirror incumbents are refined before choosing the one final design. No initial1mm filtering, no best-count dense-audit selection. Paired on task/donor/repeat; same sourcepool in each refiner.',refinement='Shared refine_design default last_freeform: last curvature+4bounded polynomialcoefficients, slope fit with3line-search fractions thenCMA. Last-parabolic and joint-parabolic controls use the same261call budget and105rays/trace, including5initialization traces. last arms preserve layout and detector. joint_parabolic changes all3n original genes.',budget='2056 inherited27-ray search calls +4*261 new105-ray refinement calls=3100 per returned design; one new15x101 dense audit. All refiner/sampling comparisons have same allocation. Different numerical kernels have different wallclock cost.',ray_evaluations_per_returned_design=2056*27+1044*105,source_development_cost=old['source_cost'],old_results='Prior one-audit results at2056calls are descriptive only; they have less total budget. The equal-budget joint_parabolic control is needed for a causal refiner comparison.',seeds='120000000+repeat*10000+donor*100+mirrorcount; shared across arms/refiners, disjoint prior refiner seeds. Same tasks intentionally share seeds; bootstrap donor blocks.',quality='Post-refinement worst-field mapping RMS<=1mm,>=99% worst-field support,no sampled crossing. Report continuous quality/routing too; no pre-refinement layout pass/fail screening.',limits='Ideal2D parabolic mirrors with polynomial final-mirror profile. Extra terms±2mm,3rd..6th powers of u/30. No3D,manufacturing,tolerance,diffraction or enclosure constraint in these two cases. No learned model. Target environments were selected after an earlier sweep. Eight correlated donor families.'))
 for f in [Path(__file__),Path('nslens/analysis/optical_refinement.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/search/cma_me.py')]:shutil.copy(f,out/f.name)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(run,jobs):print(json.dumps(result),flush=True)
 save(out/'completion.json',dict(runs=len(jobs),new_coarse_calls=len(jobs)*1044,new_dense_calls=len(jobs),unique_cached_search_pools=len(files),inherited_discovery_calls_unique=len(files)*2056))
if __name__=='__main__':main()
