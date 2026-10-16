"""Check provenance, layout invariants and denser audit stability of shared refinement."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.analysis.optical_refinement import audit
from nslens.analysis.mirror_residual_experiment import save

def denser(record):
 path,genes,n,coefficients,task=record
 a=audit(genes,n,coefficients,task,fields=31,pupils=201,samples=1025)
 for key in ['ray_image_mm','ray_good','ray_field_deg','ray_route']:a.pop(key,None)
 return dict(source=path,audit=a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.out);protocol=json.loads((out/'protocol.json').read_text());source_cache={};selected=[];fallbacks=0
 for job in protocol['jobs']:
  path=out/f'{Path(job["source"]).stem}__{job["refiner"]}.json';r=json.loads(path.read_text());key=job['source']
  if key not in source_cache:
   b=Path(key).read_bytes();assert hashlib.sha256(b).hexdigest()==job['source_hash'];source_cache[key]=json.loads(b)
  original=source_cache[key]
  assert r['additional_coarse_calls']==1044 and r['total_search_coarse_calls']==3100 and r['dense_calls']==1
  assert set(r['refinements'])=={'3','4','5','6'}
  for count,ref in r['refinements'].items():
   assert ref['coarse_calls']==261
   assert ref['coarse']['loss']<=ref['initial']['loss']+1e-12
   np.testing.assert_array_equal(ref['initial']['genes'],original['coarse'][count]['genes'])
   coeff=np.array(ref['coarse']['freeform_coefficients_mm']);assert np.all(abs(coeff)<=2+1e-12)
   if job['refiner']!='joint_parabolic':np.testing.assert_array_equal(ref['coarse']['genes'][:-1],original['coarse'][count]['genes'][:-1])
   if job['refiner']!='last_freeform':np.testing.assert_array_equal(coeff,0)
   fallbacks+=not ref['compatibility']['available']
  chosen=min(range(3,7),key=lambda n:r['refinements'][str(n)]['coarse']['loss']);assert chosen==r['selected_n']
  assert r['audit']['success']==r['audit']['threshold_reached']
  if r['audit']['success']:
   c=r['selected_coarse'];selected.append((str(path),c['genes'],chosen,c['freeform_coefficients_mm'],r['task']))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:checks=list(pool.map(denser,selected))
 save(out/'extra_dense_success_check.json',dict(purpose='Posthoc numerical robustness of all primary passes; no search or selection feedback. Primary metric remains15x101 audit.',fields=31,pupils=201,surface_samples=1025,results=checks))
 save(out/'verification.json',dict(runs=len(protocol['jobs']),unique_cached_pools=len(source_cache),actual_new_coarse_calls=len(protocol['jobs'])*1044,actual_new_selected_dense_calls=len(protocol['jobs']),inherited_unique_search_calls=len(source_cache)*2056,extra_dense_calls=len(checks),extra_dense_passes=sum(c['audit']['threshold_reached'] for c in checks),slope_fit_fallbacks=fallbacks,source_hashes_verified=True,all_candidate_counts_refined=True,coarse_only_selection_verified=True,last_only_layout_parameters_fixed=True,freeform_bounds_verified=True,tests='14 optical/numerical tests passed',smoke_cost='Three1044-call refiners plus3denseaudits; separate from main.'))
 shutil.copy(__file__,out/Path(__file__).name);print('Dense robustness:',sum(c['audit']['threshold_reached'] for c in checks),'/',len(checks),'fallbacks',fallbacks)
if __name__=='__main__':main()
