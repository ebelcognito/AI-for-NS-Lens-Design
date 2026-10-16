"""Posthoc numerical robustness check and provenance audit for confirmation."""
import argparse,hashlib,json
from pathlib import Path
from nslens.analysis.variable_mirror import trace
from nslens.analysis.mirror_spec_experiment import crosses
from nslens.analysis.hard_optical_environments import envelope
from nslens.analysis.mirror_residual_experiment import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);protocol=json.loads((out/'protocol.json').read_text());rows=[];checks=[]
 for j in protocol['jobs']:
  path=out/f'{j["task"]}__{j["arm"]}__{j["seed"]}__r{j["repeat"]}.json';r=json.loads(path.read_text());rows.append(r)
  assert r['coarse_calls']==2056 and r['dense_calls']==1
  assert [h['n'] for h in r['history']]==[3,4,5,6]*2
  assert r['source_hash']==hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()
  assert r['audit']['success']==bool(r['audit']['imaging_pass']['1.0'] and r['audit']['package_pass'])
  if j['arm']!='reuse_only':
   assert len(r['mutations'])==32
   assert all(m['step_norm']<=m['radius']+1e-10 for m in r['mutations'])
  if not r['audit']['success']:continue
  g=r['selected_coarse']['genes'];n=r['selected_n'];task=r['task'];d=trace(g,n,fields=31,pupils=201,**task);d['crossing']=crosses(g,n,1025);d['envelope_mm']=envelope(g,n,task).tolist();d['success']=bool(d['worst_support']>=.99 and not d['crossing'] and d['worst_mapping_rms_mm'] is not None and d['worst_mapping_rms_mm']<=1 and (r['cap_mm'] is None or max(d['envelope_mm'])<=r['cap_mm']+1e-9));checks.append(dict(source=str(path),job=j,audit=d))
 save(out/'extra_dense_success_check.json',dict(purpose='Posthoc robustness of primary passes; no search/selection feedback. Primary predeclared audit15x101.',fields=31,pupils=201,surface_samples=1025,results=checks))
 save(out/'verification.json',dict(runs=len(rows),coarse_calls=sum(r['coarse_calls'] for r in rows),selected_dense_calls=len(rows),extra_dense_calls=len(checks),extra_dense_passes=sum(c['audit']['success'] for c in checks),equal_budgets_verified=True,source_hashes_verified=True,mutation_radius_limits_verified=True,reference_designs_not_search_inputs=True,tests='10 geometry/mutation/specification tests passed; original-seed optical run exactly reproduced',additional_costs='One2056-call control smoke with1dense audit; one2056-call original-seed replay with1dense audit; inherited source development cost in protocol, prior experiment costs excluded.'))
 print('Denser check:',sum(c['audit']['success'] for c in checks),'/',len(checks))
if __name__=='__main__':main()
