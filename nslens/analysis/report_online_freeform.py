"""Report and numerically verify fresh searches using in-loop freeform refinement."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.analysis.optical_refinement import audit
from nslens.analysis.mirror_residual_experiment import save
ARMS=['reuse_only','optical_residual','direction_random']
TASKS=['field8','aperture42']

def check_dense(r):
 c=r['selected_coarse'];a=audit(c['genes'],r['selected_n'],c['freeform_coefficients_mm'],r['task'],fields=31,pupils=201,samples=1025)
 for key in ['ray_good','ray_image_mm','ray_route','ray_field_deg']:a.pop(key,None)
 return dict(job=r['job'],audit=a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();out=Path(a.out);protocol=json.loads((out/'protocol.json').read_text());rows=[]
 for j in protocol['jobs']:
  r=json.loads((out/f'{j["task"]}__{j["arm"]}__{j["seed"]}__r{j["repeat"]}.json').read_text());rows.append(r)
  assert r['coarse_calls']==4144 and r['discovery_calls']==2056 and r['refinement_calls']==2088 and r['dense_calls']==1
  assert r['source_hash']==hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==protocol['source_sha256'][r['source']]
  assert [h['n'] for h in r['history']]==[3,4,5,6]*2
  for h in r['history']:assert h['discovery_calls']==257 and h['refinement_calls']==261
  for n in range(3,7):
   hist=[h for h in r['history'] if h['n']==n];assert hist[1]['incumbent_loss']<=hist[0]['incumbent_loss']
  assert r['selected_n']==min(map(int,r['coarse']),key=lambda n:r['coarse'][str(n)]['loss'])
  assert r['audit']['success']==r['audit']['threshold_reached']
 assert len(rows)==48
 summary={}
 for t in TASKS+['pooled_descriptive']:
  summary[t]={}
  for arm in ARMS:
   rr=[r for r in rows if r['job']['arm']==arm and (t=='pooled_descriptive' or r['job']['task']==t)];rms=[r['audit']['worst_mapping_rms_mm'] for r in rr if r['audit']['worst_mapping_rms_mm'] is not None]
   summary[t][arm]=dict(successes=sum(r['audit']['success'] for r in rr),trials=len(rr),routing_passes=sum(r['audit']['routing_pass'] for r in rr),median_error_mm=float(np.median(rms)),mirror_counts=[r['selected_n'] for r in rr])
 comparisons={};lookup={(r['job']['task'],r['job']['arm'],r['job']['seed']):r for r in rows};rng=np.random.default_rng(193)
 for control in ['reuse_only','direction_random']:
  per_donor=[np.mean([int(lookup[t,'optical_residual',d]['audit']['success'])-int(lookup[t,control,d]['audit']['success']) for t in TASKS]) for d in range(8)];bs=np.mean(rng.choice(per_donor,(10000,8),replace=True),axis=1)
  comparisons['optical_vs_'+control]=dict(delta=float(np.mean(per_donor)),donor_blocks=8,donor_deltas=per_donor,bootstrap95=np.quantile(bs,[.025,.975]).tolist())
 with ProcessPoolExecutor(max_workers=a.workers) as pool:checks=list(pool.map(check_dense,[r for r in rows if r['audit']['success']]))
 save(out/'summary.json',summary);save(out/'paired_comparisons.json',comparisons);save(out/'extra_dense_success_check.json',checks)
 save(out/'verification.json',dict(runs=48,discovery_calls=48*2056,refinement_calls=48*2088,total_coarse_calls=48*4144,dense_calls=48,extra_dense_calls=len(checks),extra_dense_passes=sum(c['audit']['threshold_reached'] for c in checks),all_ledgers_source_hashes_and_count_selection_verified=True))
 lines=['Fresh searches with shared freeform refinement inside the loop','', 'One fresh optimizer repeat(index4) on eight existing donor families, two specifications, three proposal methods. Each round performs257 parabolic discovery evaluations, then261 shared freeform-refinement evaluations BEFORE updating its incumbent. Final design selected by refined coarse merit only.4144 total calls per run; only one dense audit after selection.', '', 'Proposals still operate on parabolic genes: higher-order terms are reset on each new discovery trial and refit. This is a limitation of the sampler representation, not freeform-aware mutation. All three methods share the same refiner and budget.', '', 'Successful final designs (<=1mm mapping RMS,>=99% support,no sampled crossing):']
 for t,arms in summary.items():
  lines.append(t)
  for arm,v in arms.items():lines.append(f"  {arm}: {v['successes']}/{v['trials']}; routing {v['routing_passes']}/{v['trials']}; median RMS {v['median_error_mm']:.3f}mm")
 lines+=['','Paired optical-mutation differences; donor-block bootstrap, only8families:',json.dumps(comparisons,indent=2),'',f'Denser numerical audit: {sum(c["audit"]["threshold_reached"] for c in checks)}/{len(checks)} primary passes survive31fields x201rays and1025crossing samples.','', 'The cached-pool comparison had a different placement and smaller total budget(3100calls). Compare methods within each experiment; do not attribute raw differences between these experiments to in-loop placement alone. Source donor development costs are inherited and recorded separately in protocol. This remains2D ideal optics, with no manufacturing,3D,diffraction or tolerance guarantee.']
 (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');shutil.copy(__file__,out/Path(__file__).name);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
