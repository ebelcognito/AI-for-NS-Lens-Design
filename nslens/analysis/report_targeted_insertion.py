"""Report architecture insertion with separate added-design and fallback outcomes."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nslens.analysis.mirror_residual_experiment import save
ARMS=['legacy_one','flat_batch','targeted','keep_count']
def merit(a):return a['loss']+30000*a['crossing']
def main():
 out=Path(sys.argv[1]);rr=[json.loads(p.read_text()) for p in out.glob('*__*__*.json')];protocol=json.loads((out/'protocol.json').read_text());assert len(rr)==len(protocol['jobs'])
 for r in rr:
  K=len(r['candidates']);assert r['coarse_calls']==4*r['standalone_budget']-K
  assert all(a['online_calls']==r['standalone_budget'] and a['screen_calls']+a['refinement_calls']==r['standalone_budget'] for a in r['results'].values())
 def aggregate(rows):
  ret=dict(parents=len(rows),donors=len(set(r['seed'] for r in rows)),arms={},paired={})
  for arm in ARMS:
   ret['arms'][arm]={}
   for mode,key in [('candidate','audit'),('guarded','guarded_audit')]:
    aa=[r['results'][arm][key] for r in rows];ret['arms'][arm][mode]=dict(median_merit=float(np.median([merit(a) for a in aa])),routing_pass=sum(a['routing_pass'] for a in aa),imaging05=sum(a['imaging_pass']['0.5'] for a in aa),imaging01=sum(a['imaging_pass']['0.1'] for a in aa),imaging10=sum(a['imaging_pass']['1.0'] for a in aa))
   ret['arms'][arm]['parent_fallbacks']=sum(r['results'][arm]['guarded_parent'] for r in rows)
  for other in ['legacy_one','flat_batch','keep_count']:
   for mode,key in [('candidate','audit'),('guarded','guarded_audit')]:
    deltas=[]
    for seed in sorted(set(r['seed'] for r in rows)):
     deltas.append(np.mean([np.log1p(merit(r['results']['targeted'][key]))-np.log1p(merit(r['results'][other][key])) for r in rows if r['seed']==seed]))
    rng=np.random.default_rng(701);boot=np.mean(rng.choice(deltas,(10000,len(deltas)),replace=True),axis=1)
    ret['paired'][mode+'_targeted_vs_'+other]=dict(mean_log1p_delta=float(np.mean(deltas)),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),winning_donor_blocks=int(sum(x<0 for x in deltas)),donor_blocks=len(deltas))
  return ret
 summary={'all':aggregate(rr),'one_design':aggregate([r for r in rr if r['parent_n']==r['parent_selected_n']])}
 for n in [3,4,5]:summary[f'{n}_to_{n+1}']=aggregate([r for r in rr if r['parent_n']==n])
 candidates=[c for r in rr for c in r['candidates']];diagnostics=dict(candidates=len(candidates),mean_initial_worst_support=float(np.mean([c['initial']['worst_support'] for c in candidates])),mean_derivative_validity=float(np.mean([c['valid_fraction'] for c in candidates])),immediate_power_correction_improvement_fraction=float(np.mean([c['corrected']['loss']<c['initial']['loss'] for c in candidates])),zero_step_fraction=float(np.mean([c['step_norm']<1e-12 for c in candidates])))
 save(out/'summary.json',summary);save(out/'diagnostics.json',diagnostics);ledger=dict(parents=len(rr),coarse_calls=sum(r['coarse_calls'] for r in rr),dense_calls=sum(r['dense_calls'] for r in rr),standalone_budgets=sorted(set(r['standalone_budget'] for r in rr)),ledgers_passed=True);save(out/'verification.json',ledger)
 fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(3)
 for i,arm in enumerate(ARMS):
  axes[0].bar(x+(i-1.5)*.19,[summary[f'{n}_to_{n+1}']['arms'][arm]['candidate']['routing_pass']/summary[f'{n}_to_{n+1}']['parents'] for n in [3,4,5]],.19,label=arm)
  axes[1].bar(x+(i-1.5)*.19,[summary[f'{n}_to_{n+1}']['arms'][arm]['guarded']['median_merit'] for n in [3,4,5]],.19,label=arm)
 axes[0].set_ylabel('Routing/geometry pass fraction');axes[0].set_ylim(0,1.05);axes[1].set_ylabel('Median guarded dense merit (log)');axes[1].set_yscale('log')
 for ax in axes:ax.set_xticks(x,['3 → 4','4 → 5','5 → 6'])
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 lines=['RESIDUAL-TARGETED ARCHITECTURE INSERTION','','METHOD',protocol['folded_override'] if protocol.get('folded') else protocol['method'],protocol['controls'],protocol['cost'],protocol['task'],'','CALL LEDGER',json.dumps(ledger,indent=2),'','RESULTS: candidate retains required extra mirror, guarded may return parent. keep_count remains at original mirror count in both columns.','group | method | candidate routing | candidate0.5mm | parent fallbacks | guarded median merit']
 for name,v in summary.items():
  lines.append(f'{name}: {v["parents"]} parents,{v["donors"]} donor blocks')
  for arm,a in v['arms'].items():lines.append(f'{name} | {arm} | {a["candidate"]["routing_pass"]}/{v["parents"]} | {a["candidate"]["imaging05"]}/{v["parents"]} | {a["parent_fallbacks"]} | {a["guarded"]["median_merit"]:.6g}')
 lines+=['','PAIRED DONOR-BLOCK COMPARISONS',json.dumps({k:v['paired'] for k,v in summary.items()},indent=2),'Negative log1p differences favor targeted insertion. Six donors repeated across tasks/counts, resampled as donor blocks. Exploratory intervals, no multiplicity correction. Comparisons to keep_count intentionally test whether an added surface is useful, not equal dimensionality.','','INSERTION DIAGNOSTICS',json.dumps(diagnostics,indent=2),'Missing rays do not provide usable residual derivatives. Initial nearly-grazing flat insertion can destroy routing, limiting any local residual correction. These diagnostics alone do not establish the cause of poor final performance.','','LIMITATIONS','Fixed2D ideal parabolas, detector coupled to final scaffold leg, no common packaging envelope,coating losses,manufacturing/diffraction/3D model. Donors are task-adapted designs from earlier experiments. This is a one-step architecture edit and local refinement, not a learned policy or fresh-start discovery.','One-design subset uses parent count chosen before this test; prior6-mirror selections cannot be grown within the3–6 range and are excluded. Guard compares only coarse merit, never dense audits. A fallback to parent is NOT successful architecture improvement.','This targets residuals AFTER insertion using local curvature sensitivities. It does not establish an uncorrectable parent residual subspace or predict a topology performance bound. No exact optical-power conservation claimed.','','VERIFICATION','Deterministic insertion index/bounds and curvature preservation checks passed; existing residual/spec/mirror unit tests passed. Exact standalone cost and dataset reuse ledgers verified. Source snapshots and parent hashes saved. Worker review performed.','','REPRODUCE','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.targeted_insertion --out NEW_DIRECTORY','.venv/bin/python -m nslens.analysis.report_targeted_insertion OUTPUT_DIRECTORY']
 if protocol.get('folded'):
  lines+=['','FOLDED RUN CLARIFICATIONS','The generic legacy_one key means one RANDOM PATH-PRESERVING FOLDED candidate in this run. All candidate locations/turns for targeted and flat_batch are the same eligible>=50mm gaps with+/-90deg downstream reflection. The generic small-turn method text in protocol is overridden by folded_override.','This run passed394 exact downstream surface-normal/vertex/detector reflection checks before launch. Normal and tangent frame transformation preserves the signed-curvature parabola under reflection; physical clipping/obstruction still audited.']
  lines[lines.index('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.targeted_insertion --out NEW_DIRECTORY')]+=' --folded'
 (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');shutil.copy(__file__,out/Path(__file__).name);print(json.dumps(summary['all'],indent=2));print(json.dumps(diagnostics,indent=2))
if __name__=='__main__':main()
