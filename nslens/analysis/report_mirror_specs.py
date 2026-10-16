"""Summarize predeclared optical-task transfer tests; no model fitting."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=['restart','same_count','cross_count','source_reuse']
def main():
 out=Path(sys.argv[1]);protocol=json.loads((out/'protocol.json').read_text());rows=[json.loads(p.read_text()) for p in sorted(out.glob('*__*__*.json'))];expected=protocol['rounds']*(1+12*protocol['generations'])
 assert len(rows)==len(protocol['tasks'])*len(protocol['seeds'])*len(ARMS)
 assert all(r['coarse_calls']==expected and r['geometry_checks']==expected for r in rows)
 summaries={};paired={}
 def aggregate(rr,key='selected_audit'):
  aa=[r[key] for r in rr];finite=[a['worst_mapping_rms_mm'] for a in aa if a['worst_mapping_rms_mm'] is not None]
  return dict(runs=len(rr),median_loss=float(np.median([a['loss'] for a in aa])),routing_pass=sum(a['routing_pass'] for a in aa),imaging_pass={str(t):sum(a['imaging_pass'][str(t)] for a in aa) for t in [.1,.5,1.]},median_worst_mapping_rms_mm_measurable=float(np.median(finite)) if finite else None,measurable=len(finite),count_choices={str(n):sum(r.get('selected_n')==n for r in rr) for n in range(3,7)})
 for task in protocol['tasks']:
  summaries[task]={};paired[task]={}
  for arm in ARMS:
   rr=[r for r in rows if r['taskname']==task and r['arm']==arm];summaries[task][arm]=aggregate(rr)
   summaries[task][arm]['by_count']={str(n):aggregate([dict(selected_audit=r['audits'][str(n)],selected_n=n) for r in rr]) for n in range(3,7)}
  donor=[]
  for r in rows:
   if r['taskname']==task and r['arm']=='source_reuse':
    k=min(r['source_initial'],key=lambda k:r['source_initial'][k]['loss']);donor.append(dict(selected_audit=r['source_audits'][k],selected_n=int(k)))
  summaries[task]['source_without_adaptation']=aggregate(donor)
  for baseline in ['restart','same_count','cross_count']:
   d=[];lograt=[]
   for seed in protocol['seeds']:
    rr={r['arm']:r for r in rows if r['taskname']==task and r['seed']==seed};a=rr['source_reuse']['selected_audit']['loss'];b=rr[baseline]['selected_audit']['loss'];d.append(a-b);lograt.append(np.log1p(a)-np.log1p(b))
   rng=np.random.default_rng(49);d=np.array(d);boot=np.mean(rng.choice(d,(10000,len(d)),replace=True),axis=1)
   paired[task][baseline]=dict(source_wins=int(sum(d<0)),ties=int(sum(d==0)),mean_loss_delta=float(d.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),mean_log1p_delta=float(np.mean(lograt)))
 # Shared donor IDs across tasks mean task/seed combinations are not independent.
 heldout=[k for k in protocol['tasks'] if k!='source_control'];block=[]
 for seed in protocol['seeds']:
  deltas=[]
  for task in heldout:
   rr={r['arm']:r for r in rows if r['taskname']==task and r['seed']==seed};deltas.append(np.log1p(rr['source_reuse']['selected_audit']['loss'])-np.log1p(rr['same_count']['selected_audit']['loss']))
  block.append(np.mean(deltas))
 rng=np.random.default_rng(177);boot=np.mean(rng.choice(block,(10000,len(block)),replace=True),axis=1)
 overall=dict(heldout_tasks=heldout,independent_donor_seed_blocks=len(block),source_vs_same_count_mean_log1p_delta=float(np.mean(block)),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),winning_seed_blocks=int(sum(np.array(block)<0)))
 for name,obj in [('summary.json',summaries),('paired.json',paired),('heldout_summary.json',overall)]: (out/name).write_text(json.dumps(obj,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(protocol['tasks']));width=.18
 for i,arm in enumerate(ARMS):
  vals=[summaries[t][arm]['median_loss'] for t in protocol['tasks']];axes[0].bar(x+(i-1.5)*width,vals,width,label=arm)
  vals=[summaries[t][arm]['imaging_pass']['0.5']/summaries[t][arm]['runs'] for t in protocol['tasks']];axes[1].bar(x+(i-1.5)*width,vals,width,label=arm)
 axes[0].set_yscale('log');axes[0].set_ylabel('Median selected dense loss (log scale)');axes[1].set_ylabel('Fraction passing 0.5mm mapping threshold');axes[1].set_ylim(0,1.05)
 for ax in axes:ax.set_xticks(x,list(protocol['tasks']),rotation=20,ha='right')
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 report=['OPTICAL SPECIFICATION GENERALIZATION','','Scope: fixed2D reflective parameterization, variable3–6 mirror count, changes to task requirements. No learned sampler; this tests transfer of existing designs as a prerequisite/baseline.','Task table: focal mapping(mm), entrance aperture(mm), half field(degrees):',json.dumps(protocol['tasks'],indent=2),'','Methods: fresh restarts; same-count reuse within task; adjacent-count reuse within task; source-task same-count initialization followed by same-count reuse. Source donors come only from original50mm/27.78mm/+/-4deg task.','Main question: does source experience help find ONE design on a new optical specification under equal online budgets? Select count using coarse penalized merit before auditing.','All methods share1 initial+192 refinement traces per round,8 rounds unless protocol says otherwise. Source-only zero-adaptation audits are diagnostic and never feed search.','','SELECTED DESIGN RESULTS','task | method | median dense loss | routing/geometry passes | imaging passes at0.1/0.5/1.0mm']
 for task,ss in summaries.items():
  for arm,v in ss.items():report.append(f'{task} | {arm} | {v["median_loss"]:.5g} | {v["routing_pass"]}/{v["runs"]} | '+ '/'.join(str(v['imaging_pass'][str(t)]) for t in [.1,.5,1.]))
 report+=['','Thresholds measure worst-field RMS error relative to desired signed image mapping, on received rays. Every pass additionally requires>=99% support at every field and no sampled mirror crossings. They are diagnostic thresholds, not diffraction or3D imaging specifications.','Mapping RMS, spot spread and centroid error are recorded separately. Missing fields yield unavailable metrics and cannot pass.','Source-control task is not held out. Remaining four task specifications were fixed before the run; no hyperparameter fitting on target outcomes. Same donor seed appears across tasks, so pooled inference uses donor-seed blocks.','','HELDOUT PAIRED SUMMARY',json.dumps(overall,indent=2),'Negative log1p loss difference favors source reuse. This transformed statistic avoids averaging raw errors across different specifications; it is not an optical-quality percentage. Small sample, exploratory bootstrap, no multiplicity correction.','','COST',f'New coarse traces: {sum(r["coarse_calls"] for r in rows)}. New dense audits: {sum(r["dense_calls"] for r in rows)}. Geometry checks accompany every trace and audit.',f'Source donor generation: {protocol["source_cost"]} prior coarse traces across the selected donors, plus their original audits. Source reuse is NOT a zero-cost training method. The equal-online-budget comparison does not demonstrate total-compute savings or an amortization break-even point. Earlier exploratory project costs are also not included in the source generation number.','','LIMITS','No arbitrary optical tasks, freeform3D mirrors, independent detector tilt/decenter, packaging envelope, coating losses or tolerancing. Fixed60mm full mirror width and25–105mm leg bounds may limit harder tasks.','Counts are allocated uniformly with an ascending schedule; this is not learned architecture allocation. Source donor reuse is an explicit warm start, not learned mutation or unsupervised learning.','A transfer gain can come entirely from better initial fitness. Source-without-adaptation rows help distinguish this; no claim that the optimizer learned a transferable direction.','','VERIFICATION','32 default-evaluator parity cases against saved previous implementation passed. Task-specific metric tests and prior mirror tests passed. All run counts, trace ledgers and geometry-check counts reconciled.','Smoke dataset separate and not included. No dense audit affects candidate selection.','','REPRODUCE',f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.mirror_spec_experiment --out NEW_DIRECTORY --seeds {len(protocol["seeds"])} --rounds {protocol["rounds"]} --gens {protocol["generations"]}','Source data hashes in protocol.json. Report: .venv/bin/python -m nslens.analysis.report_mirror_specs OUTPUT_DIRECTORY']
 if (out/'source_cost.json').exists():report+=['','DETAILED SOURCE COST LEDGER',(out/'source_cost.json').read_text()]
 (out/'RESULTS.txt').write_text('\n'.join(report)+'\n');(out/'verification.json').write_text(json.dumps(dict(runs=len(rows),coarse_calls=sum(r['coarse_calls'] for r in rows),dense_calls=sum(r['dense_calls'] for r in rows),expected_coarse_per_run=expected,ledgers_passed=True),indent=2));print(json.dumps({t:{a:{k:v for k,v in x.items() if k!='by_count'} for a,x in ss.items()} for t,ss in summaries.items()},indent=2))
if __name__=='__main__':main()
