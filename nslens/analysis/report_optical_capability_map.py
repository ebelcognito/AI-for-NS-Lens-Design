"""Readable per-failure and per-case capability map, with explicit budget limits."""
from pathlib import Path
import json,csv,argparse,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

ARMS=['qwen_cma','qwen_edit','fixed_cma','random_edit']
LABELS=['Qwen → CMA','Qwen direct edits','Fixed CMA','Random edits']
GROUPS=['detector_miss','insufficient_TIR','TIR_margin','blur_or_mixed_route']
NAMES={'detector_miss':'Detector misses','insufficient_TIR':'Insufficient TIR folds','TIR_margin':'TIR-margin losses','blur_or_mixed_route':'Blur / mixed routes'}

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def main(root):
 summary=read(root/'summary.json');tasks={t['id']:t for t in read(root/'tasks_with_initial.json')};rows=summary['cases']
 comparisons=[];effects=[];calls=[];costs=dict(candidate_evaluations=0,candidate_source_rays=0,final_audit_evaluations=0,final_audit_source_rays=0,model_requests=0,input_tokens=0,output_tokens=0,model_failures=0)
 for file in sorted((root/'model_calls').glob('*.json')):
  if file.stem.endswith('_attempts'):continue
  call=read(file);calls.append(call);costs['model_failures']+=call['model_failure']
  for attempt in call['attempts']:
   costs['model_requests']+=1;usage=attempt.get('response',{}).get('usage',{});costs['input_tokens']+=usage.get('prompt_tokens',0);costs['output_tokens']+=usage.get('completion_tokens',0)
 for row in rows:
  task=tasks[row['case_id']];before=task['initial_audits']['stress'];base_diag=before.get('diagnostics',{});baseline_loss=np.asarray(base_diag.get('per_field_exit_energy',[]))
  before_failure=None
  if baseline_loss.size:
   worst=base_diag['worst_throughput_field_index'];before_failure=base_diag['loss_categories'][int(np.argmax(baseline_loss[worst,1:4]))+1]
  for arm in ARMS:
   for seed in [160901,160902]:
    r=read(root/'episodes'/row['case_id']/f'{arm}_{seed}.json');after=r['outcome']['stress'];diag=after.get('diagnostics',{});loss=np.asarray(diag.get('per_field_exit_energy',[]))
    costs['candidate_evaluations']+=r['candidate_evaluations'];costs['candidate_source_rays']+=r['candidate_rays'];costs['final_audit_evaluations']+=r['outcome']['evaluations'];costs['final_audit_source_rays']+=r['outcome']['source_rays']
    record=dict(case_id=row['case_id'],failure_group=row['failure_group'],initial_dense_dominant_exit_loss=before_failure,ancestry=row['ancestry'],source_task_id=row['source_task_id'],arm=arm,seed=seed,success=r['outcome']['success'],initial_dense_pass=row['initial_dense_pass'],before_throughput=before.get('worst_folded_throughput'),after_throughput=after.get('worst_folded_throughput'),before_rms_mm=before.get('worst_spot_rms'),after_rms_mm=after.get('worst_spot_rms'),after_root_support=after.get('support'),model_failure=r['model_failure'],candidate_calls=r['candidate_evaluations'],coarse_success=r['best_search']['feasible'],dense_success=r['outcome']['success'])
    for i,name in enumerate(['accepted','detector_miss','insufficient_TIR','TIR_margin','nonexit']):
     record[f'before_mean_{name}']=float(baseline_loss[:,i].mean()) if baseline_loss.size else None
     record[f'after_mean_{name}']=float(loss[:,i].mean()) if loss.size else None
    comparisons.append(record)
    effects.append(dict(case_id=row['case_id'],ancestry=row['ancestry'],arm=arm,seed=seed,observation=f'observations/{row["case_id"]}.json',action=r['action'],final_genes=r['genes'],verified_outcome=r['outcome'],candidate_records=f'episodes/{row["case_id"]}/{arm}_{seed}.json',label_scope='Only this selected outcome is dense-audited; candidate records are provisional. CMA candidate genes are solver-generated, not Qwen-authored edits.'))
 write(root/'action_outcomes.json',effects);write(root/'costs.json',costs)
 if comparisons:
  with (root/'case_metrics.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
 rows=sorted(rows,key=lambda r:(GROUPS.index(r['failure_group']),r['case_id']))
 values=np.array([[sum(r['passes'][a]) for a in ARMS] for r in rows])
 fig,ax=plt.subplots(figsize=(11,max(6,len(rows)*.35)),layout='constrained')
 colors=ListedColormap(['#f3d7d1','#f5d486','#79bda4']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
 ax.imshow(values,cmap=colors,norm=norm,aspect='auto')
 ax.set_xticks(range(4),LABELS);ax.xaxis.tick_top();ax.set_yticks(range(len(rows)),[f"{r['case_id']}  {NAMES[r['failure_group']]}" for r in rows])
 for i,row in enumerate(rows):
  for j,arm in enumerate(ARMS):
   partial=any(row['material_improvement'][arm]) and not any(row['passes'][arm]);ax.text(j,i,f"{values[i,j]}/2"+(' •' if partial else ''),ha='center',va='center',fontsize=10)
 ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
 fig.suptitle('Verified repairs by optical failure and method\nTwo attempts per state; 16 candidate evaluations per attempt',fontsize=14)
 fig.supxlabel('0/2 = not solved in this bounded test; 1/2 = solved once; 2/2 = solved in both attempts.\n• = material improvement without full repair. Development data; related designs are grouped by ancestry.',fontsize=10)
 fig.savefig(root/'capability_map.png',dpi=180);plt.close(fig)
 lines=['# Optical failure capability map','',f"Completed cases: {summary['completed']}/{summary['expected']}. Frozen Qwen3.8-27B Q8_0; no training.",'', '## Verified repairs by initial failure group','', 'Counts below are states solved at least once in two attempts, not single-attempt success rates. All methods get two attempts and 16 candidates per attempt.','', '| Initial failure group | States | Qwen → CMA | Qwen direct edits | Fixed CMA | Random edits |','|---|---:|---:|---:|---:|---:|']
 for group in GROUPS:
  g=summary['by_failure'].get(group)
  if not g:continue
  lines.append('| '+NAMES[group]+f" | {g['cases']} | "+' | '.join(str(g['solved_any_of_two'][a]) for a in ARMS)+' |')
 lines+=['','## Individual states','', '| State | Failure group | Qwen CMA | Qwen edits | Fixed CMA | Random | Interpretation |','|---|---|---:|---:|---:|---:|---|']
 for row in rows:lines.append(f"| {row['case_id']} | {NAMES[row['failure_group']]} | "+' | '.join(f"{sum(row['passes'][a])}/2" for a in ARMS)+f" | {row['category']} |")
 lines+=['','## Interpretation rules','', '- `qwen_solved_in_test`: at least one Qwen arm obtained a fully verified repair in one of its two attempts. This does not imply a reliable general capability.','- `control_solved_qwen_missed`: a numerical/random arm succeeded, but neither Qwen arm did. This provides evidence of a policy/interface opportunity at this budget.','- `no_tested_method_solved`: unresolved with these actions and budgets. It is not evidence that the optical task is impossible or beyond Qwen in another interface.','- Existing witness-backed routing tasks are known feasible, but the witness may be outside the +/-0.1 one-batch trust region. The imaging targets are not all known feasible.','- Selected failure groups come from initial screening. `case_metrics.csv` also records the dense initial dominant loss; categories can shift with sampling and are not causal surface diagnoses.','', '## Material improvements, including full repairs','', 'Routing improvement requires >=2 percentage points worst-field throughput gain on both audit grids, with no increased maximum normalized constraint violation. Imaging improvement requires >=10% RMS reduction on both grids, throughput >=max(80%, baseline minus 1 percentage point), and no increased maximum violation. The counts below include full repairs as well as partial gains. A partial gain alone is not a repair.','', '| Failure group | Qwen CMA | Qwen edits | Fixed CMA | Random edits |','|---|---:|---:|---:|---:|']
 for group in GROUPS:
  g=summary['by_failure'].get(group)
  if g:lines.append('| '+NAMES[group]+' | '+' | '.join(str(g['material_improvement_any'][a]) for a in ARMS)+' |')
 lines+=['','## Scope and limitations','', 'This is development data: 24 selected states, shared parent settings, and two seeds. No untouched generalization claim. The first two routing groups span starting-throughput quantiles; the TIR-margin group is scarcer. Six imaging states include related C designs.','', 'Qwen gets physical face IDs, geometry, codec bounds, per-field optical metrics, disjoint exit-loss categories and face-level footprint/TIR ranges. No hidden witness, corruption recipe, task label or alternative outcomes enter its observation. Rationales are hypotheses, not verified explanations.','', 'CMA uses two sequential populations of eight. Direct editing proposes sixteen alternatives in one model response, all from the same starting state; it has no within-attempt feedback. Equal candidate counts do not mean equal token cost or identical search structure. This pilot does not test iterative direct-edit reasoning, freeform coefficient control, or a trained policy.','', 'Candidate scoring uses 9 fields × 25 rays. Final success requires 17 × 101 and 33 × 151 audits and repeat stress at doubled branch capacity. TIR and branching are re-evaluated at every geometry. Only selected candidates are densely audited; apparent coarse successes may disappear.','', 'Search ranks infeasible candidates by maximum normalized violation, then summed violation. On severe blur cases this can favor lower RMS at the expense of throughput. Such outcomes fail the final constraints; this objective limitation affects all four arms.','', 'The legacy known_feasible_witness flag concerns source feasibility and is not proof of the added imaging target. In particular, C15300 does not have a certified <=0.5 mm witness in this pilot.','', 'Bounded stochastic tests cannot establish model inability. A failed model response after three attempts is retained as a model failure with no numerical fallback.','', '## Costs','', f"Candidate evaluations: {costs['candidate_evaluations']}; candidate source rays: {costs['candidate_source_rays']}; final-audit evaluations: {costs['final_audit_evaluations']}; final-audit source rays: {costs['final_audit_source_rays']}.",f"Model requests including schema retries: {costs['model_requests']}; prompt tokens: {costs['input_tokens']}; completion tokens: {costs['output_tokens']}; exhausted decisions: {costs['model_failures']}.",'', 'These totals exclude initial screening, initial task audits and archived preflight work, whose artifacts are retained. Context caching and wall-clock concurrency affect timing; ray counts are launched source rays, not all internal branches.','', '## Files','', '- `capability_map.png`: per-state success matrix.','- `case_metrics.csv`: before/after physical metrics and loss components for every attempt.','- `action_outcomes.json`: model/numerical actions paired with verified selected outcomes.','- `observations/`, `model_calls/`, `episodes/`: full provenance; candidate-level results are provisional.','- `protocol.json`, `selection.json`, `source_snapshot/`: declared controls and frozen executable source.']
 (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 # Small self-contained browser view with an inline filter and links to raw cases.
 body=[]
 for row in rows:
  values=''.join(f'<td>{sum(row["passes"][a])}/2</td>' for a in ARMS)
  body.append(f'<tr data-group="{row["failure_group"]}"><td><a href="observations/{row["case_id"]}.json">{row["case_id"]}</a> · <a href="before_after/{row["case_id"]}_before_after.png">before/after</a></td><td>{html.escape(NAMES[row["failure_group"]])}</td>{values}<td>{html.escape(row["category"])}</td></tr>')
 options='<option value="all">All failure types</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in NAMES.items())
 page='''<!doctype html><meta charset="utf-8"><title>Optical capability map</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#24333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f5}select{padding:8px;margin:12px 0}a{color:#087e83}img{max-width:100%}</style><h1>Which optical failures improved?</h1><p>Frozen Qwen; two attempts per state; 16 candidates per attempt. A failed test is not proof of inability.</p><label>Show: <select id="filter">'''+options+'''</select></label><table><thead><tr><th>State</th><th>Initial failure</th><th>Qwen CMA</th><th>Qwen edits</th><th>Fixed CMA</th><th>Random edits</th><th>Finding</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table><p><a href="RESULTS.md">Full report and limitations</a> · <a href="case_metrics.csv">Before/after metrics</a> · <a href="PHYSICAL_EFFECTS.md">Physical changes and matched cases</a></p><img src="capability_map.png" alt="Per-state verified repair matrix"><script>document.getElementById('filter').onchange=function(){document.querySelectorAll('tbody tr').forEach(r=>r.hidden=this.value!=='all'&&r.dataset.group!==this.value)}</script>'''
 (root/'index.html').write_text(page)
 print('report saved',root,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
