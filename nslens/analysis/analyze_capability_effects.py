"""Summarize matched case outcomes and physical changes, without causal attribution."""
import json,argparse
from pathlib import Path
import numpy as np

ARMS=['qwen_cma','qwen_edit','fixed_cma','random_edit']

def main(root):
 read=lambda p:json.loads(p.read_text());summary=read(root/'summary.json');tasks={t['id']:t for t in read(root/'tasks_with_initial.json')}
 groups={};examples=[];coarse_rejections=[]
 for row in summary['cases']:
  task=tasks[row['case_id']];before=task['initial_audits']['stress'];episodes=[]
  for arm in ARMS:
   for seed in [160901,160902]:
    r=read(root/'episodes'/row['case_id']/f'{arm}_{seed}.json');episodes.append(r)
    if r['best_search']['feasible'] and not r['outcome']['success']:coarse_rejections.append(dict(case_id=row['case_id'],arm=arm,seed=seed))
  if not row['initial_dense_pass']:
   group=groups.setdefault(row['failure_group'],dict(qwen_only=[],controls_only=[],both=[],neither=[],attempt_passes={a:0 for a in ARMS},attempts_per_arm=0))
   q=any(row['passes']['qwen_cma']+row['passes']['qwen_edit']);c=any(row['passes']['fixed_cma']+row['passes']['random_edit'])
   group['both' if q and c else 'qwen_only' if q else 'controls_only' if c else 'neither'].append(row['case_id']);group['attempts_per_arm']+=2
   for arm in ARMS:group['attempt_passes'][arm]+=sum(row['passes'][arm])
  for arm in ARMS:
   eligible=[e for e in episodes if e['arm']==arm]
   def score(e):
    m=e['outcome']['stress'];return (e['outcome']['success'],-max(m['violations'].values()),m['worst_folded_throughput'],-m['worst_spot_rms'])
   best=max(eligible,key=score);after=best['outcome']['stress'];names=before['diagnostics']['loss_categories'];bl=np.mean(before['diagnostics']['per_field_exit_energy'],axis=0);al=np.mean(after['diagnostics']['per_field_exit_energy'],axis=0)
   examples.append(dict(case_id=row['case_id'],source_task_id=row['source_task_id'],failure_group=row['failure_group'],arm=arm,seed=best['seed'],success=best['outcome']['success'],throughput_before=before['worst_folded_throughput'],throughput_after=after['worst_folded_throughput'],rms_before=before['worst_spot_rms'],rms_after=after['worst_spot_rms'],mean_energy_changes={n:float(a-b) for n,a,b in zip(names,al,bl)},selection='Best of two by verified outcome, for descriptive display only.'))
 result=dict(groups=groups,examples=examples,coarse_success_rejected=coarse_rejections)
 (root/'physical_effects.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# Matched cases and physical changes','','Best-of-two examples are descriptive selections after auditing. They do not measure a single-call success rate.','','## Overlap of repaired states','','| Initial failure | Both Qwen and controls | Qwen only | Controls only | Neither |','|---|---:|---:|---:|---:|']
 for name,g in groups.items():lines.append('| '+name+' | '+' | '.join(str(len(g[k])) for k in ['both','qwen_only','controls_only','neither'])+' |')
 lines+=['','## Best verified outcome for each method and state','','Throughput is the worst field; RMS is worst-field RMS spot radius. Each before/after pair is measured on the same 33 × 151 grid. Full success also requires the separate 17 × 101 audit and capacity agreement.','','| State | Method | Throughput before → after | RMS mm before → after | Full repair |','|---|---|---:|---:|---|']
 for e in examples:lines.append(f"| {e['case_id']} | {e['arm']} | {e['throughput_before']:.1%} → {e['throughput_after']:.1%} | {e['rms_before']:.3f} → {e['rms_after']:.3f} | {'yes' if e['success'] else 'no'} |")
 lines+=['','## Sparse-grid false passes','',f'{len(coarse_rejections)} selected outcomes passed search but failed final verification. See physical_effects.json for the individual attempts.','', 'Energy-loss changes in physical_effects.json are descriptive, field-averaged partitions. They do not establish a causal surface diagnosis.']
 (root/'PHYSICAL_EFFECTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
