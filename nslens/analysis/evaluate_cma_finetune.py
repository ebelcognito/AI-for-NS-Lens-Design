"""Simulator score the frozen and trained action policies on development validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from nslens.workbench.interactive_cma_pilot import read,write,finalize
from nslens.workbench.cma_midpoint_pilot import evaluate_action

def job(args):
    root,i,row,actions=args
    path=root/'simulator_evaluation'/f'{i:03d}.json'
    if path.exists():return read(path)
    task=next(t for t in read(row['task_source']) if t['task_id']==row['task_id'])
    midpoint=read(row['midpoint_path'])['state'];arms={}
    for name in ['frozen','trained']:
        action=actions[name]['action']
        if action is None:arms[name]=dict(action=None,outcome=finalize(task,midpoint),model_failure=True)
        else:arms[name]=evaluate_action(task,midpoint,action,row['seed'])
    source=read(row['source_result'])
    for name in ['fixed','half','double']:arms[name]=source['arms'][name]
    result=dict(id=row['id'],family_id=row['family_id'],arms=arms)
    write(path,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.out
    val=read(r/'validation.json');actions=read(r/'evaluation_actions.json')
    for phase in ['frozen','trained']:assert [x['id'] for x in actions[phase]]==[x['id'] for x in val]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(job,(r,i,row,{k:actions[k][i] for k in actions})) for i,row in enumerate(val)]
        results=[]
        for f in as_completed(futures):
            results.append(f.result());write(r/'status.json',dict(phase='simulator_evaluation',completed=len(results),expected=len(val)))
    names=['fixed','half','double','frozen','trained']
    counts={n:sum(x['arms'][n]['outcome']['success'] for x in results) for n in names}
    differences={}
    for other in ['frozen','fixed','half','double']:
        differences[other]=dict(trained_only=sum(x['arms']['trained']['outcome']['success'] and not x['arms'][other]['outcome']['success'] for x in results),other_only=sum(x['arms'][other]['outcome']['success'] and not x['arms']['trained']['outcome']['success'] for x in results))
    summary=dict(complete=True,validation_states=len(val),validation_settings=len({x['family_id'] for x in results}),successes=counts,paired_comparisons=differences,model_failures={n:sum(x['arms'][n]['action'] is None for x in results) for n in ['frozen','trained']},adapter_update=read(r/'adapter_update.json'))
    write(r/'summary.json',summary)
    lines=['# First Qwen CMA fine-tuning pilot','','Three epochs of rank-8 QLoRA action supervision on 76 development states.','Validation comprises 60 states from eight task settings excluded from training.','These settings were previously used for development; this is not a new untouched test.','','| Controller | Verified repairs |','|---|---:|']
    lines += [f'| {n} | {counts[n]}/{len(val)} |' for n in names]
    lines += ['',f'Paired comparisons: {differences}','','Frozen/trained use the same NF4 checkpoint, prompts, greedy decoding, and disabled thinking. The prior GGUF Q8 results are not the matched frozen baseline here. Labels are single-seed simulator winners and may reflect search luck. All final repairs use dense audits and doubled branch capacity. Each action receives 16 candidates from the same saved 16-candidate midpoint. No direct geometry edits.','', 'The final third-epoch adapter was predeclared, not selected by validation outcomes. Empty rationales avoid supervision on unverified optical explanations. This experiment does not isolate optical understanding or establish an AZR curriculum gain.']
    (r/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(r/'status.json',dict(phase='complete',summary=summary));print(summary)
if __name__=='__main__':main()
