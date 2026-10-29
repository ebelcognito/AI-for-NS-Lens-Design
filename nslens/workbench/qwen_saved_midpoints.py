"""Run frozen Qwen on a preselected set of saved failing CMA midpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from nslens.workbench.interactive_cma_pilot import Qwen, MODEL, digest, read, write
from nslens.workbench.cma_midpoint_pilot import run_case


def report(root, expected):
    rows=[read(p) for p in sorted((root/'episodes').glob('*/result.json'))]
    names=['fixed','half','double','qwen']
    totals={n:sum(r['arms'][n]['outcome']['success'] for r in rows) for n in names}
    pairs=[]
    comparison={}
    for n in names[:-1]:
        comparison[n]=dict(qwen_only=sum(r['arms']['qwen']['outcome']['success'] and not r['arms'][n]['outcome']['success'] for r in rows),control_only=sum(r['arms'][n]['outcome']['success'] and not r['arms']['qwen']['outcome']['success'] for r in rows))
    for r in rows:
        q=r['arms']['qwen']
        if q['action'] is None:continue
        for n in names[:-1]:
            b=r['arms'][n]
            if abs(q['outcome']['reward']-b['outcome']['reward'])<.05:continue
            a,z=(q,b) if q['outcome']['reward']>b['outcome']['reward'] else (b,q)
            pairs.append(dict(key=r['key'],family_id=r['family_id'],midpoint_hash=r['midpoint_hash'],observation_path=f'episodes/{r["key"]}/observation.json',chosen=a['action'],rejected=z['action'],chosen_reward=a['outcome']['reward'],rejected_reward=z['outcome']['reward']))
    summary=dict(complete=len(rows)==expected,completed=len(rows),expected=expected,verified_successes=totals,comparisons=comparison,preference_pairs=len(pairs),model_failures=sum(r['arms']['qwen']['action'] is None for r in rows),qwen_operations={op:sum(r['arms']['qwen']['action'] is not None and r['arms']['qwen']['action']['operation']==op for r in rows) for op in ['continue','restart']})
    write(root/'summary.json',summary);write(root/'decision_preferences.json',pairs)
    lines=['# Frozen Qwen on saved failing midpoints','',f'Completed: {len(rows)}/{expected}.','', '| Controller | Verified repairs |','|---|---:|']
    lines += [f'| {n} | {totals[n]}/{len(rows)} |' for n in names]
    lines += ['',f'Paired comparisons: {comparison}', '', 'One recorded numerical seed per task. All arms start from the exact saved failing midpoint and receive 16 additional candidates. Final 5x25/7x51 audits and doubled-capacity agreement determine success.','', 'This is a development subset selected for failure after the fixed first batch. It is not an unbiased full-dataset or held-out evaluation. Related tasks share witnesses/settings. No training, direct edits, or best-of selection. Model failures retain the midpoint without a fallback.','', 'The initial batch and task generation were previously computed and reused. Arm files preserve candidate/ray costs and all audit results.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=a.out;cases=read(root/'cases.json');tasks={t['task_id']:t for t in read(root/'tasks.json')}
    qwen=Qwen('http://127.0.0.1:8080/v1',root)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[]
        for case in cases:
            task=tasks[case['task_id']];key=f'{task["task_id"]}_seed_{case["seed"]}'
            saved=read(root/'episodes'/key/'midpoint.json')
            assert digest(saved['state'])==case['midpoint_hash']
            assert saved['state']['candidates']==16 and not saved['state']['result']['success']
            futures.append(pool.submit(run_case,(root,task,case['seed'],qwen)))
        for future in as_completed(futures):
            future.result();s=report(root,len(cases));print(s,flush=True)


if __name__=='__main__':main()
