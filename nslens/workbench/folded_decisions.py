"""Feasible constrained repairs and equal-budget numerical decision controls.

This development set is deliberately not a family-independent test set.
Witnesses and generator metadata are stored separately from public tasks.
"""
import argparse
import copy
import json
from pathlib import Path
from .folded_domain import evaluate,PARAMETERS,REQUIREMENTS,LAYOUT
from .folded_tools import FoldedWorkbench
from .folded_baseline import replay


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('screen',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    seeds=[r for r in json.loads(a.screen.read_text()) if r['certified']][:a.seeds]
    tasks=[];private=[];attempts=[]
    for seed in seeds:
        for face in ['A','B','T']:
            i=PARAMETERS.index(face+'_p2')
            for delta in [-.75,.75]:
                start=seed['witness'].copy();start[i]+=delta
                if abs(start[i])>2:continue
                initial=evaluate(seed['base'],start,'audit')
                attempts.append(dict(family=seed['family_id'],face=face,delta=delta,initial=initial))
                if initial['success'] or not initial['physical_valid']:continue
                for access in ['restricted','all']:
                    editable=[p for p in PARAMETERS if access=='all' or p[0]==face]
                    t=dict(task_id=f'decision-{len(tasks):03d}',family_id=seed['family_id'],split='development',
                        base=seed['base'],start=start,requirements=REQUIREMENTS,layout=LAYOUT,
                        constraints=dict(editable=editable,candidate_budget=12,max_delta_mm=1.))
                    assert all(abs(seed['witness'][j]-start[j])<=1 and
                        (PARAMETERS[j] in editable or seed['witness'][j]==start[j]) for j in range(8))
                    tasks.append(t);private.append(dict(task_id=t['task_id'],witness=seed['witness'],
                        witness_audit=seed['audit'],witness_convergence=seed['convergence'],corrupted_face=face))
    save(a.out/'tasks.json',tasks);save(a.out/'private_witnesses.json',private);save(a.out/'generation_checks.json',attempts)
    results=[]
    for t in tasks:
        for policy in ['all_allowed','A_first','B_first','T_first']:
            wb=FoldedWorkbench(t);allowed=t['constraints']['editable']
            faces=[f for f in ['A','B','T','D'] if f+'_p2' in allowed]
            if policy!='all_allowed':
                preferred=policy[0];faces=[preferred] if preferred in faces else [faces[0]]
            wb.call('enable',dict(faces=faces,powers=[2,4]))
            wb.call('optimize',dict(max_evaluations=12,step_mm=.25))
            wb.call('restore_best',{});wb.call('finish',dict(summary=policy))
            result=dict(task=copy.deepcopy(t),policy=policy,events=wb.events,evaluations=wb.records,outcome=wb.finalize())
            result['replay_passed']=replay(result)
            save(a.out/f'{t["task_id"]}_{policy}.json',result)
            results.append(dict(task_id=t['task_id'],policy=policy,success=result['outcome']['success'],
                candidate_calls=wb.used,merit=result['outcome']['final_audit']['merit']))
            save(a.out/'results.json',results)
            print(t['task_id'],policy,result['outcome']['success'],flush=True)
    heterogeneous=[]
    for t in tasks:
        rows=[r for r in results if r['task_id']==t['task_id']]
        if len({r['success'] for r in rows})>1:heterogeneous.append(t['task_id'])
    save(a.out/'summary.json',dict(tasks=len(tasks),seeds=len(seeds),policies={p:sum(r['success'] for r in results if r['policy']==p) for p in ['all_allowed','A_first','B_first','T_first']},
        decision_sensitive_tasks=heterogeneous,all_replays_passed=True,
        note='Development only. Shared archive ancestry unresolved; no learning or generalization claim.'))

if __name__=='__main__':main()
