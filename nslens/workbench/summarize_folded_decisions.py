"""Summarize a completed folded decision pilot without treating variants as families."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('numerical',type=Path);p.add_argument('qwen',type=Path)
    a=p.parse_args()
    tasks=json.loads((a.numerical/'tasks.json').read_text())
    results=json.loads((a.numerical/'results.json').read_text())
    rows=[]
    for task in tasks:
        episodes=[json.loads(f.read_text()) for f in a.qwen.glob('episode*.json')]
        q=next(e for e in episodes if e['task']['task_id']==task['task_id'])
        assert q['task']==task and q['replay_passed']
        assert q['outcome']['candidate_calls']<=task['constraints']['candidate_budget']
        numerical={r['policy']:r['success'] for r in results if r['task_id']==task['task_id']}
        assert len(numerical)==4
        rows.append(dict(task_id=task['task_id'],numerical=numerical,qwen=q['outcome']['success'],
            qwen_status=q['status'],qwen_candidates=q['outcome']['candidate_calls'],
            qwen_verifications=q['outcome']['verification_calls'],qwen_rays=q['outcome']['nominal_rays'],
            qwen_tokens=sum(q['usage'].values()),qwen_invalid_requests=q['outcome']['invalid_tool_calls']))
    summary=dict(rows=rows,tasks=len(rows),qwen_successes=sum(r['qwen'] for r in rows),
        numerical_successes={p:sum(r['numerical'][p] for r in rows) for p in rows[0]['numerical']} if rows else {},
        limitations=['Single certified ancestor; development tasks only.',
            'Only candidate budget is matched. Qwen can use optional verification; total rays reported separately.',
            'All retained corruptions affect A; this does not test diagnosis across defect locations.',
            'No training or learned proposer was run.'])
    (a.qwen/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
