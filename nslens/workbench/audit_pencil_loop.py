"""Replay every numerical counterfactual and audit the pencil curriculum boundaries."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .pencil_loop import replay,write


def check(job):
    task,path=job;episode=json.loads(Path(path).read_text())
    return dict(path=str(path),passed=replay(task,episode))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    tasks=json.loads((a.corpus/'tasks.json').read_text())
    jobs=[(t,a.corpus/'private_plans'/f'{t["task_id"]}_plan{i}.json') for t in tasks for i in range(8)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=[]
        for row in pool.map(check,jobs):
            rows.append(row)
            if len(rows)%16==0:print('replayed',len(rows),'/',len(jobs),flush=True)
    write(a.corpus/'replay_audit.json',dict(episodes=len(rows),all_passed=all(r['passed'] for r in rows),rows=rows))

if __name__=='__main__':main()
