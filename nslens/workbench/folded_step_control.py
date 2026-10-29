"""Post-hoc step-size sensitivity control; never a held-out model comparison."""
import argparse,json
from pathlib import Path
from .folded_tools import FoldedWorkbench
from .folded_baseline import replay


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('tasks',type=Path);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for t in json.loads(a.tasks.read_text()):
  wb=FoldedWorkbench(t)
  faces=[f for f in ['A','B','T','D'] if f+'_p2' in t['constraints']['editable']]
  wb.call('enable',dict(faces=faces,powers=[2,4]));wb.call('optimize',dict(max_evaluations=12,step_mm=.5))
  wb.call('restore_best',{});wb.call('finish',dict(summary='Post-hoc fixed 0.5 mm control'))
  r=dict(task=t,events=wb.events,evaluations=wb.records,outcome=wb.finalize())
  r['replay_passed']=replay(r)
  (a.out/(t['task_id']+'.json')).write_text(json.dumps(r,indent=2)+'\n')
  rows.append(dict(task_id=t['task_id'],success=r['outcome']['success']))
  (a.out/'summary.json').write_text(json.dumps(dict(note='Step selected after observing first Qwen episode; exploratory sensitivity control.',rows=rows),indent=2)+'\n')
  print(t['task_id'],r['outcome']['success'],flush=True)

if __name__=='__main__':main()
