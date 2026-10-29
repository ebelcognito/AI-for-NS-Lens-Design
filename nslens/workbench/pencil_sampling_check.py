"""Fixed additional decoding seeds; evaluation only, never checkpoint selection."""
import argparse,gc,json
from pathlib import Path
from .pencil_loop import write
from .qwen_baseline import LocalQwen
from .train_pencil_loop import assess


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('run',type=Path)
 p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
 if not (a.run/'summary.json').exists():raise ValueError('finish primary experiment first')
 protocol=json.loads((a.run/'sampling_protocol.json').read_text())
 tasks=[t for t in json.loads((a.corpus/'tasks.json').read_text()) if t['split']=='test']
 cache={}
 for f in (a.corpus/'private_plans').glob('*.json'):
  r=json.loads(f.read_text());cache[(r['task_id'],r['plan_id'])]=r
 rows=[]
 for arm in ['baseline','uniform','fixed','adaptive']:
  adapter=None if arm=='baseline' else a.run/arm/'round2'/'adapter'
  transport=LocalQwen(a.weights,adapter)
  for seed in protocol['seeds']:
   decisions=assess(transport,tasks,cache,seed)
   rows.append(dict(arm=arm,seed=seed,decisions=decisions))
   write(a.run/'sampling_check.json',rows)
  del transport;gc.collect()
  import torch
  torch.cuda.empty_cache()
 print('Additional decoding check complete',flush=True)

if __name__=='__main__':main()
