"""Replay CMA populations without new traces; reconcile all pilot ledgers."""
import argparse
from pathlib import Path
import numpy as np
from nslens.workbench.stock_probe import read,dump,indices,rank,digest
from nslens.search.cma_me import CMAES


def audit(root):
 expected=read(root.parent/"source_hashes.json")
 amendment=read(root/"transport_amendment.json") if (root/"transport_amendment.json").exists() else None
 for name,sha in expected.items():
  if amendment and name.endswith("nslens/workbench/stock_probe.py"):sha=amendment["new_source_sha256"]
  assert digest(name)==sha, name
 reports=[]
 for folder in sorted(p.parent for p in root.glob('*/task.json')):
  task,state,out=read(folder/'task.json'),read(folder/'state.json'),read(folder/'outcome.json')
  assert len(state['events'])==6 and state['done']
  best={r['probe']:r['result'] for r in state['records'] if r['stage']=='initial'}
  search=[]
  for number,event in enumerate(state['events']):
   batch=read(folder/f'batch_{number:02d}.json');a=event['action'];probe=a['probe']
   assert batch['event']==event
   base=np.asarray(task['design']['genes'] if a['parent']=='start' else best[probe]['genes'])
   ix=indices(task['design']['corners'],a['group'])
   cma=CMAES(base[ix],a['sigma'],popsize=10,seed=event['seed'],bounds=(-np.ones(len(ix)),np.ones(len(ix))),boundary_handling='reflect')
   for g in range(4):
    assert cma.state_snapshot()==batch['generations'][g]['before']
    pop=cma.ask();rows=[r['result'] for r in batch['records'][g*10:(g+1)*10]]
    for x,row in zip(pop,rows):
     genes=base.copy();genes[ix]=x
     assert genes.tolist()==row['genes']
     if rank(row)>rank(best[probe]):best[probe]=row
    order=sorted(range(10),key=lambda i:rank(rows[i]),reverse=True)
    assert order==batch['generations'][g]['order']
    cma.tell(np.asarray(order));assert cma.state_snapshot()==batch['generations'][g]['after']
   assert event['source_rays']==sum(r['result']['source_rays'] for r in batch['records'])
   search.extend(batch['records'])
  assert search==[r for r in state['records'] if r['stage']=='search']
  assert best==state['best']
  budget={}
  for r in state['records']:
   b=budget.setdefault(r['stage'],dict(evaluations=0,source_rays=0));b['evaluations']+=1;b['source_rays']+=r['result']['source_rays']
  assert budget==out['budget'] and budget['search']['evaluations']==240 and budget['search']['source_rays']<=54000
  calls=[__import__('json').loads(l) for l in (folder/'model_calls.jsonl').read_text().splitlines()] if (folder/'model_calls.jsonl').exists() else []
  if root.name=='qwen': assert [c['event'] for c in calls if 'event' in c]==state['events']
  reports.append(dict(task_id=task['task_id'],cma_replay_exact=True,budgets_match=True,model_responses=len(calls),invalid_responses=sum('error'in c for c in calls),
   model_usage={key:sum(c['response'].get('usage',{}).get(key,0) for c in calls) for key in ['prompt_tokens','completion_tokens','total_tokens']},
   audit_replay_exact=all(v['replay_exact'] for v in out['outcomes'].values()),capacity_agrees=all(v['capacity_agrees'] for v in out['outcomes'].values())))
 result=dict(complete=len(reports)==6,source_hashes_match=True,results=reports)
 dump(root/'audit.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(audit(a.root))
