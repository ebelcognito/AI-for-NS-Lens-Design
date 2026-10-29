"""Run local Qwen or an explicit numerical control on folded freeform tasks."""
import argparse
import json
from pathlib import Path
import time
from .folded_tools import FoldedWorkbench,TOOLS,INSTRUCTIONS
from .qwen_baseline import LocalQwen,MODEL,REVISION
from .tools import canonical_hash
from nslens.analysis.run_repair_curriculum import digest,write_json


def replay(result):
 wb=FoldedWorkbench(result['task'])
 for e in result['events']:
  assert canonical_hash(wb.state())==e['before_hash']
  assert wb.call(e['tool'],e['arguments'])==e['output']
  assert canonical_hash(wb.state())==e['after_hash']
 assert wb.finalize()==result['outcome']
 assert wb.records==result['evaluations']
 return True


def run(task,transport=None,*,max_tokens=64000):
 wb=FoldedWorkbench(task);responses=[];usage=dict(input_tokens=0,output_tokens=0)
 conversation=[dict(role='user',content=json.dumps(task))];status='action_limit';start=time.monotonic()
 if transport is None:
  for name,args in [('inspect',{}),('enable',dict(faces=['A','B','T','D'],powers=[2,4])),
                    ('optimize',dict(max_evaluations=48,step_mm=.25)),('restore_best',{}),
                    ('verify',{}),('finish',dict(summary='Numerical coordinate-search control.'))]:wb.call(name,args)
  status='submitted'
 else:
  while not wb.done and wb.actions<16 and len(responses)<16:
   payload=dict(instructions=INSTRUCTIONS,input=conversation,tools=TOOLS,parallel_tool_calls=False,max_output_tokens=1500)
   remaining=max_tokens-sum(usage.values())-transport.input_token_count(payload)
   if remaining<=0:status='token_limit';break
   payload['max_output_tokens']=min(remaining,1500)
   try:response=transport(payload)
   except Exception as error:status='transport_error:'+type(error).__name__;break
   responses.append(response)
   for key in usage:usage[key]+=response['usage'][key]
   conversation.extend(response['output'])
   calls=[c for c in response['output'] if c['type']=='function_call']
   if not calls:status='stopped_without_submission';break
   for call in calls:
    result=wb.call(call['name'],json.loads(call['arguments']))
    conversation.append(dict(type='function_call_output',call_id=call['call_id'],output=json.dumps(result,allow_nan=False)))
   if wb.done:status='submitted'
 result=dict(task=task,status=status,provenance='frozen_language_model' if transport else 'scripted_coordinate_control',
  model=MODEL if transport else None,response_log=responses,conversation=conversation,usage=usage,
  elapsed_seconds=time.monotonic()-start,events=wb.events,evaluations=wb.records,outcome=wb.finalize())
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('tasks',type=Path);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--weights',type=Path);p.add_argument('--scripted',action='store_true')
 p.add_argument('--max-tokens',type=int,default=128000)
 a=p.parse_args()
 if a.out.exists() or (not a.scripted and not a.weights):p.error('new output directory and local weights (or --scripted) required')
 tasks=json.loads(a.tasks.read_text());a.out.mkdir(parents=True)
 sources={str(p):digest(p) for p in Path('nslens/workbench').glob('*.py')}
 sources.update({str(p):digest(p) for p in Path('nslens/freeform_body').glob('*.py')})
 write_json(a.out/'config.json',dict(model=MODEL,revision=REVISION,scripted=a.scripted,
  task_sha256=digest(a.tasks),seed=20260910,max_tokens=a.max_tokens,sources=sources,
  note='Folded smoke tasks share one ancestor; no independent-family generalization claim.'))
 transport=None if a.scripted else LocalQwen(a.weights)
 results=[]
 for i,task in enumerate(tasks):
  if transport:transport.torch.manual_seed(20260910+i)
  result=run(task,transport,max_tokens=a.max_tokens)
  write_json(a.out/f'episode{i:03d}.json',result)
  result['replay_passed']=replay(result)
  write_json(a.out/f'episode{i:03d}.json',result);results.append(result)
  write_json(a.out/'summary.json',dict(episodes=len(results),successes=sum(r['outcome']['success'] for r in results),
   complete=len(results)==len(tasks),all_replays_passed=all(r['replay_passed'] for r in results),
   outcomes=[dict(task_id=r['task']['task_id'],status=r['status'],outcome=r['outcome']) for r in results]))
  print(task['task_id'],result['status'],result['outcome']['success'],flush=True)


if __name__=='__main__':main()
