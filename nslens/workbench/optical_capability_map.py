"""Failure-stratified, frozen-Qwen direct-edit versus CMA capability pilot.

A development capability map, not a proof of model limits or independent families.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,hashlib,time,argparse,urllib.request,traceback
from pathlib import Path
from dataclasses import replace
from functools import partial
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import numpy as np
from nslens.discovery.folded import RoutingBrief,evaluate
from nslens.discovery.polygon import PolygonCodec,exact_extent
from nslens.freeform_body.trace import trace_bundle
from nslens.search.cma_me import CMAES
from nslens.workbench.interactive_cma_pilot import parameter_names

ROOT=Path(__file__).resolve().parents[2]/'runs/optical_capability_map_20260916'
SOURCE=Path('/playpen/hisham26/overnight_repair_tasks_20260915')
MODEL='qwen3.8-27b'
BUDGET=16
SEEDS=[160901,160902]
LOSS_NAMES=['accepted','detector_miss','insufficient_TIR','TIR_margin','nonexit']


def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def measure(task,genes,fidelity='search',cap=200000,diagnostics=False):
 fields,rays,samples={'screen':(3,25,256),'search':(9,25,256),'audit':(17,101,1024),'stress':(33,151,1024)}[fidelity]
 codec=PolygonCodec(**task['config']['codec']);base=RoutingBrief(**task['config']['brief'])
 brief=replace(base,fields_deg=tuple(np.linspace(min(base.fields_deg),max(base.fields_deg),fields)),rays_per_field=rays,min_folded_throughput=max(.8,base.min_folded_throughput),min_root_fraction=max(.8,base.min_root_fraction))
 with patch('nslens.discovery.folded.trace_bundle',partial(trace_bundle,max_branches=cap)):
  output=evaluate(codec,genes,brief,geometry_samples=samples,return_trace=diagnostics)
 r=output[0] if isinstance(output,tuple) else output
 if r['status']!='measured':return r
 if task.get('spot_target_mm') is not None:
  r['violations']['spot_rms']=max(0.,r['worst_spot_rms']/task['spot_target_mm']-1)
  centers=np.array([f['detector_centroid'] for f in r['fields']]);delta=np.diff(centers)
  r['violations']['mapping']=0. if np.all(delta>0) or np.all(delta<0) else 1.
 if task.get('max_dimensions_mm') is not None:
  lo,hi=np.asarray(r['bounds']);r['violations']['body_growth']=max(0.,float(np.max((hi-lo)/task['max_dimensions_mm'])-1)-1e-9)
 r['feasible']=all(v==0 for v in r['violations'].values())
 if diagnostics:
  _,body,tr=output;axis,pos,center,half,sign=brief.detector();cats=np.zeros((fields,5));face_rows=[dict(face=i,min_tir_margin_deg=None,hit_fraction_min=None,hit_fraction_max=None,events={}) for i in range(codec.corners)]
  for ray in tr.exits:
   field=ray.source_id//rays;tir=[ev for ev in ray.events if ev.kind=='tir'];d=np.asarray(ray.direction)
   distance=(pos-ray.origin[axis])/d[axis] if abs(d[axis])>1e-12 else -1
   if len(tir)<brief.min_tir:cat=2
   elif any(ev.critical_margin_deg<brief.min_tir_margin_deg for ev in tir):cat=3
   elif d[axis]*sign<=1e-9 or distance<=0 or abs(ray.origin[1-axis]+distance*d[1-axis]-center)>half:cat=1
   else:cat=0
   cats[field,cat]+=ray.energy/rays
   # Significant exiting branches only; repeated prefix events across Fresnel
   # branches are not counted as independent interactions. Ranges/minima are used.
   if ray.energy>=.01:
    for ev in ray.events:
     i,f=body._locate(ev.u);row=face_rows[i]
     row['hit_fraction_min']=float(f) if row['hit_fraction_min'] is None else min(row['hit_fraction_min'],float(f))
     row['hit_fraction_max']=float(f) if row['hit_fraction_max'] is None else max(row['hit_fraction_max'],float(f))
     row['events'][ev.kind]=True
     if ev.kind=='tir':row['min_tir_margin_deg']=ev.critical_margin_deg if row['min_tir_margin_deg'] is None else min(row['min_tir_margin_deg'],ev.critical_margin_deg)
  cats[:,4]=1-cats[:,:4].sum(axis=1)
  worst=int(np.argmin([f['folded_throughput'] for f in r['fields']]))
  r['diagnostics']=dict(loss_categories=LOSS_NAMES,per_field_exit_energy=cats.tolist(),worst_throughput_field_index=worst,face_usage=face_rows,critical_angle_deg=float(np.degrees(np.arcsin(1/body.n))),note='Exit-energy categories are disjoint with priority fold count, TIR margin, receiver miss. Face ranges concern exiting branches with >=1% incident ray energy; no causal surface attribution.')
 return r


def rank(r):
 if r['status']!='measured':return (0,)
 numerical=[r['violations'].get(k,0) for k in ['closure','unresolved','pruned']]
 if max(numerical)>0:return (1,-max(numerical),-sum(numerical))
 v=list(r['violations'].values())
 return (3 if r['feasible'] else 2,-max(v),-sum(v),r['worst_folded_throughput'],r['support'],-r['worst_spot_rms'])


def agreement(a,b):
 if a['status']!=b['status']:return False
 if a['status']!='measured':return a.get('reason')==b.get('reason')
 return all(np.isclose(a[k],b[k],rtol=1e-8,atol=1e-9) for k in ['worst_folded_throughput','worst_spot_rms','support','closure','unresolved','pruned']) and all(np.allclose([f[k] for f in a['fields']],[f[k] for f in b['fields']],rtol=1e-8,atol=1e-9) for k in ['folded_throughput','spot_rms','root_fraction','detector_centroid'])


def audit(task,x):
 a=measure(task,x,'audit');b=measure(task,x,'stress',diagnostics=True);c=measure(task,x,'stress',400000)
 return dict(audit=a,stress=b,capacity_agrees=agreement(b,c),success=bool(a['feasible'] and b['feasible'] and agreement(b,c)),source_rays=sum(r['source_rays'] for r in [a,b,c]),evaluations=3)


def screen(task):
 return dict(task=task,screen=measure(task,task['start'],'screen',diagnostics=True))


def prepare(workers):
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'tasks.json').exists():return read(ROOT/'tasks.json')
 candidates=[]
 for t in read(SOURCE/'tasks.json'):
  candidates.append(dict(source_task_id=t['task_id'],ancestry=t['family_id'],config=t['config'],start=t['start'],spot_target_mm=None,max_dimensions_mm=None,source='overnight_repair',known_feasible_witness=True))
 if (ROOT/'screened.json').exists():screened=read(ROOT/'screened.json')
 else:
  with ProcessPoolExecutor(max_workers=workers) as pool:screened=list(pool.map(screen,candidates))
  dump(ROOT/'screened.json',screened)
 counts={};groups={}
 for row in screened:
  r=row['screen']
  if r['status']!='measured':continue
  d=r['diagnostics'];i=d['worst_throughput_field_index'];loss=np.asarray(d['per_field_exit_energy'][i]);kind=LOSS_NAMES[int(np.argmax(loss[1:4]))+1]
  row['primary_failure']=kind;counts[kind]=counts.get(kind,0)+1;groups.setdefault(kind,[]).append(row)
 selected=[];used=set()
 for kind in ['detector_miss','insufficient_TIR','TIR_margin']:
  # No policy outcomes used in selection. Use one strong failure anchor plus throughput quantiles
  # and distinct parent settings, so difficulty is not uniformly severe. If scarce, keep fewer rather than relabel.
  rows=sorted(groups.get(kind,[]),key=lambda z:(-z['screen']['diagnostics']['per_field_exit_energy'][z['screen']['diagnostics']['worst_throughput_field_index']][LOSS_NAMES.index(kind)],z['task']['source_task_id']))
  chosen=[];families=set()
  if rows:
   chosen.append(rows[0]);families.add(rows[0]['task']['ancestry'])
   levels=np.quantile([r['screen']['worst_folded_throughput'] for r in rows],[.95,.75,.55,.35,.15])
   for target in levels:
    eligible=[r for r in rows if r['task']['ancestry'] not in families]
    if not eligible:break
    row=min(eligible,key=lambda r:(abs(r['screen']['worst_folded_throughput']-target),r['task']['source_task_id']))
    chosen.append(row);families.add(row['task']['ancestry'])
  for row in chosen:
   t=copy.deepcopy(row['task']);t['failure_group']=kind;t['screen']=row['screen'];selected.append(t);used.add(t['source_task_id'])
 # Six imaging examples, including known C sibling designs and distinct A/B.
 p=Path('/playpen/hisham26/concentration_pilot_20260915')
 specs=[('A_seed15100','initial',None),('B_seed15100','initial',None),('C_seed15100','initial',None),('C_seed15300','final',None),('C_seed15100','final',6.),('C_seed15200','final',6.)]
 for name,stage,full in specs:
  t=read(p/name/'task.json');r=read(p/name/'result.json');config=copy.deepcopy(t['design']['config']);config={k:config[k] for k in ['codec','brief']}
  genes=t['design']['genes'] if stage=='initial' else r['best']['genes']
  if full is not None:config['brief']['fields_deg']=[-full/2,0.,full/2]
  task=dict(source_task_id=f'{name}_{stage}_{full}',ancestry=t['design']['family_id'],config=config,start=genes,spot_target_mm=.5,max_dimensions_mm=t['initial_dimensions'],source='imaging_refinement',known_feasible_witness=name.startswith('C') and full is None,failure_group='blur_or_mixed_route')
  task['screen']=measure(task,genes,'screen',diagnostics=True);selected.append(task)
 for i,t in enumerate(selected):t['id']=f'case_{i:02d}'
 dump(ROOT/'tasks.json',selected)
 dump(ROOT/'selection.json',dict(available_failure_groups=counts,selected_failure_groups={k:sum(t['failure_group']==k for t in selected) for k in ['detector_miss','insufficient_TIR','TIR_margin','blur_or_mixed_route']},ancestry_count=len(set(t['ancestry'] for t in selected)),selection='Measured dominant exit loss at worst-throughput field; one hard anchor plus five throughput quantiles per routing group with distinct ancestry; six predeclared imaging examples. No model/control outcome filtering.',development_only=True))
 return selected


def validate_edits(obj,n):
 if not isinstance(obj,dict) or set(obj)!={'edits','rationale'} or not isinstance(obj['rationale'],str) or len(obj['rationale'])>1500:raise ValueError('Expected exactly edits and a short rationale')
 if not isinstance(obj['edits'],list) or len(obj['edits'])!=BUDGET:raise ValueError('Exactly 16 candidate edits required')
 for edit in obj['edits']:
  if not isinstance(edit,dict) or set(edit)!={'parameters','deltas'}:raise ValueError('Each edit requires parameters and deltas only')
  ids,ds=edit['parameters'],edit['deltas']
  if not isinstance(ids,list) or not 1<=len(ids)<=3 or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<n for i in ids):raise ValueError('Edit 1–3 unique valid parameter IDs')
  if not isinstance(ds,list) or len(ds)!=len(ids) or any(type(d) not in (int,float) or not np.isfinite(d) or not 0<abs(d)<=.1 for d in ds):raise ValueError('One nonzero normalized delta per parameter, absolute maximum 0.1')
 return obj


def validate_cma(obj,n):
 if not isinstance(obj,dict) or set(obj)!={'parameters','sigma','rationale'}:raise ValueError('Expected parameters, sigma, rationale')
 ids=obj['parameters'];sigma=obj['sigma']
 if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<n for i in ids):raise ValueError('Unique valid parameter IDs required')
 if type(sigma) not in (int,float) or not np.isfinite(sigma) or not .002<=sigma<=.1:raise ValueError('sigma must be 0.002 to 0.1')
 if not isinstance(obj['rationale'],str) or len(obj['rationale'])>1500:raise ValueError('Short rationale required')
 return obj


def observation(task):
 body=PolygonCodec(**task['config']['codec']).decode(task['start']);r=task['initial_search']
 return dict(immutable_config=task['config'],requirements=dict(min_folded_throughput=max(.8,task['config']['brief']['min_folded_throughput']),min_root_support=max(.8,task['config']['brief']['min_root_fraction']),spot_rms_target_mm=task['spot_target_mm'],max_dimensions_mm=task['max_dimensions_mm'],monotonic_centroids_required=task['spot_target_mm'] is not None),geometry=dict(corners_mm=body.corners.tolist(),face_sags_mm=body.sags.tolist()),parameters=[dict(id=i,name=name,value=x) for i,(name,x) in enumerate(zip(parameter_names(len(body.corners)),task['start']))],measurement={k:r[k] for k in ['status','feasible','fields','violations','worst_folded_throughput','worst_spot_rms','support','diagnostics']},candidate_budget=BUDGET)

SYSTEM='''You are controlling a 2D non-sequential optical body. Use the supplied numerical geometry and ray measurements to choose bounded geometry changes. Refractive index, source, beam, fields and detector are immutable. Material TIR conditions and all ray routes are recomputed after every edit. Face i connects physical corner i to corner (i+1)%K; physical face IDs are zero-based. Goals apply at EVERY field, not on average. Do not sharpen spots by throwing away light below the throughput/support floor. Shape genes can move several physical vertices; use the supplied codec definitions and physical geometry. Return only the requested JSON. Your rationale is a hypothesis, not verified physics. No hidden witness or other candidate outcomes are available.'''


def model_action(task,arm,seed,endpoint):
 obs=observation(task);folder=ROOT/'model_calls';path=folder/f"{task['id']}_{arm}_{seed}.json";obs_hash=digest(obs)
 if path.exists():
  saved=read(path)
  if saved['observation_hash']!=obs_hash:raise RuntimeError('cached observation changed')
  return saved
 if arm=='qwen_edit':
  instruction='Propose exactly 16 candidate geometric edits, all relative to the SAME starting geometry, not a sequential path. Each candidate changes 1–3 normalized gene values by deltas of absolute value <=0.1. Give distinct alternatives where useful. Values are clipped to [-1,1]. Schema: {"edits":[{"parameters":[integer IDs],"deltas":[nonzero numbers]} repeated 16 times],"rationale":"short testable explanation"}.'
  validator=validate_edits
 else:
  instruction='Choose one CMA-ES batch configuration. It will evaluate exactly 16 candidates in two populations of eight, starting at the current geometry. Select any nonempty subset of gene IDs and one initial normalized sigma in [0.002,0.1]. Each active gene has a trust region +/-0.1 around its current value intersected with [-1,1]. Schema: {"parameters":[integer IDs],"sigma":number,"rationale":"short testable explanation"}.'
  validator=validate_cma
 request=dict(model=MODEL,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(dict(instruction=instruction,observation=obs)))],temperature=.7,top_p=.9,seed=seed,max_tokens=4096,reasoning_budget_tokens=512,response_format=dict(type='json_object'))
 attempts=[];action=None
 for attempt in range(3):
  began=time.time();record=dict(request=copy.deepcopy(request))
  try:
   req=urllib.request.Request(endpoint+'/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=600) as response:raw=json.load(response)
   record['response']=raw
   if raw['choices'][0]['finish_reason']=='length':raise ValueError('truncated response')
   action=validator(json.loads(raw['choices'][0]['message']['content']),len(task['start']))
  except Exception as exc:
   record['error']=str(exc);action=None
   request['messages'][-1]['content']+='\nPrevious response rejected: '+str(exc)
  record['elapsed_seconds']=time.time()-began;attempts.append(record)
  dump(folder/f"{task['id']}_{arm}_{seed}_attempts.json",attempts)
  if action is not None:break
 saved=dict(observation_hash=obs_hash,action=action,attempts=attempts,model_failure=action is None);dump(path,saved);return saved


def execute(task,arm,seed,action=None):
 folder=ROOT/'episodes'/task['id'];path=folder/f'{arm}_{seed}.json'
 if path.exists():return read(path)
 started=time.time();start=np.asarray(task['start']);best=start.copy();best_r=task['initial_search'];records=[];rng=np.random.default_rng(seed)
 if arm in ['qwen_cma','fixed_cma'] and action is not None:
  ix=np.asarray(action['parameters']);lo=np.maximum(-1,start[ix]-.1);hi=np.minimum(1,start[ix]+.1)
  cma=CMAES(start[ix],action['sigma'],popsize=8,seed=seed,bounds=(lo,hi),boundary_handling='reflect')
  for generation in range(2):
   pop=cma.ask();batch=[]
   for candidate in pop:
    x=start.copy();x[ix]=candidate;r=measure(task,x);records.append(dict(genes=x.tolist(),result=r));batch.append(r)
    if rank(r)>rank(best_r):best=x.copy();best_r=r
   order=sorted(range(8),key=lambda i:rank(batch[i]),reverse=True);cma.tell(np.asarray(order))
 elif arm in ['qwen_edit','random_edit'] and action is not None:
  for edit in action['edits']:
   x=start.copy();ix=np.asarray(edit['parameters']);x[ix]=np.clip(x[ix]+edit['deltas'],-1,1);r=measure(task,x);records.append(dict(edit=edit,genes=x.tolist(),result=r))
   if rank(r)>rank(best_r):best=x.copy();best_r=r
 # Model exhaustion retains the initial geometry; no fallback action.
 final=audit(task,best)
 result=dict(case_id=task['id'],arm=arm,seed=seed,action=action,genes=best.tolist(),best_search=best_r,outcome=final,records=records,candidate_evaluations=len(records),candidate_rays=sum(r['result']['source_rays'] for r in records),elapsed_seconds=time.time()-started,model_failure=arm.startswith('qwen') and action is None)
 dump(path,result);return result


def init_task(task):
 folder=ROOT/'cases'/task['id'];path=folder/'initial.json'
 task_hash=digest({k:task[k] for k in ['config','start','spot_target_mm','max_dimensions_mm']})
 if path.exists():
  data=read(path)
  if data.get('task_hash')!=task_hash:raise RuntimeError('cached initial task changed')
  if data['initial_audits']['stress']['status']=='measured' and 'diagnostics' not in data['initial_audits']['stress']:
   data['initial_audits']['stress']=measure(task,task['start'],'stress',diagnostics=True)
   data['extra_diagnostic_evaluations']=1
   dump(path,data)
 else:
  data=dict(task_hash=task_hash,initial_search=measure(task,task['start'],'search',diagnostics=True),initial_audits=audit(task,task['start']));dump(path,data)
 return {**task,**data}


def random_action(task,seed):
 rng=np.random.default_rng(seed);edits=[]
 for i in range(BUDGET):
  count=int(rng.integers(1,4));ids=rng.choice(len(task['start']),count,replace=False).tolist();deltas=(rng.choice([-1.,1.],count)*rng.choice([.01,.025,.05,.1],count)).tolist();edits.append(dict(parameters=ids,deltas=deltas))
 return dict(edits=edits,rationale='Fixed seeded random edits, same 1–3-gene and +/-0.1 bounds as Qwen.')


def summarize(tasks):
 rows=[];groups={};arms=['qwen_cma','qwen_edit','fixed_cma','random_edit']
 for t in tasks:
  outcomes={}
  for arm in arms:
   files=[ROOT/'episodes'/t['id']/f'{arm}_{seed}.json' for seed in SEEDS]
   if all(p.exists() for p in files):outcomes[arm]=[read(p) for p in files]
  if len(outcomes)!=len(arms):continue
  passes={arm:[r['outcome']['success'] for r in rs] for arm,rs in outcomes.items()}
  initial=t['initial_audits']['success'];q=any(passes['qwen_cma']+passes['qwen_edit']);control=any(passes['fixed_cma']+passes['random_edit'])
  category='initial_already_passed_dense' if initial else ('qwen_solved_in_test' if q else ('control_solved_qwen_missed' if control else 'no_tested_method_solved'))
  improved={}
  for arm,rs in outcomes.items():
   # A material gain must appear on BOTH audit grids without worsening the
   # maximum normalized constraint violation. This is not a full repair.
   marks=[]
   for result in rs:
    checks=[]
    for fidelity in ['audit','stress']:
     before=t['initial_audits'][fidelity];after=result['outcome'][fidelity]
     if before['status']!='measured' or after['status']!='measured':checks.append(False);continue
     v0=max(before['violations'].values());v1=max(after['violations'].values())
     if t['spot_target_mm'] is None:gain=after['worst_folded_throughput']-before['worst_folded_throughput']>=.02
     else:gain=after['worst_spot_rms']<=.9*before['worst_spot_rms'] and after['worst_folded_throughput']>=max(.8,before['worst_folded_throughput']-.01)
     checks.append(bool(gain and v1<=v0+1e-9))
    marks.append(bool(all(checks) and result['outcome']['capacity_agrees']))
   improved[arm]=marks
  row=dict(case_id=t['id'],source_task_id=t['source_task_id'],failure_group=t['failure_group'],ancestry=t['ancestry'],category=category,initial_dense_pass=initial,passes=passes,material_improvement=improved,known_feasible_witness=t['known_feasible_witness'],model_failures={a:sum(r['model_failure'] for r in rs) for a,rs in outcomes.items()})
  rows.append(row)
 for group in sorted(set(t['failure_group'] for t in tasks)):
  subset=[r for r in rows if r['failure_group']==group and not r['initial_dense_pass']]
  groups[group]=dict(cases=len(subset),ancestries=len(set(r['ancestry'] for r in subset)),solved_any_of_two={a:sum(any(r['passes'][a]) for r in subset) for a in arms},solved_both={a:sum(all(r['passes'][a]) for r in subset) for a in arms},material_improvement_any={a:sum(any(r['material_improvement'][a]) for r in subset) for a in arms})
 summary=dict(complete=len(rows)==len(tasks),completed=len(rows),expected=len(tasks),by_failure=groups,cases=rows,limitations='Development-only, correlated ancestry, two rollouts, 16 candidate calls per arm/rollout. No finite failure proves inability. Candidate and audit rays/costs recorded separately. Direct edits and CMA have matched candidate counts but different proposal/update structure and model token cost.')
 dump(ROOT/'summary.json',summary);return summary


def run(workers,endpoint):
 tasks=prepare(workers)
 dump(ROOT/'status.json',dict(phase='initial_audits',tasks=len(tasks)))
 with ProcessPoolExecutor(max_workers=workers) as pool:tasks=list(pool.map(init_task,tasks))
 dump(ROOT/'tasks_with_initial.json',tasks)
 with urllib.request.urlopen(endpoint+'/models',timeout=10) as response:models=json.load(response)
 if MODEL not in [m['id'] for m in models['data']]:raise RuntimeError('Expected Qwen checkpoint alias missing')
 dump(ROOT/'served_model.json',models)
 dump(ROOT/'status.json',dict(phase='running',tasks=len(tasks),expected_arms=len(tasks)*4*len(SEEDS)))
 with ProcessPoolExecutor(max_workers=workers) as numerical,ThreadPoolExecutor(max_workers=4) as llm:
  model_jobs={};solver_jobs=[]
  for t in tasks:
   dump(ROOT/'observations'/f"{t['id']}.json",observation(t))
   for seed in SEEDS:
    fixed=dict(parameters=list(range(len(t['start']))),sigma=.025,rationale='Fixed all-parameter CMA, sigma .025.')
    solver_jobs.append(numerical.submit(execute,t,'fixed_cma',seed,fixed));solver_jobs.append(numerical.submit(execute,t,'random_edit',seed,random_action(t,seed)))
    for arm in ['qwen_cma','qwen_edit']:model_jobs[llm.submit(model_action,t,arm,seed,endpoint)]=(t,arm,seed)
  for future in as_completed(model_jobs):
   t,arm,seed=model_jobs[future];decision=future.result();solver_jobs.append(numerical.submit(execute,t,arm,seed,decision['action']))
   print('model',t['id'],arm,seed,'valid',not decision['model_failure'],flush=True)
  done=0
  for future in as_completed(solver_jobs):
   r=future.result();done+=1;summary=summarize(tasks)
   dump(ROOT/'status.json',dict(phase='running',arms_completed=done,expected_arms=len(solver_jobs),complete_cases=summary['completed'],tasks=len(tasks)))
   print('result',r['case_id'],r['arm'],r['seed'],r['outcome']['success'],flush=True)
 summarize(tasks);dump(ROOT/'status.json',dict(phase='complete',tasks=len(tasks)))


def main():
 global ROOT
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--out',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=6);p.add_argument('--endpoint',default='http://127.0.0.1:8081/v1');a=p.parse_args();ROOT=a.out.resolve();ROOT.mkdir(exist_ok=True,parents=True)
 try:
  if a.command=='prepare':prepare(a.workers)
  else:run(a.workers,a.endpoint)
 except Exception:
  dump(ROOT/'status.json',dict(phase='failed',error=traceback.format_exc()));raise
if __name__=='__main__':main()
