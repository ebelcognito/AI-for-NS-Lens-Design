"""Portable, bounded CMA correction tasks for stock polygon optics.

Model-neutral JSON actions; no model gets Python or shell execution privileges.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from unittest.mock import patch
import numpy as np
from nslens.discovery.folded import RoutingBrief, evaluate
from nslens.discovery.polygon import PolygonCodec, exact_extent
from nslens.freeform_body.trace import trace_bundle
from nslens.search.cma_me import CMAES

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'probe': {'type': 'string', 'enum': ['repair', 'compact', 'field']},
        'group': {'type': 'string', 'enum': ['all', 'shape', 'curvature', 'placement_size']},
        'sigma': {'type': 'number', 'minimum': .005, 'maximum': .15},
        'parent': {'type': 'string', 'enum': ['start', 'best']},
        'reason': {'type': 'string', 'maxLength': 1200},
    }, 'required': ['probe', 'group', 'sigma', 'parent', 'reason'],
}
INSTRUCTIONS = """You control bounded CMA batches for one optical design. Choose one JSON action conforming to action_schema. No code or markdown. Inspect per-field violations and previous batch feedback. You have two batches for EACH of repair, compact, field; choose their order, variable group, sigma, and whether to restart at start or the probe's best. Every batch runs 4 generations of 10 candidates, restarting CMA covariance. 'shape' edits rotation and corner angular/radial genes; 'curvature' edits sags; 'placement_size' edits center and half dimensions; 'all' edits everything. Normalized genes lie in [-1,1]. Use small steps when preserving a fragile route. Bad candidates remain eligible. The compact goal reduces the body's longest bounding-box side by 5%, with neither side allowed to grow; other probes cannot enlarge either side beyond the start. Field probe doubles full FOV to 2 degrees. Repair keeps 1 degree. All source, detector, material, aperture, radius and clearance constraints remain fixed. Search uses 9 fields x 25 pupil rays. Ranking prioritizes joint routing/package feasibility and then lower maximum normalized violation; throughput breaks ties. Audit is hidden until all actions finish. Spot optimization is deferred. Goal: learn recoverability, compaction response and field response, not maximize initial throughput. State facts in reason; avoid long deliberation."""

def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(task, genes, probe, *, audit=False, cap=200000):
    half = task.get('full_fov_deg', {}).get(probe, 2. if probe == 'field' else 1.) / 2
    brief = replace(RoutingBrief(**task['design']['brief']),
                    fields_deg=tuple(np.linspace(-half, half, 9)), rays_per_field=101 if audit else 25)
    codec = PolygonCodec(**task['design']['codec'])
    with patch('nslens.discovery.folded.trace_bundle', partial(trace_bundle, max_branches=cap)):
        r = evaluate(codec, genes, brief, geometry_samples=4096 if audit else 256)
    r['routing_feasible'] = r['feasible']
    try:
        lo, hi = exact_extent(codec.decode(genes))
        dims = hi-lo
        limits = np.asarray(task['initial_dimensions'])
        longest = max(limits) * (.95 if probe == 'compact' else 1.)
        package = max(0., float(max(dims / limits)-1), float(max(dims)/longest-1))
        r.update(dimensions=dims.tolist(), package_violation=package)
    except (ValueError, FloatingPointError):
        r.update(dimensions=None, package_violation=1.)
    r['feasible'] = r['routing_feasible'] and r['package_violation'] <= 1e-12
    return r


def rank(row):
    if row['status'] != 'measured':
        return (0,)
    v = list(row['violations'].values()) + [row['package_violation']]
    return (2 if row['feasible'] else 1, -max(v), -sum(v), row['quality'], row['support'])


def compact(row):
    keys = ['status','feasible','routing_feasible','quality','support','dimensions',
            'package_violation','route','reason','fields','source_clearance','detector_clearance']
    out = {k: row[k] for k in keys if k in row}
    out['violations'] = {k:v for k,v in row['violations'].items() if v > 0}
    return out


def validate(action):
    if not isinstance(action, dict) or set(action) != set(SCHEMA['required']):
        raise ValueError('action must have exactly the five schema keys')
    for k in ['probe','group','parent']:
        if action[k] not in SCHEMA['properties'][k]['enum']:
            raise ValueError('invalid '+k)
    s = action['sigma']
    if isinstance(s,bool) or not isinstance(s,(int,float)) or not np.isfinite(s) or not .005 <= s <= .15:
        raise ValueError('sigma outside [.005,.15]')
    if not isinstance(action['reason'],str) or len(action['reason']) > 1200:
        raise ValueError('reason must be <=1200 characters')


def indices(k, group):
    return {'all':np.arange(5+3*k), 'shape':np.arange(4,5+2*k),
            'curvature':np.arange(5+2*k,5+3*k), 'placement_size':np.arange(4)}[group]


def initialize(folder):
    folder = Path(folder)
    if (folder/'state.json').exists():
        return
    task = read(folder/'task.json')
    records = []
    best = {}
    for probe in task.get('batch_limits', dict(repair=2, compact=2, field=2)):
        r = measure(task, task['design']['genes'], probe)
        records.append(dict(stage='initial',probe=probe,result=r))
        best[probe] = r
    dump(folder/'state.json', dict(counts=dict.fromkeys(best,0), best=best, events=[], records=records, done=False))
    export_request(folder)


def export_request(folder):
    folder = Path(folder)
    task, state = read(folder/'task.json'), read(folder/'state.json')
    req = dict(instructions=task.get('instructions', INSTRUCTIONS), action_schema=SCHEMA, task=task,
               remaining_batches={k:task.get('batch_limits', {}).get(k, 2)-v for k,v in state['counts'].items()},
               best={k:compact(v) for k,v in state['best'].items()},history=state['events'])
    dump(folder/'request.json',req)
    return req


def step(folder, action):
    validate(action)
    folder = Path(folder)
    task, state = read(folder/'task.json'), read(folder/'state.json')
    probe = action['probe']
    if state['done'] or state['counts'].get(probe, 0) >= task.get('batch_limits', dict(repair=2, compact=2, field=2)).get(probe, 0):
        raise ValueError('probe budget exhausted or episode finalized')
    base = np.asarray(task['design']['genes'] if action['parent']=='start' else state['best'][probe]['genes'])
    ix = indices(task['design']['corners'], action['group'])
    seed = task['seed'] + len(state['events'])
    cma = CMAES(base[ix],action['sigma'],popsize=10,seed=seed,
                bounds=(-np.ones(len(ix)),np.ones(len(ix))),boundary_handling='reflect')
    before = compact(state['best'][probe])
    records, generations = [], []
    started=time.monotonic()
    for generation in range(4):
        snapshot=cma.state_snapshot()
        pop=cma.ask(); rows=[]
        for x in pop:
            genes=base.copy(); genes[ix]=x
            r=measure(task,genes,probe)
            rows.append(r)
            records.append(dict(stage='search',probe=probe,batch=len(state['events']),generation=generation,result=r))
            if rank(r)>rank(state['best'][probe]): state['best'][probe]=r
        order=sorted(range(10),key=lambda i:rank(rows[i]),reverse=True)
        cma.tell(np.asarray(order))
        generations.append(dict(before=snapshot,order=order,after=cma.state_snapshot()))
    state['counts'][probe]+=1
    event=dict(action=action,seed=seed,before=before,after=compact(state['best'][probe]),
               proposals=len(records),source_rays=sum(r['result']['source_rays'] for r in records),
               invalid_geometry=sum(r['result']['status']!='measured' for r in records),
               elapsed_seconds=time.monotonic()-started)
    dump(folder/f'batch_{len(state["events"]):02d}.json',dict(event=event,generations=generations,records=records))
    state['events'].append(event);state['records'].extend(records)
    dump(folder/'state.json',state);export_request(folder)
    return event


def finalize(folder):
    folder=Path(folder);task,state=read(folder/'task.json'),read(folder/'state.json')
    if state['done']: return read(folder/'outcome.json')
    if any(v != task.get('batch_limits', {}).get(k, 2) for k,v in state['counts'].items()): raise ValueError('all declared batches required before audit')
    outcomes={}
    for probe in state['counts']:
        start=measure(task,task['design']['genes'],probe,audit=True)
        end=measure(task,state['best'][probe]['genes'],probe,audit=True)
        replay=measure(task,state['best'][probe]['genes'],probe,audit=True)
        high=measure(task,state['best'][probe]['genes'],probe,audit=True,cap=400000)
        for label,r in [('audit_start',start),('audit_final',end),('audit_replay',replay),('audit_capacity',high)]:
            state['records'].append(dict(stage=label,probe=probe,result=r))
        outcomes[probe]=dict(initial=start,final=end,replay_exact=end==replay,capacity_agrees=end==high,
                             verified=end['feasible'] and end==replay and end==high)
    counts={}
    for row in state['records']:
        c=counts.setdefault(row['stage'],dict(evaluations=0,source_rays=0))
        c['evaluations']+=1;c['source_rays']+=row['result']['source_rays']
    result=dict(task_id=task['task_id'],label=task['label'],outcomes=outcomes,budget=counts,
                note='One optimizer trajectory per probe; 2D routing, not imaging certification or model comparison.')
    dump(folder/'outcome.json',result)
    state['done']=True;dump(folder/'state.json',state)
    return result


def run_one(job):
    folder,endpoint,model=job
    folder=Path(folder);initialize(folder)
    calls_path = folder/'model_calls.jsonl'
    prior = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
    last_error = prior[-1].get('error') if prior else None
    for _ in range(12-len(prior)):
        state=read(folder/'state.json')
        if all(v == read(folder/'task.json').get('batch_limits', {}).get(k, 2) for k,v in state['counts'].items()):return finalize(folder)
        request=export_request(folder)
        if last_error: request['previous_action_error'] = last_error
        payload=dict(model=model,messages=[dict(role='system',content=request['instructions']),
                    dict(role='user',content=json.dumps(request))],temperature=.2,max_tokens=4096,
                    response_format={'type':'json_object'})
        headers={'Content-Type':'application/json'}
        if os.environ.get('MODEL_API_KEY'):headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
        t=time.monotonic()
        req=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=300) as res: response=json.load(res)
        except Exception as error:
            record=dict(request=payload,response={},elapsed_seconds=time.monotonic()-t,error='transport: '+str(error))
            with calls_path.open('a') as f: f.write(json.dumps(record)+'\n')
            raise
        record=dict(request=payload,response=response,elapsed_seconds=time.monotonic()-t)
        try:
            action=json.loads(response['choices'][0]['message']['content'])
            event=step(folder,action)
            record['event']=event
            last_error=None
            print(folder.name,action['probe'],action['group'],event['after']['feasible'],flush=True)
        except (ValueError,TypeError,KeyError) as e:
            record['error']=str(e)
            last_error=str(e)
        with (folder/'model_calls.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    raise RuntimeError('model exceeded response limit; partial records preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    for name in ['init','request','step','finalize']:
        s=sub.add_parser(name);s.add_argument('folder',type=Path)
        if name=='step':s.add_argument('--action-file',type=Path,required=True)
    s=sub.add_parser('run');s.add_argument('root',type=Path);s.add_argument('--endpoint',default='http://127.0.0.1:8080/v1');s.add_argument('--model',default='qwen3.8-27b');s.add_argument('--workers',type=int,default=5)
    a=p.parse_args()
    if a.cmd=='init':initialize(a.folder)
    elif a.cmd=='request':print(json.dumps(export_request(a.folder),indent=2))
    elif a.cmd=='step':print(json.dumps(step(a.folder,read(a.action_file)),indent=2))
    elif a.cmd=='finalize':print(json.dumps(finalize(a.folder),indent=2))
    else:
        tasks=sorted(x.parent for x in a.root.glob('*/task.json'))
        with ProcessPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(run_one,[(str(t),a.endpoint,a.model) for t in tasks]))
        dump(a.root/'summary.json',dict(model=a.model,endpoint=a.endpoint,complete=True,results=results))

if __name__=='__main__':main()
