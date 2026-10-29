"""Budgeted inception and direct-geometry model runs on the same camera task."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import numpy as np
from nslens.search.cma_me import CMAES
from nslens.discovery.portfolio import RestartPool, constraint_rank
from .problem import CameraCodec, CameraBrief, evaluate, audit


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path,value):
    with path.open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')


def camera_rank(row):
    if row['status'] != 'measured':
        return (0,)
    if row['feasible']:
        return (2, row['quality'], row['support'])
    graded = row.get('sequential_grade') or {}
    fields = row.get('fields') or ()
    throughputs = [field['throughput'] for field in fields]
    if fields and min(throughputs) >= .05:
        return (1, 1, min(field['root_support'] for field in fields),
                min(throughputs),
                -max(field['centroid_error_mm'] if field['centroid_error_mm'] is not None else 1e3
                     for field in fields),
                -max(field['rms_mm'] if field['rms_mm'] is not None else 1e3
                     for field in fields))
    return (1, 0, graded.get('worst_score', 0.), graded.get('mean_score', 0.),
            *constraint_rank(row))


def task(codec,brief):
    return dict(codec=asdict(codec),brief=asdict(brief),geometry_schema=codec.design(np.zeros(codec.gene_size)),
        instructions='Construct a 2D thin imaging camera from scratch. Propose complete geometry, trace it, and revise. '
        'The schema example is only a format example and may overlap; it is not a starting prescription. '
        'No optimizer is available. Each proposal costs one candidate, including invalid requests. '
        'Output JSON only: {"action":"propose","design":{"bodies":[...]}} or '
        '{"action":"finish","candidate":integer}. Candidate indices start at zero. '
        'The source is at (4, thickness/2+2), aims down -y, aperture=EFL/f-number; '
        'sensor is at x=max_length, spanning y=+-sensor_half. All bodies must lie in '
        'x=[0,max_length-.1], y=+-thickness/2. Body bounding boxes must be separated by .02 mm '
        '(conservative geometry restriction). Coordinates are mm. Bodies are closed CCW polygons '
        'with circular edges. For K corners: corner angle=360*i/K+rotation+angular_offset; '
        'corner=center+half_size*radial_fraction*(cos(angle),sin(angle)). '
        'center bounds: x uses codec.placement_bounds by body, or 1..22 when empty; '
        'y -1..1; half sizes: x .5..3,y .5..3; rotation -180..180; '
        'angular offsets +-72/K degrees; radial fractions .65..1. '
        'curvature_controls are -1..1: sag=control*min(.5,minimum-radius-limit)*half-chord, '
        'positive outward. Minimum radius=.5 mm, edge=.1 mm. '
        'mirrors are per-edge boolean ideal lossless coatings; false means uncoated Fresnel/TIR. '
        'departures_mm are Kx3 polynomial coefficients, each bounded by departure_bound_mm; '
        'when bound=0 all must be zero. Departure=16*t^2*(1-t)^2*(c0+c1*(2*t-1)+c2*(6*t^2-6*t+1)), '
        'along outward chord normal. Material index is fixed. '
        'No required fold count or surface sequence. Score all sensor-reaching branches that interacted with optics. '
        'Image targets use one common parity times EFL*tan(field), no fitted focal length. '
        'Use the physical observations to improve imaging; no patent prescription is supplied.')


class Session:
    def __init__(self, out, codec, brief, budget, ray_budget):
        self.out=out; self.codec=codec; self.brief=brief; self.budget=budget; self.ray_budget=ray_budget
        self.rows=[]; self.rays=0; self.best=None

    def available(self):
        return len(self.rows)<self.budget and self.rays+len(self.brief.rays())<=self.ray_budget

    def propose(self, design=None, genes=None):
        if not self.available(): raise ValueError('candidate or source-ray budget exhausted')
        try:
            if genes is None: genes=self.codec.encode(design)
            row=evaluate(self.codec,genes,self.brief)
        except (ValueError,TypeError,KeyError,IndexError) as e:
            row=dict(status='protocol',reason=str(e),genes=None,feasible=False,source_rays=0,
                     fields=[],quality=0.,support=0.,violations={},progress=(0.,0.,0.))
        row['candidate']=len(self.rows); self.rays+=row['source_rays']; self.rows.append(row)
        append(self.out/'candidates.jsonl',row)
        if row['genes'] is not None and (self.best is None or camera_rank(row)>camera_rank(self.best)):
            self.best=row
        return row

    def finalize(self, candidate=None):
        selected=self.best if candidate is None else self.rows[candidate]
        result=None
        if selected is not None and selected['genes'] is not None:
            result=audit(self.codec,selected['genes'],self.brief)
            write(self.out/'design.json',self.codec.design(selected['genes']))
            write(self.out/'selected.json',selected)
        summary=dict(candidates=len(self.rows),source_rays=self.rays,
            selected_candidate=selected['candidate'] if selected else None,
            selection='best measured candidate' if candidate is None else 'explicit model submission',
            audit=result,success=bool(result and result['passed']))
        write(self.out/'summary.json',summary); return summary


def inception(session,seed,popsize=4,initial_genes=None,sigma=.2):
    rng=np.random.default_rng(seed); pool=RestartPool(ranking=camera_rank); emitter=None; age=0
    if initial_genes is not None:
        initial_genes=np.asarray(initial_genes,dtype=float)
        first=session.propose(genes=initial_genes)
        pool.update([first])
    while session.available():
        if emitter is None:
            start=initial_genes if initial_genes is not None else pool.start()
            initial_genes=None
            if start is None: start=rng.uniform(-1,1,session.codec.gene_size)
            emitter=CMAES(start,sigma,popsize=popsize,seed=int(rng.integers(2**30)),
                bounds=(-np.ones(len(start)),np.ones(len(start))),boundary_handling='reflect'); age=0
        population=emitter.ask(); rows=[]
        for genes in population:
            if not session.available(): break
            rows.append(session.propose(genes=genes))
        pool.update(rows)
        if len(rows)!=len(population): break
        order=np.asarray(sorted(range(len(rows)),key=lambda i:camera_rank(rows[i]),reverse=True))
        emitter.tell(order); age+=1
        if emitter.restart_needed or age>=5: emitter=None
    return session.finalize()


def model_run(session,endpoint,model,seed,max_output_tokens,max_calls,reasoning_tokens=None):
    public=task(session.codec,session.brief); history=[]; total_tokens=0; submitted=None; error=None
    for request_index in range(max_calls):
        if not session.available(): break
        observation=dict(task=public,remaining_candidates=session.budget-len(session.rows),
                         previous=history[-3:],best=session.best,
                         best_design=session.codec.design(session.best['genes']) if session.best else None)
        payload=dict(model=model,messages=[dict(role='system',content=public['instructions']),
            dict(role='user',content=json.dumps(observation))],temperature=.2,seed=seed,
            max_tokens=max_output_tokens,response_format={'type':'json_object'})
        if reasoning_tokens is not None: payload['reasoning_budget_tokens']=reasoning_tokens
        headers={'Content-Type':'application/json'}
        if os.environ.get('MODEL_API_KEY'): headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
        request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(),headers=headers)
        started=time.monotonic()
        try:
            with urllib.request.urlopen(request,timeout=300) as response: data=json.load(response)
        except Exception as exc:
            error='transport: '+str(exc)
            append(session.out/'model_calls.jsonl',dict(request=payload,error=error,wall_seconds=time.monotonic()-started))
            break
        total_tokens+=data.get('usage',{}).get('total_tokens',0)
        append(session.out/'model_calls.jsonl',dict(request=payload,response=data,wall_seconds=time.monotonic()-started))
        action=None
        try:
            action=json.loads(data['choices'][0]['message']['content'])
            if not isinstance(action,dict): raise ValueError('action must be a JSON object')
            if action.get('action')=='finish':
                index=action.get('candidate')
                if type(index) is not int or not 0<=index<len(session.rows) or session.rows[index]['genes'] is None:
                    raise ValueError('finish requires a previously evaluated valid candidate index')
                submitted=index; break
            if action.get('action')!='propose' or set(action)!={'action','design'}:
                raise ValueError('expected propose with design')
            row=session.propose(design=action['design'])
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            # Every malformed response consumes a candidate slot; no free retries.
            row=session.propose(design={'error':str(exc)})
            row=dict(row,response_error=str(exc))
        history.append(dict(design=action.get('design') if isinstance(action,dict) else None,result=row))
    result=session.finalize(submitted)
    result.update(model=model,endpoint=endpoint,reported_total_tokens=total_tokens,
                  submitted=submitted is not None,transport_error=error,
                  success=result['success'] and error is None)
    write(session.out/'summary.json',result); return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm',choices=['inception','model','task'])
    p.add_argument('--out',type=Path,required=True); p.add_argument('--budget',type=int,default=100)
    p.add_argument('--ray-budget',type=int); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--corners',type=int,nargs='+',default=[4,3,3])
    p.add_argument('--placement-bounds',help='x-centre intervals per body, e.g. 3,6;10,13;18,21')
    p.add_argument('--seed-design',type=Path,help='existing physical design; charged as candidate 0')
    p.add_argument('--sigma',type=float,default=.2,help='initial CMA sigma in normalized coordinates')
    p.add_argument('--freeform-bound',type=float,default=0.)
    p.add_argument('--endpoint',default='http://127.0.0.1:8080/v1')
    p.add_argument('--model',default='qwen3.8-27b'); p.add_argument('--max-output-tokens',type=int,default=4096)
    p.add_argument('--max-calls',type=int,default=24)
    p.add_argument('--reasoning-tokens',type=int,help='Optional llama.cpp per-request thinking budget; omit for other endpoints')
    a=p.parse_args(argv)
    if a.reasoning_tokens is not None and not 0<=a.reasoning_tokens<a.max_output_tokens: p.error('reasoning budget must be nonnegative and smaller than output cap')
    if min(a.budget,a.max_output_tokens,a.max_calls)<1 or not np.isfinite(a.sigma) or a.sigma<=0:
        p.error('budgets and sigma must be positive')
    try:
        placement = tuple(tuple(float(v) for v in part.split(','))
                          for part in a.placement_bounds.split(';')) if a.placement_bounds else ()
        codec=CameraCodec(tuple(a.corners),departure_bound_mm=a.freeform_bound,
                          placement_bounds=placement)
    except ValueError as exc:
        p.error(str(exc))
    brief=CameraBrief()
    ray_budget=a.ray_budget if a.ray_budget is not None else a.budget*len(brief.rays())
    if ray_budget<1: p.error('ray budget must be positive')
    initial_design = json.loads(a.seed_design.read_text()) if a.seed_design else None
    initial_genes = codec.encode(initial_design) if initial_design is not None else None
    a.out.mkdir(parents=True,exist_ok=False)
    sources=[*Path(__file__).parent.glob('*.py'),Path(__file__).parents[1]/'freeform_body'/'trace.py',
             Path(__file__).parents[1]/'freeform_body'/'arc_polygon.py',Path(__file__).parents[1]/'freeform_body'/'polynomial_faces.py',
             Path(__file__).parents[1]/'discovery'/'polygon.py',Path(__file__).parents[1]/'discovery'/'portfolio.py',
             Path(__file__).parents[1]/'search'/'cma_me.py']
    write(a.out/'protocol.json',dict(arm=a.arm,seed=a.seed,candidate_budget=a.budget,ray_budget=ray_budget,
        population_size=4, sigma=a.sigma,max_calls=a.max_calls,max_output_tokens=a.max_output_tokens,reasoning_tokens=a.reasoning_tokens,
        seed_design=str(a.seed_design) if a.seed_design else None,
        task=task(codec,brief),sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        caveat='2D monochromatic development pilot; ideal mirrors; conservative disjoint bounding boxes; no 3D camera claim.'))
    if a.arm=='task': return
    session=Session(a.out,codec,brief,a.budget,ray_budget); started=time.monotonic()
    result=inception(session,a.seed,initial_genes=initial_genes,sigma=a.sigma) if a.arm=='inception' else model_run(session,a.endpoint,a.model,a.seed,a.max_output_tokens,a.max_calls,a.reasoning_tokens)
    result['wall_seconds']=time.monotonic()-started; write(a.out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='audit'},indent=2))

if __name__=='__main__': main()
