"""Witness-backed, bounded polygon repair corpus for an AZR-style pilot.

No witness, perturbation label, split, or counterfactual outcome reaches the solver.
The generator and solver use finite action menus; this is not open-ended AZR.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nslens.discovery.folded import RoutingBrief, evaluate
from nslens.discovery.polygon import PolygonCodec
from nslens.freeform_body.trace import trace_bundle
from nslens.search.cma_me import CMAES

STYLES = [(group, amplitude) for group in ('pose', 'curvature', 'shape')
          for amplitude in (.04, .12)]
PLANS = [(group, sigma) for group in ('all', 'curvature', 'shape_pose')
         for sigma in (.025, .10)]
FLOOR = .80


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def measure(config, genes, fidelity='search', cap=200000):
    codec = PolygonCodec(**config['codec'])
    original = RoutingBrief(**config['brief'])
    count, fields, samples = {'search': (9, 3, 256), 'audit': (25, 5, 1024),
                              'stress': (51, 7, 4096)}[fidelity]
    brief = replace(original, min_folded_throughput=max(FLOOR, original.min_folded_throughput),
                    min_root_fraction=max(FLOOR, original.min_root_fraction),
                    rays_per_field=count,
                    fields_deg=tuple(np.linspace(min(original.fields_deg), max(original.fields_deg), fields)))
    with patch('nslens.discovery.folded.trace_bundle', partial(trace_bundle, max_branches=cap)):
        r = evaluate(codec, genes, brief, geometry_samples=samples)
    # Use throughput explicitly: do not silently redefine quality if spot_weight changes.
    q = min((f['folded_throughput'] for f in r['fields']), default=0.)
    return dict(success=bool(r['feasible']), measured=r['status']=='measured',
                throughput=q, support=r['support'], fields=r['fields'],
                violations=r['violations'], rays=r['source_rays'],
                route=r.get('route'), reason=r.get('reason'), bounds=r.get('bounds'),
                unresolved=r.get('unresolved'), pruned=r.get('pruned'), closure=r.get('closure'))


def reward(r):
    # Continuous signal only for physically measured outcomes with a sound energy ledger.
    if not r['measured'] or any(r['violations'].get(k, 0)>0 for k in ('closure','unresolved','pruned')):
        return 0.
    return float(.5 * min(r['throughput']/FLOOR, r['support']/FLOOR, 1.) + .5*r['success'])


def indices(k, group):
    if group == 'all': return np.arange(5+3*k)
    if group == 'curvature': return np.arange(5+2*k, 5+3*k)
    if group == 'shape_pose': return np.arange(5+2*k)
    if group == 'pose': return np.array([0,1,4])
    if group == 'shape': return np.arange(2,5+2*k)
    raise ValueError(group)


def generate(job):
    family, style, seed = job
    rng = np.random.default_rng(seed)
    group, amplitude = STYLES[style]
    config = family['config']
    original = np.array(family['witness'])
    ix = indices(config['codec']['corners'], group)
    attempts = []
    for attempt in range(32):
        start = original.copy()
        start[ix] = np.clip(start[ix] + rng.normal(0, amplitude, len(ix)), -1, 1)
        coarse = measure(config, start)
        attempts.append(dict(genes=start.tolist(), fidelity='search', result=coarse))
        if not coarse['measured'] or coarse['success'] or not .20 <= coarse['throughput'] < FLOOR-.025:
            continue
        audit = measure(config, start, 'audit')
        attempts.append(dict(genes=start.tolist(), fidelity='audit', result=audit))
        if not audit['measured'] or audit['success'] or not .20 <= audit['throughput'] < FLOOR-.025:
            continue
        # A second grid verifies failure; no corrupted task is already solved.
        stress = measure(config, start, 'stress')
        attempts.append(dict(genes=start.tolist(), fidelity='stress', result=stress))
        if not stress['measured'] or stress['success'] or not .20 <= stress['throughput'] < FLOOR-.025:
            continue
        task = dict(task_id=f"{family['family_id']}-style{style}", family_id=family['family_id'],
                    split=family['split'], config=config, start=start.tolist(), initial=coarse,
                    initial_audit=audit, initial_stress=stress, style=style)
        return dict(task=task, attempts=attempts)
    return dict(task=None, family_id=family['family_id'], style=style, attempts=attempts)


def solve(job):
    task, plan_id, budget, seed = job
    group, sigma = PLANS[plan_id]
    x = np.array(task['start'])
    ix = indices(task['config']['codec']['corners'], group)
    cma = CMAES(x[ix], sigma, popsize=8, seed=seed,
                bounds=(-np.ones(len(ix)), np.ones(len(ix))), boundary_handling='reflect')
    best, best_r = x.copy(), task['initial']
    records = []
    for _ in range(budget//8):
        population = cma.ask()
        scores = []
        for candidate in population:
            genes = x.copy(); genes[ix] = candidate
            r = measure(task['config'], genes)
            score = reward(r); scores.append(score)
            records.append(dict(genes=genes.tolist(), result=r))
            if score > reward(best_r): best, best_r = genes.copy(), r
        cma.tell(np.argsort(-np.asarray(scores), kind='stable'))
    audit = measure(task['config'], best, 'audit')
    stress = measure(task['config'], best, 'stress')
    capacity = measure(task['config'], best, 'stress', cap=400000)
    success = bool(audit['success'] and stress['success'] and stress == capacity)
    # Reward requires success on both grids; final audit never feeds CMA.
    final_reward = min(reward(audit), reward(stress)) if stress == capacity else 0.
    return dict(task_id=task['task_id'], plan_id=plan_id, seed=seed, genes=best.tolist(),
                success=success, reward=final_reward, audit=audit, stress=stress,
                capacity_agrees=stress == capacity, records=records,
                evaluations=budget+3, rays=sum(r['result']['rays'] for r in records)+
                audit['rays']+stress['rays']+capacity['rays'])


def public_observation(task):
    return dict(config=task['config'], start=task['start'], initial=task['initial'],
                requirements=dict(worst_field_folded_throughput=FLOOR, root_support=FLOOR,
                                  preserve_original_brief=True), candidate_budget=40,
                plans=PLANS)


def select_families(root):
    grouped = {}
    for path in sorted(root.glob('discovery_ri1.50*/*/archive.json')):
        config = json.loads(path.with_name('summary.json').read_text())['config']
        if config['codec']['n'] != 1.5: continue
        # Hold out entire optical brief/geometry-range groups, not just individual elites.
        key = config['brief_id']
        for audit in json.loads(path.read_text()).get('audits', []):
            r = audit['result']
            if r['feasible'] and r['quality'] >= FLOOR and r['support'] >= FLOOR:
                grouped.setdefault(key, []).append((r['quality'], str(path), r['genes'], config))
    # Prefer corner diversity; ordering and split are fixed before solver experiments.
    ordered = sorted(grouped, key=lambda k: (int(k.rsplit('_k',1)[1]), k))
    rng = np.random.default_rng(20260915)
    rng.shuffle(ordered)
    families, admission = [], []
    for key in ordered:
        for _, path, genes, config in sorted(grouped[key], key=lambda x:-x[0])[:4]:
            clean = dict(codec=config['codec'], brief=config['brief'])
            coarse = measure(clean, genes)
            audit = measure(clean, genes, 'audit')
            stress = measure(clean, genes, 'stress')
            capacity = measure(clean, genes, 'stress', 400000)
            passed = coarse['success'] and audit['success'] and stress['success'] and stress == capacity
            admission.append(dict(source=path, genes=genes, coarse=coarse, audit=audit,
                                  stress=stress, capacity=capacity, passed=bool(passed)))
            if passed:
                families.append(dict(family_id=key, config=clean, witness=genes,
                                     source=path, witness_audit=audit, witness_stress=stress))
                break
        if len(families) == 10: break
    if len(families) < 8: raise RuntimeError(f'Only {len(families)} admitted families; need at least 8')
    for i, family in enumerate(families):
        family['split'] = 'train' if i<4 else 'validation' if i<6 else 'test'
    return families, admission


def build(args):
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    sources = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
               for base in ['nslens/discovery','nslens/freeform_body','nslens/search']
               for p in Path(base).glob('*.py')}
    sources[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(out/'protocol.json', dict(threshold=FLOOR, index=1.5, seed=20260915, plans=PLANS,
          styles=STYLES, calls_per_plan=args.budget, sources=sources,
          scope='Quality restoration, not compaction. Finite-generator AZR-style feasibility pilot.',
          split='Entire brief/range groups held out; shared representation and search priors remain.',
          leakage='Witness and corruption metadata private; no counterfactual rewards in solver input.',
          audit='5x25 plus 7x51 rays, latter at 200k/400k capacity; same physics engine.',
          costs='All generation, admission and counterfactual solver traces retained and counted.'))
    families, admission = select_families(Path('runs'))
    write(out/'private_families.json', families); write(out/'admission.json', admission)
    print('Admitted',len(families),'families', flush=True)
    jobs = [(f,s,20260915+1000*i+s) for i,f in enumerate(families)
            for s in (range(6) if f['split']=='train' else (0,3,4))]
    tasks=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(generate, jobs)):
            write(out/'private_generation'/f'{i:03d}.json',row)
            if row['task']: tasks.append(row['task'])
            print('generated',i+1,'/',len(jobs),'accepted',len(tasks),flush=True)
        write(out/'tasks.json',tasks)
        if any(not any(t['split']==s for t in tasks) for s in ('train','validation','test')):
            raise RuntimeError('Missing task split')
        jobs=[(t,p,args.budget,73000+i) for i,t in enumerate(tasks) for p in range(6)]
        for i,row in enumerate(pool.map(solve,jobs)):
            write(out/'private_outcomes'/f"{row['task_id']}_plan{row['plan_id']}.json",row)
            print('solved',i+1,'/',len(jobs),'pass',row['success'],flush=True)
    outcomes=[json.loads(p.read_text()) for p in (out/'private_outcomes').glob('*.json')]
    summary={s:dict(tasks=sum(t['split']==s for t in tasks),
                   recoverable=sum(any(r['task_id']==t['task_id'] and r['success'] for r in outcomes)
                                   for t in tasks if t['split']==s)) for s in ('train','validation','test')}
    write(out/'build_complete.json',dict(summary=summary,episodes=len(outcomes),
                                        solver_rays=sum(r['rays'] for r in outcomes)))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--budget',type=int,default=40)
    a=p.parse_args()
    if a.budget<8 or a.budget%8: p.error('budget must be a positive multiple of 8')
    build(a)
