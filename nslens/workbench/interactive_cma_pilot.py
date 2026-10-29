"""Development-only Qwen/CMA continuations; collect evidence before training.

Four first actions share an observation. Each branch gets its own second decision,
with no sibling outcomes or hidden witness data. All optical requirements are fixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nslens.analysis.azr_repair_pilot import generate, measure, reward, write
from nslens.discovery.polygon import PolygonCodec, exact_extent
from nslens.search.cma_me import CMAES

MODEL = 'qwen3.8-27b'
SEED = 2026091407
BATCH = 16
SYSTEM = """You control bounded CMA-ES to repair a 2D folded optical body.
Inspect the numerical optical results, physical geometry and search history.
Keep the optical brief and material immutable. Choose search controls, not direct edits.
Return only the requested JSON object in the final answer. Use a short testable rationale.
Genes are normalized to [-1,1]. A surface i connects corner i to corner (i+1)%k.
The route string uses canonical encounter IDs, NOT physical surface IDs; do not equate them.
No surface-resolved loss/TIR-margin diagnostics are available in this first interface.
Each call spends exactly 16 candidate evaluations (two populations of eight).
Restart action: {"operation":"restart","parameters":[integer gene IDs],
"scales":[positive initial standard deviations, one per parameter, 0.002 to 0.2],
"radius":number from 0.01 to 0.5,"rationale":"short explanation"}.
Restart starts at the best retained geometry, discarding old covariance and paths.
Radius defines a symmetric normalized trust region around that geometry, intersected
with [-1,1]. Scales set each parameter's initial standard deviation. All unselected
genes remain fixed. A restart uses the next recorded deterministic seed.
Continue action (only after a batch): {"operation":"continue",
"sigma_multiplier":number from 0.25 to 4,"rationale":"short explanation"}.
Continue preserves the active parameters, bounds, covariance, paths, mean and RNG;
only the global search scale is multiplied. CMA's mean may differ from the best geometry.
Search success is provisional; dense final audits determine verified success.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def compact(result):
    return {k: result[k] for k in ('success', 'measured', 'throughput', 'support',
                                   'fields', 'violations', 'reason', 'route')}


def score_key(result):
    return (reward(result), result['throughput'] if result['measured'] else 0.,
            result['support'] if result['measured'] else 0.)


def parameter_names(k):
    return (['center_x', 'center_y', 'half_width', 'half_height', 'rotation']
            + [f'corner_{i}_angle' for i in range(k)]
            + [f'corner_{i}_radial_fraction' for i in range(k)]
            + [f'face_{i}_sag' for i in range(k)])


def initial_state(task):
    return dict(genes=task['start'], result=task['initial'], optimizer=None,
                history=[], candidates=0, candidate_rays=0)


def observation(task, state):
    # Explicit allowlist: never include family/split/corruption or admission audits.
    codec = PolygonCodec(**task['config']['codec'])
    body = codec.decode(state['genes'])
    lo, hi = exact_extent(body)
    axis = int(np.argmax(np.abs([np.cos(np.deg2rad(task['config']['brief']['aim_deg'])),
                                np.sin(np.deg2rad(task['config']['brief']['aim_deg']))])))
    optimizer = state['optimizer']
    search = None if optimizer is None else dict(
        parameters=optimizer['parameters'], scales=optimizer['scales'],
        bounds=optimizer['bounds'], mean=optimizer['snapshot']['mean'],
        global_sigma=optimizer['snapshot']['sigma'],
        axis_scales=optimizer['snapshot']['axis_scales'],
        generations=optimizer['snapshot']['generation'])
    return dict(immutable_config=task['config'],
        requirements=dict(throughput=max(.8, task['config']['brief']['min_folded_throughput']),
                          support=max(.8, task['config']['brief']['min_root_fraction'])),
        parameters=[dict(id=i, name=name, value=float(state['genes'][i]))
                    for i, name in enumerate(parameter_names(codec.corners))],
        geometry=dict(corners_mm=body.corners.tolist(), face_sags_mm=body.sags.tolist(),
                      fixed_incoming_axis_thickness_mm=float((hi-lo)[axis]),
                      transverse_width_mm=float((hi-lo)[1-axis])),
        measured=compact(state['result']), search=search, history=state['history'],
        remaining_candidate_evaluations=32-state['candidates'], batch_evaluations=BATCH)


def validate_action(action, size, can_continue):
    if not isinstance(action, dict):
        raise ValueError('action must be an object')
    if not isinstance(action.get('rationale'), str) or len(action['rationale']) > 1500:
        raise ValueError('short rationale required')
    if action.get('operation') == 'continue':
        if set(action) != {'operation', 'sigma_multiplier', 'rationale'} or not can_continue:
            raise ValueError('continue requires an existing optimizer and exact keys')
        x = action['sigma_multiplier']
        if type(x) not in (int, float) or not np.isfinite(x) or not .25 <= x <= 4:
            raise ValueError('sigma multiplier outside [0.25,4]')
    elif action.get('operation') == 'restart':
        if set(action) != {'operation', 'parameters', 'scales', 'radius', 'rationale'}:
            raise ValueError('restart requires exact keys')
        ix, scales = action['parameters'], action['scales']
        if (not isinstance(ix, list) or not ix or
            any(type(i) is not int or not 0 <= i < size for i in ix) or len(set(ix)) != len(ix)):
            raise ValueError('unique valid parameter IDs required')
        if (not isinstance(scales, list) or len(scales) != len(ix) or
            any(type(s) not in (int, float) or not np.isfinite(s) or not .002 <= s <= .2 for s in scales)):
            raise ValueError('one scale in [0.002,0.2] per parameter required')
        radius = action['radius']
        if type(radius) not in (int, float) or not np.isfinite(radius) or not .01 <= radius <= .5:
            raise ValueError('radius outside [0.01,0.5]')
    else:
        raise ValueError('unknown operation')


def run_batch(task, state, action, seed):
    started = time.monotonic()
    validate_action(action, len(state['genes']), state['optimizer'] is not None)
    if state['candidates'] + BATCH > 32:
        raise ValueError('candidate budget exhausted')
    state = copy.deepcopy(state)
    before = compact(state['result'])
    if action['operation'] == 'restart':
        ix = np.array(action['parameters'])
        scales = np.array(action['scales'])
        anchor = np.array(state['genes'])
        lo = (np.maximum(-1, anchor[ix]-action['radius']) - anchor[ix]) / scales
        hi = (np.minimum(1, anchor[ix]+action['radius']) - anchor[ix]) / scales
        cma = CMAES(np.zeros(len(ix)), 1., popsize=8, seed=seed,
                    bounds=(lo, hi), boundary_handling='reflect')
    else:
        old = state['optimizer']
        ix, scales, anchor = np.array(old['parameters']), np.array(old['scales']), np.array(old['anchor'])
        lo, hi = np.array(old['bounds'])
        cma = CMAES.from_state_snapshot(old['snapshot'], bounds=(lo, hi))
        cma.sigma *= action['sigma_multiplier']
    records, snapshots = [], []
    for _ in range(2):
        snapshot = cma.state_snapshot()
        population = cma.ask()
        rows = []
        for candidate in population:
            genes = anchor.copy()
            genes[ix] = np.clip(anchor[ix] + scales*candidate, -1, 1)
            result = measure(task['config'], genes)
            rows.append(result)
            records.append(dict(genes=genes.tolist(), result=result))
            if score_key(result) > score_key(state['result']):
                state['genes'], state['result'] = genes.tolist(), result
        order = np.array(sorted(range(8), key=lambda i: score_key(rows[i]), reverse=True))
        cma.tell(order)
        snapshots.append(dict(before=snapshot, order=order.tolist(), after=cma.state_snapshot()))
    state['optimizer'] = dict(parameters=ix.tolist(), scales=scales.tolist(), anchor=anchor.tolist(),
                              bounds=[lo.tolist(), hi.tolist()], snapshot=cma.state_snapshot())
    state['candidates'] += BATCH
    rays = sum(row['result']['rays'] for row in records)
    state['candidate_rays'] += rays
    scores = [reward(row['result']) for row in records]
    event = dict(action=action, seed=seed, before=before, after=compact(state['result']),
                 throughput_delta=state['result']['throughput']-before['throughput'],
                 invalid_candidates=sum(not r['result']['measured'] for r in records),
                 population_reward_min=min(scores), population_reward_max=max(scores),
                 population_reward_std=float(np.std(scores)), candidate_rays=rays)
    state['history'].append(event)
    return dict(state=state, event=event, records=records, generations=snapshots,
                elapsed_seconds=time.monotonic()-started)


def finalize(task, state):
    start = time.monotonic()
    audit = measure(task['config'], state['genes'], 'audit')
    stress = measure(task['config'], state['genes'], 'stress')
    capacity = measure(task['config'], state['genes'], 'stress', 400000)
    agrees = stress == capacity
    return dict(success=bool(audit['success'] and stress['success'] and agrees),
                reward=min(reward(audit), reward(stress)) if agrees else 0.,
                audit=audit, stress=stress, capacity=capacity, capacity_agrees=agrees,
                candidate_evaluations=state['candidates'], audit_evaluations=3,
                source_rays=state['candidate_rays']+audit['rays']+stress['rays']+capacity['rays'],
                audit_seconds=time.monotonic()-start)


def fixed_baseline(task):
    n = len(task['start'])
    action = dict(operation='restart', parameters=list(range(n)), scales=[.025]*n,
                  radius=.5, rationale='Fixed all-parameter local CMA baseline.')
    first = run_batch(task, initial_state(task), action, SEED)
    second = run_batch(task, first['state'], dict(operation='continue', sigma_multiplier=1.,
                       rationale='Continue unchanged.'), SEED+1)
    return dict(batches=[first, second], outcome=finalize(task, second['state']))


class Qwen:
    def __init__(self, endpoint, out):
        self.endpoint, self.out = endpoint.rstrip('/'), Path(out)
        with urllib.request.urlopen(self.endpoint+'/models', timeout=10) as response:
            models = json.load(response)
        if MODEL not in [m['id'] for m in models['data']]:
            raise RuntimeError(f'required local model {MODEL} not served')
        write(self.out/'served_model.json', models)

    def decide(self, obs, count, key):
        path = self.out/'model_calls'/f'{key}.json'
        if path.exists():
            cached = read(path)
            if cached['observation_hash'] != digest(obs):
                raise ValueError('cached observation mismatch')
            return cached['actions']
        instruction = (f'Return {{"actions":[{count} action objects]}}. '
                       + ('Propose four distinct plausible first search strategies from this same state.'
                          if count == 4 else 'Choose the next action using only this branch history.'))
        request = dict(model=MODEL, messages=[dict(role='system', content=SYSTEM),
                       dict(role='user', content=json.dumps(dict(instruction=instruction, observation=obs)))],
                       temperature=.7, top_p=.9, seed=SEED,
                       max_tokens=3072, reasoning_budget_tokens=512,
                       response_format=dict(type='json_object'))
        attempts_path = path.with_name(path.stem+'_attempts.json')
        attempts = read(attempts_path) if attempts_path.exists() else []
        for attempt in range(len(attempts), 3):
            start = time.monotonic()
            req = urllib.request.Request(self.endpoint+'/chat/completions',
                  data=json.dumps(request).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=300) as response:
                raw = json.load(response)
            record = dict(request=copy.deepcopy(request), response=raw,
                          elapsed_seconds=time.monotonic()-start)
            attempts.append(record)
            try:
                choice = raw['choices'][0]
                if choice['finish_reason'] == 'length':
                    raise ValueError('truncated model response')
                obj = json.loads(choice['message']['content'])
                if set(obj) != {'actions'} or len(obj['actions']) != count:
                    raise ValueError(f'exactly {count} actions required')
                for action in obj['actions']:
                    validate_action(action, len(obs['parameters']), obs['search'] is not None)
            except (ValueError, TypeError, KeyError) as exc:
                record['error'] = str(exc)
                write(attempts_path, attempts)
                request['messages'][-1]['content'] += '\nPrevious response rejected: '+str(exc)
                continue
            write(path, dict(observation_hash=digest(obs), actions=obj['actions'], attempts=attempts))
            return obj['actions']
        raise RuntimeError(f'model failed three valid-action attempts: {path}')


def prepare(root, corpus, count, workers):
    if (root/'tasks.json').exists():
        return read(root/'tasks.json')
    families = [f for f in read(corpus/'private_families.json') if f['split'] != 'test']
    tasks, seen = [], {digest(t['start']) for t in read(corpus/'tasks.json')}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for wave in range(8):
            jobs = [(family, (wave+i)%6, SEED+wave*100+i)
                    for i, family in enumerate(families)]
            for i, row in enumerate(pool.map(generate, jobs)):
                write(root/'private_generation'/f'{wave:02d}_{i:02d}.json', row)
                if row['task'] and digest(row['task']['start']) not in seen and len(tasks) < count:
                    task = row['task']
                    task['task_id'] = f'dev_{len(tasks):03d}'
                    task['split'] = 'development'
                    tasks.append(task)
                    seen.add(digest(task['start']))
            print(f'generated {len(tasks)}/{count} fresh development tasks', flush=True)
            if len(tasks) == count:
                break
    if len(tasks) != count:
        raise RuntimeError(f'only generated {len(tasks)} tasks')
    write(root/'tasks.json', tasks)
    return tasks


def report(root, tasks):
    rows, preferences = [], []
    for task in tasks:
        folder = root/'episodes'/task['task_id']
        branches = [read(p) for p in sorted(folder.glob('branch_*.json'))]
        if len(branches) != 4:
            continue
        baseline = read(folder/'baseline.json')['outcome']
        outcomes = [b['outcome'] for b in branches]
        rewards = [r['reward'] for r in outcomes]
        eligible = [i for i,b in enumerate(branches) if b.get('status') != 'model_failure']
        for offset, i in enumerate(eligible):
            for j in eligible[offset+1:]:
                # A 0.05 gap avoids labeling tiny coarse sampling differences.
                if abs(rewards[i]-rewards[j]) < .05:
                    continue
                win, lose = (i,j) if rewards[i]>rewards[j] else (j,i)
                preferences.append(dict(task_id=task['task_id'], family_id=task['family_id'],
                    observation=observation(task, initial_state(task)),
                    chosen=branches[win]['actions'], rejected=branches[lose]['actions'],
                    chosen_reward=rewards[win], rejected_reward=rewards[lose],
                    label_scope='two-action trajectory preference, NOT immediate-action causality'))
        rows.append(dict(task_id=task['task_id'], family_id=task['family_id'],
            successes=[r['success'] for r in outcomes], rewards=rewards,
            baseline_success=baseline['success'], baseline_reward=baseline['reward'],
            reward_spread=max(rewards)-min(rewards),
            initial_actions=[b['actions'][0] for b in branches],
            followup_operations=[b['actions'][1]['operation'] if len(b['actions'])>1
                                 else 'model_failure' for b in branches]))
    calls = [read(p) for p in (root/'model_calls').glob('*.json') if not p.stem.endswith('_attempts')]
    attempts = [a for c in calls for a in c['attempts']]
    # Exhausted calls have only an attempts file. Include their costs exactly once.
    for p in (root/'model_calls').glob('*_attempts.json'):
        if not p.with_name(p.name.replace('_attempts.json','.json')).exists():
            attempts.extend(read(p))
    artifacts = [read(p) for p in (root/'episodes').glob('*/*.json')
                 if p.name == 'baseline.json' or p.name.startswith('branch_')]
    summary = dict(complete=len(rows)==len(tasks), completed_tasks=len(rows), total_tasks=len(tasks),
        model=MODEL, model_weights_trained=False, task_families=len(set(t['family_id'] for t in tasks)),
        qwen_continuations=len(rows)*4, verified_qwen_repairs=sum(sum(r['successes']) for r in rows),
        fixed_baseline_repairs=sum(r['baseline_success'] for r in rows),
        any_of_four_repairs=sum(any(r['successes']) for r in rows),
        first_branch_repairs=sum(r['successes'][0] for r in rows),
        tasks_with_success_disagreement=sum(0<sum(r['successes'])<4 for r in rows),
        tasks_with_reward_gap_005=sum(r['reward_spread']>=.05 for r in rows),
        trajectory_preference_pairs=len(preferences), model_requests=len(attempts),
        rejected_model_responses=sum('error' in a for a in attempts),
        input_tokens=sum(a['response'].get('usage',{}).get('prompt_tokens',0) for a in attempts),
        output_tokens=sum(a['response'].get('usage',{}).get('completion_tokens',0) for a in attempts),
        model_seconds=sum(a['elapsed_seconds'] for a in attempts),
        solver_source_rays=sum(a['outcome']['source_rays'] for a in artifacts),
        candidate_evaluations=sum(a['outcome']['candidate_evaluations'] for a in artifacts),
        audit_evaluations=sum(a['outcome']['audit_evaluations'] for a in artifacts),
        model_failed_continuations=sum(a.get('status')=='model_failure' for a in artifacts),
        limitations=['Development families only; no held-out generalization claim.',
            'Any-of-four spends four times one-continuation search budget; diagnostic only.',
            'Single numerical seed per strategy; preference gaps can include search luck.',
            'Numerical geometry/field diagnostics only; no images or surface loss maps.',
            'Two CMA decisions; no direct edits or separate diagnostic probes.',
            'Four first alternatives requested jointly; not four independent policy samples.',
            'Malformed-response exhaustion aborts that branch; unused evaluations are not replaced.'], rows=rows)
    write(root/'summary.json', summary)
    write(root/'trajectory_preferences.json', preferences)
    return summary


def run_branch(job):
    root, task, action, i, qwen = job
    folder = root/'episodes'/task['task_id']
    path = folder/f'branch_{i}.json'
    if path.exists():
        return
    first_path = folder/f'first_{i}.json'
    first = read(first_path) if first_path.exists() else run_batch(task, initial_state(task), action, SEED)
    write(first_path, first)
    try:
        following = qwen.decide(observation(task, first['state']), 1, task['task_id']+f'_followup_{i}')[0]
    except RuntimeError as exc:
        if not str(exc).startswith('model failed three valid-action attempts'):
            raise
        outcome = finalize(task, first['state'])
        # Score the actual retained geometry, without silently executing a fallback.
        write(path, dict(actions=[action], batches=[first], outcome=outcome,
                         status='model_failure', error=str(exc)))
        print(task['task_id'], f'branch {i+1}/4 aborted: invalid model responses', flush=True)
        return
    second = run_batch(task, first['state'], following, SEED+1)
    outcome = finalize(task, second['state'])
    write(path, dict(actions=[action, following], batches=[first, second], outcome=outcome))
    print(task['task_id'], f'branch {i+1}/4', 'verified', outcome['success'],
          'reward', round(outcome['reward'],4), flush=True)


def run(root, tasks, endpoint, limit):
    qwen = Qwen(endpoint, root)
    for task in tasks[:limit]:
        folder = root/'episodes'/task['task_id']
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder/'baseline.json').exists():
            write(folder/'baseline.json', fixed_baseline(task))
        obs = observation(task, initial_state(task))
        first_actions = qwen.decide(obs, 4, task['task_id']+'_initial')
        # Isolated processes keep tracer monkeypatches independent. Model requests
        # occupy separate server slots; observations never mix branch outcomes.
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(run_branch, [(root,task,a,i,qwen) for i,a in enumerate(first_actions)]))
        summary = report(root, tasks)
        print(f"Completed {summary['completed_tasks']}/{len(tasks)} tasks; "
              f"{summary['trajectory_preference_pairs']} preference pairs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','report'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=Path('runs/azr_repair_pilot_20260914_v1/corpus'))
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        write(args.out/'protocol.json', dict(model=MODEL, seed=SEED, tasks=args.count,
              continuations_per_task=4, decisions_per_continuation=2, evaluations_per_decision=BATCH,
              scope='Frozen Qwen development data collection; quality repair, not compaction or training.',
              corpus=str(args.corpus.resolve()), system_prompt=SYSTEM,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        prepare(args.out, args.corpus, args.count, args.workers)
    elif args.command == 'run':
        run(args.out, read(args.out/'tasks.json'), args.endpoint, args.limit)
    else:
        print(json.dumps(report(args.out, read(args.out/'tasks.json')), indent=2))


if __name__ == '__main__':
    main()
