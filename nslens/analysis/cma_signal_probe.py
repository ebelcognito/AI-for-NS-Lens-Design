"""Development diagnostic: repeated, paired CMA actions; no automatic SFT labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nslens.discovery.polygon import PolygonCodec
from nslens.workbench.cma_midpoint_pilot import continuation, evaluate_action
from nslens.workbench.interactive_cma_pilot import digest, observation, read, write


def fork_rng(state, seed):
    """Preserve the optimizer distribution; replace only future sampling randomness.

    The existing continuation executor ignores its seed argument and restores RNG
    from the snapshot. Explicit replacement is essential for independent repeats.
    """
    result = copy.deepcopy(state)
    result['optimizer']['snapshot']['rng_state'] = np.random.default_rng(seed).bit_generator.state
    return result


def run_repeat(job):
    folder, task, original, seed = job
    state = fork_rng(original, seed)
    codec = PolygonCodec(**task['config']['codec'])
    base = codec.decode(state['genes'])
    arms = {}
    for name, multiplier in [('half', .5), ('fixed', 1.), ('double', 2.)]:
        arm = evaluate_action(task, state, continuation(multiplier), seed)
        for row in arm['batch']['records']:
            body = codec.decode(row['genes'])
            row['change_from_incumbent'] = {
                'genes': (np.array(row['genes']) - state['genes']).tolist(),
                'corners_mm': (body.corners - base.corners).tolist(),
                'face_sags_mm': (body.sags - base.sags).tolist(),
                'route_changed': row['result']['route'] != state['result']['route'],
            }
        arms[name] = arm
    write(Path(folder) / f'repeat_{seed}.json', dict(seed=seed, arms=arms))
    return str(folder), seed


def summarize(root):
    report = []
    for folder in sorted(root.glob('state_*')):
        repeats = [read(p) for p in sorted(folder.glob('repeat_*.json'))]
        if not repeats:
            continue
        comparisons = {}
        for name in ['half', 'fixed', 'double']:
            arms = [r['arms'][name] for r in repeats]
            delta = np.array([r['arms'][name]['outcome']['reward'] -
                              r['arms']['fixed']['outcome']['reward'] for r in repeats])
            # Descriptive paired bootstrap, not a multiple-testing corrected label rule.
            rng = np.random.default_rng(1010)
            means = delta[rng.integers(len(delta), size=(4000, len(delta)))].mean(axis=1)
            records = [x for a in arms for x in a['batch']['records']]
            corner_shifts = [np.linalg.norm(x['change_from_incumbent']['corners_mm'], axis=1).max()
                             for x in records]
            comparisons[name] = dict(
                successes=sum(a['outcome']['success'] for a in arms),
                mean_reward=float(np.mean([a['outcome']['reward'] for a in arms])),
                mean_reward_delta_vs_fixed=float(delta.mean()),
                paired_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                win_repeats=int((delta > 1e-8).sum()), loss_repeats=int((delta < -1e-8).sum()),
                invalid_fraction=float(np.mean([not x['result']['measured'] for x in records])),
                route_change_fraction=float(np.mean([x['change_from_incumbent']['route_changed']
                                                       for x in records])),
                max_corner_displacement_mm_quantiles=np.quantile(corner_shifts, [0, .5, .9, 1]).tolist(),
            )
        report.append(dict(state=folder.name, repeats=len(repeats), actions=comparisons))
    write(root / 'summary.json', dict(
        scope='Development-only conditional action effects; no new held-out evidence or training labels.',
        caveats=['Bootstrap intervals are exploratory with few repeats and no multiplicity adjustment.',
                 'Repeated futures share the same starting state; they are not independent task families.',
                 'Route changes are canonical route-label changes, not exact physical path differences.',
                 'All arms retain the incumbent; worse candidates do not imply a worse final result.'],
        states=report))
    print(__import__('json').dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/playpen/hisham26/qwen_cma_finetune_20260915/train.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--states', type=int, default=4)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if min(args.states, args.repeats, args.workers) < 1:
        p.error('counts must be positive')
    # Only old training states, selected by ID/family before inspecting action outcomes.
    selected, families = [], set()
    for row in sorted(read(args.source), key=lambda r: r['id']):
        if row['family_id'] not in families:
            families.add(row['family_id'])
            selected.append(row)
        if len(selected) == args.states:
            break
    if len(selected) != args.states:
        p.error('not enough distinct training settings')
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for package in ['nslens/search', 'nslens/discovery', 'nslens/freeform_body', 'nslens/workbench']:
        for path in Path(package).glob('*.py'):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [Path(__file__), Path('nslens/analysis/azr_repair_pilot.py'), args.source]:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(args.out / 'protocol.json', dict(
        source=str(args.source), selected=[r['id'] for r in selected], source_hashes=sources,
        selection='First distinct family IDs in ID-sorted old training states; no outcome filtering.',
        repeats=args.repeats, actions=[.5, 1., 2.], candidates_per_action=16,
        rng='New future RNG per repeat; identical future RNG across actions within repeat.',
        budget='Only second batches and final audits are newly executed; first batches reused.',
        labels='None; do not force ties to unchanged continuation.'))
    jobs = []
    for i, row in enumerate(selected):
        tasks = read(row['task_source'])
        task = next(t for t in tasks if t['task_id'] == row['task_id'])
        state = read(row['midpoint_path'])['state']
        folder = args.out / f'state_{i:03d}'
        write(folder / 'private_source.json', dict(id=row['id'], family_id=row['family_id'],
              task=task, midpoint=state, midpoint_hash=digest(state)))
        write(folder / 'observation.json', observation(task, state))
        jobs += [(folder, task, state, 101000 + j) for j in range(args.repeats)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for folder, seed in pool.map(run_repeat, jobs):
            print(f'Completed {Path(folder).name}, seed {seed}', flush=True)
    summarize(args.out)


if __name__ == '__main__':
    main()
