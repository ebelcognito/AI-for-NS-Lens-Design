"""Collect accepted geometry repair moves from independent CMA searches.

Each CMA population is evaluated against the incumbent at the start of its
generation. A saved delta is therefore an observed improvement from that
incumbent, not the optimizer's sampled step from its internal mean.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from nslens.analysis.adapter_refinement import evaluate_state
from nslens.analysis.mirror_count_experiment import initial
from nslens.search.cma_me import CMAES


def _task_map():
    # Keep the collector importable without the pilot's optional sklearn stack.
    from nslens.analysis.mirror_adapter_pilot import TASKS
    return TASKS


def features(state, row, task):
    from nslens.analysis.mirror_adapter_pilot import features as pilot_features
    return pilot_features(state, row, task)


def save(path, obj):
    from nslens.analysis.mirror_adapter_pilot import save as pilot_save
    pilot_save(path, obj)


def collect(job: dict, budget: int = 257) -> dict:
    """Collect improving parent-to-child observations using exactly budget calls."""
    if not isinstance(job, dict) or 'seed' not in job or 'task' not in job:
        raise ValueError("job must contain seed and task")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")

    seed = int(job['seed'])
    task_spec = job['task']
    if isinstance(task_spec, str):
        tasks = _task_map()
        if task_spec not in tasks:
            raise ValueError(f"unknown task: {task_spec}")
        task_name, task = task_spec, tasks[task_spec]
    elif isinstance(task_spec, dict):
        task_name, task = job.get('task_name', 'custom'), task_spec
    else:
        raise ValueError("task must be a task name or task dictionary")

    rng = np.random.default_rng(seed)
    state = np.r_[initial(3, rng), np.zeros(4)]
    incumbent_state = state.copy()
    incumbent = evaluate_state(state, task)
    initial_loss = float(incumbent['loss'])
    evaluations = 1
    cma = CMAES(state[:9], .12, popsize=8, seed=seed,
                bounds=(-np.ones(9), np.ones(9)), boundary_handling='reflect')
    transitions = []
    diagnostics = dict(children=0, improving=0, rejected_loss=0,
                       rejected_crossing=0, rejected_support=0)

    while evaluations < budget:
        population = cma.ask()
        count = min(len(population), budget - evaluations)
        generation_parent = incumbent_state.copy()
        generation_row = incumbent
        parent_loss = float(generation_row['loss'])
        parent_support = float(generation_row['worst_support'])
        parent_features = features(generation_parent, generation_row, task)
        rows = []
        for vector in population[:count]:
            child_state = generation_parent.copy()
            child_state[:9] = np.asarray(vector, dtype=float)
            child_row = evaluate_state(child_state, task)
            rows.append(child_row)
            evaluations += 1
            diagnostics['children'] += 1
            better = child_row['loss'] < parent_loss
            clear = not child_row['crossing']
            supported = child_row['worst_support'] >= parent_support - .01
            if better:
                diagnostics['improving'] += 1
            if not better:
                diagnostics['rejected_loss'] += 1
            if not clear:
                diagnostics['rejected_crossing'] += 1
            if not supported:
                diagnostics['rejected_support'] += 1
            if better and clear and supported:
                transitions.append(dict(
                    features=parent_features.copy(),
                    delta=(child_state[:9] - generation_parent[:9]).copy(),
                    parent=generation_parent.copy(), child=child_state.copy(),
                    task=task_name, parent_loss=parent_loss,
                    child_loss=float(child_row['loss']),
                    parent_support=parent_support,
                    child_support=float(child_row['worst_support']),
                    seed=seed, evaluations=evaluations,
                ))
            if child_row['loss'] < incumbent['loss']:
                incumbent, incumbent_state = child_row, child_state.copy()
        if count == len(population):
            cma.tell(np.argsort([row['loss'] for row in rows]))

    return dict(transitions=transitions, final_state=incumbent_state,
                coarse_calls=evaluations, initial_loss=initial_loss,
                final_loss=float(incumbent['loss']), seed=seed,
                task=task_name, diagnostics=diagnostics)


def _run_job(args):
    seed, task_name, budget = args
    return collect(dict(seed=seed, task=task_name), budget=budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seeds', type=int, default=48)
    parser.add_argument('--seed-start', type=int, default=910000)
    parser.add_argument('--budget', type=int, default=257)
    args = parser.parse_args()
    if args.workers < 1 or args.seeds < 1:
        parser.error('--workers and --seeds must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tasks = ('field8', 'aperture42')
    jobs = [(args.seed_start + i, tasks[i % len(tasks)], args.budget)
            for i in range(args.seeds)]
    save(out / 'protocol.json', dict(
        tasks=list(tasks), seeds=[args.seed_start, args.seed_start + args.seeds - 1],
        budget_per_seed=args.budget, initial='initial(3, default_rng(seed)) + four zero freeform coefficients',
        search='Independent 9-parameter CMAES, sigma .12, population 8, reflected [-1,1] bounds; freeform coefficients remain zero.',
        parent='Incumbent at the start of each generation; a saved delta is the observed improvement from that parent, not a step from CMA internal mean.',
        retention='Keep only lower-merit children that do not cross and preserve worst support within .01.',
        records='Parent optical features at 5 fields x 21 pupils, 9D geometry delta, parent/child 13D states and evaluations count.',
    ))
    by_seed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_run_job, jobs):
            path = out / f"seed_{result['seed']}_{result['task']}.json"
            save(path, result)
            by_seed.append(result)
    summary = dict(runs=len(by_seed), tasks=list(tasks), budget_per_seed=args.budget,
                   evaluations=sum(row['coarse_calls'] for row in by_seed),
                   transitions=sum(len(row['transitions']) for row in by_seed),
                   children=sum(row['diagnostics']['children'] for row in by_seed),
                   diagnostics={key: sum(row['diagnostics'][key] for row in by_seed)
                                for key in by_seed[0]['diagnostics']})
    save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
