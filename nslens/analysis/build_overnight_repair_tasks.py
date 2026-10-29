"""Development repair tasks from the overnight inventory, with fixed-CMA profiling."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil

from nslens.analysis.azr_repair_pilot import generate, measure, STYLES
from nslens.workbench.interactive_cma_pilot import (
    digest, finalize, initial_state, observation, read, run_batch, write,
)
from nslens.workbench.cma_midpoint_pilot import fixed_start, continuation

SEED = 2026091510


def family_job(job):
    root, index, source = job
    folder = root/'private_generation'/f'family_{index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    config = source['config']
    witness_path = folder/'witness_check.json'
    if not witness_path.exists():
        outcome = finalize(dict(config=config), dict(genes=source['genes'], candidates=0, candidate_rays=0))
        write(witness_path, outcome)
    assert read(witness_path)['success'], 'Source witness failed fresh verification'
    family = dict(family_id=source['family_id'], split='development', config=config, witness=source['genes'])
    results = []
    for style in range(len(STYLES)):
        seed = SEED + index*100 + style
        path = folder/f'style_{style}.json'
        if not path.exists():
            row = generate((family, style, seed))
            row.update(seed=seed, witness_id=source['id'])
            if row['task']:
                task = row['task']
                capacity = measure(config, task['start'], 'stress', 400000)
                row['attempts'].append(dict(genes=task['start'], fidelity='stress_capacity_400000', result=capacity))
                if capacity != task['initial_stress']:
                    row['rejection'] = 'Initial stress disagrees at doubled capacity'
                    row['task'] = None
                else:
                    task.update(task_id=f'newdev_{index:02d}_{style}',
                                initial_capacity=capacity, witness_id=source['id'], generation_seed=seed)
            write(path, row)
        row = read(path)
        if row['task']:
            task = row['task']
            obs_path = root/'observations'/f'{task["task_id"]}.json'
            write(obs_path, observation(task, initial_state(task)))
            results.append(task)
    return results


def profile_job(job):
    root, task = job
    path = root/'profiles'/f'{task["task_id"]}.json'
    if path.exists():
        return read(path)
    seed = task['generation_seed']+100000
    first = run_batch(task, initial_state(task), fixed_start(task), seed)
    midpoint = finalize(task, first['state'])
    second = run_batch(task, first['state'], continuation(1.), seed+1)
    result = finalize(task, second['state'])
    row = dict(task_id=task['task_id'], seed=seed, first=first, midpoint=midpoint,
               second=second, outcome=result)
    write(path, row)
    return row


def report(root):
    tasks = read(root/'tasks.json')
    profiles = [read(p) for p in sorted((root/'profiles').glob('*.json'))]
    rows = [read(p) for p in (root/'private_generation').glob('*/style_*.json')]
    witness_checks = [read(p) for p in (root/'private_generation').glob('*/witness_check.json')]
    attempts = [a for r in rows for a in r['attempts']]
    eligible = [p['task_id'] for p in profiles if not p['first']['state']['result']['success']]
    dense_failing = [p['task_id'] for p in profiles if not p['midpoint']['success']]
    write(root/'midpoint_coarse_failing_ids.json', eligible)
    write(root/'midpoint_audit_failing_ids.json', dense_failing)
    summary = dict(complete=len(profiles)==len(tasks), tasks=len(tasks), attempted_recipes=len(rows),
        rejected_recipes=sum(r['task'] is None for r in rows),
        task_settings=len({t['family_id'] for t in tasks}),
        corners=dict(Counter(t['config']['codec']['corners'] for t in tasks)),
        styles=dict(Counter(STYLES[t['style']][0]+f'_sigma{STYLES[t["style"]][1]}' for t in tasks)),
        profiles_completed=len(profiles), provisional_midpoint_failures=len(eligible),
        audited_midpoint_failures=len(dense_failing),
        fixed_cma_16_verified=sum(p['midpoint']['success'] for p in profiles),
        fixed_cma_32_verified=sum(p['outcome']['success'] for p in profiles),
        witness_recheck_evaluations=3*len(witness_checks),
        witness_recheck_source_rays=sum(w['source_rays'] for w in witness_checks),
        generation_evaluations=len(attempts), generation_source_rays=sum(a['result']['rays'] for a in attempts),
        profiling_evaluations=38*len(profiles),
        profiling_source_rays=sum(p['outcome']['source_rays']+p['midpoint']['source_rays']-
                                  p['first']['state']['candidate_rays'] for p in profiles))
    write(root/'summary.json', summary)
    (root/'RESULTS.md').write_text(f'''# Repair tasks from overnight discoveries

Status: {'complete' if summary['complete'] else 'profiling in progress'}.

- Accepted repair tasks: **{len(tasks)}** from **{summary['task_settings']}** task settings.
- Recipes attempted: {len(rows)}; exhausted/rejected: {summary['rejected_recipes']}.
- Shape coverage (corner count → tasks): {summary['corners']}.
- Fixed-CMA profiles completed: {len(profiles)}/{len(tasks)}.
- Verified after 16 candidates: {summary['fixed_cma_16_verified']}.
- Verified after 32 candidates: {summary['fixed_cma_32_verified']}.
- Still failing the coarse midpoint observation: {len(eligible)}.
- Still failing the dense midpoint audit: {len(dense_failing)}.

One highest worst-field-throughput witness per qualifying task setting was selected
before perturbation. Every selected witness passed fresh 5x25 / 7x51 checks and
exact stress agreement at doubled branch capacity. Accepted degraded starts fail
search, audit, and stress requirements with throughput between 20% and 77.5%, and
their stress results agree at doubled branch capacity. Original optical requirements
remain fixed; the repair floor is 80% throughput and root support.

Each setting receives pose, curvature, and shape recipes at normalized scales
0.04 and 0.12. All attempted recipes and rejected perturbations are retained.
The complete task set is retained regardless of baseline repairability. The two
midpoint failure lists are diagnostic subsets, not an unbiased evaluation benchmark.
Every baseline profile uses one numerical seed, 16+16 candidates, all parameters,
initial scale 0.025, radius 0.5, and unchanged continuation.

All tasks are DEVELOPMENT data. Related tasks share a witness and task setting;
counts do not represent independent optical families. No held-out test claim is
made. Witnesses, generation recipes, and provenance are private metadata: model
inputs are the separately allowlisted observations. No Qwen calls or training ran.

Files: tasks.json, observations/, private_witnesses.json, private_generation/,
profiles/, midpoint_coarse_failing_ids.json, midpoint_audit_failing_ids.json.
Generation, witness recheck, and profile costs are separate in summary.json;
original nine-hour discovery/certification cost is reused, not included again.
''')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/playpen/hisham26/overnight_inception_20260915/discovery/repair_ready_designs.json'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    inventory = read(args.source)
    def quality(r): return min(r['outcome'][f]['throughput'] for f in ['audit','stress'])
    sources = [max((r for r in inventory if r['family_id']==family), key=lambda r:(quality(r),r['id']))
               for family in sorted({r['family_id'] for r in inventory})]
    paths = [Path(__file__), Path('nslens/analysis/azr_repair_pilot.py'),
             Path('nslens/workbench/interactive_cma_pilot.py'), Path('nslens/workbench/cma_midpoint_pilot.py')]
    protocol = dict(source=str(args.source), source_hash=digest(inventory), seed=SEED,
        selected_witnesses=[s['id'] for s in sources], styles=STYLES, split='development',
        source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    # JSON round-trip normalizes tuples before comparison on resume.
    import json
    protocol = json.loads(json.dumps(protocol))
    if (root/'protocol.json').exists():
        assert read(root/'protocol.json')==protocol, 'Use a fresh output directory for changed protocol/source'
    write(root/'protocol.json',protocol)
    write(root/'private_witnesses.json',sources)
    for p in paths:
        target=root/'source_snapshot'/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(family_job,(root,i,s)) for i,s in enumerate(sources)]
        tasks=[]
        for future in as_completed(futures):
            tasks.extend(future.result())
            print(f'Accepted {len(tasks)} tasks so far',flush=True)
        tasks.sort(key=lambda t:t['task_id'])
        assert len({digest(dict(config=t['config'],start=t['start'])) for t in tasks})==len(tasks), 'Duplicate starts'
        assert tasks, 'No valid tasks generated'
        write(root/'tasks.json',tasks)
        report(root)
        futures=[pool.submit(profile_job,(root,t)) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%16==0 or i==len(tasks):
                s=report(root)
                print(f'Profiled {s["profiles_completed"]}/{len(tasks)}; midpoint failures {s["provisional_midpoint_failures"]}',flush=True)
    print(report(root),flush=True)


if __name__=='__main__':
    main()
