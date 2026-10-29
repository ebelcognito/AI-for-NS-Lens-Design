"""Inventory dense-audited discoveries and certify the stricter repair quality floor."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from nslens.workbench.interactive_cma_pilot import digest, finalize, read, write


def certify(job):
    root, candidate = job
    path = root/'repair_certification'/f'{candidate["id"]}.json'
    if path.exists():
        return read(path)
    state = dict(genes=candidate['genes'], candidates=0, candidate_rays=0)
    outcome = finalize(dict(config=candidate['config']), state)
    record = dict(**candidate, repair_ready=outcome['success'], outcome=outcome)
    write(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    root = args.root
    rows = read(root/'summary.json')
    candidates = {}
    for row in rows:
        config = {key: row['config'][key] for key in ['codec', 'brief']}
        archive = read(Path(row['config']['out'])/'archive.json')
        for audit in archive['audits']:
            result = audit['result']
            if not result['feasible']:
                continue
            identity = digest(dict(config=config, genes=result['genes']))
            candidate = dict(id=identity, family_id=row['brief_id'], config=config,
                genes=result['genes'], route=result['route'], discovery_quality=result['quality'],
                source_archive=str(Path(row['config']['out'])/'archive.json'),
                source_seed=row['seed'], source_evaluation=audit['coarse_evaluation'])
            candidates.setdefault(identity, candidate)
    write(root/'inventory.json', list(candidates.values()))
    certified = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(certify, (root, c)) for c in candidates.values()]
        for future in as_completed(futures):
            certified.append(future.result())
            write(root/'harvest_status.json', dict(complete=False, completed=len(certified), expected=len(candidates)))
    ready = [c for c in certified if c['repair_ready']]
    write(root/'repair_ready_designs.json', ready)
    stats = dict(complete=True, completed_jobs=len(rows),
        task_settings_completed=len({r['brief_id'] for r in rows}),
        unique_dense_audited_candidates=len(candidates), repair_ready_designs=len(ready),
        repair_ready_task_settings=len({c['family_id'] for c in ready}),
        repair_ready_corner_counts=dict(Counter(c['config']['codec']['corners'] for c in ready)),
        search_source_rays=sum(r['search_source_rays'] for r in rows),
        discovery_audit_source_rays=sum(r['audit_source_rays'] for r in rows),
        certification_source_rays=sum(c['outcome']['source_rays'] for c in certified))
    write(root/'harvest_summary.json', stats)
    write(root/'harvest_status.json', dict(complete=True, completed=len(certified), expected=len(candidates)))
    text = f'''# Overnight inception inventory

- Completed portfolio jobs: {len(rows)}
- Task settings visited: {stats['task_settings_completed']}
- Unique candidates passing discovery's dense audit: {len(candidates)}
- Candidates passing the stricter repair floor: {len(ready)}
- Task settings with repair-ready candidates: {stats['repair_ready_task_settings']}

The repair floor requires at least 80% worst-field folded throughput and root
support, retaining any stricter original requirement. Each certified candidate
passes 5x25 and 7x51 sampling and exact agreement with doubled branch capacity.
These checks share one physics engine. This is 2D circular-face polygon discovery,
not imaging qualification, compaction, model training, or independent physical validation.

Deduplication is exact geometry/configuration identity only. Candidate count is
not a count of independent optical families; task settings can share topology.
No train/test split has been assigned. Preserve source ancestry when splitting.
Discovery and certification source-ray costs are recorded separately in harvest_summary.json.
'''
    (root/'RESULTS.md').write_text(text)
    print(stats, flush=True)


if __name__ == '__main__':
    main()
