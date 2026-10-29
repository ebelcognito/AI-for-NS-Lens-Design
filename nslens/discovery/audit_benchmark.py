"""Replay frozen confirmation audits and stress selected examples at higher density."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from .folded import RoutingBrief, evaluate
from .polygon import PolygonCodec
from .run import write


def check(item):
    summary, stored, stress = item
    brief = RoutingBrief(**summary['config']['brief'])
    fields = sorted(brief.fields_deg)
    if stress:
        brief = replace(brief, rays_per_field=101,
                        fields_deg=tuple(np.linspace(fields[0], fields[-1], 9)))
    else:
        brief = replace(brief, rays_per_field=summary['config']['audit_rays'],
                        fields_deg=tuple(sorted(set(fields+[(a+b)/2 for a,b in zip(fields, fields[1:])]))))
    actual = evaluate(PolygonCodec(**summary['config']['codec']), stored['genes'], brief,
                      geometry_samples=4096 if stress else 1024)
    # JSON roundtrip normalizes tuple/list differences. Require exact replay of all
    # recorded physics, not just matching pass/fail or rounded summary metrics.
    equal = json.loads(json.dumps(actual)) == json.loads(json.dumps(stored))
    return dict(unit=summary['unit'], stress=stress, exact_replay=equal if not stress else None,
                original_feasible=stored['feasible'], feasible=actual['feasible'],
                source_rays=actual['source_rays'], result=actual if stress else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    protocol = json.loads((a.run/'protocol.json').read_text())
    candidate = protocol['candidate']
    sources_match = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                     for path,digest in protocol['sources'].items()}
    if not all(sources_match.values()):
        raise RuntimeError('benchmark sources changed; use the saved source snapshot for replay')
    summaries = json.loads((a.run/'summary.json').read_text())
    budget_checks = []
    for summary in summaries:
        folder = Path(summary['config']['out'])
        rows = [json.loads(line) for line in (folder/'candidates.jsonl').read_text().splitlines()]
        archive = json.loads((folder/'archive.json').read_text())
        valid = (len(rows) == summary['proposals']
                 and sum(r['source_rays'] for r in rows) == summary['search_source_rays']
                 and sum(r['source_rays'] > 0 for r in rows) == summary['search_calls']
                 and summary['search_calls'] <= summary['config']['search_calls']
                 and len(archive['audits']) <= summary['config']['audit_limit']
                 and sum(r['result']['source_rays'] for r in archive['audits']) == summary['audit_source_rays']
                 and summary['search_source_rays']+summary['audit_source_rays'] <= summary['reserved_source_rays'])
        budget_checks.append(dict(unit=summary['unit'],arm=summary['arm'],valid=valid))
    if not all(r['valid'] for r in budget_checks):
        raise RuntimeError('saved candidate traces do not support declared budgets')
    jobs, examples = [], {}
    for summary in summaries:
        if summary['arm'] != candidate:
            continue
        archive = json.loads((Path(summary['config']['out'])/'archive.json').read_text())
        for audit in archive['audits']:
            stored = audit['result']
            jobs.append((summary, stored, False))
            key = (summary['brief_id'],summary['corners'])
            if stored['feasible'] and (key not in examples or stored['quality'] > examples[key][1]['quality']):
                examples[key] = (summary, stored, True)
    jobs.extend(examples.values())
    with get_context('spawn').Pool(a.workers) as pool:
        results = pool.map(check, jobs)
    replay = [r for r in results if not r['stress']]
    stress = [r for r in results if r['stress']]
    result = dict(candidate=candidate, source_hashes_match=sources_match,
                  run_budget_logs_checked=len(budget_checks), all_budget_logs_match=True,
                  replayed_audits=len(replay), exact_replays=sum(r['exact_replay'] for r in replay),
                  additional_verification_source_rays=sum(r['source_rays'] for r in results),
                  stress_examples=stress, replay_mismatches=[r for r in replay if not r['exact_replay']],
                  caveat='Stress examples are selected per brief/corner after the experiment; not used to estimate search success. Same optical engine, no independent-engine certification.')
    write(a.run/'replay_and_stress.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes_match','stress_examples')},indent=2))
    print('Selected stress passes:',sum(r['feasible'] for r in stress),'/',len(stress))
    if result['replay_mismatches']:
        raise RuntimeError('stored audit did not replay exactly')


if __name__ == '__main__':
    main()
