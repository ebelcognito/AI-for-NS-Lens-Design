"""Paired, seed-block uncertainty for a predeclared discovery benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .run import write


def paired_comparison(rows, candidate, control, *, metric="audited_success"):
    left = {r["unit"]:r for r in rows if r["arm"] == candidate}
    right = {r["unit"]:r for r in rows if r["arm"] == control}
    if not left or set(left) != set(right):
        raise ValueError("comparison requires complete paired units")
    differences = {}
    for unit, row in left.items():
        other = right[unit]
        for key in ("codec", "brief", "search_calls", "popsize", "audit_rays", "audit_limit"):
            if row["config"][key] != other["config"][key]:
                raise ValueError(f"unmatched {key}: {unit}")
        if row["reserved_source_rays"] != other["reserved_source_rays"]:
            raise ValueError("unmatched reserved physical budgets")
        differences.setdefault(row["seed"], []).append(float(row[metric])-float(other[metric]))
    sizes = {len(v) for v in differences.values()}
    if len(sizes) != 1:
        raise ValueError("incomplete seed blocks")
    blocks = np.asarray([np.mean(differences[seed]) for seed in sorted(differences)])
    rng = np.random.default_rng(20260911)
    samples = blocks[rng.integers(len(blocks), size=(20000,len(blocks)))].mean(axis=1)
    wins = int(np.sum(blocks > 0))
    losses = int(np.sum(blocks < 0))
    n = wins+losses
    p = sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.
    return dict(candidate=candidate, control=control, metric=metric, paired_units=len(left),
                seed_blocks=len(blocks), mean_difference=float(blocks.mean()),
                seed_block_bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),
                seed_blocks_won=wins, seed_blocks_lost=losses, seed_blocks_tied=len(blocks)-n,
                one_sided_seed_sign_p=p,
                differences_by_seed={str(seed):float(np.mean(v)) for seed,v in sorted(differences.items())},
                scope="Variation over fresh search seeds on the fixed declared briefs and corner families.")


def summarize(root, candidate):
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["stage"] == "confirmation" and protocol.get("candidate") != candidate:
        raise ValueError("candidate differs from predeclared confirmation policy")
    rows = json.loads((root/"summary.json").read_text())
    expected = {(r["unit"],r["arm"]) for r in protocol["jobs"]}
    actual = [(r["unit"],r["arm"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("missing or duplicate benchmark results")
    arms = sorted({r["arm"] for r in rows})
    totals = {}
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        totals[arm] = dict(runs=len(group), successes=sum(r["audited_success"] for r in group),
                          audited_designs=sum(r["audited_designs"] for r in group),
                          audited_routes=sum(r["audited_routes"] for r in group),
                          mean_best_audited_quality=float(np.mean([r["best_audited_quality"] for r in group])),
                          search_source_rays=sum(r["search_source_rays"] for r in group),
                          audit_source_rays=sum(r["audit_source_rays"] for r in group),
                          reserved_source_rays=sum(r["reserved_source_rays"] for r in group),
                          total_worker_seconds=sum(r["wall_seconds"] for r in group))
    comparisons = [paired_comparison(rows,candidate,arm) for arm in arms if arm != candidate]
    primary = next(c for c in comparisons if c["control"] == "naive")
    accepted = (protocol["stage"] == "confirmation" and primary["seed_block_bootstrap_95"][0] > 0
                and primary["one_sided_seed_sign_p"] < .05)
    result = dict(stage=protocol["stage"], candidate=candidate, totals=totals,
                  comparisons=comparisons, improvement_over_naive_confirmed=accepted,
                  acceptance="Confirmation only: paired 95% seed-block interval above zero and one-sided seed sign p < .05.",
                  caveat="Secondary comparisons are exploratory; no inference to all optical specifications.")
    write(root/"comparison.json", result)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--candidate",required=True)
    a=p.parse_args()
    summarize(a.run,a.candidate)


if __name__ == "__main__":
    main()
