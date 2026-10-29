"""Stronger local baseline: finite-difference initialization plus Broyden updates.

Uses the locked test tasks and exactly eight candidate calls, with no training data.
It complements the original teacher's conservative four-probe/four-line-search step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from nslens.analysis.run_repair_curriculum import digest, metrics, paired_summary, write_json
from nslens.learning.freeform_repair import BOUND, ImagingSpec, Recorder, bounded_action, merit


def broyden_solve(spec, start, rec, budget=8):
    if budget < 5:
        raise ValueError("four Jacobian probes and at least one repair attempt required")
    x = np.asarray(start, dtype=float)
    initial = current = rec(x, source="initial")
    cols = []
    for j in range(4):
        h = .001 if x[j] < BOUND - .001 else -.001
        probe = x.copy()
        probe[j] += h
        v = rec(probe, source="jacobian", parent=x.tolist())
        cols.append((np.asarray(v.residual) - initial.residual) / h)
    jac = np.array(cols).T
    scale = 1.
    for _ in range(budget - 4):
        delta = bounded_action(np.linalg.solve(jac.T @ jac + .01 * np.eye(4),
                                               -jac.T @ np.asarray(current.residual)))
        candidate = np.clip(x + scale * delta, -BOUND, BOUND)
        v = rec(candidate, source="broyden", parent=x.tolist())
        step = candidate - x
        if current.physical_valid and v.physical_valid and step @ step > 1e-14:
            difference = np.asarray(v.residual) - current.residual
            jac += np.outer(difference - jac @ step, step) / (step @ step)
        if merit(v) < merit(current):
            x, current, scale = candidate, v, 1.
        else:
            scale = max(.125, scale * .5)
    # The initial derivative probes are real candidates too.
    best = min(rec.records, key=lambda row: (not row["verdict"]["physical_valid"],
                                            row["verdict"]["rms"]))
    return np.asarray(best["x"]), initial


def worker(task):
    spec = ImagingSpec(**task["spec"])
    rec = Recorder(spec, task["task_id"], "test/broyden")
    end, initial = broyden_solve(spec, task["start"], rec)
    coarse_calls = len(rec.records)
    dense = rec(end, "dense", source="final_dense")
    audit = rec(end, "audit", source="final_audit")
    assert coarse_calls == 9
    return dict(task_id=task["task_id"], family_id=spec.family_id,
                initial_success=initial.success, initial_rms=initial.rms,
                final_x=end.tolist(), dense=asdict(dense), audit=asdict(audit),
                coarse_calls=coarse_calls), rec.records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    directory = args.root / "broyden_baseline"
    directory.mkdir()
    shutil.copy2(__file__, directory / "benchmark_source.py")
    tasks, learned = [], []
    for seed in manifest["config"]["seeds"]:
        tasks.extend(json.loads((args.root / f"seed{seed}/heldout_tasks.json").read_text())["test"])
        learned.extend(json.loads((args.root / f"seed{seed}/test_adaptive.json").read_text()))
    rows = []
    with (directory / "evaluations.jsonl").open("w") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, records in pool.map(worker, tasks):
                rows.append(row)
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
    write_json(directory / "results.json", rows)
    result = dict(source_sha256=digest(__file__), candidate_calls=8, initial_calls=1,
                  metrics=metrics(rows, "audit"),
                  learned_eight_calls_vs_broyden=paired_summary(learned, rows))
    write_json(directory / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
