"""Verify and summarize a fixed equal-ray-engine-call benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "nslens.fixed_call_budget_summary.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_metrics(run: dict) -> dict:
    trace = Path(run["trace"])
    if _sha256(trace) != run["trace_sha256"]:
        raise ValueError(f"trace checksum mismatch: {trace}")
    rows = [json.loads(line) for line in trace.open() if line.strip()]
    calls = sum(int(row["engine_calls"]) for row in rows)
    if calls != int(run["engine_calls_used"]):
        raise ValueError(f"engine-call mismatch in {trace}: {calls}")
    status = Counter(row["status"] for row in rows)
    rejections = Counter((row.get("rejection") or {}).get("stage", "unclassified")
                         for row in rows if row["status"] == "invalid")
    proposal_kinds = Counter(row["proposal_kind"] for row in rows)
    model_candidates = sum(int(row["model_candidates_considered"]) for row in rows)
    return {
        **{key: run[key] for key in (
            "environment_id", "algorithm", "seed", "engine_call_budget",
            "engine_calls_used", "proposals_evaluated", "cells", "best_fitness",
            "wall_seconds")},
        "unused_engine_calls": int(run["engine_call_budget"])
        - int(run["engine_calls_used"]),
        "valid": len(rows) - status.get("invalid", 0),
        "valid_fraction": (len(rows) - status.get("invalid", 0)) / len(rows),
        "status_counts": dict(sorted(status.items())),
        "rejection_stage_counts": dict(sorted(rejections.items())),
        "proposal_kind_counts": dict(sorted(proposal_kinds.items())),
        "model_candidates_considered": model_candidates,
    }


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def summarize(manifest: dict) -> dict:
    runs = [_run_metrics(run) for run in manifest["runs"]]
    runs.sort(key=lambda row: (row["environment_id"], row["seed"], row["algorithm"]))
    aggregate = {}
    for algorithm in sorted({row["algorithm"] for row in runs}):
        rows = [row for row in runs if row["algorithm"] == algorithm]
        aggregate[algorithm] = {
            "runs": len(rows), "mean_best_fitness": _mean(rows, "best_fitness"),
            "mean_valid_fraction": _mean(rows, "valid_fraction"),
            "mean_cells": _mean(rows, "cells"),
            "mean_engine_calls_used": _mean(rows, "engine_calls_used"),
            "mean_proposals_evaluated": _mean(rows, "proposals_evaluated"),
            "mean_wall_seconds": _mean(rows, "wall_seconds"),
        }
    keyed = defaultdict(dict)
    for row in runs:
        keyed[(row["environment_id"], row["seed"])][row["algorithm"]] = row
    comparisons = []
    totals = {"learned_vs_cma": Counter(), "learned_vs_random": Counter()}
    deltas = {"learned_vs_cma": [], "learned_vs_random": []}
    for (environment_id, seed), algorithms in sorted(keyed.items()):
        if set(algorithms) != {"cma", "random", "learned"}:
            raise ValueError(f"incomplete algorithm set for {environment_id} seed {seed}")
        row = {"environment_id": environment_id, "seed": seed}
        learned = float(algorithms["learned"]["best_fitness"])
        for opponent in ("cma", "random"):
            delta = learned - float(algorithms[opponent]["best_fitness"])
            name = f"learned_vs_{opponent}"
            winner = "learned" if delta > 1e-12 else opponent if delta < -1e-12 else "tie"
            totals[name][winner] += 1
            deltas[name].append(delta)
            row[name] = {"delta": delta, "winner": winner}
        row["best_fitness"] = {algorithm: algorithms[algorithm]["best_fitness"]
                               for algorithm in ("cma", "random", "learned")}
        comparisons.append(row)
    comparison_summary = {}
    for name in totals:
        comparison_summary[name] = {
            "wins": dict(sorted(totals[name].items())),
            "mean_fitness_delta": statistics.mean(deltas[name]),
            "median_fitness_delta": statistics.median(deltas[name]),
        }
    return {"schema_version": SCHEMA, "runs": runs,
            "algorithm_aggregate": aggregate,
            "paired_comparisons": comparisons,
            "paired_summary": comparison_summary,
            "checks": {
                "all_runs_within_budget": all(
                    0 <= row["unused_engine_calls"] < 9 for row in runs),
                "all_pairs_complete": len(comparisons) * 3 == len(runs),
                "trace_call_sums_match_manifests": True,
            }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    payload = summarize(json.loads(manifest_path.read_text()))
    payload["benchmark_manifest"] = str(manifest_path)
    payload["benchmark_manifest_sha256"] = _sha256(manifest_path)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"out": str(output), "checks": payload["checks"],
                      "aggregate": payload["algorithm_aggregate"],
                      "paired": payload["paired_summary"]}, indent=2))


if __name__ == "__main__":
    main()
