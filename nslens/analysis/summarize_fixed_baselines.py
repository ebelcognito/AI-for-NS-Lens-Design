"""Summarize equal-budget fixed-environment baseline traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "nslens.fixed_baseline_comparison.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _engine_calls(row: Mapping[str, Any]) -> int:
    """Exact for new traces; reconstructable for pre-counter fixed traces."""
    if row.get("engine_calls") is not None:
        return int(row["engine_calls"])
    if row["status"] != "invalid":
        return 9
    rejection = row.get("rejection") or {}
    stage = rejection.get("stage")
    reason = str(rejection.get("reason", ""))
    if stage == "invalid_geometry":
        return 0
    if stage in {"no_light", "beam_gate", "exit_gate"}:
        return 1
    if stage in {"intended_root_support", "intended_not_dominant"}:
        return 1 if "field 0:" in reason else 9
    if stage in {"beam_gate_all_fields", "exit_gate_all_fields",
                 "intended_tir_margin", "intended_low_energy"}:
        return 9
    raise ValueError(f"cannot reconstruct engine calls for rejection {rejection}")


def _best_at_engine_budget(run: Mapping[str, Any], budget: int) -> dict:
    used = evaluations = 0
    best = None
    with Path(run["trace"]).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cost = _engine_calls(row)
            if used + cost > budget:
                break
            used += cost
            evaluations += 1
            if row["status"] != "invalid":
                value = float(row["fitness"])
                best = value if best is None else max(best, value)
    return {"best_fitness": best, "evaluations": evaluations,
            "engine_calls_used": used}


def _trace_metrics(run: Mapping[str, Any]) -> dict:
    trace_path = Path(run["trace"]).resolve()
    expected = run.get("trace_sha256")
    if expected and _sha256(trace_path) != expected:
        raise ValueError(f"trace checksum mismatch: {trace_path}")
    total = valid = updates = engine_calls = 0
    first_valid = first_update = None
    best = None
    best_sum = 0.0
    rejections = Counter()
    with trace_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            engine_calls += _engine_calls(row)
            if row["status"] == "invalid":
                rejection = row.get("rejection") or {}
                rejections[str(rejection.get("stage", "unclassified"))] += 1
            else:
                valid += 1
                if first_valid is None:
                    first_valid = int(row["eval"])
                value = float(row["fitness"])
                best = value if best is None else max(best, value)
                if row["status"] in {"new", "improved"}:
                    updates += 1
                    if first_update is None:
                        first_update = int(row["eval"])
            best_sum += 0.0 if best is None else best
    if total != int(run["evals"]):
        raise ValueError(f"trace has {total} rows, expected {run['evals']}: {trace_path}")
    final_best = run.get("best_fitness")
    if best is not None and final_best is not None and abs(best - float(final_best)) > 2e-6:
        raise ValueError(f"trace/archive best mismatch: {trace_path}")
    return {
        "evaluations": total,
        "engine_calls": engine_calls,
        "valid": valid,
        "valid_fraction": valid / total,
        "archive_updates": updates,
        "archive_update_fraction": updates / total,
        "first_valid_eval": first_valid,
        "first_archive_update_eval": first_update,
        "best_fitness": final_best,
        "normalized_best_so_far_auc": (
            best_sum / (total * float(final_best))
            if final_best is not None and float(final_best) > 0 else None),
        "rejection_stage_counts": dict(sorted(rejections.items())),
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize(manifest: Mapping[str, Any]) -> dict:
    runs = []
    for source in manifest.get("runs", []):
        runs.append({
            "environment_id": source["environment_id"],
            "algorithm": source["algorithm"], "seed": int(source["seed"]),
            "cells": int(source["cells"]),
            **_trace_metrics(source),
        })
    runs.sort(key=lambda row: (row["environment_id"], row["seed"], row["algorithm"]))

    by_algorithm = defaultdict(list)
    for row in runs:
        by_algorithm[row["algorithm"]].append(row)
    aggregate = {}
    for algorithm, rows in sorted(by_algorithm.items()):
        aggregate[algorithm] = {
            "runs": len(rows),
            "mean_cells": _mean(rows, "cells"),
            "mean_best_fitness": _mean(rows, "best_fitness"),
            "mean_valid_fraction": _mean(rows, "valid_fraction"),
            "mean_archive_update_fraction": _mean(rows, "archive_update_fraction"),
            "mean_normalized_best_so_far_auc": _mean(
                rows, "normalized_best_so_far_auc"),
        }

    keyed = defaultdict(dict)
    for row in runs:
        keyed[(row["environment_id"], row["seed"])][row["algorithm"]] = row
    pairs = []
    winners = Counter()
    call_winners = Counter()
    source_by_key = {(row["environment_id"], int(row["seed"]), row["algorithm"]): row
                     for row in manifest.get("runs", [])}
    for (environment_id, seed), algorithms in sorted(keyed.items()):
        if set(algorithms) != {"cma", "random"}:
            continue
        cma, random = algorithms["cma"], algorithms["random"]
        cma_best, random_best = cma["best_fitness"], random["best_fitness"]
        if cma_best is None and random_best is None:
            winner = "tie"
        elif random_best is None or (cma_best is not None and cma_best > random_best + 1e-12):
            winner = "cma"
        elif cma_best is None or random_best > cma_best + 1e-12:
            winner = "random"
        else:
            winner = "tie"
        winners[winner] += 1
        call_budget = min(cma["engine_calls"], random["engine_calls"])
        cma_at_calls = _best_at_engine_budget(
            source_by_key[(environment_id, seed, "cma")], call_budget)
        random_at_calls = _best_at_engine_budget(
            source_by_key[(environment_id, seed, "random")], call_budget)
        if cma_at_calls["best_fitness"] > random_at_calls["best_fitness"] + 1e-12:
            call_winner = "cma"
        elif random_at_calls["best_fitness"] > cma_at_calls["best_fitness"] + 1e-12:
            call_winner = "random"
        else:
            call_winner = "tie"
        call_winners[call_winner] += 1
        pairs.append({
            "environment_id": environment_id, "seed": seed, "winner": winner,
            "cma_best": cma_best, "random_best": random_best,
            "best_delta_cma_minus_random": (
                None if cma_best is None or random_best is None
                else float(cma_best) - float(random_best)),
            "cell_delta_cma_minus_random": cma["cells"] - random["cells"],
            "valid_fraction_delta_cma_minus_random": (
                cma["valid_fraction"] - random["valid_fraction"]),
            "auc_delta_cma_minus_random": (
                None if cma["normalized_best_so_far_auc"] is None
                or random["normalized_best_so_far_auc"] is None
                else cma["normalized_best_so_far_auc"]
                - random["normalized_best_so_far_auc"]),
            "equal_engine_calls": {
                "budget": call_budget, "winner": call_winner,
                "cma": cma_at_calls, "random": random_at_calls,
            },
        })
    return {
        "schema_version": SCHEMA,
        "runs": runs,
        "algorithm_aggregate": aggregate,
        "paired_comparisons": pairs,
        "best_fitness_wins": {key: winners.get(key, 0)
                              for key in ("cma", "random", "tie")},
        "equal_engine_call_best_wins": {
            key: call_winners.get(key, 0) for key in ("cma", "random", "tie")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    payload = summarize(json.loads(manifest_path.read_text()))
    payload["baseline_manifest"] = str(manifest_path)
    payload["baseline_manifest_sha256"] = _sha256(manifest_path)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "out": str(output),
        "best_fitness_wins": payload["best_fitness_wins"],
        "algorithm_aggregate": payload["algorithm_aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
