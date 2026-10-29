"""Summarize paired complete-search units from a fold EvoTF ablation artifact."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "filled", "coverage", "qd_score", "best_fitness", "route_families", "engine_calls",
    "deep_fold_route_families", "calls_to_first_valid", "calls_to_first_improving",
    "invalid_rate", "clipped_candidate_rate", "route_persistence",
    "optimizer_seconds", "compile_seconds", "wall_seconds",
)
LOWER_IS_BETTER = {
    "calls_to_first_valid", "calls_to_first_improving", "invalid_rate",
    "clipped_candidate_rate", "engine_calls", "optimizer_seconds",
    "compile_seconds", "wall_seconds",
}


def _finite(values):
    return np.asarray([float(value) for value in values
                       if value is not None and np.isfinite(float(value))])


def _describe(values):
    values = _finite(values)
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {
        "n": len(values), "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def _paired_interval(differences, *, lower_is_better=False, seed=0, samples=10000):
    differences = _finite(differences)
    if not len(differences):
        return {
            "n": 0, "direction": "lower" if lower_is_better else "higher",
            "median_arm_minus_cma": None, "median_improvement": None,
            "improvement_bootstrap_95": [None, None],
        }
    improvements = -differences if lower_is_better else differences
    rng = np.random.default_rng(seed)
    draws = rng.choice(improvements, size=(samples, len(improvements)), replace=True)
    medians = np.median(draws, axis=1)
    return {
        "n": len(differences),
        "direction": "lower" if lower_is_better else "higher",
        "median_arm_minus_cma": float(np.median(differences)),
        "median_improvement": float(np.median(improvements)),
        "improvement_bootstrap_95": [
            float(value) for value in np.percentile(medians, [2.5, 97.5])],
        "wins": int(np.count_nonzero(improvements > 0)),
        "ties": int(np.count_nonzero(improvements == 0)),
        "losses": int(np.count_nonzero(improvements < 0)),
    }


def summarize(payload: dict) -> dict:
    by_arm = defaultdict(list)
    by_seed = defaultdict(dict)
    for record in payload["records"]:
        by_arm[record["arm"]].append(record["metrics"])
        by_seed[int(record["seed"])][record["arm"]] = record["metrics"]
    summary = {
        arm: {metric: _describe([row.get(metric) for row in rows])
              for metric in METRICS}
        for arm, rows in by_arm.items()
    }
    paired = {}
    for arm in by_arm:
        if arm == "cma":
            continue
        metric_rows = {}
        for metric in METRICS:
            differences = []
            for seed, arms in by_seed.items():
                if "cma" not in arms or arm not in arms:
                    continue
                left, right = arms[arm].get(metric), arms["cma"].get(metric)
                if left is not None and right is not None:
                    differences.append(float(left) - float(right))
            metric_rows[metric] = _paired_interval(
                differences, lower_is_better=metric in LOWER_IS_BETTER)
        paired[f"{arm}_minus_cma"] = metric_rows
    return {
        "schema_version": "nslens.fold_evotf_ablation_summary.v1",
        "source_schema": payload.get("schema_version"),
        "paired_unit": "complete search task/seed",
        "arms": summary,
        "paired_differences": paired,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.results.read_text())
    if payload.get("schema_version") != "nslens.fold_evotf_ablation.v1":
        parser.error("input is not a fold EvoTF ablation artifact")
    summary = summarize(payload)
    output = args.output or args.results.with_name("summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "metric", "n", "median", "q25", "q75"))
        for arm, metrics in summary["arms"].items():
            for metric, values in metrics.items():
                writer.writerow((arm, metric, values["n"], values["median"],
                                 values["q25"], values["q75"]))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
