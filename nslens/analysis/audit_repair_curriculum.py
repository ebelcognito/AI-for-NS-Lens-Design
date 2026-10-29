"""Audit leakage, provenance, accounting, labels, and deterministic physical replay.

python -m nslens.analysis.audit_repair_curriculum runs/repair_loop_verified_20260910_v3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from nslens.analysis.run_repair_curriculum import digest, metrics, write_json
from nslens.learning.freeform_repair import ImagingSpec, evaluate, features, Verdict


def audit(root: Path, replay_per_seed=3):
    manifest = json.loads((root / "manifest.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    for name, expected in manifest["sources"].items():
        assert digest(root / "source_snapshot" / name) == expected, f"snapshot changed: {name}"
        assert digest(repo / name) == expected, f"runtime differs from experiment: {name}"
    summaries = []
    for seed in manifest["config"]["seeds"]:
        folder = root / f"seed{seed}"
        specs = [ImagingSpec(**s) for s in json.loads((folder / "specs.json").read_text())]
        by_id = {s.family_id: s for s in specs}
        assert len(by_id) == len(specs), "duplicate family ID"
        physics_keys = [tuple(s.context()) for s in specs]
        assert len(set(physics_keys)) == len(physics_keys), "duplicate specification across families"
        counts = defaultdict(Counter)
        accounting = defaultdict(lambda: dict(evaluations=0, launched_rays=0))
        train_initial = {}
        invalid_geometry = 0
        for line in (folder / "evaluations.jsonl").open():
            row = json.loads(line)
            spec = by_id[row["family_id"]]
            assert row["split"] == spec.split
            stage = row["stage"]
            if stage != "certification":
                expected_split = ("test" if stage.startswith("test/") else
                                  "validation" if "validation" in stage else "train")
                assert spec.split == expected_split, (stage, spec.split)
            v = row["verdict"]
            assert np.isfinite(v["residual"]).all()
            assert bool(v["success"]) == bool(v["physical_valid"] and
                                              v["rms"] <= spec.max_rms and
                                              v["worst_field_rms"] <= spec.max_field_rms)
            if v["physical_valid"]:
                assert v["geometry_valid"] and v["route_fraction"] == 1.
                assert v["min_ray_energy"] >= spec.min_ray_energy
                assert v["closure_error"] < 1e-9 and v["unresolved_energy_fraction"] < 1e-5
            if not v["geometry_valid"]:
                invalid_geometry += 1
            counts[(stage, row["task_id"])][row["fidelity"]] += 1
            key = stage + "/" + row["fidelity"]
            accounting[key]["evaluations"] += 1
            accounting[key]["launched_rays"] += v["rays"]
            if spec.split == "train" and row["source"] == "initial":
                train_initial[(stage, row["task_id"])] = row
        assert dict(accounting) == json.loads((folder / "accounting.json").read_text())
        for (stage, _), count in counts.items():
            if stage == "certification":
                expected = dict(coarse=33, dense=1)
            elif stage.startswith("test/"):
                expected = dict(coarse=1 + manifest["config"]["budget"], dense=3, audit=2)
            elif "validation" in stage:
                expected = dict(coarse=1 + manifest["config"]["budget"], dense=3)
            else:
                expected = dict(coarse=10)
            assert dict(count) == expected, (stage, dict(count), expected)
        examples = 0
        for line in (folder / "training_examples.jsonl").open():
            row = json.loads(line)
            assert row["split"] == "train"
            initial = train_initial[(row["stage"], row["task_id"])]
            spec = by_id[row["family_id"]]
            np.testing.assert_allclose(row["features"], features(
                spec, initial["x"], Verdict(**initial["verdict"])), rtol=1e-12, atol=1e-12)
            if row.get("jacobian") is not None:
                np.testing.assert_array_equal(row["response_features"],
                                              np.r_[spec.context(), initial["x"]])
            examples += 1
        selection = json.loads((folder / "selection.json").read_text())
        for selected in selection.values():
            assert digest(folder / selected["checkpoint"]) == selected["sha256"]
        summary = json.loads((folder / "summary.json").read_text())
        for name, expected in summary["test"].items():
            rows = json.loads((folder / f"test_{name}.json").read_text())
            assert metrics(rows, "early_audit") == expected
        tasks = {t["task_id"]: t for t in json.loads((folder / "heldout_tasks.json").read_text())["test"]}
        rows = json.loads((folder / "test_adaptive.json").read_text())
        replayed = []
        rng = np.random.default_rng(seed + 991)
        for i in rng.choice(len(rows), size=min(replay_per_seed, len(rows)), replace=False):
            row = rows[i]
            spec = ImagingSpec(**tasks[row["task_id"]]["spec"])
            measured = evaluate(spec, row["final_x"], "audit")
            np.testing.assert_allclose(measured.residual, row["audit"]["residual"], rtol=1e-10, atol=1e-10)
            assert measured.success == row["audit"]["success"]
            replayed.append(row["task_id"])
        summaries.append(dict(seed=seed, families=len(specs), training_examples=examples,
                              task_groups=len(counts), evaluations=sum(v["evaluations"] for v in accounting.values()),
                              invalid_geometry_attempts=invalid_geometry, replayed=replayed))
    report = dict(passed=True, seeds=summaries,
                  checks=["source snapshots and current runtime match", "whole-family split isolation",
                          "no exact duplicate specifications", "policy features reconstruct without witnesses",
                          "physical gates and acceptance labels agree", "all attempted calls accounted",
                          "candidate allowances identical across curricula", "checkpoint hashes match",
                          "reported metrics recompute exactly", "independent deterministic ray replay"],
                  limitation="Same physics implementation; does not establish agreement with an external tracer.")
    baseline = root / "broyden_baseline"
    if (baseline / "summary.json").exists():
        bsummary = json.loads((baseline / "summary.json").read_text())
        assert digest(baseline / "benchmark_source.py") == bsummary["source_sha256"]
        results = json.loads((baseline / "results.json").read_text())
        assert metrics(results, "audit") == bsummary["metrics"]
        counts = defaultdict(Counter)
        for line in (baseline / "evaluations.jsonl").open():
            record = json.loads(line)
            counts[record["task_id"]][record["fidelity"]] += 1
        assert len(counts) == len(results)
        assert all(dict(v) == dict(coarse=9, dense=1, audit=1) for v in counts.values())
        report["broyden"] = dict(tasks=len(results), budget_and_metrics_verified=True)
    write_json(root / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--replay-per-seed", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.replay_per_seed), indent=2))


if __name__ == "__main__":
    main()
