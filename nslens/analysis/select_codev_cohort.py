"""Select a small paired cohort for direct CODE V validation.

This emits prescriptions and expected audit outcomes only.  It deliberately does not emit
bridge jobs, commands, executable paths, or CODE V sequences; transport/schema ownership
remains with the separate bridge implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..rl import env_config

SCHEMA = "nslens.codev_validation_cohort.v1"
CONTROL_LABELS = ("field-dependent", "spurious", "unstable")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _normalized_design(record: dict) -> np.ndarray:
    env = env_config.get("fold")
    x = np.array([record["design"][name] for name in env.param_names], dtype=float)
    return (x - np.asarray(env.problem.lo, float)) / (
        np.asarray(env.problem.hi, float) - np.asarray(env.problem.lo, float))


def _summary(record: dict) -> dict:
    hardened = record.get("source_hardened_verdict") or {}
    return {
        "design_id": record["design_id"], "prescription": record["design"],
        "environment": {
            "source_center": record["design"]["ray_y"],
            "source_aim_deg": record["design"]["ang"],
            "refractive_index": record["design"]["n"],
            "fields_deg": [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            "beam_width": 8.0, "detector_x": 120.0,
        },
        "expected_interaction_sequence": record["historical_sequence"],
        "expected_audit_label": record["audit_label"],
        "internal_metrics": {
            "hardened_fitness": hardened.get("fitness"),
            "historical_path_delivered_energy_share_mean":
                record.get("historical_full_delivered_energy_share_mean"),
            "dense_dominance_fraction": record.get("historical_dense_dominance_fraction"),
            "field_persistence": record.get("global_topology_field_persistence"),
            "perturbation_persistence": record.get(
                "global_topology_perturbation_persistence"),
        },
    }


def run(audit_path: Path, environment_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    records = _load_jsonl(audit_path)
    by_id = {record["design_id"]: record for record in records}
    environments = json.loads(environment_path.read_text())["environments"]
    positives = [by_id[spec["seed_design_id"]] for spec in environments
                 if spec.get("status") == "accepted"]
    negatives = [record for record in records if record.get("audit_label") in CONTROL_LABELS]
    used = set()
    cohort = []
    pair_manifest = []
    for index, positive in enumerate(positives):
        desired = CONTROL_LABELS[index % len(CONTROL_LABELS)]
        pool = [record for record in negatives
                if record["design_id"] not in used
                and record["historical_sequence"] == positive["historical_sequence"]]
        if not pool:
            raise ValueError(f"no matched negative control for {positive['design_id']}")
        pvec = _normalized_design(positive)
        # Prefer the rotating control label only when it costs no more than 0.05 in
        # normalized nine-gene distance. Geometric matching is more valuable than a
        # perfectly balanced label count in a paired external-engine check.
        control = min(pool, key=lambda record: float(
            np.linalg.norm(_normalized_design(record) - pvec))
            + (0.0 if record["audit_label"] == desired else 0.05))
        used.add(control["design_id"])
        pair_id = f"codev-pair-{index:02d}"
        pos = {"schema_version": SCHEMA, "pair_id": pair_id,
               "role": "intentional-anchor", **_summary(positive)}
        neg = {"schema_version": SCHEMA, "pair_id": pair_id,
               "role": "nearby-negative-control", **_summary(control)}
        cohort.extend((pos, neg))
        pair_manifest.append({
            "pair_id": pair_id, "positive_design_id": positive["design_id"],
            "control_design_id": control["design_id"],
            "control_label": control["audit_label"],
            "normalized_parameter_distance": float(
                np.linalg.norm(_normalized_design(control) - pvec)),
            "matched_sequence": positive["historical_sequence"],
        })

    cohort_path = output / "cohort.jsonl"
    temporary = cohort_path.with_name(cohort_path.name + ".tmp")
    temporary.write_text("".join(_canonical_json(record) + "\n" for record in cohort))
    os.replace(temporary, cohort_path)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "paired direct-CODE-V validation of the interaction audit",
        "not_bridge_jobs": True,
        "source_audit": str(audit_path), "source_audit_sha256": _sha256_file(audit_path),
        "source_environments": str(environment_path),
        "source_environments_sha256": _sha256_file(environment_path),
        "candidate_count": len(cohort), "pair_count": len(pair_manifest),
        "pairs": pair_manifest,
        "output": {"path": str(cohort_path), "sha256": _sha256_file(cohort_path)},
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    readme = output / "README.md"
    temporary = readme.with_name(readme.name + ".tmp")
    temporary.write_text(
        "# Direct CODE V validation cohort\n\n"
        "Eight verified stable anchors are paired with nearby candidates implementing the "
        "same claimed sequence but labelled field-dependent, spurious, or unstable. This "
        "is prescription data, not an executable bridge job. The bridge layer should map "
        "these records into its own allowlisted declarative schema.\n")
    os.replace(temporary, readme)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-records", required=True)
    parser.add_argument("--environment-specs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run(Path(args.audit_records).resolve(),
                   Path(args.environment_specs).resolve(), Path(args.out).resolve())
    print(json.dumps({"candidate_count": manifest["candidate_count"],
                      "pair_count": manifest["pair_count"],
                      "output": str(Path(args.out).resolve())}, indent=2))


if __name__ == "__main__":
    main()
