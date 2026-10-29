"""Dense-audit fixed-environment baseline archives and admit strict positives.

This is the handoff between fresh equal-budget search and dataset construction.  It
deduplicates archive elites within each environment, runs the interaction audit under
that environment's fixed source/material conditions, and writes a deliberately strict
positive partition.  Archive admission alone is never treated as a training label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from . import interaction_audit

SCHEMA = "nslens.fixed_baseline_audit.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _load_specs(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    specs = {
        str(row["environment_id"]): row
        for row in payload.get("environments", [])
        if row.get("status") == "accepted"
    }
    if not specs:
        raise ValueError(f"no accepted environments in {path}")
    return specs


def _candidate_key(environment_id: str, genotype: list[float]) -> str:
    body = _canonical_json({
        "environment_id": environment_id,
        "genotype": [float(value) for value in genotype],
    })
    return hashlib.sha256(body.encode()).hexdigest()


def _collect_candidates(manifest: Mapping[str, Any], specs: Mapping[str, dict]
                        ) -> dict[str, list[dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for run in manifest.get("runs", []):
        environment_id = str(run["environment_id"])
        if environment_id not in specs:
            raise ValueError(f"baseline references unknown environment {environment_id!r}")
        archive_path = Path(run["archive"]).resolve()
        expected_hash = run.get("archive_sha256")
        if expected_hash and _sha256(archive_path) != expected_hash:
            raise ValueError(f"archive checksum mismatch: {archive_path}")
        archive = json.loads(archive_path.read_text())
        param_names = list(archive["meta"]["param_names"])
        if param_names != ["cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b"]:
            raise ValueError(f"unexpected parameters in {archive_path}: {param_names}")
        for cell in archive.get("cells", []):
            genotype = [float(value) for value in cell["genotype"]]
            digest = _candidate_key(environment_id, genotype)
            source = {
                "algorithm": run["algorithm"], "seed": int(run["seed"]),
                "evals": int(run.get("evals", run.get("proposals_evaluated", 0))),
                "archive_cell": {"i0": cell.get("i0"), "i1": cell.get("i1")},
                "fitness": float(cell["fitness"]),
            }
            existing = grouped[environment_id].get(digest)
            if existing is None:
                grouped[environment_id][digest] = {
                    "design_id": f"fixed:{environment_id}:{digest[:20]}",
                    "split": "candidate",
                    "design": dict(zip(param_names, genotype)),
                    "historical_sequence": specs[environment_id]["intended_sequence"],
                    "hardened_verdict": {
                        "training_eligible": True,
                        "fitness": float(cell["fitness"]),
                        "full_signature": specs[environment_id]["intended_sequence"],
                    },
                    "provenance": {"sources": [source]},
                }
            else:
                existing["provenance"]["sources"].append(source)
                existing["hardened_verdict"]["fitness"] = max(
                    existing["hardened_verdict"]["fitness"], float(cell["fitness"]))
    return {
        environment_id: sorted(rows.values(), key=lambda row: row["design_id"])
        for environment_id, rows in grouped.items()
    }


def _strict_rejection_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons = []
    if record.get("audit_label") != "intentional":
        reasons.append(f"audit_label={record.get('audit_label')}")
    if float(record.get("historical_dense_dominance_fraction", 0.0)) < 1.0 - 1e-12:
        reasons.append("intended route is not dominant at every dense field")
    if float(record.get("global_topology_field_persistence", 0.0)) < 1.0 - 1e-12:
        reasons.append("dominant topology changes across dense fields")
    if float(record.get("global_topology_perturbation_persistence", 0.0)) < 0.80:
        reasons.append("dominant topology is not perturbation-stable")
    if float(record.get("historical_full_delivered_energy_share_mean", 0.0)) < 0.05:
        reasons.append("mean intended delivered-energy share is below 0.05")
    for contact in record.get("contacts", []):
        label = f"contact[{contact.get('index')}]"
        if float(contact.get("minimum_root_support", 0.0)) < 0.80:
            reasons.append(f"{label} lacks 0.80 root support at a dense field")
        if float(contact.get("perturbation_persistence", 0.0)) < 0.80:
            reasons.append(f"{label} lacks 0.80 perturbation persistence")
        critical = contact.get("tir_critical_margin_min_deg")
        grazing = contact.get("tir_grazing_margin_min_deg")
        if critical is not None and float(critical) < 0.50:
            reasons.append(f"{label} has a TIR critical margin below 0.50 degrees")
        if grazing is not None and float(grazing) < 0.50:
            reasons.append(f"{label} has a grazing margin below 0.50 degrees")
    return reasons


def run(args) -> dict:
    baseline_manifest_path = Path(args.baseline_manifest).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    baseline = json.loads(baseline_manifest_path.read_text())
    specs_path = Path(args.environment_specs or baseline["environment_specs"]).resolve()
    specs = _load_specs(specs_path)
    candidates = _collect_candidates(baseline, specs)
    output.mkdir(parents=True)

    all_records = []
    audit_manifests = []
    started = time.time()
    for environment_id in sorted(candidates):
        catalogue_dir = output / "catalogues" / environment_id
        catalogue_dir.mkdir(parents=True)
        catalogue_path = catalogue_dir / "candidates.jsonl"
        _atomic_text(catalogue_path, "".join(
            _canonical_json(row) + "\n" for row in candidates[environment_id]))
        audit_dir = output / "audits" / environment_id
        audit_args = SimpleNamespace(
            catalogue=str(catalogue_dir), out=str(audit_dir),
            fixed_environment_id=environment_id, environment_specs=str(specs_path),
            workers=args.workers, limit=0, fov=args.fov,
            dense_fields=args.dense_fields, dense_rays=args.dense_rays,
            perturbations=args.perturbations, perturb_rays=args.perturb_rays,
            perturb_scale=args.perturb_scale,
            max_fingerprints_per_field=args.max_fingerprints_per_field,
            seed=args.seed,
        )
        audit_manifest = interaction_audit.run(audit_args)
        audit_manifests.append({
            "environment_id": environment_id,
            "manifest": str((audit_dir / "manifest.json").resolve()),
            "manifest_sha256": _sha256(audit_dir / "manifest.json"),
            "records": audit_manifest["records_audited"],
            "label_counts": audit_manifest["label_counts"],
        })
        with (audit_dir / "records.jsonl").open() as stream:
            for line in stream:
                if line.strip():
                    all_records.append(json.loads(line))

    strict_rows = []
    rejection_counts = Counter()
    for record in all_records:
        reasons = _strict_rejection_reasons(record)
        if not reasons:
            strict_rows.append(record)
        else:
            rejection_counts.update(reasons)
    strict_path = output / "strict_intentional.jsonl"
    _atomic_text(strict_path, "".join(
        _canonical_json(row) + "\n" for row in strict_rows))

    label_counts = Counter(row.get("audit_label", "missing") for row in all_records)
    candidate_counts = {key: len(value) for key, value in sorted(candidates.items())}
    strict_counts = Counter(row["fixed_environment_id"] for row in strict_rows)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_manifest": str(baseline_manifest_path),
        "baseline_manifest_sha256": _sha256(baseline_manifest_path),
        "environment_specs": str(specs_path),
        "environment_specs_sha256": _sha256(specs_path),
        "candidate_counts": candidate_counts,
        "candidates_total": sum(candidate_counts.values()),
        "audit_label_counts": dict(sorted(label_counts.items())),
        "strict_positive_counts": {
            key: strict_counts.get(key, 0) for key in sorted(candidate_counts)},
        "strict_positives_total": len(strict_rows),
        "strict_gate": {
            "audit_label": "intentional",
            "intended_dominance_all_dense_fields": True,
            "same_dominant_topology_all_dense_fields": True,
            "minimum_contact_root_support": 0.80,
            "minimum_mean_intended_delivered_energy_share": 0.05,
            "minimum_topology_perturbation_persistence": 0.80,
            "minimum_contact_perturbation_persistence": 0.80,
            "minimum_tir_critical_and_grazing_margin_deg": 0.50,
        },
        "strict_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "audits": audit_manifests,
        "outputs": {
            "strict_intentional": {"path": str(strict_path),
                                   "sha256": _sha256(strict_path)},
        },
        "wall_seconds": time.time() - started,
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_text(output / "README.md", (
        "# Fixed baseline dense audit\n\n"
        "`strict_intentional.jsonl` contains only archive candidates that pass the "
        "interaction audit plus the stricter all-dense-field admission gate recorded "
        "in `manifest.json`. These are internal ray-engine labels, not CODE V validation.\n"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--environment-specs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--fov", type=float, default=8.0)
    parser.add_argument("--dense-fields", type=int, default=17)
    parser.add_argument("--dense-rays", type=int, default=101)
    parser.add_argument("--perturbations", type=int, default=4)
    parser.add_argument("--perturb-rays", type=int, default=33)
    parser.add_argument("--perturb-scale", type=float, default=0.005)
    parser.add_argument("--max-fingerprints-per-field", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.workers <= 0 or args.dense_fields < 3 or args.dense_fields % 2 == 0:
        parser.error("workers must be positive and dense-fields must be odd and >= 3")
    if min(args.dense_rays, args.perturb_rays) < 3:
        parser.error("ray counts must be at least 3")
    if args.perturbations < 0 or args.perturb_scale < 0:
        parser.error("perturbation count and scale must be non-negative")
    manifest = run(args)
    print(json.dumps({
        "output": str(Path(args.out).resolve()),
        "candidates": manifest["candidates_total"],
        "strict_positives": manifest["strict_positives_total"],
        "wall_seconds": round(manifest["wall_seconds"], 3),
    }, indent=2))


if __name__ == "__main__":
    main()
