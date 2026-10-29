"""Select a balanced, topology-diverse cohort for lossless fingerprint export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..rl import env_config

SCHEMA = "nslens.interaction_training_cohort.v1"
LABELS = ("intentional", "field-dependent", "spurious", "unstable")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _feature(record: dict) -> np.ndarray:
    env = env_config.get("fold")
    x = np.array([record["design"][name] for name in env.param_names], float)
    return (x - np.asarray(env.problem.lo, float)) / (
        np.asarray(env.problem.hi, float) - np.asarray(env.problem.lo, float))


def _fitness(record: dict) -> float:
    return float((record.get("source_hardened_verdict") or {}).get("fitness") or 0.0)


def _select(pool: list[dict], count: int) -> list[dict]:
    """Seed with the best member of each sequence, then farthest-point fill."""
    by_sequence = defaultdict(list)
    for record in pool:
        by_sequence[record["historical_sequence"]].append(record)
    sequence_best = sorted((max(group, key=_fitness) for group in by_sequence.values()),
                           key=_fitness, reverse=True)
    selected = sequence_best[:count]
    remaining = [record for record in pool if record not in selected]
    while len(selected) < min(count, len(pool)):
        if not selected:
            chosen = max(remaining, key=_fitness)
        else:
            chosen = max(remaining, key=lambda record: (
                min(float(np.linalg.norm(_feature(record) - _feature(other)))
                    for other in selected), _fitness(record)))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def run(audit_path: Path, catalogue: Path, output: Path, per_label: int) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    with audit_path.open() as fh:
        audits = [json.loads(line) for line in fh if line.strip()]
    selected = []
    counts = {}
    for label in LABELS:
        pool = [record for record in audits if record.get("audit_label") == label]
        chosen = _select(pool, per_label)
        selected.extend(chosen)
        counts[label] = len(chosen)
    labels_of = {record["design_id"]: record["audit_label"] for record in selected}

    source_records = {}
    source_files = []
    for path in sorted(catalogue.glob("*.jsonl")):
        source_files.append({"path": str(path), "sha256": _sha(path)})
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    source_records[record["design_id"]] = record
    missing = set(labels_of) - set(source_records)
    if missing:
        raise ValueError(f"selected designs missing from source catalogue: {sorted(missing)}")
    ordered = [source_records[design_id] for design_id in sorted(labels_of)]
    selected_path = output / "selected.jsonl"
    _atomic(selected_path, "".join(json.dumps(record, separators=(",", ":")) + "\n"
                                    for record in ordered))
    labels_path = output / "selection_labels.json"
    _atomic(labels_path, json.dumps(labels_of, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": "best-per-sequence then normalized-prescription farthest-point fill",
        "source_audit": str(audit_path), "source_audit_sha256": _sha(audit_path),
        "source_catalogue": str(catalogue), "source_files": source_files,
        "target_per_label": per_label, "counts": counts, "total": len(ordered),
        "outputs": {
            "selected_catalogue": {"path": str(selected_path),
                                   "sha256": _sha(selected_path), "rows": len(ordered)},
            "selection_labels": {"path": str(labels_path), "sha256": _sha(labels_path)},
        },
    }
    _atomic(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-records", required=True)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-label", type=int, default=36)
    args = parser.parse_args()
    manifest = run(Path(args.audit_records).resolve(), Path(args.catalogue).resolve(),
                   Path(args.out).resolve(), args.per_label)
    print(json.dumps({"output": str(Path(args.out).resolve()),
                      "counts": manifest["counts"], "total": manifest["total"]}, indent=2))


if __name__ == "__main__":
    main()
