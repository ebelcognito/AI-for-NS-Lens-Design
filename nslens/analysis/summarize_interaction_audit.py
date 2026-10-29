"""Create compact, reproducible summary tables from an interaction-audit JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "nslens.interaction_audit_summary.v1"


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


def run(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    with source.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    traced = [row for row in rows if row["audit_label"] != "pretrace-invalid"]
    eligible = [row for row in rows
                if (row.get("source_hardened_verdict") or {}).get("training_eligible")]
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row.get("source_split", "unknown")][row["audit_label"]] += 1
    reasons = Counter(reason for row in rows for reason in row.get("reasons", []))
    intentional = [row for row in rows if row["audit_label"] == "intentional"]
    compact_intentional = [{
        "design_id": row["design_id"], "source_split": row["source_split"],
        "training_eligible": row["source_hardened_verdict"].get("training_eligible"),
        "hardened_fitness": row["source_hardened_verdict"].get("fitness"),
        "historical_sequence": row["historical_sequence"], "design": row["design"],
        "delivered_energy_share_mean": row["historical_full_delivered_energy_share_mean"],
        "dense_dominance_fraction": row["historical_dense_dominance_fraction"],
        "field_persistence": row["global_topology_field_persistence"],
        "perturbation_persistence": row["global_topology_perturbation_persistence"],
    } for row in sorted(intentional, key=lambda item: -float(
        item["source_hardened_verdict"].get("fitness") or 0.0))]
    intentional_path = output / "intentional_candidates.jsonl"
    _atomic(intentional_path, "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in compact_intentional))
    summary = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source), "source_sha256": _sha(source),
        "all_candidates": len(rows), "traceable_candidates": len(traced),
        "label_counts_all": dict(Counter(row["audit_label"] for row in rows)),
        "label_counts_traceable": dict(Counter(row["audit_label"] for row in traced)),
        "training_eligible_candidates": len(eligible),
        "label_counts_training_eligible": dict(Counter(
            row["audit_label"] for row in eligible)),
        "label_counts_by_source_split": {
            split: dict(counts) for split, counts in sorted(by_split.items())},
        "top_reasons": [{"reason": reason, "count": count}
                        for reason, count in reasons.most_common(20)],
        "intentional_candidates": len(compact_intentional),
        "intentional_sequences": dict(Counter(
            row["historical_sequence"] for row in intentional)),
        "outputs": {"intentional_candidates": {
            "path": str(intentional_path), "sha256": _sha(intentional_path),
            "rows": len(compact_intentional)}},
    }
    _atomic(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = run(Path(args.records).resolve(), Path(args.out).resolve())
    print(json.dumps({"output": str(Path(args.out).resolve()),
                      "label_counts": summary["label_counts_all"]}, indent=2))


if __name__ == "__main__":
    main()
