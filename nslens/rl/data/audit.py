"""Measure leakage, redundancy, and effective diversity in generated RL datasets."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .schema import validate_proposer


def _rows(paths: Iterable[Path]):
    for path in paths:
        for lineno, line in enumerate(path.open(), 1):
            if line.strip():
                yield path, lineno, json.loads(line)


def _nearest_distances(rows: List[dict]) -> dict:
    """Nearest-neighbour distance in normalized parameter space, blockwise."""
    if len(rows) < 2:
        return {"n": len(rows), "nearest_distance_percentiles": {}}
    X = []
    for rec in rows:
        design = rec["target"]["design"]
        bounds = rec["input"]["parameter_bounds"]
        names = list(design)
        X.append([(float(design[n]) - float(bounds[n][0]))
                  / (float(bounds[n][1]) - float(bounds[n][0])) for n in names])
    X = np.asarray(X, float)
    nearest = np.full(len(X), np.inf)
    block = 512
    for start in range(0, len(X), block):
        stop = min(start + block, len(X))
        d2 = np.sum((X[start:stop, None, :] - X[None, :, :]) ** 2, axis=2)
        d2[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest[start:stop] = np.sqrt(np.min(d2, axis=1))
    return {
        "n": len(rows),
        "nearest_distance_percentiles": {
            str(q): float(np.percentile(nearest, q)) for q in (0, 10, 25, 50, 75, 90, 100)
        },
        "fraction_with_neighbor_below": {
            str(t): float(np.mean(nearest < t)) for t in (0.001, 0.005, 0.01, 0.02)
        },
    }


def audit_proposer(root: Path) -> dict:
    paths = sorted(root.glob("*/*.jsonl"))
    records = []
    invalid = []
    for path, lineno, rec in _rows(paths):
        errors = validate_proposer(rec)
        if errors:
            invalid.append({"path": str(path), "line": lineno, "errors": errors})
        records.append(rec)
    by_corpus = defaultdict(list)
    for rec in records:
        by_corpus[rec["dataset_kind"]].append(rec)

    exact = defaultdict(lambda: defaultdict(set))
    topology_counts = defaultdict(Counter)
    lineage_counts = defaultdict(Counter)
    physical_anchor_counts = defaultdict(Counter)
    for rec in records:
        corpus, split = rec["dataset_kind"], rec["split"]
        key = tuple((k, float(v)) for k, v in rec["target"]["design"].items())
        exact[corpus][split].add(key)
        topo = (rec.get("labels") or {}).get("topology_index")
        if topo is not None:
            topology_counts[split][int(topo)] += 1
        lineage = str((rec.get("provenance") or {}).get("spec_id") or rec["example_id"])
        lineage_counts[corpus][lineage] += 1
        anchor = str((rec.get("provenance") or {}).get("anchor_id") or lineage)
        physical_anchor_counts[corpus][anchor] += 1

    leak = {}
    for corpus, splits in exact.items():
        for a in splits:
            for b in splits:
                if a < b:
                    leak[f"{corpus}:{a}<->{b}"] = len(splits[a] & splits[b])
    nearest = {}
    for corpus, rows in by_corpus.items():
        for split in sorted({r["split"] for r in rows}):
            nearest[f"{corpus}:{split}"] = _nearest_distances(
                [r for r in rows if r["split"] == split])

    weights = {}
    for corpus, rows in by_corpus.items():
        vals = [float(r.get("sample_weight", 1.0)) for r in rows]
        weights[corpus] = {
            "records": len(rows), "lineages": len(lineage_counts[corpus]),
            "physical_anchor_lineages": len(physical_anchor_counts[corpus]),
            "weight_sum": sum(vals), "min": min(vals), "max": max(vals),
            "largest_lineage": max(lineage_counts[corpus].values()),
        }
    train_top = set(topology_counts.get("train", {})) \
        | set(topology_counts.get("validation_seen", {}))
    test_top = set(topology_counts.get("test_cell", {}))
    return {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root), "files": [str(p) for p in paths],
        "records": len(records), "invalid_records": invalid,
        "counts": {
            f"{corpus}:{split}": n for (corpus, split), n in sorted(
                Counter((r["dataset_kind"], r["split"])
                        for r in records).items())},
        "lineage_and_weight_audit": weights,
        "topology_counts_by_split": {
            k: {str(t): n for t, n in sorted(v.items())}
            for k, v in topology_counts.items()},
        "nearest_neighbor_normalized_parameter_space": nearest,
        "exact_design_overlap_between_splits": leak,
        "checks": {
            "all_records_valid": not invalid,
            "no_exact_design_overlap_between_splits": not any(leak.values()),
            "train_test_cell_topology_overlap": sorted(train_top & test_top),
            "sample_weight_equals_effective_lineages": {
                corpus: math.isclose(v["weight_sum"], v["lineages"],
                                     rel_tol=1e-9, abs_tol=1e-7)
                for corpus, v in weights.items()},
        },
        "interpretation": [
            "Record count is not independent sample size; certified spec lineage is.",
            "Small nearest-neighbour distance is expected for traced basin perturbations.",
            "Directed and unconditioned corpora intentionally share targets and must not "
            "be concatenated as if independent.",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    report = audit_proposer(Path(a.root))
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0 if report["checks"]["all_records_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
