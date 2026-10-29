"""Build leakage-safe CMA distillation and optical-hindsight trajectory views.

Input files contain one ``nslens.optimizer_transition.v1`` row per complete emitter
generation. Splits operate on whole lifecycle units; adjacent generations are never
randomized independently.

Example::

    python -m nslens.rl.data.build_optimizer_trajectories \
        --runs runs --out runs/evotf_trajectories --hindsight 5
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schema import OPTIMIZER_TRANSITION_SCHEMA, validate_optimizer_transition


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_token(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_sources(paths: Sequence[Path]) -> tuple[list[dict], list[dict]]:
    records = []
    sources = []
    seen_ids = set()
    for path in sorted(paths):
        sha = _hash_file(path)
        count = 0
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                errors = validate_optimizer_transition(record)
                if errors:
                    raise ValueError(f"{path}:{line_number}: {errors}")
                source_id = f"{sha[:16]}:{record['lifecycle_id']}"
                record["source_lifecycle_id"] = source_id
                record["source"] = {
                    "path": str(path.resolve()), "sha256": sha,
                    "line": line_number,
                }
                if record["transition_id"] in seen_ids:
                    raise ValueError(f"duplicate transition_id {record['transition_id']}")
                seen_ids.add(record["transition_id"])
                records.append(record)
                count += 1
        sources.append({"path": str(path.resolve()), "sha256": sha, "records": count})
    return records, sources


def _candidate_score(candidate: dict) -> tuple[float, float]:
    tier = {"invalid": 0.0, "none": 1.0, "improved": 2.0, "new": 3.0}[
        candidate["status"]]
    if candidate["status"] == "improved":
        within = float(candidate.get("archive_improvement") or 0.0)
    else:
        within = float(candidate.get("fitness") or 0.0)
    return tier, within


def add_targets(records: Sequence[dict], horizon: int) -> list[dict]:
    """Attach marginal-CMA teacher and fixed-horizon verified hindsight targets."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    by_lifecycle = defaultdict(list)
    for source in records:
        by_lifecycle[source["source_lifecycle_id"]].append(source)
    output = []
    for lifecycle in by_lifecycle.values():
        lifecycle.sort(key=lambda row: row["generation"])
        for position, source in enumerate(lifecycle):
            row = copy.deepcopy(source)
            after = row["state_after"]
            covariance = np.asarray(after["covariance"], dtype=np.float64)
            marginal_std = float(after["sigma"]) * np.sqrt(np.diag(covariance))
            row["teacher_target"] = {
                "next_mean": [float(value) for value in after["mean"]],
                "next_diagonal_std": [float(value) for value in marginal_std],
                "cross_covariance_omitted": True,
            }
            window = lifecycle[position:position + horizon]
            choices = [
                (candidate, future)
                for future in window
                for candidate in future["candidates"]
                if candidate["status"] != "invalid"
            ]
            best = max(choices, key=lambda pair: _candidate_score(pair[0])) \
                if choices else None
            complete = len(window) == horizon
            if best is None:
                row["hindsight_target"] = {
                    "mask": False, "truncated": not complete,
                    "lookahead_generations": len(window),
                    "candidate_transition_id": None, "candidate_index": None,
                    "displacement": None, "standardized_displacement": None,
                }
            else:
                candidate, future = best
                mean = np.asarray(row["state_before"]["mean"], dtype=np.float64)
                sigma = float(row["state_before"]["sigma"])
                candidate_x = np.asarray(candidate["search_coordinates"], dtype=np.float64)
                displacement = candidate_x - mean
                row["hindsight_target"] = {
                    # Preserve the shorter-window target for audit/analysis, but mask
                    # it from the fixed-horizon loss.  Otherwise tail states receive a
                    # systematically different objective from complete windows.
                    "mask": complete,
                    "truncated": not complete,
                    "lookahead_generations": len(window),
                    "candidate_transition_id": future["transition_id"],
                    "candidate_index": candidate["candidate_index"],
                    "status": candidate["status"],
                    "displacement": [float(value) for value in displacement],
                    "standardized_displacement": [
                        float(value) for value in displacement / max(sigma, 1e-12)
                    ],
                }
            output.append(row)
    return output


def _lifecycle_groups(records: Sequence[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["source_lifecycle_id"]].append(record)
    for rows in groups.values():
        rows.sort(key=lambda record: record["generation"])
    return dict(groups)


def _seed_assignment(groups: dict[str, list[dict]]) -> dict[str, str]:
    search_seeds = sorted({int(rows[0].get("search_seed", rows[0]["seed"]))
                           for rows in groups.values()})
    if len(search_seeds) >= 3:
        split_of = {seed: ("test" if seed == search_seeds[-1] else
                           "validation" if seed == search_seeds[-2] else "train")
                    for seed in search_seeds}
    elif len(search_seeds) == 2:
        split_of = {search_seeds[0]: "train", search_seeds[1]: "test"}
    else:
        split_of = {seed: "train" for seed in search_seeds}
    return {key: split_of[int(rows[0].get("search_seed", rows[0]["seed"]))]
            for key, rows in groups.items()}


def _component_assignment(
    groups: dict[str, list[dict]], token_of, *, test_label: str = "test"
) -> dict[str, str]:
    """Keep all lifecycle units sharing a route/environment token in one split."""
    token_groups = defaultdict(set)
    for lifecycle_id, rows in groups.items():
        for token in token_of(rows):
            token_groups[token].add(lifecycle_id)
    adjacency = {key: set() for key in groups}
    for lifecycle_ids in token_groups.values():
        for lifecycle_id in lifecycle_ids:
            adjacency[lifecycle_id].update(lifecycle_ids)
    components = []
    unseen = set(groups)
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            neighbors = adjacency[stack.pop()] & unseen
            unseen -= neighbors
            component |= neighbors
            stack.extend(neighbors)
        components.append(sorted(component))
    components.sort(key=lambda component: hashlib.sha256(
        "|".join(component).encode()).hexdigest())
    assignment = {}
    cutoff = max(1, len(components) // 5) if len(components) > 1 else 0
    held = components[-cutoff:] if cutoff else []
    held_ids = {item for component in held for item in component}
    for lifecycle_id in groups:
        assignment[lifecycle_id] = test_label if lifecycle_id in held_ids else "train"
    return assignment


def _design_tokens(rows: Sequence[dict]) -> set[str]:
    tokens = set()
    for row in rows:
        tokens.add(_json_token(row["state_before"]["mean"]))
        tokens.update(_json_token(candidate["search_coordinates"])
                      for candidate in row["candidates"])
    return tokens


def _drop_design_leakage(
    groups: dict[str, list[dict]], assignment: dict[str, str]
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    order = ("train", "validation", "test")
    claimed = set()
    selected = defaultdict(list)
    dropped = Counter()
    for split in order:
        current = set()
        for lifecycle_id in sorted(key for key, value in assignment.items() if value == split):
            tokens = _design_tokens(groups[lifecycle_id])
            if tokens & claimed:
                dropped[split] += 1
                continue
            selected[split].extend(groups[lifecycle_id])
            current |= tokens
        claimed |= current
    return dict(selected), dict(dropped)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return len(rows)


def _view(
    name: str, groups: dict[str, list[dict]], assignment: dict[str, str], out_dir: Path
) -> dict:
    selected, dropped = _drop_design_leakage(groups, assignment)
    counts = {split: _write_jsonl(out_dir / name / f"{split}.jsonl", rows)
              for split, rows in selected.items()}
    lifecycle_counts = Counter()
    for lifecycle_id, split in assignment.items():
        if any(record in selected.get(split, []) for record in groups[lifecycle_id]):
            lifecycle_counts[split] += 1
    return {
        "records": counts,
        "lifecycles": dict(lifecycle_counts),
        "dropped_lifecycles_for_exact_design_leakage": dropped,
    }


def build(paths: Sequence[Path], out_dir: Path, *, hindsight: int = 5) -> dict:
    records, source_manifest = _read_sources(paths)
    if not records:
        raise RuntimeError("no optimizer transition records found")
    records = add_targets(records, hindsight)
    groups = _lifecycle_groups(records)
    seed_assignment = _seed_assignment(groups)
    route_assignment = _component_assignment(
        groups,
        lambda rows: {row.get("route_family") for row in rows
                      if row.get("route_family") not in (None, "invalid")},
    )
    for lifecycle_id, rows in groups.items():
        if not any(row.get("route_family") not in (None, "invalid") for row in rows):
            route_assignment[lifecycle_id] = "train"
    environment_assignment = _component_assignment(
        groups,
        lambda rows: {row["fingerprints"]["objective"] for row in rows},
    )
    views = {
        "held_seed": _view("held_seed", groups, seed_assignment, out_dir),
        "held_route": _view("held_route", groups, route_assignment, out_dir),
        "held_environment": _view(
            "held_environment", groups, environment_assignment, out_dir),
    }
    statuses = Counter(candidate["status"] for row in records
                       for candidate in row["candidates"])
    failures = Counter((candidate.get("failure") or {}).get("stage")
                       for row in records for candidate in row["candidates"]
                       if candidate["status"] == "invalid")
    route_counts = Counter(row.get("route_family") or "invalid" for row in records)
    hindsight_statuses = Counter(
        (row["hindsight_target"].get("status") or "none")
        for row in records if row["hindsight_target"].get("mask"))
    manifest = {
        "schema_version": "nslens.optimizer_trajectory_manifest.v1",
        "record_schema": OPTIMIZER_TRANSITION_SCHEMA,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hindsight_generations": hindsight,
        "source_files": source_manifest,
        "complete_generations": len(records),
        "candidate_rows": sum(len(row["candidates"]) for row in records),
        "unique_emitter_states": len({
            _json_token(row["state_before"]) for row in records}),
        "emitter_lifecycles": len(groups),
        "hindsight_targets": {
            "usable": sum(bool(row["hindsight_target"].get("mask")) for row in records),
            "masked": sum(not bool(row["hindsight_target"].get("mask"))
                          for row in records),
            "truncated": sum(bool(row["hindsight_target"].get("truncated"))
                             for row in records),
            "status_composition": dict(hindsight_statuses),
        },
        "route_families": dict(route_counts),
        "status_composition": dict(statuses),
        "failure_composition": {str(key): value for key, value in failures.items()},
        "objective_fingerprints": sorted({
            row["fingerprints"]["objective"] for row in records}),
        "views": views,
        "checks": {
            "all_input_records_validated": True,
            "whole_lifecycle_splits": True,
            "adjacent_generations_kept_together": True,
            "exact_parent_and_candidate_cross_split_leakage_removed": True,
            "teacher_target_is_marginal_full_cma": True,
            "hindsight_target_shared_by_numeric_and_optical_arms": True,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hindsight", type=int, default=5)
    args = parser.parse_args(argv)
    paths = args.input or sorted(args.runs.glob("**/optimizer_transitions.jsonl"))
    manifest = build(paths, args.out, hindsight=args.hindsight)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
