"""Derive training views from raw CODE V transition logs without another COM run.

Raw transitions remain the source of truth. This creates three disposable views:
successful-action chat SFT, same-state chosen/rejected preferences, and flat surrogate
regression rows. The split unit is ``design_id`` so an optical starting point and all of
its optimization episodes stay in exactly one split.

    python -m nslens.rl.data.derive_codev transitions.jsonl \
      --out codev_dataset_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

from .schema import CODEV_TRANSITION_SCHEMA, validate_codev_transition


DERIVED_SCHEMA = "nslens.codev_derived.v1"
SYSTEM = (
    "You refine a folded optical design. Return exactly one "
    "<action>{\"normalized_delta\":[numbers],\"trust_scale\":number}</action> "
    "block and no prose."
)


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _split(design_id: str) -> str:
    bucket = int(_hash("codev-design-split", design_id)[:8], 16) % 10
    return "test_design" if bucket == 0 else (
        "validation_design" if bucket == 1 else "train")


def _state_hash(state: dict) -> str:
    return hashlib.sha256(json.dumps(
        state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _feasible(rec: dict) -> bool:
    result = rec["result"]
    if not result["trace_ok"] or result.get("topology_preserved") is False:
        return False
    slacks = result.get("constraint_slacks_after") or {}
    return all(isinstance(v, (int, float)) and float(v) >= 0 for v in slacks.values())


def _rank_key(rec: dict):
    result = rec["result"]
    return (_feasible(rec), bool(result["trace_ok"]), float(result["delta_merit"]))


def _action_block(rec: dict) -> str:
    action = rec["action"]
    payload = {
        "normalized_delta": action["normalized_delta"],
        "trust_scale": float(action.get("trust_scale", 1.0)),
    }
    return "<action>" + json.dumps(payload, separators=(",", ":")) + "</action>"


def _prompt(state: dict) -> List[dict]:
    # Fixed-size numeric state only. Full reports remain artifacts referenced by the raw
    # transition; sending them through the language model caused the original token waste.
    compact = {k: state.get(k) for k in (
        "parameter_names", "parameters", "parameter_scales", "active_mask",
        "merit_before", "metrics_before", "constraint_slacks", "topology_id",
        "budget_remaining")}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(compact, separators=(",", ":"))}]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def build(paths: List[Path], out_dir: Path) -> dict:
    rows, bad, source_hashes = [], [], {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for lineno, raw in enumerate(fh, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except Exception as exc:
                    bad.append({"path": str(path), "line": lineno,
                                "errors": [f"invalid JSON: {exc}"]})
                    continue
                errors = validate_codev_transition(rec)
                if errors:
                    bad.append({"path": str(path), "line": lineno, "errors": errors})
                    continue
                rec["_source"] = {"path": str(path), "line": lineno}
                rows.append(rec)
        source_hashes[str(path)] = digest.hexdigest()
    if bad:
        raise ValueError(f"refusing to derive from {len(bad)} invalid transitions; "
                         f"first={bad[0]}")

    groups = defaultdict(list)
    for rec in rows:
        groups[(rec["episode_id"], rec["step"])].append(rec)
    sft, prefs, surrogate = defaultdict(list), defaultdict(list), defaultdict(list)
    skipped = Counter()
    for (episode, step), candidates in groups.items():
        states = {_state_hash(c["state"]) for c in candidates}
        if len(states) != 1:
            skipped["state_mismatch_inside_group"] += 1
            continue
        split = _split(candidates[0]["design_id"])
        group_id = _hash(episode, step, next(iter(states)))[:24]
        ordered = sorted(candidates, key=_rank_key, reverse=True)
        for rec in candidates:
            surrogate[split].append({
                "schema_version": DERIVED_SCHEMA, "dataset_kind": "surrogate_transition",
                "transition_id": rec["transition_id"], "group_id": group_id,
                "split": split, "state": rec["state"], "action": rec["action"],
                "labels": {
                    "merit_after": rec["result"]["merit_after"],
                    "delta_merit": rec["result"]["delta_merit"],
                    "trace_ok": rec["result"]["trace_ok"],
                    "feasible": _feasible(rec),
                    "accepted": rec["result"]["accepted"],
                    "metrics_after": rec["result"].get("metrics_after"),
                    "constraint_slacks_after": rec["result"].get(
                        "constraint_slacks_after"),
                    "topology_preserved": rec["result"].get("topology_preserved"),
                },
                "source": rec["_source"],
            })
        if len(ordered) < 2 or _rank_key(ordered[0]) == _rank_key(ordered[1]):
            skipped["no_preference_gradient"] += 1
            continue
        chosen, rejected = ordered[0], ordered[1]
        prompt = _prompt(chosen["state"])
        pref = {
            "schema_version": DERIVED_SCHEMA, "dataset_kind": "codev_preference",
            "preference_id": group_id, "group_id": group_id, "split": split,
            "prompt_messages": prompt,
            "chosen_response": _action_block(chosen),
            "rejected_response": _action_block(rejected),
            "labels": {
                "chosen_delta_merit": chosen["result"]["delta_merit"],
                "rejected_delta_merit": rejected["result"]["delta_merit"],
                "chosen_feasible": _feasible(chosen),
                "rejected_feasible": _feasible(rejected),
            },
            "ranking_rule": "feasible > trace_ok > delta_merit; accepted is not a label",
        }
        prefs[split].append(pref)
        if _feasible(chosen) and float(chosen["result"]["delta_merit"]) > 0:
            sft[split].append({
                "schema_version": DERIVED_SCHEMA,
                "dataset_kind": "successful_action_sft", "example_id": group_id,
                "group_id": group_id, "split": split,
                "messages": prompt + [
                    {"role": "assistant", "content": _action_block(chosen)}],
                "labels": {
                    "delta_merit": chosen["result"]["delta_merit"],
                    "merit_after": chosen["result"]["merit_after"],
                    "feasible": True,
                },
            })
        else:
            skipped["best_action_not_successful"] += 1

    counts = {"sft": {}, "preferences": {}, "surrogate": {}}
    for name, corpus in (("sft", sft), ("preferences", prefs),
                         ("surrogate", surrogate)):
        for split, part in corpus.items():
            counts[name][split] = _write_jsonl(out_dir / name / f"{split}.jsonl", part)
    design_splits = defaultdict(set)
    for rec in rows:
        design_splits[_split(rec["design_id"])].add(rec["design_id"])
    split_names = list(design_splits)
    overlap = {f"{a}<->{b}": sorted(design_splits[a] & design_splits[b])
               for i, a in enumerate(split_names) for b in split_names[i + 1:]}
    manifest = {
        "schema_version": DERIVED_SCHEMA, "raw_schema_version": CODEV_TRANSITION_SCHEMA,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_sha256": source_hashes, "raw_transitions": len(rows),
        "same_state_groups": len(groups), "counts": counts,
        "skipped_groups": dict(skipped),
        "split_unit": "design_id", "design_counts": {
            k: len(v) for k, v in design_splits.items()},
        "checks": {"all_raw_records_valid": True,
                   "design_overlap_between_splits": overlap,
                   "accepted_used_as_training_rank": False},
        "ranking_rule": "feasible > trace_ok > delta_merit",
        "limitations": [
            "At least two candidates from an identical state are required for preference data.",
            "A missing constraint slack cannot be inferred; log every enforced slack.",
            "These labels inherit the exact merit-function and translator versions in raw data.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(build([Path(p) for p in a.paths], Path(a.out)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
