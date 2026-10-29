"""Re-label historical action groups under the current hardened optical environment.

The base action corpus preserves what CMA-ME knew at the time. This derived view is the one
to use for refinement SFT/preferences: every parent and candidate is freshly evaluated with
the detector/source clearances, curvature cap, nine-field clean/exit tests, and TIR margin.

    python -m nslens.rl.data.harden_search_actions \
      --input runs/rl_datasets/search_actions_v1 \
      --out runs/rl_datasets/search_actions_hardened_v1 --workers 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .. import env_config
from ..reward import HARD_GATE_KEYS, gate_score, hardened_pass
from ..server import RewardPool
from .build_hardened_catalogue import _jsonable, yardstick_payload
from .build_search_actions import _action_block
from .schema import (SEARCH_GROUP_SCHEMA, SEARCH_PREFERENCE_SCHEMA,
                     HARDENED_ACTION_SFT_SCHEMA, validate_hardened_action_sft,
                     validate_search_group, validate_search_preference)


SYSTEM = (
    "Refine a nine-parameter folded optic while preserving its realized topology. "
    "Parameter order is ray_y,ang,cx,thick,fsag,bsag,tilt_a,tilt_b,n. Return exactly "
    "one <action>{\"normalized_delta\":[nine numbers]}</action> block and no prose."
)


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _verdict_payload(v) -> dict:
    return {
        "stage": v.stage, "reason": v.reason, "hard_pass": hardened_pass(v),
        "fitness": float(v.fitness), "topology_index": v.topology,
        "thinness_bin": v.thinness_bin, "thinness_T": v.thinness_T,
        "full_signature": v.signature,
        "per_field": [float(x) for x in v.per_field],
        "critical_margin_deg": v.crit_margin,
        "gates": _jsonable(v.gates or {}),
    }


def _prompt(state: dict) -> List[dict]:
    parent = state["parent_hardened"]
    compact = {
        "parent_normalized": state["parent_normalized"], "sigma": state["sigma"],
        "fov_half_angle_deg": max(abs(float(v)) for v in state["fields_deg"]),
        "focus_window_half_width": state["focus_window_half_width"],
        "parent_optics": {
            "stage": parent["stage"], "hard_pass": parent["hard_pass"],
            "fitness": parent["fitness"], "topology_index": parent["topology_index"],
            "thinness_bin": parent["thinness_bin"],
            "clean_all": parent["gates"].get("clean_all"),
            "exit_ok_all": parent["gates"].get("exit_ok_all"),
            "margin_ok": parent["gates"].get("margin_ok"),
        },
        "objective": "preserve topology, clear every hard gate, improve fitness",
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(compact, separators=(",", ":"))}]


def _rank(v, parent) -> tuple:
    """Lexicographic safety first; quality only breaks ties after acceptance."""
    preserve_required = parent.topology is not None and parent.stage == "scored"
    preserved = not preserve_required or v.topology == parent.topology
    hp = hardened_pass(v)
    if hp and preserved:
        tier = 5
        value = 0.8 * float(v.fitness) + 0.2 * gate_score(
            v, env_config.get("fold", fov=8.0, focus_w=8.0))
    elif hp:
        tier, value = 4, float(v.fitness)
    elif v.stage == "scored":
        tier = 3
        value = 0.6 * gate_score(v, env_config.get("fold", fov=8.0, focus_w=8.0)) \
            + 0.4 * float(v.fitness)
    elif v.stage == "off_class":
        tier, value = 2, float(v.fitness)
    elif v.stage == "no_light":
        tier, value = 1, 0.0
    else:
        tier, value = 0, 0.0
    return tier, float(value), bool(preserved)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def _load_groups(root: Path) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for path in sorted((root / "groups").glob("*.jsonl")):
        for line in path.open():
            if line.strip():
                rec = json.loads(line)
                errors = validate_search_group(rec)
                if errors:
                    raise ValueError(f"invalid base group in {path}: {errors}")
                out[rec["split"]].append(rec)
    return out


def build(input_dir: Path, out_dir: Path, *, workers: int | None = None,
          batch_size: int = 10000) -> dict:
    groups_by_split = _load_groups(input_dir)
    groups = [g for rows in groups_by_split.values() for g in rows]
    if not groups:
        raise RuntimeError("no base action groups")
    base_manifest_path = input_dir / "manifest.json"
    base_bytes = base_manifest_path.read_bytes()

    # One canonical variant for this first refinement prior. All accepted source traces
    # already use the same relative-sag nine-gene layout and +-8 deg / 8 mm target.
    variant = ("fold", 8.0, 8.0)
    unique: Dict[tuple, list] = {}
    for group in groups:
        unique.setdefault(tuple(float(v) for v in group["state"]["parent"]),
                          group["state"]["parent"])
        for cand in group["candidates"]:
            unique.setdefault(tuple(float(v) for v in cand["genotype"]), cand["genotype"])
    keys = list(unique)
    values = [unique[k] for k in keys]
    verdict_of = {}
    with RewardPool(workers=workers, cache_size=max(200000, len(values) + 1000),
                    chunksize=16) as pool:
        for start in range(0, len(values), batch_size):
            part = values[start:start + batch_size]
            verdicts = pool.verdicts([(variant, x) for x in part])
            for key, verdict in zip(keys[start:start + batch_size], verdicts):
                verdict_of[key] = verdict
            print(f"[harden-actions] {min(start + len(part), len(values))}/{len(values)}",
                  flush=True)
        pool_stats = {
            "unique_designs": pool.stats.designs,
            "engine_seconds": pool.stats.engine_seconds,
            "designs_per_second": pool.stats.designs_per_second,
        }

    hardened_groups = defaultdict(list)
    sft = defaultdict(list)
    preferences = defaultdict(list)
    dropped = Counter()
    stage_counts = Counter()
    candidate_pass = Counter()
    for group in groups:
        parent_key = tuple(float(v) for v in group["state"]["parent"])
        parent_v = verdict_of[parent_key]
        group["state"]["parent_hardened"] = _verdict_payload(parent_v)
        ranked = []
        for cand in group["candidates"]:
            key = tuple(float(v) for v in cand["genotype"])
            v = verdict_of[key]
            tier, value, preserved = _rank(v, parent_v)
            cand["hardened"] = _verdict_payload(v)
            cand["hardened_preference_tier"] = tier
            cand["hardened_within_tier_value"] = value
            cand["topology_preserved_from_parent"] = preserved
            ranked.append((tier, value, -cand["candidate_index"], cand["candidate_index"]))
            stage_counts[v.stage] += 1
            candidate_pass["hard_pass" if hardened_pass(v) else "not_hard_pass"] += 1
        order = [x[3] for x in sorted(ranked, reverse=True)]
        distinct = {(x[0], round(x[1], 12)) for x in ranked}
        if len(distinct) < 2:
            dropped["no_hardened_preference_gradient"] += 1
            continue
        group["dataset_kind"] = "hardened_refinement_group"
        group["ranking"] = order
        group["best_candidate_index"] = order[0]
        for rank, idx in enumerate(order):
            group["candidates"][idx]["rank"] = rank
        group["supervision"] = {
            "basis": "canonical hard gates and topology preservation before quality",
            "comparable_scope": "single fingerprinted hardened yardstick",
            "historical_absolute_quality_label": False,
        }
        prompt = _prompt(group["state"])
        chosen = group["candidates"][order[0]]
        rejected = group["candidates"][order[1]]
        group["messages"] = prompt + [
            {"role": "assistant", "content": _action_block(chosen)}]
        errors = validate_search_group(group)
        if errors:
            raise ValueError(f"invalid hardened group {group['group_id']}: {errors}")
        split = group["split"]
        hardened_groups[split].append(group)

        pair = {
            "schema_version": SEARCH_PREFERENCE_SCHEMA,
            "preference_id": f"{group['group_id']}:hardened",
            "group_id": group["group_id"], "state_id": group["state_id"],
            "sample_weight": group["sample_weight"], "split": split,
            "preference_type": "archive_rank", "state": group["state"],
            "prompt_messages": prompt, "chosen": chosen, "rejected": rejected,
            "chosen_response": _action_block(chosen),
            "rejected_response": _action_block(rejected),
            "source": group["source"],
        }
        errors = validate_search_preference(pair)
        if errors:
            raise ValueError(f"invalid hardened preference: {errors}")
        preferences[split].append(pair)

        chosen_v = chosen["hardened"]
        preserve_ok = bool(chosen["topology_preserved_from_parent"])
        parent_ok = hardened_pass(parent_v)
        improves = (not parent_ok or float(chosen_v["fitness"]) > float(parent_v.fitness))
        if chosen_v["hard_pass"] and preserve_ok and improves:
            sft[split].append({
                "schema_version": HARDENED_ACTION_SFT_SCHEMA,
                "example_id": f"{group['group_id']}:hardened",
                "state_id": group["state_id"], "split": split,
                "sample_weight": group["sample_weight"],
                "messages": prompt + [
                    {"role": "assistant", "content": _action_block(chosen)}],
                "labels": {
                    "parent_hard_pass": parent_ok,
                    "candidate_hard_pass": True,
                    "topology_preserved": True,
                    "parent_fitness": float(parent_v.fitness),
                    "candidate_fitness": float(chosen_v["fitness"]),
                },
                "source_group_id": group["group_id"],
            })

    # Reweight after filtering so each distinct parent/context contributes unit total mass.
    for split, part in hardened_groups.items():
        per_state = Counter(g["state_id"] for g in part)
        for group in part:
            group["sample_weight"] = 1.0 / per_state[group["state_id"]]
        pair_by_group = {p["group_id"]: p for p in preferences[split]}
        for group in part:
            pair_by_group[group["group_id"]]["sample_weight"] = group["sample_weight"]
        sft_per_state = Counter(r["state_id"] for r in sft[split])
        for rec in sft[split]:
            rec["sample_weight"] = 1.0 / sft_per_state[rec["state_id"]]
            errors = validate_hardened_action_sft(rec)
            if errors:
                raise ValueError(f"invalid hardened action SFT: {errors}")

    counts = {"groups": {}, "preferences": {}, "sft": {}}
    for name, corpus in (("groups", hardened_groups), ("preferences", preferences),
                         ("sft", sft)):
        for split, rows in corpus.items():
            counts[name][split] = _write_jsonl(out_dir / name / f"{split}.jsonl", rows)
    yardstick = yardstick_payload()
    manifest = {
        "schema_versions": [SEARCH_GROUP_SCHEMA, SEARCH_PREFERENCE_SCHEMA,
                            HARDENED_ACTION_SFT_SCHEMA],
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dataset": str(input_dir),
        "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "yardstick": yardstick, "counts": counts,
        "source_groups": len(groups), "unique_designs_rescored": len(values),
        "candidate_stages": dict(stage_counts),
        "candidate_hard_gate_counts": dict(candidate_pass),
        "dropped_groups": dict(dropped), "pool_stats": pool_stats,
        "effective_parent_states": {
            split: len({g["state_id"] for g in rows})
            for split, rows in hardened_groups.items()},
        "required_final_gates": [
            "nondegenerate_geometry", "max_sag_ratio", "detector_clearance_12mm",
            "source_clearance_2mm", "in_class_topology_and_thinness",
            *HARD_GATE_KEYS, "topology_preserved_from_parent"],
        "ranking_rule": (
            "hard-pass+topology-preserved > hard-pass topology change > scored near-miss "
            "> off-class > no-light > physical/engine failure; quality breaks ties"),
        "checks": {
            "every_parent_and_candidate_canonically_rescored": True,
            "single_yardstick_fingerprint": yardstick["config_fingerprint"],
            "historical_status_used_for_ranking": False,
            "sft_requires_every_hard_gate_and_topology_preservation": True,
        },
        "limitations": [
            "This remains a local 2D pretraining label; CODE V transitions are authoritative.",
            "Groups were selected from historical QD traces before canonical re-ranking.",
            "The fixed first version covers the current nine-parameter relative-sag fold.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="runs/rl_datasets/search_actions_v1")
    ap.add_argument("--out", default="runs/rl_datasets/search_actions_hardened_v1")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=10000)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.input), Path(a.out), workers=a.workers,
                           batch_size=a.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
