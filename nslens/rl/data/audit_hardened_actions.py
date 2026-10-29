"""Audit whether a hardened local-action corpus contains useful LoRA signal.

The builder manifest proves provenance and gate configuration.  This audit measures the
properties that row counts hide: independent parent states, contradictory same-prompt
actions, feasibility-versus-quality supervision, preference strength, bound saturation,
and held-seed shift.

    python -m nslens.rl.data.audit_hardened_actions \
      runs/rl_datasets/search_actions_hardened_v1 \
      --out runs/rl_datasets/search_actions_hardened_v1/audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SPLITS = ("train", "validation_seed", "test_seed")
FITNESS_THRESHOLDS = (0.001, 0.05, 0.1, 0.2, 0.5, 0.75)
ACTION_RE = re.compile(r"<action>(.*)</action>")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float)
    if not x.size:
        return {}
    return {str(q): float(np.percentile(x, q))
            for q in (0, 5, 10, 25, 50, 75, 90, 95, 100)}


def _counter(values: Iterable[object]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).most_common()}


def _pair_counter(values: Iterable[tuple[object, object]]) -> dict[str, int]:
    return {f"{a}|{b}": int(n) for (a, b), n in Counter(values).most_common()}


def _action(content: str) -> np.ndarray:
    match = ACTION_RE.fullmatch(content)
    if match is None:
        raise ValueError(f"not one exact action block: {content[:80]!r}")
    payload = json.loads(match.group(1))
    return np.asarray(payload["normalized_delta"], dtype=float)


def _message_chars(messages: Sequence[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _prompt_consistency(rows: Sequence[dict]) -> dict:
    """Measure directional agreement when an identical prompt has several SFT targets."""
    by_prompt: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = tuple(m["content"] for m in row["messages"][:-1])
        by_prompt[key].append(_action(row["messages"][-1]["content"]))

    cosines: list[float] = []
    resultants: list[float] = []
    negative_mean_groups = 0
    for actions in by_prompt.values():
        if len(actions) < 2:
            continue
        unit = [a / (float(np.linalg.norm(a)) + 1e-12) for a in actions]
        group_cosines = [float(a @ b) for a, b in combinations(unit, 2)]
        cosines.extend(group_cosines)
        resultants.append(float(np.linalg.norm(np.mean(unit, axis=0))))
        negative_mean_groups += int(float(np.mean(group_cosines)) < 0.0)
    return {
        "unique_prompts": len(by_prompt),
        "prompts_with_multiple_targets": sum(len(v) > 1 for v in by_prompt.values()),
        "maximum_targets_for_one_prompt": max(map(len, by_prompt.values()), default=0),
        "pairwise_action_cosine_quantiles": _quantiles(cosines),
        "mean_directional_resultant": (
            float(np.mean(resultants)) if resultants else None),
        "repeated_prompt_groups_with_negative_mean_cosine": negative_mean_groups,
    }


def _sft_stats(rows: Sequence[dict], group_of: dict[str, dict]) -> dict:
    states = Counter(r["state_id"] for r in rows)
    parent = [json.loads(r["messages"][1]["content"])["parent_optics"] for r in rows]
    candidate_fitness = np.asarray(
        [r["labels"]["candidate_fitness"] for r in rows], dtype=float)
    parent_fitness = np.asarray(
        [r["labels"]["parent_fitness"] for r in rows], dtype=float)
    delta = candidate_fitness - parent_fitness
    actions = np.asarray(
        [_action(r["messages"][-1]["content"]) for r in rows], dtype=float)
    clipped = np.asarray([
        group_of[r["source_group_id"]]["candidates"][
            group_of[r["source_group_id"]]["best_candidate_index"]]["clipped_mask"]
        for r in rows
    ], dtype=bool)
    chars = [_message_chars(r["messages"]) for r in rows]

    actual_improvement = delta > 1e-12
    known_topology = np.asarray([p["topology_index"] is not None for p in parent])
    return {
        "records": len(rows),
        "effective_parent_states": len(states),
        "weight_sum": float(sum(float(r["sample_weight"]) for r in rows)),
        "maximum_examples_per_state": max(states.values(), default=0),
        "parent_stage_counts": _counter(p["stage"] for p in parent),
        "parent_topology_counts": _counter(p["topology_index"] for p in parent),
        "parent_hard_pass": int(sum(bool(r["labels"]["parent_hard_pass"])
                                    for r in rows)),
        "gate_recovery_examples": int(sum(not bool(r["labels"]["parent_hard_pass"])
                                          for r in rows)),
        "candidate_strictly_improves_scalar_fitness": int(actual_improvement.sum()),
        "candidate_strictly_improves_fraction": (
            float(actual_improvement.mean()) if len(rows) else 0.0),
        "candidate_fitness_quantiles": _quantiles(candidate_fitness),
        "candidate_minus_parent_fitness_quantiles": _quantiles(delta),
        "candidate_fitness_threshold_counts": {
            str(t): int((candidate_fitness >= t).sum()) for t in FITNESS_THRESHOLDS},
        "objective_consistent_fit05_known_topology": int(np.sum(
            actual_improvement & (candidate_fitness >= 0.05) & known_topology)),
        "action_l2_norm_quantiles": _quantiles(np.linalg.norm(actions, axis=1)),
        "action_dimension_mean": [float(v) for v in np.mean(actions, axis=0)],
        "action_dimension_std": [float(v) for v in np.std(actions, axis=0)],
        "action_dimension_abs_p95": [
            float(v) for v in np.percentile(np.abs(actions), 95, axis=0)],
        "fraction_with_any_parameter_at_bound": float(clipped.any(axis=1).mean()),
        "fraction_at_bound_by_dimension": [float(v) for v in clipped.mean(axis=0)],
        "message_character_quantiles": _quantiles(chars),
        "rough_tokens_characters_divided_by_four": int(round(sum(chars) / 4.0)),
        "same_prompt_action_consistency": _prompt_consistency(rows),
    }


def _preference_stats(rows: Sequence[dict]) -> dict:
    states = {r["state_id"] for r in rows}
    hp = [(r["chosen"]["hardened"]["hard_pass"],
           r["rejected"]["hardened"]["hard_pass"]) for r in rows]
    preservation = [(r["chosen"]["topology_preserved_from_parent"],
                     r["rejected"]["topology_preserved_from_parent"]) for r in rows]
    tier_gap = [r["chosen"]["hardened_preference_tier"]
                - r["rejected"]["hardened_preference_tier"] for r in rows]
    same_tier_gap = [
        r["chosen"]["hardened_within_tier_value"]
        - r["rejected"]["hardened_within_tier_value"]
        for r in rows
        if r["chosen"]["hardened_preference_tier"]
        == r["rejected"]["hardened_preference_tier"]
    ]

    feasibility_pairs = []
    quality_pairs = []
    for row in rows:
        chosen, rejected = row["chosen"], row["rejected"]
        chosen_ok = bool(chosen["hardened"]["hard_pass"]
                         and chosen["topology_preserved_from_parent"])
        rejected_ok = bool(rejected["hardened"]["hard_pass"]
                           and rejected["topology_preserved_from_parent"])
        feasibility_pairs.append(chosen_ok and not rejected_ok)
        quality_pairs.append(
            chosen_ok and rejected_ok
            and float(chosen["hardened"]["fitness"])
            >= float(rejected["hardened"]["fitness"]) + 0.01)

    chosen_at_bound = np.asarray(
        [r["chosen"]["clipped_mask"] for r in rows], dtype=bool)
    rejected_at_bound = np.asarray(
        [r["rejected"]["clipped_mask"] for r in rows], dtype=bool)
    chars = [_message_chars(r["prompt_messages"])
             + len(r["chosen_response"]) + len(r["rejected_response"])
             for r in rows]
    return {
        "records": len(rows),
        "effective_parent_states": len(states),
        "weight_sum": float(sum(float(r["sample_weight"]) for r in rows)),
        "chosen_rejected_hard_pass_counts": _pair_counter(hp),
        "chosen_rejected_topology_preservation_counts": _pair_counter(preservation),
        "both_candidates_fail_hard_gates": int(sum(not a and not b for a, b in hp)),
        "feasible_preserved_vs_unacceptable_pairs": int(sum(feasibility_pairs)),
        "both_acceptable_quality_gap_ge_0.01_pairs": int(sum(quality_pairs)),
        "direct_policy_core_pairs": int(sum(feasibility_pairs) + sum(quality_pairs)),
        "chosen_changes_topology_while_rejected_preserves": int(sum(
            not a and b for a, b in preservation)),
        "tier_gap_quantiles": _quantiles(tier_gap),
        "same_tier_pairs": len(same_tier_gap),
        "same_tier_value_gap_quantiles": _quantiles(same_tier_gap),
        "same_tier_nonpositive_value_gap": int(sum(v <= 0.0 for v in same_tier_gap)),
        "chosen_fraction_with_any_parameter_at_bound": float(
            chosen_at_bound.any(axis=1).mean()),
        "chosen_fraction_at_bound_by_dimension": [
            float(v) for v in chosen_at_bound.mean(axis=0)],
        "rejected_fraction_with_any_parameter_at_bound": float(
            rejected_at_bound.any(axis=1).mean()),
        "message_character_quantiles": _quantiles(chars),
        "rough_tokens_characters_divided_by_four": int(round(sum(chars) / 4.0)),
    }


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    report = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(root),
        "manifest_config_fingerprint": manifest["yardstick"]["config_fingerprint"],
        "parameter_order": manifest["yardstick"]["parameter_names"],
        "splits": {},
    }
    state_ids: dict[str, set[str]] = {}
    count_matches = True
    group_weight_ok = True
    sft_weight_ok = True
    for split in SPLITS:
        groups = _rows(root / "groups" / f"{split}.jsonl")
        preferences = _rows(root / "preferences" / f"{split}.jsonl")
        sft = _rows(root / "sft" / f"{split}.jsonl")
        group_of = {g["group_id"]: g for g in groups}
        states = {g["state_id"] for g in groups}
        state_ids[split] = states
        group_weight = float(sum(float(g["sample_weight"]) for g in groups))
        group_weight_ok &= math.isclose(group_weight, len(states), abs_tol=1e-7)
        sft_states = {r["state_id"] for r in sft}
        sft_weight = float(sum(float(r["sample_weight"]) for r in sft))
        sft_weight_ok &= math.isclose(sft_weight, len(sft_states), abs_tol=1e-7)
        count_matches &= all(
            len(rows) == int(manifest["counts"][kind].get(split, 0))
            for kind, rows in (("groups", groups), ("preferences", preferences),
                               ("sft", sft)))
        report["splits"][split] = {
            "groups": {
                "records": len(groups), "effective_parent_states": len(states),
                "weight_sum": group_weight,
                "source_run_counts": _counter(g["source"]["run"] for g in groups),
            },
            "preferences": _preference_stats(preferences),
            "sft": _sft_stats(sft, group_of),
        }

    overlap = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap[f"{a}<->{b}"] = len(state_ids[a] & state_ids[b])
    report["checks"] = {
        "manifest_counts_match_files": bool(count_matches),
        "group_weight_sum_equals_effective_states": bool(group_weight_ok),
        "sft_weight_sum_equals_effective_states": bool(sft_weight_ok),
        "parent_state_overlap_between_splits": overlap,
        "no_parent_state_overlap_between_splits": not any(overlap.values()),
    }
    report["interpretation"] = [
        "SFT rows teach applied normalized actions; hard_pass is feasibility, not a minimum-"
        "quality guarantee.",
        "Direct policy-core preferences choose a hard-passing topology-preserved action over "
        "an unacceptable action, or separate two acceptable actions by >=0.01 fitness.",
        "A high same-prompt action conflict rate indicates multimodal search labels; retain "
        "sampling/ranking rather than interpreting token loss as regression accuracy.",
        "The token estimate is characters/4 and must be replaced by the selected base model's "
        "actual chat tokenizer before training.",
    ]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = audit(Path(args.root))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0 if all((
        report["checks"]["manifest_counts_match_files"],
        report["checks"]["group_weight_sum_equals_effective_states"],
        report["checks"]["sft_weight_sum_equals_effective_states"],
        report["checks"]["no_parent_state_overlap_between_splits"],
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
