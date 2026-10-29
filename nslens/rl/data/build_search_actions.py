"""Build same-parent action preferences from complete CMA-ME trace populations.

This dataset trains the local refinement policy to choose a useful *direction and scale*
without asking a language model to narrate an optimization loop. It does not claim that
fitness values from different historical experiments are comparable: ranking happens only
inside the ten siblings proposed from one emitter state.

    python -m nslens.rl.data.build_search_actions --runs runs \
      --out runs/rl_datasets/search_actions_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .schema import (SEARCH_GROUP_SCHEMA, SEARCH_PREFERENCE_SCHEMA,
                     validate_search_group, validate_search_preference)
from .trace_sources import TraceSource, compatible_sources


STATUS_TIER = {"invalid": 0, "none": 1, "improved": 2, "new": 3}
ACTION_SYSTEM = (
    "You propose one local move for a nine-parameter folded optic. Parameter order is "
    "ray_y,ang,cx,thick,fsag,bsag,tilt_a,tilt_b,n. Return exactly one "
    "<action>{\"normalized_delta\":[nine numbers]}</action> block and no prose."
)


def _hash(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(raw).hexdigest()


def _split_by_seed(sources: Sequence[TraceSource]) -> Dict[int, str]:
    """Hold complete random-search seeds apart, including across run families."""
    seeds = sorted({s.seed for s in sources})
    if len(seeds) >= 3:
        return {s: ("test_seed" if s == seeds[-1] else
                    "validation_seed" if s == seeds[-2] else "train") for s in seeds}
    if len(seeds) == 2:
        return {seeds[0]: "train", seeds[1]: "test_seed"}
    return {s: "train" for s in seeds}


def _within_status_value(row: dict) -> float:
    status = row.get("status")
    if status == "improved":
        return float(row.get("delta") or 0.0)
    if status in ("new", "none"):
        return float(row.get("fitness") or 0.0)
    return 0.0


def _candidate(row: dict, i: int, source: TraceSource) -> dict:
    parent = np.asarray(row["parent"], float)
    x = np.asarray(row["genotype"], float)
    lo, hi = np.asarray(source.lo), np.asarray(source.hi)
    out = {
        "candidate_index": i,
        "genotype": [round(float(v), 6) for v in x],
        "normalized_delta": [round(float(v), 6) for v in ((x - parent) / (hi - lo))],
        "clipped_mask": [int(abs(v - a) < 5e-7 or abs(v - b) < 5e-7)
                         for v, a, b in zip(x, lo, hi)],
        "status": str(row["status"]),
        "preference_tier": STATUS_TIER[str(row["status"])],
        "within_tier_value": _within_status_value(row),
    }
    for key in ("fitness", "delta", "bc"):
        if row.get(key) is not None:
            out[key] = row[key]
    info = row.get("info") or {}
    if info:
        out["observed"] = {k: info[k] for k in (
            "tir_count", "spot", "efficiency", "dom_signature", "thinness_T",
            "per_field", "worst_focthru") if k in info}
    return out


def _action_block(candidate: dict) -> str:
    payload = {"normalized_delta": candidate["normalized_delta"]}
    return "<action>" + json.dumps(payload, separators=(",", ":")) + "</action>"


def _policy_prompt(state: dict) -> List[dict]:
    compact = {
        "parent_normalized": state["parent_normalized"],
        "sigma": state["sigma"],
        "fov_half_angle_deg": max(abs(float(v)) for v in state["fields_deg"]),
        "focus_window_half_width": state["focus_window_half_width"],
        "field_aggregate": state["field_aggregate"],
        "archive_axes": state["archive_axes"],
    }
    return [{"role": "system", "content": ACTION_SYSTEM},
            {"role": "user", "content": json.dumps(compact, separators=(",", ":"))}]


def _group(source: TraceSource, rows: Sequence[dict], trace_sha: str,
           split: str) -> dict | None:
    if len(rows) != 10:
        return None
    parents = {tuple(r.get("parent") or ()) for r in rows}
    if len(parents) != 1 or any(r.get("status") not in STATUS_TIER for r in rows):
        return None
    if any(len(r.get("genotype") or ()) != len(source.parameter_names) for r in rows):
        return None
    candidates = [_candidate(r, i, source) for i, r in enumerate(rows)]
    ranking = sorted(range(len(candidates)), key=lambda i: (
        candidates[i]["preference_tier"], candidates[i]["within_tier_value"], -i),
        reverse=True)
    # An all-invalid population has no demonstrated recovery direction. A population with
    # identical effective labels provides no preference gradient either.
    values = {(c["preference_tier"], c["within_tier_value"]) for c in candidates}
    if candidates[ranking[0]]["status"] == "invalid" or len(values) < 2:
        return None
    parent = [float(v) for v in rows[0]["parent"]]
    lo, hi = np.asarray(source.lo), np.asarray(source.hi)
    group_id = _hash(source.trace_path, rows[0]["eval"], rows[-1]["eval"], parent)[:24]
    rec = {
        "schema_version": SEARCH_GROUP_SCHEMA,
        "group_id": group_id,
        "state_id": _hash("state", source.experiment_label, parent)[:24],
        # Recomputed after dataset selection so every repeated parent state has unit mass.
        "sample_weight": 1.0,
        "split": split,
        "source": {
            "run": source.run, "seed": source.seed,
            "experiment": source.experiment_label,
            "experiment_key": source.experiment_key,
            "trace": str(source.trace_path), "trace_sha256": trace_sha,
            "eval_first": int(rows[0]["eval"]), "eval_last": int(rows[-1]["eval"]),
            "emitter": int(rows[0]["emitter"]), "generation": int(rows[0]["gen"]),
        },
        "state": {
            "parameter_names": list(source.parameter_names),
            "parameter_bounds": [[a, b] for a, b in zip(source.lo, source.hi)],
            "parent": parent,
            "parent_normalized": [round(float(v), 6)
                                  for v in ((np.asarray(parent) - lo) / (hi - lo))],
            "sigma": float(rows[0]["sigma"]),
            "fields_deg": list(source.meta.get("fields") or [0.0]),
            "focus_window_half_width": float(source.meta.get("focus_w", 8.0)),
            "field_aggregate": source.meta.get("field_aggregate"),
            "param_scaling": source.meta.get("param_scaling"),
            "archive_axes": source.meta.get("bc_axes"),
            "archive_bounds": source.meta.get("bc_bounds"),
        },
        "candidates": candidates,
        "ranking": ranking,
        "best_candidate_index": ranking[0],
        "supervision": {
            "basis": "exact CMA-ME archive priority: new > improved > none > invalid; "
                     "fitness/delta breaks ties inside a tier",
            "comparable_scope": "this_group_only",
            "historical_absolute_quality_label": False,
        },
    }
    for rank, i in enumerate(ranking):
        candidates[i]["rank"] = rank
    rec["messages"] = _policy_prompt(rec["state"]) + [
        {"role": "assistant", "content": _action_block(candidates[ranking[0]])}]
    errors = validate_search_group(rec)
    if errors:
        raise ValueError(f"invalid search group {group_id}: {errors}")
    return rec


def _preferences(group: dict) -> List[dict]:
    ranking = group["ranking"]
    candidates = group["candidates"]
    chosen = candidates[ranking[0]]
    # The highest-ranked loser is the hard negative.
    rejected = candidates[ranking[1]]
    pairs = [("archive_rank", chosen, rejected)]
    # Also retain the closest failed sibling. This explicitly teaches the validity
    # boundary, and is different from a random easy negative.
    invalid = [c for c in candidates if c["status"] == "invalid"]
    if invalid:
        near = min(invalid, key=lambda c: sum(v * v for v in c["normalized_delta"]))
        if near["candidate_index"] != rejected["candidate_index"]:
            pairs.append(("validity_contrast", chosen, near))
    out = []
    for kind, win, lose in pairs:
        rec = {
            "schema_version": SEARCH_PREFERENCE_SCHEMA,
            "preference_id": f"{group['group_id']}:{kind}",
            "group_id": group["group_id"], "state_id": group["state_id"],
            "sample_weight": group["sample_weight"] / len(pairs),
            "split": group["split"],
            "preference_type": kind, "state": group["state"],
            "chosen": win, "rejected": lose,
            "prompt_messages": _policy_prompt(group["state"]),
            "chosen_response": _action_block(win),
            "rejected_response": _action_block(lose),
            "source": group["source"],
        }
        errors = validate_search_preference(rec)
        if errors:
            raise ValueError(f"invalid search preference: {errors}")
        out.append(rec)
    return out


def _read_groups(source: TraceSource, split: str) -> Tuple[List[dict], dict]:
    digest = hashlib.sha256()
    raw_rows: List[dict] = []
    groups: List[dict] = []
    status = Counter()
    malformed = 0
    # Hash and parse in one pass. Group boundaries are consecutive CMA ask() populations,
    # not (emitter, generation): a restarted emitter may reuse both identifiers.
    with source.trace_path.open("rb") as fh:
        for raw in fh:
            digest.update(raw)
            try:
                row = json.loads(raw)
            except Exception:
                malformed += 1
                continue
            raw_rows.append(row)
            status[row.get("status")] += 1
    trace_sha = digest.hexdigest()
    for start in range(0, len(raw_rows), 10):
        rec = _group(source, raw_rows[start:start + 10], trace_sha, split)
        if rec is not None:
            groups.append(rec)
    audit = {
        "trace": str(source.trace_path), "sha256": trace_sha,
        "rows": len(raw_rows), "malformed": malformed,
        "status": dict(status), "usable_groups": len(groups),
        "discarded_groups": (len(raw_rows) + 9) // 10 - len(groups),
    }
    return groups, audit


def _balanced(groups: Sequence[dict], cap_per_tier: int, seed: int) -> List[dict]:
    by = defaultdict(list)
    for g in groups:
        tier = g["candidates"][g["best_candidate_index"]]["status"]
        by[tier].append(g)
    out = []
    for tier, rows in by.items():
        rows = sorted(rows, key=lambda g: _hash("sample", seed, g["group_id"]))
        out.extend(rows[:cap_per_tier] if cap_per_tier > 0 else rows)
    return sorted(out, key=lambda g: g["group_id"])


def _drop_cross_split_design_leakage(groups_by_split: Dict[str, List[dict]]) \
        -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """Remove held-out groups that reuse any exact parent or candidate from an earlier split.

    Warm-start elites are shared across search seeds. A seed split alone therefore leaves a
    small but real memorization path. Training has priority, then validation, then test;
    affected held-out groups are discarded rather than reassigned to another random seed.
    """
    claimed = set()
    out: Dict[str, List[dict]] = defaultdict(list)
    dropped: Dict[str, int] = {}
    for split in ("train", "validation_seed", "test_seed"):
        n_drop = 0
        current_tokens = set()
        for group in groups_by_split.get(split, []):
            tokens = {tuple(group["state"]["parent"])}
            tokens.update(tuple(c["genotype"]) for c in group["candidates"])
            if claimed & tokens:
                n_drop += 1
                continue
            out[split].append(group)
            current_tokens.update(tokens)
        claimed.update(current_tokens)
        dropped[split] = n_drop
    return out, dropped


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def build(runs_root: Path, out_dir: Path, *, cap_per_tier: int = 5000,
          seed: int = 0) -> dict:
    sources, excluded = compatible_sources(runs_root)
    if not sources:
        raise RuntimeError("no trace sources match the current nine-parameter fold semantics")
    split_of = _split_by_seed(sources)
    all_groups = defaultdict(list)
    audits = []
    for i, source in enumerate(sources, 1):
        print(f"[actions] {i}/{len(sources)} {source.run}/seed{source.seed}", flush=True)
        groups, audit = _read_groups(source, split_of[source.seed])
        all_groups[split_of[source.seed]].extend(groups)
        audits.append(audit)

    all_groups, leakage_drops = _drop_cross_split_design_leakage(all_groups)

    counts, status_counts, pair_counts = {}, {}, {}
    effective_states, group_weight_sums = {}, {}
    seen_group_ids = set()
    for split, rows in all_groups.items():
        selected = _balanced(rows, cap_per_tier, seed)
        per_state = Counter(g["state_id"] for g in selected)
        for group in selected:
            group["sample_weight"] = 1.0 / per_state[group["state_id"]]
            errors = validate_search_group(group)
            if errors:
                raise ValueError(f"weighted search group invalid: {errors}")
        overlap = seen_group_ids & {r["group_id"] for r in selected}
        if overlap:
            raise AssertionError(f"group leakage across splits: {sorted(overlap)[:3]}")
        seen_group_ids.update(r["group_id"] for r in selected)
        pairs = [p for g in selected for p in _preferences(g)]
        counts[split] = _write_jsonl(out_dir / "groups" / f"{split}.jsonl", selected)
        pair_counts[split] = _write_jsonl(
            out_dir / "preferences" / f"{split}.jsonl", pairs)
        status_counts[split] = dict(Counter(
            g["candidates"][g["best_candidate_index"]]["status"] for g in selected))
        effective_states[split] = len(per_state)
        group_weight_sums[split] = sum(float(g["sample_weight"]) for g in selected)

    reason_counts = Counter(e["reason"] for e in excluded)
    manifest = {
        "schema_versions": [SEARCH_GROUP_SCHEMA, SEARCH_PREFERENCE_SCHEMA],
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(runs_root), "source_traces_included": len(sources),
        "source_traces_excluded": len(excluded),
        "exclusion_reasons": dict(reason_counts),
        "included_experiments": dict(Counter(s.experiment_key for s in sources)),
        "seed_split": {str(k): v for k, v in split_of.items()},
        "selection": {
            "population_size": 10,
            "complete_same_parent_groups_only": True,
            "all_invalid_groups_removed": True,
            "no_preference_gradient_groups_removed": True,
            "cap_per_best_status_per_split": cap_per_tier,
        },
        "group_counts": counts, "best_status_counts": status_counts,
        "preference_counts": pair_counts,
        "effective_parent_states": effective_states,
        "group_sample_weight_sums": group_weight_sums,
        "groups_dropped_for_exact_design_leakage": leakage_drops,
        "checks": {
            "source_seed_disjoint_splits": True,
            "group_id_disjoint_splits": True,
            "parent_and_candidate_designs_disjoint_splits": True,
            "every_record_validated": True,
            "cross_run_absolute_scores_compared": False,
        },
        "source_audit": audits,
        "excluded_sources": excluded,
        "limitations": [
            "Archive status is an optimizer supervision label, not canonical optical quality.",
            "Only registered relative-sag, normalized, nine-parameter fold traces are used.",
            "Random seed is the split unit; with three historical seeds each split is one seed.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/rl_datasets/search_actions_v1")
    ap.add_argument("--cap-per-tier", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.runs), Path(a.out),
                           cap_per_tier=a.cap_per_tier, seed=a.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
