"""Build leakage-safe fixed-environment proposer and refinement corpora.

Only dense-audited strict positives become proposer or refinement SFT targets.  Complete
fresh traces remain available as outcome/preference supervision, while legacy nine-gene
audit records are exported solely as negatives or ablation-only examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA = "nslens.fixed_learning_data.v1"
PARAMETERS = ("cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b")
SPLITS = ("train", "validation_env", "test_env")
PROPOSER_SYSTEM = (
    "Propose one six-parameter geometry for the fixed folded-optic environment. "
    "Return exactly one <design> JSON object with keys "
    "cx,thick,fsag,bsag,tilt_a,tilt_b and no prose."
)
ACTION_SYSTEM = (
    "Propose one local geometry move for the fixed folded-optic environment. "
    "Parameter order is cx,thick,fsag,bsag,tilt_a,tilt_b. Return exactly one "
    "<action>{\"normalized_delta\":[six numbers]}</action> block and no prose."
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(*values: Any) -> str:
    return hashlib.sha256("|".join(_canonical(v) for v in values).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    _atomic_text(path, "".join(_canonical(row) + "\n" for row in rows))
    return len(rows)


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_specs(path: Path) -> tuple[dict[str, dict], list[str]]:
    payload = json.loads(path.read_text())
    rows = [row for row in payload.get("environments", [])
            if row.get("status") == "accepted"]
    return {row["environment_id"]: row for row in rows}, [
        row["environment_id"] for row in rows]


def _split_map(environment_ids: list[str], validation: set[str], test: set[str]
               ) -> dict[str, str]:
    known = set(environment_ids)
    if validation & test:
        raise ValueError("validation and test environment sets overlap")
    missing = (validation | test) - known
    if missing:
        raise ValueError(f"unknown held-out environments: {sorted(missing)}")
    if not validation or not test:
        raise ValueError("at least one validation and one test environment are required")
    result = {environment_id: (
        "test_env" if environment_id in test else
        "validation_env" if environment_id in validation else "train")
        for environment_id in environment_ids}
    if not any(value == "train" for value in result.values()):
        raise ValueError("the split leaves no training environments")
    return result


def _environment_payload(spec: Mapping[str, Any]) -> dict:
    fixed = spec["fixed"]
    return {
        "environment_id": spec["environment_id"],
        "fixed": {key: fixed[key] for key in (
            "ray_y", "ang", "n", "x_sensor", "beam_width",
            "focus_window_half_width")},
        "intended_sequence": spec["intended_sequence"],
        "fields_deg": [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
        "parameter_names": list(PARAMETERS),
        "parameter_bounds": [spec["variables"][name] for name in PARAMETERS],
    }


def _design_block(design: Mapping[str, float]) -> str:
    payload = {name: float(design[name]) for name in PARAMETERS}
    return "<design>" + json.dumps(payload, separators=(",", ":")) + "</design>"


def _action_block(delta: list[float]) -> str:
    return "<action>" + json.dumps(
        {"normalized_delta": delta}, separators=(",", ":")) + "</action>"


def _proposer_records(strict_path: Path, specs: Mapping[str, dict],
                      split_of: Mapping[str, str]) -> tuple[dict[str, list[dict]], dict]:
    rows = [json.loads(line) for line in strict_path.open() if line.strip()]
    per_environment = Counter(row["fixed_environment_id"] for row in rows)
    output = defaultdict(list)
    strict_keys = defaultdict(dict)
    for row in rows:
        environment_id = row["fixed_environment_id"]
        spec = specs[environment_id]
        design = {name: float(row["design"][name]) for name in PARAMETERS}
        split = split_of[environment_id]
        evidence = {
            "audit_label": row["audit_label"],
            "dense_dominance_fraction": row["historical_dense_dominance_fraction"],
            "topology_field_persistence": row["global_topology_field_persistence"],
            "topology_perturbation_persistence":
                row["global_topology_perturbation_persistence"],
            "intended_energy_share_mean":
                row["historical_full_delivered_energy_share_mean"],
            "intended_energy_share_min":
                row["historical_full_delivered_energy_share_min"],
        }
        env_payload = _environment_payload(spec)
        record = {
            "schema_version": "nslens.fixed_proposer.v1",
            "dataset_kind": "strict_intentional_seed",
            "example_id": row["design_id"], "split": split,
            "sample_weight": 1.0 / per_environment[environment_id],
            "input": env_payload,
            "target": {"design": design}, "labels": evidence,
            "messages": [
                {"role": "system", "content": PROPOSER_SYSTEM},
                {"role": "user", "content": _canonical(env_payload)},
                {"role": "assistant", "content": _design_block(design)},
            ],
            "source": {
                "dense_audit": str(strict_path),
                "baseline_sources": row["source_provenance"]["sources"],
            },
        }
        output[split].append(record)
        strict_keys[environment_id][tuple(round(design[name], 6)
                                          for name in PARAMETERS)] = row
    for split in output:
        output[split].sort(key=lambda row: row["example_id"])
    return output, strict_keys


def _rank_candidates(candidates: list[dict]) -> list[int]:
    return sorted(range(len(candidates)), key=lambda index: (
        bool(candidates[index]["hard_pass"]),
        float(candidates[index].get("fitness") or 0.0), -index), reverse=True)


def _candidate(row: Mapping[str, Any], parent: np.ndarray, lo: np.ndarray,
               hi: np.ndarray, index: int) -> dict:
    genotype = np.asarray(row["genotype"], float)
    result = {
        "candidate_index": index,
        "genotype": [float(value) for value in genotype],
        "normalized_delta": [round(float(value), 6)
                             for value in ((genotype - parent) / (hi - lo))],
        "status": row["status"], "hard_pass": row["status"] != "invalid",
    }
    if row.get("fitness") is not None:
        result["fitness"] = float(row["fitness"])
    if row.get("bc") is not None:
        result["bc"] = row["bc"]
    if row.get("rejection"):
        result["rejection"] = row["rejection"]
    if row.get("info"):
        result["observed"] = row["info"]
    return result


def _refinement_data(baseline: Mapping[str, Any], specs: Mapping[str, dict],
                     split_of: Mapping[str, str], strict_keys: Mapping[str, dict]
                     ) -> tuple[dict[str, list[dict]], dict[str, list[dict]],
                                dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    groups = defaultdict(list)
    preferences = defaultdict(list)
    sft = defaultdict(list)
    outcomes = defaultdict(list)
    sources = []
    group_counts = Counter()

    for run in baseline["runs"]:
        environment_id = run["environment_id"]
        split = split_of[environment_id]
        trace_path = Path(run["trace"]).resolve()
        if _sha256(trace_path) != run["trace_sha256"]:
            raise ValueError(f"trace checksum mismatch: {trace_path}")
        spec = specs[environment_id]
        env_payload = _environment_payload(spec)
        bounds = np.asarray(env_payload["parameter_bounds"], float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        raw = [json.loads(line) for line in trace_path.open() if line.strip()]
        sources.append({
            "environment_id": environment_id, "algorithm": run["algorithm"],
            "seed": run["seed"], "trace": str(trace_path),
            "trace_sha256": run["trace_sha256"], "rows": len(raw),
        })
        for row in raw:
            outcome = {
                "schema_version": "nslens.fixed_outcome.v1",
                "outcome_id": _hash(environment_id, run["algorithm"], run["seed"],
                                    row["eval"], row["genotype"])[:24],
                "split": split, "sample_weight": 1.0,
                "environment": env_payload,
                "genotype": row["genotype"], "status": row["status"],
                "hard_pass": row["status"] != "invalid",
                "fitness": row.get("fitness"), "rejection": row.get("rejection"),
                "source": {"algorithm": run["algorithm"], "seed": run["seed"],
                           "eval": row["eval"], "trace_sha256": run["trace_sha256"]},
            }
            outcomes[split].append(outcome)

        if run["algorithm"] != "cma":
            continue
        if len(raw) % 10:
            raise ValueError(f"CMA trace is not a whole number of populations: {trace_path}")
        for start in range(0, len(raw), 10):
            population = raw[start:start + 10]
            parents = {tuple(row.get("parent") or ()) for row in population}
            if len(parents) != 1:
                raise ValueError(f"mixed parents in CMA population at {trace_path}:{start}")
            parent = np.asarray(population[0]["parent"], float)
            candidates = [_candidate(row, parent, lo, hi, index)
                          for index, row in enumerate(population)]
            ranking = _rank_candidates(candidates)
            if not candidates[ranking[0]]["hard_pass"]:
                continue
            group_id = _hash(environment_id, run["seed"], start, parent.tolist())[:24]
            state = {
                "environment": env_payload,
                "parent": [float(value) for value in parent],
                "parent_normalized": [round(float(value), 6)
                                      for value in ((parent - lo) / (hi - lo))],
                "sigma": float(population[0]["sigma"]),
            }
            for rank, index in enumerate(ranking):
                candidates[index]["rank"] = rank
            prompt = [
                {"role": "system", "content": ACTION_SYSTEM},
                {"role": "user", "content": _canonical(state)},
            ]
            group = {
                "schema_version": "nslens.fixed_refinement_group.v1",
                "group_id": group_id, "state_id": _hash(environment_id, parent.tolist())[:24],
                "split": split, "sample_weight": 1.0,
                "state": state, "candidates": candidates, "ranking": ranking,
                "best_candidate_index": ranking[0],
                "source": {"trace": str(trace_path), "trace_sha256": run["trace_sha256"],
                           "seed": run["seed"], "eval_first": population[0]["eval"],
                           "eval_last": population[-1]["eval"]},
            }
            groups[split].append(group)
            group_counts[environment_id] += 1

            chosen = candidates[ranking[0]]
            invalid = [candidate for candidate in candidates if not candidate["hard_pass"]]
            if invalid:
                rejected = min(invalid, key=lambda candidate: sum(
                    value * value for value in candidate["normalized_delta"]))
                preference_type = "validity_contrast"
            elif len(candidates) > 1:
                rejected = candidates[ranking[1]]
                preference_type = "fitness_contrast"
            else:
                rejected = None
            if rejected is not None and chosen["candidate_index"] != rejected["candidate_index"]:
                preferences[split].append({
                    "schema_version": "nslens.fixed_refinement_preference.v1",
                    "preference_id": f"{group_id}:{preference_type}",
                    "group_id": group_id, "state_id": group["state_id"], "split": split,
                    "sample_weight": 1.0, "preference_type": preference_type,
                    "state": state, "chosen": chosen, "rejected": rejected,
                    "prompt_messages": prompt,
                    "chosen_response": _action_block(chosen["normalized_delta"]),
                    "rejected_response": _action_block(rejected["normalized_delta"]),
                })

            strict_matches = []
            for candidate in candidates:
                key = tuple(round(float(value), 6) for value in candidate["genotype"])
                audit = strict_keys.get(environment_id, {}).get(key)
                if audit is not None and candidate["status"] in {"new", "improved"}:
                    strict_matches.append((float(candidate.get("fitness") or 0.0),
                                           candidate, audit))
            if strict_matches:
                _fitness, demonstration, audit = max(strict_matches, key=lambda item: item[0])
                sft[split].append({
                    "schema_version": "nslens.fixed_refinement_sft.v1",
                    "example_id": f"{group_id}:strict", "group_id": group_id,
                    "state_id": group["state_id"], "split": split,
                    "sample_weight": 1.0, "state": state,
                    "messages": prompt + [{"role": "assistant", "content":
                                           _action_block(demonstration["normalized_delta"])}],
                    "target": demonstration,
                    "labels": {"dense_strict_positive": True,
                               "audit_design_id": audit["design_id"]},
                })

    for corpus in (groups, preferences, sft):
        for split, rows in corpus.items():
            by_environment = Counter(row["state"]["environment"]["environment_id"]
                                     for row in rows)
            for row in rows:
                environment_id = row["state"]["environment"]["environment_id"]
                row["sample_weight"] = 1.0 / by_environment[environment_id]
            rows.sort(key=lambda row: row.get("group_id", row.get("example_id")))
    for split in outcomes:
        by_environment = Counter(row["environment"]["environment_id"]
                                 for row in outcomes[split])
        for row in outcomes[split]:
            row["sample_weight"] = 1.0 / by_environment[
                row["environment"]["environment_id"]]
        outcomes[split].sort(key=lambda row: row["outcome_id"])
    return groups, preferences, sft, outcomes, sources


def _legacy_records(path: Path, cap_per_label: int) -> tuple[list[dict], list[dict]]:
    by_label = defaultdict(list)
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            by_label[row["audit_label"]].append(row)
    negatives, ablations = [], []
    for label, rows in sorted(by_label.items()):
        rows.sort(key=lambda row: _hash("legacy", row["design_id"]))
        selected = rows[:cap_per_label] if cap_per_label > 0 else rows
        for row in selected:
            compact = {
                "schema_version": "nslens.legacy_interaction_example.v1",
                "example_id": row["design_id"], "legacy_label": label,
                "legacy_nine_dof_design": row.get("design"),
                "historical_sequence": row.get("historical_sequence"),
                "reasons": row.get("reasons", []),
                "training_role": ("ablation_only" if label == "intentional"
                                  else "negative_only"),
                "usable_as_fixed_geometry_positive": False,
                "source": str(path),
            }
            (ablations if label == "intentional" else negatives).append(compact)
    return negatives, ablations


def build(args) -> dict:
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    baseline_manifest_path = Path(args.baseline_manifest).resolve()
    strict_path = Path(args.strict_positives).resolve()
    specs_path = Path(args.environment_specs).resolve()
    legacy_path = Path(args.legacy_audit_records).resolve()
    baseline = json.loads(baseline_manifest_path.read_text())
    specs, environment_ids = _load_specs(specs_path)
    split_of = _split_map(environment_ids, _csv(args.validation_environments),
                          _csv(args.test_environments))
    proposer, strict_keys = _proposer_records(strict_path, specs, split_of)
    groups, preferences, sft, outcomes, sources = _refinement_data(
        baseline, specs, split_of, strict_keys)
    negatives, ablations = _legacy_records(legacy_path, args.legacy_cap_per_label)

    output.mkdir(parents=True)
    counts = {"proposer": {}, "groups": {}, "preferences": {}, "sft": {},
              "outcomes": {}}
    for name, corpus in (("proposer", proposer), ("groups", groups),
                         ("preferences", preferences), ("sft", sft),
                         ("outcomes", outcomes)):
        for split in SPLITS:
            counts[name][split] = _write_jsonl(
                output / name / f"{split}.jsonl", corpus.get(split, []))
    counts["legacy_negative_only"] = _write_jsonl(
        output / "legacy" / "negative_only.jsonl", negatives)
    counts["legacy_ablation_only"] = _write_jsonl(
        output / "legacy" / "ablation_only.jsonl", ablations)

    checks = {
        "environment_disjoint_splits": (
            set(split_of) == set(environment_ids)
            and all(split in SPLITS for split in split_of.values())),
        "strict_positive_only_proposer": all(
            row["labels"]["audit_label"] == "intentional"
            for rows in proposer.values() for row in rows),
        "strict_positive_only_refinement_sft": all(
            row["labels"]["dense_strict_positive"] is True
            for rows in sft.values() for row in rows),
        "legacy_never_fixed_positive": all(
            not row["usable_as_fixed_geometry_positive"]
            for row in negatives + ablations),
        "all_fresh_outcomes_retained": sum(map(len, outcomes.values())) == sum(
            int(run["evals"]) for run in baseline["runs"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"dataset invariant failed: {checks}")
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_unit": "fixed_environment_id",
        "environment_split": split_of, "counts": counts, "checks": checks,
        "sources": {
            "baseline_manifest": str(baseline_manifest_path),
            "baseline_manifest_sha256": _sha256(baseline_manifest_path),
            "strict_positives": str(strict_path),
            "strict_positives_sha256": _sha256(strict_path),
            "environment_specs": str(specs_path),
            "environment_specs_sha256": _sha256(specs_path),
            "legacy_audit_records": str(legacy_path),
            "legacy_audit_records_sha256": _sha256(legacy_path),
            "traces": sources,
        },
        "weighting": "each fixed environment contributes unit total weight per corpus",
        "limitations": [
            "The fresh search pilot has one seed; generalization is tested by whole environment.",
            "Refinement SFT is intentionally small because only dense-audited final elites qualify.",
            "Legacy nine-DOF records are incompatible with this fixed six-DOF policy and are "
            "exported only as negative or ablation data.",
            "All quality labels are local ray-engine labels; CODE V remains asynchronous authority.",
        ],
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_text(output / "README.md", (
        "# Fixed-environment learning data v1\n\n"
        "Whole fixed environments are the split unit. Proposer and refinement SFT targets "
        "come only from `strict_intentional.jsonl`; complete fresh CMA/random outcomes are "
        "retained for feasibility/ranking models. Legacy nine-DOF examples are isolated "
        "under `legacy/` and must never be used as fixed-geometry positives.\n"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--strict-positives", required=True)
    parser.add_argument("--environment-specs", required=True)
    parser.add_argument("--legacy-audit-records", required=True)
    parser.add_argument("--validation-environments", required=True,
                        help="comma-separated fixed environment IDs")
    parser.add_argument("--test-environments", required=True,
                        help="comma-separated fixed environment IDs")
    parser.add_argument("--legacy-cap-per-label", type=int, default=36)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.legacy_cap_per_label < 0:
        parser.error("--legacy-cap-per-label must be non-negative")
    manifest = build(args)
    print(json.dumps({"output": str(Path(args.out).resolve()),
                      "counts": manifest["counts"],
                      "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
