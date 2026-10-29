"""Dependency-free validation for proposer and future CODE V refinement records.

The validators enforce the load-bearing invariants and return every error in one pass,
which is friendlier to a remote logging harness than failing on the first missing field.
"""
from __future__ import annotations

import math
import string
from typing import Any, List


PROPOSER_SCHEMA = "nslens.proposer.v1"
CODEV_TRANSITION_SCHEMA = "nslens.codev_transition.v1"
SEARCH_GROUP_SCHEMA = "nslens.search_action_group.v1"
SEARCH_PREFERENCE_SCHEMA = "nslens.search_preference.v1"
HARDENED_DESIGN_SCHEMA = "nslens.hardened_design.v1"
HARDENED_ACTION_SFT_SCHEMA = "nslens.hardened_action_sft.v1"
OPTIMIZER_TRANSITION_SCHEMA = "nslens.optimizer_transition.v1"

SEARCH_STATUSES = ("invalid", "none", "improved", "new")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _number_list(value: Any) -> bool:
    return isinstance(value, list) and all(_finite(v) for v in value)


def _matrix(value: Any, rows: int, columns: int) -> bool:
    return (isinstance(value, list) and len(value) == rows
            and all(_number_list(row) and len(row) == columns for row in value))


def _sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in string.hexdigits for character in value))


def validate_optimizer_transition(rec: dict) -> List[str]:
    """Validate one complete emitter generation without importing JAX or the engine."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != OPTIMIZER_TRANSITION_SCHEMA:
        errors.append(f"schema_version must be {OPTIMIZER_TRANSITION_SCHEMA!r}")
    for key in ("transition_id", "lifecycle_id", "task", "representation"):
        if not rec.get(key):
            errors.append(f"{key} is required")
    fingerprints = rec.get("fingerprints") or {}
    for key in ("simulator", "objective", "archive", "representation"):
        value = fingerprints.get(key)
        if not _sha256(value):
            errors.append(f"fingerprints.{key} must be a SHA-256 hex digest")
    context = rec.get("context") or {}
    if any(not isinstance(context.get(key), dict)
           for key in ("representation", "objective", "archive")):
        errors.append("context must contain representation, objective, and archive objects")
    elif context["representation"].get("id") != rec.get("representation"):
        errors.append("context.representation.id must agree with representation")
    if rec.get("emitter_algorithm") != "full_covariance_cma_es":
        errors.append("emitter_algorithm must be 'full_covariance_cma_es'")
    if not isinstance(rec.get("emitter_index"), int) or rec.get("emitter_index", -1) < 0:
        errors.append("emitter_index must be a non-negative integer")
    if not isinstance(rec.get("generation"), int) or rec.get("generation", -1) < 0:
        errors.append("generation must be a non-negative integer")
    if not isinstance(rec.get("seed"), int):
        errors.append("seed must be an integer")
    if not isinstance(rec.get("remaining_budget"), int) \
            or rec.get("remaining_budget", -1) < 0:
        errors.append("remaining_budget must be a non-negative integer")

    names = rec.get("parameter_names")
    if not isinstance(names, list) or not names or not all(
            isinstance(name, str) and name for name in names):
        errors.append("parameter_names must be a non-empty string list")
        dimensions = -1
    else:
        dimensions = len(names)
    bounds = rec.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != dimensions or any(
            not _number_list(bound) or len(bound) != 2 or bound[0] >= bound[1]
            for bound in bounds or []):
        errors.append(f"bounds must contain {dimensions} increasing finite pairs")

    before, after = rec.get("state_before") or {}, rec.get("state_after") or {}
    for label, state in (("state_before", before), ("state_after", after)):
        if state.get("algorithm") != "full_covariance_cma_es":
            errors.append(f"{label}.algorithm must be full_covariance_cma_es")
        if not isinstance(state.get("generation"), int) or state.get("generation", -1) < 0:
            errors.append(f"{label}.generation must be a non-negative integer")
        if state.get("population_size") != before.get("population_size"):
            errors.append(f"{label}.population_size must match state_before")
        if not _number_list(state.get("mean")) or len(state.get("mean") or []) != dimensions:
            errors.append(f"{label}.mean must contain {dimensions} finite numbers")
        if not _finite(state.get("sigma")) or float(state.get("sigma", 0)) <= 0:
            errors.append(f"{label}.sigma must be positive finite numeric")
        for key in ("evolution_path_c", "evolution_path_sigma", "axis_scales"):
            if not _number_list(state.get(key)) or len(state.get(key) or []) != dimensions:
                errors.append(f"{label}.{key} must contain {dimensions} finite numbers")
        for key in ("covariance", "eigenvectors", "inverse_sqrt_covariance"):
            matrix = state.get(key)
            if not isinstance(matrix, list) or len(matrix) != dimensions or any(
                    not _number_list(row) or len(row) != dimensions for row in matrix or []):
                errors.append(f"{label}.{key} must be a finite {dimensions}x{dimensions} matrix")
        if not isinstance(state.get("rng_state"), dict):
            errors.append(f"{label}.rng_state is required")
    if isinstance(before.get("generation"), int) and isinstance(after.get("generation"), int) \
            and after["generation"] != before["generation"] + 1:
        errors.append("state_after.generation must equal state_before.generation + 1")

    candidates = rec.get("candidates")
    population_size = before.get("population_size")
    if not isinstance(population_size, int) or population_size <= 0:
        errors.append("state_before.population_size must be a positive integer")
        population_size = 0
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates must be a non-empty list")
        candidates = []
    elif isinstance(population_size, int) and len(candidates) != population_size:
        errors.append("candidate count must match state_before.population_size")
    ranks = []
    for index, candidate in enumerate(candidates):
        if candidate.get("candidate_index") != index:
            errors.append(f"candidates[{index}].candidate_index must equal {index}")
        if candidate.get("status") not in SEARCH_STATUSES:
            errors.append(f"candidates[{index}].status is unknown")
        for key in ("search_coordinates", "genotype"):
            if not _number_list(candidate.get(key)) \
                    or len(candidate.get(key) or []) != dimensions:
                errors.append(f"candidates[{index}].{key} must contain {dimensions} numbers")
        mask = candidate.get("clipping_mask")
        if not isinstance(mask, list) or len(mask) != dimensions \
                or any(value not in (0, 1, False, True) for value in mask):
            errors.append(f"candidates[{index}].clipping_mask must be a length-{dimensions} mask")
        rank = candidate.get("rank")
        if not isinstance(rank, int):
            errors.append(f"candidates[{index}].rank must be an integer")
        else:
            ranks.append(rank)
            percentile = candidate.get("rank_percentile")
            expected = rank / max(len(candidates) - 1, 1)
            if not _finite(percentile) or abs(float(percentile) - expected) > 1e-7:
                errors.append(
                    f"candidates[{index}].rank_percentile must agree with rank")
        if candidate.get("status") == "invalid":
            failure = candidate.get("failure") or {}
            if not failure.get("reason"):
                errors.append(f"candidates[{index}] invalid result requires failure.reason")
        elif not _finite(candidate.get("fitness")):
            errors.append(f"candidates[{index}] valid result requires finite fitness")
    if ranks and sorted(ranks) != list(range(len(candidates))):
        errors.append("candidate ranks must be a permutation of population ranks")

    ranking = rec.get("ranking_order")
    if not isinstance(ranking, list) or sorted(ranking) != list(range(len(candidates))):
        errors.append("ranking_order must be a candidate-index permutation")
    elif all(isinstance(c.get("rank"), int) for c in candidates):
        if any(candidates[candidate_index]["rank"] != rank
               for rank, candidate_index in enumerate(ranking)):
            errors.append("candidate ranks must agree with ranking_order")

    sampling = rec.get("sampling") or {}
    if not sampling.get("sample_id"):
        errors.append("sampling.sample_id is required")
    for key in ("standard_normal", "transformed_steps", "raw_population", "population"):
        if not _matrix(sampling.get(key), population_size, dimensions):
            errors.append(
                f"sampling.{key} must be a finite {population_size}x{dimensions} matrix")
    sampling_mask = sampling.get("clipping_mask")
    if not (isinstance(sampling_mask, list) and len(sampling_mask) == population_size
            and all(isinstance(row, list) and len(row) == dimensions
                    and all(value in (0, 1, False, True) for value in row)
                    for row in sampling_mask)):
        errors.append(
            f"sampling.clipping_mask must be a {population_size}x{dimensions} mask")
    elif len(candidates) == population_size:
        if any(candidate.get("clipping_mask") != sampling_mask[index]
               for index, candidate in enumerate(candidates)):
            errors.append("candidate clipping masks must agree with sampling.clipping_mask")
    if _matrix(sampling.get("population"), population_size, dimensions) \
            and len(candidates) == population_size:
        if any(candidate.get("search_coordinates") != sampling["population"][index]
               for index, candidate in enumerate(candidates)):
            errors.append(
                "candidate search_coordinates must agree with sampling.population")
    for label in ("archive_before", "archive_after"):
        archive = rec.get(label) or {}
        if not isinstance(archive.get("filled"), int) or archive.get("filled", -1) < 0:
            errors.append(f"{label}.filled must be non-negative integer")
        if not _finite(archive.get("coverage")) \
                or not 0 <= float(archive.get("coverage", -1)) <= 1:
            errors.append(f"{label}.coverage must lie in [0, 1]")
    restart = rec.get("restart") or {}
    if not isinstance(restart.get("triggered"), bool):
        errors.append("restart.triggered must be boolean")
    if restart.get("triggered") and not restart.get("reason"):
        errors.append("a triggered restart requires restart.reason")
    return errors


def validate_proposer(rec: dict) -> List[str]:
    """Return validation errors for one Model-A JSONL record."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != PROPOSER_SCHEMA:
        errors.append(f"schema_version must be {PROPOSER_SCHEMA!r}")
    if not rec.get("example_id"):
        errors.append("example_id is required")
    if rec.get("dataset_kind") not in (
            "format_validity", "directed_seed", "unconditioned_archive_seed"):
        errors.append("unknown dataset_kind")
    if rec.get("split") not in (
            "train", "validation_seen", "test_cell", "test_env", "test_format"):
        errors.append("unknown split")
    if "sample_weight" in rec and (not _finite(rec["sample_weight"])
                                    or float(rec["sample_weight"]) <= 0):
        errors.append("sample_weight must be positive finite numeric")

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs
                                      if isinstance(m, dict)] != [
                                          "system", "user", "assistant"]:
        errors.append("messages must contain system, user, assistant in that order")
    target = rec.get("target") or {}
    design = target.get("design")
    inputs = rec.get("input") or {}
    bounds = inputs.get("parameter_bounds")
    if not isinstance(design, dict) or not design:
        errors.append("target.design must be a non-empty object")
    if not isinstance(bounds, dict) or not bounds:
        errors.append("input.parameter_bounds must be a non-empty object")
    if isinstance(design, dict) and isinstance(bounds, dict):
        if set(design) != set(bounds):
            errors.append("design keys must exactly match parameter_bounds keys")
        for name, value in design.items():
            b = bounds.get(name)
            if not _finite(value):
                errors.append(f"design.{name} is not finite numeric")
            elif not (_number_list(b) and len(b) == 2 and b[0] <= value <= b[1]):
                errors.append(f"design.{name} is outside or missing its bound")

    if rec.get("dataset_kind") == "directed_seed":
        labels = rec.get("labels") or {}
        if not isinstance(labels.get("topology_index"), int):
            errors.append("directed_seed requires labels.topology_index")
        if not labels.get("full_signature"):
            errors.append("directed_seed requires labels.full_signature")
    return errors


def validate_codev_transition(rec: dict) -> List[str]:
    """Return validation errors for one future CODE V refinement transition."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != CODEV_TRANSITION_SCHEMA:
        errors.append(f"schema_version must be {CODEV_TRANSITION_SCHEMA!r}")
    for key in ("transition_id", "episode_id", "design_id"):
        if not rec.get(key):
            errors.append(f"{key} is required")
    if not isinstance(rec.get("step"), int) or rec.get("step", -1) < 0:
        errors.append("step must be a non-negative integer")

    state, action, result = (rec.get(k) or {} for k in ("state", "action", "result"))
    names = state.get("parameter_names")
    values = state.get("parameters")
    scales = state.get("parameter_scales")
    delta = action.get("normalized_delta")
    mask = state.get("active_mask")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        errors.append("state.parameter_names must be a non-empty string list")
    n = len(names) if isinstance(names, list) else -1
    for label, seq in (("state.parameters", values),
                       ("state.parameter_scales", scales),
                       ("action.normalized_delta", delta)):
        if not _number_list(seq) or len(seq) != n:
            errors.append(f"{label} must be a finite numeric list of length {n}")
    if not isinstance(mask, list) or len(mask) != n or any(v not in (0, 1) for v in mask):
        errors.append(f"state.active_mask must be a 0/1 list of length {n}")
    if _number_list(scales) and any(float(v) <= 0 for v in scales):
        errors.append("state.parameter_scales must be positive")

    if not _finite(state.get("merit_before")):
        errors.append("state.merit_before must be finite numeric")
    for key in ("merit_after", "delta_merit"):
        if not _finite(result.get(key)):
            errors.append(f"result.{key} must be finite numeric")
    if not isinstance(result.get("trace_ok"), bool):
        errors.append("result.trace_ok must be boolean")
    if not isinstance(result.get("accepted"), bool):
        errors.append("result.accepted must be boolean")
    if result.get("accepted") is True and result.get("trace_ok") is not True:
        errors.append("an accepted transition cannot have trace_ok=false")
    if _finite(state.get("merit_before")) and _finite(result.get("merit_after")) \
            and _finite(result.get("delta_merit")):
        # Convention: lower CODE V merit is better, so delta is before - after.
        expected = float(state["merit_before"]) - float(result["merit_after"])
        if abs(expected - float(result["delta_merit"])) > 1e-7:
            errors.append("delta_merit must equal merit_before - merit_after")
    provenance = rec.get("provenance") or {}
    for key in ("codev_version", "merit_function_version", "translator_version"):
        if not provenance.get(key):
            errors.append(f"provenance.{key} is required")
    return errors


def validate_search_group(rec: dict) -> List[str]:
    """Validate one same-parent CMA-ME candidate group.

    The group boundary is load-bearing: candidates can only be ranked against siblings
    proposed from the same emitter state. Historical fitness values from different runs
    are intentionally never compared here.
    """
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != SEARCH_GROUP_SCHEMA:
        errors.append(f"schema_version must be {SEARCH_GROUP_SCHEMA!r}")
    if not rec.get("group_id"):
        errors.append("group_id is required")
    if not rec.get("state_id"):
        errors.append("state_id is required")
    if not _finite(rec.get("sample_weight")) or float(rec.get("sample_weight", 0)) <= 0:
        errors.append("sample_weight must be positive finite numeric")
    if rec.get("split") not in ("train", "validation_seed", "test_seed"):
        errors.append("unknown split")
    state = rec.get("state") or {}
    names = state.get("parameter_names")
    parent = state.get("parent")
    bounds = state.get("parameter_bounds")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        errors.append("state.parameter_names must be a non-empty string list")
    n = len(names) if isinstance(names, list) else -1
    if not _number_list(parent) or len(parent) != n:
        errors.append(f"state.parent must be a finite numeric list of length {n}")
    if not isinstance(bounds, list) or len(bounds) != n or any(
            not _number_list(b) or len(b) != 2 or b[0] >= b[1] for b in bounds):
        errors.append(f"state.parameter_bounds must contain {n} increasing pairs")
    candidates = rec.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        errors.append("candidates must contain at least two siblings")
        candidates = []
    for i, cand in enumerate(candidates):
        if cand.get("candidate_index") != i:
            errors.append(f"candidates[{i}].candidate_index must equal {i}")
        if cand.get("status") not in SEARCH_STATUSES:
            errors.append(f"candidates[{i}].status is unknown")
        genotype = cand.get("genotype")
        delta = cand.get("normalized_delta")
        if not _number_list(genotype) or len(genotype) != n:
            errors.append(f"candidates[{i}].genotype must have length {n}")
        if not _number_list(delta) or len(delta) != n:
            errors.append(f"candidates[{i}].normalized_delta must have length {n}")
        if _number_list(parent) and _number_list(genotype) and _number_list(delta) \
                and isinstance(bounds, list) and len(bounds) == n:
            for j, (p, x, d, b) in enumerate(zip(parent, genotype, delta, bounds)):
                if _number_list(b) and len(b) == 2:
                    expected = (float(x) - float(p)) / (float(b[1]) - float(b[0]))
                    if abs(expected - float(d)) > 3e-5:
                        errors.append(
                            f"candidates[{i}].normalized_delta[{j}] disagrees with "
                            "(genotype-parent)/range")
                        break
    ranking = rec.get("ranking")
    if not isinstance(ranking, list) or sorted(ranking) != list(range(len(candidates))):
        errors.append("ranking must be a permutation of candidate indices")
    elif rec.get("best_candidate_index") != ranking[0]:
        errors.append("best_candidate_index must equal ranking[0]")
    source = rec.get("source") or {}
    if not source.get("trace_sha256") or not source.get("experiment"):
        errors.append("source experiment and trace_sha256 are required")
    messages = rec.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages
                                          if isinstance(m, dict)] != [
                                              "system", "user", "assistant"]:
        errors.append("messages must contain system, user, assistant in that order")
    elif not (messages[-1].get("content", "").startswith("<action>")
              and messages[-1].get("content", "").endswith("</action>")):
        errors.append("assistant must contain one action block")
    return errors


def validate_search_preference(rec: dict) -> List[str]:
    """Validate a compact chosen/rejected view derived from a search group."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != SEARCH_PREFERENCE_SCHEMA:
        errors.append(f"schema_version must be {SEARCH_PREFERENCE_SCHEMA!r}")
    for key in ("preference_id", "group_id", "state", "chosen", "rejected"):
        if key not in rec:
            errors.append(f"{key} is required")
    if not rec.get("state_id"):
        errors.append("state_id is required")
    if not _finite(rec.get("sample_weight")) or float(rec.get("sample_weight", 0)) <= 0:
        errors.append("sample_weight must be positive finite numeric")
    if rec.get("split") not in ("train", "validation_seed", "test_seed"):
        errors.append("unknown split")
    chosen, rejected = rec.get("chosen") or {}, rec.get("rejected") or {}
    if chosen.get("candidate_index") == rejected.get("candidate_index"):
        errors.append("chosen and rejected must be different candidates")
    if rec.get("preference_type") not in ("archive_rank", "validity_contrast"):
        errors.append("unknown preference_type")
    if isinstance(chosen.get("rank"), int) and isinstance(rejected.get("rank"), int):
        if chosen["rank"] >= rejected["rank"]:
            errors.append("chosen.rank must be better (smaller) than rejected.rank")
    else:
        errors.append("chosen and rejected require integer ranks")
    prompt = rec.get("prompt_messages")
    if not isinstance(prompt, list) or [m.get("role") for m in prompt
                                        if isinstance(m, dict)] != ["system", "user"]:
        errors.append("prompt_messages must contain system and user")
    for key in ("chosen_response", "rejected_response"):
        value = rec.get(key)
        if not isinstance(value, str) or not value.startswith("<action>") \
                or not value.endswith("</action>"):
            errors.append(f"{key} must be an action block")
    if rec.get("chosen_response") == rec.get("rejected_response"):
        errors.append("chosen_response and rejected_response must differ")
    return errors


def validate_hardened_design(rec: dict) -> List[str]:
    """Validate a design re-scored under the single canonical hardened environment."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != HARDENED_DESIGN_SCHEMA:
        errors.append(f"schema_version must be {HARDENED_DESIGN_SCHEMA!r}")
    if not rec.get("design_id"):
        errors.append("design_id is required")
    if rec.get("split") not in ("train", "validation_topology", "test_topology",
                                "rejected"):
        errors.append("unknown split")
    design, bounds = rec.get("design"), rec.get("parameter_bounds")
    if not isinstance(design, dict) or not design:
        errors.append("design must be a non-empty object")
    if not isinstance(bounds, dict) or set(bounds) != set(design or {}):
        errors.append("parameter_bounds must match design keys")
    verdict = rec.get("hardened_verdict") or {}
    if verdict.get("stage") not in (
            "degenerate", "bulb_face", "detector_intersect", "embedded_source",
            "engine_error", "no_light", "off_class", "scored"):
        errors.append("unknown hardened_verdict.stage")
    if verdict.get("stage") == "scored":
        if not _finite(verdict.get("fitness")):
            errors.append("scored design requires finite fitness")
        if not isinstance(verdict.get("topology_index"), int):
            errors.append("scored design requires topology_index")
        if not verdict.get("full_signature"):
            errors.append("scored design requires full_signature")
    yardstick = rec.get("yardstick") or {}
    if yardstick.get("env") != "fold" or not yardstick.get("config_fingerprint"):
        errors.append("canonical fold yardstick and config_fingerprint are required")
    return errors


def validate_hardened_action_sft(rec: dict) -> List[str]:
    """Validate a topology-preserving, all-gates-passing refinement SFT example."""
    errors: List[str] = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != HARDENED_ACTION_SFT_SCHEMA:
        errors.append(f"schema_version must be {HARDENED_ACTION_SFT_SCHEMA!r}")
    for key in ("example_id", "state_id", "source_group_id"):
        if not rec.get(key):
            errors.append(f"{key} is required")
    if rec.get("split") not in ("train", "validation_seed", "test_seed"):
        errors.append("unknown split")
    if not _finite(rec.get("sample_weight")) or float(rec.get("sample_weight", 0)) <= 0:
        errors.append("sample_weight must be positive finite numeric")
    messages = rec.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages
                                          if isinstance(m, dict)] != [
                                              "system", "user", "assistant"]:
        errors.append("messages must contain system, user, assistant in that order")
    elif not (messages[-1].get("content", "").startswith("<action>")
              and messages[-1].get("content", "").endswith("</action>")):
        errors.append("assistant must contain one action block")
    labels = rec.get("labels") or {}
    if labels.get("candidate_hard_pass") is not True:
        errors.append("candidate_hard_pass must be true")
    if labels.get("topology_preserved") is not True:
        errors.append("topology_preserved must be true")
    for key in ("parent_fitness", "candidate_fitness"):
        if not _finite(labels.get(key)):
            errors.append(f"labels.{key} must be finite numeric")
    if labels.get("parent_hard_pass") is True \
            and _finite(labels.get("parent_fitness")) \
            and _finite(labels.get("candidate_fitness")) \
            and float(labels["candidate_fitness"]) <= float(labels["parent_fitness"]):
        errors.append("a hard-pass parent requires a strictly improving candidate")
    return errors


def example_codev_transition() -> dict:
    """A copyable record for the remote COM harness."""
    return {
        "schema_version": CODEV_TRANSITION_SCHEMA,
        "transition_id": "design_042_run03_step0017_try0003",
        "episode_id": "design_042_run03",
        "design_id": "design_042",
        "step": 17,
        "candidate_index": 3,
        "state": {
            "parameter_names": ["XYP_1", "XYP_2", "ADE_S4", "YDE_S4"],
            "parameters": [0.0142, -0.0031, 1.228, 0.0],
            "parameter_scales": [0.02, 0.02, 2.0, 1.0],
            "active_mask": [1, 1, 1, 0],
            "merit_before": 1.743,
            "metrics_before": {"rms_spot": 2.1, "throughput": 0.81},
            "constraint_slacks": {"tir_margin_min_deg": 1.4, "clearance_mm": 3.8},
            "topology_id": "T@A -> TIR@D -> TIR@B -> T@B",
            "budget_remaining": 83,
        },
        "action": {
            "normalized_delta": [-0.12, 0.05, 0.08, 0.0],
            "trust_scale": 0.5,
            "source": "policy",
        },
        "result": {
            "merit_after": 1.612,
            "delta_merit": 0.131,
            "metrics_after": {"rms_spot": 1.93, "throughput": 0.82},
            "constraint_slacks_after": {"tir_margin_min_deg": 1.3,
                                          "clearance_mm": 3.7},
            "trace_ok": True,
            "topology_preserved": True,
            "bounds_clipped": False,
            "accepted": True,
            "failure_code": None,
            "runtime_s": 2.8,
        },
        "provenance": {
            "codev_version": "2026",
            "merit_function_version": "mf_v4",
            "translator_version": "json2codev_v7",
            "baseline_json_sha256": "replace-on-remote",
            "random_seed": 1234,
        },
    }
