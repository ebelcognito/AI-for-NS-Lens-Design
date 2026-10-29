"""Compact prompts for the optical-design proposer.

The prompt is deliberately a serialization of the task, not a conversation. Training
and inference should never resend a .seq file, a render, or an accumulated transcript:
the policy emits a coarse starting point and the deterministic harness owns the rest.

Modes:
    directed       Names a requested TIR-face multiset.
    unconditioned  Contains functional requirements but no topology label.
    format         Asks only for an in-bounds, buildable vector.
"""
from __future__ import annotations

import json
from typing import Dict

from .env_config import RLEnv
from .reward import Spec


SYSTEM = (
    "You propose coarse starting points for a 2D non-sequential folded optic. "
    "Return exactly one <design>{...}</design> block and no prose. Every key must appear "
    "once, every value must be numeric, and every value must lie inside its stated bound."
)

# A certification records the best score found by three expensive probes. Asking every
# SFT target to equal that record would make a slightly perturbed, still excellent seed a
# false label. Eighty percent is the actual task threshold; the exact certified value is
# retained in labels and remains the RL normalization denominator.
TRAIN_TARGET_FRACTION = 0.8


def thinness_interval(env: RLEnv, bin_index: int) -> tuple[float, float]:
    """Physical thinness interval represented by one grid bin."""
    lo, hi = (float(v) for v in env.cfg.bc_bounds[1])
    bins = int(env.cfg.bc_bins[1])
    if not (0 <= int(bin_index) < bins):
        raise ValueError(f"thinness bin {bin_index} outside [0, {bins})")
    width = (hi - lo) / bins
    return lo + int(bin_index) * width, lo + (int(bin_index) + 1) * width


def _bounds(env: RLEnv) -> Dict[str, list[float]]:
    return {name: [lo, hi] for name, (lo, hi) in env.bounds().items()}


def spec_payload(spec: Spec, env: RLEnv, *, mode: str = "directed") -> dict:
    """Structured input retained beside the rendered prompt for auditability."""
    if mode not in ("directed", "unconditioned"):
        raise ValueError("mode must be 'directed' or 'unconditioned'")
    t0, t1 = thinness_interval(env, spec.thinness_bin)
    payload = {
        "task": "propose_folded_optic_starting_point",
        "mode": mode,
        "environment": spec.env,
        "half_fov_deg": float(spec.fov),
        "detector_window_half_width_mm": float(spec.focus_w),
        "minimum_worst_field_focused_throughput":
            float(TRAIN_TARGET_FRACTION * spec.achievable),
        "thinness_T_interval_mm": [round(t0, 6), round(t1, 6)],
        "parameter_bounds": _bounds(env),
    }
    if mode == "directed":
        payload["requested_tir_face_multiset"] = spec.topology_label
        payload["requested_topology_index"] = int(spec.topology)
    return payload


def spec_prompt(spec: Spec, env: RLEnv, *, mode: str = "directed") -> str:
    return json.dumps(spec_payload(spec, env, mode=mode),
                      sort_keys=True, separators=(",", ":"))


def format_payload(env: RLEnv) -> dict:
    """Topology-free scaffold task: syntax, bounds and cheap geometry validity only."""
    return {
        "task": "emit_buildable_in_bounds_geometry",
        "mode": "format",
        "environment": env.name,
        "parameter_bounds": _bounds(env),
        "quality_or_topology_target": None,
    }


def format_prompt(env: RLEnv) -> str:
    return json.dumps(format_payload(env), sort_keys=True, separators=(",", ":"))


def messages(user_payload: dict, assistant_design: str) -> list[dict]:
    """Chat-template-neutral SFT record."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(
            user_payload, sort_keys=True, separators=(",", ":"))},
        {"role": "assistant", "content": assistant_design},
    ]
