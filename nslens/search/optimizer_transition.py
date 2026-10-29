"""Stable provenance and route helpers for optimizer-generation records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def json_value(value: Any) -> Any:
    """Convert nested scientific values to strict JSON without changing precision."""
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hash(paths: Sequence[Path]) -> str:
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        else:
            rows.append({"path": path.name, "missing": True})
    return canonical_hash(rows)


def optimizer_fingerprints(cfg, problem) -> tuple[dict[str, str], dict[str, Any]]:
    """Fingerprint the exact representation, verifier settings, simulator, and archive."""
    root = Path(__file__).resolve().parents[2]
    representation = {
        "id": ("fold9_relative_sag" if len(problem.param_names) == 9
               and bool(getattr(problem, "relative_sag", False))
               else f"{type(problem).__name__}:{len(problem.param_names)}"),
        "problem_class": type(problem).__name__,
        "parameter_names": list(problem.param_names),
        "bounds": [[float(lo), float(hi)] for lo, hi in zip(problem.lo, problem.hi)],
        "normalized_search": cfg.param_scaling == "normalized",
        "relative_sag": bool(getattr(problem, "relative_sag", False)),
    }
    objective = {
        "fitness": cfg.fitness,
        "fields": list(cfg.fields or (0.0,)),
        "field_aggregate": cfg.field_aggregate,
        "focus_window": float(problem.focus_w),
        "beam_width": float(getattr(problem, "beam_w", 0.0)),
        "ray_count": int(problem.n_rays),
        "sensor_position": float(problem.x_sensor),
        "max_sag_ratio": getattr(problem, "max_sag_ratio", None),
        "detector_clearance": getattr(problem, "det_clearance", None),
        "intended_sequence": cfg.intended_sequence or None,
        "min_intended_root_support": cfg.min_intended_root_support,
        "min_intended_delivered_energy_share": cfg.min_intended_delivered_energy_share,
        "require_intended_dominant_all_fields": cfg.require_intended_dominant_all_fields,
        "min_intended_critical_margin_deg": cfg.min_intended_critical_margin_deg,
        "require_monotone_convergence": cfg.require_monotone_convergence,
        "convergence_gate": cfg.convergence_gate,
        "convergence_tol": list(cfg.convergence_tol),
        "gate_all_fields": cfg.gate_all_fields,
        "require_exit_convergence": cfg.require_exit_convergence,
        "exit_window_mult": cfg.exit_window_mult,
        "reject_out_of_bounds": cfg.reject_out_of_bounds,
    }
    archive = {
        "descriptors": list(cfg.descriptors),
        "bins": list(cfg.bc_bins),
        "bounds": [list(bound) for bound in cfg.bc_bounds],
        "cell_keys": list(cfg.cell_bc_keys),
    }
    simulator_hash = _source_hash((
        root / "nslens" / "engine" / "runner.py",
        root / "ray-optics" / "integrations" / "simulation_core.js",
        root / "ray-optics" / "src" / "core" / "Simulator.js",
    ))
    payloads = {
        "representation": representation,
        "objective": objective,
        "archive": archive,
    }
    fingerprints = {
        "representation": canonical_hash(representation),
        "objective": canonical_hash(objective),
        "archive": canonical_hash(archive),
        "simulator": simulator_hash,
    }
    return fingerprints, payloads


def route_from_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    """Encode the ordered dominant route already emitted by the hardened evaluator."""
    info = info or {}
    signature = str(info.get("dom_signature") or "")
    normalized = signature.replace("→", "->")
    tokens = [token.strip() for token in normalized.split("->") if token.strip()]
    repeats: dict[str, int] = {}
    events = []
    for position, token in enumerate(tokens):
        operation, separator, face = token.partition("@")
        upper = operation.strip().upper()
        if upper.startswith("TIR"):
            kind = "tir"
        elif upper in {"R", "PR", "M", "REFLECT", "FRESNEL"} \
                or upper.startswith("R("):
            kind = "fresnel_reflection"
        elif upper in {"MISS", "ESCAPE"}:
            kind = "miss"
        elif upper in {"END", "TERMINAL", "SENSOR"}:
            kind = "terminal"
        else:
            kind = "transmission_refraction"
        face = face.strip().upper() if separator else "unknown"
        repeat = repeats.get(face, 0)
        repeats[face] = repeat + 1
        events.append({
            "position": position,
            "kind": kind,
            "face": face if face in {"A", "B", "T", "D"} else "unknown",
            "repeat_index": repeat,
            "field_support": info.get("intended_root_support_min"),
            "incidence_angle_deg": None,
            "turning_angle_deg": None,
            "critical_margin_deg": info.get("intended_critical_margin_min",
                                             info.get("crit_margin")),
            "energy_share": info.get("dom_energy_share"),
        })
    return {
        "signature": signature or None,
        "events": events,
        "per_field": json_value(info.get("per_field") or []),
        "support": info.get("intended_root_support_min"),
        "persistence": info.get("intended_dominant_all"),
        "worst_critical_margin_deg": info.get(
            "intended_critical_margin_min", info.get("crit_margin")),
    }


__all__ = ["canonical_hash", "json_value", "optimizer_fingerprints", "route_from_info"]
