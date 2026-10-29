"""Canonical, geometry-invariant fingerprints for recorded ray histories.

The v1 fingerprint intentionally removes four nuisance symmetries: global translation,
global rotation, mirror reflection, and arbitrary surface naming.  It preserves the
ordered optical behavior needed to distinguish intentional interaction motifs:

* interaction type and partial-reflection mechanism;
* first/repeated use of a physical surface;
* normalized contact location on that surface;
* unsigned ray turning angle; and
* terminal outcome and whether the history is exact and complete.

Energy is not part of the hash.  It is a weight attached to a fingerprint when a bundle
is aggregated.  Absolute positions and directions are likewise diagnostic data, not
identity data.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

Vertex = Mapping[str, Any]
SurfaceOf = Callable[[Vertex], Hashable]
ContactCoordinateOf = Callable[[Vertex, Hashable], float | None]
IncidenceOf = Callable[[Vertex, Hashable], float | None]

FINGERPRINT_SCHEMA = "nslens.interaction_fingerprint.v1"
FINGERPRINT_INVARIANCES = (
    "global_translation",
    "global_rotation",
    "mirror_reflection",
    "surface_relabeling",
    "surface_coordinate_orientation",
)
_NON_INTERACTIONS = frozenset({"source", "sensor", "depth_limit"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identity_key(value: Hashable) -> tuple[str, str]:
    """Stable comparison key without exposing the original label in the fingerprint."""
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError):
        encoded = repr(value)
    return type(value).__name__, encoded


def _nearest_bin(value: float, quantum: float) -> int:
    """Round symmetrically to the nearest integer, including exact half-bin values."""
    if not math.isfinite(value):
        raise ValueError(f"fingerprint descriptors must be finite, got {value!r}")
    scaled = abs(value) / quantum
    rounded = int(math.floor(scaled + 0.5))
    return -rounded if value < 0 else rounded


def _turning_angle_deg(vertex: Vertex) -> float | None:
    values = (vertex.get("dirInX"), vertex.get("dirInY"),
              vertex.get("dirOutX"), vertex.get("dirOutY"))
    if any(value is None for value in values):
        return None
    ix, iy, ox, oy = (float(value) for value in values)
    in_norm = math.hypot(ix, iy)
    out_norm = math.hypot(ox, oy)
    if in_norm == 0 or out_norm == 0:
        return None
    cosine = max(-1.0, min(1.0, (ix * ox + iy * oy) / (in_norm * out_norm)))
    return math.degrees(math.acos(cosine))


def fingerprint_record(
    record: Mapping[str, Any],
    surface_of: SurfaceOf | None = None,
    contact_coordinate_of: ContactCoordinateOf | None = None,
    incidence_deg_of: IncidenceOf | None = None,
    *,
    contact_quantum: float = 0.1,
    angle_quantum_deg: float = 2.0,
) -> dict[str, Any]:
    """Return the canonical v1 fingerprint document for one terminal ray record.

    ``surface_of`` should identify physical faces, not merely scene objects.  If it is
    omitted, the recorder's object index ``i`` is used.  ``contact_coordinate_of`` must
    return a dimensionless surface-local coordinate (normally -1 to +1 across the clear
    aperture).  Its sign is canonicalized independently on every surface, making the
    result invariant to mirror reflection and arbitrary tangent orientation.

    ``path_hash`` identifies optical interactions only.  ``fingerprint_hash`` additionally
    distinguishes delivered/escaped/absorbed/truncated/unfinished and exact/incomplete
    histories.  This separation lets audits compare the same path across different fates.
    """
    if contact_quantum <= 0 or angle_quantum_deg <= 0:
        raise ValueError("fingerprint quantization steps must be positive")

    seq = record.get("seq")
    if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
        raise ValueError("terminal record must contain a sequence in 'seq'")

    raw_events: list[dict[str, Any]] = []
    surface_names: dict[tuple[str, str], str] = {}
    repeat_counts: dict[tuple[str, str], int] = {}
    contact_bins_by_surface: dict[str, list[int]] = {}

    for vertex in seq:
        event_type = str(vertex.get("t") or "other")
        if event_type in _NON_INTERACTIONS:
            continue

        raw_surface = surface_of(vertex) if surface_of else vertex.get("i", "unknown")
        key = _identity_key(raw_surface)
        if key not in surface_names:
            surface_names[key] = f"s{len(surface_names)}"
        surface = surface_names[key]
        repeat = repeat_counts.get(key, 0)
        repeat_counts[key] = repeat + 1

        contact_bin = None
        if contact_coordinate_of is not None:
            coordinate = contact_coordinate_of(vertex, raw_surface)
            if coordinate is not None:
                contact_bin = _nearest_bin(float(coordinate), contact_quantum)
                contact_bins_by_surface.setdefault(surface, []).append(contact_bin)

        angle = _turning_angle_deg(vertex)
        turning_bin = (_nearest_bin(angle, angle_quantum_deg)
                       if angle is not None else None)

        incidence_bin = None
        if incidence_deg_of is not None:
            incidence = incidence_deg_of(vertex, raw_surface)
            if incidence is not None:
                incidence_bin = _nearest_bin(abs(float(incidence)), angle_quantum_deg)

        raw_events.append({
            "interaction": event_type,
            "mechanism": (str(vertex.get("mech"))
                          if vertex.get("mech") is not None else None),
            "surface": surface,
            "repeat_index": repeat,
            "contact_bin": contact_bin,
            "turning_angle_bin": turning_bin,
            "incidence_angle_bin": incidence_bin,
        })

    # A reflected geometry, or the opposite parameterization of a surface tangent,
    # negates all its contact coordinates.  Select the lexicographically smaller of
    # the two orientations per surface and apply that choice to the event stream.
    flip_surface: dict[str, bool] = {}
    for surface, bins in contact_bins_by_surface.items():
        forward = tuple(bins)
        reversed_sign = tuple(-value for value in bins)
        flip_surface[surface] = reversed_sign < forward
    for event in raw_events:
        if event["contact_bin"] is not None and flip_surface.get(event["surface"], False):
            event["contact_bin"] = -event["contact_bin"]

    quantization = {
        "contact_coordinate": float(contact_quantum),
        "angle_degrees": float(angle_quantum_deg),
    }
    path_payload = {
        "schema_version": FINGERPRINT_SCHEMA,
        "quantization": quantization,
        "events": raw_events,
    }
    path_hash = _digest(path_payload)
    terminal = {
        "outcome": str(record.get("outcome", "unknown")),
        "history_complete": bool(record.get("historyComplete", False)),
        "aggregate": bool(record.get("aggregate", False)),
    }
    fingerprint_payload = {"path_hash": path_hash, "terminal": terminal}
    return {
        "schema_version": FINGERPRINT_SCHEMA,
        "invariances": list(FINGERPRINT_INVARIANCES),
        "quantization": quantization,
        "events": raw_events,
        "path_hash": path_hash,
        "terminal": terminal,
        "fingerprint_hash": _digest(fingerprint_payload),
    }


def make_single_element_contact_coordinate(
    cx: float,
    thickness: float,
    tilt_a_deg: float,
    tilt_b_deg: float,
    elem_h: float,
) -> ContactCoordinateOf:
    """Surface-local coordinate callback for the project's single polygonal element.

    Optical faces A/B use ``y / elem_h``.  Top/bottom faces use a coordinate normalized
    between their A and B edge vertices.  Values outside [-1, 1] are retained so malformed
    or edge-tolerance contacts remain visible to the audit rather than being clipped away.
    """
    if elem_h <= 0 or thickness <= 0:
        raise ValueError("element height and thickness must be positive")
    left_x = float(cx) - float(thickness) / 2
    right_x = float(cx) + float(thickness) / 2
    delta_a = math.tan(math.radians(float(tilt_a_deg))) * float(elem_h)
    delta_b = math.tan(math.radians(float(tilt_b_deg))) * float(elem_h)

    def coordinate(vertex: Vertex, surface: Hashable) -> float | None:
        label = str(surface)
        if label in {"A", "B"}:
            return float(vertex["y"]) / float(elem_h)
        if label == "T":
            x_a, x_b = left_x + delta_a, right_x + delta_b
        elif label == "D":
            x_a, x_b = left_x - delta_a, right_x - delta_b
        else:
            return None
        half_width = (x_b - x_a) / 2
        if abs(half_width) < 1e-15:
            return None
        return (float(vertex["x"]) - (x_a + x_b) / 2) / half_width

    return coordinate


def terminal_energy_by_outcome(result: Mapping[str, Any]) -> dict[str, float]:
    """Sum terminal-record energy by fate; aggregate incomplete records are included."""
    totals = {name: 0.0 for name in
              ("delivered", "escaped", "absorbed", "truncated", "unfinished")}
    for record in result.get("rayTerminations", []):
        outcome = str(record.get("outcome", "unknown"))
        totals[outcome] = totals.get(outcome, 0.0) + float(record.get("b", 0.0))
    return totals


def terminal_ledger_residuals(result: Mapping[str, Any]) -> dict[str, float]:
    """Terminal-stream minus energy-ledger residual for each fate and for the total."""
    ledger = result.get("energyLedger") or {}
    totals = terminal_energy_by_outcome(result)
    residuals = {name: totals.get(name, 0.0) - float(ledger.get(name, 0.0))
                 for name in ("delivered", "escaped", "absorbed",
                              "truncated", "unfinished")}
    overflow = result.get("rayTerminationOverflow") or {}
    residuals["overflow_energy"] = float(overflow.get("energy", 0.0))
    residuals["total"] = sum(residuals[name] for name in
                             ("delivered", "escaped", "absorbed",
                              "truncated", "unfinished"))
    return residuals


__all__ = [
    "FINGERPRINT_SCHEMA",
    "FINGERPRINT_INVARIANCES",
    "fingerprint_record",
    "make_single_element_contact_coordinate",
    "terminal_energy_by_outcome",
    "terminal_ledger_residuals",
]
