"""Semantic registry and fixed-shape optical-context encoding."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GeneRole:
    name: str
    kind: str
    group: str
    units: str


GENE_REGISTRY = {
    role.name: role for role in (
        GeneRole("ray_y", "source_position", "source", "normalized_length"),
        GeneRole("ang", "source_angle", "source", "normalized_degrees"),
        GeneRole("cx", "element_position", "pose_package", "normalized_length"),
        GeneRole("thick", "element_thickness", "pose_package", "normalized_length"),
        GeneRole("fsag", "relative_face_sag", "surface_shape", "relative_sag"),
        GeneRole("bsag", "relative_face_sag", "surface_shape", "relative_sag"),
        GeneRole("tilt_a", "face_tilt", "surface_pose", "normalized_degrees"),
        GeneRole("tilt_b", "face_tilt", "surface_pose", "normalized_degrees"),
        GeneRole("n", "refractive_index", "material", "normalized_index"),
    )
}


def _vocabulary(values: Sequence[str]) -> dict[str, int]:
    return {value: index + 1 for index, value in enumerate(sorted(set(values)))}


NAME_IDS = _vocabulary(tuple(GENE_REGISTRY))
KIND_IDS = _vocabulary(tuple(role.kind for role in GENE_REGISTRY.values()))
GROUP_IDS = _vocabulary(tuple(role.group for role in GENE_REGISTRY.values()))
UNIT_IDS = _vocabulary(tuple(role.units for role in GENE_REGISTRY.values()))
STATUS_IDS = {"unknown": 0, "invalid": 1, "none": 2, "improved": 3, "new": 4}
FAILURE_NAMES = (
    "unknown", "none", "invalid_geometry", "engine_error", "no_light",
    "descriptor_out_of_bounds", "beam_gate", "exit_gate", "intended_root_support",
    "beam_gate_all_fields", "exit_gate_all_fields", "intended_not_dominant",
    "intended_low_energy", "intended_tir_margin",
)
FAILURE_IDS = {name: index for index, name in enumerate(FAILURE_NAMES)}
ROUTE_KIND_IDS = {
    "padding": 0, "transmission_refraction": 1, "tir": 2,
    "fresnel_reflection": 3, "miss": 4, "terminal": 5,
}
FACE_IDS = {"padding": 0, "unknown": 1, "A": 2, "B": 3, "T": 4, "D": 5}
REPRESENTATION_IDS = {"unknown": 0, "fold9_relative_sag": 1, "fixed_fold6": 2}
AGGREGATE_IDS = {"unknown": 0, "mean": 1, "worst": 2}

ENV_CONTINUOUS_SIZE = 24
CANDIDATE_CONTINUOUS_SIZE = 48
ROUTE_CONTINUOUS_SIZE = 5
MAX_ROUTE_EVENTS = 12


def parameter_metadata(parameter_names: Sequence[str]) -> dict[str, np.ndarray]:
    roles = [GENE_REGISTRY.get(name, GeneRole(name, "unknown", "unknown", "unknown"))
             for name in parameter_names]
    return {
        "name": np.asarray([NAME_IDS.get(role.name, 0) for role in roles], np.int32),
        "kind": np.asarray([KIND_IDS.get(role.kind, 0) for role in roles], np.int32),
        "group": np.asarray([GROUP_IDS.get(role.group, 0) for role in roles], np.int32),
        "units": np.asarray([UNIT_IDS.get(role.units, 0) for role in roles], np.int32),
    }


def _fingerprint_features(value: str | None) -> list[float]:
    try:
        digest = bytes.fromhex(value) if value and len(value) == 64 else None
    except ValueError:
        digest = None
    if digest is None:
        digest = hashlib.sha256(str(value).encode()).digest()
    return [int.from_bytes(digest[offset:offset + 4], "big") / (2**32 - 1)
            for offset in range(0, 16, 4)]


def encode_environment(context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Encode explicitly-scaled continuous and small categorical environment values."""
    representation = context.get("representation") or {}
    objective = context.get("objective") or {}
    archive = context.get("archive") or {}
    fingerprints = context.get("fingerprints") or {}
    fields = np.asarray(objective.get("fields") or [0.0], dtype=np.float64)
    archive_bounds = np.asarray(archive.get("bounds") or [[0, 0], [0, 0]], dtype=float)
    archive_bins = np.asarray(archive.get("bins") or [0, 0], dtype=float)
    values = [
        float(np.min(fields)) / 90.0,
        float(np.max(fields)) / 90.0,
        float(np.mean(fields)) / 90.0,
        float(np.std(fields)) / 90.0,
        len(fields) / 32.0,
        float(objective.get("sensor_position") or 0.0) / 250.0,
        float(objective.get("focus_window") or 0.0) / 50.0,
        float(objective.get("beam_width") or 0.0) / 50.0,
        float(objective.get("ray_count") or 0.0) / 128.0,
        float(objective.get("max_sag_ratio") or 0.0),
        float(objective.get("detector_clearance") or 0.0) / 100.0,
        float(objective.get("min_intended_critical_margin_deg") or 0.0) / 20.0,
        float(objective.get("min_intended_root_support") or 0.0),
        float(objective.get("min_intended_delivered_energy_share") or 0.0),
        float(bool(objective.get("require_monotone_convergence"))),
        float(bool(objective.get("require_exit_convergence"))),
        *(archive_bins[:2] / 256.0).tolist(),
        *(archive_bounds.reshape(-1)[:4] / 250.0).tolist(),
    ]
    values.extend(_fingerprint_features(fingerprints.get("objective"))[:2])
    values = (values + [0.0] * ENV_CONTINUOUS_SIZE)[:ENV_CONTINUOUS_SIZE]
    representation_id = str(representation.get("id") or "unknown")
    if representation_id.startswith("FixedFoldProblem"):
        representation_id = "fixed_fold6"
    return {
        "continuous": np.asarray(values, dtype=np.float32),
        "categorical": np.asarray([
            REPRESENTATION_IDS.get(representation_id, 0),
            AGGREGATE_IDS.get(str(objective.get("field_aggregate") or "unknown"), 0),
        ], dtype=np.int32),
    }


def _finite(value: Any, scale: float = 1.0) -> tuple[float, float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not np.isfinite(number):
        return 0.0, 0.0
    return number / scale, 1.0


def _pair(value: Any, scale: float) -> list[float]:
    sequence = value if isinstance(value, (list, tuple, np.ndarray)) else ()
    output = []
    for index in range(2):
        output.extend(_finite(sequence[index] if index < len(sequence) else None, scale))
    return output


def encode_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    population_size: int,
    num_dims: int,
    max_route_events: int = MAX_ROUTE_EVENTS,
) -> dict[str, np.ndarray]:
    """Pad one evaluated population; masks make absent values exactly zero."""
    status = np.zeros(population_size, dtype=np.int32)
    failure = np.zeros(population_size, dtype=np.int32)
    continuous = np.zeros((population_size, CANDIDATE_CONTINUOUS_SIZE), np.float32)
    clipping = np.zeros((population_size, num_dims), np.float32)
    route_kind = np.zeros((population_size, max_route_events), np.int32)
    route_face = np.zeros((population_size, max_route_events), np.int32)
    route_repeat = np.zeros((population_size, max_route_events), np.int32)
    route_continuous = np.zeros(
        (population_size, max_route_events, ROUTE_CONTINUOUS_SIZE), np.float32)
    route_mask = np.zeros((population_size, max_route_events), np.float32)

    for row_index, observation in enumerate(observations[:population_size]):
        status[row_index] = STATUS_IDS.get(str(observation.get("status") or "unknown"), 0)
        failure_name = str((observation.get("failure") or {}).get("stage") or "none")
        failure[row_index] = FAILURE_IDS.get(failure_name, FAILURE_IDS["unknown"])
        clip = np.asarray(observation.get("clipping_mask") or [], dtype=np.float32)
        clipping[row_index, :min(num_dims, len(clip))] = clip[:num_dims]
        metrics = observation.get("optical_metrics") or {}
        archive = observation.get("archive_context") or {}
        field_values = []
        for value in metrics.get("per_field") or []:
            parsed, present = _finite(value)
            if present:
                field_values.append(parsed)
        finite_fields = np.asarray(field_values, dtype=np.float64)
        improvement, improvement_mask = _finite(observation.get("archive_improvement"))
        fitness, fitness_mask = _finite(observation.get("fitness"))
        previous, previous_mask = _finite(archive.get("previous_elite_fitness"))
        parent = archive.get("parent") or {}
        parent_route = parent.get("route_family")
        margin, margin_mask = _finite(
            metrics.get("intended_critical_margin_min", metrics.get("crit_margin")), 20.0)
        values = [
            *_finite(observation.get("rank_percentile")),
            improvement, improvement_mask,
            fitness, fitness_mask,
            float(np.mean(finite_fields)) if len(finite_fields) else 0.0,
            float(np.min(finite_fields)) if len(finite_fields) else 0.0,
            float(np.max(finite_fields)) if len(finite_fields) else 0.0,
            float(np.std(finite_fields)) if len(finite_fields) else 0.0,
            min(len(finite_fields), 32) / 32.0,
            *_finite(metrics.get("efficiency")),
            *_finite(metrics.get("spot"), 50.0),
            margin, margin_mask,
            float(archive.get("coverage") or 0.0),
            float(archive.get("filled") or 0.0) / 2048.0,
            previous, previous_mask,
            float(np.mean(clipping[row_index])) if num_dims else 0.0,
            float(archive.get("cell_was_empty") is True),
            *_pair(archive.get("proposed_cell"), 256.0),
            *_pair(observation.get("behavior"), 250.0),
            *_pair(parent.get("cell"), 256.0),
            *_finite(parent.get("fitness")),
            *_pair(parent.get("behavior"), 250.0),
            *(_fingerprint_features(str(parent_route))[:2]
              if parent_route else [0.0, 0.0]),
            float(bool(parent_route)),
        ]
        continuous[row_index] = np.asarray(
            (values + [0.0] * CANDIDATE_CONTINUOUS_SIZE)[:CANDIDATE_CONTINUOUS_SIZE],
            dtype=np.float32)
        events = (observation.get("route") or {}).get("events") or []
        for event_index, event in enumerate(events[:max_route_events]):
            route_mask[row_index, event_index] = 1.0
            route_kind[row_index, event_index] = ROUTE_KIND_IDS.get(
                str(event.get("kind") or "padding"), 0)
            route_face[row_index, event_index] = FACE_IDS.get(
                str(event.get("face") or "unknown"), FACE_IDS["unknown"])
            route_repeat[row_index, event_index] = min(
                int(event.get("repeat_index") or 0) + 1, max_route_events)
            event_values = []
            for key, scale in (("field_support", 1.0), ("incidence_angle_deg", 180.0),
                               ("turning_angle_deg", 180.0),
                               ("critical_margin_deg", 90.0), ("energy_share", 1.0)):
                event_values.append(_finite(event.get(key), scale)[0])
            route_continuous[row_index, event_index] = event_values
    return {
        "status": status,
        "failure": failure,
        "continuous": continuous,
        "clipping": clipping,
        "route_kind": route_kind,
        "route_face": route_face,
        "route_repeat": route_repeat,
        "route_continuous": route_continuous,
        "route_mask": route_mask,
    }


__all__ = [
    "AGGREGATE_IDS", "CANDIDATE_CONTINUOUS_SIZE", "ENV_CONTINUOUS_SIZE",
    "FACE_IDS", "FAILURE_IDS", "GENE_REGISTRY", "GROUP_IDS", "KIND_IDS",
    "MAX_ROUTE_EVENTS", "NAME_IDS", "REPRESENTATION_IDS", "ROUTE_CONTINUOUS_SIZE",
    "ROUTE_KIND_IDS", "STATUS_IDS", "UNIT_IDS", "encode_environment",
    "encode_observations", "parameter_metadata",
]
