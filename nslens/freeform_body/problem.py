"""Outcome-only Stage-1 objective for continuous freeform discovery.

The evaluator rewards energy delivered near requested image points.  It never rewards
TIR, reflection count, a particular footprint, or a particular interaction sequence.
Those properties leave through the separate diversity descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Mapping, Protocol

import numpy as np

from .codec import FreeformBodyCodec
from .descriptors import (Descriptor, interaction_descriptor,
                          ranked_route_signatures, shape_descriptor)
from .trace import InputRay, project_exits_to_x, trace_bundle


@dataclass(frozen=True)
class SensorTarget:
    source_id: int
    y: float
    half_height: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.y, self.half_height)).all() or self.half_height <= 0:
            raise ValueError("sensor target values must be finite and half-height positive")


@dataclass(frozen=True)
class FieldScore:
    source_id: int
    throughput: float
    rms_error: float
    fitness: float
    delivered_energy: float
    emitted_energy: float


@dataclass(frozen=True)
class DiscoveryEvaluation:
    """Result of one candidate evaluation.

    ``fitness`` is the archive currency and is meaningful only when ``feasible``.
    ``search_score`` is the separate signal the emitter ranks on; an objective with a
    feasibility ladder uses it to order infeasible candidates that all share
    ``fitness == 0``.  Keeping the two apart is what lets a ladder shape the *search*
    without shaping what counts as a discovered design.
    """

    feasible: bool
    fitness: float
    fields: tuple[FieldScore, ...]
    shape: Descriptor | None
    interaction: Descriptor | None
    route_keys: tuple[tuple, ...]
    geometry_report: object | None
    trace_result: object | None
    reason: str | None = None
    search_score: float = 0.0


class DiscoveryProblem(Protocol):
    """What the archive and emitter require of any Stage-1 objective."""

    codec: FreeformBodyCodec

    def evaluate(self, genes) -> DiscoveryEvaluation:
        ...


@dataclass(frozen=True)
class FreeformDiscoveryProblem:
    """Decode, trace, score, and describe one Stage-1 freeform candidate."""

    codec: FreeformBodyCodec
    rays: tuple[InputRay, ...]
    sensor_x: float
    targets: tuple[SensorTarget, ...]
    spot_scale: float = 1.0
    minimum_field_throughput: float = 1e-3
    max_interactions: int = 16
    energy_floor: float = 1e-4

    def __post_init__(self) -> None:
        if not self.rays:
            raise ValueError("a discovery problem requires input rays")
        if not self.targets:
            raise ValueError("a discovery problem requires sensor targets")
        if self.spot_scale <= 0:
            raise ValueError("spot_scale must be positive")
        if not np.isfinite((self.sensor_x, self.spot_scale,
                            self.minimum_field_throughput, self.energy_floor)).all():
            raise ValueError("problem scalar settings must be finite")
        if not 0.0 <= self.minimum_field_throughput <= 1.0:
            raise ValueError("minimum_field_throughput must lie in [0, 1]")
        if self.max_interactions < 1 or self.energy_floor < 0:
            raise ValueError("max_interactions must be positive and energy_floor non-negative")
        target_ids = [target.source_id for target in self.targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("each source_id may have only one sensor target")
        ray_ids = {ray.source_id for ray in self.rays}
        missing = set(target_ids) - ray_ids
        if missing:
            raise ValueError(f"targets have no corresponding input rays: {sorted(missing)}")

    def evaluate(self, genes) -> DiscoveryEvaluation:
        try:
            body = self.codec.decode(genes)
            geometry = body.geometry_report()
        except (ValueError, FloatingPointError) as exc:
            return DiscoveryEvaluation(False, 0.0, (), None, None, (), None, None,
                                       str(exc), search_score=0.0)
        if not geometry.valid:
            return DiscoveryEvaluation(
                False, 0.0, (), None, None, (), geometry, None,
                "; ".join(geometry.reasons), search_score=0.0,
            )

        trace = trace_bundle(
            body,
            self.rays,
            max_interactions=self.max_interactions,
            energy_floor=self.energy_floor,
        )
        projected = project_exits_to_x(trace, self.sensor_x)
        emitted_by_source: Mapping[int, float] = {
            source_id: sum(ray.energy_s + ray.energy_p for ray in self.rays
                           if ray.source_id == source_id)
            for source_id in {ray.source_id for ray in self.rays}
        }

        fields = []
        for target in self.targets:
            records = [record for record in projected
                       if record["source_id"] == target.source_id
                       and abs(record["y"] - target.y) <= target.half_height]
            delivered = sum(record["energy"] for record in records)
            emitted = emitted_by_source[target.source_id]
            throughput = delivered / max(emitted, 1e-30)
            if delivered > 0:
                mse = sum(record["energy"] * (record["y"] - target.y) ** 2
                          for record in records) / delivered
                rms = float(np.sqrt(max(mse, 0.0)))
                fitness = throughput * exp(-rms / self.spot_scale)
            else:
                rms = float("inf")
                fitness = 0.0
            fields.append(FieldScore(
                source_id=target.source_id,
                throughput=float(throughput),
                rms_error=rms,
                fitness=float(fitness),
                delivered_energy=float(delivered),
                emitted_energy=float(emitted),
            ))

        feasible = all(field.throughput >= self.minimum_field_throughput for field in fields)
        # Geometric mean makes a missing/poor field load-bearing without assigning an
        # interaction route to any field.
        fitness = (float(np.exp(np.mean(np.log(np.maximum(
            [field.fitness for field in fields], 1e-30))))) if feasible else 0.0)
        route_keys = ranked_route_signatures(trace)
        return DiscoveryEvaluation(
            feasible=feasible,
            fitness=fitness,
            fields=tuple(fields),
            shape=shape_descriptor(body),
            interaction=interaction_descriptor(trace),
            route_keys=route_keys,
            geometry_report=geometry,
            trace_result=trace,
            reason=None if feasible else "one or more fields missed the throughput floor",
            search_score=fitness,
        )
