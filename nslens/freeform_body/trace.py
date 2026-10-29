"""Non-sequential Fresnel/TIR tracing against a continuous freeform element.

This is a Stage-1 discovery tracer.  It does not assume an interaction order and it
does not classify hits as front/back/top/bottom.  Transmitted and Fresnel-reflected
branches are both retained until they escape, fall below the energy floor, or reach a
hard interaction/branch limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, degrees, sqrt
from typing import Iterable

import numpy as np

from .boundary import FreeformBody2D

_EPS = 1e-12


def _unit(v: Iterable[float]) -> np.ndarray:
    a = np.asarray(tuple(v), dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("vectors must be finite and two-dimensional")
    norm = float(np.linalg.norm(a))
    if norm <= _EPS:
        raise ValueError("direction must be non-zero")
    return a / norm


@dataclass(frozen=True)
class InputRay:
    origin: tuple[float, float]
    direction: tuple[float, float]
    energy_s: float = 0.5
    energy_p: float = 0.5
    source_id: int = 0

    def __post_init__(self) -> None:
        _unit(self.direction)
        if len(self.origin) != 2 or not np.isfinite(self.origin).all():
            raise ValueError("origin must be a finite 2-vector")
        if (not np.isfinite((self.energy_s, self.energy_p)).all()
                or self.energy_s < 0 or self.energy_p < 0):
            raise ValueError("ray energy must be finite and non-negative")


@dataclass(frozen=True)
class BoundaryEvent:
    """An interaction on the one continuous surface.

    ``arc_s`` is normalized boundary arclength and is the stable replacement for a
    legacy face label.  ``critical_margin_deg`` is populated only for an internal
    encounter; it is positive exactly when that encounter is TIR.
    """

    kind: str
    point: tuple[float, float]
    u: float
    arc_s: float
    curvature: float
    incidence_deg: float
    critical_margin_deg: float | None
    energy_in: float
    energy_out: float
    body_id: int = 0
    face_id: int | None = None


@dataclass(frozen=True)
class ExitRay:
    source_id: int
    origin: tuple[float, float]
    direction: tuple[float, float]
    energy_s: float
    energy_p: float
    events: tuple[BoundaryEvent, ...]

    @property
    def energy(self) -> float:
        return self.energy_s + self.energy_p


@dataclass(frozen=True)
class TerminatedRay:
    """A branch that did not escape because a numerical safety limit stopped it."""

    source_id: int
    energy: float
    events: tuple[BoundaryEvent, ...]
    reason: str


@dataclass(frozen=True)
class TraceResult:
    exits: tuple[ExitRay, ...]
    terminations: tuple[TerminatedRay, ...]
    emitted_energy: float
    pruned_energy: float
    depth_limited_energy: float
    branch_limited_energy: float

    @property
    def escaped_energy(self) -> float:
        return sum(ray.energy for ray in self.exits)

    @property
    def closure_residual(self) -> float:
        accounted = (self.escaped_energy + self.pruned_energy
                     + self.depth_limited_energy + self.branch_limited_energy)
        return self.emitted_energy - accounted

    def dominant_exit(self, source_id: int | None = None) -> ExitRay | None:
        rays = self.exits if source_id is None else tuple(
            r for r in self.exits if r.source_id == source_id
        )
        return max(rays, key=lambda r: r.energy, default=None)


@dataclass
class _Branch:
    source_id: int
    origin: np.ndarray
    direction: np.ndarray
    energy_s: float
    energy_p: float
    inside: bool
    depth: int
    events: tuple[BoundaryEvent, ...]

    @property
    def energy(self) -> float:
        return self.energy_s + self.energy_p


def _event(kind: str, hit, incidence: float, critical_margin: float | None,
           energy_in: float, energy_out: float) -> BoundaryEvent:
    return BoundaryEvent(
        kind=kind,
        point=hit.point,
        u=hit.u,
        arc_s=hit.arc_s,
        curvature=hit.curvature,
        incidence_deg=incidence,
        critical_margin_deg=critical_margin,
        energy_in=energy_in,
        energy_out=energy_out,
        body_id=getattr(hit, "body_id", 0),
        face_id=getattr(hit, "face_id", None),
    )


def trace_bundle(body: FreeformBody2D, rays: Iterable[InputRay], *,
                 max_interactions: int = 16, energy_floor: float = 1e-4,
                 max_branches: int = 20000, intersection_scan: int = 512,
                 ray_epsilon: float = 1e-7) -> TraceResult:
    """Trace input rays until every branch terminates.

    The branch cap is a safety device for trapping cavities.  If it is reached, all
    pending energy is reported separately rather than silently discarded.
    """
    if max_interactions < 1 or max_branches < 1:
        raise ValueError("interaction and branch limits must be positive")
    if (not np.isfinite((energy_floor, ray_epsilon)).all()
            or energy_floor < 0 or ray_epsilon <= 0):
        raise ValueError("energy_floor must be non-negative and ray_epsilon positive")

    input_rays = tuple(rays)
    emitted = sum(r.energy_s + r.energy_p for r in input_rays)
    stack: list[_Branch] = []
    for ray in input_rays:
        stack.append(_Branch(
            source_id=ray.source_id,
            origin=np.asarray(ray.origin, dtype=float),
            direction=_unit(ray.direction),
            energy_s=float(ray.energy_s),
            energy_p=float(ray.energy_p),
            inside=body.contains(ray.origin),
            depth=0,
            events=(),
        ))

    exits: list[ExitRay] = []
    terminations: list[TerminatedRay] = []
    pruned = depth_limited = branch_limited = 0.0
    processed = 0

    while stack:
        if processed >= max_branches:
            branch_limited += sum(branch.energy for branch in stack)
            terminations.extend(TerminatedRay(
                source_id=branch.source_id,
                energy=branch.energy,
                events=branch.events,
                reason="branch_limit",
            ) for branch in stack)
            stack.clear()
            break
        branch = stack.pop()
        processed += 1
        # The equality matters when energy_floor == 0: a zero-power Fresnel branch
        # must terminate rather than continue splitting until the branch cap.
        if branch.energy <= energy_floor:
            pruned += branch.energy
            terminations.append(TerminatedRay(
                source_id=branch.source_id,
                energy=branch.energy,
                events=branch.events,
                reason="energy_floor",
            ))
            continue

        hit = body.nearest_intersection(
            branch.origin, branch.direction,
            t_min=0.5 * ray_epsilon,
            scan_count=intersection_scan,
        )
        if hit is None:
            exits.append(ExitRay(
                source_id=branch.source_id,
                origin=(float(branch.origin[0]), float(branch.origin[1])),
                direction=(float(branch.direction[0]), float(branch.direction[1])),
                energy_s=branch.energy_s,
                energy_p=branch.energy_p,
                events=branch.events,
            ))
            continue
        if branch.depth >= max_interactions:
            depth_limited += branch.energy
            terminations.append(TerminatedRay(
                source_id=branch.source_id,
                energy=branch.energy,
                events=branch.events,
                reason="interaction_limit",
            ))
            continue

        point = np.asarray(hit.point)
        outward = np.asarray(hit.outward_normal)
        # The vector Snell form expects the normal to oppose the incoming ray.
        normal = outward if float(np.dot(branch.direction, outward)) < 0.0 else -outward
        cos1 = float(np.clip(-np.dot(branch.direction, normal), 0.0, 1.0))
        incidence = degrees(acos(cos1))
        # A disjoint multi-body scene supplies the material at the nearest hit.
        # The original single-body path retains exactly its previous physics.
        index = getattr(hit, "material_index", body.n)
        critical_deg = degrees(asin(1.0 / index))
        eta = index if branch.inside else 1.0 / index
        discriminant = 1.0 - eta * eta * (1.0 - cos1 * cos1)
        margin = incidence - critical_deg if branch.inside else None
        reflected = branch.direction + 2.0 * cos1 * normal
        reflected = _unit(reflected)

        mirror = getattr(hit, "ideal_mirror", False)
        if discriminant < 0.0 or mirror:
            ev = _event("mirror" if mirror else "tir", hit, incidence, margin,
                        branch.energy, branch.energy)
            stack.append(_Branch(
                source_id=branch.source_id,
                origin=point + ray_epsilon * reflected,
                direction=reflected,
                energy_s=branch.energy_s,
                energy_p=branch.energy_p,
                inside=branch.inside,
                depth=branch.depth + 1,
                events=branch.events + (ev,),
            ))
            continue

        cos2 = sqrt(max(discriminant, 0.0))
        transmitted = eta * branch.direction + (eta * cos1 - cos2) * normal
        transmitted = _unit(transmitted)
        den_s = eta * cos1 + cos2
        den_p = eta * cos2 + cos1
        rs = ((eta * cos1 - cos2) / den_s) ** 2 if abs(den_s) > _EPS else 1.0
        rp = ((eta * cos2 - cos1) / den_p) ** 2 if abs(den_p) > _EPS else 1.0
        rs, rp = float(np.clip(rs, 0.0, 1.0)), float(np.clip(rp, 0.0, 1.0))

        ts, tp = branch.energy_s * (1.0 - rs), branch.energy_p * (1.0 - rp)
        transmitted_energy = ts + tp
        tev = _event("refract", hit, incidence, margin, branch.energy, transmitted_energy)
        stack.append(_Branch(
            source_id=branch.source_id,
            origin=point + ray_epsilon * transmitted,
            direction=transmitted,
            energy_s=ts,
            energy_p=tp,
            inside=not branch.inside,
            depth=branch.depth + 1,
            events=branch.events + (tev,),
        ))

        r_s, r_p = branch.energy_s * rs, branch.energy_p * rp
        reflected_energy = r_s + r_p
        rev = _event("fresnel_reflect", hit, incidence, margin,
                     branch.energy, reflected_energy)
        stack.append(_Branch(
            source_id=branch.source_id,
            origin=point + ray_epsilon * reflected,
            direction=reflected,
            energy_s=r_s,
            energy_p=r_p,
            inside=branch.inside,
            depth=branch.depth + 1,
            events=branch.events + (rev,),
        ))

    return TraceResult(
        exits=tuple(exits),
        terminations=tuple(terminations),
        emitted_energy=float(emitted),
        pruned_energy=float(pruned),
        depth_limited_energy=float(depth_limited),
        branch_limited_energy=float(branch_limited),
    )


def project_exits_to_line(result: TraceResult, *, point, normal) -> list[dict]:
    """Project escaped rays onto an arbitrary plane, given as a point and a normal.

    ``normal`` faces the side the light arrives from, so a ray counts only when it is
    travelling into the plane.  ``coord`` is the signed in-plane position measured from
    ``point`` along ``(-n_y, n_x)``; for a horizontal plane whose normal points down
    that reduces to plain x, which is what the axis-aligned projection returned.
    """
    origin_p = np.asarray(tuple(point), dtype=float)
    n = np.asarray(tuple(normal), dtype=float)
    if origin_p.shape != (2,) or n.shape != (2,):
        raise ValueError("plane point and normal must be 2-vectors")
    if not np.isfinite(origin_p).all() or not np.isfinite(n).all():
        raise ValueError("plane point and normal must be finite")
    norm = float(np.linalg.norm(n))
    if norm <= _EPS:
        raise ValueError("plane normal must be non-zero")
    n = n / norm
    along = np.array((-n[1], n[0]))

    records = []
    for ray in result.exits:
        origin = np.asarray(ray.origin)
        direction = np.asarray(ray.direction)
        approach = float(np.dot(direction, n))
        if approach >= -_EPS:            # parallel, or receding from the plane
            continue
        t = float(np.dot(origin_p - origin, n)) / approach
        if t <= 0:
            continue
        crossing = origin + t * direction
        records.append({
            "source_id": ray.source_id,
            "exit_origin": ray.origin,
            "x": float(crossing[0]),
            "y": float(crossing[1]),
            "coord": float(np.dot(crossing - origin_p, along)),
            "direction": ray.direction,
            "energy": ray.energy,
            "events": ray.events,
        })
    return records


def project_exits_to_plane(result: TraceResult, *, axis: int, value: float,
                           travel_sign: int = 1) -> list[dict]:
    """Project escaped rays onto an axis-aligned plane without assigning valid routes.

    ``axis`` selects the plane normal: 0 for a vertical plane at constant x, 1 for a
    horizontal plane at constant y.  ``travel_sign`` states which way a ray must be
    travelling along that normal to count, so a reverse-propagated bundle needs no
    separate code path.  ``coord`` is the in-plane coordinate; ``x`` and ``y`` always
    carry the full crossing point.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (constant x) or 1 (constant y)")
    if travel_sign not in (-1, 0, 1):
        raise ValueError("travel_sign must be -1, 0, or 1")
    if not np.isfinite(value):
        raise ValueError("plane value must be finite")

    other = 1 - axis
    records = []
    for ray in result.exits:
        origin = np.asarray(ray.origin)
        direction = np.asarray(ray.direction)
        if abs(direction[axis]) <= _EPS:
            continue
        if travel_sign and direction[axis] * travel_sign <= 0:
            continue
        t = (value - origin[axis]) / direction[axis]
        if t <= 0:
            continue
        crossing = origin + t * direction
        records.append({
            "source_id": ray.source_id,
            "exit_origin": ray.origin,
            "x": float(crossing[0]),
            "y": float(crossing[1]),
            "coord": float(crossing[other]),
            "direction": ray.direction,
            "energy": ray.energy,
            "events": ray.events,
        })
    return records


def project_exits_to_x(result: TraceResult, sensor_x: float, *, forward_only: bool = True
                       ) -> list[dict]:
    """Project escaped rays to a vertical sensor plane without assigning valid routes."""
    return project_exits_to_plane(result, axis=0, value=sensor_x,
                                  travel_sign=1 if forward_only else 0)
