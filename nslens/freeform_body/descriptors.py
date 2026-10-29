"""Segment-label-free descriptors for Stage-1 freeform discovery."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .boundary import FreeformBody2D
from .trace import BoundaryEvent, ExitRay, TraceResult


@dataclass(frozen=True)
class Descriptor:
    names: tuple[str, ...]
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values)}


def _sign_changes(values: np.ndarray, tolerance: float) -> int:
    signs = np.sign(np.where(abs(values) <= tolerance, 0.0, values))
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.count_nonzero(nonzero != np.roll(nonzero, 1)))


def shape_descriptor(body: FreeformBody2D, *, sample_count: int = 256,
                     fourier_orders: int = 8) -> Descriptor:
    """Translation-, rotation-, scale-, and start-index-invariant shape descriptor."""
    # Use an integer number of samples per control interval.  Then changing which
    # interpolation anchor is called u=0 produces an exact cyclic sample shift rather
    # than a slightly different quadrature grid.
    sample_count = int(np.ceil(sample_count / body.control_count) * body.control_count)
    u, points = body.sample(sample_count)
    report = body.geometry_report(sample_count)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / sample_count
    eigenvalues = np.linalg.eigvalsh(covariance)
    aspect = float(np.sqrt(max(eigenvalues[-1], 1e-15) / max(eigenvalues[0], 1e-15)))
    radial = np.linalg.norm(centered, axis=1)
    radial_mean = max(float(radial.mean()), 1e-15)
    spectrum = np.abs(np.fft.rfft(radial / radial_mean)) / sample_count
    radial_modes = [float(spectrum[k]) if k < len(spectrum) else 0.0
                    for k in range(2, fourier_orders + 2)]

    curvature = body.curvature(u)
    dimensionless_k = curvature * report.perimeter
    concave_fraction = float(np.mean(dimensionless_k < -1e-4))
    curvature_changes = float(_sign_changes(dimensionless_k, tolerance=1e-4))
    compactness = float(4.0 * pi * report.signed_area / max(report.perimeter ** 2, 1e-15))
    q = np.quantile(dimensionless_k, (0.05, 0.25, 0.5, 0.75, 0.95))

    names = (
        "aspect_ratio",
        "isoperimetric_compactness",
        "concave_fraction",
        "curvature_sign_changes",
        "curvature_q05",
        "curvature_q25",
        "curvature_q50",
        "curvature_q75",
        "curvature_q95",
        *(f"radial_fourier_{k}" for k in range(2, fourier_orders + 2)),
    )
    values = np.asarray((aspect, compactness, concave_fraction, curvature_changes,
                         *q.tolist(), *radial_modes), dtype=float)
    return Descriptor(tuple(names), values)


def route_signature(ray_or_events: ExitRay | tuple[BoundaryEvent, ...], *,
                    gap_bins: int = 24) -> tuple:
    """Cyclic-boundary-origin-invariant signature for one escaped branch.

    Absolute boundary coordinates are omitted.  The signature retains event types and
    signed circular displacement between consecutive interaction footprints.  A revisit
    therefore remains visible while arbitrary control-point reindexing does not change
    the route.
    """
    events = ray_or_events.events if isinstance(ray_or_events, ExitRay) else ray_or_events
    kinds = tuple(event.kind for event in events)
    gaps = []
    for a, b in zip(events, events[1:]):
        gap = ((b.arc_s - a.arc_s + 0.5) % 1.0) - 0.5
        gaps.append(int(np.clip(np.rint(gap * gap_bins), -gap_bins // 2, gap_bins // 2)))
    return kinds, tuple(gaps)


def ranked_route_signatures(result: TraceResult, *,
                            minimum_tir_bounces: int = 0) -> tuple[tuple, ...]:
    """Routed signatures ranked by their total escaped energy.

    A physical route is normally shared by many pupil rays and may also contain
    multiple Fresnel descendants.  Ranking individual exit branches makes the archive
    hinge on one unusually bright ray.  Sum all interacted exit branches with the same
    signature instead, across every pupil sample and field, and use a stable signature
    ordering to break exact energy ties.

    Zero-event exits are deliberately excluded: they missed the element and therefore
    do not describe a route through it.
    """
    energy_by_signature: dict[tuple, float] = {}
    if minimum_tir_bounces < 0:
        raise ValueError("minimum_tir_bounces must be non-negative")
    for ray in result.exits:
        if not ray.events or ray.energy <= 0.0:
            continue
        if sum(event.kind == "tir" for event in ray.events) < minimum_tir_bounces:
            continue
        signature = route_signature(ray)
        energy_by_signature[signature] = (
            energy_by_signature.get(signature, 0.0) + ray.energy
        )
    return tuple(sorted(
        energy_by_signature,
        key=lambda signature: (-energy_by_signature[signature], repr(signature)),
    ))


def _footprint_count(events: list[BoundaryEvent], bins: int = 32) -> int:
    if not events:
        return 0
    occupied = np.zeros(bins, dtype=bool)
    for event in events:
        occupied[int(np.floor(event.arc_s * bins)) % bins] = True
    # Merge adjacent occupied bins into circular footprint components.
    if occupied.all():
        return 1
    return int(np.count_nonzero(occupied & ~np.roll(occupied, 1)))


def interaction_descriptor(result: TraceResult, *, energy_floor: float = 1e-8) -> Descriptor:
    """Energy-weighted optical-path morphology over escaped branches.

    This descriptor characterizes diversity; it is not an imaging reward.  Missed rays
    with no boundary events remain represented through ``zero_event_energy_fraction``.
    """
    rays = [ray for ray in result.exits if ray.energy > energy_floor]
    total = sum(ray.energy for ray in rays)
    if total <= 0:
        values = np.zeros(13, dtype=float)
        names = (
            "mean_interactions", "interaction_std", "tir_fraction",
            "fresnel_reflect_fraction", "refract_fraction", "revisit_fraction",
            "footprint_count", "route_entropy", "dominant_route_share",
            "zero_event_energy_fraction", "escaped_energy_fraction",
            "pruned_energy_fraction", "depth_limited_energy_fraction",
        )
        emitted = max(result.emitted_energy, 1e-30)
        values[10] = result.escaped_energy / emitted
        values[11] = result.pruned_energy / emitted
        values[12] = result.depth_limited_energy / emitted
        return Descriptor(names, values)

    weights = np.asarray([ray.energy / total for ray in rays])
    counts = np.asarray([len(ray.events) for ray in rays], dtype=float)
    mean_count = float(np.dot(weights, counts))
    count_std = float(np.sqrt(np.dot(weights, (counts - mean_count) ** 2)))

    # Zero-event exits are represented separately and must not dilute the relative
    # frequencies of actual surface-event types.
    weighted_event_total = float(np.dot(weights, counts))
    type_counts = {kind: 0.0 for kind in ("tir", "fresnel_reflect", "refract")}
    revisit = 0.0
    all_events: list[BoundaryEvent] = []
    signature_energy: dict[tuple, float] = {}
    zero_event = 0.0
    for ray, weight in zip(rays, weights):
        if not ray.events:
            zero_event += float(weight)
        all_events.extend(ray.events)
        for event in ray.events:
            if event.kind in type_counts:
                type_counts[event.kind] += float(weight)
        if len(ray.events) >= 2:
            seen = []
            repeated = False
            for event in ray.events:
                if any(abs(((event.arc_s - old + 0.5) % 1.0) - 0.5) < 1.0 / 32.0
                       for old in seen):
                    repeated = True
                    break
                seen.append(event.arc_s)
            revisit += float(weight) * float(repeated)
        signature = route_signature(ray)
        signature_energy[signature] = signature_energy.get(signature, 0.0) + ray.energy

    shares = np.asarray(list(signature_energy.values()), dtype=float) / total
    entropy = float(-np.sum(shares * np.log(np.maximum(shares, 1e-30))))
    dominant = float(np.max(shares)) if len(shares) else 0.0
    escaped_fraction = result.escaped_energy / max(result.emitted_energy, 1e-30)
    names = (
        "mean_interactions", "interaction_std", "tir_fraction",
        "fresnel_reflect_fraction", "refract_fraction", "revisit_fraction",
        "footprint_count", "route_entropy", "dominant_route_share",
        "zero_event_energy_fraction", "escaped_energy_fraction",
        "pruned_energy_fraction", "depth_limited_energy_fraction",
    )
    values = np.asarray((
        mean_count,
        count_std,
        type_counts["tir"] / max(weighted_event_total, 1e-30),
        type_counts["fresnel_reflect"] / max(weighted_event_total, 1e-30),
        type_counts["refract"] / max(weighted_event_total, 1e-30),
        revisit,
        float(_footprint_count(all_events)),
        entropy,
        dominant,
        zero_event,
        escaped_fraction,
        result.pruned_energy / max(result.emitted_energy, 1e-30),
        result.depth_limited_energy / max(result.emitted_energy, 1e-30),
    ), dtype=float)
    return Descriptor(names, values)
