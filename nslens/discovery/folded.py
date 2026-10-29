"""Finite-aperture folded routing, before imaging refinement or compaction."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from nslens.freeform_body.trace import InputRay, trace_bundle, project_exits_to_plane
from .polygon import exact_extent, route_key


@dataclass(frozen=True)
class RoutingBrief:
    beam_width_mm: float = 4.
    source_center: tuple = (55., 50.)
    aim_deg: float = -90.
    fields_deg: tuple = (-.5, 0., .5)
    rays_per_field: int = 9
    detector_x: float = 120.
    detector_center_y: float = 0.
    detector_half_height: float = 30.
    # Generic detector coordinates.  None preserves the historical vertical detector
    # encoded by detector_x/detector_center_y/detector_half_height.
    detector_axis: int = 0
    detector_travel_sign: int = 1
    detector_position: float | None = None
    detector_center: float | None = None
    detector_half_length: float | None = None
    clearance_mm: float = 2.
    source_clearance_bounds: tuple | None = None
    detector_clearance_bounds: tuple | None = None
    min_tir: int = 2
    min_folded_throughput: float = .25
    min_root_fraction: float = .60
    min_energy_per_root: float = .10
    min_tir_margin_deg: float = .5
    max_unresolved_fraction: float = 1e-4
    max_pruned_fraction: float = 1e-3
    max_closure_error: float = 1e-9
    max_interactions: int = 24
    # Zero preserves the historical throughput-only objective.  Positive values
    # penalize the energy-weighted RMS detector spot relative to the input width.
    spot_weight: float = 0.

    def __post_init__(self):
        from dataclasses import asdict
        for name, value in asdict(self).items():
            if value is None:
                continue
            if not np.isfinite(np.asarray(value, dtype=float)).all():
                raise ValueError(f"nonfinite routing requirement: {name}")
        if self.beam_width_mm < 1:
            raise ValueError("discovery beam must be at least 1 mm wide")
        if len(self.source_center) != 2 or not self.fields_deg or len(set(self.fields_deg)) != len(self.fields_deg):
            raise ValueError("invalid source or repeated/empty fields")
        if type(self.rays_per_field) is not int or self.rays_per_field < 5:
            raise ValueError("at least five independent source rays per field required")
        if type(self.min_tir) is not int or self.min_tir < 1:
            raise ValueError("positive integer TIR requirement required")
        if type(self.max_interactions) is not int or self.max_interactions < self.min_tir + 2:
            raise ValueError("interaction cap cannot realize requested path")
        if self.detector_axis not in (0, 1) or self.detector_travel_sign not in (-1, 1):
            raise ValueError("detector axis/sign must describe a directed x or y plane")
        if self.detector_half_length is not None and self.detector_half_length <= 0:
            raise ValueError("detector half length must be positive")
        for name in ("source_clearance_bounds", "detector_clearance_bounds"):
            bounds = getattr(self, name)
            if bounds is not None and (len(bounds) != 2 or bounds[0] < self.clearance_mm
                                       or bounds[0] >= bounds[1]):
                raise ValueError(f"{name} must be an increasing interval above minimum clearance")
        if (self.detector_half_height <= 0 or self.clearance_mm <= 0
                or self.min_tir_margin_deg < 0 or self.spot_weight < 0):
            raise ValueError("invalid lengths or TIR margin")
        for name in ("min_folded_throughput", "min_root_fraction", "min_energy_per_root",
                     "max_unresolved_fraction", "max_pruned_fraction", "max_closure_error"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")

    def rays(self):
        # Aperture is perpendicular to the common chief direction. Each angular field
        # illuminates the ENTIRE aperture; distinct IDs survive Fresnel branching.
        aim = np.deg2rad(self.aim_deg)
        tangent = np.array((-np.sin(aim), np.cos(aim)))
        rays = []
        for field in self.fields_deg:
            angle = np.deg2rad(self.aim_deg + field)
            for s in np.linspace(-self.beam_width_mm/2, self.beam_width_mm/2, self.rays_per_field):
                origin = np.asarray(self.source_center) + s * tangent
                rays.append(InputRay(tuple(origin), (np.cos(angle), np.sin(angle)), source_id=len(rays)))
        return rays

    def detector(self):
        """Return axis, plane coordinate, in-plane centre, half length, travel sign."""
        position = self.detector_x if self.detector_position is None else self.detector_position
        center = self.detector_center_y if self.detector_center is None else self.detector_center
        half = self.detector_half_height if self.detector_half_length is None else self.detector_half_length
        return self.detector_axis, position, center, half, self.detector_travel_sign


def _plane_clearance(lo, hi, point, axis, travel_sign):
    """Clearance from a launch/receive plane to a body lying in its travel direction."""
    coordinate = float(point[axis])
    return ((lo[axis] - coordinate) if travel_sign > 0 else
            (coordinate - hi[axis]))


def evaluate(codec, genes, brief, *, return_trace=False, geometry_samples=256):
    row = dict(genes=np.asarray(genes).tolist(), corners=codec.corners, feasible=False,
               status="geometry", source_rays=0, fields=[], quality=0., support=0.,
               route=None, violations={}, progress=(0., 0., 0.))
    try:
        body = codec.decode(genes)
        geometry = body.geometry_report(sample_count=geometry_samples)
        if not geometry.valid:
            row["reason"] = "; ".join(geometry.reasons)
            return row
        lo, hi = exact_extent(body)
        aim = np.asarray((np.cos(np.deg2rad(brief.aim_deg)), np.sin(np.deg2rad(brief.aim_deg))))
        source_axis = int(np.argmax(abs(aim)))
        if abs(aim[source_axis]) < 1 - 1e-10:
            raise ValueError("discovery clearance currently requires a cardinal source aim")
        source_sign = 1 if aim[source_axis] > 0 else -1
        detector_axis, detector_position, detector_center, detector_half_length, detector_sign = brief.detector()
        # Keep the complete curved body downstream of the source and upstream of the
        # directed detector plane.  exact_extent includes circular-arc bulges.
        source_clearance = _plane_clearance(lo, hi, brief.source_center, source_axis, source_sign)
        detector_clearance = _plane_clearance(lo, hi,
                                              (detector_position, detector_position),
                                              detector_axis, -detector_sign)
        if min(source_clearance, detector_clearance) < brief.clearance_mm:
            row["reason"] = "source/detector plane clearance"
            return row
        if (brief.source_clearance_bounds is not None
                and not brief.source_clearance_bounds[0] <= source_clearance <= brief.source_clearance_bounds[1]):
            row["reason"] = "source clearance bounds"
            return row
        if (brief.detector_clearance_bounds is not None
                and not brief.detector_clearance_bounds[0] <= detector_clearance <= brief.detector_clearance_bounds[1]):
            row["reason"] = "detector clearance bounds"
            return row
        rays = brief.rays()
        if any(body.contains(ray.origin) for ray in rays):
            row["reason"] = "source inside body"
            return row
    except (ValueError, FloatingPointError) as exc:
        row["reason"] = str(exc)
        return row
    trace = trace_bundle(body, rays, max_interactions=brief.max_interactions, energy_floor=1e-6)
    row["source_rays"] = len(rays)
    projected = project_exits_to_plane(trace, axis=detector_axis, value=detector_position,
                                       travel_sign=detector_sign)
    records = [r for r in projected if abs(r["coord"] - detector_center) <= detector_half_length]
    closure = abs(trace.closure_residual) / trace.emitted_energy
    unresolved = (trace.depth_limited_energy + trace.branch_limited_energy) / trace.emitted_energy
    pruned = trace.pruned_energy / trace.emitted_energy
    routes = defaultdict(float)
    violations = {"closure": max(0., closure/brief.max_closure_error-1),
                  "unresolved": max(0., unresolved/brief.max_unresolved_fraction-1),
                  "pruned": max(0., pruned/brief.max_pruned_fraction-1)}
    for i, field in enumerate(brief.fields_deg):
        rows = [r for r in records if r["source_id"]//brief.rays_per_field == i]
        folded, onefold, roots, folded_records = 0., 0., defaultdict(float), []
        for r in rows:
            tir = [e for e in r["events"] if e.kind == "tir"]
            if tir:
                onefold += r["energy"]
            if len(tir) >= brief.min_tir and all(e.critical_margin_deg >= brief.min_tir_margin_deg for e in tir):
                folded += r["energy"]
                roots[r["source_id"]] += r["energy"]
                routes[route_key(body, r["events"])] += r["energy"]
                folded_records.append(r)
        energy = folded / brief.rays_per_field
        support = sum(v >= brief.min_energy_per_root for v in roots.values()) / brief.rays_per_field
        if folded_records and folded > 0:
            coordinates = np.asarray([r["coord"] for r in folded_records], dtype=float)
            weights = np.asarray([r["energy"] for r in folded_records], dtype=float)
            centroid = float(np.average(coordinates, weights=weights))
            spot_rms = float(np.sqrt(np.average((coordinates-centroid)**2, weights=weights)))
        else:
            centroid = float(detector_center)
            spot_rms = float(2*detector_half_length)
        imaging_quality = energy / (1. + brief.spot_weight*spot_rms/brief.beam_width_mm)
        violations[f"field={field:g}/folded_energy"] = max(0., 1-energy/brief.min_folded_throughput)
        violations[f"field={field:g}/root_support"] = max(0., 1-support/brief.min_root_fraction)
        row["fields"].append(dict(field_deg=field, folded_throughput=energy, root_fraction=support,
                                  one_fold_throughput=onefold/brief.rays_per_field,
                                  detector_centroid=centroid, spot_rms=spot_rms,
                                  imaging_quality=imaging_quality))
    entered_roots = {r.source_id for r in (*trace.exits, *trace.terminations) if r.events}
    row.update(status="measured", feasible=all(v == 0 for v in violations.values()),
               violations=violations, quality=min(f["imaging_quality"] for f in row["fields"]),
               worst_spot_rms=max(f["spot_rms"] for f in row["fields"]),
               worst_folded_throughput=min(f["folded_throughput"] for f in row["fields"]),
               support=min(f["root_fraction"] for f in row["fields"]),
               progress=(min(f["folded_throughput"] for f in row["fields"]),
                         min(f["one_fold_throughput"] for f in row["fields"]), len(entered_roots)/len(rays)),
               route=repr(max(routes, key=routes.get)) if routes else None,
               route_count=len(routes), closure=closure, unresolved=unresolved, pruned=pruned,
               source_clearance=float(source_clearance), detector_clearance=float(detector_clearance),
               bounds=[lo.tolist(), hi.tolist()], vertices=body.corners.tolist(), sags=body.sags.tolist())
    if return_trace:
        return row, body, trace
    return row


def ranking(row, feedback=True):
    if row["status"] != "measured":
        return (0, 0., 0., 0., 0.)
    if row["feasible"]:
        return (2, row["quality"], row["support"], 0., 0.)
    if not feedback:
        return (0, 0., 0., 0., 0.)
    # Minimize worst relative requirement violation; break ties with actual routing
    # progress. No weighted sum, prescribed surface sequence, or reward for trapping.
    return (1, -max(row["violations"].values()), *row["progress"])
