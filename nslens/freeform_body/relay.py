"""Stage-1 AR/VR three-field relay, stated in the reverse (collimated-in) direction.

Ray paths are reciprocal, so a body that takes three microdisplay points to three
collimated eyebox beams is the same body that takes three collimated eyebox fields to
three focused display-plane spots.  Stating the problem in reverse buys two things:

* the eyebox becomes the *entrance* aperture, so every field fills the fixed box by
  construction and no field can score off a single ray through a thin slice of it;
* image quality is a spot size again, which is what the Stage-1 evaluator in
  ``problem.py`` already measures.

What it costs is that field of view stops being an emergent quantity: the three input
angles are part of the specification, not a discovered descriptor.

Nothing here names a face, a fold, a bounce count, or a route.  It states where light
enters, at what angles, and where it must land.  How the body gets it there is left to
the search and is reported only through the diversity descriptors.

A feasibility ladder (``relay_field_report``) grades candidates that deliver nothing,
so the emitter has a gradient out of the dead region.  It is modelled on the staged
status score of the two-stage freeform-prism GA, but that method's bands are keyed to
named surfaces (the second hit on A, mirror B, surface C), which would smuggle a
prescribed path back into a label-free objective.  The bands here are keyed to how far
light *got* instead: entered, escaped, reached the plane, landed in the window.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from math import cos, exp, radians, sin
from typing import Sequence

import numpy as np

from .codec import FreeformBodyCodec
from .descriptors import (interaction_descriptor, ranked_route_signatures,
                          shape_descriptor)
from .problem import DiscoveryEvaluation, FieldScore
from .trace import InputRay, project_exits_to_line, trace_bundle

# --- the fixed specification --------------------------------------------------------
# The fixed object plane and eyebox define the relay layout; the OBJECT PLANE is tilted:
#
#       \  ->  (element)  ->  |
#     object                 eyebox
#
# Their relative offset and the tilt ask for off-axis correction without assigning any
# optical role to a named body face.
EYEBOX_X = 150.0           # eyebox plane; reversed light enters here travelling -x
EYEBOX_Y = 30.0             # eyebox centre, on the axis
EYEBOX_HALF = 3.0          # eyebox half-height
EYEBOX_TILT_DEG = -10.0    # negative is clockwise from the vertical eyebox plane
OBJECT_CENTER = (0.0, 0.0)         # object plane centre, on the same axis
OBJECT_TILT_DEG = 135.0            # orientation of the object plane itself ("\")
TARGET_OFFSETS = (-6.0, 0.0, 6.0)  # the three object points, 6 mm apart ALONG the plane
FIELD_DEGS = (-6.0, 0.0, 6.0)      # 12 deg full FOV
N_PUPIL = 17               # rays per field across the eyebox
SPOT_SCALE = 1.0           # mm, spot tolerance in the fitness
TARGET_HALF_WIDTH = 2.0    # mm, half-width of an object target window

# Ladder constants.  Progress is capped strictly below 1 so that a band can never be
# overtaken from below, including after the mean-score tiebreak is added.
_BAND_INVALID = -1.0
_PROGRESS_CAP = 0.99
_TIEBREAK = 1e-3


@dataclass(frozen=True)
class EyeboxSpec:
    """The fixed entrance aperture.  Its position and size are not searchable."""

    x: float = EYEBOX_X
    y_center: float = EYEBOX_Y
    half_height: float = EYEBOX_HALF
    tilt_deg: float = EYEBOX_TILT_DEG
    ray_count: int = N_PUPIL

    def __post_init__(self) -> None:
        if not np.isfinite(
                (self.x, self.y_center, self.half_height, self.tilt_deg)).all():
            raise ValueError("eyebox geometry must be finite")
        if self.half_height <= 0:
            raise ValueError("eyebox half-height must be positive")
        if not isinstance(self.ray_count, (int, np.integer)) or self.ray_count < 1:
            raise ValueError("an eyebox needs at least one ray")

    @property
    def along(self) -> np.ndarray:
        """Unit vector from the lower to upper end of the tilted aperture."""
        theta = radians(self.tilt_deg)
        return np.array((-sin(theta), cos(theta)))

    @property
    def inward_normal(self) -> np.ndarray:
        """Chief-ray direction, perpendicular to the aperture and toward the body."""
        theta = radians(self.tilt_deg)
        return np.array((-cos(theta), -sin(theta)))

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        center = np.array((self.x, self.y_center))
        points = center + np.array((-self.half_height, self.half_height))[:, None] \
            * self.along
        return tuple(tuple(float(value) for value in point) for point in points)


@dataclass(frozen=True)
class ObjectPlane:
    """The fixed, tilted object plane.  Its pose is specification, not genotype."""

    center: tuple[float, float] = OBJECT_CENTER
    tilt_deg: float = OBJECT_TILT_DEG

    def __post_init__(self) -> None:
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("object-plane centre must be a finite 2-vector")
        if not np.isfinite(self.tilt_deg):
            raise ValueError("object-plane tilt must be finite")

    @property
    def along(self) -> np.ndarray:
        """Unit vector along the plane; target offsets are measured on this."""
        t = radians(self.tilt_deg)
        return np.array((cos(t), sin(t)))

    @property
    def normal(self) -> np.ndarray:
        """Unit normal, perpendicular to ``along`` and facing the incoming light."""
        t = radians(self.tilt_deg)
        return np.array((sin(t), -cos(t)))

    def point_at(self, offset: float) -> np.ndarray:
        return np.asarray(self.center, dtype=float) + float(offset) * self.along

    def signed_distance(self, points) -> np.ndarray:
        """Distance from the plane along the normal; positive is the element side."""
        rel = np.asarray(points, dtype=float) - np.asarray(self.center, dtype=float)
        return rel @ self.normal


@dataclass(frozen=True)
class DisplayTarget:
    """One object point, located by its signed offset along the tilted plane."""

    offset: float
    half_width: float = TARGET_HALF_WIDTH

    def __post_init__(self) -> None:
        if not np.isfinite((self.offset, self.half_width)).all() or self.half_width <= 0:
            raise ValueError("object target values must be finite and half-width positive")


@dataclass(frozen=True)
class VerticalBlocker:
    """Finite opaque stop placed in the free-space leg before the object plane."""

    x: float
    y_center: float
    half_height: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.x, self.y_center, self.half_height)).all():
            raise ValueError("blocker geometry must be finite")
        if self.half_height <= 0.0:
            raise ValueError("blocker half-height must be positive")

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.x, self.y_center - self.half_height),
                (self.x, self.y_center + self.half_height))

    def intersection(self, start, end) -> tuple[float, float] | None:
        """Intersection with the open line segment ``start -> end``, if blocked."""
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite((a, b)).all():
            raise ValueError("blocker intersection needs two finite 2-vectors")
        dx = float(b[0] - a[0])
        if abs(dx) <= 1e-12:
            return None
        t = (self.x - float(a[0])) / dx
        if not 0.0 < t < 1.0:
            return None
        y = float(a[1] + t * (b[1] - a[1]))
        if abs(y - self.y_center) > self.half_height:
            return None
        return float(self.x), y

    def distance_to_points(self, points) -> np.ndarray:
        """Euclidean distance from points to this finite vertical segment."""
        p = np.asarray(points, dtype=float)
        y0, y1 = self.endpoints[0][1], self.endpoints[1][1]
        dy = p[..., 1] - np.clip(p[..., 1], y0, y1)
        return np.hypot(p[..., 0] - self.x, dy)


@dataclass(frozen=True)
class RelayFieldReport:
    """How far one field's light travelled, on the label-free feasibility ladder."""

    source_id: int
    target_offset: float
    band: int
    progress: float
    score: float
    throughput: float
    rms_error: float
    fitness: float
    delivered_energy: float
    emitted_energy: float

    def as_field_score(self) -> FieldScore:
        return FieldScore(
            source_id=self.source_id,
            throughput=self.throughput,
            rms_error=self.rms_error,
            fitness=self.fitness,
            delivered_energy=self.delivered_energy,
            emitted_energy=self.emitted_energy,
        )


@dataclass(frozen=True)
class RelayEvaluation(DiscoveryEvaluation):
    """A ``DiscoveryEvaluation`` that also records which parity the design used."""

    parity: int = 1
    field_reports: tuple[RelayFieldReport, ...] = ()

    @property
    def worst_band(self) -> int:
        return min((report.band for report in self.field_reports), default=-1)


def collimated_eyebox_rays(eyebox: EyeboxSpec = EyeboxSpec(),
                           field_degs: Sequence[float] = FIELD_DEGS,
                           ) -> tuple[InputRay, ...]:
    """Fill the fixed eyebox with one collimated bundle per field angle.

    The chief direction is normal to the eyebox: reversed light enters at the eye and
    travels back toward the element.  Field angle rotates relative to that normal;
    every ray of a field is parallel, which makes the bundle collimated at the eyebox.
    """
    angles = tuple(float(a) for a in field_degs)
    if not angles:
        raise ValueError("a relay needs at least one field angle")
    if not np.isfinite(angles).all():
        raise ValueError("field angles must be finite")
    center = np.array((eyebox.x, eyebox.y_center))
    # One sample is an explicit chief-ray diagnostic and therefore starts at the
    # aperture centre.  Two or more samples span the complete eyebox.
    offsets = (np.array((0.0,)) if eyebox.ray_count == 1 else
               np.linspace(-eyebox.half_height, eyebox.half_height,
                           eyebox.ray_count))
    rays = []
    for source_id, degrees_ in enumerate(angles):
        theta = radians(eyebox.tilt_deg + degrees_)
        direction = (-cos(theta), -sin(theta))
        for offset in offsets:
            origin = center + offset * eyebox.along
            rays.append(InputRay(origin=(float(origin[0]), float(origin[1])),
                                 direction=direction,
                                 source_id=source_id))
    return tuple(rays)


def relay_field_report(*, source_id: int, target: DisplayTarget, records: Sequence[dict],
                       emitted: float, entered: float, escaped: float,
                       minimum_throughput: float, spot_scale: float,
                       distance_scale: float) -> RelayFieldReport:
    """Place one field on the ladder and, at the top band, score it for real.

    Bands, worst to best: never touched the body; entered but never escaped; escaped
    but never reached the image plane; reached it but missed the window; landed in the
    window.  Only the top band produces an optical score.
    """
    emitted = max(float(emitted), 1e-30)
    in_window = [r for r in records
                 if abs(r["coord"] - target.offset) <= target.half_width]
    delivered = float(sum(r["energy"] for r in in_window))
    throughput = delivered / emitted
    plane_energy = float(sum(r["energy"] for r in records))

    rms = float("inf")
    fitness = 0.0
    if entered <= 0.0:
        band, progress = 0, 0.0
    elif escaped <= 0.0:
        band, progress = 1, entered / emitted
    elif plane_energy <= 0.0:
        band, progress = 2, escaped / emitted
    elif throughput < minimum_throughput:
        # Energy-weighted mean miss distance: smooth, and not dominated by one stray ray.
        mean_miss = sum(r["energy"] * abs(r["coord"] - target.offset)
                        for r in records) / plane_energy
        band, progress = 3, 1.0 / (1.0 + mean_miss / distance_scale)
    else:
        mse = sum(r["energy"] * (r["coord"] - target.offset) ** 2
                  for r in in_window) / delivered
        rms = float(np.sqrt(max(mse, 0.0)))
        fitness = throughput * exp(-rms / spot_scale)
        band, progress = 4, fitness

    score = (4.0 + fitness) if band == 4 else band + min(float(progress), _PROGRESS_CAP)
    return RelayFieldReport(
        source_id=source_id,
        target_offset=float(target.offset),
        band=int(band),
        progress=float(progress),
        score=float(score),
        throughput=float(throughput),
        rms_error=rms,
        fitness=float(fitness),
        delivered_energy=delivered,
        emitted_energy=float(emitted),
    )


@dataclass(frozen=True)
class FreeformRelayProblem:
    """A fixed-eyebox, fixed-display AR/VR relay, traced in reverse."""

    codec: FreeformBodyCodec
    eyebox: EyeboxSpec = EyeboxSpec()
    field_degs: tuple[float, ...] = FIELD_DEGS
    object_plane: ObjectPlane = ObjectPlane()
    targets: tuple[DisplayTarget, ...] = dataclass_field(
        default_factory=lambda: tuple(DisplayTarget(s) for s in TARGET_OFFSETS))
    blocker: VerticalBlocker | None = None
    minimum_tir_bounces: int = 0
    spot_scale: float = SPOT_SCALE
    # 1e-2, not the 1e-3 this started at: a 1200-eval pilot showed the looser floor
    # admitting designs whose optics were fine (0.4-1.0 mm RMS) but which routed ~99%
    # of the light somewhere other than the target window.  Those squat on niches that
    # a real design should hold.  Candidates below the floor are not discarded -- they
    # sit on ladder band 3 and still steer the emitter.
    minimum_field_throughput: float = 1e-2
    distance_scale: float = 10.0   # mm; grades a missed window and a plane overshoot
    separation_scale: float = 1.0
    clearance: float = 2.0
    max_interactions: int = 16
    energy_floor: float = 1e-4

    def __post_init__(self) -> None:
        if len(self.field_degs) != len(self.targets):
            raise ValueError("each field angle needs exactly one display target")
        if not self.targets:
            raise ValueError("a relay needs at least one field")
        angles = np.asarray(self.field_degs, dtype=float)
        if not np.all(np.diff(angles) > 0):
            raise ValueError("field angles must be strictly increasing")
        offsets = np.asarray([t.offset for t in self.targets], dtype=float)
        if not np.all(np.diff(offsets) > 0):
            raise ValueError("object targets must be strictly increasing along the plane")
        # Disjoint windows make ordering automatic: a design cannot satisfy every field
        # and still scramble the image, so no separate monotonicity gate is needed.
        widths = np.asarray([t.half_width for t in self.targets], dtype=float)
        if np.any(widths[:-1] + widths[1:] >= np.diff(offsets)):
            raise ValueError("object target windows must be disjoint")
        scalars = (self.spot_scale, self.minimum_field_throughput,
                   self.distance_scale, self.separation_scale, self.clearance,
                   self.energy_floor)
        if not np.isfinite(scalars).all():
            raise ValueError("relay scalar settings must be finite")
        if min(self.spot_scale, self.distance_scale, self.separation_scale) <= 0:
            raise ValueError("spot, distance and separation scales must be positive")
        if not 0.0 <= self.minimum_field_throughput <= 1.0:
            raise ValueError("minimum_field_throughput must lie in [0, 1]")
        if self.clearance < 0 or self.energy_floor < 0:
            raise ValueError("clearance and energy_floor must be non-negative")
        if self.max_interactions < 1:
            raise ValueError("max_interactions must be positive")
        if (not isinstance(self.minimum_tir_bounces, (int, np.integer))
                or self.minimum_tir_bounces < 0):
            raise ValueError("minimum_tir_bounces must be a non-negative integer")
        # The eyebox must lie on the element side of the object plane, or the two
        # fixed surfaces are on top of each other and nothing can image between them.
        eyebox_side = float(self.object_plane.signed_distance(
            [(self.eyebox.x, self.eyebox.y_center)])[0])
        if eyebox_side <= 0.0:
            raise ValueError("the eyebox must sit on the element side of the object plane")

    @property
    def rays(self) -> tuple[InputRay, ...]:
        return collimated_eyebox_rays(self.eyebox, self.field_degs)

    def clearance_report(self, body) -> tuple[tuple[str, ...], float]:
        """How the body violates the fixed planes, and by how far.

        The overshoot matters: grading a too-tall body by its self-intersection
        distance would tell the emitter nothing about the thing it needs to change.
        """
        _, points = body.sample(256)
        reasons: list[str] = []
        overshoot = 0.0
        # Signed distance, so a tilted plane is handled like any other.
        past_object = self.clearance - float(
            self.object_plane.signed_distance(points).min())
        if past_object > 0.0:
            reasons.append("body reaches the object plane")
            overshoot = max(overshoot, past_object)
        center = np.array((self.eyebox.x, self.eyebox.y_center))
        inside_distances = (points - center) @ self.eyebox.inward_normal
        past_eyebox = self.clearance - float(inside_distances.min())
        if past_eyebox > 0.0:
            reasons.append("body reaches the eyebox plane")
            overshoot = max(overshoot, past_eyebox)
        if self.blocker is not None:
            blocker_distance = float(np.min(self.blocker.distance_to_points(points)))
            p0, p1 = self.blocker.endpoints
            direction = np.asarray(p1) - np.asarray(p0)
            length = float(np.linalg.norm(direction))
            crosses = any(hit.t <= length + 1e-9 for hit in
                          body.all_intersections(p0, direction / length))
            embedded = (body.contains(p0) or body.contains(p1)
                        or body.contains((self.blocker.x, self.blocker.y_center)))
            if crosses or embedded or blocker_distance < self.clearance:
                reasons.append("body reaches the direct-path blocker")
                overshoot = max(overshoot, self.clearance - blocker_distance
                                if blocker_distance < self.clearance else self.clearance)
        return tuple(reasons), overshoot

    def _rejected(self, score: float, reason: str, geometry=None) -> RelayEvaluation:
        return RelayEvaluation(
            feasible=False, fitness=0.0, fields=(), shape=None, interaction=None,
            route_keys=(), geometry_report=geometry, trace_result=None,
            reason=reason, search_score=float(score),
        )

    def evaluate(self, genes) -> RelayEvaluation:
        try:
            body = self.codec.decode(genes)
            geometry = body.geometry_report()
        except (ValueError, FloatingPointError) as exc:
            return self._rejected(_BAND_INVALID, str(exc))

        clearance_reasons, overshoot = self.clearance_report(body)
        reasons = tuple(geometry.reasons) + clearance_reasons
        if not geometry.valid or reasons:
            # The graded analogue of the GA's wedge penalty, on both axes it can fail:
            # a body that only just self-touches ranks above one grossly folded through
            # itself, and one just past a fixed plane ranks above one far past it.
            separation = float(geometry.min_nonlocal_separation)
            separation_progress = (min(separation / self.separation_scale, _PROGRESS_CAP)
                                   if np.isfinite(separation) else _PROGRESS_CAP)
            clearance_progress = 1.0 / (1.0 + overshoot / self.distance_scale)
            progress = min(separation_progress, clearance_progress, _PROGRESS_CAP)
            return self._rejected(_BAND_INVALID + progress, "; ".join(reasons), geometry)

        rays = self.rays
        trace = trace_bundle(body, rays, max_interactions=self.max_interactions,
                             energy_floor=self.energy_floor)
        projected = project_exits_to_line(trace, point=self.object_plane.center,
                                          normal=self.object_plane.normal)
        if self.blocker is not None:
            projected = [record for record in projected
                         if self.blocker.intersection(record["exit_origin"],
                                                      (record["x"], record["y"])) is None]
        def follows_allowed_route(events) -> bool:
            return (bool(events) and
                    sum(event.kind == "tir" for event in events)
                    >= self.minimum_tir_bounces)

        # A ray that never touched the body is not routed light, however it crosses.
        # In blocked-direct mode, zero-TIR energy is also physically inadmissible and
        # cannot contribute to either the ladder or the final image score.
        routed = [record for record in projected
                  if follows_allowed_route(record["events"])]

        source_ids = tuple(range(len(self.targets)))
        emitted = {s: 0.0 for s in source_ids}
        for ray in rays:
            emitted[ray.source_id] += ray.energy_s + ray.energy_p
        escaped = {s: 0.0 for s in source_ids}
        for exit_ray in trace.exits:
            if follows_allowed_route(exit_ray.events):
                escaped[exit_ray.source_id] += exit_ray.energy
        entered = dict(escaped)
        for stopped in trace.terminations:
            if follows_allowed_route(stopped.events):
                entered[stopped.source_id] += stopped.energy
        by_source = {s: [r for r in routed if r["source_id"] == s] for s in source_ids}

        # Reflection parity decides whether field angle maps to display x increasing or
        # decreasing.  Pinning one pairing would silently exclude every design with the
        # other parity -- that is, would prescribe an even or odd number of reflections.
        # Both pairings are scored and the better one is kept.
        best: tuple[float, int, tuple[RelayFieldReport, ...]] | None = None
        # A single on-axis field has no ordering parity to discover.
        parities = (1,) if len(self.targets) == 1 else (1, -1)
        for parity in parities:
            ordered = self.targets if parity == 1 else tuple(reversed(self.targets))
            reports = tuple(
                relay_field_report(
                    source_id=s, target=ordered[s], records=by_source[s],
                    emitted=emitted[s], entered=entered[s], escaped=escaped[s],
                    minimum_throughput=self.minimum_field_throughput,
                    spot_scale=self.spot_scale, distance_scale=self.distance_scale,
                )
                for s in source_ids
            )
            scores = [report.score for report in reports]
            search_score = min(scores) + _TIEBREAK * float(np.mean(scores))
            if best is None or search_score > best[0]:
                best = (search_score, parity, reports)

        search_score, parity, reports = best
        feasible = all(report.band == 4 for report in reports)
        # Geometric mean: a dead field cannot be hidden behind two good ones.
        fitness = (float(np.exp(np.mean(np.log(np.maximum(
            [report.fitness for report in reports], 1e-30))))) if feasible else 0.0)
        route_keys = ranked_route_signatures(
            trace, minimum_tir_bounces=self.minimum_tir_bounces)
        return RelayEvaluation(
            feasible=feasible,
            fitness=fitness,
            fields=tuple(report.as_field_score() for report in reports),
            shape=shape_descriptor(body),
            interaction=interaction_descriptor(trace),
            route_keys=route_keys,
            geometry_report=geometry,
            trace_result=trace,
            reason=None if feasible else
            f"worst field on ladder band {min(r.band for r in reports)}",
            search_score=float(search_score),
            parity=int(parity),
            field_reports=reports,
        )
