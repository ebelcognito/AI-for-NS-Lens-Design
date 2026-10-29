"""A corner-and-arc element: spherical faces, sampled corners, analytic intersection.

This replaces the periodic-spline genotype for relay work.  A body is K corners joined
by K circular arcs -- the classical spherical-face element -- but each corner is drawn
from its own bounded region rather than being derived from a thickness and two tilts.
That is the point: it gives genuinely distinct starting geometries while keeping every
face a sphere.

Two practical gains over the spline body:

* intersection is closed-form (ray/circle), not a polyline scan plus Brent refinement,
  so tracing is several times faster;
* a design is describable as corners and sags, which is what a sequential optical-design
  tool expects.

The class deliberately matches the interface the Stage-1 tracer and descriptors already
use -- ``nearest_intersection``, ``contains``, ``sample``, ``curvature``, ``control_count``,
``geometry_report`` -- so the relay problem, ladder, archive and renderer are unchanged.
"""
from __future__ import annotations

from math import atan2, cos, hypot, pi, sin, sqrt
from typing import Iterable

import numpy as np
from scipy.spatial.distance import cdist

from .boundary import RI_MAX, RI_MIN, BoundaryHit, GeometryReport, _signed_polygon_area

_EPS = 1e-12
_STRAIGHT = 1e-9          # |sag| below this is a flat face


class ArcPolygonBody:
    """K corners joined by K circular arcs, in a homogeneous medium of index ``n``.

    ``sags[i]`` is the bulge of the arc from corner ``i`` to corner ``i+1``, measured
    at its midpoint along the outward normal.  Positive bulges outward (convex), zero
    is a flat face, negative is concave.  Winding is canonicalized counter-clockwise so
    the reported normal is always outward.
    """

    def __init__(self, corners: Iterable[Iterable[float]], sags: Iterable[float], n: float):
        pts = np.asarray(tuple(corners), dtype=float)
        sag = np.asarray(tuple(sags), dtype=float)
        if pts.ndim != 2 or pts.shape[1:] != (2,):
            raise ValueError("corners must have shape (K, 2)")
        if len(pts) < 3:
            raise ValueError("an arc polygon needs at least three corners")
        if sag.shape != (len(pts),):
            raise ValueError("there must be exactly one sag per edge")
        if not np.isfinite(pts).all() or not np.isfinite(sag).all():
            raise ValueError("corners and sags must be finite")
        if not RI_MIN <= float(n) <= RI_MAX:
            raise ValueError(f"refractive index must be in [{RI_MIN:.2f}, {RI_MAX:.2f}]")
        area = _signed_polygon_area(pts)
        if abs(area) <= _EPS:
            raise ValueError("corner polygon has zero signed area")
        if area < 0:
            # Reversing vertex order also reverses which edge each sag belongs to.
            pts = pts[::-1].copy()
            sag = sag[::-1].copy()
            sag = np.roll(sag, -1)

        self._corners = pts
        self._corners.setflags(write=False)
        self._sags = sag
        self._sags.setflags(write=False)
        self.n = float(n)
        self._build_edges()

    # ------------------------------------------------------------------ construction
    def _build_edges(self) -> None:
        pts = self._corners
        following = np.roll(pts, -1, axis=0)
        chord = following - pts
        length = np.linalg.norm(chord, axis=1)
        if float(length.min()) <= _EPS:
            raise ValueError("coincident corners leave a zero-length edge")
        # Counter-clockwise winding puts the outward normal at (dy, -dx).
        normals = np.stack((chord[:, 1], -chord[:, 0]), axis=1) / length[:, None]
        midpoints = pts + 0.5 * chord

        self._edges = []
        for i in range(len(pts)):
            s = float(self._sags[i])
            half = 0.5 * float(length[i])
            if abs(s) < _STRAIGHT:
                self._edges.append(dict(kind="line", start=pts[i], end=following[i],
                                        normal=normals[i], length=float(length[i]),
                                        curvature=0.0))
                continue
            radius = (half * half + s * s) / (2.0 * abs(s))
            center = midpoints[i] - np.sign(s) * (radius - abs(s)) * normals[i]
            theta0 = atan2(pts[i][1] - center[1], pts[i][0] - center[0])
            theta1 = atan2(following[i][1] - center[1], following[i][0] - center[0])
            apex = midpoints[i] + s * normals[i]
            theta_apex = atan2(apex[1] - center[1], apex[0] - center[0])
            # Pick the sweep direction that actually passes through the apex, which is
            # what distinguishes a minor arc from a major one when |sag| > half-chord.
            ccw = (theta1 - theta0) % (2.0 * pi)
            through = (theta_apex - theta0) % (2.0 * pi)
            sweep = ccw if through <= ccw else ccw - 2.0 * pi
            if abs(sweep) <= _EPS:
                raise ValueError("degenerate arc sweep")
            self._edges.append(dict(kind="arc", center=center, radius=radius,
                                    theta0=theta0, sweep=sweep,
                                    outward=float(np.sign(s)),
                                    length=radius * abs(sweep),
                                    curvature=float(np.sign(s)) / radius))

        # Column arrays so evaluate/curvature/normal can run without a Python loop:
        # sample() is called hundreds of times per trace and dominated the cost.
        self._is_arc = np.array([e["kind"] == "arc" for e in self._edges])
        self._start = np.stack([e["start"] if e["kind"] == "line" else e["center"]
                                for e in self._edges])
        self._end = np.stack([e["end"] if e["kind"] == "line" else e["center"]
                              for e in self._edges])
        self._center = np.stack([e["center"] if e["kind"] == "arc" else e["start"]
                                 for e in self._edges])
        self._radius = np.array([e["radius"] if e["kind"] == "arc" else 0.0
                                 for e in self._edges])
        self._theta0 = np.array([e["theta0"] if e["kind"] == "arc" else 0.0
                                 for e in self._edges])
        self._sweep = np.array([e["sweep"] if e["kind"] == "arc" else 0.0
                                for e in self._edges])
        self._outward = np.array([e["outward"] if e["kind"] == "arc" else 0.0
                                  for e in self._edges])
        self._flat_normal = np.stack([e["normal"] if e["kind"] == "line"
                                      else np.zeros(2) for e in self._edges])
        self._curvatures = np.array([e["curvature"] for e in self._edges])

        lengths = np.array([edge["length"] for edge in self._edges])
        self._perimeter = float(lengths.sum())
        if self._perimeter <= _EPS:
            raise ValueError("boundary has zero perimeter")
        self._cumulative = np.concatenate(([0.0], np.cumsum(lengths))) / self._perimeter

    # ------------------------------------------------------------------ description
    @property
    def corners(self) -> np.ndarray:
        return self._corners.copy()

    @property
    def sags(self) -> np.ndarray:
        return self._sags.copy()

    @property
    def control_count(self) -> int:
        return len(self._corners)

    @property
    def perimeter(self) -> float:
        return self._perimeter

    def _locate(self, u: float) -> tuple[int, float]:
        """Map normalized arclength to (edge index, fraction along that edge)."""
        uw = float(u) % 1.0
        index = int(np.searchsorted(self._cumulative, uw, side="right") - 1)
        index = min(max(index, 0), len(self._edges) - 1)
        span = self._cumulative[index + 1] - self._cumulative[index]
        return index, (uw - self._cumulative[index]) / max(span, _EPS)

    def _point_on(self, index: int, fraction: float) -> np.ndarray:
        edge = self._edges[index]
        if edge["kind"] == "line":
            return edge["start"] + fraction * (edge["end"] - edge["start"])
        theta = edge["theta0"] + fraction * edge["sweep"]
        return edge["center"] + edge["radius"] * np.array((cos(theta), sin(theta)))

    def _normal_on(self, index: int, fraction: float) -> np.ndarray:
        edge = self._edges[index]
        if edge["kind"] == "line":
            return edge["normal"]
        point = self._point_on(index, fraction)
        radial = (point - edge["center"]) / edge["radius"]
        return radial * edge["outward"]

    def _locate_many(self, u) -> tuple[np.ndarray, np.ndarray]:
        uw = np.asarray(u, dtype=float) % 1.0
        index = np.clip(np.searchsorted(self._cumulative, uw, side="right") - 1,
                        0, len(self._edges) - 1)
        span = self._cumulative[index + 1] - self._cumulative[index]
        return index, (uw - self._cumulative[index]) / np.maximum(span, _EPS)

    def evaluate(self, u) -> np.ndarray:
        scalar = np.asarray(u).ndim == 0
        values = np.atleast_1d(np.asarray(u, dtype=float))
        index, f = self._locate_many(values)
        straight = self._start[index] + f[:, None] * (self._end[index] - self._start[index])
        theta = self._theta0[index] + f * self._sweep[index]
        curved = self._center[index] + self._radius[index][:, None] * np.stack(
            (np.cos(theta), np.sin(theta)), axis=1)
        out = np.where(self._is_arc[index][:, None], curved, straight)
        return out[0] if scalar else out

    def sample(self, count: int = 256) -> tuple[np.ndarray, np.ndarray]:
        if count < max(32, 2 * self.control_count):
            raise ValueError("sample count is too small for a geometry audit")
        u = np.linspace(0.0, 1.0, count, endpoint=False)
        return u, self.evaluate(u)

    def curvature(self, u) -> np.ndarray:
        scalar = np.asarray(u).ndim == 0
        index, _ = self._locate_many(np.atleast_1d(np.asarray(u, dtype=float)))
        out = self._curvatures[index]
        return out[0] if scalar else out

    def outward_normal(self, u) -> np.ndarray:
        scalar = np.asarray(u).ndim == 0
        values = np.atleast_1d(np.asarray(u, dtype=float))
        index, f = self._locate_many(values)
        theta = self._theta0[index] + f * self._sweep[index]
        radial = np.stack((np.cos(theta), np.sin(theta)), axis=1) * self._outward[index][:, None]
        out = np.where(self._is_arc[index][:, None], radial, self._flat_normal[index])
        return out[0] if scalar else out

    def arclength_coordinates(self, u, table_size: int = 2048) -> np.ndarray:
        """Identity: this body is already parameterized by normalized arclength."""
        return np.asarray(u, dtype=float) % 1.0

    # An off-axis direction, so a probe ray does not graze a flat face or clip a corner.
    _PROBE = (0.9331234, 0.3595678)

    def contains(self, point: Iterable[float], sample_count: int = 0) -> bool:
        """Exact even-odd test using the closed-form intersections themselves.

        The spline body has to approximate this against a dense polyline.  Here the
        crossings are already exact, so counting them is both cheaper and truthful.
        """
        p = np.asarray(tuple(point), dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("point must be a finite 2-vector")
        crossings = self.all_intersections(p, self._PROBE, t_min=1e-12)
        return bool(len(crossings) % 2)

    def geometry_report(self, sample_count: int = 256) -> GeometryReport:
        """Deterministic audit.  ``simple`` is the no-crossing-edges requirement."""
        count = max(int(sample_count), 8 * self.control_count, 32)
        _, p = self.sample(count)
        finite = bool(np.isfinite(p).all())
        area = _signed_polygon_area(p) if finite else float("nan")
        max_kappa = float(np.max(np.abs([e["curvature"] for e in self._edges])))

        simple = finite
        if simple:
            starts = p
            edges = np.roll(p, -1, axis=0) - p
            ei, ej = edges[:, None, :], edges[None, :, :]
            rel = starts[None, :, :] - starts[:, None, :]
            den = ei[..., 0] * ej[..., 1] - ei[..., 1] * ej[..., 0]
            num_t = rel[..., 0] * ej[..., 1] - rel[..., 1] * ej[..., 0]
            num_v = rel[..., 0] * ei[..., 1] - rel[..., 1] * ei[..., 0]
            safe = np.where(abs(den) > 1e-12, den, 1.0)
            t, v = num_t / safe, num_v / safe
            idx = np.arange(count)
            gap = np.minimum(abs(idx[:, None] - idx[None, :]),
                             count - abs(idx[:, None] - idx[None, :]))
            eligible = (idx[:, None] < idx[None, :]) & (gap > 1)
            simple = not bool(np.any(eligible & (abs(den) > 1e-12)
                                     & (t >= -1e-10) & (t <= 1.0 + 1e-10)
                                     & (v >= -1e-10) & (v <= 1.0 + 1e-10)))

        if finite:
            distances = cdist(p, p)
            idx = np.arange(count)
            circular = np.minimum(abs(idx[:, None] - idx[None, :]),
                                  count - abs(idx[:, None] - idx[None, :]))
            distances[circular <= 2] = np.inf
            min_sep = float(np.min(distances))
        else:
            min_sep = 0.0

        reasons = []
        if not finite:
            reasons.append("boundary is not finite")
        if not simple:
            reasons.append("faces cross one another")
        # Winding is canonicalized on the corner cage, but a sag deep enough to sweep
        # past its neighbours can turn the enclosed region inside out without any two
        # faces actually crossing.  The area of the real boundary catches that, and it
        # matters: an inverted body reports its outward normals pointing inward.
        if finite and area <= 0.0:
            reasons.append("faces invert the body")
        if finite and min_sep < 1e-6:
            reasons.append("faces are separated by less than 1e-6")
        return GeometryReport(
            valid=bool(finite and simple and not reasons),
            simple=bool(simple), finite=finite,
            signed_area=float(area) if finite else float("nan"),
            perimeter=self._perimeter,
            min_speed=self._perimeter,
            max_abs_curvature=max_kappa,
            min_nonlocal_separation=min_sep,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------ intersection
    def all_intersections(self, origin: Iterable[float], direction: Iterable[float],
                          *, t_min: float = 1e-7, scan_count: int = 0) -> list[BoundaryHit]:
        """Every forward crossing, nearest first.  Closed form -- no root search.

        ``scan_count`` is accepted and ignored; it exists so this body can stand in for
        the spline body, whose polyline bracketing needed a resolution.
        """
        o = np.asarray(tuple(origin), dtype=float)
        d = np.asarray(tuple(direction), dtype=float)
        if o.shape != (2,) or d.shape != (2,):
            raise ValueError("origin and direction must be 2-vectors")
        if not np.isfinite(o).all() or not np.isfinite(d).all():
            raise ValueError("origin and direction must be finite")
        norm = float(np.linalg.norm(d))
        if norm <= _EPS:
            raise ValueError("ray direction must be non-zero")
        d = d / norm

        hits: list[tuple[float, int, float]] = []
        for index, edge in enumerate(self._edges):
            if edge["kind"] == "line":
                seg = edge["end"] - edge["start"]
                den = d[0] * seg[1] - d[1] * seg[0]
                if abs(den) <= 1e-14:
                    continue
                rel = edge["start"] - o
                t = (rel[0] * seg[1] - rel[1] * seg[0]) / den
                f = (rel[0] * d[1] - rel[1] * d[0]) / den
                if t > t_min and -1e-12 <= f <= 1.0 + 1e-12:
                    hits.append((t, index, min(max(f, 0.0), 1.0)))
                continue

            rel = o - edge["center"]
            b = float(np.dot(rel, d))
            c = float(np.dot(rel, rel)) - edge["radius"] ** 2
            disc = b * b - c
            if disc <= 0.0:                 # miss, or a tangency we deliberately drop
                continue
            root = sqrt(disc)
            for t in (-b - root, -b + root):
                if t <= t_min:
                    continue
                point = o + t * d
                theta = atan2(point[1] - edge["center"][1], point[0] - edge["center"][0])
                delta = (theta - edge["theta0"]) % (2.0 * pi)
                if edge["sweep"] < 0:
                    delta -= 2.0 * pi
                f = delta / edge["sweep"]
                if -1e-12 <= f <= 1.0 + 1e-12:
                    hits.append((t, index, min(max(f, 0.0), 1.0)))

        hits.sort(key=lambda item: item[0])
        out: list[BoundaryHit] = []
        for t, index, f in hits:
            if out and abs(t - out[-1].t) < 1e-7:
                continue
            point = self._point_on(index, f)
            normal = self._normal_on(index, f)
            u = float(self._cumulative[index]
                      + f * (self._cumulative[index + 1] - self._cumulative[index]))
            out.append(BoundaryHit(
                t=float(t),
                point=(float(point[0]), float(point[1])),
                outward_normal=(float(normal[0]), float(normal[1])),
                u=u % 1.0,
                arc_s=u % 1.0,
                curvature=float(self._edges[index]["curvature"]),
            ))
        return out

    def nearest_intersection(self, origin: Iterable[float], direction: Iterable[float],
                             *, t_min: float = 1e-7, scan_count: int = 0
                             ) -> BoundaryHit | None:
        hits = self.all_intersections(origin, direction, t_min=t_min)
        return hits[0] if hits else None
