"""A single, smooth, closed freeform optical boundary.

The legacy discovery domains are four-sided prisms whose freeform coefficients only
deform pre-named faces.  This module deliberately has no face concept.  The optical
surface is a periodic cubic spline and a hit is located by normalized boundary
arclength.

The implementation is two-dimensional because the present discovery engine is 2-D.
It is a geometry kernel, not a waveguide model: there is no centerline, port, width,
or preferred propagation direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.distance import cdist

RI_MIN = 1.50
RI_MAX = 1.70
_EPS = 1e-12


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_polygon_area(points: np.ndarray) -> float:
    q = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * q[:, 1] - points[:, 1] * q[:, 0]))


@dataclass(frozen=True)
class BoundaryHit:
    """One forward ray intersection with the continuous boundary."""

    t: float
    point: tuple[float, float]
    outward_normal: tuple[float, float]
    u: float
    arc_s: float
    curvature: float


@dataclass(frozen=True)
class GeometryReport:
    """Deterministic geometry audit; no optical merit is included."""

    valid: bool
    simple: bool
    finite: bool
    signed_area: float
    perimeter: float
    min_speed: float
    max_abs_curvature: float
    min_nonlocal_separation: float
    reasons: tuple[str, ...]


class FreeformBody2D:
    """A homogeneous optical element bounded by one periodic C2 spline.

    ``control_points`` are interpolation anchors rather than four physical corners.
    The constructor canonicalizes their winding to counter-clockwise, so the returned
    surface normal is always outward.  Material index is a hard domain invariant and
    may not leave [1.50, 1.70].
    """

    def __init__(self, control_points: Iterable[Iterable[float]], n: float):
        pts = np.asarray(tuple(control_points), dtype=float)
        if pts.ndim != 2 or pts.shape[1:] != (2,):
            raise ValueError("control_points must have shape (K, 2)")
        if len(pts) < 6:
            raise ValueError("a freeform boundary requires at least 6 control points")
        if not np.isfinite(pts).all():
            raise ValueError("control_points must be finite")
        if not RI_MIN <= float(n) <= RI_MAX:
            raise ValueError(f"refractive index must be in [{RI_MIN:.2f}, {RI_MAX:.2f}]")
        if abs(_signed_polygon_area(pts)) <= _EPS:
            raise ValueError("control-point cage has zero signed area")
        if _signed_polygon_area(pts) < 0:
            pts = pts[::-1].copy()

        nodes = np.linspace(0.0, 1.0, len(pts) + 1)
        closed = np.vstack((pts, pts[0]))
        self._points = pts.copy()
        self._points.setflags(write=False)
        # One vector-valued spline rather than two scalar ones.  CubicSpline solves
        # each column independently, so this is the same curve, evaluated in half the
        # scipy calls -- and tracing calls it millions of times per search.
        self._spline = CubicSpline(nodes, closed, axis=0, bc_type="periodic")
        self.n = float(n)
        self._scan_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Piecewise coefficients as plain floats.  Root refinement runs millions of
        # scalar evaluations per search, where allocating two-element arrays costs more
        # than the arithmetic does.
        coeff = np.ascontiguousarray(self._spline.c)
        self._segment_count = len(pts)
        self._segment_width = 1.0 / len(pts)
        self._coeff_rows = tuple(
            (float(coeff[0, i, 0]), float(coeff[1, i, 0]),
             float(coeff[2, i, 0]), float(coeff[3, i, 0]),
             float(coeff[0, i, 1]), float(coeff[1, i, 1]),
             float(coeff[2, i, 1]), float(coeff[3, i, 1]))
            for i in range(len(pts))
        )
        # Keep arclength and containment/intersection tables resolved even when a
        # caller elects to use substantially more anchors than the default codec.
        arc_intervals = max(2048, 32 * len(pts))
        self._arc_grid = np.linspace(0.0, 1.0, arc_intervals + 1)
        arc_points = self.evaluate(self._arc_grid)
        self._arc_cumulative = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(arc_points, axis=0), axis=1))
        ))
        self._arc_cumulative /= self._arc_cumulative[-1]

    @property
    def control_points(self) -> np.ndarray:
        return self._points.copy()

    @property
    def control_count(self) -> int:
        return len(self._points)

    @staticmethod
    def _wrap_u(u):
        arr = np.asarray(u, dtype=float)
        return np.mod(arr, 1.0)

    def evaluate(self, u, derivative: int = 0) -> np.ndarray:
        """Evaluate boundary position or its first/second parameter derivative."""
        if derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1, or 2")
        return self._spline(self._wrap_u(u), nu=derivative)

    def _local_point(self, u: float) -> tuple[float, float]:
        """Scalar position straight from the piecewise coefficients.

        Root refinement calls this millions of times per search, where SciPy's array
        entry cost dominates the arithmetic.  It agrees with ``evaluate`` to about
        1e-14 -- six orders below the tracer's tightest geometric tolerance -- and is
        used only to *locate* a root.  Accepted hits are still built with ``evaluate``.
        """
        uw = u % 1.0
        i = int(uw * self._segment_count)
        if i >= self._segment_count:
            i = self._segment_count - 1
        ax, bx, cx, dx_, ay, by, cy, dy_ = self._coeff_rows[i]
        t = uw - i * self._segment_width
        return (((ax * t + bx) * t + cx) * t + dx_,
                ((ay * t + by) * t + cy) * t + dy_)

    def _local_tangent(self, u: float) -> tuple[float, float]:
        """Scalar first derivative from the same coefficients."""
        uw = u % 1.0
        i = int(uw * self._segment_count)
        if i >= self._segment_count:
            i = self._segment_count - 1
        ax, bx, cx, _, ay, by, cy, _ = self._coeff_rows[i]
        t = uw - i * self._segment_width
        return ((3.0 * ax * t + 2.0 * bx) * t + cx,
                (3.0 * ay * t + 2.0 * by) * t + cy)

    def sample(self, count: int = 256) -> tuple[np.ndarray, np.ndarray]:
        if count < max(32, 2 * self.control_count):
            raise ValueError("sample count is too small for a geometry audit")
        u = np.linspace(0.0, 1.0, count, endpoint=False)
        return u, self.evaluate(u)

    def outward_normal(self, u) -> np.ndarray:
        tangent = self.evaluate(u, derivative=1)
        speed = np.linalg.norm(tangent, axis=-1, keepdims=True)
        if np.any(speed <= _EPS):
            raise ValueError("boundary has a singular tangent")
        return np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1) / speed

    def curvature(self, u) -> np.ndarray:
        d1 = self.evaluate(u, derivative=1)
        d2 = self.evaluate(u, derivative=2)
        den = np.linalg.norm(d1, axis=-1) ** 3
        num = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        return num / np.maximum(den, _EPS)

    def arclength_coordinates(self, u, table_size: int = 2048) -> np.ndarray:
        """Convert spline parameter to normalized counter-clockwise arclength."""
        query = self._wrap_u(u)
        if table_size == 2048:
            return np.interp(query, self._arc_grid, self._arc_cumulative)
        if table_size < 32:
            raise ValueError("table_size must be at least 32")
        grid = np.linspace(0.0, 1.0, table_size + 1)
        pts = self.evaluate(grid)
        cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        if cumulative[-1] <= _EPS:
            raise ValueError("boundary has zero perimeter")
        return np.interp(query, grid, cumulative / cumulative[-1])

    def contains(self, point: Iterable[float], sample_count: int = 512) -> bool:
        """Odd-even containment test on a dense, deterministic boundary sampling."""
        p = np.asarray(tuple(point), dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("point must be a finite 2-vector")
        effective_count = max(int(sample_count), 8 * self.control_count, 32)
        _, poly = self.sample(effective_count)
        a = poly
        b = np.roll(poly, -1, axis=0)
        crosses_y = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
        den = b[:, 1] - a[:, 1]
        den = np.where(abs(den) < _EPS, np.copysign(_EPS, den + _EPS), den)
        x_cross = a[:, 0] + (p[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / den
        return bool(np.count_nonzero(crosses_y & (x_cross > p[0])) % 2)

    def geometry_report(self, sample_count: int = 256) -> GeometryReport:
        effective_count = max(int(sample_count), 8 * self.control_count, 32)
        u, p = self.sample(effective_count)
        d1 = self.evaluate(u, derivative=1)
        finite = bool(np.isfinite(p).all() and np.isfinite(d1).all())
        area = _signed_polygon_area(p) if finite else float("nan")
        edge = np.roll(p, -1, axis=0) - p
        perimeter = float(np.linalg.norm(edge, axis=1).sum()) if finite else float("nan")
        speed = np.linalg.norm(d1, axis=1)
        min_speed = float(speed.min()) if finite else 0.0
        kappa = self.curvature(u) if finite and min_speed > _EPS else np.array([np.inf])
        max_kappa = float(np.max(np.abs(kappa)))

        simple = finite
        if simple:
            # Pairwise segment intersection in one vectorized pass. Parallel overlaps
            # are caught by the nonlocal-separation test below.
            starts = p
            edges = np.roll(p, -1, axis=0) - p
            ei, ej = edges[:, None, :], edges[None, :, :]
            rel = starts[None, :, :] - starts[:, None, :]
            den = ei[..., 0] * ej[..., 1] - ei[..., 1] * ej[..., 0]
            num_t = rel[..., 0] * ej[..., 1] - rel[..., 1] * ej[..., 0]
            num_v = rel[..., 0] * ei[..., 1] - rel[..., 1] * ei[..., 0]
            safe = np.where(abs(den) > 1e-12, den, 1.0)
            t, v = num_t / safe, num_v / safe
            idx = np.arange(effective_count)
            gap = np.minimum(abs(idx[:, None] - idx[None, :]),
                             effective_count - abs(idx[:, None] - idx[None, :]))
            eligible = (idx[:, None] < idx[None, :]) & (gap > 1)
            crossing = (eligible & (abs(den) > 1e-12)
                        & (t >= -1e-10) & (t <= 1.0 + 1e-10)
                        & (v >= -1e-10) & (v <= 1.0 + 1e-10))
            simple = not bool(np.any(crossing))

        distances = (cdist(p, p) if finite
                     else np.full((effective_count, effective_count), np.nan))
        if finite:
            idx = np.arange(effective_count)
            circular_gap = np.minimum(abs(idx[:, None] - idx[None, :]),
                                      effective_count - abs(idx[:, None] - idx[None, :]))
            distances[circular_gap <= 2] = np.inf
            min_sep = float(np.min(distances))
        else:
            min_sep = 0.0

        reasons = []
        if not finite:
            reasons.append("non-finite boundary")
        if not simple:
            reasons.append("self-intersecting boundary")
        if not np.isfinite(area) or area <= 1e-6:
            reasons.append("non-positive enclosed area")
        if min_speed <= 1e-7:
            reasons.append("singular boundary tangent")
        if min_sep <= 1e-5:
            reasons.append("pinched or unresolved boundary")
        return GeometryReport(
            valid=not reasons,
            simple=simple,
            finite=finite,
            signed_area=area,
            perimeter=perimeter,
            min_speed=min_speed,
            max_abs_curvature=max_kappa,
            min_nonlocal_separation=min_sep,
            reasons=tuple(reasons),
        )

    def all_intersections(self, origin: Iterable[float], direction: Iterable[float],
                          *, t_min: float = 1e-7, scan_count: int = 512) -> list[BoundaryHit]:
        """Return all ordinary forward intersections, nearest first.

        Tangent-only contacts are intentionally omitted: they are topology-unstable and
        should not become discovery interactions.  A polyline pass finds candidate
        brackets and Brent refinement places the hit on the actual spline.
        """
        o = np.asarray(tuple(origin), dtype=float)
        d = np.asarray(tuple(direction), dtype=float)
        norm = float(np.linalg.norm(d))
        if o.shape != (2,) or d.shape != (2,) or not np.isfinite(o).all() or not np.isfinite(d).all():
            raise ValueError("origin and direction must be finite 2-vectors")
        if norm <= _EPS:
            raise ValueError("ray direction must be non-zero")
        if scan_count < 16:
            raise ValueError("scan_count must be at least 16")
        d = d / norm

        effective_scan = max(int(scan_count), 16 * self.control_count)
        us, p0, edge = self._scan_grid(effective_scan)
        den = d[0] * edge[:, 1] - d[1] * edge[:, 0]
        rel = p0 - o
        safe = np.where(abs(den) > 1e-14, den, 1.0)
        t_poly = (rel[:, 0] * edge[:, 1] - rel[:, 1] * edge[:, 0]) / safe
        alpha = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / safe
        candidate_idx = np.flatnonzero(
            (abs(den) > 1e-14) & (t_poly > t_min)
            & (alpha >= -1e-6) & (alpha <= 1.0 + 1e-6)
        )
        candidates = [
            (us[i], us[i + 1],
             us[i] + np.clip(alpha[i], 0.0, 1.0) / effective_scan)
            for i in candidate_idx
        ]

        ox, oy, dx, dy = float(o[0]), float(o[1]), float(d[0]), float(d[1])
        point_at = self._local_point

        def line_residual(x: float) -> float:
            px, py = point_at(x)
            return (px - ox) * dy - (py - oy) * dx

        roots: list[tuple[float, float]] = []
        for lo, hi, guess in candidates:
            flo, fhi = line_residual(lo), line_residual(hi)
            try:
                if abs(flo) <= 1e-12:
                    root = lo
                elif abs(fhi) <= 1e-12:
                    root = hi
                elif flo * fhi < 0.0:
                    root = float(brentq(line_residual, lo, hi, xtol=1e-12, rtol=1e-12))
                else:
                    root = guess
                    for _ in range(8):
                        f = line_residual(root)
                        tx, ty = self._local_tangent(root)
                        fp = tx * dy - ty * dx
                        if abs(fp) <= 1e-14:
                            break
                        proposal = root - f / fp
                        if proposal < lo or proposal > hi:
                            break
                        root = proposal
                    if abs(line_residual(root)) > 1e-8:
                        continue
            except ValueError:
                continue
            point = self.evaluate(root)
            t = float(np.dot(point - o, d))
            if t <= t_min:
                continue
            root_wrapped = float(root % 1.0)
            if any(abs(t - old_t) < 1e-7 for old_t, _ in roots):
                continue
            roots.append((t, root_wrapped))

        roots.sort(key=lambda item: item[0])
        hits = []
        arc_coordinates = self.arclength_coordinates([u for _, u in roots]) if roots else []
        for (t, u), arc_s in zip(roots, arc_coordinates):
            point = self.evaluate(u)
            normal = self.outward_normal(u)
            hits.append(BoundaryHit(
                t=t,
                point=(float(point[0]), float(point[1])),
                outward_normal=(float(normal[0]), float(normal[1])),
                u=u,
                arc_s=float(arc_s),
                curvature=float(self.curvature(u)),
            ))
        return hits

    def _scan_grid(self, effective_scan: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached polyline bracketing grid.  The boundary is immutable, so the scan
        samples are identical for every ray and are computed once per resolution."""
        cached = self._scan_cache.get(effective_scan)
        if cached is None:
            us = np.linspace(0.0, 1.0, effective_scan + 1)
            pts = self.evaluate(us)
            cached = (us, pts[:-1], pts[1:] - pts[:-1])
            self._scan_cache[effective_scan] = cached
        return cached

    def nearest_intersection(self, origin: Iterable[float], direction: Iterable[float],
                             *, t_min: float = 1e-7, scan_count: int = 512) -> BoundaryHit | None:
        hits = self.all_intersections(origin, direction, t_min=t_min, scan_count=scan_count)
        return hits[0] if hits else None
