"""Bounded 3/4/5-corner circular-face bodies for folded-path discovery."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from nslens.freeform_body.arc_polygon import ArcPolygonBody
from nslens.freeform_body.boundary import RI_MIN, RI_MAX


@dataclass(frozen=True)
class PolygonCodec:
    corners: int
    n: float = 1.60
    center_x: tuple = (40., 70.)
    center_y: tuple = (-10., 15.)
    half_width: tuple = (12., 30.)
    half_height: tuple = (12., 30.)
    radial_fraction: tuple = (.65, 1.)
    angular_jitter_fraction: float = .20
    max_sag_ratio: float = .268
    min_radius_mm: float = 8.
    min_edge_mm: float = 4.

    def __post_init__(self):
        if type(self.corners) is not int or self.corners not in (3, 4, 5):
            raise ValueError("corners must be 3, 4, or 5")
        for name in ("center_x", "center_y", "half_width", "half_height", "radial_fraction"):
            pair = np.asarray(getattr(self, name), dtype=float)
            if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
                raise ValueError(f"{name} must be a finite increasing interval")
        scalars = (self.n, self.angular_jitter_fraction, self.max_sag_ratio,
                   self.min_radius_mm, self.min_edge_mm)
        if not np.isfinite(scalars).all():
            raise ValueError("codec scalars must be finite")
        if not RI_MIN <= self.n <= RI_MAX:
            raise ValueError("material index is outside the tracer's supported range")
        if min(self.half_width[0], self.half_height[0], self.min_radius_mm, self.min_edge_mm) <= 0:
            raise ValueError("length scales must be positive")
        if not 0 < self.radial_fraction[0] < self.radial_fraction[1] <= 1:
            raise ValueError("radial fractions must lie in (0, 1]")
        if not 0 <= self.angular_jitter_fraction < .5:
            raise ValueError("angular sectors must not overlap")
        if not 0 < self.max_sag_ratio < 1:
            raise ValueError("sag/half-chord must lie in (0, 1), excluding major arcs")

    @property
    def gene_size(self):
        return 5 + 3 * self.corners

    def decode(self, genes):
        g = np.asarray(genes, dtype=float)
        if g.shape != (self.gene_size,) or not np.isfinite(g).all() or np.any(abs(g) > 1):
            raise ValueError("invalid normalized polygon genes")
        def interval(value, bounds):
            return bounds[0] + (value + 1) * .5 * (bounds[1] - bounds[0])
        cx, cy, hx, hy = [interval(g[i], bounds) for i, bounds in enumerate(
            (self.center_x, self.center_y, self.half_width, self.half_height))]
        k = self.corners
        # Cyclic angular sectors guarantee ordered vertices, without corner boxes or
        # fixed top/bottom/left/right roles. Radial variation permits concave cages.
        theta = (2 * np.pi * np.arange(k) / k + np.pi * g[4]
                 + g[5:5+k] * self.angular_jitter_fraction * 2 * np.pi / k)
        radial = interval(g[5+k:5+2*k], self.radial_fraction)
        corners = np.column_stack((cx + hx * radial * np.cos(theta),
                                   cy + hy * radial * np.sin(theta)))
        half_chords = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) / 2
        if np.any(half_chords * 2 < self.min_edge_mm):
            raise ValueError("edge shorter than min_edge_mm")
        # R = a*(1+q*q)/(2*|q|), a=half-chord, q=sag/a. Solve the
        # minimum-radius constraint on the minor-arc branch without cancellation.
        radius_limit = np.ones(k)
        small = half_chords < self.min_radius_mm
        a = half_chords[small]
        radius_limit[small] = a / (self.min_radius_mm + np.sqrt(self.min_radius_mm**2 - a*a))
        limits = np.minimum(self.max_sag_ratio, radius_limit)
        sags = g[5+2*k:] * limits * half_chords
        return ArcPolygonBody(corners, sags, n=self.n)


def route_key(body, events):
    """Typed edge/revisit pattern, invariant to arbitrary cyclic vertex numbering."""
    names, sequence = {}, []
    for event in events:
        index, _ = body._locate(event.u)
        if index not in names:
            names[index] = len(names)
        sequence.append((event.kind, names[index]))
    return tuple(sequence)


def exact_extent(body):
    """Include circular-arc extrema, not just corners or a sampled boundary."""
    points = list(body.corners)
    for edge in body._edges:
        if edge["kind"] != "arc":
            continue
        for theta in (0., math.pi/2, math.pi, 3*math.pi/2):
            delta = (theta - edge["theta0"]) % (2*math.pi)
            if edge["sweep"] < 0:
                delta -= 2*math.pi
            fraction = delta / edge["sweep"]
            if -1e-12 <= fraction <= 1 + 1e-12:
                points.append(edge["center"] + edge["radius"] * np.array((math.cos(theta), math.sin(theta))))
    p = np.asarray(points)
    return p.min(axis=0), p.max(axis=0)
