"""Nearest-hit scene adapter for the existing Fresnel/TIR tracer.

Bodies must be disjoint, with air gaps. No cemented or nested media. Mirrors are
explicit ideal, lossless coatings, not a model of real spectral coating efficiency.
"""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from nslens.discovery.polygon import exact_extent


class OpticalScene:
    def __init__(self, bodies, mirrors=None, *, clearance=.02):
        self.bodies = tuple(bodies)
        if not self.bodies:
            raise ValueError('at least one body required')
        self.n = self.bodies[0].n
        self.mirrors = tuple(tuple(m) for m in mirrors) if mirrors is not None else tuple(
            (False,) * b.control_count for b in self.bodies)
        if len(self.mirrors) != len(self.bodies) or any(
                len(m) != b.control_count or any(type(v) is not bool for v in m)
                for b, m in zip(self.bodies, self.mirrors)):
            raise ValueError('one boolean mirror flag per canonical CCW face required')
        self.clearance = clearance

    def contains(self, point):
        return any(b.contains(point) for b in self.bodies)

    def nearest_intersection(self, origin, direction, **kwargs):
        nearest = None
        for i, body in enumerate(self.bodies):
            hit = body.nearest_intersection(origin, direction, **kwargs)
            if hit is not None and (nearest is None or hit.t < nearest.t):
                face, _ = body._locate(hit.u)
                nearest = SimpleNamespace(**asdict(hit), body_id=i, face_id=int(face),
                    material_index=body.n, ideal_mirror=self.mirrors[i][face])
        return nearest

    def validate(self, samples=256):
        extents = []
        for b in self.bodies:
            report = b.geometry_report(sample_count=samples)
            if not report.valid:
                raise ValueError('invalid body: ' + '; '.join(report.reasons))
            # Safe overbound for polynomial departures: envelope <= 1, each of
            # the three basis functions has absolute value <= 1 on [0,1].
            lo, hi = exact_extent(b)
            pad = float(np.max(np.sum(abs(b.coefficients), axis=1))) if hasattr(b, 'coefficients') else 0.
            extents.append((lo-pad, hi+pad))
        # Conservative separation avoids unverified thin intersections and nested
        # media. It excludes interleaved AABBs even when actual bodies are disjoint.
        for i, (lo, hi) in enumerate(extents):
            for lo2, hi2 in extents[:i]:
                if not np.any((hi + self.clearance <= lo2) | (hi2 + self.clearance <= lo)):
                    raise ValueError('body bounding boxes must have an air gap')
        return np.min([e[0] for e in extents], axis=0), np.max([e[1] for e in extents], axis=0)
