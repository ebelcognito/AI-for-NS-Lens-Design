"""Genotype for `ArcPolygonBody`: each corner drawn from its own bounded region.

The legacy prism genotype derived four corners from a thickness and two face tilts, so
every design was the same quadrilateral seen at different aspect.  Here each corner has
its own sampling box and moves independently, which is what produces genuinely distinct
starting geometries while keeping every face spherical.

The regions are sized so that neighbouring corners cannot swap places, but nothing stops
a bad draw from producing crossing faces -- that is caught by the body's geometry audit
and reported as an invalid candidate, not silently repaired.

As in the spline codec, the refractive index is problem configuration and not a gene, so
a search cannot rescue a poor shape by drifting to a higher index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arc_polygon import ArcPolygonBody
from .boundary import RI_MAX, RI_MIN

# A quadrilateral spanning the space between the object plane and the eyebox, sized so
# it clears both with room to move.
NOMINAL_CORNERS = ((50.0, -24.0), (82.0, -24.0), (82.0, 50.0), (38.0, 60.0))
CORNER_REGION = 15.0       # mm, half-width of each corner's sampling box
SAG_LIMIT = 20.0           # mm, maximum bulge of any face


@dataclass(frozen=True)
class ArcPolygonCodec:
    """Decode bounded genes into a spherical-faced element with sampled corners.

    Genes are normalized to [-1, 1]: two per corner (its offset inside that corner's
    region) followed by one per face (its sag).  ``gene_size`` is therefore ``3 * K``.
    """

    nominal_corners: tuple[tuple[float, float], ...] = NOMINAL_CORNERS
    corner_region: float = CORNER_REGION
    sag_limit: float = SAG_LIMIT
    n: float = 1.60

    def __post_init__(self) -> None:
        corners = np.asarray(self.nominal_corners, dtype=float)
        if corners.ndim != 2 or corners.shape[1:] != (2,) or len(corners) < 3:
            raise ValueError("nominal_corners must be at least three 2-vectors")
        if not np.isfinite(corners).all():
            raise ValueError("nominal_corners must be finite")
        if not np.isfinite((self.corner_region, self.sag_limit, self.n)).all():
            raise ValueError("codec scalars must be finite")
        if self.corner_region <= 0 or self.sag_limit < 0:
            raise ValueError("corner_region must be positive and sag_limit non-negative")
        if not RI_MIN <= self.n <= RI_MAX:
            raise ValueError(f"n must be in [{RI_MIN}, {RI_MAX}]")
        # Regions that overlap would let two corners land on the same point, or swap,
        # which turns the polygon inside out rather than merely crossing a face.
        following = np.roll(corners, -1, axis=0)
        gaps = np.linalg.norm(following - corners, axis=1)
        if float(gaps.min()) <= 2.0 * self.corner_region:
            raise ValueError(
                "corner regions overlap: halve corner_region or spread the corners")

    @property
    def corner_count(self) -> int:
        return len(self.nominal_corners)

    @property
    def gene_size(self) -> int:
        return 3 * self.corner_count

    def decode(self, genes) -> ArcPolygonBody:
        g = np.asarray(genes, dtype=float)
        if g.shape != (self.gene_size,):
            raise ValueError(f"expected {self.gene_size} genes, got shape {g.shape}")
        if not np.isfinite(g).all() or np.any(np.abs(g) > 1.0 + 1e-12):
            raise ValueError("genes must be finite and contained in [-1, 1]")

        count = self.corner_count
        offsets = g[: 2 * count].reshape(count, 2) * self.corner_region
        corners = np.asarray(self.nominal_corners, dtype=float) + offsets
        sags = g[2 * count:] * self.sag_limit
        return ArcPolygonBody(corners, sags, n=self.n)

    def corner_regions(self) -> list[tuple[tuple[float, float], float]]:
        """Each corner's nominal centre and region half-width, for plotting."""
        return [(tuple(map(float, corner)), float(self.corner_region))
                for corner in self.nominal_corners]
