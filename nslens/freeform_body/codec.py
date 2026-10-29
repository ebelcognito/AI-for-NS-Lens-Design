"""A multiresolution, non-quadrilateral genotype for ``FreeformBody2D``.

The material index belongs to the problem specification, not to the genotype.  That
prevents a search from repairing every difficult shape by drifting to a higher-index
material.  More geometric detail is exposed by constructing a codec with additional
Fourier harmonics; the decoded optical object is always the same periodic spline type.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import FreeformBody2D, RI_MAX, RI_MIN


@dataclass(frozen=True)
class FreeformBodyCodec:
    """Decode bounded genes into a smooth closed freeform optical element.

    Genes are all normalized to [-1, 1]: five pose/envelope values followed by four
    values per harmonic (radial cosine/sine and tangential cosine/sine).  Harmonics
    begin at order two because translation and global scale/rotation already have
    explicit parameters.
    """

    center: tuple[float, float] = (60.0, 0.0)
    base_radii: tuple[float, float] = (20.0, 30.0)
    n: float = 1.60
    control_count: int = 16
    harmonic_count: int = 3
    translation_limit: tuple[float, float] = (20.0, 20.0)
    log_scale_limit: float = 0.35
    rotation_limit_deg: float = 60.0
    radial_fraction: float = 0.45
    tangential_fraction: float = 0.25

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.base_radii, dtype=float)
        translation = np.asarray(self.translation_limit, dtype=float)
        scalars = np.asarray((self.n, self.log_scale_limit, self.rotation_limit_deg,
                              self.radial_fraction, self.tangential_fraction), dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite 2-vector")
        if radii.shape != (2,) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("base radii must be finite and positive")
        if (translation.shape != (2,) or not np.isfinite(translation).all()
                or np.any(translation < 0)):
            raise ValueError("translation limits must be finite and non-negative")
        if not np.isfinite(scalars).all():
            raise ValueError("codec scalar settings must be finite")
        if not RI_MIN <= self.n <= RI_MAX:
            raise ValueError(f"n must be in [{RI_MIN}, {RI_MAX}]")
        if not isinstance(self.control_count, (int, np.integer)):
            raise ValueError("control_count must be an integer")
        if not isinstance(self.harmonic_count, (int, np.integer)):
            raise ValueError("harmonic_count must be an integer")
        if self.control_count < max(8, 2 * self.harmonic_count + 4):
            raise ValueError("control_count is too small for the requested harmonics")
        if self.harmonic_count < 1:
            raise ValueError("harmonic_count must be positive")
        if min(self.log_scale_limit, self.rotation_limit_deg,
               self.radial_fraction, self.tangential_fraction) < 0:
            raise ValueError("codec limits and deformation fractions must be non-negative")

    @property
    def gene_size(self) -> int:
        return 5 + 4 * self.harmonic_count

    def decode(self, genes) -> FreeformBody2D:
        g = np.asarray(genes, dtype=float)
        if g.shape != (self.gene_size,):
            raise ValueError(f"expected {self.gene_size} genes, got shape {g.shape}")
        if not np.isfinite(g).all() or np.any(np.abs(g) > 1.0 + 1e-12):
            raise ValueError("genes must be finite and contained in [-1, 1]")

        dx = g[0] * self.translation_limit[0]
        dy = g[1] * self.translation_limit[1]
        sx = np.exp(g[2] * self.log_scale_limit)
        sy = np.exp(g[3] * self.log_scale_limit)
        rotation = np.radians(g[4] * self.rotation_limit_deg)

        theta = np.linspace(0.0, 2.0 * np.pi, self.control_count, endpoint=False)
        rx, ry = self.base_radii[0] * sx, self.base_radii[1] * sy
        base = np.stack((rx * np.cos(theta), ry * np.sin(theta)), axis=1)
        deriv = np.stack((-rx * np.sin(theta), ry * np.cos(theta)), axis=1)
        tangent = deriv / np.linalg.norm(deriv, axis=1, keepdims=True)
        normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1)

        radial = np.zeros(self.control_count)
        tangential = np.zeros(self.control_count)
        coeff = g[5:].reshape(self.harmonic_count, 4)
        # Normalize by harmonic count so adding resolution does not automatically add
        # unbounded geometric departure.
        for row, mode in zip(coeff, range(2, self.harmonic_count + 2)):
            radial += row[0] * np.cos(mode * theta) + row[1] * np.sin(mode * theta)
            tangential += row[2] * np.cos(mode * theta) + row[3] * np.sin(mode * theta)
        radial /= self.harmonic_count
        tangential /= self.harmonic_count
        length_scale = min(rx, ry)
        points = (base
                  + self.radial_fraction * length_scale * radial[:, None] * normal
                  + self.tangential_fraction * length_scale * tangential[:, None] * tangent)

        c, s = np.cos(rotation), np.sin(rotation)
        rotate = np.array(((c, -s), (s, c)))
        points = points @ rotate.T
        points += np.asarray(self.center) + np.array((dx, dy))
        return FreeformBody2D(points, n=self.n)
