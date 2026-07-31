"""Random-projection descriptor for ablation studies.

Projects the lens theta vector onto fixed random directions.  These
descriptors carry NO optical meaning — they exist solely as a control
condition to test whether Seidel-based behavior characterizations
provide genuine structure beyond arbitrary diversity maintenance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..problem.lens_template import LensTemplate


class RandomProjectionDescriptor:
    """2-D descriptor from random linear projection of the theta vector.

    The projection matrix is seeded deterministically so runs are
    reproducible.  The descriptor is differentiable (linear in theta)
    so CMA-MEGA gradient machinery works unchanged.
    """

    def __init__(
        self,
        dim_out: int = 2,
        seed: int = 12345,
        theta_dim: int = 19,
        theta_bounds: Optional[list] = None,
    ) -> None:
        self._dim = dim_out
        self._seed = seed
        self._theta_dim = theta_dim
        self._theta_bounds = theta_bounds
        self._proj: Optional[torch.Tensor] = None  # (theta_dim, dim_out)
        self._bounds: Optional[np.ndarray] = None

    # ---- Descriptor protocol ----

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def coord_names(self) -> Tuple[str, ...]:
        return tuple(f"rand_proj_{i}" for i in range(self._dim))

    @property
    def fixed_bounds(self) -> Optional[np.ndarray]:
        return self._bounds

    def prepare(self, template: LensTemplate, *, device: str) -> None:
        """Build a fixed random projection matrix."""
        theta_dim = self._theta_dim

        rng = np.random.RandomState(self._seed)
        # Gaussian random projection, normalized columns
        W = rng.randn(theta_dim, self._dim).astype(np.float32)
        for j in range(self._dim):
            W[:, j] /= np.linalg.norm(W[:, j])

        self._proj = torch.tensor(W, dtype=torch.float32, device=device)

        # Estimate bounds from theta bounds if provided.
        if self._theta_bounds is not None:
            lo = np.array([b[0] for b in self._theta_bounds], dtype=np.float32)
            hi = np.array([b[1] for b in self._theta_bounds], dtype=np.float32)

            bounds = np.zeros((self._dim, 2), dtype=np.float64)
            for j in range(self._dim):
                w = W[:, j]
                proj_lo = np.sum(np.where(w > 0, lo * w, hi * w))
                proj_hi = np.sum(np.where(w > 0, hi * w, lo * w))
                margin = 0.05 * (proj_hi - proj_lo)
                bounds[j] = [proj_lo - margin, proj_hi + margin]
            self._bounds = bounds

    def compute(
        self,
        lens,
        *,
        field_y_deg: float,
        object_depth_mm: float,
        wavelength_um: float,
        device: str,
        theta_tensor: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Project the current theta onto random directions.

        The theta_tensor is passed through kwargs by the problem evaluator.
        If not available, fall back to extracting curvatures from the lens
        (non-differentiable path — should not happen in normal operation).
        """
        if theta_tensor is None:
            raise RuntimeError(
                "RandomProjectionDescriptor.compute() requires theta_tensor "
                "kwarg — check that the problem passes it."
            )

        proj = self._proj
        if proj.device != theta_tensor.device:
            proj = proj.to(theta_tensor.device)
            self._proj = proj

        return theta_tensor @ proj  # shape (dim_out,)
