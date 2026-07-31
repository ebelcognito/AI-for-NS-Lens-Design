"""Parameter-space (geometric) descriptor for the Antonov-style ablation.

Uses raw lens *parameters* (specific entries of the theta vector, e.g. the
first and last element curvatures) as the behavior descriptor axes, instead
of an optically meaningful quantity like a Seidel coefficient.  This is the
faithful "parameter-space QD" / curvature-descriptor control condition
(cf. Antonov et al. 2025): the archive is organised by lens geometry rather
than optical behavior, so a HyperLens-vs-this comparison isolates whether the
*optical* meaning of the descriptor (not merely having any 2-D archive) is
what drives the diversity advantage.

Like the random-projection descriptor, this is a fixed linear selection of the
theta vector and is therefore differentiable — the CMA-MEGA gradient machinery
works unchanged.  We deliberately auto-calibrate the archive bounds
(``fixed_bounds`` returns None) so the geometric archive gets its best shot at
covering the realised parameter distribution (no strawman).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..problem.lens_template import LensTemplate


class LensParamDescriptor:
    """K-D descriptor selecting K raw entries of the theta vector.

    Parameters
    ----------
    indices:
        Positions in the (full-DOF) theta vector to use as BC axes.  For the
        sixlet, theta is ordered [12 curvatures, 12 gaps, flange, 12 materials],
        so ``indices=[0, 11]`` picks the first (E1 front) and last (E6 back)
        element curvatures.
    names:
        Optional human-readable coord names for plots/CSV headers.
    theta_dim:
        Length of the theta vector (for validation only).
    """

    def __init__(
        self,
        indices: List[int],
        names: Optional[List[str]] = None,
        theta_dim: Optional[int] = None,
    ) -> None:
        self._indices = [int(i) for i in indices]
        self._dim = len(self._indices)
        if names is not None and len(names) != self._dim:
            raise ValueError("lens_param: len(names) must match len(indices)")
        self._names = tuple(names) if names is not None else tuple(
            f"theta_{i}" for i in self._indices
        )
        self._theta_dim = theta_dim
        self._idx_t: Optional[torch.Tensor] = None

    # ---- Descriptor protocol ----

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def coord_names(self) -> Tuple[str, ...]:
        return self._names

    @property
    def fixed_bounds(self) -> Optional[np.ndarray]:
        # Auto-calibrate from seeding — the geometric archive adapts its bounds
        # to the realised parameter distribution.
        return None

    def prepare(self, template: LensTemplate, *, device: str) -> None:
        if self._theta_dim is not None:
            for i in self._indices:
                if not (0 <= i < self._theta_dim):
                    raise ValueError(
                        f"lens_param index {i} out of range for theta_dim "
                        f"{self._theta_dim}"
                    )
        self._idx_t = torch.tensor(self._indices, dtype=torch.long, device=device)

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
        """Select the chosen raw theta entries (differentiable gather)."""
        if theta_tensor is None:
            raise RuntimeError(
                "LensParamDescriptor.compute() requires theta_tensor kwarg — "
                "check that the problem passes it."
            )
        idx = self._idx_t
        if idx is None or idx.device != theta_tensor.device:
            idx = torch.tensor(self._indices, dtype=torch.long,
                               device=theta_tensor.device)
            self._idx_t = idx
        return theta_tensor.index_select(0, idx)  # shape (K,)
