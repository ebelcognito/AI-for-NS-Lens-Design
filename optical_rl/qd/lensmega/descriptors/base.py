from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np
import torch

from ..problem.lens_template import LensTemplate


class Descriptor(Protocol):
    """Behavior-characterization producer used by CMA-MEGA.

    The descriptor is the ONLY place that knows how to turn a traced lens into
    BC coordinates.  The runner, problem, and archive never inspect its kind:
    they just ask for ``dim``, ``coord_names``, and ``compute(...)``.
    """

    @property
    def dim(self) -> int:
        """Number of behavior coordinates produced, K."""
        ...

    @property
    def coord_names(self) -> Tuple[str, ...]:
        """Human-readable names for each coordinate; used for plot / CSV headers."""
        ...

    @property
    def fixed_bounds(self) -> Optional[np.ndarray]:
        """Optional (K, 2) archive bounds.  None means 'use calibration'."""
        ...

    def prepare(self, template: LensTemplate, *, device: str) -> None:
        """One-shot init given the lens template.  Called once per worker."""
        ...

    def compute(
        self,
        lens,
        *,
        field_y_deg: float,
        object_depth_mm: float,
        wavelength_um: float,
        device: str,
        **kwargs,
    ) -> torch.Tensor:
        """Return a shape-(K,) differentiable torch tensor of BC coordinates."""
        ...
