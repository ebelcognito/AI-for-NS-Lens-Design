from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


Descriptor2D = Tuple[float, float]


@dataclass(frozen=True)
class IrradianceDescriptor:
    centroid: float
    spread: float


def descriptor_from_bounding_box(points_xy: Iterable[Tuple[float, float]]) -> Descriptor2D:
    pts = np.asarray(list(points_xy), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise ValueError("Expected a non-empty sequence of (x,y) points")
    width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    return width, height


def descriptor_from_irradiance_map(
    detector_data: dict, *, eps: float = 1e-12
) -> Optional[IrradianceDescriptor]:
    """Compute (centroid, spread) from detector irradiance bins if available.

    Returns None if the necessary keys are missing.
    """
    if not isinstance(detector_data, dict):
        return None
    irr = detector_data.get("irradianceMap")
    pos = detector_data.get("binPositions")
    if not (isinstance(irr, list) and isinstance(pos, list) and len(irr) == len(pos) and len(irr) > 0):
        return None

    w = np.asarray(irr, dtype=np.float64)
    x = np.asarray(pos, dtype=np.float64)
    total = float(np.sum(w))
    if total <= eps:
        return IrradianceDescriptor(centroid=float(x[len(x) // 2]), spread=float(x[-1] - x[0]))

    mu = float(np.sum(w * x) / total)
    var = float(np.sum(w * (x - mu) ** 2) / total)
    return IrradianceDescriptor(centroid=mu, spread=float(np.sqrt(max(var, 0.0))))

