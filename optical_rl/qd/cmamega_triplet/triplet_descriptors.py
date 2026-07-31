from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "global_power_proxy",
    "surface_power_skew",
    "total_power_mag",
    "offaxis_aberr_proxy",
    "invalid_fraction",
)


def surface_power_skew(phi: Sequence) -> object:
    eps = 1e-8
    p_front = phi[0] + phi[1] + phi[2]
    p_rear = phi[3] + phi[4] + phi[5]
    return (p_front - p_rear) / (abs(p_front) + abs(p_rear) + eps)


def build_descriptor_dict(
    *,
    mean_offaxis_y,
    alpha_deg: float,
    phi: Sequence,
    aberr_proxy,
    invalid_fraction,
) -> Dict[str, object]:
    alpha = max(float(alpha_deg), 1e-6)
    d0 = mean_offaxis_y / math.tan(math.radians(alpha))
    p_total = phi[0] + phi[1] + phi[2] + phi[3] + phi[4] + phi[5]
    d1 = surface_power_skew(phi)
    d2 = abs(p_total)
    d3 = aberr_proxy
    d4 = invalid_fraction
    return {
        "global_power_proxy": d0,
        "surface_power_skew": d1,
        "total_power_mag": d2,
        "offaxis_aberr_proxy": d3,
        "invalid_fraction": d4,
    }
