from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple


DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "power_proxy",
    "power_split_front_rear",
    "total_power_mag",
    "offaxis_aberr_proxy",
)


def power_split_front_rear(phi: Sequence) -> object:
    eps = 1e-8
    p_front = phi[0] + phi[1]
    p_rear = phi[2] + phi[3]
    return (p_front - p_rear) / (abs(p_front) + abs(p_rear) + eps)


def build_descriptor_dict(
    *,
    mean_offaxis_y,
    alpha_deg: float,
    phi: Sequence,
    aberr_proxy,
) -> Dict[str, object]:
    alpha = max(float(alpha_deg), 1e-6)
    d0 = mean_offaxis_y / math.tan(math.radians(alpha))
    p_total = phi[0] + phi[1] + phi[2] + phi[3]
    d1 = power_split_front_rear(phi)
    d2 = abs(p_total)
    d3 = aberr_proxy
    return {
        "power_proxy": d0,
        "power_split_front_rear": d1,
        "total_power_mag": d2,
        "offaxis_aberr_proxy": d3,
    }
