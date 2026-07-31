from __future__ import annotations

import torch


def intersection_penalty(
    lens, *, air_min_mm: float, thick_min_mm: float, device: str
) -> torch.Tensor:
    """Soft penalty for adjacent surfaces that are too close or intersecting.

    Walks every adjacent pair of surfaces in ``lens.surfaces``, computes the
    center-axis distance and the minimum edge distance along a 16-point radial
    sample, and adds ``(min_dist - actual)`` whenever the actual distance drops
    below the applicable threshold.  Air gaps use ``air_min_mm`` (0.1 mm), glass
    thicknesses use ``thick_min_mm`` (1.0 mm).  Mirrors ``_intersection_penalty``
    in the unrolled EF50 code.
    """
    loss = torch.tensor(0.0, device=device)
    for i in range(len(lens.surfaces) - 1):
        curr = lens.surfaces[i]
        nxt = lens.surfaces[i + 1]

        r0 = torch.tensor(0.0, device=device)
        z_curr = curr.surface_with_offset(r0, 0.0, valid_check=False)
        z_nxt = nxt.surface_with_offset(r0, 0.0, valid_check=False)
        dist_center = z_nxt - z_curr

        r_edge = torch.linspace(0.5, 1.0, 16, device=device) * curr.r
        z_curr_edge = curr.surface_with_offset(r_edge, 0.0, valid_check=False)
        z_nxt_edge = nxt.surface_with_offset(r_edge, 0.0, valid_check=False)
        dist_edge = torch.min(z_nxt_edge - z_curr_edge)

        is_air = curr.mat2.name == "air"
        min_dist = air_min_mm if is_air else thick_min_mm

        if float(dist_center.detach()) < min_dist:
            loss = loss + (min_dist - dist_center)
        if float(dist_edge.detach()) < min_dist:
            loss = loss + (min_dist - dist_edge)

    return loss


def invalid_fraction_penalty(
    invalid_frac: torch.Tensor,
    *,
    tol: float,
    weight: float,
) -> torch.Tensor:
    """Soft penalty ``weight * max(mean_invalid - tol, 0)``.

    ``invalid_frac`` is typically the per-field invalid fraction from
    ``rms_spot_batch``; the penalty triggers when the mean exceeds ``tol`` and
    scales linearly above the threshold.
    """
    mean_inv = invalid_frac.mean() if invalid_frac.ndim > 0 else invalid_frac
    return weight * torch.clamp(mean_inv - tol, min=0.0)
