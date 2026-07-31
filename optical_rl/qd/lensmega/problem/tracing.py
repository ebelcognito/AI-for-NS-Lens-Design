from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch

from deeplens.optics.light.ray import Ray


def disk_samples(num_rays: int) -> np.ndarray:
    """Fibonacci spiral on the unit disk.

    Identical to the implementation copy-pasted into every old variant
    (`_disk_samples`), extracted here so it lives in exactly one place.
    """
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(max(num_rays, 1)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((max(n, 0), dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / float(n)
    return out


def project_box(theta: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(np.asarray(theta, dtype=np.float64).reshape(-1), lo, hi)


def theta_to_unit(
    theta: np.ndarray, bounds: Sequence[Tuple[float, float]]
) -> np.ndarray:
    theta = project_box(theta, bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (theta - center) / np.where(half > 0.0, half, 1.0)


def unit_to_theta(
    unit: np.ndarray, bounds: Sequence[Tuple[float, float]]
) -> np.ndarray:
    u = np.clip(np.asarray(unit, dtype=np.float64).reshape(-1), -1.0, 1.0)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return project_box(center + u * half, bounds)


def make_object_point(
    object_depth_mm: float, field_x_deg: float, field_y_deg: float
) -> Tuple[float, float, float]:
    depth = float(object_depth_mm)
    x = depth * math.tan(math.radians(field_x_deg))
    y = depth * math.tan(math.radians(field_y_deg))
    return x, y, depth


def trace_fields_batch(
    lens,
    field_points: Iterable[Tuple[float, float]],
    aperture_xy: torch.Tensor,
    pupil_z: float,
    object_depth_mm: float,
    wavelength_um: float,
    *,
    device: str,
    dtype: torch.dtype,
):
    """Trace all given field points at one wavelength in a single batched call.

    ``field_points`` is a sequence of ``(field_x_deg, field_y_deg)`` tuples.
    Returns a ``Ray`` with batch shape ``(F, num_rays)`` already traced to the
    sensor.  Mirrors ``_trace_fields_batch`` in the unrolled EF50 code.
    """
    N = aperture_xy.shape[0]
    F = len(list(field_points))
    field_points = list(field_points)

    origins = torch.empty((F, N, 3), dtype=dtype, device=device)
    targets = torch.empty((F, N, 3), dtype=dtype, device=device)

    for i, (fx, fy) in enumerate(field_points):
        x_obj, y_obj, z_obj = make_object_point(object_depth_mm, fx, fy)
        origins[i, :, 0] = x_obj
        origins[i, :, 1] = y_obj
        origins[i, :, 2] = z_obj
        targets[i, :, :2] = aperture_xy.to(dtype=dtype)
        targets[i, :, 2] = pupil_z

    ray = Ray(origins, targets - origins, wvln=wavelength_um, device=device)
    return lens.trace2sensor(ray)


def rms_spot_batch(ray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-field RMS spot and invalid-fraction for a batched ray.

    Returns ``(rms, invalid_fraction)`` each of shape ``(F,)``.
    """
    valid = ray.is_valid
    xy = ray.o[..., :2]
    denom = valid.sum(dim=-1).clamp_min(1e-8)
    centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)
    err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)
    rms = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()
    inv_frac = 1.0 - valid.sum(dim=-1) / valid.shape[-1]
    return rms, inv_frac
