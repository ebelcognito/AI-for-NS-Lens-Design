"""Batched GPU Cooke-triplet CMA-MEGA with Zernike behavioral descriptors.

Identical physics to triplet_cooke_zernike.py, but with a custom batched
spherical ray tracer that evaluates B lens designs simultaneously on GPU.
No DeepLens dependency for the inner tracing loop — pure PyTorch tensor ops.

Key speedup: seeding and calibration evaluate all designs in ONE GPU call
instead of looping one-by-one.

Usage:
  python -m optical_rl.qd.triplet_batched --config optical_rl/configs/triplet_cooke_zernike.yaml
"""
from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project / DeepLens paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = _PROJECT_ROOT / "DeepLens"


def _ensure_paths() -> None:
    for p in [str(_PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Reuse helpers from existing modules
# ---------------------------------------------------------------------------

from optical_rl.qd.singlet_zernike_measures import (  # noqa: E402
    NOLL_N_TERMS,
    NOLL_TERM_NAMES,
    build_zernike_pinv,
)


def _disk_samples(num_rays: int) -> np.ndarray:
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(max(num_rays, 1)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    result = np.empty((n, dim), dtype=np.float64)
    for d in range(dim):
        perm = rng.permutation(n)
        result[:, d] = (perm + rng.uniform(size=n)) / n
    return result


def _project_box(theta, bounds):
    return np.array(
        [np.clip(float(theta[i]), float(lo), float(hi)) for i, (lo, hi) in enumerate(bounds)],
        dtype=np.float64,
    )


def _theta_to_unit(theta, bounds):
    return np.array(
        [2.0 * (float(theta[i]) - lo) / max(hi - lo, 1e-12) - 1.0 for i, (lo, hi) in enumerate(bounds)],
        dtype=np.float64,
    )


def _unit_to_theta(unit, bounds):
    return np.array(
        [lo + 0.5 * (float(unit[i]) + 1.0) * (hi - lo) for i, (lo, hi) in enumerate(bounds)],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Config (same as TripletCookeZernikeConfig)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchedTripletConfig:
    lens_path: str = str(_PROJECT_ROOT / "DeepLens/datasets/lenses/cooke_smaller_ap_1mm.json")
    object_depth_mm: float = -1000.0
    wavelength_um: float = 0.587
    fitness_wavelengths_um: Tuple[float, ...] = (0.656, 0.587, 0.486)

    theta_bounds: Tuple = (
        (-0.06, 0.06),
        (-0.06, 0.06),
        (-0.06, 0.06),
        (-0.06, 0.06),
        (-0.06, 0.06),
    )

    stop_surface_index: int = 4
    focal_length_target_mm: Optional[float] = None

    descriptor_zernike_indices: Tuple[int, int] = (7, 4)
    archive_bins: Tuple[int, int] = (64, 64)
    archive_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    invalid_tol: float = 0.05
    invalid_penalty_weight: float = 5.0

    cma_eta: float = 0.05
    seed_count: int = 128
    calibration_count: int = 512
    num_rays: int = 256

    stage: str = "stage1"
    zernike_field_index: int = 0

    device: str = "auto"
    initial_sigma: float = 0.75
    population_size: int = 16
    cma_seed: int = 0
    results_dir: str = "results/models"
    run_name: str = "triplet_batched"
    log_every: int = 50

    # Stop-and-go checkpointing. Every checkpoint_every CMA-MEGA generations
    # the runner pickles a CMAMEGACheckpoint into result_dir/checkpoint.pkl.
    # 0 disables mid-run snapshots; the final checkpoint is always written
    # at clean shutdown (and on SIGINT). See _checkpoint.py for full doc.
    checkpoint_every: int = 50

    # Bonus added to candidate fitness in CMA-MEGA tell() when the candidate
    # filled a previously-empty cell. Implements canonical CMA-MEGA's
    # two-tier improvement ranking (new cells > replaced elites > none).
    new_cell_bonus: float = 1.0e6


# ---------------------------------------------------------------------------
# Eval result
# ---------------------------------------------------------------------------

@dataclass
class BatchedEvalResult:
    theta: np.ndarray
    fitness: float
    measures: Tuple[float, float]
    metrics: Dict[str, float]
    grads: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Batched Spherical Ray Tracer — pure PyTorch, no DeepLens objects
# ---------------------------------------------------------------------------

class BatchedSphericalTracer:
    """Traces rays through B different spherical-surface lens designs in parallel.

    All surfaces share the same geometry (z positions, aperture radii, glass types)
    — only curvatures differ per design.  This is the case for QD curvature
    optimisation where the template is fixed.
    """

    def __init__(self, surface_z, aperture_r, n_before, n_after,
                 refractive_surfaces, stop_index, device):
        """
        Args:
            surface_z: list of float — axial z-positions of each surface
            aperture_r: list of float — aperture radius of each surface
            n_before: list of float — refractive index before each surface
            n_after: list of float — refractive index after each surface
            refractive_surfaces: list of int — indices of refractive surfaces (skip stop)
            stop_index: int — index of the stop surface
            device: torch device
        """
        import torch
        self.torch = torch
        self.device = device
        self.surface_z = surface_z
        self.aperture_r = aperture_r
        self.n_before = n_before
        self.n_after = n_after
        self.refractive_surfaces = refractive_surfaces
        self.stop_index = stop_index
        self.n_surfaces = len(surface_z)

    def trace_batch(self, c_batch, origins, directions, d_sensor_batch, coherent=False):
        """Trace rays through B lens designs simultaneously.

        Args:
            c_batch: (B, n_refractive) curvatures per design
            origins: (B, ..., N, 3) ray origins
            directions: (B, ..., N, 3) ray directions (normalized)
            d_sensor_batch: (B,) sensor z-position per design
            coherent: bool — track OPL for wavefront analysis
        Returns:
            ray_o: (B, ..., N, 3) final positions at sensor
            valid: (B, ..., N) validity mask
            opl: (B, ..., N) optical path length (None if not coherent)
        """
        torch = self.torch
        ray_o = origins.clone()
        ray_d = directions.clone()
        shape = ray_o.shape[:-1]  # (B, ..., N)
        valid = torch.ones(shape, device=self.device, dtype=ray_o.dtype)
        opl = torch.zeros(shape, device=self.device, dtype=torch.float64) if coherent else None

        surf_k = 0  # index into c_batch columns
        for surf_idx in range(self.n_surfaces):
            if surf_idx == self.stop_index:
                continue

            c = c_batch[:, surf_k]  # (B,)
            # Reshape c for broadcasting: (B, 1, ..., 1) to match (B, ..., N)
            for _ in range(len(shape) - 1):
                c = c.unsqueeze(-1)

            n_in = self.n_before[surf_idx]
            n_out = self.n_after[surf_idx]
            z_surf = self.surface_z[surf_idx]
            r_ap = self.aperture_r[surf_idx]

            ray_o, ray_d, valid, opl = self._intersect_refract(
                ray_o, ray_d, valid, opl, c, z_surf, r_ap, n_in, n_out, coherent
            )
            surf_k += 1

        # Propagate to sensor
        d_s = d_sensor_batch
        for _ in range(len(shape) - 1):
            d_s = d_s.unsqueeze(-1)
        dz = ray_d[..., 2]
        dz_safe = torch.where(dz.abs() > 1e-12, dz, torch.full_like(dz, 1e-12))
        t = (d_s - ray_o[..., 2]) / dz_safe
        ray_o = ray_o + t.unsqueeze(-1) * ray_d
        if coherent:
            opl = opl + t  # air (n=1)

        return ray_o, valid, opl

    @staticmethod
    def _intersect_refract(ray_o, ray_d, valid, opl, c, z_surf, r_ap, n_in, n_out, coherent):
        """Intersect + refract at one spherical surface for all B designs.

        Uses the exact analytical ray-sphere intersection (quadratic formula),
        matching DeepLens spheric.py.

        c: (B, 1, ..., 1) curvatures, broadcastable with (B, ..., N)
        """
        import torch
        EPSILON = 1e-12

        # ---- 1. Transform to local coordinates (surface vertex at z=0) ----
        local_o = ray_o.clone()
        local_o[..., 2] = local_o[..., 2] - z_surf

        # ---- 2. Analytical ray-sphere intersection ----
        # Sphere: x² + y² + (z - R)² = R² where R = 1/c
        # For flat surfaces (c ≈ 0), use plane intersection
        c_abs = c.abs()
        is_flat = (c_abs < EPSILON)

        # Plane intersection: t = -z / dz
        t_plane = -local_o[..., 2] / (ray_d[..., 2] + EPSILON)

        # Sphere intersection: quadratic formula
        R = 1.0 / (c + torch.where(is_flat, torch.ones_like(c), torch.zeros_like(c)))
        oc = local_o.clone()
        oc[..., 2] = oc[..., 2] - R

        a = (ray_d * ray_d).sum(dim=-1)           # (B, ..., N)
        b = 2.0 * (oc * ray_d).sum(dim=-1)        # (B, ..., N)
        c_coeff = (oc * oc).sum(dim=-1) - R ** 2   # (B, ..., N)

        discriminant = b ** 2 - 4 * a * c_coeff
        valid_intersect = (discriminant >= 0)
        sqrt_disc = torch.sqrt(discriminant.clamp_min(EPSILON))

        t1 = (-b - sqrt_disc) / (2 * a + EPSILON)
        t2 = (-b + sqrt_disc) / (2 * a + EPSILON)

        # Choose root closest to surface vertex (z=0 in local coords)
        z1 = local_o[..., 2] + t1 * ray_d[..., 2]
        z2 = local_o[..., 2] + t2 * ray_d[..., 2]
        use_t1 = torch.abs(z1) < torch.abs(z2)
        t_sphere = torch.where(use_t1, t1, t2)

        # Use plane or sphere result
        t = torch.where(is_flat, t_plane, t_sphere)

        # New position (in local coords)
        new_local_o = local_o + t.unsqueeze(-1) * ray_d

        # ---- 3. Aperture check ----
        r2 = new_local_o[..., 0] ** 2 + new_local_o[..., 1] ** 2
        within_aperture = r2 <= (r_ap ** 2 + EPSILON)
        surf_valid = valid_intersect & within_aperture
        valid = valid * surf_valid.float()

        # Update ray origin (back to global coords)
        new_global_o = new_local_o.clone()
        new_global_o[..., 2] = new_global_o[..., 2] + z_surf
        ray_o = torch.where(surf_valid.unsqueeze(-1), new_global_o, ray_o)

        # ---- 4. OPL accumulation ----
        if coherent and opl is not None:
            opl = opl + n_in * t * surf_valid.float()

        # ---- 5. Surface normal from sag derivatives ----
        x, y = new_local_o[..., 0], new_local_o[..., 1]
        r2 = x ** 2 + y ** 2
        sf = (1 - r2 * c ** 2 + EPSILON).sqrt()
        dfdx = c * x / sf
        dfdy = c * y / sf
        # Normal = (-dfdx, -dfdy, 1) normalized
        nx = -dfdx
        ny = -dfdy
        nz = torch.ones_like(nx)
        n_norm = (nx ** 2 + ny ** 2 + nz ** 2).sqrt()
        normal = torch.stack([nx / n_norm, ny / n_norm, nz / n_norm], dim=-1)

        # ---- 6. Snell's law refraction ----
        eta = n_in / n_out
        cos_i = (-normal * ray_d).sum(-1, keepdim=True)
        k = 1 - eta ** 2 * (1 - cos_i ** 2)
        tir = (k < 0).squeeze(-1)
        valid = valid * (~tir).float()
        k = k.clamp_min(EPSILON)
        new_d = eta * ray_d + (eta * cos_i - k.sqrt()) * normal
        new_d = new_d / (new_d.norm(dim=-1, keepdim=True) + EPSILON)
        ray_d = torch.where(surf_valid.unsqueeze(-1) & (~tir.unsqueeze(-1)), new_d, ray_d)

        return ray_o, ray_d, valid, opl


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class BatchedTripletProblem:
    """Cooke-triplet problem with batched GPU evaluation."""

    _SURF_IDX = (0, 1, 2, 3, 5, 6)

    def __init__(self, config: BatchedTripletConfig):
        self.config = config
        self._backend_loaded = False
        self._torch = None
        self._device = None
        self._tracer: Optional[BatchedSphericalTracer] = None
        self._template_lens = None
        self._aperture_xy = None
        self._pinv_Z = None
        # Paraxial constants
        self._n1 = self._n2 = self._n3 = None
        self._d01 = self._d12 = self._d23 = self._d35 = self._d56 = None
        self._z_s6 = None
        self._f_target: Optional[float] = None
        # Multi-wavelength refractive indices: {wvln: (n1, n2, n3)}
        self._n_per_wvln: Dict[float, Tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return
        _ensure_paths()
        import torch
        from deeplens.optics import GeoLens

        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        self._torch = torch
        self._device = device

        # Load template lens via DeepLens (only for extracting geometry)
        lens = GeoLens(filename=self.config.lens_path, device=device)

        wvln = float(self.config.wavelength_um)
        self._n1 = float(lens.surfaces[0].mat2.refractive_index(wvln))
        self._n2 = float(lens.surfaces[2].mat2.refractive_index(wvln))
        self._n3 = float(lens.surfaces[5].mat2.refractive_index(wvln))

        d = [float(lens.surfaces[i].d.item()) for i in range(7)]
        self._d01 = d[1] - d[0]
        self._d12 = d[2] - d[1]
        self._d23 = d[3] - d[2]
        self._d35 = d[5] - d[3]
        self._d56 = d[6] - d[5]
        self._z_s6 = d[6]

        if self.config.focal_length_target_mm is not None:
            self._f_target = float(self.config.focal_length_target_mm)
        else:
            self._f_target = None

        # Precompute refractive indices for each fitness wavelength
        for w in self.config.fitness_wavelengths_um:
            n1w = float(lens.surfaces[0].mat2.refractive_index(w))
            n2w = float(lens.surfaces[2].mat2.refractive_index(w))
            n3w = float(lens.surfaces[5].mat2.refractive_index(w))
            self._n_per_wvln[w] = (n1w, n2w, n3w)
        # Also add the d-line if not already
        if wvln not in self._n_per_wvln:
            self._n_per_wvln[wvln] = (self._n1, self._n2, self._n3)

        # Surface geometry (shared across all designs)
        surface_z = [float(lens.surfaces[i].d.item()) for i in range(7)]
        aperture_r = [float(lens.surfaces[i].r) for i in range(7)]

        # Build per-surface n_before/n_after for d-line (default).
        # For multi-wavelength, we'll create per-wavelength tracers.
        self._surface_z = surface_z
        self._aperture_r = aperture_r
        self._sensor_half_size = float(lens.sensor_size[0]) / 2.0

        self._tracers: Dict[float, BatchedSphericalTracer] = {}
        for w, (n1w, n2w, n3w) in self._n_per_wvln.items():
            n_before = [1.0, n1w, 1.0, n2w, 1.0, 1.0, n3w]
            n_after = [n1w, 1.0, n2w, 1.0, 1.0, n3w, 1.0]
            self._tracers[w] = BatchedSphericalTracer(
                surface_z=surface_z,
                aperture_r=aperture_r,
                n_before=n_before,
                n_after=n_after,
                refractive_surfaces=list(self._SURF_IDX),
                stop_index=int(self.config.stop_surface_index),
                device=device,
            )

        # Aperture disk
        stop_r = float(lens.surfaces[self.config.stop_surface_index].r)
        disk = _disk_samples(int(self.config.num_rays))
        self._aperture_xy = torch.tensor(
            disk * stop_r, dtype=torch.float32, device=device
        )
        self._disk_unit = disk  # unit-disk for Zernike

        # Zernike pseudo-inverse
        j_max = max(int(j) for j in self.config.descriptor_zernike_indices)
        pinv_np = build_zernike_pinv(disk, j_max=j_max)
        self._pinv_Z = torch.tensor(pinv_np, dtype=torch.float64, device=device)

        self._template_lens = lens  # kept for fallback single-design evaluation
        self._backend_loaded = True

    # ------------------------------------------------------------------
    # Batched paraxial solve
    # ------------------------------------------------------------------

    def _batched_paraxial_fixed_efl(self, c_batch):
        """c_batch: (B, 5). Returns c6: (B,), d_sensor: (B,)."""
        torch = self._torch
        B = c_batch.shape[0]
        n1, n2, n3 = self._n1, self._n2, self._n3
        d01, d12, d23 = self._d01, self._d12, self._d23
        d35, d56 = self._d35, self._d56

        dtype, dev = c_batch.dtype, c_batch.device
        y = torch.ones(B, dtype=dtype, device=dev)
        nu = torch.zeros(B, dtype=dtype, device=dev)

        nu = nu - (n1 - 1.0) * c_batch[:, 0] * y;  y = y + (d01 / n1) * nu
        nu = nu - (1.0 - n1) * c_batch[:, 1] * y;  y = y + d12 * nu
        nu = nu - (n2 - 1.0) * c_batch[:, 2] * y;  y = y + (d23 / n2) * nu
        nu = nu - (1.0 - n2) * c_batch[:, 3] * y;  y = y + d35 * nu
        nu = nu - (n3 - 1.0) * c_batch[:, 4] * y;  y = y + (d56 / n3) * nu

        P = 1.0 / self._f_target
        safe_y = torch.where(torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6))
        c6 = (nu + P) / safe_y / (1.0 - n3)
        bfl = y * float(self._f_target)
        d_sensor = self._z_s6 + bfl
        return c6, d_sensor

    def _batched_paraxial_bfl(self, c_batch):
        """c_batch: (B, 6). Returns d_sensor: (B,)."""
        torch = self._torch
        B = c_batch.shape[0]
        n1, n2, n3 = self._n1, self._n2, self._n3
        d01, d12, d23 = self._d01, self._d12, self._d23
        d35, d56 = self._d35, self._d56

        dtype, dev = c_batch.dtype, c_batch.device
        y = torch.ones(B, dtype=dtype, device=dev)
        nu = torch.zeros(B, dtype=dtype, device=dev)

        nu = nu - (n1 - 1.0) * c_batch[:, 0] * y;  y = y + (d01 / n1) * nu
        nu = nu - (1.0 - n1) * c_batch[:, 1] * y;  y = y + d12 * nu
        nu = nu - (n2 - 1.0) * c_batch[:, 2] * y;  y = y + (d23 / n2) * nu
        nu = nu - (1.0 - n2) * c_batch[:, 3] * y;  y = y + d35 * nu
        nu = nu - (n3 - 1.0) * c_batch[:, 4] * y;  y = y + (d56 / n3) * nu
        nu = nu - (1.0 - n3) * c_batch[:, 5] * y

        safe_nu = torch.where(torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6))
        bfl = -y / safe_nu
        return self._z_s6 + bfl

    def _batched_calc_hfov(self, efl_batch):
        """efl_batch: (B,) effective focal lengths. Returns (B,) HFOV in degrees."""
        torch = self._torch
        half_sensor = self._sensor_half_size
        return torch.rad2deg(torch.atan(half_sensor / efl_batch.abs().clamp_min(1e-6)))

    # ------------------------------------------------------------------
    # Build batched curvatures + d_sensor
    # ------------------------------------------------------------------

    def _prepare_batch(self, theta_batch_t):
        """theta_batch_t: (B, D) tensor. Returns c_all: (B, 6), d_sensor: (B,), bfl: (B,)."""
        torch = self._torch
        if self._f_target is not None:
            c6, d_sensor = self._batched_paraxial_fixed_efl(theta_batch_t)
            c_all = torch.cat([theta_batch_t, c6.unsqueeze(-1)], dim=-1)  # (B, 6)
            hfov = self._batched_calc_hfov(
                torch.full((theta_batch_t.shape[0],), self._f_target,
                           device=self._device, dtype=theta_batch_t.dtype)
            )
        else:
            c_all = theta_batch_t  # (B, 6)
            d_sensor = self._batched_paraxial_bfl(theta_batch_t)
            # EFL from nu_final
            B = theta_batch_t.shape[0]
            n1, n2, n3 = self._n1, self._n2, self._n3
            d01, d12, d23 = self._d01, self._d12, self._d23
            d35, d56 = self._d35, self._d56
            dtype, dev = theta_batch_t.dtype, theta_batch_t.device
            y = torch.ones(B, dtype=dtype, device=dev)
            nu = torch.zeros(B, dtype=dtype, device=dev)
            nu = nu - (n1-1)*theta_batch_t[:,0]*y; y = y + (d01/n1)*nu
            nu = nu - (1-n1)*theta_batch_t[:,1]*y; y = y + d12*nu
            nu = nu - (n2-1)*theta_batch_t[:,2]*y; y = y + (d23/n2)*nu
            nu = nu - (1-n2)*theta_batch_t[:,3]*y; y = y + d35*nu
            nu = nu - (n3-1)*theta_batch_t[:,4]*y; y = y + (d56/n3)*nu
            nu = nu - (1-n3)*theta_batch_t[:,5]*y
            efl = 1.0 / nu.abs().clamp_min(1e-8)
            hfov = self._batched_calc_hfov(efl)

        # BFL check
        bfl = d_sensor - self._z_s6
        return c_all, d_sensor, hfov, bfl

    # ------------------------------------------------------------------
    # Build ray origins for field points
    # ------------------------------------------------------------------

    def _build_field_rays(self, B, field_points, dtype):
        """Build ray origins and directions for all B designs × F field points.

        Returns origins: (B, F, N, 3), directions: (B, F, N, 3)
        """
        torch = self._torch
        N = self._aperture_xy.shape[0]
        F = len(field_points)
        depth = float(self.config.object_depth_mm)
        s0_z = self._surface_z[self._SURF_IDX[0]]

        origins = torch.empty((B, F, N, 3), dtype=dtype, device=self._device)
        targets = torch.empty((B, F, N, 3), dtype=dtype, device=self._device)

        for fi, (fx, fy) in enumerate(field_points):
            x_obj = depth * math.tan(math.radians(fx))
            y_obj = depth * math.tan(math.radians(fy))
            origins[:, fi, :, 0] = x_obj
            origins[:, fi, :, 1] = y_obj
            origins[:, fi, :, 2] = depth
            targets[:, fi, :, :2] = self._aperture_xy.to(dtype=dtype)
            targets[:, fi, :, 2] = s0_z

        directions = targets - origins
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-12)
        return origins, directions

    # ------------------------------------------------------------------
    # Batched RMS spot
    # ------------------------------------------------------------------

    @staticmethod
    def _batched_rms(ray_pos, valid):
        """ray_pos: (B, F, N, 3), valid: (B, F, N). Returns rms: (B, F), inv_frac: (B, F)."""
        xy = ray_pos[..., :2]                                                    # (B, F, N, 2)
        denom = valid.sum(dim=-1).clamp_min(1e-8)                                # (B, F)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)   # (B, F, 2)
        err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)                  # (B, F, N)
        rms = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()                # (B, F)
        inv_frac = 1.0 - valid.sum(dim=-1) / valid.shape[-1]                     # (B, F)
        return rms, inv_frac

    # ------------------------------------------------------------------
    # Batched Zernike OPD
    # ------------------------------------------------------------------

    def _batched_zernike(self, c_all, d_sensor, hfov_deg):
        """Compute Zernike coefficients for B designs in parallel.

        Returns coeffs: (B, j_max) tensor.
        """
        torch = self._torch
        B = c_all.shape[0]
        N = self._aperture_xy.shape[0]
        wvln = float(self.config.wavelength_um)
        tracer = self._tracers[wvln]

        # Field angle for Zernike
        idx = int(self.config.zernike_field_index)
        # hfov_deg is (B,) — use the mean for field points (all designs are evaluated
        # at the same field fractions; we use per-design HFOV)
        # For simplicity, use the mean HFOV across the batch for the Zernike field angle
        mean_hfov = float(hfov_deg.mean().item())
        field_pts = self._field_points_from_hfov(mean_hfov)
        field = field_pts[idx][1]  # meridional (y) component

        z_obj_val = float(self.config.object_depth_mm)
        z_obj_abs = abs(z_obj_val)
        y_obj = z_obj_val * math.tan(math.radians(field))
        s0_z = self._surface_z[self._SURF_IDX[0]]

        # Build coherent ray bundle: (B, N, 3)
        dtype = torch.float64
        origins = torch.zeros((B, N, 3), dtype=dtype, device=self._device)
        origins[:, :, :2] = self._aperture_xy.to(dtype=dtype)
        origins[:, :, 2] = s0_z

        obj = torch.tensor([0.0, y_obj, z_obj_val], dtype=dtype, device=self._device)
        directions = origins - obj.unsqueeze(0).unsqueeze(0)  # (B, N, 3)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-12)

        # Coherent trace
        ray_o, valid, opl = tracer.trace_batch(
            c_all.to(dtype), origins, directions, d_sensor.to(dtype), coherent=True
        )

        # Propagate to exit pupil: approximate as BFL from sensor
        # Exit pupil z ≈ d_sensor - BFL (paraxial approximation)
        bfl = d_sensor - self._z_s6  # (B,)
        pupil_z = d_sensor - bfl     # = z_s6 for all designs
        # Actually for a fixed-EFL triplet, exit pupil ≈ z_s6.
        # More precisely: we'll use z_s6 as the exit pupil reference.
        # Propagate rays to pupil_z
        pupil_z_scalar = float(self._z_s6)
        t_pupil = (pupil_z_scalar - ray_o[..., 2]) / (ray_o.new_ones(ray_o[..., 2].shape) * 1e-12 + 1.0)
        # Actually prop_to with z = pupil_z:
        t_to_pupil = (pupil_z_scalar - ray_o[..., 2])  # This isn't right for diverging rays
        # Use full prop formula: need ray_d after trace — but we only have ray_o from trace_batch
        # The tracer returns final positions but not final directions.

        # We need to also return ray directions from the tracer for OPD prop.
        # Let me fix the approach: compute OPD at the sensor plane directly
        # and skip exit pupil propagation for the batched version.
        # OPD at sensor: opl already accumulated up to sensor.

        # Incoming OPL correction
        xy_stop = self._aperture_xy.to(dtype)  # (N, 2)
        xp, yp = xy_stop[:, 0], xy_stop[:, 1]
        opl_marginal_in = torch.sqrt(xp ** 2 + (yp - y_obj) ** 2 + z_obj_abs ** 2)
        opl_chief_in = math.sqrt(y_obj ** 2 + z_obj_abs ** 2)
        opl = opl + (opl_marginal_in.unsqueeze(0) - opl_chief_in)  # (B, N)

        # Chief ray: trace a single ray per design through center of aperture
        chief_origins = torch.zeros((B, 1, 3), dtype=dtype, device=self._device)
        chief_origins[:, 0, 2] = s0_z
        chief_dir = torch.tensor([0.0, math.sin(math.radians(field)),
                                  math.cos(math.radians(field))],
                                 dtype=dtype, device=self._device)
        chief_directions = chief_dir.unsqueeze(0).unsqueeze(0).expand(B, 1, 3).clone()

        chief_o, chief_valid, chief_opl = tracer.trace_batch(
            c_all.to(dtype), chief_origins, chief_directions, d_sensor.to(dtype), coherent=True
        )
        opl_chief = chief_opl.squeeze(-1)  # (B,)

        # Reference sphere: use sensor-plane geometry
        # BFL = distance from last surface to sensor
        bfl_t = (d_sensor - self._z_s6).to(dtype)  # (B,)

        # At sensor: chief ray lands at chief_o, marginal rays land at ray_o
        x_chief = chief_o[:, 0, 0]  # (B,)
        y_chief = chief_o[:, 0, 1]  # (B,)
        x_marg = ray_o[..., 0]  # (B, N)
        y_marg = ray_o[..., 1]  # (B, N)

        dx = x_marg - x_chief.unsqueeze(-1)
        dy = y_marg - y_chief.unsqueeze(-1)
        ref_sphere = torch.sqrt(dx ** 2 + dy ** 2 + bfl_t.unsqueeze(-1) ** 2) - bfl_t.unsqueeze(-1)

        # OPD
        opd = ((opl - opl_chief.unsqueeze(-1)) + ref_sphere) * valid  # (B, N)

        # Zernike fit: pinv_Z is (j_max, N), opd is (B, N)
        # Result: (B, j_max)
        coeffs = torch.matmul(self._pinv_Z, opd.to(self._pinv_Z.dtype).T).T  # (B, j_max)

        return coeffs

    # ------------------------------------------------------------------
    # Field points helper
    # ------------------------------------------------------------------

    def _field_points_from_hfov(self, hfov_deg: float):
        if str(self.config.stage).lower().strip() == "stage2":
            return ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))
        return ((0.0, 0.0),)

    # ------------------------------------------------------------------
    # Batched evaluate — the main speedup
    # ------------------------------------------------------------------

    def batched_evaluate(self, theta_batch: np.ndarray) -> list:
        """Evaluate B designs in parallel on GPU.

        Args:
            theta_batch: (B, D) numpy array
        Returns:
            list of BatchedEvalResult
        """
        self._ensure_backend()
        torch = self._torch
        B = theta_batch.shape[0]

        # Project to bounds
        projected = np.array([_project_box(theta_batch[i], self.config.theta_bounds)
                              for i in range(B)], dtype=np.float64)
        theta_t = torch.tensor(projected, dtype=torch.float32, device=self._device)

        # Batched paraxial solve
        c_all, d_sensor, hfov, bfl = self._prepare_batch(theta_t)

        # BFL validity mask
        bfl_valid = (bfl > 0).float()  # (B,)

        # Field points — use mean HFOV for field angle positions
        mean_hfov = float(hfov.mean().item())
        field_pts = self._field_points_from_hfov(mean_hfov)

        # Build rays for all B × F field points
        origins, directions = self._build_field_rays(B, field_pts, theta_t.dtype)

        # Trace at each wavelength
        wavelengths = self.config.fitness_wavelengths_um
        rms_per_wvln = []
        inv_per_wvln = []
        for w in wavelengths:
            tracer = self._tracers[w]
            # c_all: (B, 6) → expand for (B, F, N, 3) via tracer broadcasting
            ray_o, valid, _ = tracer.trace_batch(
                c_all, origins, directions, d_sensor, coherent=False
            )
            rms, inv_frac = self._batched_rms(ray_o, valid)  # (B, F)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        # Average over wavelengths and field points
        rms_all = torch.stack(rms_per_wvln)    # (W, B, F)
        inv_all = torch.stack(inv_per_wvln)    # (W, B, F)
        rms_mean = rms_all.mean(dim=(0, 2))    # (B,)
        inv_mean = inv_all.mean(dim=(0, 2))    # (B,)

        fitness = -rms_mean  # (B,)

        # Invalid-fraction penalty (proportional to excess above tolerance)
        excess = (inv_mean - self.config.invalid_tol).clamp_min(0)
        fitness = fitness - self.config.invalid_penalty_weight * excess

        # BFL penalty: designs with BFL <= 0
        fitness = torch.where(bfl_valid > 0, fitness, torch.full_like(fitness, -1e9))

        # Batched Zernike
        coeffs = self._batched_zernike(c_all, d_sensor, hfov)  # (B, j_max)
        j0, j1 = self.config.descriptor_zernike_indices
        m0 = coeffs[:, j0 - 1]  # (B,)
        m1 = coeffs[:, j1 - 1]  # (B,)

        # Convert to results
        results = []
        for i in range(B):
            metrics = {
                "rms_spot": float(rms_mean[i].detach().cpu().item()),
                "invalid_fraction": float(inv_mean[i].detach().cpu().item()),
                f"z{j0}_{NOLL_TERM_NAMES.get(j0, 'unknown')}_mm": float(m0[i].detach().cpu().item()),
                f"z{j1}_{NOLL_TERM_NAMES.get(j1, 'unknown')}_mm": float(m1[i].detach().cpu().item()),
            }
            results.append(BatchedEvalResult(
                theta=projected[i].astype(np.float64, copy=False),
                fitness=float(fitness[i].detach().cpu().item()),
                measures=(float(m0[i].detach().cpu().item()), float(m1[i].detach().cpu().item())),
                metrics=metrics,
            ))

        return results

    # ------------------------------------------------------------------
    # Single-design evaluate (for CMA-MEGA loop with gradients)
    # ------------------------------------------------------------------

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> BatchedEvalResult:
        """Single-design evaluation. Uses batched_evaluate for no-grad path."""
        if not with_grads:
            return self.batched_evaluate(theta.reshape(1, -1))[0]

        # With gradients: use the batched tracer but with B=1
        self._ensure_backend()
        torch = self._torch
        theta_proj = _project_box(theta, self.config.theta_bounds)
        theta_t = torch.tensor(
            theta_proj, dtype=torch.float32, device=self._device, requires_grad=True
        )

        # Single-design paraxial
        c_batch = theta_t.unsqueeze(0)  # (1, D)
        c_all, d_sensor, hfov, bfl = self._prepare_batch(c_batch)

        if float(bfl[0].detach()) <= 0:
            bad_f = theta_t.sum() * 0.0 - 1e9
            return BatchedEvalResult(
                theta=theta_proj, fitness=-1e9,
                measures=(0.0, 0.0), metrics={"invalid_fraction": 1.0},
                grads=None,
            )

        mean_hfov = float(hfov[0].detach().item())
        field_pts = self._field_points_from_hfov(mean_hfov)
        origins, directions = self._build_field_rays(1, field_pts, theta_t.dtype)

        wavelengths = self.config.fitness_wavelengths_um
        rms_per_wvln = []
        for w in wavelengths:
            tracer = self._tracers[w]
            ray_o, valid, _ = tracer.trace_batch(c_all, origins, directions, d_sensor)
            rms, _ = self._batched_rms(ray_o, valid)
            rms_per_wvln.append(rms)

        rms_mean = torch.stack(rms_per_wvln).mean()
        fitness = -rms_mean

        # Zernike (single)
        coeffs = self._batched_zernike(c_all, d_sensor, hfov)
        j0, j1 = self.config.descriptor_zernike_indices
        m0 = coeffs[0, j0 - 1]
        m1 = coeffs[0, j1 - 1]

        grads = None
        if with_grads:
            g_f = torch.autograd.grad(fitness, theta_t, retain_graph=True, allow_unused=True)[0]
            g_m0 = torch.autograd.grad(m0, theta_t, retain_graph=True, allow_unused=True)[0]
            g_m1 = torch.autograd.grad(m1, theta_t, retain_graph=False, allow_unused=True)[0]
            k0, k1 = f"z{j0}", f"z{j1}"
            grads = {
                "fitness": g_f.detach().cpu().numpy().astype(np.float64) if g_f is not None else np.zeros_like(theta_proj),
                k0: g_m0.detach().cpu().numpy().astype(np.float64) if g_m0 is not None else np.zeros_like(theta_proj),
                k1: g_m1.detach().cpu().numpy().astype(np.float64) if g_m1 is not None else np.zeros_like(theta_proj),
            }

        return BatchedEvalResult(
            theta=theta_proj,
            fitness=float(fitness.detach().cpu().item()),
            measures=(float(m0.detach().cpu().item()), float(m1.detach().cpu().item())),
            metrics={
                "rms_spot": float(rms_mean.detach().cpu().item()),
                f"z{j0}_{NOLL_TERM_NAMES.get(j0, 'unknown')}_mm": float(m0.detach().cpu().item()),
                f"z{j1}_{NOLL_TERM_NAMES.get(j1, 'unknown')}_mm": float(m1.detach().cpu().item()),
            },
            grads=grads,
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def initial_theta(self) -> np.ndarray:
        return np.zeros(len(self.config.theta_bounds), dtype=np.float64)

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        dim = len(self.config.theta_bounds)
        unit = _latin_hypercube(int(n), dim, int(seed))
        lo = np.array([b[0] for b in self.config.theta_bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self.config.theta_bounds], dtype=np.float64)
        return lo + unit * (hi - lo)

    def project_theta(self, theta):
        return _project_box(theta, self.config.theta_bounds)


# ---------------------------------------------------------------------------
# Archive (reuse from triplet)
# ---------------------------------------------------------------------------

class BatchedArchive2D:
    def __init__(self, bins, bounds, *, seed: int = 0):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bounds = bounds
        self._rng = np.random.default_rng(int(seed))
        self._cells: Dict[Tuple[int, int], BatchedEvalResult] = {}
        self._best_fitness = -float("inf")

    def _to_index(self, measures):
        (x_lo, x_hi), (y_lo, y_hi) = self.bounds
        x, y = float(measures[0]), float(measures[1])
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return None
        def _bin(v, lo, hi, n):
            return int(np.clip(np.floor((v - lo) / (hi - lo) * n), 0, n - 1))
        return (_bin(x, x_lo, x_hi, self.bins[0]), _bin(y, y_lo, y_hi, self.bins[1]))

    def insert(self, result: BatchedEvalResult) -> int:
        """Insert a candidate. Returns INSERT_{NONE,REPLACED,NEW} from
        optical_rl.qd._checkpoint. Truthy if any kind of improvement
        occurred, so legacy `if archive.insert(...)` call sites still work.
        """
        from optical_rl.qd._checkpoint import INSERT_NONE, INSERT_REPLACED, INSERT_NEW
        idx = self._to_index(result.measures)
        if idx is None:
            return INSERT_NONE
        current = self._cells.get(idx)
        self._best_fitness = max(self._best_fitness, float(result.fitness))
        if current is None:
            self._cells[idx] = result
            return INSERT_NEW
        if float(result.fitness) > float(current.fitness):
            self._cells[idx] = result
            return INSERT_REPLACED
        return INSERT_NONE

    def sample_parent(self) -> np.ndarray:
        if not self._cells:
            raise RuntimeError("archive is empty")
        keys = list(self._cells.keys())
        idx = keys[int(self._rng.integers(0, len(keys)))]
        return self._cells[idx].theta.copy()

    def stats(self) -> Dict[str, float]:
        total = self.bins[0] * self.bins[1]
        filled = len(self._cells)
        return {
            "filled": float(filled),
            "total_cells": float(total),
            "coverage": float(filled / max(total, 1)),
            "best_fitness": float(self._best_fitness if filled else -float("inf")),
        }

    @property
    def cells(self):
        return dict(self._cells)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA  # noqa: E402


class BatchedTripletRunner:
    """CMA-MEGA runner with batched GPU seeding."""

    def __init__(self, config: BatchedTripletConfig):
        self.config = config
        self.problem = BatchedTripletProblem(config)
        self.archive: Optional[BatchedArchive2D] = None
        self.emitter = GradientCoefficientCMA(
            dim=1 + len(config.descriptor_zernike_indices),
            seed=int(config.cma_seed),
            sigma=float(config.initial_sigma),
            population_size=int(config.population_size),
        )
        # Persistent runner state — survives checkpoint/resume.
        self.completed: int = 0
        self.generation: int = 0
        self.restart_count: int = 0
        self.wallclock_seconds: float = 0.0
        self.result_dir: Optional["Path"] = None

    def calibrate_archive_bounds(self):
        """Batched calibration — all designs evaluated in one GPU call."""
        samples = self.problem.sample_theta(self.config.calibration_count, self.config.cma_seed + 17)
        results = self.problem.batched_evaluate(samples)
        m0_vals = [r.measures[0] for r in results]
        m1_vals = [r.measures[1] for r in results]

        def _bounds(arr):
            lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
            span = max(hi - lo, 1e-6)
            return (lo - 0.05 * span, hi + 0.05 * span)

        bounds = (
            _bounds(np.asarray(m0_vals, dtype=np.float64)),
            _bounds(np.asarray(m1_vals, dtype=np.float64)),
        )
        self.archive = BatchedArchive2D(self.config.archive_bins, bounds, seed=self.config.cma_seed)
        return bounds

    def seed_archive(self) -> Dict[str, float]:
        """Batched seeding — all designs evaluated in one GPU call."""
        if self.archive is None:
            if self.config.archive_bounds is not None:
                self.archive = BatchedArchive2D(
                    self.config.archive_bins,
                    self.config.archive_bounds,
                    seed=self.config.cma_seed,
                )
            else:
                self.calibrate_archive_bounds()

        # Initial + seed designs — all at once
        flat = self.problem.initial_theta()
        seed_thetas = self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31)
        all_thetas = np.vstack([flat.reshape(1, -1), seed_thetas])

        results = self.problem.batched_evaluate(all_thetas)
        for r in results:
            self.archive.insert(r)
        return self.archive.stats()

    def _normalized_gradients(self, grads, theta_bounds):
        half = 0.5 * np.array([hi - lo for lo, hi in theta_bounds], dtype=np.float64)
        j0, j1 = self.config.descriptor_zernike_indices

        def norm(name):
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        return {"fitness": norm("fitness"), f"z{j0}": norm(f"z{j0}"), f"z{j1}": norm(f"z{j1}")}

    def run(self, iterations: int, *, resume: bool = False) -> Dict[str, float]:
        """Canonical CMA-MEGA loop with restart on no-improvement,
        three-tier improvement ranking, periodic checkpointing, and a
        per-generation trajectory CSV. Mirrors EF50ZernikeRunner.run with
        the only difference being batched candidate evaluation.
        """
        import signal
        import time
        from optical_rl.qd._checkpoint import (
            CMAMEGACheckpoint,
            INSERT_NEW,
            INSERT_REPLACED,
            TrajectoryLogger,
            hash_config,
            now_iso,
            save_checkpoint,
        )

        if not resume and (self.archive is None or not self.archive.cells):
            self.seed_archive()

        j0, j1 = self.config.descriptor_zernike_indices
        theta_dim = len(self.config.theta_bounds)
        scores: list = []
        eval_budget = int(iterations)
        if not resume:
            self.completed = 0
            self.generation = 0
            self.restart_count = 0
            self.wallclock_seconds = 0.0
        target_completed = self.completed + eval_budget

        result_dir = self.result_dir
        ckpt_path = (Path(result_dir) / "checkpoint.pkl") if result_dir is not None else None
        traj_logger = (
            TrajectoryLogger(Path(result_dir) / "trajectory.csv")
            if result_dir is not None else None
        )
        cfg_hash = hash_config(self.config)
        checkpoint_every = max(0, int(self.config.checkpoint_every))
        new_cell_bonus = float(self.config.new_cell_bonus)

        def _build_ckpt() -> "CMAMEGACheckpoint":
            return CMAMEGACheckpoint(
                archive=self.archive,
                emitter_state=self.emitter.get_state(),
                completed_evaluations=int(self.completed),
                generation=int(self.generation),
                restart_count=int(self.restart_count),
                best_fitness=float(self.archive.stats()["best_fitness"]),
                config_hash=cfg_hash,
                timestamp=now_iso(),
                wallclock_seconds=float(self.wallclock_seconds),
            )

        def _flush_checkpoint() -> None:
            if ckpt_path is None:
                return
            save_checkpoint(_build_ckpt(), ckpt_path)

        sigint_received = {"flag": False}
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):  # noqa: ARG001
            sigint_received["flag"] = True
            print(
                "[triplet-batched] SIGINT received — checkpointing then exiting.",
                flush=True,
            )

        signal.signal(signal.SIGINT, _handle_sigint)
        loop_t0 = time.perf_counter()

        try:
            while self.completed < target_completed:
                if sigint_received["flag"]:
                    break
                # Sample a parent with valid gradients (retry up to 10 times)
                parent_eval = None
                for _retry in range(10):
                    parent_theta = self.archive.sample_parent()
                    parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
                    if parent_eval.grads is not None:
                        break
                if parent_eval is None or parent_eval.grads is None:
                    parent_theta = self.problem.sample_theta(1, self.config.cma_seed + self.completed)[0]
                    parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
                    if parent_eval.grads is None:
                        self.completed += 1
                        continue

                unit_parent = _theta_to_unit(parent_theta, self.config.theta_bounds)
                grads = self._normalized_gradients(parent_eval.grads, self.config.theta_bounds)
                batch_n = min(int(self.config.population_size), target_completed - self.completed)
                coeffs = self.emitter.ask(batch_n)

                candidate_thetas = []
                for local_idx, coeff in enumerate(coeffs):
                    direction = (
                        abs(coeff[0]) * grads["fitness"]
                        + coeff[1] * grads[f"z{j0}"]
                        + coeff[2] * grads[f"z{j1}"]
                    )
                    dir_norm = np.linalg.norm(direction)
                    if not np.isfinite(dir_norm) or dir_norm < 1e-12:
                        rng = np.random.default_rng(self.config.cma_seed + self.completed + local_idx)
                        direction = rng.normal(0.0, 1.0, size=(theta_dim,))
                        direction /= np.linalg.norm(direction) + 1e-12

                    candidate_unit = np.clip(
                        unit_parent + float(self.config.cma_eta) * direction, -1.0, 1.0
                    )
                    candidate_thetas.append(_unit_to_theta(candidate_unit, self.config.theta_bounds))

                candidate_batch = np.array(candidate_thetas, dtype=np.float64)
                candidate_results = self.problem.batched_evaluate(candidate_batch)

                batch_scores: list = []
                batch_status: list = []
                for candidate_eval in candidate_results:
                    status = int(self.archive.insert(candidate_eval))
                    scores.append(candidate_eval.fitness)
                    if status == INSERT_NEW:
                        batch_scores.append(candidate_eval.fitness + new_cell_bonus)
                    elif status == INSERT_REPLACED:
                        batch_scores.append(candidate_eval.fitness)
                    else:
                        batch_scores.append(-np.inf)
                    batch_status.append(status)

                self.emitter.tell(coeffs, np.asarray(batch_scores, dtype=np.float64))
                self.completed += batch_n
                self.generation += 1

                n_new = sum(1 for s in batch_status if s == INSERT_NEW)
                n_replaced = sum(1 for s in batch_status if s == INSERT_REPLACED)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted = sum(1 for i in elite_idx if batch_status[i] == 0)

                restart_reason = None
                if (n_new + n_replaced) == 0:
                    restart_reason = "no_improvement"
                else:
                    restart_flag, hansen_reason = self.emitter.should_restart()
                    if restart_flag:
                        restart_reason = hansen_reason or "hansen_term"
                if restart_reason is not None:
                    sigma_pre = float(self.emitter.sigma)
                    self.emitter.reset()
                    self.restart_count += 1
                    print(
                        f"[triplet-batched] [restart] gen={self.generation} "
                        f"reason={restart_reason} sigma_pre={sigma_pre:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                self.wallclock_seconds += (time.perf_counter() - loop_t0)
                loop_t0 = time.perf_counter()

                if traj_logger is not None:
                    s = self.archive.stats()
                    traj_logger.log(
                        gen=self.generation,
                        evals=self.completed,
                        coverage=float(s["coverage"]),
                        best_fitness=float(s["best_fitness"]),
                        new_cells=n_new,
                        replaced_elites=n_replaced,
                        wasted_elites=n_wasted,
                        sigma=float(self.emitter.sigma),
                        restarts=self.restart_count,
                        wallclock_s=float(self.wallclock_seconds),
                    )

                if self.generation % max(int(self.config.log_every), 1) == 0:
                    s = self.archive.stats()
                    print(
                        f"[triplet-batched] gen={self.generation} evals={self.completed} "
                        f"coverage={s['coverage']:.4f} best={s['best_fitness']:.6f} | "
                        f"new={n_new}/{batch_n} replaced={n_replaced}/{batch_n} "
                        f"wasted={n_wasted}/{elite_n} "
                        f"sigma={self.emitter.sigma:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                if checkpoint_every > 0 and self.generation % checkpoint_every == 0:
                    _flush_checkpoint()
        finally:
            try:
                _flush_checkpoint()
            except Exception as e:  # pragma: no cover
                print(f"[triplet-batched] WARNING: final checkpoint failed: {e}", flush=True)
            signal.signal(signal.SIGINT, prev_handler)

        s = self.archive.stats()
        s["iterations"] = float(eval_budget)
        s["generations"] = float(self.generation)
        s["restart_count"] = float(self.restart_count)
        valid_scores = [v for v in scores if v > -1e8 and np.isfinite(v)]
        s["mean_score"] = float(np.mean(valid_scores)) if valid_scores else float("nan")
        return s


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> BatchedTripletConfig:
    _ensure_paths()
    from optical_rl.utils.config import load_config
    raw = load_config(config_path)
    qd = raw.get("qd", raw)

    _fl_raw = qd.get("focal_length_target_mm")
    focal_length_target_mm = float(_fl_raw) if _fl_raw is not None else None
    theta_dim = 5 if focal_length_target_mm is not None else 6
    theta_bounds_default = [[-0.06, 0.06]] * theta_dim

    archive_bounds = qd.get("archive_bounds")
    if archive_bounds is not None:
        archive_bounds = (tuple(archive_bounds[0]), tuple(archive_bounds[1]))

    return BatchedTripletConfig(
        lens_path=str(qd["lens_path"]),
        object_depth_mm=float(qd.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd.get("wavelength_um", 0.587)),
        fitness_wavelengths_um=tuple(
            float(x) for x in qd.get("fitness_wavelengths_um", [0.587])
        ),
        theta_bounds=tuple(tuple(b) for b in qd.get("theta_bounds", theta_bounds_default)),
        stop_surface_index=int(qd.get("stop_surface_index", 4)),
        focal_length_target_mm=focal_length_target_mm,
        descriptor_zernike_indices=tuple(int(x) for x in qd.get("descriptor_zernike_indices", [7, 4])),
        archive_bins=tuple(int(x) for x in qd.get("archive_bins", [64, 64])),
        archive_bounds=archive_bounds,
        invalid_tol=float(qd.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd.get("invalid_penalty_weight", 5.0)),
        cma_eta=float(qd.get("cma_eta", 0.05)),
        seed_count=int(qd.get("seed_count", 128)),
        calibration_count=int(qd.get("calibration_count", 512)),
        num_rays=int(qd.get("num_rays", 256)),
        stage=str(qd.get("stage", "stage1")),
        zernike_field_index=int(qd.get("zernike_field_index", 0)),
        device=str(qd.get("device", "auto")),
        initial_sigma=float(qd.get("initial_sigma", 0.75)),
        population_size=int(qd.get("population_size", 16)),
        cma_seed=int(qd.get("seed", 0)),
        results_dir=str(qd.get("results_dir", "results/models")),
        run_name=str(qd.get("run_name", "triplet_batched")),
        log_every=int(qd.get("log_every", 50)),
        checkpoint_every=int(qd.get("checkpoint_every", 50)),
        new_cell_bonus=float(qd.get("new_cell_bonus", 1.0e6)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Batched GPU Cooke-triplet QD CMA-MEGA")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing run directory to resume from")
    args = parser.parse_args()

    if args.config is not None:
        cfg = _load_config(args.config)
    else:
        cfg = BatchedTripletConfig()

    yaml_iters = int(cfg.__dict__.get("iterations", 1000)) if hasattr(cfg, "__dict__") else 1000
    # Check if YAML config has an iterations key (not in dataclass, read directly)
    if args.config is not None:
        from optical_rl.utils.config import load_config
        _raw = load_config(args.config)
        _qd = _raw.get("qd", _raw)
        yaml_iters = int(_qd.get("iterations", 1000))
    iterations = args.iterations if args.iterations is not None else yaml_iters

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.results_dir) / f"{cfg.run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        import shutil
        shutil.copy(args.config, out_dir / Path(args.config).name)

    runner = BatchedTripletRunner(cfg)
    runner.result_dir = out_dir

    import time
    resumed = False
    if args.resume is not None:
        from optical_rl.qd._checkpoint import hash_config, load_checkpoint
        ckpt_path = Path(args.resume) / "checkpoint.pkl"
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path)
            runner.archive = ckpt.archive
            runner.emitter.set_state(ckpt.emitter_state)
            runner.completed = int(ckpt.completed_evaluations)
            runner.generation = int(ckpt.generation)
            runner.restart_count = int(ckpt.restart_count)
            runner.wallclock_seconds = float(ckpt.wallclock_seconds)
            current_hash = hash_config(cfg)
            if ckpt.config_hash != current_hash:
                print(
                    f"[triplet-batched] WARNING: config hash drift on resume "
                    f"(checkpoint={ckpt.config_hash[:12]}, "
                    f"current={current_hash[:12]}). Continuing anyway.",
                    flush=True,
                )
            print(
                f"[triplet-batched] Resumed from {ckpt_path}: "
                f"completed={runner.completed} gen={runner.generation} "
                f"restarts={runner.restart_count} "
                f"best={runner.archive.stats()['best_fitness']:.6f}",
                flush=True,
            )
            resumed = True
        else:
            print(f"[triplet-batched] No checkpoint.pkl in {args.resume}, starting fresh.",
                  flush=True)

    t0 = time.perf_counter()
    if not resumed:
        print(f"[triplet-batched] Seeding archive ...", flush=True)
        seed_stats = runner.seed_archive()
        t_seed = time.perf_counter() - t0
        print(f"[triplet-batched] Seed stats: {seed_stats} ({t_seed:.2f}s)", flush=True)
    else:
        t_seed = 0.0

    print(f"[triplet-batched] Running {iterations} evaluations ...", flush=True)
    t0 = time.perf_counter()
    final_stats = runner.run(iterations, resume=resumed)
    t_run = time.perf_counter() - t0
    print(f"[triplet-batched] Final stats: {final_stats} ({t_run:.2f}s)", flush=True)

    # Save archive
    cells_data = []
    for cell_idx, result in runner.archive.cells.items():
        cells_data.append({
            "cell": list(cell_idx),
            "theta": result.theta.tolist(),
            "fitness": float(result.fitness),
            "measures": list(result.measures),
            "metrics": result.metrics,
        })
    with open(out_dir / "archive.json", "w") as f:
        json.dump(cells_data, f, indent=2)

    # Archive plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        j0, j1 = cfg.descriptor_zernike_indices
        label0 = f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)"
        label1 = f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)"

        bins_x, bins_y = runner.archive.bins
        neg_grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
        xs, ys, neg_cs = [], [], []
        for (i, j), result in runner.archive.cells.items():
            f = float(result.fitness)
            neg_grid[i, j] = -f
            xs.append(float(result.measures[0]))
            ys.append(float(result.measures[1]))
            neg_cs.append(-f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        valid_grid = neg_grid[np.isfinite(neg_grid) & (neg_grid > 0)]
        norm = LogNorm(vmin=float(valid_grid.min()), vmax=float(valid_grid.max())) if valid_grid.size > 0 else None
        im = axes[0].imshow(neg_grid.T, origin="lower", aspect="auto", norm=norm)
        axes[0].set_title("Archive Fitness Heatmap")
        fig.colorbar(im, ax=axes[0], shrink=0.85).set_label("-fitness (log)")

        if xs:
            pos_cs = [v for v in neg_cs if v > 0]
            sn = LogNorm(vmin=min(pos_cs), vmax=max(pos_cs)) if pos_cs else None
            sc = axes[1].scatter(xs, ys, c=neg_cs, s=24, norm=sn)
            (x_lo, x_hi), (y_lo, y_hi) = runner.archive.bounds
            axes[1].set_xlim(x_lo, x_hi)
            axes[1].set_ylim(y_lo, y_hi)
            fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("-fitness (log)")
        axes[1].set_title("Archive (Batched Triplet)")
        axes[1].set_xlabel(label0)
        axes[1].set_ylabel(label1)
        fig.tight_layout()
        fig.savefig(out_dir / "archive_map.png", dpi=120)
        plt.close(fig)
        print(f"[triplet-batched] Archive plot saved", flush=True)
    except Exception as e:
        print(f"[triplet-batched] Plot failed: {e}", flush=True)

    print(f"[triplet-batched] Saved to {out_dir}", flush=True)
    print(f"[triplet-batched] Seed time: {t_seed:.2f}s, Run time: {t_run:.2f}s", flush=True)


if __name__ == "__main__":
    main()
