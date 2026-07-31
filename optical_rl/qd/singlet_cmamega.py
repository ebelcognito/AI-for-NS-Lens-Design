from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .singlet_zernike_measures import (
    NOLL_N_TERMS,
    NOLL_TERM_NAMES,
    build_zernike_pinv,
)


# Descriptor2D: two Zernike coefficients selected by descriptor_zernike_indices.
Descriptor2D = Tuple[float, float]
BoundsND = Tuple[Tuple[float, float], ...]


def project_box(theta: np.ndarray, bounds: BoundsND) -> np.ndarray:
    dim = len(bounds)
    arr = np.asarray(theta, dtype=np.float64).reshape(dim)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(arr, lo, hi)


def theta_to_unit(theta: np.ndarray, bounds: BoundsND) -> np.ndarray:
    theta = project_box(theta, bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (theta - center) / np.where(half > 0.0, half, 1.0)


def unit_to_theta(unit: np.ndarray, bounds: BoundsND) -> np.ndarray:
    dim = len(bounds)
    u = np.clip(np.asarray(unit, dtype=np.float64).reshape(dim), -1.0, 1.0)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return project_box(center + u * half, bounds)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deeplens_root() -> Path:
    return _repo_root() / "DeepLens"


def _ensure_deeplens_path() -> None:
    root = str(_deeplens_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    out = np.zeros((n, dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / float(n)
    return out


def _disk_samples(num_rays: int) -> np.ndarray:
    if num_rays <= 0:
        raise ValueError(f"num_rays must be > 0, got {num_rays}")
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(num_rays))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class SingletCMAMEGAConfig:
    lens_path: str
    object_depth_mm: float = -1000.0
    wavelength_um: float = 0.587
    # 1D theta: [c1] only — c2 and d_sensor are derived via paraxial solve.
    theta_bounds: BoundsND = ((-0.1, 0.1),)
    # None → use lens.foclen loaded from the JSON.
    focal_length_target_mm: Optional[float] = None
    # Two 1-based Noll indices to use as archive descriptors.
    # Any pair from j=1..16; see singlet_zernike_measures.NOLL_TERM_NAMES.
    # Examples: (4,11) defocus+spherical, (11,16) spherical+secondary-coma,
    #           (7,11) vertical-coma+spherical, (5,11) oblique-astigmatism+spherical
    descriptor_zernike_indices: Tuple[int, int] = (4, 11)
    archive_bins: Tuple[int, int] = (64, 64)
    archive_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    invalid_tol: float = 0.05
    invalid_penalty_weight: float = 5.0
    cma_eta: float = 0.05
    seed_count: int = 128
    calibration_count: int = 512
    num_rays: int = 256
    alpha_deg: float = 0.5  # retained for config compatibility; not used in evaluation
    stage: str = "stage1"
    stage2_field_deg: float = 2.0
    # Field angle (degrees) used for the Zernike OPD measurement.
    # 0.0 = on-axis (default, backward-compatible); set e.g. 2.0 for off-axis coma/astigmatism.
    zernike_field_deg: float = 0.0
    device: str = "auto"
    initial_sigma: float = 0.75
    population_size: int = 16
    cma_seed: int = 0
    results_dir: str = "results/models"
    run_name: str = "singlet_cmamega"
    log_every: int = 50


@dataclass(frozen=True)
class SingletEvalResult:
    theta: np.ndarray
    fitness: float
    measures: Descriptor2D
    metrics: Dict[str, float]
    grads: Optional[Dict[str, np.ndarray]] = None


class CMAMEGAArchive2D:
    def __init__(
        self,
        bins: Tuple[int, int],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        *,
        seed: int = 0,
    ):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bounds = bounds
        self._rng = np.random.default_rng(int(seed))
        self._cells: Dict[Tuple[int, int], SingletEvalResult] = {}
        self._best_fitness = -float("inf")

    def _to_index(self, measures: Descriptor2D) -> Optional[Tuple[int, int]]:
        (x_lo, x_hi), (y_lo, y_hi) = self.bounds
        x, y = float(measures[0]), float(measures[1])
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return None

        def to_bin(value: float, lo: float, hi: float, n: int) -> int:
            if hi <= lo:
                return 0
            frac = (value - lo) / (hi - lo)
            return int(np.clip(np.floor(frac * n), 0, n - 1))

        return (
            to_bin(x, x_lo, x_hi, self.bins[0]),
            to_bin(y, y_lo, y_hi, self.bins[1]),
        )

    def insert(self, result: SingletEvalResult) -> bool:
        idx = self._to_index(result.measures)
        if idx is None:
            return False
        current = self._cells.get(idx)
        self._best_fitness = max(self._best_fitness, float(result.fitness))
        if current is None or float(result.fitness) > float(current.fitness):
            self._cells[idx] = result
            return True
        return False

    def sample_parent(self, seed: int | None = None) -> np.ndarray:
        if not self._cells:
            raise RuntimeError("archive is empty")
        rng = self._rng if seed is None else np.random.default_rng(int(seed))
        keys = list(self._cells.keys())
        idx = keys[int(rng.integers(0, len(keys)))]
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
    def cells(self) -> Dict[Tuple[int, int], SingletEvalResult]:
        return dict(self._cells)


# GradientCoefficientCMA is now the canonical CMA-ES engine implemented in
# optical_rl/qd/_cma.py. This module re-exports it for backwards compatibility
# with existing imports (`from optical_rl.qd.singlet_cmamega import
# GradientCoefficientCMA`). New code should import from optical_rl.qd._cma.
from ._cma import GradientCoefficientCMA  # noqa: F401


class FixedSingletProblem:
    def __init__(self, config: SingletCMAMEGAConfig):
        self.config = config
        self._backend_loaded = False
        self._torch = None
        self._GeoLens = None
        self._Ray = None
        self._device = None
        self._template_lens = None
        self._aperture_xy = None
        # Paraxial solve constants (set in _ensure_backend)
        self._n_glass: float = 1.0
        self._t_glass: float = 0.0
        self._z_last: float = 0.0
        self._f_target: float = 0.0
        # Precomputed Zernike pseudo-inverse (j_max × N), as a torch tensor
        self._pinv_Z = None

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return

        _ensure_deeplens_path()
        import torch
        from deeplens.optics import GeoLens
        from deeplens.optics.light import Ray

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
        self._GeoLens = GeoLens
        self._Ray = Ray
        self._device = device
        self._template_lens = self._load_template_lens()

        # Aperture samples scaled to the stop radius (surfaces[0].r).
        disk_unit = _disk_samples(int(self.config.num_rays))
        aperture = disk_unit * float(self._template_lens.surfaces[0].r)
        self._aperture_xy = torch.tensor(aperture, dtype=torch.float32, device=self._device)

        # Precompute Zernike pseudo-inverse — only compute up to the highest
        # requested index so we don't waste work on unused terms.
        j_max_needed = max(self.config.descriptor_zernike_indices)
        if j_max_needed > NOLL_N_TERMS:
            raise ValueError(
                f"descriptor_zernike_indices requests j={j_max_needed} but "
                f"only j=1..{NOLL_N_TERMS} are implemented."
            )
        pinv_np = build_zernike_pinv(disk_unit, j_max=j_max_needed)  # (j_max, N)
        self._pinv_Z = torch.tensor(pinv_np, dtype=torch.float32, device=self._device)

        self._backend_loaded = True

    def _load_template_lens(self):
        """Load the template lens, extract paraxial constants, then zero curvatures."""
        torch = self._torch
        lens = self._GeoLens(filename=self.config.lens_path, device=self._device)

        # Read focal length before zeroing (GeoLens computes it from the JSON curvatures).
        if self.config.focal_length_target_mm is not None:
            self._f_target = float(self.config.focal_length_target_mm)
        else:
            self._f_target = float(lens.foclen)

        # surfaces[0]: Stop, surfaces[1]: front glass, surfaces[2]: rear glass.
        wvln = float(self.config.wavelength_um)
        self._n_glass = float(lens.surfaces[1].mat2.refractive_index(wvln))
        self._t_glass = float(lens.surfaces[2].d.item() - lens.surfaces[1].d.item())
        self._z_last = float(lens.surfaces[-1].d.item())

        # Zero curvatures so the template is a neutral (plano) baseline.
        lens.surfaces[1].c = torch.tensor(0.0, dtype=lens.surfaces[1].c.dtype, device=self._device)
        lens.surfaces[2].c = torch.tensor(0.0, dtype=lens.surfaces[2].c.dtype, device=self._device)
        lens.d_sensor = torch.tensor(float(lens.d_sensor.item()), device=self._device)
        return lens

    def _paraxial_solve(self, c1_t):
        """Derive c2 and d_sensor from c1 to enforce focal length = f_target.

        Uses the thick-lens paraxial (ABCD) relations for a singlet in air:
            phi_1 = (n-1)*c1
            phi_2 = (P - phi_1) / (1 - t*phi_1/n)   where P = 1/f_target
            c2    = -phi_2 / (n-1)
            BFL   = (1 - t*phi_1/n) / P
            d_sensor = z_last + BFL

        Returns:
            c2 (tensor), d_sensor (tensor) — differentiable w.r.t. c1_t.
        """
        n = self._n_glass
        t = self._t_glass
        P = 1.0 / self._f_target

        phi_1 = (n - 1.0) * c1_t
        raw_denom = 1.0 - t * phi_1 / n
        # Guard: clamp denominator away from zero (singularity near c1 ~ n/(t*(n-1))).
        safe_denom = raw_denom.clamp(min=1e-4) if raw_denom > 0 else raw_denom.clamp(max=-1e-4)
        phi_2 = (P - phi_1) / safe_denom
        c2 = -phi_2 / (n - 1.0)
        bfl = safe_denom / P
        d_sensor = self._z_last + bfl
        return c2, d_sensor

    def _build_lens_from_theta_tensor(self, theta_t):
        """Build a GeoLens from theta_t = [c1], deriving c2 and d_sensor via paraxial solve."""
        c1 = theta_t[0]
        c2, d_sensor = self._paraxial_solve(c1)
        lens = copy.deepcopy(self._template_lens)
        dtype = lens.surfaces[1].c.dtype
        lens.surfaces[1].c = c1.to(dtype=dtype)
        lens.surfaces[2].c = c2.to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens

    def theta_to_lens(self, theta: np.ndarray):
        """Public API: build a GeoLens from a 1D numpy theta = [c1]."""
        self._ensure_backend()
        torch = self._torch
        theta = self.project_theta(theta)
        c1_t = torch.tensor(float(theta[0]), dtype=torch.float32, device=self._device)
        c2_t, d_sensor_t = self._paraxial_solve(c1_t)
        lens = copy.deepcopy(self._template_lens)
        dtype = lens.surfaces[1].c.dtype
        lens.surfaces[1].c = c1_t.to(dtype=dtype)
        lens.surfaces[2].c = c2_t.to(dtype=dtype)
        lens.d_sensor = d_sensor_t.to(dtype=dtype)
        return lens

    def initial_theta(self) -> np.ndarray:
        self._ensure_backend()
        return np.zeros(1, dtype=np.float64)

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        dim = len(self.config.theta_bounds)
        unit = _latin_hypercube(int(n), dim, int(seed))
        lo = np.array([b[0] for b in self.config.theta_bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self.config.theta_bounds], dtype=np.float64)
        return lo + unit * (hi - lo)

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        return project_box(theta, self.config.theta_bounds)

    def _make_object_point(self, field_x_deg: float, field_y_deg: float):
        depth = float(self.config.object_depth_mm)
        x = depth * math.tan(math.radians(field_x_deg))
        y = depth * math.tan(math.radians(field_y_deg))
        return x, y, depth

    def _trace_field(self, lens, field_x_deg: float, field_y_deg: float):
        torch = self._torch
        x_obj, y_obj, z_obj = self._make_object_point(field_x_deg, field_y_deg)
        ray_count = self._aperture_xy.shape[0]
        dtype = lens.surfaces[1].c.dtype
        origins = torch.empty((ray_count, 3), dtype=dtype, device=self._device)
        origins[:, 0] = x_obj
        origins[:, 1] = y_obj
        origins[:, 2] = z_obj

        targets = torch.empty((ray_count, 3), dtype=dtype, device=self._device)
        targets[:, :2] = self._aperture_xy.to(dtype=dtype)
        targets[:, 2] = float(lens.surfaces[0].d.item())
        ray = self._Ray(origins, targets - origins, wvln=float(self.config.wavelength_um), device=self._device)
        return lens.trace2sensor(ray)

    def _trace_coherent(self, lens, field_x_deg: float = 0.0, field_y_deg: float = 0.0):
        """Trace a coherent ray bundle from the stop plane for OPL accumulation.

        Rays start at (x_p, y_p, z_stop) — on the stop plane — directed toward the
        object point implied by (field_x_deg, field_y_deg).  Starting at the stop
        avoids the >100mm propagation that would trigger the float32 OPL precision
        warning in DeepLens; the missing incoming OPL is restored in _zernike_measures.

        field_x_deg = field_y_deg = 0.0 reproduces the former on-axis behaviour.

        Float64 is required by DeepLens' coherent Ray.prop_to().
        """
        torch = self._torch
        ray_count = self._aperture_xy.shape[0]
        dtype = torch.float64

        z_stop = float(lens.surfaces[0].d.item())
        z_obj = float(self.config.object_depth_mm)
        x_obj = z_obj * math.tan(math.radians(field_x_deg))
        y_obj = z_obj * math.tan(math.radians(field_y_deg))

        origins = torch.zeros((ray_count, 3), dtype=dtype, device=self._device)
        origins[:, :2] = self._aperture_xy.to(dtype=dtype)
        origins[:, 2] = z_stop

        obj = torch.tensor([x_obj, y_obj, z_obj], dtype=dtype, device=self._device)
        d = torch.nn.functional.normalize(origins - obj.unsqueeze(0), dim=-1)

        ray = self._Ray(origins, d, wvln=float(self.config.wavelength_um),
                        coherent=True, device=self._device)
        return lens.trace2sensor(ray)

    def _field_points(self) -> Tuple[Tuple[float, float], ...]:
        if str(self.config.stage).lower().strip() == "stage2":
            f = float(self.config.stage2_field_deg)
            return ((0.0, 0.0), (f, 0.0), (-f, 0.0), (0.0, f), (0.0, -f))
        return ((0.0, 0.0),)

    def _rms_spot(self, ray):
        torch = self._torch
        valid = ray.is_valid
        xy = ray.o[..., :2]
        denom = valid.sum().clamp_min(1e-8)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
        err2 = ((xy - centroid) ** 2).sum(dim=-1)
        rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
        return rms, 1.0 - (valid.sum() / max(float(valid.numel()), 1.0))

    def _zernike_measures(self, lens):
        """Fit Noll Zernike coefficients to the OPD at the exit pupil.

        The field angle is controlled by config.zernike_field_deg (default 0 = on-axis).
        At 0° the computation is identical to the original on-axis pipeline.  At any
        non-zero angle an off-axis coherent trace is used so that coma, astigmatism,
        and other non-rotationally-symmetric terms become non-trivial.

        The off-axis bundle is traced in the meridional (Y) plane:
            y_obj = z_obj * tan(zernike_field_deg)   x_obj = 0

        Pipeline:
          1. Trace marginal rays coherently to sensor, then prop_to exit pupil.
          2. Restore missing incoming OPL: differential path from object to each stop
             sample minus the chief-ray incoming path.  At 0° this reduces to
             sqrt(x²+y²+z_obj²) - |z_obj| (original formula).
          3. Trace chief ray at the configured field angle (meridional plane).
             calc_chief_ray_infinite takes degrees; pass [0, field] and take index [-1]
             so the function fills the correct [1:] slice.
          4. Reference sphere centred on chief ray's exit-pupil position:
               ref_i = sqrt((x_ep_i-x_c)² + (y_ep_i-y_c)² + BFL²) - BFL
             At 0° x_c = y_c = 0, recovering the original formula.
          5. OPD_i = (OPL_i - OPL_chief) + ref_i,  zeroed for invalid rays.
          6. Linear Zernike fit:  coeffs = pinv_Z @ opd.

        Returns full Zernike coefficient vector (j_max,); caller selects indices.
        """
        torch = self._torch

        field = float(self.config.zernike_field_deg)
        z_obj_val = float(self.config.object_depth_mm)
        z_obj_abs = abs(z_obj_val)
        # Off-axis object in the meridional (Y) plane; x_obj = 0 throughout.
        y_obj = z_obj_val * math.tan(math.radians(field))

        # --- 1. Marginal rays: coherent trace → sensor → exit pupil ---
        ray = self._trace_coherent(lens, field_x_deg=0.0, field_y_deg=field)
        pupil_z, _ = lens.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        opl   = ray.opl.squeeze(-1)   # (N,)
        valid = ray.is_valid.float()  # (N,)
        x_ep  = ray.o[:, 0]          # (N,)
        y_ep  = ray.o[:, 1]          # (N,)

        # --- 2. Restore incoming OPL (object → stop) missing because rays start at stop ---
        # Differential correction: marginal path - chief path (both in air, n=1).
        # At field=0: y_obj=0 → opl_chief_in = z_obj_abs → reduces to original formula.
        xy_stop = self._aperture_xy.to(torch.float64)
        xp = xy_stop[:, 0]
        yp = xy_stop[:, 1]
        opl_marginal_in = torch.sqrt(xp ** 2 + (yp - y_obj) ** 2 + z_obj_abs ** 2)
        opl_chief_in = math.sqrt(y_obj ** 2 + z_obj_abs ** 2)
        opl_incoming = opl_marginal_in - opl_chief_in   # (N,), relative to chief
        opl = opl + opl_incoming

        # --- 3. Chief ray at configured field angle (degrees, meridional plane) ---
        # calc_chief_ray_infinite fills output[1:] for non-zero entries, so pass
        # [0, field] and take the last row to always get a single (1,3) tensor.
        if field == 0.0:
            chief_rfov = torch.tensor([0.0], device=self._device)
        else:
            chief_rfov = torch.tensor([0.0, field], device=self._device)
        chief_o, chief_d = lens.calc_chief_ray_infinite(
            chief_rfov, plane="meridional"
        )
        chief_ray = self._Ray(
            chief_o[-1:].to(torch.float64),
            chief_d[-1:].to(torch.float64),
            wvln=float(self.config.wavelength_um),
            coherent=True,
            device=self._device,
        )
        chief_ray = lens.trace2sensor(chief_ray)
        chief_ray = chief_ray.prop_to(pupil_z)
        opl_chief = chief_ray.opl.squeeze()          # scalar
        x_chief_ep = chief_ray.o[0, 0].to(torch.float64)  # chief exit-pupil x
        y_chief_ep = chief_ray.o[0, 1].to(torch.float64)  # chief exit-pupil y

        # --- 4. Reference sphere centred on chief ray's exit-pupil position ---
        # At field=0: x_chief_ep ≈ y_chief_ep ≈ 0 → reduces to original formula.
        BFL = lens.d_sensor - pupil_z   # gradient flows through d_sensor
        ref_sphere = torch.sqrt(
            (x_ep - x_chief_ep) ** 2 + (y_ep - y_chief_ep) ** 2 + BFL ** 2
        ) - BFL

        # --- 5. OPD at exit pupil, invalid rays zeroed ---
        # Sign convention identical to original: + ref_sphere gives W=0 for perfect lens.
        opd = ((opl - opl_chief) + ref_sphere) * valid

        # --- 6. Linear Zernike fit: pinv_Z (j_max, N) @ opd (N,) → (j_max,) ---
        return self._pinv_Z @ opd.to(self._pinv_Z.dtype)

    def _fitness_and_metrics(self, theta_t):
        torch = self._torch
        lens = self._build_lens_from_theta_tensor(theta_t)

        # Fitness: negative mean RMS spot over field points.
        rms_terms = []
        invalid_fracs = []
        for fx, fy in self._field_points():
            ray = self._trace_field(lens, fx, fy)
            rms, invalid_fraction = self._rms_spot(ray)
            rms_terms.append(rms)
            invalid_fracs.append(invalid_fraction)

        rms_mean = torch.stack(rms_terms).mean()
        invalid_fraction_mean = torch.stack(invalid_fracs).mean()
        fitness = -rms_mean

        # Behavioral descriptors: select the two configured Noll Zernike indices.
        # coeffs[j-1] is the coefficient for Noll term j (1-based).
        coeffs = self._zernike_measures(lens)
        j0, j1 = self.config.descriptor_zernike_indices
        m0, m1 = coeffs[j0 - 1], coeffs[j1 - 1]

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(invalid_fraction_mean.detach().cpu().item()),
            f"z{j0}_{NOLL_TERM_NAMES.get(j0, 'unknown')}_mm": float(m0.detach().cpu().item()),
            f"z{j1}_{NOLL_TERM_NAMES.get(j1, 'unknown')}_mm": float(m1.detach().cpu().item()),
        }
        return fitness, m0, m1, metrics

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> SingletEvalResult:
        self._ensure_backend()
        torch = self._torch
        theta_proj = self.project_theta(theta)
        theta_t = torch.tensor(theta_proj, dtype=torch.float32, device=self._device, requires_grad=with_grads)
        fitness, m0, m1, metrics = self._fitness_and_metrics(theta_t)

        j0, j1 = self.config.descriptor_zernike_indices
        k0, k1 = f"z{j0}", f"z{j1}"

        grads = None
        if with_grads:
            g_f  = torch.autograd.grad(fitness, theta_t, retain_graph=True,  allow_unused=False)[0]
            g_m0 = torch.autograd.grad(m0,      theta_t, retain_graph=True,  allow_unused=False)[0]
            g_m1 = torch.autograd.grad(m1,      theta_t, retain_graph=False, allow_unused=False)[0]
            grads = {
                "fitness": g_f.detach().cpu().numpy().astype(np.float64, copy=False),
                k0: g_m0.detach().cpu().numpy().astype(np.float64, copy=False),
                k1: g_m1.detach().cpu().numpy().astype(np.float64, copy=False),
            }

        return SingletEvalResult(
            theta=theta_proj.astype(np.float64, copy=False),
            fitness=float(fitness.detach().cpu().item()),
            measures=(float(m0.detach().cpu().item()), float(m1.detach().cpu().item())),
            metrics=metrics,
            grads=grads,
        )


class FixedSingletCMAMEGARunner:
    def __init__(self, config: SingletCMAMEGAConfig):
        self.config = config
        self.problem = FixedSingletProblem(config)
        self.archive: Optional[CMAMEGAArchive2D] = None
        # dim = 1 (fitness) + 2 (one per descriptor).
        self.emitter = GradientCoefficientCMA(
            dim=1 + len(config.descriptor_zernike_indices),
            seed=int(config.cma_seed),
            sigma=float(config.initial_sigma),
        )

    def calibrate_archive_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Sample the design space and fit 2-percentile bounds for both Zernike measures."""
        samples = self.problem.sample_theta(self.config.calibration_count, self.config.cma_seed + 17)
        m0_vals: list[float] = []
        m1_vals: list[float] = []
        for theta in samples:
            result = self.problem.evaluate(theta, with_grads=False)
            m0_vals.append(result.measures[0])
            m1_vals.append(result.measures[1])

        def _percentile_bounds(arr: np.ndarray) -> Tuple[float, float]:
            lo = float(np.percentile(arr, 2.0))
            hi = float(np.percentile(arr, 98.0))
            span = max(hi - lo, 1e-6)
            return (lo - 0.05 * span, hi + 0.05 * span)

        bounds = (
            _percentile_bounds(np.asarray(m0_vals, dtype=np.float64)),
            _percentile_bounds(np.asarray(m1_vals, dtype=np.float64)),
        )
        self.archive = CMAMEGAArchive2D(self.config.archive_bins, bounds, seed=self.config.cma_seed)
        return bounds

    def seed_archive(self) -> Dict[str, float]:
        if self.archive is None:
            if self.config.archive_bounds is not None:
                self.archive = CMAMEGAArchive2D(
                    self.config.archive_bins,
                    self.config.archive_bounds,
                    seed=self.config.cma_seed,
                )
            else:
                self.calibrate_archive_bounds()

        flat = self.problem.initial_theta()
        self.archive.insert(self.problem.evaluate(flat, with_grads=False))
        seed_thetas = self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31)
        for theta in seed_thetas:
            self.archive.insert(self.problem.evaluate(theta, with_grads=False))
        return self.archive.stats()

    def _normalized_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        half = 0.5 * np.array([hi - lo for lo, hi in self.config.theta_bounds], dtype=np.float64)

        def norm_grad(name: str) -> np.ndarray:
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        return {"fitness": norm_grad("fitness"), "z4": norm_grad("z4"), "z11": norm_grad("z11")}

    def run(self, iterations: int) -> Dict[str, float]:
        if self.archive is None or not self.archive.cells:
            self.seed_archive()

        theta_dim = len(self.config.theta_bounds)
        scores = []
        eval_budget = int(iterations)
        completed = 0
        generation = 0
        while completed < eval_budget:
            parent_theta = self.archive.sample_parent()
            parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
            assert parent_eval.grads is not None

            unit_parent = theta_to_unit(parent_theta, self.config.theta_bounds)
            grads = self._normalized_gradients(parent_eval.grads)
            batch_n = min(int(self.config.population_size), eval_budget - completed)
            coeffs = self.emitter.ask(batch_n)
            batch_scores = []
            batch_improved = []
            for local_idx, coeff in enumerate(coeffs):
                direction = (
                    abs(coeff[0]) * grads["fitness"]
                    + coeff[1] * grads["z4"]
                    + coeff[2] * grads["z11"]
                )
                dir_norm = np.linalg.norm(direction)
                if not np.isfinite(dir_norm) or dir_norm < 1e-12:
                    rng = np.random.default_rng(self.config.cma_seed + completed + local_idx)
                    direction = rng.normal(0.0, 1.0, size=(theta_dim,))
                    direction /= np.linalg.norm(direction) + 1e-12

                candidate_unit = np.clip(unit_parent + float(self.config.cma_eta) * direction, -1.0, 1.0)
                candidate_theta = unit_to_theta(candidate_unit, self.config.theta_bounds)
                candidate_eval = self.problem.evaluate(candidate_theta, with_grads=False)
                improved = self.archive.insert(candidate_eval)
                scores.append(candidate_eval.fitness)
                batch_scores.append(candidate_eval.fitness if improved else -np.inf)
                batch_improved.append(improved)

            self.emitter.tell(coeffs, np.asarray(batch_scores, dtype=np.float64))
            completed += batch_n
            generation += 1

            if generation % max(int(self.config.log_every), 1) == 0:
                stats = self.archive.stats()

                n_improved = sum(batch_improved)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted_elites = sum(1 for i in elite_idx if not batch_improved[i])
                print(
                    f"[singlet-cmamega] gen={generation} evals={completed} "
                    f"coverage={stats['coverage']:.4f} best={stats['best_fitness']:.6f} | "
                    f"archive_improved={n_improved}/{batch_n} "
                    f"wasted_elites={n_wasted_elites}/{elite_n} "
                    f"sigma={self.emitter.sigma:.4f}",
                    flush=True,
                )

        stats = self.archive.stats()
        stats["iterations"] = float(iterations)
        stats["generations"] = float(generation)
        stats["mean_score"] = float(np.mean(scores)) if scores else float("nan")
        return stats
