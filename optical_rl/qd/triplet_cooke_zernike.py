"""Cooke triplet CMA-MEGA with Zernike behavioral descriptors.

theta = [c0, c1, c2, c3, c5, c6]  (6 surface curvatures; indices match the JSON
surface list where surface 4 is the aperture stop and is never touched)

d_sensor is derived from a paraxial ABCD back-focal-length solve performed in
PyTorch so that autograd gradients flow through it to all six curvatures.

Design philosophy mirrors singlet_cmamega_two_param.py exactly:
  - Auto d_sensor via differentiable paraxial BFL (no free d_sensor parameter)
  - Aperture sampling on stop aperture disk (r = stop surface radius from JSON)
  - Zernike OPD pipeline identical to the singlet (coherent trace from first
    surface, incoming OPL correction, chief ray, reference sphere, pinv fit)
  - stage1 / stage2 fitness (on-axis RMS vs mean over 5 field points)
  - Zernike field angle is independent of fitness field angle
  - Diverging-lens rejection (BFL <= 0 → heavy penalty)
"""
from __future__ import annotations

import copy
import json
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
# Reuse Zernike helpers from singlet module
# ---------------------------------------------------------------------------

from optical_rl.qd.singlet_zernike_measures import (  # noqa: E402
    NOLL_N_TERMS,
    NOLL_TERM_NAMES,
    build_zernike_pinv,
)


# ---------------------------------------------------------------------------
# Helpers (local copies so this file has zero extra imports)
# ---------------------------------------------------------------------------

def _disk_samples(num_rays: int) -> np.ndarray:
    """Fibonacci spiral on the unit disk — identical to singlet implementation."""
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(max(num_rays, 1)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((max(n, 0), dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / float(n)
    return out


def _project_box(theta: np.ndarray, bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(np.asarray(theta, dtype=np.float64).reshape(-1), lo, hi)


def _theta_to_unit(theta: np.ndarray, bounds) -> np.ndarray:
    theta = _project_box(theta, bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (theta - center) / np.where(half > 0.0, half, 1.0)


def _unit_to_theta(unit: np.ndarray, bounds) -> np.ndarray:
    u = np.clip(np.asarray(unit, dtype=np.float64).reshape(-1), -1.0, 1.0)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return _project_box(center + u * half, bounds)


# ---------------------------------------------------------------------------
# EvalResult (matches singlet shape exactly)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc, field as _field  # noqa: E402


@_dc
class TripletEvalResult:
    theta: np.ndarray
    fitness: float
    measures: Tuple[float, float]
    metrics: Dict[str, float]
    grads: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TripletCookeZernikeConfig:
    lens_path: str = str(_PROJECT_ROOT / "DeepLens/datasets/lenses/cooke_smaller_ap_1mm.json")
    object_depth_mm: float = -1000.0
    wavelength_um: float = 0.587

    # Wavelengths for polychromatic RMS spot evaluation.
    # Single entry = monochromatic (backward compatible).
    # (0.656, 0.587, 0.486) = Fraunhofer C, d, F lines (R, G, B).
    fitness_wavelengths_um: Tuple[float, ...] = (0.656, 0.587, 0.486)

    # theta = [c0, c1, c2, c3, c5, c6] — all surface curvatures, ±0.06
    theta_bounds: Tuple = (
        (-0.06, 0.06),  # c0
        (-0.06, 0.06),  # c1
        (-0.06, 0.06),  # c2
        (-0.06, 0.06),  # c3
        (-0.06, 0.06),  # c5
        (-0.06, 0.06),  # c6
    )

    # Index of the aperture stop surface in the JSON surface list.
    stop_surface_index: int = 4

    # When set, the last refractive surface curvature (c6) is derived from the
    # paraxial ABCD solve to enforce EFL = focal_length_target_mm.  theta becomes
    # 5-D: [c0, c1, c2, c3, c5] — theta_bounds must also have 5 entries.
    # When None the original 6-D free behaviour is preserved.
    focal_length_target_mm: Optional[float] = None

    # Noll Zernike indices used as archive behavioral descriptors (1-based).
    descriptor_zernike_indices: Tuple[int, int] = (7, 4)

    archive_bins: Tuple[int, int] = (64, 64)
    archive_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    invalid_tol: float = 0.05
    invalid_penalty_weight: float = 5.0

    cma_eta: float = 0.05
    seed_count: int = 128
    calibration_count: int = 512
    num_rays: int = 256

    # Fitness stage: "stage1" = on-axis RMS only;
    # "stage2" = mean over 3 meridional field pts (0, 0.5, 1.0 × HFOV).
    stage: str = "stage1"

    # Which of the 3 hfov3 field positions to use for Zernike OPD measurement:
    #   0 = on-axis (0.0°), 1 = mid-field (0.5 × HFOV), 2 = full-field (1.0 × HFOV)
    zernike_field_index: int = 0

    device: str = "auto"
    initial_sigma: float = 0.75
    population_size: int = 16
    cma_seed: int = 0
    results_dir: str = "results/models"
    run_name: str = "triplet_cooke_zernike"
    log_every: int = 50

    # Stop-and-go checkpointing — see _checkpoint.py for full doc.
    checkpoint_every: int = 50

    # Two-tier improvement-rank bonus for new cells.
    new_cell_bonus: float = 1.0e6


# ---------------------------------------------------------------------------
# Archive (identical structure to CMAMEGAArchive2D in singlet_cmamega)
# ---------------------------------------------------------------------------

class TripletArchive2D:
    def __init__(self, bins, bounds, *, seed: int = 0):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bounds = bounds
        self._rng = np.random.default_rng(int(seed))
        self._cells: Dict[Tuple[int, int], TripletEvalResult] = {}
        self._best_fitness = -float("inf")

    def _to_index(self, measures):
        (x_lo, x_hi), (y_lo, y_hi) = self.bounds
        x, y = float(measures[0]), float(measures[1])
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return None
        def _bin(v, lo, hi, n):
            return int(np.clip(np.floor((v - lo) / (hi - lo) * n), 0, n - 1))
        return (_bin(x, x_lo, x_hi, self.bins[0]), _bin(y, y_lo, y_hi, self.bins[1]))

    def insert(self, result: TripletEvalResult) -> int:
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
# GradientCoefficientCMA (re-import from singlet module)
# ---------------------------------------------------------------------------

from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA  # noqa: E402


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class TripletCookeZernikeProblem:
    """Cooke-triplet CMA-MEGA problem with Zernike behavioral descriptors.

    Surfaces in the JSON (cooke_smaller_ap_1mm.json):
      0  Spheric  n-sk16  front element 1
      1  Spheric  air     rear  element 1
      2  Spheric  f2      front element 2
      3  Spheric  air     rear  element 2
      4  Stop     ---     aperture stop  (r = 1 mm)
      5  Spheric  n-sk16  front element 3
      6  Spheric  air     rear  element 3

    theta = [c0, c1, c2, c3, c5, c6]  — all ±0.06 mm⁻¹
    d_sensor derived via paraxial ABCD BFL solve (differentiable).
    """

    # Surface indices of the six refractive surfaces
    _SURF_IDX = (0, 1, 2, 3, 5, 6)

    def __init__(self, config: TripletCookeZernikeConfig):
        self.config = config
        self._backend_loaded = False
        self._torch = None
        self._GeoLens = None
        self._Ray = None
        self._device = None
        self._template_lens = None
        self._aperture_xy = None   # (N,2) stop-plane positions in mm
        self._pinv_Z = None        # (j_max, N) Zernike pseudo-inverse tensor
        # Paraxial constants extracted from the lens JSON (set in _load_template_lens)
        self._n1 = None   # refractive index of element 1 glass at wavelength
        self._n2 = None   # refractive index of element 2 glass at wavelength
        self._n3 = None   # refractive index of element 3 glass at wavelength
        self._d01 = None  # glass thickness  element 1  (surface 0→1)
        self._d12 = None  # air gap          element 1→2
        self._d23 = None  # glass thickness  element 2  (surface 2→3)
        self._d35 = None  # air gap from surface 3 exit through stop to surface 5
        self._d56 = None  # glass thickness  element 3  (surface 5→6)
        self._z_s6 = None # z-position of surface 6 (used as BFL origin)
        self._f_target: Optional[float] = None  # set by _load_template_lens from config

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return
        _ensure_paths()
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

        # Aperture: stop-radius disk at the first glass surface plane
        stop_r = float(self._template_lens.surfaces[self.config.stop_surface_index].r)
        disk = _disk_samples(int(self.config.num_rays))
        self._aperture_xy = torch.tensor(
            disk * stop_r, dtype=torch.float32, device=device
        )

        # Zernike pseudo-inverse (built from unit-disk coords, same as singlet)
        j_max = max(int(j) for j in self.config.descriptor_zernike_indices)
        pinv_np = build_zernike_pinv(disk, j_max=j_max)
        self._pinv_Z = torch.tensor(pinv_np, dtype=torch.float32, device=device)

        # Reusable lens — mutated in-place instead of deepcopy per evaluation
        self._reusable_lens = copy.deepcopy(self._template_lens)

        self._backend_loaded = True

    def _load_template_lens(self):
        torch = self._torch
        lens = self._GeoLens(filename=self.config.lens_path, device=self._device)

        wvln = float(self.config.wavelength_um)

        # Element 1: surfaces 0 (air→glass) and 1 (glass→air)
        self._n1 = float(lens.surfaces[0].mat2.refractive_index(wvln))
        # Element 2: surfaces 2 (air→glass) and 3 (glass→air)
        self._n2 = float(lens.surfaces[2].mat2.refractive_index(wvln))
        # Element 3: surfaces 5 (air→glass) and 6 (glass→air)
        self._n3 = float(lens.surfaces[5].mat2.refractive_index(wvln))

        # Fixed axial gaps (read from surface positions in lens)
        d = [float(lens.surfaces[i].d.item()) for i in range(7)]
        self._d01 = d[1] - d[0]           # glass thickness element 1
        self._d12 = d[2] - d[1]           # air gap element 1→2
        self._d23 = d[3] - d[2]           # glass thickness element 2
        self._d35 = d[5] - d[3]           # air gap through stop to element 3
        self._d56 = d[6] - d[5]           # glass thickness element 3
        self._z_s6 = d[6]                 # z-position of last refractive surface

        # Focal-length target: use config value or read from the loaded lens.
        if self.config.focal_length_target_mm is not None:
            self._f_target = float(self.config.focal_length_target_mm)
        else:
            self._f_target = None

        # Zero all curvatures — template is a neutral (plano) baseline
        for idx in self._SURF_IDX:
            lens.surfaces[idx].c = torch.tensor(
                0.0, dtype=lens.surfaces[idx].c.dtype, device=self._device
            )
        lens.d_sensor = torch.tensor(float(lens.d_sensor.item()), device=self._device)
        return lens

    # ------------------------------------------------------------------
    # Paraxial ABCD back-focal-length solve  (differentiable w.r.t. c_t)
    # ------------------------------------------------------------------

    def _paraxial_bfl(self, c_t):
        """Chain paraxial ABCD matrices for the Cooke triplet → d_sensor.

        Uses reduced-coordinates [y, nu = n·u] convention.
          Refraction:  nu_new = nu_old − phi · y       (phi = delta_n · c)
          Transfer  :  y_new  = y_old + (t / n) · nu

        Trace a marginal ray [y=1, nu=0] through all 6 surfaces and compute
        BFL = −y / nu in the final air medium.
        """
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        d01, d12, d23 = self._d01, self._d12, self._d23
        d35, d56 = self._d35, self._d56

        dtype = c_t.dtype
        dev = c_t.device
        y = torch.ones(1, dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        # Surface 0: air → glass n1   (phi = (n1−1)·c0)
        nu = nu - (n1 - 1.0) * c_t[0] * y
        y = y + (d01 / n1) * nu                # transfer through element-1 glass

        # Surface 1: glass n1 → air  (phi = (1−n1)·c1)
        nu = nu - (1.0 - n1) * c_t[1] * y
        y = y + d12 * nu                        # transfer through air gap

        # Surface 2: air → glass n2   (phi = (n2−1)·c2)
        nu = nu - (n2 - 1.0) * c_t[2] * y
        y = y + (d23 / n2) * nu                # transfer through element-2 glass

        # Surface 3: glass n2 → air  (phi = (1−n2)·c3)
        nu = nu - (1.0 - n2) * c_t[3] * y
        y = y + d35 * nu                        # transfer through air (incl. stop)

        # Surface 5: air → glass n3   (phi = (n3−1)·c5)
        nu = nu - (n3 - 1.0) * c_t[4] * y
        y = y + (d56 / n3) * nu                # transfer through element-3 glass

        # Surface 6: glass n3 → air  (phi = (1−n3)·c6)
        nu = nu - (1.0 - n3) * c_t[5] * y
        # BFL in air: distance until y = 0
        safe_nu = torch.where(torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6))
        bfl = -y / safe_nu                     # scalar tensor

        return self._z_s6 + bfl.squeeze()

    def _paraxial_fixed_efl(self, c_t):
        """Trace paraxial marginal ray through surfaces 0-5, then derive c6 and d_sensor
        to enforce EFL = f_target (differentiable w.r.t. c_t).

        c_t: tensor of shape (5,) = [c0, c1, c2, c3, c5]

        Convention (reduced coordinates):
            phi_k = delta_n_k * c_k
            Refraction : nu_new = nu - phi * y
            Transfer   : y_new  = y + (t/n) * nu

        After tracing through surfaces 0-5 we obtain (y_5, nu_5).  Surface 6 has
        power phi_6 = (1-n3)*c6.  Enforcing EFL = f_target means the total system
        must satisfy nu_final = -P where P = 1/f_target:
            nu_5 - phi_6 * y_5 = -P
            phi_6 = (nu_5 + P) / y_5
            c6    = phi_6 / (1 - n3)

        BFL after surface 6 (nu_final = -P exactly):
            bfl = -y_5 / (-P) = y_5 * f_target

        Returns: (c6, d_sensor, bfl) — all scalar tensors, all differentiable.
        """
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        d01, d12, d23 = self._d01, self._d12, self._d23
        d35, d56 = self._d35, self._d56

        dtype = c_t.dtype
        dev = c_t.device
        y = torch.ones(1, dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        # Surface 0: air → glass n1
        nu = nu - (n1 - 1.0) * c_t[0] * y
        y = y + (d01 / n1) * nu
        # Surface 1: glass n1 → air
        nu = nu - (1.0 - n1) * c_t[1] * y
        y = y + d12 * nu
        # Surface 2: air → glass n2
        nu = nu - (n2 - 1.0) * c_t[2] * y
        y = y + (d23 / n2) * nu
        # Surface 3: glass n2 → air
        nu = nu - (1.0 - n2) * c_t[3] * y
        y = y + d35 * nu
        # Surface 5: air → glass n3
        nu = nu - (n3 - 1.0) * c_t[4] * y
        y = y + (d56 / n3) * nu
        # Now at exit of element-3 glass; next is surface 6 (glass n3 → air).

        P = 1.0 / self._f_target
        safe_y = torch.where(torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6))
        phi_6 = (nu + P) / safe_y
        c6 = phi_6 / (1.0 - n3)
        bfl = y * float(self._f_target)  # exact: -y / (-P)
        d_sensor = self._z_s6 + bfl.squeeze()
        return c6.squeeze(), d_sensor, bfl.squeeze()

    def _paraxial_efl(self, c_t):
        """Paraxial effective focal length (free mode, 6-D theta)."""
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        d01, d12, d23 = self._d01, self._d12, self._d23
        d35, d56 = self._d35, self._d56

        dtype, dev = c_t.dtype, c_t.device
        y = torch.ones(1, dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        nu = nu - (n1 - 1.0) * c_t[0] * y;  y = y + (d01 / n1) * nu
        nu = nu - (1.0 - n1) * c_t[1] * y;  y = y + d12 * nu
        nu = nu - (n2 - 1.0) * c_t[2] * y;  y = y + (d23 / n2) * nu
        nu = nu - (1.0 - n2) * c_t[3] * y;  y = y + d35 * nu
        nu = nu - (n3 - 1.0) * c_t[4] * y;  y = y + (d56 / n3) * nu
        nu = nu - (1.0 - n3) * c_t[5] * y
        # EFL = 1 / |total power|;  for initial (y=1, nu=0), nu_final = -P
        return 1.0 / torch.abs(nu).clamp_min(1e-8).squeeze()

    def _calc_hfov_deg(self, theta_t):
        """Paraxial half-field of view in degrees for the given design."""
        if self._f_target is not None:
            efl = self._f_target
        else:
            efl = float(self._paraxial_efl(theta_t).detach())
        half_sensor = self._template_lens.sensor_size[0] / 2.0
        return math.degrees(math.atan(half_sensor / abs(efl)))

    # ------------------------------------------------------------------
    # Lens construction
    # ------------------------------------------------------------------

    def _build_lens_from_theta_tensor(self, theta_t):
        """Build a GeoLens from theta_t; d_sensor via paraxial solve.

        Free mode   (focal_length_target_mm is None):
            theta_t = [c0,c1,c2,c3,c5,c6]  (6-D); d_sensor from _paraxial_bfl.
        Fixed-EFL mode (focal_length_target_mm is set):
            theta_t = [c0,c1,c2,c3,c5]      (5-D); c6 and d_sensor from
            _paraxial_fixed_efl.
        """
        if self._f_target is not None:
            c6, d_sensor, _ = self._paraxial_fixed_efl(theta_t)
            curvatures = [theta_t[0], theta_t[1], theta_t[2], theta_t[3], theta_t[4], c6]
        else:
            d_sensor = self._paraxial_bfl(theta_t)
            curvatures = [theta_t[k] for k in range(6)]
        lens = self._reusable_lens
        dtype = lens.surfaces[self._SURF_IDX[0]].c.dtype
        for k, idx in enumerate(self._SURF_IDX):
            lens.surfaces[idx].c = curvatures[k].to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _make_object_point(self, field_x_deg: float, field_y_deg: float):
        depth = float(self.config.object_depth_mm)
        x = depth * math.tan(math.radians(field_x_deg))
        y = depth * math.tan(math.radians(field_y_deg))
        return x, y, depth

    def initial_theta(self) -> np.ndarray:
        self._ensure_backend()
        return np.zeros(len(self.config.theta_bounds), dtype=np.float64)

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        dim = len(self.config.theta_bounds)
        unit = _latin_hypercube(int(n), dim, int(seed))
        lo = np.array([b[0] for b in self.config.theta_bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self.config.theta_bounds], dtype=np.float64)
        return lo + unit * (hi - lo)

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        return _project_box(theta, self.config.theta_bounds)

    def theta_to_lens(self, theta: np.ndarray):
        """Public API: build a GeoLens from a 6D numpy theta."""
        self._ensure_backend()
        theta_proj = self.project_theta(theta)
        theta_t = self._torch.tensor(
            theta_proj, dtype=self._torch.float32, device=self._device
        )
        return self._build_lens_from_theta_tensor(theta_t)

    # ------------------------------------------------------------------
    # Incoherent spot-size trace (fitness)
    # ------------------------------------------------------------------

    def _trace_field(self, lens, field_x_deg: float, field_y_deg: float, wvln=None):
        torch = self._torch
        if wvln is None:
            wvln = float(self.config.wavelength_um)
        x_obj, y_obj, z_obj = self._make_object_point(field_x_deg, field_y_deg)
        N = self._aperture_xy.shape[0]
        dtype = lens.surfaces[self._SURF_IDX[0]].c.dtype

        origins = torch.empty((N, 3), dtype=dtype, device=self._device)
        origins[:, 0] = x_obj
        origins[:, 1] = y_obj
        origins[:, 2] = z_obj

        s0_z = float(lens.surfaces[self._SURF_IDX[0]].d.item())
        targets = torch.empty((N, 3), dtype=dtype, device=self._device)
        targets[:, :2] = self._aperture_xy.to(dtype=dtype)
        targets[:, 2] = s0_z

        ray = self._Ray(
            origins, targets - origins,
            wvln=wvln, device=self._device
        )
        return lens.trace2sensor(ray)

    def _rms_spot(self, ray):
        torch = self._torch
        valid = ray.is_valid
        xy = ray.o[..., :2]
        denom = valid.sum().clamp_min(1e-8)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
        err2 = ((xy - centroid) ** 2).sum(dim=-1)
        rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
        return rms, 1.0 - (valid.sum() / max(float(valid.numel()), 1.0))

    def _field_points(self, hfov_deg: float):
        if str(self.config.stage).lower().strip() == "stage2":
            # 3 meridional positions: on-axis, mid-field, full-field
            return ((0.0, 0.0), (0.0, 0.5 * hfov_deg), (0.0, hfov_deg))
        return ((0.0, 0.0),)

    def _trace_fields_batch(self, lens, field_points, wvln):
        """Trace all field points at one wavelength in one batched call.

        Returns ray with batch shape (F, num_rays).
        """
        torch = self._torch
        N = self._aperture_xy.shape[0]
        F = len(field_points)
        dtype = lens.surfaces[self._SURF_IDX[0]].c.dtype

        origins = torch.empty((F, N, 3), dtype=dtype, device=self._device)
        targets = torch.empty((F, N, 3), dtype=dtype, device=self._device)
        s0_z = float(lens.surfaces[self._SURF_IDX[0]].d.item())

        for i, (fx, fy) in enumerate(field_points):
            x_obj, y_obj, z_obj = self._make_object_point(fx, fy)
            origins[i, :, 0] = x_obj
            origins[i, :, 1] = y_obj
            origins[i, :, 2] = z_obj
            targets[i, :, :2] = self._aperture_xy.to(dtype=dtype)
            targets[i, :, 2] = s0_z

        ray = self._Ray(origins, targets - origins, wvln=wvln, device=self._device)
        return lens.trace2sensor(ray)

    def _rms_spot_batch(self, ray):
        """RMS spot for each field point in a batched ray. Returns (F,) tensors."""
        torch = self._torch
        valid = ray.is_valid                    # (F, num_rays)
        xy = ray.o[..., :2]                     # (F, num_rays, 2)
        denom = valid.sum(dim=-1).clamp_min(1e-8)                          # (F,)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)  # (F, 2)
        err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)            # (F, num_rays)
        rms = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()          # (F,)
        inv_frac = 1.0 - valid.sum(dim=-1) / valid.shape[-1]              # (F,)
        return rms, inv_frac

    # ------------------------------------------------------------------
    # Coherent trace for Zernike OPD
    # ------------------------------------------------------------------

    def _trace_coherent(self, lens, field_x_deg: float = 0.0, field_y_deg: float = 0.0):
        """Coherent ray bundle from the first glass surface toward the object point.

        Rays start at (x_p, y_p, z_s0=0) — the first glass-surface plane — using
        the stop-aperture Fibonacci disk.  The missing incoming OPL (object → first
        surface) is restored in _zernike_measures.  float64 is required by DeepLens
        coherent prop_to.
        """
        torch = self._torch
        N = self._aperture_xy.shape[0]
        dtype = torch.float64

        z_obj = float(self.config.object_depth_mm)
        x_obj = z_obj * math.tan(math.radians(field_x_deg))
        y_obj = z_obj * math.tan(math.radians(field_y_deg))

        s0_z = float(lens.surfaces[self._SURF_IDX[0]].d.item())  # = 0.0

        origins = torch.zeros((N, 3), dtype=dtype, device=self._device)
        origins[:, :2] = self._aperture_xy.to(dtype=dtype)
        origins[:, 2] = s0_z

        obj = torch.tensor([x_obj, y_obj, z_obj], dtype=dtype, device=self._device)
        d = torch.nn.functional.normalize(origins - obj.unsqueeze(0), dim=-1)

        ray = self._Ray(origins, d, wvln=float(self.config.wavelength_um),
                        coherent=True, device=self._device)
        return lens.trace2sensor(ray)

    # ------------------------------------------------------------------
    # Zernike OPD measurement
    # ------------------------------------------------------------------

    def _zernike_measures(self, lens, hfov_deg: float):
        """Fit Noll Zernike coefficients to the wavefront OPD at the exit pupil.

        Pipeline is identical to the singlet _zernike_measures:
          1. Coherent trace → sensor → propagate to exit pupil.
          2. Restore incoming OPL (object → first surface).
          3. Trace chief ray at selected field index.
          4. Reference sphere centred on chief-ray exit-pupil position.
          5. OPD_i = (OPL_i − OPL_chief) + ref_i,  zeroed for invalid rays.
          6. Zernike fit: coeffs = pinv_Z @ opd.
        """
        torch = self._torch

        idx = int(self.config.zernike_field_index)
        field = self._field_points(hfov_deg)[idx][1]  # meridional (y) component
        z_obj_val = float(self.config.object_depth_mm)
        z_obj_abs = abs(z_obj_val)
        y_obj = z_obj_val * math.tan(math.radians(field))

        # Step 1: coherent trace → exit pupil
        ray = self._trace_coherent(lens, field_x_deg=0.0, field_y_deg=field)
        pupil_z, _ = lens.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        opl   = ray.opl.squeeze(-1)
        valid = ray.is_valid.float()
        x_ep  = ray.o[:, 0].to(torch.float64)
        y_ep  = ray.o[:, 1].to(torch.float64)

        # Step 2: restore incoming OPL (object → first surface plane, z=0)
        xy_stop = self._aperture_xy.to(torch.float64)
        xp, yp = xy_stop[:, 0], xy_stop[:, 1]
        opl_marginal_in = torch.sqrt(xp ** 2 + (yp - y_obj) ** 2 + z_obj_abs ** 2)
        opl_chief_in = math.sqrt(y_obj ** 2 + z_obj_abs ** 2)
        opl = opl + (opl_marginal_in - opl_chief_in)

        # Step 3: chief ray at selected field index (meridional plane)
        if field == 0.0:
            chief_rfov = torch.tensor([0.0], device=self._device)
        else:
            chief_rfov = torch.tensor([0.0, field], device=self._device)
        chief_o, chief_d = lens.calc_chief_ray_infinite(chief_rfov, plane="meridional")
        chief_ray = self._Ray(
            chief_o[-1:].to(torch.float64),
            chief_d[-1:].to(torch.float64),
            wvln=float(self.config.wavelength_um),
            coherent=True,
            device=self._device,
        )
        chief_ray = lens.trace2sensor(chief_ray)
        chief_ray = chief_ray.prop_to(pupil_z)
        opl_chief = chief_ray.opl.squeeze()
        x_chief_ep = chief_ray.o[0, 0].to(torch.float64)
        y_chief_ep = chief_ray.o[0, 1].to(torch.float64)

        # Step 4: reference sphere centred on chief-ray exit-pupil position
        BFL = lens.d_sensor - pupil_z
        ref_sphere = torch.sqrt(
            (x_ep - x_chief_ep) ** 2 + (y_ep - y_chief_ep) ** 2 + BFL ** 2
        ) - BFL

        # Step 5: OPD (invalid rays zeroed)
        opd = ((opl - opl_chief) + ref_sphere) * valid

        # Step 6: Zernike fit
        return self._pinv_Z @ opd.to(self._pinv_Z.dtype)

    # ------------------------------------------------------------------
    # Fitness + metrics
    # ------------------------------------------------------------------

    def _fitness_and_metrics(self, theta_t):
        torch = self._torch

        # Reject diverging lenses: BFL <= 0 → heavy penalty (graph stays connected)
        if self._f_target is not None:
            _, _, bfl_t = self._paraxial_fixed_efl(theta_t)
            bfl_val = float(bfl_t.detach().item())
        else:
            bfl_val = float((self._paraxial_bfl(theta_t) - self._z_s6).detach().item())
        if bfl_val <= 0.0:
            bad = theta_t.sum() * 0.0 - 1e9
            return bad, theta_t[0] * 0.0, theta_t[1] * 0.0, {"invalid_fraction": 1.0}

        lens = self._build_lens_from_theta_tensor(theta_t)
        hfov = self._calc_hfov_deg(theta_t)

        # Spot-size fitness: batch all field points per wavelength
        wavelengths = self.config.fitness_wavelengths_um
        field_pts = self._field_points(hfov)
        rms_per_wvln, inv_per_wvln = [], []
        for w in wavelengths:
            ray = self._trace_fields_batch(lens, field_pts, wvln=w)
            rms, inv_frac = self._rms_spot_batch(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        rms_all = torch.stack(rms_per_wvln)          # (n_wvln, F)
        inv_all = torch.stack(inv_per_wvln)          # (n_wvln, F)
        rms_mean = rms_all.mean()
        invalid_fraction_mean = inv_all.mean()
        fitness = -rms_mean

        # Invalid-fraction penalty (same as singlet two-param)
        if float(invalid_fraction_mean.detach().item()) > self.config.invalid_tol:
            fitness = fitness - float(self.config.invalid_penalty_weight) * 1e5

        # Zernike behavioral descriptors
        coeffs = self._zernike_measures(lens, hfov)
        j0, j1 = self.config.descriptor_zernike_indices
        m0, m1 = coeffs[j0 - 1], coeffs[j1 - 1]

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(invalid_fraction_mean.detach().cpu().item()),
            f"z{j0}_{NOLL_TERM_NAMES.get(j0, 'unknown')}_mm": float(m0.detach().cpu().item()),
            f"z{j1}_{NOLL_TERM_NAMES.get(j1, 'unknown')}_mm": float(m1.detach().cpu().item()),
        }
        return fitness, m0, m1, metrics

    # ------------------------------------------------------------------
    # Public evaluate
    # ------------------------------------------------------------------

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> TripletEvalResult:
        self._ensure_backend()
        torch = self._torch
        theta_proj = self.project_theta(theta)
        theta_t = torch.tensor(
            theta_proj, dtype=torch.float32, device=self._device,
            requires_grad=with_grads
        )
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
                k0:        g_m0.detach().cpu().numpy().astype(np.float64, copy=False),
                k1:        g_m1.detach().cpu().numpy().astype(np.float64, copy=False),
            }

        return TripletEvalResult(
            theta=theta_proj.astype(np.float64, copy=False),
            fitness=float(fitness.detach().cpu().item()),
            measures=(float(m0.detach().cpu().item()), float(m1.detach().cpu().item())),
            metrics=metrics,
            grads=grads,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TripletCookeZernikeRunner:
    """CMA-MEGA runner for the Cooke-triplet Zernike problem."""

    def __init__(self, config: TripletCookeZernikeConfig):
        self.config = config
        self.problem = TripletCookeZernikeProblem(config)
        self.archive: Optional[TripletArchive2D] = None
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
        self.result_dir = None  # type: ignore[assignment]

    def calibrate_archive_bounds(self):
        samples = self.problem.sample_theta(self.config.calibration_count, self.config.cma_seed + 17)
        m0_vals, m1_vals = [], []
        for theta in samples:
            result = self.problem.evaluate(theta, with_grads=False)
            m0_vals.append(result.measures[0])
            m1_vals.append(result.measures[1])

        def _bounds(arr):
            lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
            span = max(hi - lo, 1e-6)
            return (lo - 0.05 * span, hi + 0.05 * span)

        bounds = (
            _bounds(np.asarray(m0_vals, dtype=np.float64)),
            _bounds(np.asarray(m1_vals, dtype=np.float64)),
        )
        self.archive = TripletArchive2D(self.config.archive_bins, bounds, seed=self.config.cma_seed)
        return bounds

    def seed_archive(self) -> Dict[str, float]:
        if self.archive is None:
            if self.config.archive_bounds is not None:
                self.archive = TripletArchive2D(
                    self.config.archive_bins,
                    self.config.archive_bounds,
                    seed=self.config.cma_seed,
                )
            else:
                self.calibrate_archive_bounds()

        flat = self.problem.initial_theta()
        self.archive.insert(self.problem.evaluate(flat, with_grads=False))
        for theta in self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31):
            self.archive.insert(self.problem.evaluate(theta, with_grads=False))
        return self.archive.stats()

    def _normalized_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        half = 0.5 * np.array(
            [hi - lo for lo, hi in self.config.theta_bounds], dtype=np.float64
        )
        j0, j1 = self.config.descriptor_zernike_indices

        def norm(name: str) -> np.ndarray:
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        return {"fitness": norm("fitness"), f"z{j0}": norm(f"z{j0}"), f"z{j1}": norm(f"z{j1}")}

    def run(self, iterations: int, *, resume: bool = False) -> Dict[str, float]:
        """Canonical CMA-MEGA loop. Mirrors EF50ZernikeRunner.run."""
        import signal
        import time
        from pathlib import Path
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
                "[triplet-zernike] SIGINT received — checkpointing then exiting.",
                flush=True,
            )

        signal.signal(signal.SIGINT, _handle_sigint)
        loop_t0 = time.perf_counter()

        try:
            while self.completed < target_completed:
                if sigint_received["flag"]:
                    break
                parent_theta = self.archive.sample_parent()
                parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
                assert parent_eval.grads is not None

                unit_parent = _theta_to_unit(parent_theta, self.config.theta_bounds)
                grads = self._normalized_gradients(parent_eval.grads)
                batch_n = min(int(self.config.population_size), target_completed - self.completed)
                coeffs = self.emitter.ask(batch_n)

                batch_scores: list = []
                batch_status: list = []
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
                    candidate_theta = _unit_to_theta(candidate_unit, self.config.theta_bounds)
                    candidate_eval = self.problem.evaluate(candidate_theta, with_grads=False)
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
                        f"[triplet-zernike] [restart] gen={self.generation} "
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
                        f"[triplet-zernike] gen={self.generation} evals={self.completed} "
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
                print(f"[triplet-zernike] WARNING: final checkpoint failed: {e}", flush=True)
            signal.signal(signal.SIGINT, prev_handler)

        s = self.archive.stats()
        s["iterations"] = float(eval_budget)
        s["generations"] = float(self.generation)
        s["restart_count"] = float(self.restart_count)
        s["mean_score"] = float(np.mean(scores)) if scores else float("nan")
        return s
