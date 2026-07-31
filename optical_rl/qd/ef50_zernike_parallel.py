"""Canon EF50mm f/1.8 CMA-MEGA with Zernike behavioral descriptors — PARALLEL VARIANT.

This is a copy of ef50_zernike.py where ONLY the seeding and calibration
phases have been switched to the multi-GPU multiprocessing pipeline from
seed_parallel_experiment.py.  The CMA-MEGA optimization loop, fitness
function, problem class, archive, and CLI are otherwise identical to the
sequential file.

If anything goes wrong with the parallel path, fall back to the original
optical_rl.qd.ef50_zernike — both files share the same YAML config.

theta = [c0, c1, c2, c3, c4, c5, c7, c8, c9, c10]  (10 surface curvatures; surface
6 is the aperture stop and is never touched; c11 is derived from the focal constraint)

d_sensor is derived from a paraxial ABCD back-focal-length solve performed in
PyTorch so that autograd gradients flow through it to all ten curvatures.

Lens layout (ef50mm_f1.8.json):
  0  Spheric  n1=1.6280   element 1 front
  1  Spheric  air         element 1 rear
  2  Spheric  n2=1.70154  element 2 front
  3  Spheric  air         element 2 rear
  4  Spheric  n3=1.6727   element 3 front
  5  Spheric  air         element 3 rear
  6  Stop     ---         aperture stop  (never touched)
  7  Spheric  n4=1.6990   cemented doublet front
  8  Spheric  n5=1.7995   cemented doublet interface  (n4→n5, not air!)
  9  Spheric  air         cemented doublet rear
 10  Spheric  n6=1.7200   element 5 front
 11  Spheric  air         element 5 rear  (derived from focal constraint)

Design philosophy mirrors triplet_cooke_zernike.py exactly:
  - Auto d_sensor via differentiable paraxial BFL (no free d_sensor parameter)
  - Aperture sampling on stop aperture disk (r = stop surface radius from JSON)
  - Zernike OPD pipeline identical to the triplet (coherent trace from first
    surface, incoming OPL correction, chief ray, reference sphere, pinv fit)
  - stage1 / stage2 fitness (on-axis RMS vs mean over 5 field points)
  - Zernike field angle is independent of fitness field angle
  - Diverging-lens rejection (BFL <= 0 → heavy penalty)
"""
from __future__ import annotations

import copy
import json
import math
import os
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


_ensure_paths()  # Must run at import time so optical_rl.* imports below resolve


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
    """Fibonacci spiral on the unit disk — identical to triplet implementation."""
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
# EvalResult
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc, field as _field  # noqa: E402


@_dc
class EF50EvalResult:
    theta: np.ndarray
    fitness: float
    measures: Tuple[float, float]
    metrics: Dict[str, float]
    grads: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EF50ZernikeConfig:
    lens_path: str = str(_PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json")
    object_depth_mm: float = -20000.0
    wavelength_um: float = 0.587

    # Wavelengths for polychromatic RMS spot evaluation.
    # Single entry = monochromatic (backward compatible).
    # (0.656, 0.587, 0.486) = Fraunhofer C, d, F lines (R, G, B).
    fitness_wavelengths_um: Tuple[float, ...] = (0.656, 0.587, 0.486)

    # theta = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10] — 10 free curvatures, all ±0.09
    # c11 (rear element 5) is derived from the focal-length constraint.
    theta_bounds: Tuple = (
        (-0.09, 0.09),  # c0  element 1 front
        (-0.09, 0.09),  # c1  element 1 rear
        (-0.09, 0.09),  # c2  element 2 front
        (-0.09, 0.09),  # c3  element 2 rear
        (-0.09, 0.09),  # c4  element 3 front
        (-0.09, 0.09),  # c5  element 3 rear
        (-0.09, 0.09),  # c7  cemented doublet front
        (-0.09, 0.09),  # c8  cemented doublet interface
        (-0.09, 0.09),  # c9  cemented doublet rear
        (-0.09, 0.09),  # c10 element 5 front
        # c11 is derived — do NOT add an 11th entry here
    )

    # Index of the aperture stop surface in the JSON surface list (0-based).
    stop_surface_index: int = 6

    # c11 (rear surface of element 5) is derived from the paraxial ABCD solve
    # to enforce EFL = focal_length_target_mm.  theta is 10-D.
    # Set to None to revert to free 11-D mode (c11 free, d_sensor from BFL).
    focal_length_target_mm: Optional[float] = 50.52

    # Two 1-based Noll Zernike indices used as archive behavioral descriptors.
    # (7, 11) = vertical coma + primary spherical — dominant for fast f/1.8 lenses.
    descriptor_zernike_indices: Tuple[int, int] = (7, 11)

    archive_bins: Tuple[int, int] = (64, 64)
    archive_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    invalid_tol: float = 0.15
    invalid_penalty_weight: float = 5.0

    cma_eta: float = 0.05
    seed_count: int = 128
    calibration_count: int = 512
    num_rays: int = 512

    # Fitness stage: "stage1" = on-axis RMS only;
    # "stage2" = mean over 3 meridional field pts (0, 0.5, 1.0 × HFOV).
    stage: str = "stage1"

    # Which of the 3 hfov3 field positions to use for Zernike OPD measurement:
    #   0 = on-axis (0.0°), 1 = mid-field (0.5 × HFOV), 2 = full-field (1.0 × HFOV)
    zernike_field_index: int = 0

    # Intersection loss (soft penalty)
    intersect_penalty_weight: float = 1.0
    air_min_mm: float = 0.1
    thick_min_mm: float = 1.0

    device: str = "auto"
    initial_sigma: float = 0.75
    population_size: int = 16
    cma_seed: int = 0
    results_dir: str = "results/models"
    run_name: str = "ef50_zernike"
    log_every: int = 50
    # When True, print a line each time seeding inserts into the MAP-Elites archive.
    log_seed_accepts: bool = True
    iterations: int = 1000

    # When True, designs whose entrance pupil cannot be computed (front group is
    # degenerate / fails ray-tracing intersection in calc_entrance_pupil) get a
    # -1e9 penalty fitness instead of silently falling back to surfaces[0].r.
    # Prevents these "phantom" designs from poisoning the MAP-Elites archive
    # with misleadingly competitive scores.  Applies to all evaluations.
    strict_entrance_pupil: bool = False

    # Number of parallel worker processes used for seeding/calibration in
    # ef50_zernike_parallel.py.  None = auto-detect based on CPU/GPU count.
    # Ignored by ef50_zernike.py (sequential).
    parallel_workers: Optional[int] = None

    # Stop-and-go checkpointing. Every checkpoint_every CMA-MEGA generations
    # the runner pickles a CMAMEGACheckpoint into result_dir/checkpoint.pkl.
    # 0 disables mid-run snapshots; the final checkpoint is always written
    # at clean shutdown (and on SIGINT). Resume is via --resume <run_dir>.
    checkpoint_every: int = 50

    # Bonus added to candidate fitness in CMA-MEGA tell() when the
    # candidate filled a previously-empty archive cell. Used so that new
    # cells always rank above replaced elites in the same generation,
    # implementing canonical CMA-MEGA's two-tier improvement ranking.
    # Must be larger than any plausible |fitness|; the value only affects
    # sorting inside the emitter, never the archive.
    new_cell_bonus: float = 1.0e6


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class EF50Archive2D:
    def __init__(self, bins, bounds, *, seed: int = 0):
        self.bins = (int(bins[0]), int(bins[1]))
        self.bounds = bounds
        self._rng = np.random.default_rng(int(seed))
        self._cells: Dict[Tuple[int, int], EF50EvalResult] = {}
        self._best_fitness = -float("inf")

    def _to_index(self, measures):
        (x_lo, x_hi), (y_lo, y_hi) = self.bounds
        x, y = float(measures[0]), float(measures[1])
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return None
        def _bin(v, lo, hi, n):
            return int(np.clip(np.floor((v - lo) / (hi - lo) * n), 0, n - 1))
        return (_bin(x, x_lo, x_hi, self.bins[0]), _bin(y, y_lo, y_hi, self.bins[1]))

    def insert(self, result: EF50EvalResult) -> int:
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
# GradientCoefficientCMA
# ---------------------------------------------------------------------------

from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA  # noqa: E402


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class EF50ZernikeProblem:
    """Canon EF50mm f/1.8 CMA-MEGA problem with Zernike behavioral descriptors.

    Surfaces in the JSON (ef50mm_f1.8.json):
      0  Spheric  n1=1.6280   element 1 front
      1  Spheric  air         element 1 rear
      2  Spheric  n2=1.70154  element 2 front
      3  Spheric  air         element 2 rear
      4  Spheric  n3=1.6727   element 3 front
      5  Spheric  air         element 3 rear
      6  Stop     ---         aperture stop
      7  Spheric  n4=1.6990   cemented doublet front
      8  Spheric  n5=1.7995   cemented doublet interface
      9  Spheric  air         cemented doublet rear
     10  Spheric  n6=1.7200   element 5 front
     11  Spheric  air         element 5 rear  (derived)

    theta = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10]  — all ±0.09 mm⁻¹
    d_sensor derived via paraxial ABCD BFL solve (differentiable).
    """

    # Surface indices of the eleven refractive surfaces (skip stop at idx 6)
    _SURF_IDX = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)

    def __init__(self, config: EF50ZernikeConfig):
        self.config = config
        self._backend_loaded = False
        self._torch = None
        self._GeoLens = None
        self._Ray = None
        self._device = None
        self._template_lens = None
        self._aperture_xy = None   # (N,2) stop-plane positions in mm
        self._pinv_Z = None        # (j_max, N) Zernike pseudo-inverse tensor

        # Glass refractive indices at wavelength (set in _load_template_lens)
        self._n1 = None   # element 1        (surfaces 0–1)
        self._n2 = None   # element 2        (surfaces 2–3)
        self._n3 = None   # element 3        (surfaces 4–5)
        self._n4 = None   # cemented front   (surface 7, mat2)
        self._n5 = None   # cemented rear    (surface 8, mat2 = n4→n5 interface)
        self._n6 = None   # element 5        (surfaces 10–11)

        # Fixed axial gaps (set in _load_template_lens)
        self._d01 = None     # glass thickness  element 1
        self._d12 = None     # air gap          element 1 → element 2
        self._d23 = None     # glass thickness  element 2
        self._d34 = None     # air gap          element 2 → element 3
        self._d45 = None     # glass thickness  element 3
        self._d5_to_7 = None # air gap  element 3 rear → cemented doublet front
                             #          (spans stop at surface 6)
        self._d78 = None     # glass thickness  cemented doublet front (n4)
        self._d89 = None     # glass thickness  cemented doublet rear  (n5)
        self._d9_10 = None   # air gap          cemented doublet → element 5
        self._d10_11 = None  # glass thickness  element 5
        self._z_s11 = None   # z-position of surface 11 (BFL origin)

        self._f_target: Optional[float] = None

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

        # DeepLens coherent OPL tracer raises if t > 100 and the global default
        # dtype is float32 ("Using float32 may cause precision problem for OPL
        # calculation"). The EF50 has d_sensor ≈ 72 mm so propagation steps
        # routinely exceed 100. GeoLens.__init__ resets the global dtype to
        # float32, so we override it HERE — after the template is loaded.
        # This is consistent with how the Cooke triplet works (its "(c)" hint
        # surfaces are float64 and it never triggers the check).
        torch.set_default_dtype(torch.float64)

        # Aperture: stop-radius disk at the stop surface plane (for Zernike OPD)
        stop_r = float(self._template_lens.surfaces[self.config.stop_surface_index].r)
        disk = _disk_samples(int(self.config.num_rays))
        self._aperture_xy = torch.tensor(
            disk * stop_r, dtype=torch.float32, device=device
        )

        # Fitness aperture: unit-disk Fibonacci samples, scaled per-evaluation
        # to the built lens's paraxial entrance pupil (full beam, not paraxial pencil).
        fitness_disk = _disk_samples(int(self.config.num_rays))
        self._fitness_unit_disk = torch.tensor(
            fitness_disk, dtype=torch.float32, device=device
        )  # (N, 2) with max radius ≈ 1.0

        # Zernike pseudo-inverse (built from unit-disk coords)
        j_max = max(int(j) for j in self.config.descriptor_zernike_indices)
        pinv_np = build_zernike_pinv(disk, j_max=j_max)
        self._pinv_Z = torch.tensor(pinv_np, dtype=torch.float32, device=device)

        # Reusable lens — mutated in-place instead of deepcopy per evaluation
        self._reusable_lens = copy.deepcopy(self._template_lens)

        self._backend_loaded = True

    def _load_template_lens(self):
        """Load EF50 lens, extract glass indices + gaps, zero all curvatures."""
        torch = self._torch
        lens = self._GeoLens(filename=self.config.lens_path, device=self._device)

        wvln = float(self.config.wavelength_um)

        # Glass refractive indices from mat2 of each surface
        self._n1 = float(lens.surfaces[0].mat2.refractive_index(wvln))   # 1.6280
        self._n2 = float(lens.surfaces[2].mat2.refractive_index(wvln))   # 1.70154
        self._n3 = float(lens.surfaces[4].mat2.refractive_index(wvln))   # 1.6727
        self._n4 = float(lens.surfaces[7].mat2.refractive_index(wvln))   # 1.6990
        self._n5 = float(lens.surfaces[8].mat2.refractive_index(wvln))   # 1.7995
        self._n6 = float(lens.surfaces[10].mat2.refractive_index(wvln))  # 1.7200

        # Axial positions of each surface
        d = [float(lens.surfaces[i].d.item()) for i in range(12)]

        self._d01    = d[1]  - d[0]   # 4.50   through n1
        self._d12    = d[2]  - d[1]   # 2.40   air
        self._d23    = d[3]  - d[2]   # 4.35   through n2
        self._d34    = d[4]  - d[3]   # 0.92   air
        self._d45    = d[5]  - d[4]   # 1.40   through n3
        self._d5_to_7 = d[7] - d[5]   # 11.70  air (absorbs stop at idx 6)
        self._d78    = d[8]  - d[7]   # 1.10   through n4
        self._d89    = d[9]  - d[8]   # 5.70   through n5 (cemented)
        self._d9_10  = d[10] - d[9]   # 0.15   air
        self._d10_11 = d[11] - d[10]  # 3.00   through n6
        self._z_s11  = d[11]          # 35.22  BFL origin

        # Focal-length target
        if self.config.focal_length_target_mm is not None:
            self._f_target = float(self.config.focal_length_target_mm)
        else:
            self._f_target = None

        # Compute entrance pupil from the ORIGINAL prescription (before zeroing
        # curvatures) so that fitness ray tracing samples the full f/1.8 beam.
        # calc_pupil() sets aper_idx and computes entrance/exit pupil via
        # backward ray tracing from the aperture stop edge.
        with torch.no_grad():
            lens.calc_pupil()
        self._entr_pupilz = float(lens.entr_pupilz)
        self._entr_pupilr = float(lens.entr_pupilr)

        # Zero all curvatures in float64. DeepLens's coherent OPL tracer rejects
        # float32 surfaces ("Using float32 may cause precision problem for OPL
        # calculation"). The Cooke triplet JSON uses parenthesised "(c)" hints
        # which load as float64; the EF50 JSON stores "c" as nn.Parameter
        # (float32 by default), so we must override to float64 here.
        # _build_lens_from_theta_tensor calls .to(dtype=dtype) on every curvature,
        # so all subsequent built lenses (both incoherent and coherent traces)
        # will also run in float64.
        for idx in self._SURF_IDX:
            lens.surfaces[idx].c = torch.tensor(
                0.0, dtype=torch.float64, device=self._device
            )
        lens.d_sensor = torch.tensor(
            float(lens.d_sensor.item()), dtype=torch.float64, device=self._device
        )
        return lens

    # ------------------------------------------------------------------
    # Paraxial ABCD back-focal-length solve  (differentiable w.r.t. c_t)
    # ------------------------------------------------------------------

    def _paraxial_bfl(self, c_t):
        """Chain paraxial ABCD matrices for the EF50 → d_sensor.

        Uses reduced-coordinates [y, nu = n·u] convention.
          Refraction:  nu_new = nu_old − phi · y       (phi = delta_n · c)
          Transfer  :  y_new  = y_old + (t / n) · nu

        Trace a marginal ray [y=1, nu=0] through all 11 refractive surfaces
        and compute BFL = −y / nu in the final air medium.

        c_t: tensor of shape (11,) = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10,c11]
        """
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        n4, n5, n6 = self._n4, self._n5, self._n6
        d01, d12, d23 = self._d01, self._d12, self._d23
        d34, d45     = self._d34, self._d45
        d5_to_7      = self._d5_to_7
        d78, d89     = self._d78, self._d89
        d9_10, d10_11 = self._d9_10, self._d10_11

        dtype = c_t.dtype
        dev = c_t.device
        y  = torch.ones(1,  dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        # Surface 0: air → n1
        nu = nu - (n1 - 1.0) * c_t[0] * y
        y  = y  + (d01 / n1) * nu

        # Surface 1: n1 → air
        nu = nu - (1.0 - n1) * c_t[1] * y
        y  = y  + d12 * nu

        # Surface 2: air → n2
        nu = nu - (n2 - 1.0) * c_t[2] * y
        y  = y  + (d23 / n2) * nu

        # Surface 3: n2 → air
        nu = nu - (1.0 - n2) * c_t[3] * y
        y  = y  + d34 * nu

        # Surface 4: air → n3
        nu = nu - (n3 - 1.0) * c_t[4] * y
        y  = y  + (d45 / n3) * nu

        # Surface 5: n3 → air  (transfer absorbs stop gap at surface 6)
        nu = nu - (1.0 - n3) * c_t[5] * y
        y  = y  + d5_to_7 * nu

        # Surface 7: air → n4
        nu = nu - (n4 - 1.0) * c_t[6] * y
        y  = y  + (d78 / n4) * nu

        # Surface 8: n4 → n5  (cemented doublet interface — glass-to-glass)
        nu = nu - (n5 - n4) * c_t[7] * y
        y  = y  + (d89 / n5) * nu

        # Surface 9: n5 → air
        nu = nu - (1.0 - n5) * c_t[8] * y
        y  = y  + d9_10 * nu

        # Surface 10: air → n6
        nu = nu - (n6 - 1.0) * c_t[9] * y
        y  = y  + (d10_11 / n6) * nu

        # Surface 11: n6 → air
        nu = nu - (1.0 - n6) * c_t[10] * y

        safe_nu = torch.where(torch.abs(nu) > 1e-6, nu, torch.full_like(nu, 1e-6))
        bfl = -y / safe_nu
        return self._z_s11 + bfl.squeeze()

    def _paraxial_fixed_efl(self, c_t):
        """Trace paraxial marginal ray through surfaces 0–10, then derive c11 and
        d_sensor to enforce EFL = f_target (differentiable w.r.t. c_t).

        c_t: tensor of shape (10,) = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10]

        After tracing surfaces 0–10 we obtain (y_10, nu_10).  Surface 11 has
        power phi_11 = (1−n6)·c11.  Enforcing EFL = f_target:
            nu_10 − phi_11 · y_10 = −P   where P = 1/f_target
            phi_11 = (nu_10 + P) / y_10
            c11    = phi_11 / (1 − n6)
        BFL after surface 11:
            bfl = y_10 · f_target   (exact: −y_10 / (−P))

        Returns: (c11, d_sensor, bfl) — all scalar tensors, all differentiable.
        """
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        n4, n5, n6 = self._n4, self._n5, self._n6
        d01, d12, d23 = self._d01, self._d12, self._d23
        d34, d45     = self._d34, self._d45
        d5_to_7      = self._d5_to_7
        d78, d89     = self._d78, self._d89
        d9_10, d10_11 = self._d9_10, self._d10_11

        dtype = c_t.dtype
        dev = c_t.device
        y  = torch.ones(1,  dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        # Surface 0: air → n1
        nu = nu - (n1 - 1.0) * c_t[0] * y
        y  = y  + (d01 / n1) * nu
        # Surface 1: n1 → air
        nu = nu - (1.0 - n1) * c_t[1] * y
        y  = y  + d12 * nu
        # Surface 2: air → n2
        nu = nu - (n2 - 1.0) * c_t[2] * y
        y  = y  + (d23 / n2) * nu
        # Surface 3: n2 → air
        nu = nu - (1.0 - n2) * c_t[3] * y
        y  = y  + d34 * nu
        # Surface 4: air → n3
        nu = nu - (n3 - 1.0) * c_t[4] * y
        y  = y  + (d45 / n3) * nu
        # Surface 5: n3 → air  (absorbs stop gap)
        nu = nu - (1.0 - n3) * c_t[5] * y
        y  = y  + d5_to_7 * nu
        # Surface 7: air → n4
        nu = nu - (n4 - 1.0) * c_t[6] * y
        y  = y  + (d78 / n4) * nu
        # Surface 8: n4 → n5  (cemented interface)
        nu = nu - (n5 - n4) * c_t[7] * y
        y  = y  + (d89 / n5) * nu
        # Surface 9: n5 → air
        nu = nu - (1.0 - n5) * c_t[8] * y
        y  = y  + d9_10 * nu
        # Surface 10: air → n6
        nu = nu - (n6 - 1.0) * c_t[9] * y
        y  = y  + (d10_11 / n6) * nu

        # Now at exit of element-5 glass; next is surface 11 (n6 → air).
        P = 1.0 / self._f_target
        safe_y = torch.where(torch.abs(y) > 1e-6, y, torch.full_like(y, 1e-6))
        phi_11 = (nu + P) / safe_y
        c11 = phi_11 / (1.0 - n6)
        bfl = y * float(self._f_target)
        d_sensor = self._z_s11 + bfl.squeeze()
        return c11.squeeze(), d_sensor, bfl.squeeze()

    def _paraxial_efl(self, c_t):
        """Paraxial effective focal length (free mode, 11-D theta)."""
        torch = self._torch
        n1, n2, n3 = self._n1, self._n2, self._n3
        n4, n5, n6 = self._n4, self._n5, self._n6
        d01, d12, d23 = self._d01, self._d12, self._d23
        d34, d45 = self._d34, self._d45
        d5_to_7 = self._d5_to_7
        d78, d89 = self._d78, self._d89
        d9_10, d10_11 = self._d9_10, self._d10_11

        dtype, dev = c_t.dtype, c_t.device
        y = torch.ones(1, dtype=dtype, device=dev)
        nu = torch.zeros(1, dtype=dtype, device=dev)

        nu = nu - (n1 - 1.0) * c_t[0] * y;  y = y + (d01 / n1) * nu
        nu = nu - (1.0 - n1) * c_t[1] * y;  y = y + d12 * nu
        nu = nu - (n2 - 1.0) * c_t[2] * y;  y = y + (d23 / n2) * nu
        nu = nu - (1.0 - n2) * c_t[3] * y;  y = y + d34 * nu
        nu = nu - (n3 - 1.0) * c_t[4] * y;  y = y + (d45 / n3) * nu
        nu = nu - (1.0 - n3) * c_t[5] * y;  y = y + d5_to_7 * nu
        nu = nu - (n4 - 1.0) * c_t[6] * y;  y = y + (d78 / n4) * nu
        nu = nu - (n5 - n4) * c_t[7] * y;   y = y + (d89 / n5) * nu
        nu = nu - (1.0 - n5) * c_t[8] * y;  y = y + d9_10 * nu
        nu = nu - (n6 - 1.0) * c_t[9] * y;  y = y + (d10_11 / n6) * nu
        nu = nu - (1.0 - n6) * c_t[10] * y
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
        """Apply theta_t to the reusable lens in-place; return it.

        Fixed-EFL mode (focal_length_target_mm is set):
            theta_t = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10]  (10-D)
            c11 and d_sensor derived from _paraxial_fixed_efl.
        Free mode (focal_length_target_mm is None):
            theta_t = [c0,c1,c2,c3,c4,c5,c7,c8,c9,c10,c11]  (11-D)
            d_sensor from _paraxial_bfl.
        """
        if self._f_target is not None:
            c11, d_sensor, _ = self._paraxial_fixed_efl(theta_t)
            curvatures = [theta_t[k] for k in range(10)] + [c11]
        else:
            d_sensor = self._paraxial_bfl(theta_t)
            curvatures = [theta_t[k] for k in range(11)]
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
        """Public API: build a GeoLens from a numpy theta vector."""
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
        N = self._fitness_aperture_xy.shape[0]
        dtype = lens.surfaces[self._SURF_IDX[0]].c.dtype

        origins = torch.empty((N, 3), dtype=dtype, device=self._device)
        origins[:, 0] = x_obj
        origins[:, 1] = y_obj
        origins[:, 2] = z_obj

        targets = torch.empty((N, 3), dtype=dtype, device=self._device)
        targets[:, :2] = self._fitness_aperture_xy.to(dtype=dtype)
        targets[:, 2] = self._pupil_z

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
        N = self._fitness_aperture_xy.shape[0]
        F = len(field_points)
        dtype = lens.surfaces[self._SURF_IDX[0]].c.dtype

        origins = torch.empty((F, N, 3), dtype=dtype, device=self._device)
        targets = torch.empty((F, N, 3), dtype=dtype, device=self._device)

        for i, (fx, fy) in enumerate(field_points):
            x_obj, y_obj, z_obj = self._make_object_point(fx, fy)
            origins[i, :, 0] = x_obj
            origins[i, :, 1] = y_obj
            origins[i, :, 2] = z_obj
            targets[i, :, :2] = self._fitness_aperture_xy.to(dtype=dtype)
            targets[i, :, 2] = self._pupil_z

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

        Rays start at the stop-aperture Fibonacci disk positions on the first
        glass-surface plane.  The missing incoming OPL (object → first surface)
        is restored in _zernike_measures.  float64 is required by DeepLens
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

        Pipeline identical to the triplet _zernike_measures:
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
    # Intersection penalty
    # ------------------------------------------------------------------

    def _intersection_penalty(self, lens):
        """Soft penalty for surfaces that are too close or intersecting."""
        torch = self._torch
        loss = torch.tensor(0.0, device=self._device)
        air_min = self.config.air_min_mm
        thick_min = self.config.thick_min_mm

        for i in range(len(lens.surfaces) - 1):
            curr = lens.surfaces[i]
            nxt = lens.surfaces[i + 1]

            # Center distance
            r0 = torch.tensor(0.0, device=self._device)
            z_curr = curr.surface_with_offset(r0, 0.0, valid_check=False)
            z_nxt = nxt.surface_with_offset(r0, 0.0, valid_check=False)
            dist_center = z_nxt - z_curr

            # Edge distances (16 radial samples from 0.5r to 1.0r)
            r_edge = torch.linspace(0.5, 1.0, 16, device=self._device) * curr.r
            z_curr_edge = curr.surface_with_offset(r_edge, 0.0, valid_check=False)
            z_nxt_edge = nxt.surface_with_offset(r_edge, 0.0, valid_check=False)
            dist_edge = torch.min(z_nxt_edge - z_curr_edge)

            is_air = (curr.mat2.name == "air")
            min_dist = air_min if is_air else thick_min

            if float(dist_center.detach()) < min_dist:
                loss = loss + (min_dist - dist_center)
            if float(dist_edge.detach()) < min_dist:
                loss = loss + (min_dist - dist_edge)

        return loss

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
            bfl_val = float((self._paraxial_bfl(theta_t) - self._z_s11).detach().item())
        if bfl_val <= 0.0:
            bad = theta_t.sum() * 0.0 - 1e9
            return bad, theta_t[0] * 0.0, theta_t[1] * 0.0, {
                "invalid_fraction": 1.0,
                "rejected": 1.0,
                "rejected_bfl": 1.0,
            }

        lens = self._build_lens_from_theta_tensor(theta_t)
        hfov = self._calc_hfov_deg(theta_t)

        # Compute entrance pupil for THIS lens configuration (paraxial, ~1% overhead).
        # Scales the unit-disk Fibonacci samples to the actual entrance pupil so
        # fitness rays fill the full beam without excessive invalid-ray waste.
        with torch.no_grad():
            pupilz, pupilr = lens.calc_entrance_pupil(paraxial=True)

        # Strict mode: if calc_entrance_pupil silently fell back to surfaces[0]
        # (front element semi-diameter), the resulting pupil is meaningless.
        # Reject the design with the same -1e9 penalty pattern used for
        # BFL-invalid designs above (lines 937-939).  Detection: the fallback
        # branch in DeepLens returns exactly (surfaces[0].d.item(), surfaces[0].r);
        # both values are stable across curvature mutations, so an exact float
        # compare reliably catches the fallback.
        if self.config.strict_entrance_pupil:
            fb_z = float(lens.surfaces[0].d.item())
            fb_r = float(lens.surfaces[0].r)
            if (abs(float(pupilr) - fb_r) < 1e-9
                    and abs(float(pupilz) - fb_z) < 1e-9):
                bad = theta_t.sum() * 0.0 - 1e9
                return bad, theta_t[0] * 0.0, theta_t[1] * 0.0, {
                    "invalid_fraction": 1.0,
                    "rejected": 1.0,
                    "rejected_pupil": 1.0,
                }

        self._fitness_aperture_xy = self._fitness_unit_disk * float(pupilr)
        self._pupil_z = float(pupilz)

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

        # Intersection penalty (soft)
        intersect_loss = self._intersection_penalty(lens)
        if float(intersect_loss.detach()) > 0:
            fitness = fitness - self.config.intersect_penalty_weight * intersect_loss

        # Invalid-fraction penalty (proportional to excess above tolerance)
        inv_frac_val = float(invalid_fraction_mean.detach().item())
        if inv_frac_val > self.config.invalid_tol:
            excess = invalid_fraction_mean - self.config.invalid_tol
            fitness = fitness - self.config.invalid_penalty_weight * excess

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

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> EF50EvalResult:
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

        return EF50EvalResult(
            theta=theta_proj.astype(np.float64, copy=False),
            fitness=float(fitness.detach().cpu().item()),
            measures=(float(m0.detach().cpu().item()), float(m1.detach().cpu().item())),
            metrics=metrics,
            grads=grads,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EF50ZernikeRunner:
    """CMA-MEGA runner for the EF50mm Zernike problem."""

    def __init__(self, config: EF50ZernikeConfig):
        self.config = config
        self.problem = EF50ZernikeProblem(config)
        self.archive: Optional[EF50Archive2D] = None
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
        self.result_dir: Optional[Path] = None
        # Long-lived multiprocessing pool for parallel candidate evaluation.
        # Built lazily inside run() so seed_archive() keeps using its own
        # per-call _parallel_evaluate path. None until run() starts.
        self._eval_pool = None

    def calibrate_archive_bounds(self):
        # PARALLEL VARIANT: dispatches calibration evaluations to the
        # multi-GPU multiprocessing pipeline from seed_parallel_experiment.
        from optical_rl.qd.seed_parallel_experiment import (
            _parallel_evaluate, _config_to_dict,
            _available_gpus, _pick_worker_count,
        )

        samples = self.problem.sample_theta(self.config.calibration_count, self.config.cma_seed + 17)
        config_dict = _config_to_dict(self.config)
        gpu_ids = _available_gpus()
        num_workers = _pick_worker_count(self.config.parallel_workers, len(gpu_ids))
        results = _parallel_evaluate(
            self.config, config_dict, samples, num_workers, gpu_ids,
            progress_label="ef50-zernike calibration",
            progress_every=500,
        )
        # Skip rejected designs (BFL <= 0 or strict-mode entrance pupil
        # failure) — their (0, 0) measures would skew percentile bounds.
        m0_vals, m1_vals = [], []
        for r in results:
            if r is None:
                continue
            if r["metrics"].get("rejected", 0.0) > 0.0:
                continue
            m0_vals.append(r["measures"][0])
            m1_vals.append(r["measures"][1])

        def _bounds(arr):
            lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
            span = max(hi - lo, 1e-6)
            return (lo - 0.05 * span, hi + 0.05 * span)

        bounds = (
            _bounds(np.asarray(m0_vals, dtype=np.float64)),
            _bounds(np.asarray(m1_vals, dtype=np.float64)),
        )
        self.archive = EF50Archive2D(self.config.archive_bins, bounds, seed=self.config.cma_seed)
        return bounds

    def _maybe_log_seed_accept(self, inserted: bool, result: EF50EvalResult) -> None:
        if not inserted or not self.config.log_seed_accepts:
            return
        s = self.archive.stats()
        print(
            f"[ef50-zernike] seed_accept fitness={result.fitness:.6f} "
            f"measures=({result.measures[0]:.6f}, {result.measures[1]:.6f}) "
            f"filled={int(s['filled'])} coverage={s['coverage']:.4f} "
            f"best={s['best_fitness']:.6f}",
            flush=True,
        )

    def seed_archive(self) -> Dict[str, float]:
        # PARALLEL VARIANT: dispatches all seeding evaluations to the multi-GPU
        # multiprocessing pipeline from seed_parallel_experiment.  Results are
        # inserted into the archive in original index order so behavior matches
        # the sequential version (modulo minor float jitter from CUDA).
        from optical_rl.qd.seed_parallel_experiment import (
            _parallel_evaluate, _config_to_dict,
            _available_gpus, _pick_worker_count,
        )

        if self.archive is None:
            if self.config.archive_bounds is not None:
                self.archive = EF50Archive2D(
                    self.config.archive_bins,
                    self.config.archive_bounds,
                    seed=self.config.cma_seed,
                )
            else:
                self.calibrate_archive_bounds()

        flat = self.problem.initial_theta()
        all_thetas = np.vstack([
            flat.reshape(1, -1),
            self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31),
        ])

        config_dict = _config_to_dict(self.config)
        gpu_ids = _available_gpus()
        num_workers = _pick_worker_count(self.config.parallel_workers, len(gpu_ids))
        results = _parallel_evaluate(
            self.config, config_dict, all_thetas, num_workers, gpu_ids,
            progress_label="ef50-zernike seeding",
            progress_every=500,
        )

        for r in results:
            if r is None:
                continue
            # Skip rejected designs entirely — do not insert into the archive.
            if r["metrics"].get("rejected", 0.0) > 0.0:
                continue
            eval_result = EF50EvalResult(
                theta=r["theta"],
                fitness=r["fitness"],
                measures=r["measures"],
                metrics=r["metrics"],
                grads=None,
            )
            self._maybe_log_seed_accept(self.archive.insert(eval_result), eval_result)
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
        """Canonical CMA-MEGA loop with restart on no-improvement,
        three-tier improvement ranking, periodic checkpointing, and a
        per-generation trajectory CSV.

        If `resume` is True the runner assumes self.archive, self.emitter,
        self.completed, and self.generation have already been restored
        from a checkpoint, and it skips the seed/calibrate prelude. The
        eval budget is interpreted as the *additional* number of fitness
        evaluations beyond the saved completed count.
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
        from optical_rl.qd._eval_pool import PersistentEvalPool
        from optical_rl.qd.seed_parallel_experiment import (
            _config_to_dict,
            _available_gpus,
            _pick_worker_count,
        )

        if not resume and (self.archive is None or not self.archive.cells):
            self.seed_archive()

        # Build the long-lived CMA-MEGA worker pool exactly once per run().
        # Seeding above kept its own per-call _parallel_evaluate (workers
        # torn down after the one big batch); the CMA-MEGA inner loop calls
        # the pool once per generation for hundreds of generations, so the
        # workers must persist or the spawn cost dominates wallclock.
        gpu_ids = _available_gpus()
        if not gpu_ids:
            raise RuntimeError(
                "PersistentEvalPool requires at least one CUDA device; "
                "none were detected"
            )
        num_workers = _pick_worker_count(self.config.parallel_workers, len(gpu_ids))
        self._eval_pool = PersistentEvalPool(
            self.config,
            _config_to_dict(self.config),
            num_workers=num_workers,
            gpu_ids=gpu_ids,
        )
        print(
            f"[ef50-zernike] CMA-MEGA worker pool: {num_workers} workers "
            f"across GPUs {gpu_ids}",
            flush=True,
        )

        j0, j1 = self.config.descriptor_zernike_indices
        theta_dim = len(self.config.theta_bounds)
        scores: list = []
        eval_budget = int(iterations)
        # Resume continues from saved counters; fresh runs start at 0.
        if not resume:
            self.completed = 0
            self.generation = 0
            self.restart_count = 0
            self.wallclock_seconds = 0.0
        target_completed = self.completed + eval_budget

        # ---- checkpoint + trajectory infra ----
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
            # Also dump archive_cells.json so refine_ef50_archive.py and
            # render_ef50_zernike.py work on a live run with no code changes.
            # Atomic write via .tmp + os.replace, same pattern as the
            # checkpoint pickle, so a SIGINT mid-write cannot corrupt it.
            cells_data = []
            for cell_idx, result in self.archive.cells.items():
                cells_data.append({
                    "cell_index": list(cell_idx),
                    "theta": result.theta.tolist(),
                    "fitness": float(result.fitness),
                    "measures": list(result.measures),
                    "metrics": result.metrics,
                })
            json_path = Path(result_dir) / "archive_cells.json"
            tmp_json = json_path.with_suffix(".json.tmp")
            with open(tmp_json, "w") as jf:
                json.dump({
                    "archive_bounds": self.archive.bounds,
                    "archive_bins": self.archive.bins,
                    "cells": cells_data,
                    "stats": self.archive.stats(),
                }, jf, indent=2)
            os.replace(tmp_json, json_path)

        # ---- SIGINT handler: save once and re-raise ----
        sigint_received = {"flag": False}
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):  # noqa: ARG001
            sigint_received["flag"] = True
            print(
                "[ef50-zernike] SIGINT received — checkpointing then exiting.",
                flush=True,
            )

        signal.signal(signal.SIGINT, _handle_sigint)
        loop_t0 = time.perf_counter()

        # --- per-phase timing instrumentation ---
        # Accumulates wallclock spent in each CMA-MEGA phase across the last
        # `log_every` generations, printed alongside the gen log. Uses
        # torch.cuda.synchronize() at phase boundaries so GPU async work is
        # attributed to the phase that launched it rather than the next one.
        _torch_ref = getattr(self.problem, "_torch", None)
        _cuda_available = bool(
            _torch_ref is not None and _torch_ref.cuda.is_available()
        )

        def _sync_cuda() -> None:
            if _cuda_available:
                _torch_ref.cuda.synchronize()

        t_parent_acc = 0.0
        t_build_acc = 0.0
        t_pool_acc = 0.0
        t_insert_acc = 0.0
        t_total_acc = 0.0
        gens_since_timing_log = 0

        try:
            while self.completed < target_completed:
                if sigint_received["flag"]:
                    break
                _sync_cuda()
                t_gen_start = time.perf_counter()
                # Defense-in-depth: if the sampled parent turns out to be a
                # rejected design (e.g. loaded from an old archive.pkl that
                # predates strict-mode filtering), retry up to a few times.
                # With strict mode on and fresh seeding this branch never fires.
                parent_eval = None
                for _attempt in range(10):
                    parent_theta = self.archive.sample_parent()
                    parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
                    if parent_eval.metrics.get("rejected", 0.0) <= 0.0:
                        break
                    parent_eval = None
                if parent_eval is None:
                    print("[ef50-zernike] Warning: archive contains only rejected "
                          "designs, aborting CMA-MEGA loop.", flush=True)
                    break
                assert parent_eval.grads is not None
                _sync_cuda()
                t_after_parent = time.perf_counter()

                unit_parent = _theta_to_unit(parent_theta, self.config.theta_bounds)
                grads = self._normalized_gradients(parent_eval.grads)
                batch_n = min(int(self.config.population_size), target_completed - self.completed)
                coeffs = self.emitter.ask(batch_n)

                # Pass 1: build all candidate thetas (cheap, no I/O).
                candidate_thetas = np.empty((batch_n, theta_dim), dtype=np.float64)
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
                    candidate_thetas[local_idx] = _unit_to_theta(
                        candidate_unit, self.config.theta_bounds
                    )
                t_after_build = time.perf_counter()

                # Pass 2: parallel evaluation across the persistent worker
                # pool. One call per generation; workers stay alive across
                # generations so the per-call overhead is just queue I/O.
                pool_results = self._eval_pool.evaluate(candidate_thetas)
                _sync_cuda()
                t_after_pool = time.perf_counter()

                # Pass 3: archive insert + scoring (cheap, sequential).
                # Order matches Pass 1 so coeffs and scores align.
                batch_scores: list = []
                batch_status: list = []  # INSERT_{NONE,REPLACED,NEW}
                for local_idx, r in enumerate(pool_results):
                    if r is None:
                        # Worker error — penalise so CMA learns to avoid
                        # the direction, but do not insert into the archive.
                        scores.append(-1e9)
                        batch_scores.append(-np.inf)
                        batch_status.append(0)
                        continue
                    candidate_eval = EF50EvalResult(
                        theta=r["theta"],
                        fitness=r["fitness"],
                        measures=r["measures"],
                        metrics=r["metrics"],
                        grads=None,
                    )
                    # Skip rejected candidates — do not insert into the archive.
                    # Their fitness still feeds back to the CMA emitter via
                    # batch_scores so the distribution learns to avoid them.
                    if candidate_eval.metrics.get("rejected", 0.0) > 0.0:
                        status = 0  # INSERT_NONE
                    else:
                        status = int(self.archive.insert(candidate_eval))
                    scores.append(candidate_eval.fitness)
                    if status == INSERT_NEW:
                        # Bonus pushes new cells above any replaced elite.
                        batch_scores.append(candidate_eval.fitness + new_cell_bonus)
                    elif status == INSERT_REPLACED:
                        batch_scores.append(candidate_eval.fitness)
                    else:
                        batch_scores.append(-np.inf)
                    batch_status.append(status)

                self.emitter.tell(coeffs, np.asarray(batch_scores, dtype=np.float64))
                t_after_insert = time.perf_counter()

                t_parent_acc += (t_after_parent - t_gen_start)
                t_build_acc += (t_after_build - t_after_parent)
                t_pool_acc += (t_after_pool - t_after_build)
                t_insert_acc += (t_after_insert - t_after_pool)
                t_total_acc += (t_after_insert - t_gen_start)
                gens_since_timing_log += 1

                self.completed += batch_n
                self.generation += 1

                # ---- per-gen telemetry & restart logic ----
                n_new = sum(1 for s in batch_status if s == INSERT_NEW)
                n_replaced = sum(1 for s in batch_status if s == INSERT_REPLACED)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted = sum(1 for i in elite_idx if batch_status[i] == 0)

                # Canonical CMA-MEGA QD restart trigger: zero archive
                # improvement this generation, OR Hansen termination fired.
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
                        f"[ef50-zernike] [restart] gen={self.generation} "
                        f"reason={restart_reason} sigma_pre={sigma_pre:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                # Update wallclock total before logging / checkpointing.
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
                        f"[ef50-zernike] gen={self.generation} evals={self.completed} "
                        f"coverage={s['coverage']:.4f} best={s['best_fitness']:.6f} | "
                        f"new={n_new}/{batch_n} replaced={n_replaced}/{batch_n} "
                        f"wasted={n_wasted}/{elite_n} "
                        f"sigma={self.emitter.sigma:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                    if gens_since_timing_log > 0:
                        g = gens_since_timing_log
                        mean_parent = t_parent_acc / g
                        mean_build = t_build_acc / g
                        mean_pool = t_pool_acc / g
                        mean_insert = t_insert_acc / g
                        mean_total = t_total_acc / g
                        pct_parent = 100.0 * t_parent_acc / t_total_acc if t_total_acc > 0 else 0.0
                        pct_pool = 100.0 * t_pool_acc / t_total_acc if t_total_acc > 0 else 0.0
                        evals_per_s = (g * batch_n) / t_total_acc if t_total_acc > 0 else 0.0
                        print(
                            f"[ef50-zernike] timing gen={self.generation} "
                            f"avg/gen over last {g} gens (ms): "
                            f"parent={mean_parent*1e3:7.1f} "
                            f"build={mean_build*1e3:6.2f} "
                            f"pool={mean_pool*1e3:7.1f} "
                            f"insert={mean_insert*1e3:6.2f} "
                            f"total={mean_total*1e3:7.1f} | "
                            f"parent%={pct_parent:5.1f} pool%={pct_pool:5.1f} "
                            f"throughput={evals_per_s:6.1f} evals/s",
                            flush=True,
                        )
                        t_parent_acc = 0.0
                        t_build_acc = 0.0
                        t_pool_acc = 0.0
                        t_insert_acc = 0.0
                        t_total_acc = 0.0
                        gens_since_timing_log = 0

                if checkpoint_every > 0 and self.generation % checkpoint_every == 0:
                    _flush_checkpoint()
        finally:
            # Always save a final checkpoint, tear down the worker pool,
            # and restore the prior signal handler. Each step is wrapped
            # so a failure in one does not block the others.
            try:
                _flush_checkpoint()
            except Exception as e:  # pragma: no cover
                print(f"[ef50-zernike] WARNING: final checkpoint failed: {e}", flush=True)
            try:
                if self._eval_pool is not None:
                    self._eval_pool.shutdown()
                    self._eval_pool = None
            except Exception as e:  # pragma: no cover
                print(f"[ef50-zernike] WARNING: pool shutdown failed: {e}", flush=True)
            signal.signal(signal.SIGINT, prev_handler)

        s = self.archive.stats()
        s["iterations"] = float(eval_budget)
        s["generations"] = float(self.generation)
        s["restart_count"] = float(self.restart_count)
        s["mean_score"] = float(np.mean(scores)) if scores else float("nan")
        return s


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_config(yaml_path: str) -> EF50ZernikeConfig:
    """Load an EF50ZernikeConfig from a YAML file (qd: section)."""
    import yaml  # type: ignore

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    qd = raw.get("qd", raw)

    def _get(key, default):
        return qd[key] if key in qd else default

    theta_bounds_raw = _get("theta_bounds", None)
    if theta_bounds_raw is not None:
        theta_bounds = tuple(tuple(b) for b in theta_bounds_raw)
    else:
        theta_bounds = EF50ZernikeConfig.theta_bounds

    archive_bounds_raw = _get("archive_bounds", None)
    if archive_bounds_raw is not None:
        archive_bounds = tuple(tuple(b) for b in archive_bounds_raw)
    else:
        archive_bounds = None

    desc_raw = _get("descriptor_zernike_indices", None)
    descriptor_zernike_indices = tuple(int(j) for j in desc_raw) if desc_raw else (7, 11)

    archive_bins_raw = _get("archive_bins", None)
    archive_bins = tuple(int(b) for b in archive_bins_raw) if archive_bins_raw else (64, 64)

    fl_target = _get("focal_length_target_mm", 50.52)
    fl_target = float(fl_target) if fl_target is not None else None

    return EF50ZernikeConfig(
        lens_path=str(_get("lens_path", EF50ZernikeConfig.lens_path)),
        object_depth_mm=float(_get("object_depth_mm", -20000.0)),
        wavelength_um=float(_get("wavelength_um", 0.587)),
        theta_bounds=theta_bounds,
        stop_surface_index=int(_get("stop_surface_index", 6)),
        focal_length_target_mm=fl_target,
        descriptor_zernike_indices=descriptor_zernike_indices,
        archive_bins=archive_bins,
        archive_bounds=archive_bounds,
        invalid_tol=float(_get("invalid_tol", 0.15)),
        invalid_penalty_weight=float(_get("invalid_penalty_weight", 5.0)),
        cma_eta=float(_get("cma_eta", 0.05)),
        seed_count=int(_get("seed_count", 128)),
        calibration_count=int(_get("calibration_count", 512)),
        num_rays=int(_get("num_rays", 512)),
        fitness_wavelengths_um=tuple(
            float(x) for x in _get("fitness_wavelengths_um", [0.587])
        ),
        iterations=int(_get("iterations", 1000)),
        stage=str(_get("stage", "stage1")),
        zernike_field_index=int(_get("zernike_field_index", 0)),
        intersect_penalty_weight=float(_get("intersect_penalty_weight", 1.0)),
        air_min_mm=float(_get("air_min_mm", 0.1)),
        thick_min_mm=float(_get("thick_min_mm", 1.0)),
        device=str(_get("device", "auto")),
        initial_sigma=float(_get("initial_sigma", 0.75)),
        population_size=int(_get("population_size", 16)),
        cma_seed=int(_get("seed", 0)),
        results_dir=str(_get("results_dir", "results/models")),
        run_name=str(_get("run_name", "ef50_zernike")),
        log_every=int(_get("log_every", 50)),
        log_seed_accepts=bool(_get("log_seed_accepts", True)),
        strict_entrance_pupil=bool(_get("strict_entrance_pupil", False)),
        parallel_workers=(
            int(_get("parallel_workers", None))
            if _get("parallel_workers", None) is not None else None
        ),
        checkpoint_every=int(_get("checkpoint_every", 50)),
        new_cell_bonus=float(_get("new_cell_bonus", 1.0e6)),
    )


# ---------------------------------------------------------------------------
# Archive save / load
# ---------------------------------------------------------------------------

def _save_archive(archive: EF50Archive2D, path: Path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(archive, f)


def _load_archive(path: Path) -> EF50Archive2D:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_archive_plot(
    path: Path, runner: "EF50ZernikeRunner", config: EF50ZernikeConfig
) -> None:
    assert runner.archive is not None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception:
        return

    j0, j1 = config.descriptor_zernike_indices
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

    valid = neg_grid[np.isfinite(neg_grid) & (neg_grid > 0)]
    norm = LogNorm(vmin=float(valid.min()), vmax=float(valid.max())) if valid.size > 0 else None
    im = axes[0].imshow(neg_grid.T, origin="lower", aspect="auto", norm=norm)
    axes[0].set_title("Archive Fitness Heatmap")
    axes[0].set_xlabel("Measure bin 0")
    axes[0].set_ylabel("Measure bin 1")
    fig.colorbar(im, ax=axes[0], shrink=0.85).set_label("-fitness (log)")

    if xs:
        pos_cs = [v for v in neg_cs if v > 0]
        scatter_norm = LogNorm(vmin=min(pos_cs), vmax=max(pos_cs)) if pos_cs else None
        sc = axes[1].scatter(xs, ys, c=neg_cs, s=24, norm=scatter_norm)
        (x_lo, x_hi), (y_lo, y_hi) = runner.archive.bounds
        axes[1].set_xlim(x_lo, x_hi)
        axes[1].set_ylim(y_lo, y_hi)
        fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("-fitness (log)")
    axes[1].set_title("Archive (EF50 Zernike)")
    axes[1].set_xlabel(label0)
    axes[1].set_ylabel(label1)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[ef50-zernike] Archive plot saved: {path}", flush=True)


def main() -> None:
    import argparse
    import pickle

    # Suppress DeepLens entrance-pupil fallback warnings in this (parent)
    # process.  Workers install the same filter independently at startup.
    # Set DEEPLENS_KEEP_WARNINGS=1 in env to disable.
    from optical_rl.qd.seed_parallel_experiment import install_deeplens_warning_filter
    install_deeplens_warning_filter()

    parser = argparse.ArgumentParser(description="EF50mm f/1.8 QD CMA-MEGA with Zernike descriptors")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (qd: section)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iterations from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing run directory to resume from")
    args = parser.parse_args()

    if args.config is not None:
        cfg = _load_config(args.config)
    else:
        cfg = EF50ZernikeConfig()

    iterations = args.iterations if args.iterations is not None else cfg.iterations

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name
    out_dir = Path(cfg.results_dir) / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    import shutil
    if args.config is not None:
        shutil.copy(args.config, out_dir / Path(args.config).name)

    runner = EF50ZernikeRunner(cfg)
    runner.result_dir = out_dir

    resumed = False
    if args.resume is not None:
        from optical_rl.qd._checkpoint import hash_config, load_checkpoint
        resume_dir = Path(args.resume)
        ckpt_path = resume_dir / "checkpoint.pkl"
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
                    f"[ef50-zernike] WARNING: config hash drift on resume "
                    f"(checkpoint={ckpt.config_hash[:12]}, "
                    f"current={current_hash[:12]}). Continuing anyway.",
                    flush=True,
                )
            print(
                f"[ef50-zernike] Resumed from {ckpt_path}: "
                f"completed={runner.completed} gen={runner.generation} "
                f"restarts={runner.restart_count} "
                f"best={runner.archive.stats()['best_fitness']:.6f}",
                flush=True,
            )
            resumed = True
        else:
            # Legacy fallback: old runs only saved archive.pkl. Loading
            # gets us the cells but not the emitter; warn loudly so the
            # user knows the resume is partial.
            archive_path = resume_dir / "archive.pkl"
            if archive_path.exists():
                runner.archive = _load_archive(archive_path)
                print(
                    f"[ef50-zernike] WARNING: legacy resume (no checkpoint.pkl). "
                    f"Loaded archive from {archive_path} but the CMA emitter "
                    f"is starting fresh. completed counter reset to 0.",
                    flush=True,
                )
                resumed = True
            else:
                print(
                    f"[ef50-zernike] No checkpoint.pkl or archive.pkl in "
                    f"{resume_dir}, starting fresh.",
                    flush=True,
                )

    print(f"[ef50-zernike] Output dir: {out_dir}", flush=True)
    if not resumed:
        print(f"[ef50-zernike] Seeding archive …", flush=True)
        seed_stats = runner.seed_archive()
        print(f"[ef50-zernike] Seed stats: {seed_stats}", flush=True)

    print(f"[ef50-zernike] Running {iterations} evaluations …", flush=True)
    final_stats = runner.run(iterations, resume=resumed)
    print(f"[ef50-zernike] Final stats: {final_stats}", flush=True)

    # Save archive
    _save_archive(runner.archive, out_dir / "archive.pkl")

    # Save archive cells as JSON
    cells_data = []
    for cell_idx, result in runner.archive.cells.items():
        cells_data.append({
            "cell_index": list(cell_idx),
            "theta": result.theta.tolist(),
            "fitness": float(result.fitness),
            "measures": list(result.measures),
            "metrics": result.metrics,
        })
    with open(out_dir / "archive_cells.json", "w") as f:
        json.dump({
            "archive_bounds": runner.archive.bounds,
            "archive_bins": runner.archive.bins,
            "cells": cells_data,
            "stats": final_stats,
        }, f, indent=2)

    # Archive population plot
    _save_archive_plot(out_dir / "archive_map.png", runner, cfg)

    print(f"[ef50-zernike] Saved archive to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
