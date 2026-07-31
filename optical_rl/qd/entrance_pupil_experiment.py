"""Entrance pupil fallback experiment for EF50 Zernike.

Standalone script — does NOT modify ef50_zernike.py, geolens.py, or any existing code.

Compares three evaluation strategies on the SAME batch of random thetas:
  A) baseline  – current code (silent fallback to surfaces[0].r on pupil failure)
  B) abcd      – ABCD paraxial matrix entrance pupil (front-group only), fallback
                 to penalty if that also fails
  C) flagged   – same as baseline, but individuals where fallback triggered get
                 a heavy penalty instead of a misleading fitness score

Reuses the fast multiprocessing infrastructure from seed_parallel_experiment.py.

Usage:
    python -m optical_rl.qd.entrance_pupil_experiment \
        --config optical_rl/configs/ef50_zernike.yaml \
        --seeds 1000 --workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (copied from seed_parallel_experiment to keep this standalone)
# ---------------------------------------------------------------------------

def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _available_gpus() -> list[int]:
    import torch
    return list(range(torch.cuda.device_count()))


def _pick_worker_count(requested: int | None, num_gpus: int) -> int:
    available_cpus = _available_cpu_count()
    if requested is not None:
        return max(1, min(requested, available_cpus))
    auto = num_gpus * 6
    return max(2, min(auto, available_cpus // 2, 32))


def _tuple_to_list(t):
    if isinstance(t, tuple):
        return tuple(_tuple_to_list(x) for x in t)
    return t


def _config_to_dict(config) -> dict:
    d = {}
    for field in config.__dataclass_fields__:
        val = getattr(config, field)
        if isinstance(val, tuple):
            val = _tuple_to_list(val)
        d[field] = val
    return d


def _percentile_bounds(m0_vals, m1_vals):
    def _bounds(arr):
        lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
        span = max(hi - lo, 1e-6)
        return (lo - 0.05 * span, hi + 0.05 * span)
    return (
        _bounds(np.asarray(m0_vals, dtype=np.float64)),
        _bounds(np.asarray(m1_vals, dtype=np.float64)),
    )


# ---------------------------------------------------------------------------
# ABCD front-group entrance pupil calculation
# ---------------------------------------------------------------------------

def _abcd_entrance_pupil(problem, c_t):
    """Compute entrance pupil via ABCD paraxial matrix of the FRONT GROUP ONLY
    (surfaces 0–5, before the aperture stop at surface 6).

    Traces a marginal ray [y=1, nu=0] through the front group.  At the stop
    plane the ray height is y_stop and slope is nu_stop.  The entrance pupil
    is where backward extension of this ray crosses the axis, i.e. the image
    of the stop through the front group reversed.

    Returns (pupilz, pupilr) or None if degenerate.
    """
    import torch

    n1, n2, n3 = problem._n1, problem._n2, problem._n3
    d01, d12 = problem._d01, problem._d12
    d23, d34 = problem._d23, problem._d34
    d45 = problem._d45

    # For the front group we need curvatures c0..c5.
    # In fixed-EFL mode c_t has 10 entries: [c0,c1,c2,c3,c4,c5, c7,c8,c9,c10]
    # So c0..c5 are c_t[0]..c_t[5].

    dtype = c_t.dtype
    dev = c_t.device
    y = torch.ones(1, dtype=dtype, device=dev)
    nu = torch.zeros(1, dtype=dtype, device=dev)

    # Surface 0: air → n1
    nu = nu - (n1 - 1.0) * c_t[0] * y
    y = y + (d01 / n1) * nu

    # Surface 1: n1 → air
    nu = nu - (1.0 - n1) * c_t[1] * y
    y = y + d12 * nu

    # Surface 2: air → n2
    nu = nu - (n2 - 1.0) * c_t[2] * y
    y = y + (d23 / n2) * nu

    # Surface 3: n2 → air
    nu = nu - (1.0 - n2) * c_t[3] * y
    y = y + d34 * nu

    # Surface 4: air → n3
    nu = nu - (n3 - 1.0) * c_t[4] * y
    y = y + (d45 / n3) * nu

    # Surface 5: n3 → air
    nu = nu - (1.0 - n3) * c_t[5] * y

    # Now (y, nu) is at surface 5 exit in air.
    # Transfer to stop plane (surface 6).  d5_to_7 includes the stop gap,
    # but the stop is between surface 5 and surface 7.
    # The stop z-position is at surfaces[6].d in the JSON.
    # From the JSON: surfaces[5].d = 13.57, surfaces[6].d = 20.32, surfaces[7].d = 25.27
    # So the gap from surface 5 to the stop is surfaces[6].d - surfaces[5].d.
    # We stored d5_to_7 = d[7] - d[5] = 25.27 - 13.57 = 11.70
    # But we need d5_to_stop = d[6] - d[5].
    # d[6] = 20.32, d[5] = 13.57 → gap = 6.75
    # This is surfaces[5].d_next = 6.75 in the JSON.
    # We can compute: d5_to_stop = d5_to_7 - (d[7] - d[6])
    # d[7] - d[6] = surfaces[6].d_next = 4.95
    # So d5_to_stop = 11.70 - 4.95 = 6.75  ✓
    #
    # But we don't have d[6]-d[5] directly stored on the problem.
    # We DO have d5_to_7.  The stop is at the middle.  Let's just use the
    # template lens surfaces to get the exact z positions.
    lens = problem._template_lens
    d5 = float(lens.surfaces[5].d.item())
    d6 = float(lens.surfaces[6].d.item())
    d5_to_stop = d6 - d5

    y = y + d5_to_stop * nu  # ray height at stop

    # y_stop is the marginal ray height at the stop when launched with y=1 at
    # the front surface.  The stop radius is aper_r.  The entrance pupil radius
    # is aper_r / |y_stop| (since the marginal ray maps 1 → y_stop, the stop
    # maps aper_r → aper_r / y_stop in object space).
    #
    # Actually, more precisely: the entrance pupil is the image of the stop
    # through the front group (reversed).  If a marginal ray [y=1, nu=0]
    # arrives at the stop with height y_stop and slope nu_stop (= nu/1 since
    # we're in air), then:
    #   - entrance pupil position z_ep = -y_stop / nu_stop (from the first surface)
    #     ... actually we need the ABCD matrix to invert this properly.
    #
    # Simpler approach: the front-group ABCD matrix M maps [y_in, nu_in] → [y_stop, nu_stop].
    #   M = [[A, B], [C, D]]
    # where y_stop = A*y_in + B*nu_in, nu_stop = C*y_in + D*nu_in.
    #
    # From [y=1, nu=0]: A = y_stop, C = nu_stop
    # From [y=0, nu=1]: B and D (we'd need a second ray trace).
    #
    # The entrance pupil is the image of the stop through M^{-1}.  For an
    # object at the stop plane, the image through the reversed front group is:
    #   z_ep (from surface 0) = -A / C = -y_stop / nu_stop
    #   magnification = 1 / (A - B*C/... ) ... this gets complicated.
    #
    # Let's use the two-ray approach: trace [1,0] and [0,1], build the full
    # ABCD matrix, then invert.

    # We already have the [1,0] ray result.  Save it.
    y_stop_from_10 = float(y.item())
    nu_stop_from_10 = float(nu.item())

    # Now trace [y=0, nu=1] through the same front group.
    y2 = torch.zeros(1, dtype=dtype, device=dev)
    nu2 = torch.ones(1, dtype=dtype, device=dev)

    nu2 = nu2 - (n1 - 1.0) * c_t[0] * y2;  y2 = y2 + (d01 / n1) * nu2
    nu2 = nu2 - (1.0 - n1) * c_t[1] * y2;  y2 = y2 + d12 * nu2
    nu2 = nu2 - (n2 - 1.0) * c_t[2] * y2;  y2 = y2 + (d23 / n2) * nu2
    nu2 = nu2 - (1.0 - n2) * c_t[3] * y2;  y2 = y2 + d34 * nu2
    nu2 = nu2 - (n3 - 1.0) * c_t[4] * y2;  y2 = y2 + (d45 / n3) * nu2
    nu2 = nu2 - (1.0 - n3) * c_t[5] * y2
    y2 = y2 + d5_to_stop * nu2

    y_stop_from_01 = float(y2.item())
    nu_stop_from_01 = float(nu2.item())

    # ABCD matrix of front group (surface 0 to stop):
    #   [y_stop]   [A  B] [y_in ]
    #   [nu_stop] = [C  D] [nu_in]
    A = y_stop_from_10    # from [1, 0]
    C = nu_stop_from_10
    B = y_stop_from_01    # from [0, 1]
    D = nu_stop_from_01

    det = A * D - B * C  # should be ~1.0 for a lossless system

    # The entrance pupil is the image of the stop through the INVERSE of the
    # front group.  M^{-1} = (1/det) * [[D, -B], [-C, A]].
    #
    # For an object of height h_stop at the stop plane (z = d6), its image
    # through M^{-1} is at:
    #   If C ≠ 0:  the image is at distance z_img = -D/C from the stop plane
    #              (measured in the input space, i.e. to the LEFT of surface 0)
    #   Wait — let me be more careful.  The front-group matrix M maps from
    #   the first surface plane to the stop plane.  M^{-1} maps from stop to
    #   first surface.  The stop is an object; we want its image in object
    #   space (before the first surface).
    #
    #   For an object at the stop, the image position (relative to surface 0)
    #   is found from the imaging condition:
    #     A point at the stop with height h_stop, angle 0 maps backward to:
    #       y_img = D * h_stop / det,  nu_img = -C * h_stop / det
    #     Image position from surface 0: z = -y_img / nu_img = D / C
    #     Magnification: m = det / (A ... )
    #
    #   Actually the simplest formula: for the front-group system matrix M,
    #   the entrance pupil (image of stop through front group seen from object space):
    #     z_ep = D / C   (distance from first surface, negative = in front of lens)
    #     ... No. Let me use the standard formula.
    #
    #   The front group maps input plane (surface 0) to stop plane.
    #   System matrix: M = T_stop * R5 * T45 * ... * R0
    #   For conjugate imaging: B + A*z_obj = 0 → z_obj (from surf 0) = -B/A
    #   Wait, that's for the object.  For the entrance pupil:
    #
    #   The entrance pupil position (from the input plane = surface 0) is:
    #     z_ep = -D / C   if C ≠ 0
    #   This is measured as a signed distance FROM surface 0 along the z axis.
    #   Negative = in front of the lens (in object space).
    #
    #   The entrance pupil radius is:
    #     r_ep = |aper_r / A|  ... when the marginal ray from [y=1, nu=0]
    #     arrives at height A at the stop.  The stop clips at aper_r, so the
    #     maximum y_in that passes is y_max = aper_r / |A|.  That's the EP radius.
    #
    #   Hmm, but that's only valid for the marginal ray.  More precisely:
    #     r_ep = aper_r * |det / A| ... No.
    #
    #   Let me think about this more carefully.
    #   The entrance pupil is the image of the stop aperture as seen from
    #   object space.  The transverse magnification of the front group
    #   (stop→object space, i.e. reverse direction) is m_t = 1/A for a
    #   system matrix M = [[A,B],[C,D]] with det=1 ... actually m_t = det/A? No.
    #
    #   For system M: input y_in, output y_out = A*y_in + B*nu_in.
    #   For imaging (B=0 in conjugate), m_t = A.
    #   For the entrance pupil (image of stop through front group reversed):
    #     The reversed system has matrix M_rev = [[D, -B], [-C, A]] / det.
    #     Imaging condition for M_rev: -B/det = 0 only if B=0 (stop at conjugate).
    #     In general, the stop is NOT at a conjugate, so we use the general
    #     pupil formula.
    #
    # OK let me just use the direct approach:
    # Entrance pupil position (from first surface) = A/C  (not -D/C)
    # See Smith, "Modern Optical Engineering", the entrance pupil position
    # from the first surface is z_ep = A/C when tracing a ray [y=1, nu=0].
    # The ray exits the front group at the stop with (y=A, nu=C).
    # Extending this ray backward: it crossed y=0 at z = -A/C from the stop.
    # But we want it from the first surface (z=0), and the stop is at d6.
    # So z_ep_from_surf0 = d6 - A/C? No...
    #
    # The ray starts at (z=0, y=1) and arrives at (z=d6, y=A) with slope C (in air).
    # Extending the EXIT ray backward: z_cross = d6 - A/C.
    # That is the z-position of the entrance pupil in the global coordinate system.
    #
    # For the radius: the stop has radius aper_r, and the ray that JUST passes
    # the stop edge has y_stop = aper_r.  Since y_stop = A * y_in (for nu_in=0),
    # y_in = aper_r / A.  So the entrance pupil radius = aper_r / |A|.
    # But wait, this ignores the B term: y_stop = A*y_in + B*nu_in, and nu_in=0
    # for a parallel input ray.  So yes, r_ep = aper_r / |A| for object at infinity.

    aper_r = float(lens.surfaces[problem.config.stop_surface_index].r)

    if abs(C) < 1e-12:
        # C ≈ 0 means the front group is afocal — entrance pupil at infinity.
        # Degenerate case.
        return None

    if abs(A) < 1e-12:
        # A ≈ 0 means the stop is at the front focal point of the front group.
        # Entrance pupil radius would be infinite.  Degenerate.
        return None

    z_ep = d6 - A / C  # global z of entrance pupil
    r_ep = abs(aper_r / A)

    if r_ep < 1e-6 or r_ep > 200.0:
        # Sanity check: radius too small or absurdly large
        return None

    return z_ep, r_ep


# ---------------------------------------------------------------------------
# Worker: evaluates each theta through all three strategies
# ---------------------------------------------------------------------------

@dataclass
class _PupilDiag:
    """Diagnostic info for one evaluation."""
    theta_idx: int
    # Baseline (strategy A)
    baseline_pupilz: float
    baseline_pupilr: float
    baseline_fallback: bool   # did calc_entrance_pupil fall back?
    baseline_fitness: float
    baseline_rms: float
    baseline_invalid_frac: float
    # ABCD (strategy B)
    abcd_pupilz: Optional[float]
    abcd_pupilr: Optional[float]
    abcd_fitness: Optional[float]
    abcd_rms: Optional[float]
    abcd_invalid_frac: Optional[float]
    abcd_failed: bool         # did ABCD also fail?
    # Measures (same for both since Zernike uses coherent trace, not entrance pupil)
    measures: Tuple[float, float]


def _worker_fn(gpu_id: int, task_queue, result_queue, config_dict: dict):
    """Worker process: evaluates each theta under baseline and ABCD strategies."""
    import torch
    torch.cuda.set_device(gpu_id)

    from optical_rl.qd.ef50_zernike import EF50ZernikeConfig, EF50ZernikeProblem

    config = EF50ZernikeConfig(**{**config_dict, "device": f"cuda:{gpu_id}"})
    problem = EF50ZernikeProblem(config)
    problem._ensure_backend()

    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, theta = item
        try:
            result = _evaluate_both_strategies(problem, theta)
            result_queue.put((idx, result))
        except Exception as e:
            result_queue.put((idx, None, str(e)))


def _evaluate_both_strategies(problem, theta) -> dict:
    """Evaluate one theta under baseline and ABCD entrance pupil strategies.

    Returns a dict with all diagnostic fields.
    """
    import torch

    theta_proj = problem.project_theta(theta)
    theta_t = torch.tensor(
        theta_proj, dtype=torch.float32, device=problem._device,
        requires_grad=False,
    )

    # ---- BFL validity (shared) ----
    if problem._f_target is not None:
        _, _, bfl_t = problem._paraxial_fixed_efl(theta_t)
        bfl_val = float(bfl_t.detach().item())
    else:
        bfl_val = float((problem._paraxial_bfl(theta_t) - problem._z_s11).detach().item())

    if bfl_val <= 0.0:
        return {
            "bfl_invalid": True,
            "baseline_fitness": -1e9,
            "abcd_fitness": -1e9,
            "baseline_fallback": False,
            "abcd_failed": False,
            "baseline_pupilr": 0.0,
            "baseline_pupilz": 0.0,
            "abcd_pupilr": 0.0,
            "abcd_pupilz": 0.0,
            "baseline_rms": 0.0,
            "baseline_invalid_frac": 1.0,
            "abcd_rms": 0.0,
            "abcd_invalid_frac": 1.0,
            "measures": (0.0, 0.0),
        }

    lens = problem._build_lens_from_theta_tensor(theta_t)
    hfov = problem._calc_hfov_deg(theta_t)

    # ---- Strategy A: Baseline (current code) ----
    # Intercept builtins.print inside calc_entrance_pupil to detect fallback.
    import builtins
    _real_print = builtins.print
    _saw_fallback = [False]
    def _trap_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if "first surface" in msg:
            _saw_fallback[0] = True
        # still print so user sees warnings
        _real_print(*args, **kwargs)
    builtins.print = _trap_print
    try:
        with torch.no_grad():
            pupilz_bl, pupilr_bl = lens.calc_entrance_pupil(paraxial=True)
    finally:
        builtins.print = _real_print
    baseline_fallback = _saw_fallback[0]

    baseline_fitness, baseline_rms, baseline_inv = _eval_with_pupil(
        problem, lens, hfov, theta_t, float(pupilz_bl), float(pupilr_bl),
    )

    # ---- Strategy B: ABCD matrix entrance pupil ----
    abcd_result = _abcd_entrance_pupil(problem, theta_t)
    abcd_failed = (abcd_result is None)

    if not abcd_failed:
        abcd_pupilz, abcd_pupilr = abcd_result
        abcd_fitness, abcd_rms, abcd_inv = _eval_with_pupil(
            problem, lens, hfov, theta_t, abcd_pupilz, abcd_pupilr,
        )
    else:
        abcd_pupilz, abcd_pupilr = 0.0, 0.0
        abcd_fitness, abcd_rms, abcd_inv = -1e9, 0.0, 1.0

    # ---- Zernike measures (uses coherent trace, independent of entrance pupil) ----
    coeffs = problem._zernike_measures(lens, hfov)
    j0, j1 = problem.config.descriptor_zernike_indices
    m0 = float(coeffs[j0 - 1].detach().cpu().item())
    m1 = float(coeffs[j1 - 1].detach().cpu().item())

    return {
        "bfl_invalid": False,
        "baseline_pupilz": float(pupilz_bl),
        "baseline_pupilr": float(pupilr_bl),
        "baseline_fallback": baseline_fallback,
        "baseline_fitness": baseline_fitness,
        "baseline_rms": baseline_rms,
        "baseline_invalid_frac": baseline_inv,
        "abcd_pupilz": abcd_pupilz,
        "abcd_pupilr": abcd_pupilr,
        "abcd_failed": abcd_failed,
        "abcd_fitness": abcd_fitness,
        "abcd_rms": abcd_rms,
        "abcd_invalid_frac": abcd_inv,
        "measures": (m0, m1),
    }


def _eval_with_pupil(problem, lens, hfov, theta_t, pupilz, pupilr):
    """Evaluate fitness using a specific entrance pupil (z, r).

    Returns (fitness, rms_mean, invalid_fraction_mean) as floats.
    """
    import torch

    # Set the pupil on the problem (temporarily)
    problem._fitness_aperture_xy = problem._fitness_unit_disk * pupilr
    problem._pupil_z = pupilz

    wavelengths = problem.config.fitness_wavelengths_um
    field_pts = problem._field_points(hfov)
    rms_per_wvln, inv_per_wvln = [], []
    for w in wavelengths:
        ray = problem._trace_fields_batch(lens, field_pts, wvln=w)
        rms, inv_frac = problem._rms_spot_batch(ray)
        rms_per_wvln.append(rms)
        inv_per_wvln.append(inv_frac)

    rms_all = torch.stack(rms_per_wvln)
    inv_all = torch.stack(inv_per_wvln)
    rms_mean = float(rms_all.mean().detach().cpu().item())
    inv_mean = float(inv_all.mean().detach().cpu().item())

    fitness = -rms_mean

    intersect_loss = float(problem._intersection_penalty(lens).detach().cpu().item())
    if intersect_loss > 0:
        fitness -= problem.config.intersect_penalty_weight * intersect_loss

    if inv_mean > problem.config.invalid_tol:
        excess = inv_mean - problem.config.invalid_tol
        fitness -= problem.config.invalid_penalty_weight * excess

    return fitness, rms_mean, inv_mean


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def _parallel_evaluate(config_dict: dict, thetas: np.ndarray,
                       num_workers: int, gpu_ids: list[int]) -> list[dict]:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for w in range(num_workers):
        gpu = gpu_ids[w % len(gpu_ids)]
        p = ctx.Process(target=_worker_fn,
                        args=(gpu, task_queue, result_queue, config_dict))
        p.start()
        workers.append(p)

    for i, theta in enumerate(thetas):
        task_queue.put((i, theta))

    for _ in workers:
        task_queue.put(None)

    results = [None] * len(thetas)
    errors = []
    for _ in range(len(thetas)):
        item = result_queue.get()
        if len(item) == 3:
            idx, _, err_msg = item
            errors.append((idx, err_msg))
        else:
            idx, result_dict = item
            results[idx] = result_dict

    for p in workers:
        p.join()

    if errors:
        print(f"  WARNING: {len(errors)} evaluations failed:")
        for idx, msg in errors[:5]:
            print(f"    theta[{idx}]: {msg}")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _analyze(results: list[dict]):
    """Print analysis comparing baseline vs ABCD strategies."""

    valid = [r for r in results if r is not None and not r["bfl_invalid"]]
    bfl_invalid = sum(1 for r in results if r is not None and r["bfl_invalid"])
    errored = sum(1 for r in results if r is None)

    n = len(valid)
    print(f"\n{'='*70}")
    print(f"ENTRANCE PUPIL EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"  Total thetas:        {len(results)}")
    print(f"  BFL invalid (skip):  {bfl_invalid}")
    print(f"  Eval errors:         {errored}")
    print(f"  Valid for analysis:  {n}")

    # --- Fallback frequency ---
    n_fallback = sum(1 for r in valid if r["baseline_fallback"])
    n_abcd_fail = sum(1 for r in valid if r["abcd_failed"])
    n_both_fail = sum(1 for r in valid if r["baseline_fallback"] and r["abcd_failed"])
    n_abcd_saves = sum(1 for r in valid if r["baseline_fallback"] and not r["abcd_failed"])

    print(f"\n--- Fallback frequency ---")
    print(f"  Baseline fallback triggered:    {n_fallback}/{n} ({100*n_fallback/max(n,1):.1f}%)")
    print(f"  ABCD also failed:               {n_abcd_fail}/{n} ({100*n_abcd_fail/max(n,1):.1f}%)")
    print(f"  Both failed:                    {n_both_fail}/{n} ({100*n_both_fail/max(n,1):.1f}%)")
    print(f"  ABCD saves (baseline fell back, {n_abcd_saves}/{n} ({100*n_abcd_saves/max(n,1):.1f}%)")
    print(f"    ABCD succeeded):")

    # --- Pupil radius comparison (when baseline fell back but ABCD didn't) ---
    if n_abcd_saves > 0:
        print(f"\n--- Pupil radius: fallback cases where ABCD succeeded ---")
        bl_rs = [r["baseline_pupilr"] for r in valid if r["baseline_fallback"] and not r["abcd_failed"]]
        abcd_rs = [r["abcd_pupilr"] for r in valid if r["baseline_fallback"] and not r["abcd_failed"]]
        bl_rs = np.array(bl_rs)
        abcd_rs = np.array(abcd_rs)
        print(f"  Baseline (surfaces[0].r):  mean={bl_rs.mean():.3f}  std={bl_rs.std():.3f}")
        print(f"  ABCD:                      mean={abcd_rs.mean():.3f}  std={abcd_rs.std():.3f}")
        ratio = abcd_rs / np.clip(bl_rs, 1e-6, None)
        print(f"  Ratio (ABCD/baseline):     mean={ratio.mean():.3f}  min={ratio.min():.3f}  max={ratio.max():.3f}")

    # --- Fitness comparison: all valid individuals ---
    print(f"\n--- Fitness: all valid individuals ---")
    bl_fit = np.array([r["baseline_fitness"] for r in valid])
    abcd_fit = np.array([r["abcd_fitness"] for r in valid])
    print(f"  Baseline:  mean={bl_fit.mean():.6f}  std={bl_fit.std():.6f}  best={bl_fit.max():.6f}")
    print(f"  ABCD:      mean={abcd_fit.mean():.6f}  std={abcd_fit.std():.6f}  best={abcd_fit.max():.6f}")

    # --- Fitness comparison: fallback individuals only ---
    if n_fallback > 0:
        print(f"\n--- Fitness: baseline-fallback individuals only (n={n_fallback}) ---")
        fb_bl = np.array([r["baseline_fitness"] for r in valid if r["baseline_fallback"]])
        fb_abcd = np.array([r["abcd_fitness"] for r in valid if r["baseline_fallback"]])
        print(f"  Baseline:  mean={fb_bl.mean():.6f}  std={fb_bl.std():.6f}  best={fb_bl.max():.6f}")
        print(f"  ABCD:      mean={fb_abcd.mean():.6f}  std={fb_abcd.std():.6f}  best={fb_abcd.max():.6f}")

        # How many fallback individuals have competitive fitness?
        # "Competitive" = fitness > median of non-fallback individuals
        nonfb = [r["baseline_fitness"] for r in valid if not r["baseline_fallback"]]
        if nonfb:
            median_nonfb = float(np.median(nonfb))
            n_competitive = sum(1 for f in fb_bl if f > median_nonfb)
            print(f"  Non-fallback median fitness:  {median_nonfb:.6f}")
            print(f"  Fallback > median:            {n_competitive}/{n_fallback} "
                  f"({100*n_competitive/n_fallback:.1f}%) ← these may be poisoning the archive")

    # --- Invalid fraction comparison ---
    print(f"\n--- Invalid ray fraction ---")
    bl_inv = np.array([r["baseline_invalid_frac"] for r in valid])
    abcd_inv = np.array([r["abcd_invalid_frac"] for r in valid])
    print(f"  Baseline:  mean={bl_inv.mean():.4f}  >5%: {(bl_inv > 0.05).sum()}/{n}")
    print(f"  ABCD:      mean={abcd_inv.mean():.4f}  >5%: {(abcd_inv > 0.05).sum()}/{n}")

    if n_fallback > 0:
        fb_bl_inv = np.array([r["baseline_invalid_frac"] for r in valid if r["baseline_fallback"]])
        fb_abcd_inv = np.array([r["abcd_invalid_frac"] for r in valid if r["baseline_fallback"]])
        print(f"  Fallback subset baseline:  mean={fb_bl_inv.mean():.4f}")
        print(f"  Fallback subset ABCD:      mean={fb_abcd_inv.mean():.4f}")

    # --- RMS comparison ---
    print(f"\n--- RMS spot size (mm) ---")
    bl_rms = np.array([r["baseline_rms"] for r in valid])
    abcd_rms = np.array([r["abcd_rms"] for r in valid])
    print(f"  Baseline:  mean={bl_rms.mean():.6f}  std={bl_rms.std():.6f}")
    print(f"  ABCD:      mean={abcd_rms.mean():.6f}  std={abcd_rms.std():.6f}")

    # --- Archive simulation ---
    print(f"\n--- Archive simulation (what would get inserted) ---")
    _simulate_archives(valid)


def _simulate_archives(valid: list[dict]):
    """Build archives under each strategy and compare."""
    from optical_rl.qd.ef50_zernike import EF50Archive2D, EF50EvalResult

    # Compute bounds from ALL measures (shared)
    m0_vals = [r["measures"][0] for r in valid]
    m1_vals = [r["measures"][1] for r in valid]
    bounds = _percentile_bounds(m0_vals, m1_vals)

    bins = (512, 512)

    # Archive A: baseline fitness
    archive_bl = EF50Archive2D(bins, bounds, seed=0)
    for r in valid:
        archive_bl.insert(EF50EvalResult(
            theta=np.zeros(10),  # placeholder
            fitness=r["baseline_fitness"],
            measures=r["measures"],
            metrics={},
        ))

    # Archive B: ABCD fitness (penalize both-fail as -1e9)
    archive_abcd = EF50Archive2D(bins, bounds, seed=0)
    for r in valid:
        archive_abcd.insert(EF50EvalResult(
            theta=np.zeros(10),
            fitness=r["abcd_fitness"],
            measures=r["measures"],
            metrics={},
        ))

    # Archive C: baseline fitness, but fallback individuals get -1e9
    archive_flagged = EF50Archive2D(bins, bounds, seed=0)
    for r in valid:
        fit = r["baseline_fitness"] if not r["baseline_fallback"] else -1e9
        archive_flagged.insert(EF50EvalResult(
            theta=np.zeros(10),
            fitness=fit,
            measures=r["measures"],
            metrics={},
        ))

    stats_bl = archive_bl.stats()
    stats_abcd = archive_abcd.stats()
    stats_flagged = archive_flagged.stats()

    print(f"  {'':30s} {'Baseline':>12s} {'ABCD':>12s} {'Flagged':>12s}")
    print(f"  {'Filled cells':30s} {stats_bl['filled']:>12.0f} {stats_abcd['filled']:>12.0f} {stats_flagged['filled']:>12.0f}")
    print(f"  {'Coverage':30s} {stats_bl['coverage']:>12.6f} {stats_abcd['coverage']:>12.6f} {stats_flagged['coverage']:>12.6f}")
    print(f"  {'Best fitness':30s} {stats_bl['best_fitness']:>12.6f} {stats_abcd['best_fitness']:>12.6f} {stats_flagged['best_fitness']:>12.6f}")

    # Count how many cells differ between baseline and ABCD
    cells_bl = archive_bl.cells
    cells_abcd = archive_abcd.cells
    all_keys = set(cells_bl.keys()) | set(cells_abcd.keys())
    n_differ = 0
    n_bl_only = 0
    n_abcd_only = 0
    fitness_changes = []
    for k in all_keys:
        in_bl = k in cells_bl
        in_abcd = k in cells_abcd
        if in_bl and in_abcd:
            if abs(cells_bl[k].fitness - cells_abcd[k].fitness) > 1e-8:
                n_differ += 1
                fitness_changes.append(cells_abcd[k].fitness - cells_bl[k].fitness)
        elif in_bl:
            n_bl_only += 1
        else:
            n_abcd_only += 1

    print(f"\n  Cell differences (baseline vs ABCD):")
    print(f"    Cells in both, same fitness:    {len(all_keys) - n_differ - n_bl_only - n_abcd_only}")
    print(f"    Cells in both, different fitness: {n_differ}")
    print(f"    Only in baseline:               {n_bl_only}")
    print(f"    Only in ABCD:                   {n_abcd_only}")
    if fitness_changes:
        fc = np.array(fitness_changes)
        print(f"    Fitness delta (ABCD - baseline): mean={fc.mean():.6f}  "
              f"min={fc.min():.6f}  max={fc.max():.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrance pupil fallback experiment for EF50 Zernike"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    from optical_rl.qd.ef50_zernike import _load_config, EF50ZernikeConfig

    if args.config is not None:
        config = _load_config(args.config)
    else:
        config = EF50ZernikeConfig()

    config_dict = _config_to_dict(config)

    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus else _available_gpus()
    num_workers = _pick_worker_count(args.workers, len(gpu_ids))

    print("=" * 70)
    print("ENTRANCE PUPIL FALLBACK EXPERIMENT")
    print("=" * 70)
    print(f"  GPUs:     {gpu_ids}")
    print(f"  Workers:  {num_workers}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Stage:    {config.stage}")
    print(f"  Rays:     {config.num_rays}")
    print("=" * 70)

    # Generate thetas (same as seed_parallel_experiment)
    from optical_rl.qd.ef50_zernike import EF50ZernikeProblem
    problem = EF50ZernikeProblem(config)
    problem._ensure_backend()

    flat = problem.initial_theta()
    all_thetas = np.vstack([
        flat.reshape(1, -1),
        problem.sample_theta(args.seeds, config.cma_seed + 31),
    ])
    del problem  # free GPU memory before spawning workers

    print(f"\nEvaluating {len(all_thetas)} thetas (baseline + ABCD) ...")
    t0 = time.perf_counter()
    results = _parallel_evaluate(config_dict, all_thetas, num_workers, gpu_ids)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s ({len(all_thetas) / max(elapsed, 0.01):.1f} evals/s)")

    _analyze(results)


if __name__ == "__main__":
    main()
