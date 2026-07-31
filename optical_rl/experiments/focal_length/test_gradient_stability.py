"""Gradient stability diagnostic for focal-length experiments.

Sweeps designs across a range of perturbations — especially varying the last
surface curvature (c6) — and checks:
  1. Gradient stability (finite, reasonable magnitude)
  2. EFL feasibility (how close to target)
  3. Fitness landscape smoothness

Usage:
    source /playpen/hisham26/optifusion_env/bin/activate
    PYTHONPATH=/home/hisham26/Desktop/RL/DeepLens:$PYTHONPATH \
    python -m optical_rl.experiments.focal_length.test_gradient_stability
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import math
import numpy as np
import torch

torch.set_num_threads(1)

from optical_rl.qd.lensmega.problem.lens_problem import LensProblemConfig
from optical_rl.qd.lensmega.descriptors.zernike import ZernikeDescriptor

from .exp_a_mid_solve import MidSolveLensProblem
from .exp_b_soft_penalty import SoftPenaltyLensProblem
from .exp_c_image_height import ImageHeightLensProblem
from .exp_d_power_budget import PowerBudgetLensProblem


LENS_PATH = "DeepLens/datasets/lenses/cooke.json"
EFL_TARGET = 50.0
DEVICE = "cpu"


def _make_6free_cfg():
    """Config with all 6 curvature surfaces free (for Exp B, C, D)."""
    return LensProblemConfig(
        lens_path=LENS_PATH,
        wavelength_um=0.587,
        fitness_wavelengths_um=(0.587,),
        object_depth_mm=-20000.0,
        stop_surface_index=4,
        free_curvature_indices=(0, 1, 2, 3, 5, 6),
        derived_curvature_index=None,
        focal_length_target_mm=None,
        theta_bounds=((-0.15, 0.15),) * 6,
        stage="stage2",
        num_rays=128,
        strict_entrance_pupil=False,
        device=DEVICE,
    )


def _make_5free_cfg():
    """Config with 5 free surfaces (for Exp A — surface 3 is derived)."""
    return LensProblemConfig(
        lens_path=LENS_PATH,
        wavelength_um=0.587,
        fitness_wavelengths_um=(0.587,),
        object_depth_mm=-20000.0,
        stop_surface_index=4,
        free_curvature_indices=(0, 1, 2, 5, 6),
        derived_curvature_index=None,
        focal_length_target_mm=None,
        theta_bounds=((-0.15, 0.15),) * 5,
        stage="stage2",
        num_rays=128,
        strict_entrance_pupil=False,
        device=DEVICE,
    )


def _make_baseline_cfg():
    """Config for baseline (last-surface solve, 5 free)."""
    return LensProblemConfig(
        lens_path=LENS_PATH,
        wavelength_um=0.587,
        fitness_wavelengths_um=(0.587,),
        object_depth_mm=-20000.0,
        stop_surface_index=4,
        free_curvature_indices=(0, 1, 2, 3, 5),
        derived_curvature_index=6,
        focal_length_target_mm=EFL_TARGET,
        theta_bounds=((-0.15, 0.15),) * 5,
        stage="stage2",
        num_rays=128,
        strict_entrance_pupil=False,
        device=DEVICE,
    )


def _make_power_cfg():
    """Config for Exp D (power budget, wider bounds)."""
    return LensProblemConfig(
        lens_path=LENS_PATH,
        wavelength_um=0.587,
        fitness_wavelengths_um=(0.587,),
        object_depth_mm=-20000.0,
        stop_surface_index=4,
        free_curvature_indices=(0, 1, 2, 3, 5, 6),
        derived_curvature_index=None,
        focal_length_target_mm=None,
        theta_bounds=((-2.0, 2.0),) * 6,
        stage="stage2",
        num_rays=128,
        strict_entrance_pupil=False,
        device=DEVICE,
    )


def build_problems():
    """Instantiate all experiment problems + baseline."""
    desc = ZernikeDescriptor(noll_indices=[7, 11], num_rays=128)

    from optical_rl.qd.lensmega.problem.lens_problem import LensProblem

    problems = {}

    # Baseline: last-surface solve (surface 6 derived)
    cfg_base = _make_baseline_cfg()
    problems["baseline"] = LensProblem(cfg_base, desc)

    # Exp A: mid-surface solve (surface 3 derived)
    cfg_a = _make_5free_cfg()
    problems["A_mid_solve"] = MidSolveLensProblem(
        cfg_a, desc, mid_solve_index=3, efl_target_mm=EFL_TARGET
    )

    # Exp B: soft penalty
    cfg_b = _make_6free_cfg()
    problems["B_soft_penalty"] = SoftPenaltyLensProblem(
        cfg_b, desc, efl_target_mm=EFL_TARGET, efl_penalty_weight=10.0
    )

    # Exp C: image height
    cfg_c = _make_6free_cfg()
    problems["C_image_height"] = ImageHeightLensProblem(
        cfg_c, desc, efl_target_mm=EFL_TARGET,
        focal_loss_weight=5.0, efl_penalty_weight=1.0
    )

    # Exp D: power budget
    cfg_d = _make_power_cfg()
    problems["D_power_budget"] = PowerBudgetLensProblem(
        cfg_d, desc, efl_target_mm=EFL_TARGET, efl_penalty_weight=5.0
    )

    for name, p in problems.items():
        p._ensure_backend()

    return problems


def sweep_last_surface(problems: dict, n_steps: int = 15):
    """Sweep the last surface curvature (c6) while keeping others fixed.

    For baseline and Exp A (5-DOF problems), c6 is either derived or free
    at a different position.  We adapt the theta accordingly.
    """
    print("\n" + "=" * 90)
    print("SWEEP: Varying last surface curvature (c6) from -0.14 to +0.14")
    print("       Other curvatures fixed at small values: [0.02, -0.01, 0.01, -0.01, 0.015]")
    print("=" * 90)

    c6_values = np.linspace(-0.14, 0.14, n_steps)
    # Fixed base curvatures for surfaces 0,1,2,3,5
    base_c = [0.02, -0.01, 0.01, -0.01, 0.015]

    for name, prob in problems.items():
        print(f"\n--- {name} ---")
        print(f"{'c6':>8s} {'fitness':>10s} {'efl':>10s} {'|grad_f|':>10s} "
              f"{'|grad_m0|':>10s} {'|grad_m1|':>10s} {'status':>12s}")

        for c6 in c6_values:
            if name == "baseline":
                # 5-DOF: [c0, c1, c2, c3, c5].  c6 is derived — we can't
                # control it directly.  Instead vary c5 (the free surface
                # closest to the end).
                theta = np.array(base_c, dtype=np.float64)
                # Override c5 (last free entry) with c6 value to test
                # what happens when the front surfaces push the derived
                # curvature to different values.
                theta[4] = c6
            elif name == "A_mid_solve":
                # 5-DOF: [c0, c1, c2, c5, c6].  c3 is derived.
                # c6 IS free at index 4.
                theta = np.array([base_c[0], base_c[1], base_c[2],
                                  base_c[4], c6], dtype=np.float64)
            elif name == "D_power_budget":
                # 6-DOF power fractions: [p0, p1, p2, p3, p5, p6]
                # Map c6 range to power-fraction range.
                theta = np.array([0.3, -0.2, 0.5, -0.15, 0.3,
                                  c6 / 0.14 * 1.5], dtype=np.float64)
            else:
                # 6-DOF: [c0, c1, c2, c3, c5, c6]
                theta = np.array([base_c[0], base_c[1], base_c[2],
                                  base_c[3], base_c[4], c6],
                                 dtype=np.float64)

            theta_proj = prob.project_theta(theta)
            try:
                r = prob.evaluate(theta_proj, with_grads=True)
                rejected = r.metrics.get("rejected", 0.0) > 0.0
                efl = r.metrics.get("efl_actual", float("nan"))

                if rejected or r.grads is None:
                    status = "REJECTED"
                    grad_f_norm = float("nan")
                    grad_m0_norm = float("nan")
                    grad_m1_norm = float("nan")
                else:
                    grad_f = r.grads["fitness"]
                    grad_m0 = r.grads["m0"]
                    grad_m1 = r.grads["m1"]
                    grad_f_norm = float(np.linalg.norm(grad_f))
                    grad_m0_norm = float(np.linalg.norm(grad_m0))
                    grad_m1_norm = float(np.linalg.norm(grad_m1))

                    if not np.isfinite(grad_f_norm):
                        status = "GRAD_NAN"
                    elif grad_f_norm > 1e6:
                        status = "GRAD_EXPLODE"
                    elif grad_f_norm < 1e-10:
                        status = "GRAD_DEAD"
                    else:
                        status = "OK"

                print(f"{c6:8.4f} {r.fitness:10.4f} {efl:10.2f} "
                      f"{grad_f_norm:10.4f} {grad_m0_norm:10.4f} "
                      f"{grad_m1_norm:10.4f} {status:>12s}")

            except Exception as e:
                print(f"{c6:8.4f} {'ERROR':>10s} {'---':>10s} "
                      f"{'---':>10s} {'---':>10s} {'---':>10s} "
                      f"{str(e)[:30]:>30s}")


def sweep_random_diverse(problems: dict, n_samples: int = 20, seed: int = 42):
    """Sample random diverse designs and check gradient stability.

    Uses wide-range random sampling to probe the design space broadly.
    """
    print("\n" + "=" * 90)
    print(f"SWEEP: {n_samples} random diverse designs (wide perturbations)")
    print("=" * 90)

    rng = np.random.default_rng(seed)

    for name, prob in problems.items():
        print(f"\n--- {name} ---")
        print(f"{'i':>4s} {'fitness':>10s} {'efl':>10s} {'|grad_f|':>10s} "
              f"{'efl_err%':>10s} {'status':>12s}")

        n_ok = 0
        n_reject = 0
        n_bad_grad = 0
        efl_errors = []
        grad_norms = []

        for i in range(n_samples):
            bounds = np.array(prob.theta_bounds)
            lo, hi = bounds[:, 0], bounds[:, 1]
            theta = lo + rng.random(len(lo)) * (hi - lo)

            try:
                r = prob.evaluate(theta, with_grads=True)
                rejected = r.metrics.get("rejected", 0.0) > 0.0
                efl = r.metrics.get("efl_actual", float("nan"))

                if rejected or r.grads is None:
                    status = "REJECTED"
                    n_reject += 1
                    efl_err = float("nan")
                    gf_norm = float("nan")
                else:
                    gf = r.grads["fitness"]
                    gf_norm = float(np.linalg.norm(gf))
                    grad_norms.append(gf_norm)

                    if np.isfinite(efl):
                        efl_err = abs(efl - EFL_TARGET) / EFL_TARGET * 100
                        efl_errors.append(efl_err)
                    else:
                        efl_err = float("nan")

                    if not np.isfinite(gf_norm) or gf_norm > 1e6:
                        status = "BAD_GRAD"
                        n_bad_grad += 1
                    else:
                        status = "OK"
                        n_ok += 1

                print(f"{i:4d} {r.fitness:10.4f} {efl:10.2f} "
                      f"{gf_norm:10.4f} {efl_err:9.1f}% {status:>12s}")

            except Exception as e:
                n_reject += 1
                print(f"{i:4d} {'ERROR':>10s} {'---':>10s} "
                      f"{'---':>10s} {'---':>10s} {str(e)[:20]:>12s}")

        print(f"\n  Summary: {n_ok} OK / {n_reject} rejected / {n_bad_grad} bad gradients")
        if efl_errors:
            print(f"  EFL error: mean={np.mean(efl_errors):.1f}% "
                  f"median={np.median(efl_errors):.1f}% "
                  f"max={np.max(efl_errors):.1f}%")
        if grad_norms:
            print(f"  |grad_f|: mean={np.mean(grad_norms):.4f} "
                  f"std={np.std(grad_norms):.4f} "
                  f"max={np.max(grad_norms):.4f}")


def main():
    print("Building all experiment problems...")
    problems = build_problems()
    print(f"Built {len(problems)} problems: {list(problems.keys())}")

    sweep_last_surface(problems, n_steps=15)
    sweep_random_diverse(problems, n_samples=20, seed=42)


if __name__ == "__main__":
    main()
