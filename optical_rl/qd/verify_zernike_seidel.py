"""Verification: Analytical Seidel W_040 vs numerically fitted Z11 for a singlet.

The thin-lens Seidel formula for primary spherical aberration (on-axis, object
at infinity) is:

    W_040 = h^4 / (8*f^3) * [n*(n+2)/(n-1)^2 * B^2 + n^2/(n-1)^2]

where:
  h = marginal ray height at the stop (stop radius)
  f = effective focal length
  n = glass refractive index at the design wavelength
  B = (c1 + c2) / (c1 - c2)   — Coddington shape factor

The numerical W_040 is extracted from the Zernike decomposition via:

    W_040_num = a11 * 6 * sqrt(5)

This comes from expressing the monomial rho^4 in the Noll Zernike basis:

    rho^4 = Z11/(6*sqrt(5)) + Z4/(2*sqrt(3)) + Z1/3

so a_11 = W_040 / (6*sqrt(5)) for a pure W_040*rho^4 wavefront.

Expected outcomes
-----------------
1. Analytical and numerical W_040 should be strongly correlated (|r| > 0.95).
2. The linear fit slope should be near +1 or -1 (sign depends on OPL convention).
   Deviation from |slope|=1 comes from thick-lens vs thin-lens and higher-order
   spherical terms (rho^6, rho^8) that matter at larger apertures.
3. The Z4/Z11 ratio (slope of Z4 vs Z11 across all c1) should be near 0.
   If it is near 3.87, the OPD is being measured at the sensor plane (bug).

Usage
-----
    cd /path/to/RL
    python -m optical_rl.qd.verify_zernike_seidel \\
        optical_rl/configs/qd_singlet_cmamega.yaml --n_sweep 21 --plot
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from optical_rl.qd.singlet_cmamega import SingletCMAMEGAConfig, FixedSingletProblem


# ──────────────────────────────────────────────────────────────────────────────
# Analytical formula
# ──────────────────────────────────────────────────────────────────────────────

def analytical_w040(c1: float, c2: float, n: float, f: float, h: float) -> float:
    """Third-order thin-lens Seidel W_040 for an on-axis object at infinity.

    Args:
        c1: front surface curvature [1/mm] (= 1/R1, positive if convex toward +z)
        c2: rear surface curvature [1/mm]
        n:  glass refractive index
        f:  effective focal length [mm]
        h:  marginal ray height at the stop (= stop radius) [mm]

    Returns:
        W_040 [mm] — positive = undercorrected (marginal focus closer than paraxial).
    """
    denom_B = c1 - c2
    B = (c1 + c2) / denom_B if abs(denom_B) > 1e-12 else 0.0
    n_m1 = n - 1.0
    phi_S = n * (n + 2.0) / n_m1**2 * B**2 + n**2 / n_m1**2
    return (h**4 / (8.0 * f**3)) * phi_S


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────

def main(
    config_yaml: str = "optical_rl/configs/qd_singlet_cmamega.yaml",
    n_sweep: int = 21,
    plot: bool = False,
) -> None:
    import yaml
    import torch

    with open(config_yaml) as fh:
        raw = yaml.safe_load(fh)["qd"]

    # Force descriptor_zernike_indices=(4, 11) so that the problem's _pinv_Z
    # covers at least j=1..11 — required for this verification regardless of
    # what is set in the YAML config.
    config = SingletCMAMEGAConfig(
        lens_path=raw["lens_path"],
        wavelength_um=float(raw.get("wavelength_um", 0.587)),
        object_depth_mm=float(raw.get("object_depth_mm", -1000.0)),
        theta_bounds=tuple(tuple(b) for b in raw["theta_bounds"]),
        focal_length_target_mm=raw.get("focal_length_target_mm"),
        descriptor_zernike_indices=(4, 11),
        num_rays=int(raw.get("num_rays", 256)),
        device=str(raw.get("device", "auto")),
    )

    problem = FixedSingletProblem(config)
    problem._ensure_backend()

    n = problem._n_glass
    f = problem._f_target
    h = float(problem._template_lens.surfaces[0].r)  # stop radius = pupil half-diameter

    print("=" * 72)
    print("Zernike–Seidel cross-verification  (on-axis singlet)")
    print(f"  n_glass  = {n:.5f}")
    print(f"  f_target = {f:.4f} mm")
    print(f"  h_stop   = {h:.4f} mm  (marginal ray height = stop radius)")
    print(f"  λ        = {config.wavelength_um:.4f} µm")
    print("=" * 72)

    c1_lo, c1_hi = config.theta_bounds[0]
    c1_vals = np.linspace(c1_lo, c1_hi, n_sweep)

    # Storage
    w040_ana_list: list[float] = []
    w040_num_list: list[float] = []
    z4_list:       list[float] = []
    z11_list:      list[float] = []
    B_list:        list[float] = []

    header = (
        f"{'c1':>10}  {'c2':>10}  {'B':>8}"
        f"  {'W040_ana[µm]':>14}  {'W040_num[µm]':>14}"
        f"  {'Z4[µm]':>10}  {'Z11[µm]':>10}"
    )
    print(header)
    print("-" * len(header))

    for c1 in c1_vals:
        c1_t = torch.tensor(float(c1), dtype=torch.float32, device=problem._device)
        c2_t, d_sensor_t = problem._paraxial_solve(c1_t)
        c2 = float(c2_t.detach().item())

        # Build lens for this c1
        lens = copy.deepcopy(problem._template_lens)
        dtype = lens.surfaces[1].c.dtype
        lens.surfaces[1].c = c1_t.to(dtype=dtype)
        lens.surfaces[2].c = c2_t.to(dtype=dtype)
        lens.d_sensor = d_sensor_t.to(dtype=dtype)

        # Numerical Zernike coefficients (no gradient needed)
        with torch.no_grad():
            # _zernike_measures returns shape (j_max,) = (11,)
            coeffs = problem._zernike_measures(lens)

        a4  = float(coeffs[3].detach().item())    # j=4  defocus coefficient [mm]
        a11 = float(coeffs[10].detach().item())   # j=11 primary spherical [mm]

        # Convert a11 to W_040 using:  a11 = W_040 / (6*sqrt(5))
        # (derived from rho^4 = Z11/(6√5) + Z4/(2√3) + Z1/3)
        w040_num = a11 * 6.0 * math.sqrt(5.0)

        # Analytical W_040 (thin-lens, 3rd-order Seidel)
        B = (c1 + c2) / (c1 - c2) if abs(c1 - c2) > 1e-12 else 0.0
        w040_ana = analytical_w040(c1, c2, n, f, h)

        w040_ana_list.append(w040_ana)
        w040_num_list.append(w040_num)
        z4_list.append(a4)
        z11_list.append(a11)
        B_list.append(B)

        print(
            f"{c1:10.5f}  {c2:10.5f}  {B:8.4f}"
            f"  {w040_ana * 1e3:14.5f}"
            f"  {w040_num * 1e3:14.5f}"
            f"  {a4 * 1e3:10.5f}"
            f"  {a11 * 1e3:10.5f}"
        )

    w040_ana_arr = np.array(w040_ana_list)
    w040_num_arr = np.array(w040_num_list)
    z4_arr       = np.array(z4_list)
    z11_arr      = np.array(z11_list)

    # ── statistics ────────────────────────────────────────────────────────────
    print()
    print("── Summary statistics ──────────────────────────────────────────────")

    std_ana = float(np.std(w040_ana_arr))
    std_num = float(np.std(w040_num_arr))
    if std_ana > 1e-12 and std_num > 1e-12:
        corr = float(np.corrcoef(w040_ana_arr, w040_num_arr)[0, 1])
        slope, intercept = np.polyfit(w040_ana_arr, w040_num_arr, 1)
        print(f"\nAnalytical vs numerical W_040:")
        print(f"  Pearson r   = {corr:.4f}    (|r| > 0.95 expected)")
        print(f"  Linear fit  W040_num = {slope:.4f} × W040_ana + {intercept:.3e} mm")
        print(f"  |slope| ≈ 1 → formula correct.")
        print(f"  Offset/slope ≠ 1 allowed: thick-lens BFL ≠ thin-lens, higher-order terms.")
    else:
        corr = float("nan")
        slope = float("nan")
        print("  Insufficient variation to compute statistics.")

    std_z11 = float(np.std(z11_arr))
    if std_z11 > 1e-12:
        z4_z11_slope, _ = np.polyfit(z11_arr, z4_arr, 1)
        print(f"\nZ4 / Z11 linear relationship:")
        print(f"  Slope of (Z4 vs Z11) = {z4_z11_slope:.3f}")
        print(f"  ≈  0    → exit-pupil OPD + ref-sphere correction  (CORRECT)")
        print(f"  ≈  3.87 → sensor-plane OPD without correction      (BUG)")
        if abs(z4_z11_slope) < 0.5:
            print("  ✓ Exit-pupil OPD appears correct.")
        elif abs(z4_z11_slope - 3.87) < 0.5:
            print("  ✗ Looks like sensor-plane OPD — check _zernike_measures.")
        else:
            print(f"  ? Unexpected ratio; investigate further.")
    else:
        z4_z11_slope = float("nan")

    # ── optional plot ─────────────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes[0]
        ax.plot(c1_vals, w040_ana_arr * 1e3, "b-o", ms=4, lw=1.5,
                label="Analytical (Seidel 3rd order)")
        ax.plot(c1_vals, w040_num_arr * 1e3, "r-s", ms=4, lw=1.5,
                label="Numerical (Z11 Zernike fit)")
        ax.set_xlabel("c1  [1/mm]")
        ax.set_ylabel("W_040  [µm]")
        ax.set_title("Spherical aberration W_040 vs bending")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        sc = ax.scatter(w040_ana_arr * 1e3, w040_num_arr * 1e3,
                        c=c1_vals, cmap="viridis", s=30, zorder=3)
        lo = min(float(w040_ana_arr.min()), float(w040_num_arr.min())) * 1e3
        hi = max(float(w040_ana_arr.max()), float(w040_num_arr.max())) * 1e3
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1.2, label="y = x")
        ax.set_xlabel("Analytical W_040  [µm]")
        ax.set_ylabel("Numerical W_040  [µm]")
        ax.set_title(f"r = {corr:.4f},  slope = {slope:.3f}")
        plt.colorbar(sc, ax=ax, label="c1  [1/mm]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        sc2 = ax.scatter(z11_arr * 1e3, z4_arr * 1e3,
                         c=c1_vals, cmap="viridis", s=30, zorder=3)
        ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.4, label="Z4 = 0")
        if not math.isnan(z4_z11_slope):
            x_fit = np.linspace(z11_arr.min(), z11_arr.max(), 100)
            z4_fit_intercept = float(np.polyfit(z11_arr, z4_arr, 1)[1])
            ax.plot(x_fit * 1e3,
                    (z4_z11_slope * x_fit + z4_fit_intercept) * 1e3,
                    "r--", lw=1.2, alpha=0.7,
                    label=f"slope = {z4_z11_slope:.2f}")
        ax.set_xlabel("Z11  [µm]  (primary spherical)")
        ax.set_ylabel("Z4   [µm]  (defocus)")
        ax.set_title("Z4 vs Z11  (slope ≈ 0 expected)")
        plt.colorbar(sc2, ax=ax, label="c1  [1/mm]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out_png = "zernike_seidel_verification.png"
        plt.savefig(out_png, dpi=150)
        print(f"\nPlot saved → {out_png}")
        plt.show()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Zernike–Seidel cross-verification for singlet.")
    p.add_argument(
        "config",
        nargs="?",
        default="optical_rl/configs/qd_singlet_cmamega.yaml",
        help="Path to the singlet CMA-MEGA YAML config.",
    )
    p.add_argument("--n_sweep", type=int, default=21,
                   help="Number of c1 values to sweep (default: 21).")
    p.add_argument("--plot", action="store_true",
                   help="Show and save a 3-panel comparison plot.")
    a = p.parse_args()
    main(a.config, a.n_sweep, a.plot)
