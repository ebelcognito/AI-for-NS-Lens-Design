"""Pupil sampling quality diagnostics for Zernike decomposition.

Checks two things for a given (num_rays, j_max) configuration:

  1. Gram matrix  G = Z.T @ Z / N
     Should be close to identity.  Large off-diagonal entries indicate
     cross-talk between Zernike terms on the discrete pupil grid.

  2. Singular values of Z
     Condition number sv_max / sv_min.  High values mean the pseudo-inverse
     amplifies OPD noise for the worst-sampled mode.

Usage
-----
  python optical_rl/qd/verify_pupil_sampling.py
  python optical_rl/qd/verify_pupil_sampling.py --num_rays 64 128 256 512 --j_max 16
  python optical_rl/qd/verify_pupil_sampling.py --num_rays 256 --j_max 11
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.singlet_zernike_measures import (
    NOLL_N_TERMS,
    NOLL_TERM_NAMES,
    _noll_zernike_matrix,
)


# ---------------------------------------------------------------------------
# Fibonacci / golden-angle disk sampler (identical to singlet_cmamega.py)
# ---------------------------------------------------------------------------

def _disk_samples(num_rays: int) -> np.ndarray:
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(num_rays))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=-1)


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(num_rays: int, j_max: int) -> dict:
    """Return Gram matrix, singular values, and summary scalars."""
    if j_max > NOLL_N_TERMS:
        raise ValueError(f"j_max={j_max} exceeds implemented maximum {NOLL_N_TERMS}.")
    if num_rays < j_max:
        raise ValueError(f"Underdetermined: num_rays={num_rays} < j_max={j_max}.")

    xy = _disk_samples(num_rays)
    x, y = xy[:, 0], xy[:, 1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    Z = _noll_zernike_matrix(rho, phi, j_max)       # (N, j_max)
    G = Z.T @ Z / num_rays                           # (j_max, j_max) Gram matrix

    # Diagonal residuals: how far G_ii is from 1 (normalization error).
    diag_res_max = float(np.abs(np.diag(G) - 1.0).max())

    # Off-diagonal cross-talk: purely the i≠j entries (cross-term leakage).
    off_diag_mask = G.copy()
    np.fill_diagonal(off_diag_mask, 0.0)
    off_diag_max = float(np.abs(off_diag_mask).max())
    off_diag_rms = float(np.sqrt((off_diag_mask**2).sum() / (j_max * (j_max - 1))))

    sv = np.linalg.svd(Z, compute_uv=False)         # descending singular values
    condition_number = float(sv[0] / sv[-1])

    return {
        "xy": xy,
        "rho": rho,
        "Z": Z,
        "G": G,
        "sv": sv,
        "diag_res_max": diag_res_max,
        "off_diag_max": off_diag_max,
        "off_diag_rms": off_diag_rms,
        "condition_number": condition_number,
        "num_rays": num_rays,
        "j_max": j_max,
    }


def print_summary(d: dict) -> None:
    n, j = d["num_rays"], d["j_max"]
    print(f"\n{'='*60}")
    print(f"  num_rays={n}   j_max={j}   overdetermination={n/j:.1f}x")
    print(f"{'='*60}")
    print(f"  Gram diagonal residual  max|G_ii - 1| = {d['diag_res_max']:.5f}")
    print(f"  Gram cross-talk (i≠j)   max|G_ij|    = {d['off_diag_max']:.5f}   "
          f"rms = {d['off_diag_rms']:.5f}")
    print(f"  Condition number   κ(Z) = {d['condition_number']:.2f}")

    if d["off_diag_max"] < 0.02:
        flag = "OK  (< 0.02 — effectively orthogonal)"
    elif d["off_diag_max"] < 0.05:
        flag = "WARN  (0.02–0.05 — yellow flag, check which pairs)"
    else:
        flag = "BAD  (> 0.05 — real cross-talk present)"
    print(f"  Orthogonality:     {flag}")

    if d["condition_number"] < 20:
        cflag = "OK  (< 20)"
    elif d["condition_number"] < 100:
        cflag = "WARN  (20–100)"
    else:
        cflag = "BAD  (> 100 — noise amplification)"
    print(f"  Conditioning:      {cflag}")
    print()

    # show the worst-offending off-diagonal pair
    G = d["G"]
    j_max = d["j_max"]
    off = np.abs(G.copy())
    np.fill_diagonal(off, 0.0)
    i_max, j_max_idx = np.unravel_index(off.argmax(), off.shape)
    j_i = i_max + 1
    j_j = j_max_idx + 1
    name_i = NOLL_TERM_NAMES.get(j_i, f"Z{j_i}")
    name_j = NOLL_TERM_NAMES.get(j_j, f"Z{j_j}")
    print(f"  Worst pair: Z{j_i} ({name_i}) ↔ Z{j_j} ({name_j})"
          f"   |G_ij| = {off[i_max, j_max_idx]:.5f}")

    print(f"\n  Singular values (j=1..{d['j_max']}):")
    sv_str = "  " + "  ".join(f"{v:.3f}" for v in d["sv"])
    print(sv_str)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _term_labels(j_max: int) -> list[str]:
    return [f"Z{j}\n{NOLL_TERM_NAMES.get(j, '')[:8]}" for j in range(1, j_max + 1)]


def plot_single(d: dict, ax_gram, ax_sv, ax_pupil, ax_rho_cdf) -> None:
    n, j = d["num_rays"], d["j_max"]
    G = d["G"]
    sv = d["sv"]
    labels = _term_labels(j)

    # --- Gram matrix heatmap ---
    vmax = max(0.1, float(np.abs(G).max()))
    im = ax_gram.imshow(G, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax_gram.set_title(f"Gram matrix  G = ZᵀZ/N\n(N={n}, j_max={j})", fontsize=9)
    ax_gram.set_xticks(range(j))
    ax_gram.set_xticklabels(labels, fontsize=6, rotation=90)
    ax_gram.set_yticks(range(j))
    ax_gram.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax_gram, fraction=0.046, pad=0.04)
    ax_gram.set_xlabel(f"off-diag max={d['off_diag_max']:.4f}  rms={d['off_diag_rms']:.4f}",
                       fontsize=7)

    # --- Singular values bar ---
    ax_sv.bar(range(1, j + 1), sv, color="steelblue", edgecolor="white", linewidth=0.5)
    ax_sv.axhline(sv[-1], color="red", linestyle="--", linewidth=0.8, label=f"sv_min={sv[-1]:.2f}")
    ax_sv.set_title(f"Singular values of Z\nκ={d['condition_number']:.1f}", fontsize=9)
    ax_sv.set_xlabel("Noll index j")
    ax_sv.set_ylabel("σ")
    ax_sv.set_xticks(range(1, j + 1))
    ax_sv.legend(fontsize=7)

    # --- Pupil point scatter ---
    xy = d["xy"]
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax_pupil.plot(np.cos(theta_circle), np.sin(theta_circle), "k-", linewidth=0.8)
    ax_pupil.scatter(xy[:, 0], xy[:, 1], s=4, c=np.arange(n), cmap="viridis", linewidths=0)
    ax_pupil.set_aspect("equal")
    ax_pupil.set_title(f"Pupil samples  N={n}", fontsize=9)
    ax_pupil.set_xlim(-1.15, 1.15)
    ax_pupil.set_ylim(-1.15, 1.15)
    ax_pupil.set_xlabel("x (unit disk)")
    ax_pupil.set_ylabel("y (unit disk)")

    # --- Radial CDF check: P(ρ < r) should equal r² for uniform area coverage ---
    rho = d["rho"]
    rho2_sorted = np.sort(rho**2)
    cdf_empirical = np.arange(1, n + 1) / n
    ax_rho_cdf.plot([0, 1], [0, 1], "r--", linewidth=1.0, label="ideal (ρ² uniform)")
    ax_rho_cdf.plot(rho2_sorted, cdf_empirical, "b-", linewidth=0.9, label="empirical")
    ax_rho_cdf.set_title("Radial CDF  P(ρ² < t)", fontsize=9)
    ax_rho_cdf.set_xlabel("t = ρ²")
    ax_rho_cdf.set_ylabel("CDF")
    ax_rho_cdf.legend(fontsize=7)
    max_dev = float(np.abs(cdf_empirical - rho2_sorted).max())
    ax_rho_cdf.set_title(f"Radial CDF  P(ρ² < t)\nmax KS dev={max_dev:.4f}", fontsize=9)


def _summary_string(d: dict) -> str:
    """Return the same text that print_summary() prints, as a plain string."""
    import io
    buf = io.StringIO()
    _orig = sys.stdout
    sys.stdout = buf
    print_summary(d)
    sys.stdout = _orig
    return buf.getvalue()


def run_single(
    num_rays: int,
    j_max: int,
    save_path: str | None = None,
    txt_path: str | None = None,
) -> None:
    d = compute_diagnostics(num_rays, j_max)
    summary = _summary_string(d)
    print(summary, end="")

    if txt_path:
        Path(txt_path).write_text(summary, encoding="utf-8")
        print(f"  Text  → {txt_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_single(d, axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
    fig.suptitle(
        f"Zernike Pupil Sampling Diagnostics  —  N={num_rays}, j_max={j_max}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  PNG   → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_sweep(
    num_rays_list: list[int],
    j_max: int,
    save_path: str | None = None,
    txt_path: str | None = None,
    per_n_dir: str | None = None,
) -> None:
    """Plot off-diagonal max and condition number vs num_rays for a fixed j_max.

    Saves:
      sweep PNG          → save_path  (the two-panel summary plot)
      per-N detail PNGs  → per_n_dir/diag_N<N>_j<j_max>.png
      combined text      → txt_path
    """
    results = [compute_diagnostics(n, j_max) for n in num_rays_list]

    # --- text summaries ---
    all_summaries = "".join(_summary_string(r) for r in results)
    print(all_summaries, end="")
    if txt_path:
        Path(txt_path).write_text(all_summaries, encoding="utf-8")
        print(f"  Text  → {txt_path}")

    # --- per-N individual diagnostic figures ---
    if per_n_dir:
        pdir = Path(per_n_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for r in results:
            n = r["num_rays"]
            fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
            plot_single(r, axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1])
            fig2.suptitle(
                f"Zernike Pupil Sampling Diagnostics  —  N={n}, j_max={j_max}",
                fontsize=11, fontweight="bold"
            )
            fig2.tight_layout()
            p = pdir / f"diag_N{n:04d}_j{j_max}.png"
            fig2.savefig(p, dpi=150)
            print(f"  PNG   → {p}")
            plt.close(fig2)

    # --- sweep summary plot ---
    off_maxs = [r["off_diag_max"] for r in results]
    conds = [r["condition_number"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(num_rays_list, off_maxs, "o-", color="steelblue")
    ax1.axhline(0.02, color="orange", linestyle="--", linewidth=0.8, label="0.02 threshold")
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=0.8, label="0.05 threshold")
    ax1.set_xlabel("num_rays (N)")
    ax1.set_ylabel("max |G_ij| (i≠j)")
    ax1.set_title(f"Gram cross-talk vs N  (j_max={j_max})")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")

    ax2.plot(num_rays_list, conds, "o-", color="darkorange")
    ax2.axhline(20, color="orange", linestyle="--", linewidth=0.8, label="κ=20")
    ax2.axhline(100, color="red", linestyle="--", linewidth=0.8, label="κ=100")
    ax2.set_xlabel("num_rays (N)")
    ax2.set_ylabel("condition number κ(Z)")
    ax2.set_title(f"Condition number vs N  (j_max={j_max})")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")

    fig.suptitle(f"Sampling quality sweep  —  j_max={j_max}", fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  PNG   → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify Zernike pupil sampling quality (Gram matrix + SVD).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single config, display interactively
  python verify_pupil_sampling.py --num_rays 256 --j_max 16

  # Single config, save PNG + text
  python verify_pupil_sampling.py --num_rays 256 --j_max 16 --out_dir results/pupil_diag

  # Sweep, save everything (sweep PNG, per-N PNGs, combined text)
  python verify_pupil_sampling.py --num_rays 32 64 128 256 512 1024 --j_max 16 --out_dir results/pupil_diag
""",
    )
    parser.add_argument(
        "--num_rays", type=int, nargs="+", default=[256],
        help="Number of pupil sample points. Multiple values → sweep mode.",
    )
    parser.add_argument(
        "--j_max", type=int, default=NOLL_N_TERMS,
        help=f"Highest Noll index to include (1-based, max {NOLL_N_TERMS}).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help=(
            "Directory to save all outputs (PNGs + text). "
            "Created automatically if it does not exist. "
            "If omitted, figures are shown interactively and text goes to stdout only."
        ),
    )
    args = parser.parse_args(argv)

    out = Path(args.out_dir) if args.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {out.resolve()}")

    if len(args.num_rays) == 1:
        n = args.num_rays[0]
        png = str(out / f"diag_N{n:04d}_j{args.j_max}.png") if out else None
        txt = str(out / f"diag_N{n:04d}_j{args.j_max}.txt") if out else None
        run_single(n, args.j_max, save_path=png, txt_path=txt)
    else:
        sweep_png = str(out / f"sweep_j{args.j_max}.png") if out else None
        sweep_txt = str(out / f"sweep_j{args.j_max}.txt") if out else None
        per_n = str(out) if out else None
        run_sweep(
            args.num_rays, args.j_max,
            save_path=sweep_png,
            txt_path=sweep_txt,
            per_n_dir=per_n,
        )


if __name__ == "__main__":
    main()
