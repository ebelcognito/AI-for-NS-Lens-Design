"""Verify Zernike OPD decomposition on specific singlet lens designs.

Loads lens designs from an archive (or from manually specified thetas), traces
real ray bundles, and reports:
  - Pupil health (valid fraction, exit-pupil radius, centroid offset, radial KS)
  - Gram matrix orthogonality from the valid-ray entrance-pupil subset
  - Zernike coefficients (all j=1..j_max) and reconstruction quality
  - Comparison to archived measures for the descriptor indices

Outputs per design: a 2x3 PNG figure + printed summary text.  All outputs
saved under --out_dir.

Usage examples
--------------
# From archive entries by 0-based index:
  python3 optical_rl/qd/verify_zernike_on_designs.py \\
      --config optical_rl/configs/qd_singlet_cmamega_two_param.yaml \\
      --archive results/models/.../archive.json \\
      --entry_indices 5 22 28 \\
      --field_deg 2.0 \\
      --out_dir results/zernike_verify

# Direct theta input:
  python3 optical_rl/qd/verify_zernike_on_designs.py \\
      --config optical_rl/configs/qd_singlet_cmamega_two_param.yaml \\
      --thetas 0.084,-0.253 -0.068,-0.471 0.207,0.011 \\
      --field_deg 2.0 \\
      --out_dir results/zernike_verify
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.utils.config import load_config
from optical_rl.qd.singlet_cmamega import SingletCMAMEGAConfig, _disk_samples
from optical_rl.qd.singlet_cmamega_two_param import TwoParamSingletProblem
from optical_rl.qd.singlet_zernike_measures import (
    NOLL_N_TERMS,
    NOLL_TERM_NAMES,
    _noll_zernike_matrix,
)


# ---------------------------------------------------------------------------
# OPD pipeline — mirrors _zernike_measures but returns all intermediates
# ---------------------------------------------------------------------------

def _compute_opd_intermediates(problem: TwoParamSingletProblem, lens, field_deg: float) -> dict:
    """Replicate _zernike_measures and return intermediate arrays as numpy."""
    torch = problem._torch

    with torch.no_grad():
        field = float(field_deg)
        z_obj_val = float(problem.config.object_depth_mm)
        z_obj_abs = abs(z_obj_val)
        y_obj = z_obj_val * math.tan(math.radians(field))

        # 1. Coherent trace → sensor → exit pupil
        ray = problem._trace_coherent(lens, field_x_deg=0.0, field_y_deg=field)
        pupil_z, _ = lens.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        opl = ray.opl.squeeze(-1)
        valid_float = ray.is_valid.float()
        valid_mask = ray.is_valid.detach().cpu().numpy().astype(bool)
        x_ep = ray.o[:, 0].to(torch.float64)
        y_ep = ray.o[:, 1].to(torch.float64)

        # 2. Restore incoming OPL (object → stop) not accumulated because rays start at stop
        xy_stop = problem._aperture_xy.to(torch.float64)
        xp, yp = xy_stop[:, 0], xy_stop[:, 1]
        opl_marginal_in = torch.sqrt(xp ** 2 + (yp - y_obj) ** 2 + z_obj_abs ** 2)
        opl_chief_in = math.sqrt(y_obj ** 2 + z_obj_abs ** 2)
        opl = opl + (opl_marginal_in - opl_chief_in)

        # 3. Chief ray at configured field angle
        if field == 0.0:
            chief_rfov = torch.tensor([0.0], device=problem._device)
        else:
            chief_rfov = torch.tensor([0.0, field], device=problem._device)
        chief_o, chief_d = lens.calc_chief_ray_infinite(chief_rfov, plane="meridional")
        chief_ray_obj = problem._Ray(
            chief_o[-1:].to(torch.float64),
            chief_d[-1:].to(torch.float64),
            wvln=float(problem.config.wavelength_um),
            coherent=True,
            device=problem._device,
        )
        chief_ray_obj = lens.trace2sensor(chief_ray_obj)
        chief_ray_obj = chief_ray_obj.prop_to(pupil_z)
        opl_chief = chief_ray_obj.opl.squeeze()
        x_chief_ep = float(chief_ray_obj.o[0, 0].to(torch.float64).item())
        y_chief_ep = float(chief_ray_obj.o[0, 1].to(torch.float64).item())

        # 4. Reference sphere centred on chief ray's exit-pupil position
        BFL = lens.d_sensor - pupil_z
        ref_sphere = torch.sqrt(
            (x_ep - x_chief_ep) ** 2 + (y_ep - y_chief_ep) ** 2 + BFL ** 2
        ) - BFL

        # 5. OPD (invalid rays zeroed)
        opd_t = ((opl - opl_chief) + ref_sphere) * valid_float

    return {
        "opd": opd_t.detach().cpu().numpy().astype(np.float64),
        "valid_mask": valid_mask,
        "x_ep": x_ep.detach().cpu().numpy().astype(np.float64),
        "y_ep": y_ep.detach().cpu().numpy().astype(np.float64),
        "x_chief_ep": x_chief_ep,
        "y_chief_ep": y_chief_ep,
        "pupil_z": float(pupil_z),
    }


# ---------------------------------------------------------------------------
# Pupil health metrics — O(N), fast
# ---------------------------------------------------------------------------

def _compute_pupil_health(res: dict, disk_unit: np.ndarray) -> dict:
    """Fast pupil-health proxy metrics.

    valid_frac         fraction of rays not vignetted        (> 0.95 = healthy)
    ep_radius_95_mm    exit-pupil 95th-pct radius from chief (mm)
    centroid_offset_mm distance of valid-ray centroid from chief exit point (mm)
    radial_ks          max |empirical CDF(rho²) - rho²| for valid entrance-pupil rho
                       (uniform disk → ≈0;  > 0.05 = non-uniform)
    """
    valid = res["valid_mask"]
    valid_frac = float(valid.mean())

    if valid.any():
        x_ep_v = res["x_ep"][valid] - res["x_chief_ep"]
        y_ep_v = res["y_ep"][valid] - res["y_chief_ep"]
        rho_ep_v = np.sqrt(x_ep_v ** 2 + y_ep_v ** 2)
        ep_radius_95 = float(np.percentile(rho_ep_v, 95))
        centroid_offset = math.sqrt(float(x_ep_v.mean()) ** 2 + float(y_ep_v.mean()) ** 2)

        rho_ent_v = np.sqrt(disk_unit[valid, 0] ** 2 + disk_unit[valid, 1] ** 2)
        rho2 = np.sort(rho_ent_v ** 2)
        ecdf = np.arange(1, len(rho2) + 1) / len(rho2)
        radial_ks = float(np.max(np.abs(ecdf - rho2)))
    else:
        ep_radius_95 = 0.0
        centroid_offset = 0.0
        radial_ks = 1.0

    return {
        "valid_frac": valid_frac,
        "ep_radius_95_mm": ep_radius_95,
        "centroid_offset_mm": centroid_offset,
        "radial_ks": radial_ks,
        "n_valid": int(valid.sum()),
        "n_total": int(valid.size),
    }


# ---------------------------------------------------------------------------
# Gram matrix + Zernike fit
# ---------------------------------------------------------------------------

def _compute_gram_and_fit(res: dict, disk_unit: np.ndarray, j_max: int) -> dict:
    """Fit j_max Zernike coefficients; compute Gram matrix from valid entrance-pupil subset.

    Returns:
        coeffs           (j_max,) fitted Zernike coefficients
        gram             (j_max, j_max) Gram matrix from valid-ray entrance-pupil coords
        condition_number SVD condition number of Z_valid
        off_diag_max     max |G_ij| for i≠j
        diag_res_max     max |G_ii - 1|
        opd_rms          RMS OPD over valid rays
        recon_rms        RMS reconstruction residual over valid rays
        recon_ratio      recon_rms / opd_rms  (ideal → 0)
    """
    valid = res["valid_mask"]
    opd = res["opd"]

    # Entrance-pupil Fibonacci coords (full N)
    rho_fib = np.sqrt(disk_unit[:, 0] ** 2 + disk_unit[:, 1] ** 2)
    phi_fib = np.arctan2(disk_unit[:, 1], disk_unit[:, 0])

    Z_full = _noll_zernike_matrix(rho_fib, phi_fib, j_max)   # (N, j_max)
    pinv_Z_full = np.linalg.pinv(Z_full)                      # (j_max, N)
    coeffs = pinv_Z_full @ opd                                 # (j_max,)

    opd_hat = Z_full @ coeffs                                  # (N,)
    residual = opd - opd_hat

    opd_rms = float(np.sqrt(np.mean(opd[valid] ** 2))) if valid.any() else 0.0
    recon_rms = float(np.sqrt(np.mean(residual[valid] ** 2))) if valid.any() else 0.0
    recon_ratio = recon_rms / opd_rms if opd_rms > 1e-18 else float("nan")

    # Gram matrix from real exit-pupil positions of valid rays, normalised to unit disk.
    # This reflects the actual sampling geometry produced by this specific lens design,
    # not the synthetic Fibonacci entrance-pupil grid (which is design-invariant).
    n_v = int(valid.sum())
    if n_v >= j_max:
        x_ep_v = res["x_ep"][valid] - res["x_chief_ep"]
        y_ep_v = res["y_ep"][valid] - res["y_chief_ep"]
        ep_radius = float(np.percentile(np.sqrt(x_ep_v ** 2 + y_ep_v ** 2), 95))
        if ep_radius < 1e-12:
            ep_radius = 1.0
        rho_real = np.sqrt(x_ep_v ** 2 + y_ep_v ** 2) / ep_radius
        phi_real = np.arctan2(y_ep_v, x_ep_v)
        Z_valid = _noll_zernike_matrix(rho_real, phi_real, j_max)
        gram = (Z_valid.T @ Z_valid) / n_v
        sv = np.linalg.svd(Z_valid, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
        off_mask = ~np.eye(j_max, dtype=bool)
        off_diag_max = float(np.abs(gram[off_mask]).max())
        diag_res_max = float(np.abs(np.diag(gram) - 1.0).max())
    else:
        gram = np.full((j_max, j_max), float("nan"))
        cond = float("inf")
        off_diag_max = float("nan")
        diag_res_max = float("nan")

    return {
        "coeffs": coeffs,
        "gram": gram,
        "condition_number": cond,
        "off_diag_max": off_diag_max,
        "diag_res_max": diag_res_max,
        "opd_rms": opd_rms,
        "recon_rms": recon_rms,
        "recon_ratio": recon_ratio,
    }


# ---------------------------------------------------------------------------
# 2x3 panel figure
# ---------------------------------------------------------------------------

def _plot_design(
    res: dict,
    health: dict,
    gram_result: dict,
    disk_unit: np.ndarray,
    j_max: int,
    descriptor_indices: tuple,
    title: str,
    out_path: Path,
    archived_measures: Optional[tuple] = None,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=10, y=0.99)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    valid = res["valid_mask"]
    opd = res["opd"]
    x_ep = res["x_ep"]
    y_ep = res["y_ep"]
    x_c = res["x_chief_ep"]
    y_c = res["y_chief_ep"]
    coeffs = gram_result["coeffs"]

    rho_fib = np.sqrt(disk_unit[:, 0] ** 2 + disk_unit[:, 1] ** 2)
    phi_fib = np.arctan2(disk_unit[:, 1], disk_unit[:, 0])
    Z_full = _noll_zernike_matrix(rho_fib, phi_fib, j_max)
    opd_hat = Z_full @ coeffs
    residual = opd - opd_hat

    vabs = float(np.abs(opd[valid]).max()) if valid.any() else 1e-10

    def _scatter(ax, values, cmap, vmin, vmax, label, cbar_label="OPD (mm)"):
        sc = ax.scatter(
            x_ep[valid] - x_c, y_ep[valid] - y_c,
            c=values[valid], cmap=cmap,
            vmin=vmin, vmax=vmax, s=8, linewidths=0,
        )
        ax.scatter(0, 0, marker="+", c="k", s=60, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x_ep − x_chief  (mm)", fontsize=8)
        ax.set_ylabel("y_ep − y_chief  (mm)", fontsize=8)
        ax.set_title(label, fontsize=9)
        plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.05)

    _scatter(fig.add_subplot(gs[0, 0]), opd, "RdBu_r", -vabs, vabs, "OPD (measured)")
    _scatter(fig.add_subplot(gs[0, 1]), opd_hat, "RdBu_r", -vabs, vabs, "OPD (reconstructed)")

    ax_res = fig.add_subplot(gs[0, 2])
    res_abs = float(np.abs(residual[valid]).max()) if valid.any() else 1e-10
    _scatter(ax_res, residual, "PiYG", -res_abs, res_abs,
             f"Residual  (RMS={gram_result['recon_rms']:.2e} mm)", "residual (mm)")

    # Zernike bar chart
    ax_bar = fig.add_subplot(gs[1, 0])
    js = np.arange(1, j_max + 1)
    bar_colors = ["tab:orange" if j in descriptor_indices else "steelblue" for j in js]
    ax_bar.bar(js, coeffs, color=bar_colors, edgecolor="none")
    ax_bar.axhline(0, color="k", linewidth=0.7)
    ax_bar.set_xticks(js)
    ax_bar.set_xticklabels([str(j) for j in js], fontsize=7)
    ax_bar.set_xlabel("Noll index j", fontsize=8)
    ax_bar.set_ylabel("Coefficient (mm)", fontsize=8)
    ax_bar.set_title("Zernike coefficients  (orange = descriptors)", fontsize=9)
    if archived_measures is not None:
        j0, j1 = descriptor_indices
        ax_bar.annotate(
            f"archived j{j0}={archived_measures[0]:.3e}\narchived j{j1}={archived_measures[1]:.3e}",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
        )

    # Radial CDF (entrance pupil, valid rays)
    ax_cdf = fig.add_subplot(gs[1, 1])
    if valid.any():
        rho_v = rho_fib[valid]
        rho2_sorted = np.sort(rho_v ** 2)
        n_v = len(rho2_sorted)
        ecdf = np.arange(1, n_v + 1) / n_v
        ax_cdf.plot(rho2_sorted, ecdf, lw=1.5, label="empirical")
        ax_cdf.plot([0, 1], [0, 1], "k--", lw=1, label="ideal uniform")
        ax_cdf.set_xlim(0, 1)
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel(r"$\rho^2$  (entrance pupil, valid rays)", fontsize=8)
        ax_cdf.set_ylabel("CDF", fontsize=8)
        ax_cdf.set_title(f"Radial CDF  (KS={health['radial_ks']:.4f})", fontsize=9)
        ax_cdf.legend(fontsize=8)
        ax_cdf.text(0.98, 0.03,
                    f"valid {health['n_valid']}/{health['n_total']} "
                    f"({100*health['valid_frac']:.1f}%)",
                    ha="right", va="bottom", transform=ax_cdf.transAxes, fontsize=8)

    # Gram matrix heatmap
    ax_gram = fig.add_subplot(gs[1, 2])
    gram = gram_result["gram"]
    if not np.any(np.isnan(gram)):
        im = ax_gram.imshow(gram, vmin=-0.05, vmax=1.05, cmap="bwr", aspect="auto")
        ax_gram.set_xlabel("j index", fontsize=8)
        ax_gram.set_ylabel("j index", fontsize=8)
        ax_gram.set_title(
            f"Gram (ZᵀZ/N, real exit pupil)  off_diag={gram_result['off_diag_max']:.4f}  "
            f"κ={gram_result['condition_number']:.1f}",
            fontsize=9,
        )
        plt.colorbar(im, ax=ax_gram, fraction=0.05)
    else:
        ax_gram.text(0.5, 0.5, "not enough valid rays",
                     ha="center", va="center", transform=ax_gram.transAxes)
        ax_gram.set_title("Gram matrix", fontsize=9)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def _summary_lines(
    label: str,
    theta: np.ndarray,
    field_deg: float,
    health: dict,
    gram_result: dict,
    descriptor_indices: tuple,
    archived_measures: Optional[tuple],
) -> list:
    coeffs = gram_result["coeffs"]
    j0, j1 = descriptor_indices
    c0 = float(coeffs[j0 - 1])
    c1v = float(coeffs[j1 - 1])

    lines = [
        f"=== {label} ===",
        f"  theta              : {np.array2string(theta, precision=6, separator=', ')}",
        f"  field_deg          : {field_deg:.3f}°",
        "",
        "  --- Pupil health ---",
        f"  valid_frac         : {health['valid_frac']:.4f}  ({health['n_valid']}/{health['n_total']} rays)",
        f"  ep_radius_95       : {health['ep_radius_95_mm']:.4f} mm",
        f"  centroid_offset    : {health['centroid_offset_mm']:.4e} mm",
        f"  radial_KS          : {health['radial_ks']:.4f}  (ideal ≈ 0)",
        "",
        "  --- Zernike fit quality ---",
        f"  cond(Z_valid)      : {gram_result['condition_number']:.2f}",
        f"  Gram off_diag_max  : {gram_result['off_diag_max']:.4f}  (ideal ≈ 0)",
        f"  Gram diag_res_max  : {gram_result['diag_res_max']:.4f}  (ideal ≈ 0)",
        f"  OPD RMS            : {gram_result['opd_rms']:.4e} mm",
        f"  Reconstruction RMS : {gram_result['recon_rms']:.4e} mm",
        f"  Recon/OPD ratio    : {gram_result['recon_ratio']:.4f}  (ideal → 0)",
        "",
        "  --- Descriptor coefficients ---",
        f"  j{j0} ({NOLL_TERM_NAMES.get(j0, '?')}) fresh   : {c0:.6e} mm",
        f"  j{j1} ({NOLL_TERM_NAMES.get(j1, '?')}) fresh   : {c1v:.6e} mm",
    ]
    if archived_measures is not None and archived_measures[0] is not None:
        lines += [
            f"  j{j0} archived        : {float(archived_measures[0]):.6e} mm"
            f"  Δ = {c0 - float(archived_measures[0]):.2e}",
            f"  j{j1} archived        : {float(archived_measures[1]):.6e} mm"
            f"  Δ = {c1v - float(archived_measures[1]):.2e}",
        ]
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Verify Zernike OPD decomposition on singlet designs."
    )
    p.add_argument("--config", required=True,
                   help="Path to YAML config (qd_singlet_cmamega_two_param.yaml).")
    p.add_argument("--out_dir", default="results/zernike_verify",
                   help="Directory for output files.")
    p.add_argument("--j_max", type=int, default=NOLL_N_TERMS,
                   help=f"Number of Zernike terms to fit (default {NOLL_N_TERMS}).")
    p.add_argument("--field_deg", type=float, default=None,
                   help="Field angle override (degrees). If omitted uses config's zernike_field_deg.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", metavar="PATH",
                     help="archive.json path. Use with --entry_indices, or omit to verify all entries.")
    src.add_argument("--thetas", nargs="+", metavar="c1,c2",
                     help="Comma-separated theta pairs e.g.  0.084,-0.253  -0.068,-0.471")

    p.add_argument("--entry_indices", nargs="+", type=int, metavar="I",
                   help="0-based archive entry indices (used with --archive).")
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    raw = load_config(args.config)
    qd_cfg = raw.get("qd", raw)

    config = SingletCMAMEGAConfig(
        lens_path=str(qd_cfg["lens_path"]),
        object_depth_mm=float(qd_cfg.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd_cfg.get("wavelength_um", 0.587)),
        theta_bounds=tuple(tuple(b) for b in qd_cfg.get("theta_bounds", [[-0.5, 0.5], [-0.5, 0.5]])),
        focal_length_target_mm=qd_cfg.get("focal_length_target_mm"),
        descriptor_zernike_indices=tuple(
            int(x) for x in qd_cfg.get("descriptor_zernike_indices", [7, 4])
        ),
        num_rays=int(qd_cfg.get("num_rays", 256)),
        zernike_field_deg=float(qd_cfg.get("zernike_field_deg", 0.0)),
        device=str(qd_cfg.get("device", "auto")),
    )

    problem = TwoParamSingletProblem(config)
    problem._ensure_backend()

    field_deg = args.field_deg if args.field_deg is not None else float(config.zernike_field_deg)
    j_max = args.j_max
    descriptor_indices = tuple(config.descriptor_zernike_indices)

    print(f"Config           : {args.config}")
    print(f"Zernike field    : {field_deg}°")
    print(f"j_max            : {j_max}")
    print(f"Descriptor indices: {descriptor_indices}")
    print(f"Output dir       : {out_dir}")
    print()

    # Collect designs
    designs: list = []

    if args.archive:
        archive_path = Path(args.archive)
        with archive_path.open() as f:
            archive = json.load(f)
        indices = args.entry_indices if args.entry_indices else list(range(len(archive)))
        for idx in indices:
            entry = archive[idx]
            theta = np.array(entry["theta"], dtype=np.float64)
            measures = entry.get("measures", [None, None])
            cell = entry.get("cell", "?")
            label = f"entry{idx:04d}_cell{cell}"
            designs.append({
                "theta": theta,
                "label": label,
                "archived_measures": tuple(measures),
            })
    else:
        for raw_t in args.thetas:
            parts = raw_t.split(",")
            theta = np.array([float(v) for v in parts], dtype=np.float64)
            label = "theta_" + "_".join(f"{v:.5f}" for v in theta)
            designs.append({
                "theta": theta,
                "label": label,
                "archived_measures": None,
            })

    # Fibonacci entrance-pupil grid (fixed for this config)
    disk_unit = _disk_samples(config.num_rays)

    all_lines: list = []

    for i, design in enumerate(designs, 1):
        theta = design["theta"]
        label = design["label"]
        archived = design["archived_measures"]

        print(f"[{i}/{len(designs)}]  {label}  theta={np.round(theta, 5)}")

        lens = problem.theta_to_lens(theta)
        res = _compute_opd_intermediates(problem, lens, field_deg)
        health = _compute_pupil_health(res, disk_unit)
        gram_result = _compute_gram_and_fit(res, disk_unit, j_max)

        lines = _summary_lines(
            label, theta, field_deg, health, gram_result, descriptor_indices, archived
        )
        for ln in lines:
            print(ln)
        all_lines.extend(lines)

        fig_path = out_dir / f"{label}.png"
        archived_tup = None
        if archived and archived[0] is not None:
            archived_tup = (float(archived[0]), float(archived[1]))

        fig_title = (
            f"{label}  |  θ={np.round(theta, 5)}  field={field_deg}°\n"
            f"valid={100*health['valid_frac']:.1f}%  "
            f"recon_ratio={gram_result['recon_ratio']:.4f}  "
            f"Gram_off_diag={gram_result['off_diag_max']:.4f}"
        )
        _plot_design(
            res, health, gram_result, disk_unit, j_max,
            descriptor_indices, fig_title, fig_path, archived_tup,
        )
        print(f"  → figure: {fig_path}")
        print()

    txt_path = out_dir / "summary.txt"
    with txt_path.open("w") as f:
        f.write(f"Zernike verification | config: {args.config}\n")
        f.write(f"field_deg: {field_deg}°   j_max: {j_max}\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(all_lines))
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    main()
