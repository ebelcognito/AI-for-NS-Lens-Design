"""Refine diverse EF50 archive elites with L-BFGS-B and identify distinct local minima.

Standalone script — no changes to any existing file.

Pipeline:
  1. Load a computed archive_cells.json (auto-discovers latest if not specified).
  2. Pick `count` diverse elites spread across the whole archive.
  3. Refine each with scipy L-BFGS-B (equivalent to Zemax DLS) minimising
     the stage-2 all-field RMS spot.  EFL stays fixed at 50.52 mm throughout.
  4. Compute the pairwise L2 distance matrix of the converged Zernike vectors.
  5. Label each converged design by which local-minimum basin it fell into.
  6. Save results.json, a pairwise-distance heatmap, and an archive-basin map.

Usage:
  # auto-discover latest archive + its YAML config
  python3 optical_rl/qd/refine_ef50_archive.py

  # explicit paths
  python3 optical_rl/qd/refine_ef50_archive.py \\
      --archive results/models/ef50_zernike_*/archive_cells.json \\
      --count 20 --max-iter 200 --num-rays 256
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Reuse existing helpers
# ---------------------------------------------------------------------------

from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    select_diverse_top,
)
from optical_rl.qd.render_ef50_zernike import (  # noqa: E402
    _find_latest_archive,
    _load_qd_config,
    load_ef50_archive,
)
from optical_rl.qd.ef50_zernike import (  # noqa: E402
    EF50ZernikeConfig,
    EF50ZernikeProblem,
)
from optical_rl.qd.singlet_zernike_measures import NOLL_TERM_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def _build_problem(qd_cfg: dict, num_rays: int) -> EF50ZernikeProblem:
    """Instantiate EF50ZernikeProblem for local refinement.

    Uses descriptor_zernike_indices=(7, 11) so that _zernike_measures returns
    Z1..Z11 — a richer vector for clustering.
    """
    fl_raw = qd_cfg.get("focal_length_target_mm")
    focal_length_target_mm = float(fl_raw) if fl_raw is not None else None

    theta_dim = 10 if focal_length_target_mm is not None else 11
    theta_bounds_default = [[-0.09, 0.09]] * theta_dim

    config = EF50ZernikeConfig(
        lens_path=str(
            qd_cfg.get(
                "lens_path",
                str(PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"),
            )
        ),
        object_depth_mm=float(qd_cfg.get("object_depth_mm", -20000.0)),
        wavelength_um=float(qd_cfg.get("wavelength_um", 0.587)),
        fitness_wavelengths_um=tuple(
            float(x) for x in qd_cfg.get("fitness_wavelengths_um", [0.656, 0.587, 0.486])
        ),
        theta_bounds=tuple(
            tuple(b) for b in qd_cfg.get("theta_bounds", theta_bounds_default)
        ),
        stop_surface_index=int(qd_cfg.get("stop_surface_index", 6)),
        focal_length_target_mm=focal_length_target_mm,
        # Override to (7, 11) so j_max=11 — gives Z1..Z11 for post-refine clustering.
        descriptor_zernike_indices=(7, 11),
        archive_bins=(64, 64),     # unused by refiner
        archive_bounds=None,
        invalid_tol=float(qd_cfg.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd_cfg.get("invalid_penalty_weight", 5.0)),
        intersect_penalty_weight=float(qd_cfg.get("intersect_penalty_weight", 1.0)),
        air_min_mm=float(qd_cfg.get("air_min_mm", 0.1)),
        thick_min_mm=float(qd_cfg.get("thick_min_mm", 1.0)),
        cma_eta=0.05,
        seed_count=0,
        calibration_count=0,
        num_rays=num_rays,
        stage="stage2",            # always all-field for refinement
        zernike_field_index=int(qd_cfg.get("zernike_field_index", 1)),
        device=str(qd_cfg.get("device", "auto")),
        initial_sigma=0.75,
        population_size=16,
        cma_seed=0,
        results_dir="results/models",
        run_name="ef50_zernike",
        log_every=50,
    )
    problem = EF50ZernikeProblem(config)
    problem._ensure_backend()
    return problem


# ---------------------------------------------------------------------------
# L-BFGS-B objective
# ---------------------------------------------------------------------------

def _make_objective(problem: EF50ZernikeProblem):
    """Return a scipy-compatible (loss, grad) callable for L-BFGS-B.

    Minimises the stage-2 all-field RMS spot directly via PyTorch autograd.
    The EFL constraint is maintained inside _build_lens_from_theta_tensor
    exactly as during the archive search — no extra work needed.
    """
    torch = problem._torch

    def objective(theta_np: np.ndarray):
        theta_t = torch.tensor(
            theta_np,
            dtype=torch.float32,
            device=problem._device,
            requires_grad=True,
        )

        # Guard against degenerate configurations (BFL <= 0).
        if problem._f_target is not None:
            _, _, bfl_t = problem._paraxial_fixed_efl(theta_t)
            bfl_val = float(bfl_t.detach().item())
        else:
            bfl_val = float(
                (problem._paraxial_bfl(theta_t) - problem._z_s11).detach().item()
            )
        if bfl_val <= 0.0:
            return 1e6, np.zeros_like(theta_np, dtype=np.float64)

        lens = problem._build_lens_from_theta_tensor(theta_t)
        hfov = problem._calc_hfov_deg(theta_t)

        wavelengths = problem.config.fitness_wavelengths_um
        field_pts = problem._field_points(hfov)
        rms_per_wvln, inv_per_wvln = [], []
        for w in wavelengths:
            ray = problem._trace_fields_batch(lens, field_pts, wvln=w)
            rms, inv_frac = problem._rms_spot_batch(ray)
            rms_per_wvln.append(rms)
            inv_per_wvln.append(inv_frac)

        rms_mean = torch.stack(rms_per_wvln).mean()
        inv_mean = torch.stack(inv_per_wvln).mean()

        # Mirror _fitness_and_metrics in ef50_zernike.py: CMA maximizes
        # (-rms - intersect_penalty - invalid_penalty), L-BFGS-B minimizes
        # the equivalent (rms + intersect_penalty + invalid_penalty).
        loss_t = rms_mean

        intersect_loss = problem._intersection_penalty(lens)
        if float(intersect_loss.detach()) > 0:
            loss_t = loss_t + problem.config.intersect_penalty_weight * intersect_loss

        inv_tol = float(problem.config.invalid_tol)
        if float(inv_mean.detach()) > inv_tol:
            loss_t = loss_t + problem.config.invalid_penalty_weight * (inv_mean - inv_tol)

        loss_t.backward()

        loss = float(loss_t.detach().cpu().item())
        grad = theta_t.grad.detach().cpu().numpy().astype(np.float64)
        return loss, grad

    return objective


# ---------------------------------------------------------------------------
# Post-convergence Zernike measurement
# ---------------------------------------------------------------------------

def _full_zernike_vector(
    problem: EF50ZernikeProblem, theta_np: np.ndarray
) -> np.ndarray:
    """Compute the full Zernike coefficient vector (Z1..Z11) at converged theta."""
    torch = problem._torch
    theta_t = torch.tensor(
        theta_np, dtype=torch.float32, device=problem._device
    )
    lens = problem._build_lens_from_theta_tensor(theta_t)
    hfov = problem._calc_hfov_deg(theta_t)
    coeffs = problem._zernike_measures(lens, hfov)
    return coeffs.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Single-design refiner
# ---------------------------------------------------------------------------

def refine_design(
    problem: EF50ZernikeProblem,
    start_theta: np.ndarray,
    max_iter: int,
    gtol: float,
    record_trajectory: bool = False,
    trajectory_every: int = 3,
    descriptor_indices: tuple[int, int] = (7, 11),
) -> dict:
    """Run L-BFGS-B from start_theta until convergence.

    Returns a dict with before/after metrics and the converged Zernike vector.
    """
    objective = _make_objective(problem)
    bounds = list(problem.config.theta_bounds)

    # Record pre-refinement RMS without gradients
    pre = problem.evaluate(start_theta, with_grads=False)
    start_rms = pre.metrics.get("rms_spot", float("nan"))

    trajectory_points: list[list[float]] = []
    step_counter = [0]
    j0, j1 = descriptor_indices

    if record_trajectory:
        z_start = _full_zernike_vector(problem, start_theta)
        trajectory_points.append([float(z_start[j0 - 1]), float(z_start[j1 - 1])])

    def callback(xk: np.ndarray) -> None:
        step_counter[0] += 1
        if record_trajectory and step_counter[0] % trajectory_every == 0:
            z = _full_zernike_vector(problem, xk)
            trajectory_points.append([float(z[j0 - 1]), float(z[j1 - 1])])

    result = minimize(
        objective,
        start_theta.copy().astype(np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
        callback=callback,
    )

    converged_theta = result.x
    converged_rms = float(result.fun)

    zernike_vec = _full_zernike_vector(problem, converged_theta)

    return {
        "start_theta": start_theta.tolist(),
        "converged_theta": converged_theta.tolist(),
        "start_rms_mm": float(start_rms),
        "converged_rms_mm": converged_rms,
        "zernike_vector": zernike_vec.tolist(),
        "n_iters": int(result.nit),
        "converged": bool(result.success),
        "scipy_message": str(result.message),
        "trajectory": trajectory_points,
    }


# ---------------------------------------------------------------------------
# Pairwise distance + basin labeling
# ---------------------------------------------------------------------------

def _pairwise_distances(z_matrix: np.ndarray) -> np.ndarray:
    N = z_matrix.shape[0]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(z_matrix[i] - z_matrix[j]))
            D[i, j] = d
            D[j, i] = d
    return D


def _label_basins(dist_matrix: np.ndarray, threshold: float) -> list[int]:
    N = dist_matrix.shape[0]
    labels = list(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if dist_matrix[i, j] < threshold:
                old, new = labels[j], labels[i]
                if old != new:
                    for k in range(N):
                        if labels[k] == old:
                            labels[k] = new
    remap: dict[int, int] = {}
    out: list[int] = []
    for lab in labels:
        if lab not in remap:
            remap[lab] = len(remap)
        out.append(remap[lab])
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_heatmap(
    dist_matrix: np.ndarray,
    basin_labels: list[int],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    N = dist_matrix.shape[0]
    n_basins = len(set(basin_labels))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_matrix, aspect="auto", cmap="viridis_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("L2 distance in Zernike space (mm)", fontsize=9)

    ax.set_title(
        f"Pairwise Zernike distance — converged EF50 designs\n"
        f"{N} elites → {n_basins} distinct local minima",
        fontsize=10,
    )
    ax.set_xlabel("Design index (basin label)", fontsize=9)
    ax.set_ylabel("Design index (basin label)", fontsize=9)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"B{b}" for b in basin_labels], fontsize=6, rotation=45)
    ax.set_yticklabels([f"B{b}" for b in basin_labels], fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_basins(
    all_designs,
    selected,
    results: list[dict],
    basin_labels: list[int],
    archive_bounds,
    descriptor_indices: tuple[int, int],
    out_path: Path,
    show_trajectories: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.cm as cm

    j0, j1 = descriptor_indices
    label0 = f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)"
    label1 = f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)"
    n_basins = len(set(basin_labels))
    colors = cm.tab10.colors
    after_color = "#E8C500"

    fig, ax = plt.subplots(figsize=(8, 7))

    # Faint background scatter of the full archive
    all_m = np.array([d.measures for d in all_designs], dtype=np.float64)
    all_f = np.array([d.fitness for d in all_designs], dtype=np.float64)
    ax.scatter(
        all_m[:, 0], all_m[:, 1],
        c=all_f, s=4, alpha=0.25, cmap="Greys", zorder=1,
    )

    # Trajectory polylines (optional)
    if show_trajectories:
        for res, basin in zip(results, basin_labels):
            traj = res.get("trajectory", [])
            if len(traj) < 2:
                continue
            col = colors[basin % len(colors)]
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=col, lw=1.0, alpha=0.6, zorder=1.5)
            ax.scatter(xs, ys, color=col, s=18, alpha=0.6, zorder=1.5)
            ax.scatter(xs[0], ys[0], color=col, s=50, marker="x",
                       linewidths=1.5, alpha=0.8, zorder=1.6)

    # Post-refinement measure pairs from stored zernike_vector (Z1..Z11)
    after_measures = []
    for res in results:
        z_vec = np.asarray(res.get("zernike_vector", []), dtype=np.float64)
        if len(z_vec) >= max(j0, j1):
            after_measures.append((float(z_vec[j0 - 1]), float(z_vec[j1 - 1])))
        else:
            after_measures.append(None)

    # Arrows: before → after
    for i, (design, res, after) in enumerate(zip(selected, results, after_measures)):
        if after is None:
            continue
        bx, by = design.measures
        ax_after, ay_after = after
        ax.annotate(
            "",
            xy=(ax_after, ay_after),
            xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="-|>",
                color="grey",
                lw=1.0,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

    # Before-points: circles colored by converged basin
    for i, (design, basin) in enumerate(zip(selected, basin_labels)):
        col = colors[basin % len(colors)]
        mx, my = design.measures
        ax.scatter(mx, my, color=col, s=120, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=3)

    # After-points: gold triangles
    for i, after in enumerate(after_measures):
        if after is None:
            continue
        ax.scatter(after[0], after[1], color=after_color, s=100, marker="^",
                   edgecolors="white", linewidths=1.0, zorder=4)

    # Basin labels next to before-points
    for i, (design, basin) in enumerate(zip(selected, basin_labels)):
        col = colors[basin % len(colors)]
        mx, my = design.measures
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = 0.002 * y_span if y_span > 0 else 0.001
        ax.text(mx, my + offset, f"B{basin}", fontsize=7, ha="center",
                va="bottom", color=col, zorder=5, fontweight="bold")

    # Legend
    basin_patches = [
        mpatches.Patch(color=colors[b % len(colors)], label=f"Basin {b}")
        for b in range(n_basins)
    ]
    before_handle = mlines.Line2D(
        [], [], color="grey", marker="o", markersize=8,
        markerfacecolor="grey", linestyle="None", label="Before refinement",
    )
    after_handle = mlines.Line2D(
        [], [], color=after_color, marker="^", markersize=8,
        markerfacecolor=after_color, linestyle="None", label="After refinement",
    )
    ax.legend(
        handles=basin_patches + [before_handle, after_handle],
        fontsize=8, loc="upper right", framealpha=0.8,
    )

    ax.set_xlabel(label0, fontsize=10)
    ax.set_ylabel(label1, fontsize=10)
    ax.set_title(
        f"EF50 Archive — before/after refinement positions in measure space\n"
        f"{len(selected)} elites → {n_basins} distinct basins",
        fontsize=10,
    )

    if archive_bounds is not None:
        (x_lo, x_hi), (y_lo, y_hi) = archive_bounds
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_theta_pca(
    all_designs,
    selected,
    results: list[dict],
    basin_labels: list[int],
    out_path: Path,
) -> None:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.cm as cm

    n_basins = len(set(basin_labels))
    colors = cm.tab10.colors
    after_color = "#E8C500"

    archive_thetas = np.array([d.theta for d in all_designs], dtype=np.float64)
    start_thetas = np.array([r["start_theta"] for r in results], dtype=np.float64)
    conv_thetas = np.array([r["converged_theta"] for r in results], dtype=np.float64)

    pca = PCA(n_components=2)
    pca.fit(archive_thetas)
    ev = pca.explained_variance_ratio_

    arch_2d = pca.transform(archive_thetas)
    start_2d = pca.transform(start_thetas)
    conv_2d = pca.transform(conv_thetas)

    label0 = f"PC1 ({ev[0] * 100:.1f}% var)"
    label1 = f"PC2 ({ev[1] * 100:.1f}% var)"

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(arch_2d[:, 0], arch_2d[:, 1], s=4, alpha=0.20, color="grey", zorder=1)

    for s_pt, c_pt in zip(start_2d, conv_2d):
        ax.annotate(
            "",
            xy=(c_pt[0], c_pt[1]),
            xytext=(s_pt[0], s_pt[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="grey",
                lw=1.0,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

    for s_pt, basin in zip(start_2d, basin_labels):
        col = colors[basin % len(colors)]
        ax.scatter(s_pt[0], s_pt[1], color=col, s=120, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=3)

    for c_pt in conv_2d:
        ax.scatter(c_pt[0], c_pt[1], color=after_color, s=100, marker="^",
                   edgecolors="white", linewidths=1.0, zorder=4)

    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = 0.002 * y_span if y_span > 0 else 0.001
    for s_pt, basin in zip(start_2d, basin_labels):
        col = colors[basin % len(colors)]
        ax.text(s_pt[0], s_pt[1] + offset, f"B{basin}", fontsize=7,
                ha="center", va="bottom", color=col, zorder=5, fontweight="bold")

    basin_patches = [
        mpatches.Patch(color=colors[b % len(colors)], label=f"Basin {b}")
        for b in range(n_basins)
    ]
    before_handle = mlines.Line2D(
        [], [], color="grey", marker="o", markersize=8,
        markerfacecolor="grey", linestyle="None", label="Before refinement",
    )
    after_handle = mlines.Line2D(
        [], [], color=after_color, marker="^", markersize=8,
        markerfacecolor=after_color, linestyle="None", label="After refinement",
    )
    ax.legend(
        handles=basin_patches + [before_handle, after_handle],
        fontsize=8, loc="upper right", framealpha=0.8,
    )

    ax.set_xlabel(label0, fontsize=10)
    ax.set_ylabel(label1, fontsize=10)
    ax.set_title(
        f"EF50 Parameter space (PCA) — before/after refinement\n"
        f"{len(results)} elites → {n_basins} distinct basins  |  "
        f"PC1+PC2 explain {(ev[0]+ev[1])*100:.1f}% of archive variance",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Refine diverse EF50 archive elites with L-BFGS-B "
            "and identify distinct local minima via pairwise Zernike distance."
        )
    )
    p.add_argument(
        "--archive", default=None,
        help="Path to archive_cells.json (default: latest ef50_zernike_* run)",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: auto-discovered from archive dir)",
    )
    p.add_argument(
        "--count", type=int, default=20,
        help="Number of diverse elites to refine (default: 20)",
    )
    p.add_argument(
        "--top-frac", type=float, default=1.0,
        help=(
            "Fraction of archive to sample from (default: 1.0 = whole archive). "
            "Keep at 1.0 to spread starting points across the whole archive."
        ),
    )
    p.add_argument(
        "--max-iter", type=int, default=200,
        help="Max L-BFGS-B iterations per design (default: 200)",
    )
    p.add_argument(
        "--gtol", type=float, default=1e-6,
        help="Gradient norm stopping threshold (default: 1e-6)",
    )
    p.add_argument(
        "--num-rays", type=int, default=256,
        help="Rays per field point during refinement (default: 256)",
    )
    p.add_argument(
        "--dist-threshold", type=float, default=0.003,
        help=(
            "Zernike L2 distance below which two converged designs are the same basin "
            "(default: 0.003 mm, from ef50_zernike.yaml dist_threshold)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for diverse-top selection (default: 0)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: refinement_<timestamp>/ next to archive_cells.json)",
    )
    p.add_argument(
        "--plot-trajectories", action="store_true", default=False,
        help="Record and overlay each design's path through Zernike measure space.",
    )
    p.add_argument(
        "--trajectory-every", type=int, default=1,
        help="Sample trajectory every N L-BFGS-B iterations when --plot-trajectories is set (default: 1)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    qd_cfg = _load_qd_config(args.config, archive_path)

    # YAML value wins over the argparse default;
    # explicit --dist-threshold on the CLI wins over the YAML.
    yaml_threshold = qd_cfg.get("dist_threshold", None)
    if yaml_threshold is not None and args.dist_threshold == 0.003:
        args.dist_threshold = float(yaml_threshold)

    yaml_refine_rays = qd_cfg.get("refine_num_rays", None)
    if yaml_refine_rays is not None and args.num_rays == 256:
        args.num_rays = int(yaml_refine_rays)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else archive_path.parent / f"refinement_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[refine-ef50] archive:    {archive_path}", flush=True)
    print(f"[refine-ef50] output dir: {out_dir}", flush=True)

    fl = qd_cfg.get("focal_length_target_mm")
    if fl is not None:
        print(f"[refine-ef50] EFL constraint: {fl} mm (10-D theta)", flush=True)
    else:
        print(f"[refine-ef50] EFL constraint: none (11-D theta)", flush=True)

    # ------------------------------------------------------------------
    # Load archive and select diverse starting points
    # ------------------------------------------------------------------
    all_designs = load_ef50_archive(archive_path)
    print(f"[refine-ef50] loaded {len(all_designs)} designs from archive", flush=True)

    if not all_designs:
        print("[refine-ef50] archive is empty — nothing to refine", flush=True)
        return

    selected = select_diverse_top(
        all_designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    print(
        f"[refine-ef50] selected {len(selected)} diverse elites "
        f"(top_frac={args.top_frac})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build problem
    # ------------------------------------------------------------------
    print(
        f"[refine-ef50] initialising problem "
        f"(num_rays={args.num_rays}, stage2, j_max=11) ...",
        flush=True,
    )
    problem = _build_problem(qd_cfg, num_rays=int(args.num_rays))

    desc_indices: tuple[int, int] = tuple(  # type: ignore[assignment]
        int(x) for x in qd_cfg.get("descriptor_zernike_indices", [7, 11])
    )

    # ------------------------------------------------------------------
    # Refine each selected design
    # ------------------------------------------------------------------
    results = []
    for i, design in enumerate(selected):
        print(
            f"[refine-ef50] {i+1:2d}/{len(selected)}  "
            f"archive_fitness={design.fitness:.5f}  ",
            end="",
            flush=True,
        )
        res = refine_design(
            problem,
            design.theta,
            max_iter=int(args.max_iter),
            gtol=float(args.gtol),
            record_trajectory=bool(args.plot_trajectories),
            trajectory_every=int(args.trajectory_every),
            descriptor_indices=desc_indices,
        )
        conv_flag = "OK" if res["converged"] else "stopped"
        print(
            f"rms {res['start_rms_mm']:.4f} → {res['converged_rms_mm']:.4f} mm  "
            f"{res['n_iters']} iters  [{conv_flag}]",
            flush=True,
        )
        res["archive_measures"] = list(design.measures)
        res["archive_fitness"] = float(design.fitness)
        results.append(res)

    # ------------------------------------------------------------------
    # Pairwise distances and basin labeling
    # ------------------------------------------------------------------
    # Skip Z1 (piston) — cluster on Z2..Z11
    zernike_matrix = np.array([r["zernike_vector"] for r in results], dtype=np.float64)
    z_for_clustering = zernike_matrix[:, 1:]
    dist_matrix = _pairwise_distances(z_for_clustering)
    basin_labels = _label_basins(dist_matrix, threshold=float(args.dist_threshold))
    n_basins = len(set(basin_labels))

    print(f"\n[refine-ef50] === Basin summary (threshold={args.dist_threshold} mm) ===", flush=True)
    print(f"[refine-ef50] {len(selected)} elites → {n_basins} distinct local minima\n", flush=True)
    for i, (res, basin) in enumerate(zip(results, basin_labels)):
        conv_flag = "converged" if res["converged"] else "stopped "
        print(
            f"  #{i+1:2d}  basin={basin}  "
            f"rms={res['converged_rms_mm']:.4f} mm  "
            f"iters={res['n_iters']:3d}  [{conv_flag}]",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    summary = {
        "archive": str(archive_path),
        "count": len(selected),
        "top_frac": args.top_frac,
        "max_iter": args.max_iter,
        "gtol": args.gtol,
        "num_rays": args.num_rays,
        "dist_threshold": args.dist_threshold,
        "n_basins": n_basins,
        "basin_labels": basin_labels,
        "dist_matrix": dist_matrix.tolist(),
        "results": results,
    }
    json_path = out_dir / "refinement_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[refine-ef50] results saved: {json_path}", flush=True)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        heatmap_path = out_dir / "refinement_heatmap.png"
        _plot_heatmap(dist_matrix, basin_labels, heatmap_path)
        print(f"[refine-ef50] heatmap saved:   {heatmap_path}", flush=True)
    except Exception as exc:
        print(f"[refine-ef50] heatmap failed: {exc}", flush=True)

    try:
        archive_bounds = qd_cfg.get("archive_bounds")
        if archive_bounds is not None:
            archive_bounds = (tuple(archive_bounds[0]), tuple(archive_bounds[1]))

        basins_path = out_dir / "refinement_basins.png"
        _plot_basins(
            all_designs, selected, results, basin_labels, archive_bounds, desc_indices,
            basins_path, show_trajectories=bool(args.plot_trajectories),
        )
        print(f"[refine-ef50] basin map saved: {basins_path}", flush=True)
    except Exception as exc:
        print(f"[refine-ef50] basin map failed: {exc}", flush=True)

    try:
        theta_pca_path = out_dir / "refinement_theta_pca.png"
        _plot_theta_pca(all_designs, selected, results, basin_labels, theta_pca_path)
        print(f"[refine-ef50] theta PCA saved: {theta_pca_path}", flush=True)
    except Exception as exc:
        print(f"[refine-ef50] theta PCA failed: {exc}", flush=True)

    print(f"[refine-ef50] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
