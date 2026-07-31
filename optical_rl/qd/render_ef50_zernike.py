"""Render diverse high-quality designs from an EF50mm f/1.8 Zernike CMA-MEGA archive.

Mirrors render_triplet_cooke_zernike.py in structure.
theta = [c0, c1, c2, c3, c4, c5, c7, c8, c9, c10]  (10 free curvatures)
c11 is derived from the paraxial ABCD EFL constraint.
d_sensor follows from the paraxial BFL solve.

Usage:
  python3 optical_rl/qd/render_ef50_zernike.py
  python3 optical_rl/qd/render_ef50_zernike.py --count 12 --grid-cols 4 --fov-fracs 0.0,0.5,0.9
  # Omit --archive to use the newest ef50_zernike_* run, or pass one/more paths / globs (newest wins).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optical_rl.utils.config import load_config  # noqa: E402

# Reuse selection, theming, rendering helpers from singlet renderer.
from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    ArchiveDesign,
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
    _parse_fov_fracs,
    select_diverse_top,
    resolve_theme,
)

# Refractive surface indices for EF50 (stop is surface 6, skipped).
_SURF_IDX = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)


# ---------------------------------------------------------------------------
# EF50 archive loader (archive_cells.json has a different schema to archive.json)
# ---------------------------------------------------------------------------

def load_ef50_archive(path: Path) -> list[ArchiveDesign]:
    """Load designs from an ef50_zernike archive_cells.json file.

    The file structure is:
      { "archive_bounds": ..., "archive_bins": ..., "cells": [ {...}, ... ], "stats": ... }
    Each cell has "cell_index", "theta", "fitness", "measures", "metrics".
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Support both the wrapped ef50 format and a bare list (for future compat)
    if isinstance(raw, list):
        rows = raw
        cell_key = "cell"
    else:
        rows = raw.get("cells", [])
        cell_key = "cell_index"

    out: list[ArchiveDesign] = []
    for row in rows:
        try:
            cell_raw = row.get(cell_key) or row.get("cell") or row.get("cell_index")
            cell = tuple(int(x) for x in cell_raw)
            fitness = float(row["fitness"])
            measures = tuple(float(x) for x in row["measures"])
            theta = np.asarray(row["theta"], dtype=np.float64).reshape(-1)
            metrics = row.get("metrics", {})
            rms_spot = metrics.get("rms_spot", None)
            if rms_spot is not None:
                rms_spot = float(rms_spot)
        except Exception:
            continue
        if not np.isfinite(fitness):
            continue
        if not np.isfinite(measures[0]) or not np.isfinite(measures[1]):
            continue
        if not np.all(np.isfinite(theta)):
            continue
        if rms_spot is not None and not np.isfinite(rms_spot):
            rms_spot = None
        out.append(
            ArchiveDesign(
                cell=(cell[0], cell[1]),
                fitness=fitness,
                measures=(measures[0], measures[1]),
                theta=theta,
                rms_spot=rms_spot,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Paraxial ABCD BFL solve (pure Python / numpy — no PyTorch needed here)
# ---------------------------------------------------------------------------

def _paraxial_bfl_ef50(
    c0, c1, c2, c3, c4, c5, c7, c8, c9, c10, c11,
    *,
    n1: float, n2: float, n3: float,
    n4: float, n5: float, n6: float,
    d01: float, d12: float, d23: float, d34: float, d45: float,
    d5_to_7: float, d78: float, d89: float, d9_10: float, d10_11: float,
) -> float:
    """Chain paraxial ABCD matrices for EF50 → BFL (mm).

    Marginal ray: [y=1, nu=0] in object space.
    Returns back focal length (distance from surface 11 to focus).
    Key: surface 8 is a cemented glass-to-glass interface (n4 → n5).
    """
    y, nu = 1.0, 0.0
    nu -= (n1 - 1.0) * c0 * y;      y += (d01 / n1) * nu    # surf 0, gap 01
    nu -= (1.0 - n1) * c1 * y;      y += d12 * nu            # surf 1, gap 12
    nu -= (n2 - 1.0) * c2 * y;      y += (d23 / n2) * nu    # surf 2, gap 23
    nu -= (1.0 - n2) * c3 * y;      y += d34 * nu            # surf 3, gap 34
    nu -= (n3 - 1.0) * c4 * y;      y += (d45 / n3) * nu    # surf 4, gap 45
    nu -= (1.0 - n3) * c5 * y;      y += d5_to_7 * nu        # surf 5, gap → 7 (absorbs stop)
    nu -= (n4 - 1.0) * c7 * y;      y += (d78 / n4) * nu    # surf 7, gap 78
    nu -= (n5 - n4)  * c8 * y;      y += (d89 / n5) * nu    # surf 8 glass→glass
    nu -= (1.0 - n5) * c9 * y;      y += d9_10 * nu          # surf 9, gap 9→10
    nu -= (n6 - 1.0) * c10 * y;     y += (d10_11 / n6) * nu # surf 10, gap 10→11
    nu -= (1.0 - n6) * c11 * y                                # surf 11
    safe_nu = nu if abs(nu) > 1e-9 else 1e-9
    return -y / safe_nu


def _derive_c11_from_efl(
    c0, c1, c2, c3, c4, c5, c7, c8, c9, c10,
    *,
    n1: float, n2: float, n3: float,
    n4: float, n5: float, n6: float,
    d01: float, d12: float, d23: float, d34: float, d45: float,
    d5_to_7: float, d78: float, d89: float, d9_10: float, d10_11: float,
    f_target: float,
) -> tuple[float, float]:
    """Numpy mirror of EF50ZernikeProblem._paraxial_fixed_efl.

    Traces the marginal ray through surfaces 0–10, then solves for c11 such
    that EFL = f_target.  Returns (c11, bfl).
    """
    y, nu = 1.0, 0.0
    nu -= (n1 - 1.0) * c0 * y;      y += (d01 / n1) * nu
    nu -= (1.0 - n1) * c1 * y;      y += d12 * nu
    nu -= (n2 - 1.0) * c2 * y;      y += (d23 / n2) * nu
    nu -= (1.0 - n2) * c3 * y;      y += d34 * nu
    nu -= (n3 - 1.0) * c4 * y;      y += (d45 / n3) * nu
    nu -= (1.0 - n3) * c5 * y;      y += d5_to_7 * nu
    nu -= (n4 - 1.0) * c7 * y;      y += (d78 / n4) * nu
    nu -= (n5 - n4)  * c8 * y;      y += (d89 / n5) * nu
    nu -= (1.0 - n5) * c9 * y;      y += d9_10 * nu
    nu -= (n6 - 1.0) * c10 * y;     y += (d10_11 / n6) * nu
    # Now at exit of element-5 glass; next is surface 11 (n6 → air).
    P = 1.0 / f_target
    safe_y = y if abs(y) > 1e-9 else 1e-9
    phi_11 = (nu + P) / safe_y
    c11 = phi_11 / (1.0 - n6)
    bfl = y * f_target
    return c11, bfl


# ---------------------------------------------------------------------------
# Apply theta to a loaded GeoLens
# ---------------------------------------------------------------------------

def _apply_theta_to_lens_ef50(
    lens,
    theta: np.ndarray,
    wvln_um: float = 0.587,
    focal_length_target_mm: float | None = None,
) -> None:
    """Set curvatures on *lens* and derive d_sensor from the paraxial BFL solve.

    Handles both modes:
      10-D theta [c0..c5, c7..c10]:        c11 derived to enforce EFL = focal_length_target_mm.
      11-D theta [c0..c5, c7..c10, c11]:   all curvatures free; d_sensor from BFL.
    """
    import torch

    arr = np.asarray(theta, dtype=np.float64).reshape(-1)

    # Read 6 glass refractive indices from mat2 of the relevant surfaces
    n1 = float(lens.surfaces[0].mat2.refractive_index(wvln_um))
    n2 = float(lens.surfaces[2].mat2.refractive_index(wvln_um))
    n3 = float(lens.surfaces[4].mat2.refractive_index(wvln_um))
    n4 = float(lens.surfaces[7].mat2.refractive_index(wvln_um))
    n5 = float(lens.surfaces[8].mat2.refractive_index(wvln_um))
    n6 = float(lens.surfaces[10].mat2.refractive_index(wvln_um))

    # Axial gaps from surface d positions
    d = [float(lens.surfaces[i].d.item()) for i in range(12)]
    d01     = d[1]  - d[0]
    d12     = d[2]  - d[1]
    d23     = d[3]  - d[2]
    d34     = d[4]  - d[3]
    d45     = d[5]  - d[4]
    d5_to_7 = d[7]  - d[5]   # absorbs stop at surface 6
    d78     = d[8]  - d[7]
    d89     = d[9]  - d[8]
    d9_10   = d[10] - d[9]
    d10_11  = d[11] - d[10]
    z_s11   = d[11]

    gap_kw = dict(
        n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6,
        d01=d01, d12=d12, d23=d23, d34=d34, d45=d45,
        d5_to_7=d5_to_7, d78=d78, d89=d89, d9_10=d9_10, d10_11=d10_11,
    )

    c0  = float(arr[0])
    c1  = float(arr[1])
    c2  = float(arr[2])
    c3  = float(arr[3])
    c4  = float(arr[4])
    c5  = float(arr[5])
    c7  = float(arr[6])
    c8  = float(arr[7])
    c9  = float(arr[8])
    c10 = float(arr[9])

    if len(arr) == 10:
        # Fixed-EFL mode: derive c11 from the paraxial solve.
        if focal_length_target_mm is None:
            raise ValueError(
                "theta is 10-D (fixed-EFL archive) but focal_length_target_mm was not supplied. "
                "Pass --config or --focal-length when rendering a fixed-EFL archive."
            )
        c11, bfl = _derive_c11_from_efl(
            c0, c1, c2, c3, c4, c5, c7, c8, c9, c10,
            f_target=float(focal_length_target_mm),
            **gap_kw,
        )
        d_sensor = z_s11 + max(bfl, 1.0)
    else:
        # Free mode: c11 is the 11th element of theta.
        c11 = float(arr[10])
        bfl = _paraxial_bfl_ef50(
            c0, c1, c2, c3, c4, c5, c7, c8, c9, c10, c11,
            **gap_kw,
        )
        d_sensor = z_s11 + max(bfl, 1.0)

    curvatures = [c0, c1, c2, c3, c4, c5, c7, c8, c9, c10, c11]
    dtype = lens.d_sensor.dtype
    dev = getattr(lens, "device", "cpu")
    for k, idx in enumerate(_SURF_IDX):
        lens.surfaces[idx].c = torch.tensor(curvatures[k], dtype=dtype, device=dev)
    lens.d_sensor = torch.tensor(d_sensor, dtype=dtype, device=dev)
    try:
        lens.post_computation()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single-design renderer
# ---------------------------------------------------------------------------

def render_design_ef50(
    *,
    lens_template: Path,
    design: ArchiveDesign,
    theme: PlotTheme,
    ax,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs,
    rank: int,
    device: str,
    focal_length_target_mm: float | None = None,
    title: str | None = None,
) -> dict[str, float]:
    from deeplens.optics import GeoLens  # type: ignore

    lens = GeoLens(filename=str(lens_template), device=device)
    _apply_theta_to_lens_ef50(
        lens, design.theta, wvln_um=wavelength_um,
        focal_length_target_mm=focal_length_target_mm,
    )

    ax.set_facecolor(theme.background)
    if theme.lens_fill is not None:
        for z_poly, x_poly in _lens_fill_polygons(lens):
            ax.fill(z_poly, x_poly, color=theme.lens_fill, alpha=1.0, linewidth=0.0, zorder=1)

    lens.draw_lens_2d(ax=ax, fig=ax.figure, color=theme.lens_outline)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_values = [float(f) * rfov_deg for f in fov_fracs] or [0.0]

    for i, fov in enumerate(fov_values):
        ray = lens.sample_point_source_2D(
            fov=fov, depth=float(depth_mm),
            num_rays=int(num_rays), wvln=float(wavelength_um),
        )
        ray.prop_to(-1.0)
        _ray_out, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        lens.draw_ray_2d(ray_o_record, ax=ax, fig=ax.figure,
                         color=theme.ray_colors[i % len(theme.ray_colors)])

    spot = design.rms_spot
    if spot is None:
        spot = _compute_rms_spot(
            lens=lens, depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um), num_rays=int(num_rays),
        )

    foclen = float(getattr(lens, "foclen", float("nan")))

    if title is not None:
        ax.set_title(title, fontsize=8)
    else:
        # c11 is derived, so only show the 10 free parameters
        surf_labels = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
        theta_arr = np.asarray(design.theta, dtype=np.float64).reshape(-1)
        theta_str = "  ".join(
            f"c{surf_labels[k]}={float(theta_arr[k]):.4f}"
            for k in range(min(len(theta_arr), len(surf_labels)))
        )
        efl_note = (
            f"  EFL={focal_length_target_mm:.2f}mm(fixed)"
            if focal_length_target_mm is not None and len(theta_arr) == 10
            else f"  f={foclen:.1f}mm"
        )
        ax.set_title(
            f"#{rank+1}  spot={spot:.4f}mm{efl_note}\n{theta_str}",
            fontsize=5,
        )
    ax.set_axis_off()

    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"spot_size_mm": float(spot), "focal_length_mm": float(foclen)}


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def render_archive_grid_ef50(
    *,
    selected,
    lens_template: Path,
    theme: PlotTheme,
    out_path: Path,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs,
    cols: int,
    dpi: int,
    device: str,
    focal_length_target_mm: float | None = None,
    panel_titles=None,
) -> list[dict[str, float]]:
    import math
    import matplotlib.pyplot as plt

    n = len(selected)
    if n == 0:
        return []
    cols = max(1, int(cols))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.9 * rows), dpi=int(dpi))
    fig.patch.set_facecolor(theme.background)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    panel_metrics: list[dict[str, float]] = []
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_facecolor(theme.background)
        ax.axis("off")
        if idx >= n:
            continue

        print(
            f"[render-ef50-zernike] rendering #{idx+1}/{n} "
            f"fitness={selected[idx].fitness:.5f}",
            flush=True,
        )
        ptitle = panel_titles[idx] if panel_titles and idx < len(panel_titles) else None
        metrics = render_design_ef50(
            lens_template=lens_template,
            design=selected[idx],
            theme=theme,
            ax=ax,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
            fov_fracs=fov_fracs,
            rank=idx,
            device=device,
            focal_length_target_mm=focal_length_target_mm,
            title=ptitle,
        )
        panel_metrics.append(metrics)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor=theme.background)
    plt.close(fig)
    return panel_metrics


# ---------------------------------------------------------------------------
# Archive auto-discovery
# ---------------------------------------------------------------------------

def save_archive_with_selected_plot(
    archive_path: Path,
    all_designs: list[ArchiveDesign],
    selected: list[ArchiveDesign],
    out_path: Path,
    *,
    coord_labels: tuple[str, str] = ("m0", "m1"),
    dpi: int = 150,
) -> None:
    """Save a two-panel archive figure with the rendered selections highlighted.

    Left panel: per-bin fitness heatmap (same as archive_map.png).  Right panel:
    scatter of every cell in BC space, with the ``selected`` designs overlaid
    as larger markers with red outlines and rank labels.  The rank is assigned
    by the order of ``selected`` (which the renderer sorts by rms_spot then
    fitness, so rank 0 is the "best" rendered design).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    raw = json.loads(archive_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        archive_bounds = None
        archive_bins = None
    else:
        archive_bounds = raw.get("archive_bounds")
        archive_bins = raw.get("archive_bins")

    # Fall back: derive from the designs if bounds aren't in the file.
    if archive_bounds is None and all_designs:
        m0s = np.array([d.measures[0] for d in all_designs])
        m1s = np.array([d.measures[1] for d in all_designs])
        archive_bounds = ((float(m0s.min()), float(m0s.max())),
                          (float(m1s.min()), float(m1s.max())))
    if archive_bins is None and all_designs:
        max_i = max(d.cell[0] for d in all_designs) + 1
        max_j = max(d.cell[1] for d in all_designs) + 1
        archive_bins = (max_i, max_j)

    bins_x, bins_y = int(archive_bins[0]), int(archive_bins[1])
    (x_lo, x_hi), (y_lo, y_hi) = archive_bounds

    # Build bin-grid of -fitness for the heatmap panel.
    neg_grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
    for d in all_designs:
        i, j = d.cell
        if 0 <= i < bins_x and 0 <= j < bins_y:
            neg_grid[i, j] = -float(d.fitness)

    # Ranks of selected designs: identify by cell tuple so we don't care about
    # object identity.
    sel_by_cell = {tuple(int(v) for v in d.cell): rank
                   for rank, d in enumerate(selected)}

    all_xs = np.array([d.measures[0] for d in all_designs])
    all_ys = np.array([d.measures[1] for d in all_designs])
    all_neg = np.array([-float(d.fitness) for d in all_designs])

    sel_xs = np.array([d.measures[0] for d in selected])
    sel_ys = np.array([d.measures[1] for d in selected])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#FFFFFF")

    valid = neg_grid[np.isfinite(neg_grid) & (neg_grid > 0)]
    norm = LogNorm(
        vmin=float(valid.min()), vmax=float(valid.max())
    ) if valid.size > 0 else None

    im = axes[0].imshow(
        neg_grid.T,
        origin="lower",
        aspect="auto",
        norm=norm,
        extent=[x_lo, x_hi, y_lo, y_hi],
        cmap="viridis",
    )
    axes[0].set_title("Archive fitness heatmap (selections marked)")
    axes[0].set_xlabel(coord_labels[0])
    axes[0].set_ylabel(coord_labels[1])
    fig.colorbar(im, ax=axes[0], shrink=0.85).set_label("-fitness (log)")

    # Overlay selected designs on the heatmap with rank labels.
    axes[0].scatter(
        sel_xs, sel_ys,
        s=120, facecolors="none", edgecolors="#E53935",
        linewidths=2.0, zorder=5,
    )
    for rank, d in enumerate(selected):
        axes[0].annotate(
            str(rank),
            xy=(d.measures[0], d.measures[1]),
            xytext=(4, 4), textcoords="offset points",
            color="#E53935", fontsize=9, fontweight="bold", zorder=6,
        )

    # Right panel: scatter of all cells (faded) + selected highlighted.
    pos_neg = all_neg[all_neg > 0]
    scatter_norm = (
        LogNorm(vmin=float(pos_neg.min()), vmax=float(pos_neg.max()))
        if pos_neg.size > 0 else None
    )
    sc = axes[1].scatter(
        all_xs, all_ys, c=all_neg, s=18, norm=scatter_norm, alpha=0.45,
        cmap="viridis",
    )
    fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("-fitness (log)")

    axes[1].scatter(
        sel_xs, sel_ys,
        s=180, facecolors="none", edgecolors="#E53935",
        linewidths=2.2, zorder=5,
        label=f"rendered ({len(selected)})",
    )
    for rank, d in enumerate(selected):
        axes[1].annotate(
            str(rank),
            xy=(d.measures[0], d.measures[1]),
            xytext=(5, 5), textcoords="offset points",
            color="#E53935", fontsize=10, fontweight="bold", zorder=6,
        )

    axes[1].set_xlim(x_lo, x_hi)
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_title("Archive scatter (selections marked)")
    axes[1].set_xlabel(coord_labels[0])
    axes[1].set_ylabel(coord_labels[1])
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor="#FFFFFF")
    plt.close(fig)


def _find_latest_archive(results_root: Path | None = None) -> Path:
    if results_root is not None:
        roots = [results_root]
    else:
        roots = [
            PROJECT_ROOT / "results" / "lensmega",
            PROJECT_ROOT / "results" / "models",
        ]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob("**/archive_cells.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No archive_cells.json found under {[str(r) for r in roots]}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_archive_path(parts: list[str] | None) -> Path:
    """Resolve --archive to a single file; if several paths or glob matches, pick newest mtime."""
    if not parts:
        return _find_latest_archive()
    candidates: list[Path] = []
    for part in parts:
        s = part.strip()
        if any(ch in s for ch in "*?[]"):
            for m in glob(s):
                p = Path(m).resolve()
                if p.is_file():
                    candidates.append(p)
        else:
            p = Path(s).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Archive not found: {s}")
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No archive file matched {parts!r}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    if len(candidates) > 1:
        print(
            f"[render-ef50-zernike] {len(candidates)} archive(s) matched; using latest: {chosen}",
            flush=True,
        )
    return chosen


def _load_qd_config(config_path: str | None, archive_path: Path) -> dict:
    """Load the qd section from the YAML config.

    Priority:
      1. Explicit --config argument.
      2. Any *.yaml file saved in the archive's run directory.
      3. Repo default ef50_zernike.yaml (training without --config does not copy YAML into the run dir).
    """
    paths_to_try: list[Path] = []
    if config_path:
        paths_to_try.append(Path(config_path))
    for p in sorted(archive_path.parent.glob("*.yaml")):
        paths_to_try.append(p)
    default_yaml = PROJECT_ROOT / "optical_rl" / "configs" / "ef50_zernike.yaml"
    if default_yaml.is_file() and default_yaml not in paths_to_try:
        paths_to_try.append(default_yaml)
    for p in paths_to_try:
        try:
            cfg = load_config(str(p)) or {}
            if isinstance(cfg, dict) and "qd" in cfg:
                return cfg["qd"]
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Render diverse EF50mm f/1.8 Zernike designs from CMA-MEGA archive."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config; used to supply defaults like qd.render_min_fitness.",
    )
    p.add_argument(
        "--archive",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "archive_cells.json path(s) or glob(s); if several match, the newest file is used. "
            "Omit entirely for latest under results/models/ef50_zernike_*/."
        ),
    )
    p.add_argument(
        "--lens-template",
        default=str(PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"),
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=-20000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-fitness", type=float, default=None,
                   help="Exclude designs with fitness below this threshold.")
    p.add_argument("--focal-length", type=float, default=None,
                   help="Override focal_length_target_mm (auto-read from YAML if not set).")
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F0EDE4")
    p.add_argument("--lens-fill-color", default="#7BAFD4")
    p.add_argument("--lens-outline-color", default="#1A4E8C")
    p.add_argument("--ray-color", default="#E07B39")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    archive_path = _resolve_archive_path(args.archive)

    qd_cfg = _load_qd_config(args.config, archive_path)

    focal_length_target_mm: float | None = args.focal_length
    if focal_length_target_mm is None and qd_cfg.get("focal_length_target_mm") is not None:
        focal_length_target_mm = float(qd_cfg["focal_length_target_mm"])

    if args.min_fitness is None and qd_cfg.get("render_min_fitness") is not None:
        args.min_fitness = float(qd_cfg["render_min_fitness"])

    lens_template = Path(args.lens_template)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else archive_path.parent / f"renders_ef50_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-ef50-zernike] archive:       {archive_path}", flush=True)
    print(f"[render-ef50-zernike] lens template:  {lens_template}", flush=True)
    print(f"[render-ef50-zernike] output dir:     {out_dir}", flush=True)
    if focal_length_target_mm is not None:
        print(
            f"[render-ef50-zernike] EFL target: {focal_length_target_mm} mm "
            f"(10-D theta: last curvature c11 derived paraxially)",
            flush=True,
        )
    else:
        print(
            "[render-ef50-zernike] EFL target: none loaded — required for 10-D archives "
            "(use --config / --focal-length, or keep ef50_zernike.yaml beside archive)",
            flush=True,
        )

    designs = load_ef50_archive(archive_path)
    print(f"[render-ef50-zernike] loaded {len(designs)} designs", flush=True)

    if args.min_fitness is not None:
        before = len(designs)
        designs = [d for d in designs if d.fitness >= float(args.min_fitness)]
        print(f"[render-ef50-zernike] min-fitness filter: kept {len(designs)}/{before}", flush=True)

    if not designs:
        print("[render-ef50-zernike] no designs to render", flush=True)
        return

    selected = select_diverse_top(
        designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot) if d.rms_spot is not None and np.isfinite(d.rms_spot)
            else float("inf"),
            -float(d.fitness),
        ),
    )
    print(f"[render-ef50-zernike] selected {len(selected)} diverse designs", flush=True)

    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid_ef50(
        selected=selected,
        lens_template=lens_template,
        theme=theme,
        out_path=grid_path,
        depth_mm=float(args.depth_mm),
        wavelength_um=float(args.wavelength_um),
        num_rays=int(args.num_rays),
        fov_fracs=fov_fracs,
        cols=int(args.grid_cols),
        dpi=int(args.dpi),
        device=str(args.device),
        focal_length_target_mm=focal_length_target_mm,
    )
    print(f"[render-ef50-zernike] grid saved: {grid_path}", flush=True)

    selected_meta = []
    for rank, (design, panel) in enumerate(zip(selected, panel_metrics)):
        selected_meta.append({
            "rank": rank,
            "cell": [int(design.cell[0]), int(design.cell[1])],
            "fitness": float(design.fitness),
            "spot_size_mm": float(panel["spot_size_mm"]),
            "focal_length_mm": float(panel["focal_length_mm"]),
            "measures": [float(design.measures[0]), float(design.measures[1])],
            "theta": [float(x) for x in design.theta.tolist()],
            "rms_spot_from_archive": (
                float(design.rms_spot)
                if design.rms_spot is not None and np.isfinite(design.rms_spot)
                else None
            ),
        })

    summary = {
        "archive": str(archive_path),
        "lens_template": str(lens_template),
        "count": len(selected),
        "top_frac": float(args.top_frac),
        "depth_mm": float(args.depth_mm),
        "fov_fracs": [float(x) for x in fov_fracs],
        "grid_image": grid_path.name,
        "selected": selected_meta,
    }
    (out_dir / "selected_designs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Archive map with rendered-selection overlay.
    sel_plot_path = out_dir / "archive_with_selected.png"
    try:
        j0 = int(qd_cfg.get("descriptor_zernike_indices", [7, 11])[0])
        j1 = int(qd_cfg.get("descriptor_zernike_indices", [7, 11])[1])
        coord_labels = (
            f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)",
            f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)",
        )
    except Exception:
        coord_labels = ("m0", "m1")
    save_archive_with_selected_plot(
        archive_path=archive_path,
        all_designs=designs,
        selected=selected,
        out_path=sel_plot_path,
        coord_labels=coord_labels,
        dpi=int(args.dpi),
    )
    print(f"[render-ef50-zernike] archive+selection plot: {sel_plot_path}", flush=True)

    print(f"[render-ef50-zernike] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
