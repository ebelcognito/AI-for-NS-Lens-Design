"""Render ray-diagram panels for every design in a refinement_results.json.

Each panel shows the converged lens (post L-BFGS-B) with three ray bundles
(on-axis, 50 % FOV, 90 % FOV) and reports the converged RMS spot size, the
pre-refinement RMS, and the basin label.

Supports both triplet (Cooke) and EF50 lens types.  The lens type is
auto-detected from the archive path stored in ``refinement_results.json``.

Usage
-----
# Auto-discover latest refinement (triplet or EF50, whichever is newest):
  python3 optical_rl/qd/render_refined_designs.py

# Explicit path:
  python3 optical_rl/qd/render_refined_designs.py \
      --results results/models/triplet_cooke_zernike_20260401_125509/refinement_20260401_141324/refinement_results.json

  python3 optical_rl/qd/render_refined_designs.py \
      --results results/models/ef50_zernike_20260409_123918/refinement_20260409_132502/refinement_results.json

# Per-design PNGs:
  python3 optical_rl/qd/render_refined_designs.py --individual-pngs

# Extra options:
  python3 optical_rl/qd/render_refined_designs.py \
      --results path/to/refinement_results.json \
      --fov-fracs 0.0,0.5,0.9 \
      --num-rays 64 \
      --grid-cols 5 \
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    ArchiveDesign,
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
    _parse_fov_fracs,
    resolve_theme,
)
from optical_rl.qd.render_triplet_cooke_zernike import (  # noqa: E402
    _apply_theta_to_lens_triplet,
    _find_latest_archive as _find_latest_triplet_archive,
    _load_qd_config as _load_triplet_qd_config,
)
from optical_rl.qd.render_ef50_zernike import (  # noqa: E402
    _apply_theta_to_lens_ef50,
    _find_latest_archive as _find_latest_ef50_archive,
    _load_qd_config as _load_ef50_qd_config,
)


# ---------------------------------------------------------------------------
# Lens-type detection
# ---------------------------------------------------------------------------

_LENS_TRIPLET = "triplet"
_LENS_EF50 = "ef50"


def _detect_lens_type(archive_path: str) -> str:
    """Infer lens type from the archive path stored in refinement_results.json."""
    p = archive_path.lower()
    if "ef50" in p:
        return _LENS_EF50
    return _LENS_TRIPLET


# ---------------------------------------------------------------------------
# Load refinement results
# ---------------------------------------------------------------------------

def _find_latest_refinement(results_root: Path | None = None) -> Path:
    """Find the newest refinement_results.json across triplet and EF50 runs."""
    root = results_root or PROJECT_ROOT / "results" / "models"
    candidates: list[Path] = []
    for pattern in (
        "triplet_cooke_zernike_*/refinement_*/refinement_results.json",
        "ef50_zernike_*/refinement_*/refinement_results.json",
    ):
        candidates.extend(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No refinement_results.json found under {root}/"
            f"{{triplet_cooke_zernike,ef50_zernike}}_*/refinement_*/"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_refinement_results(path: Path) -> tuple[dict, list[dict]]:
    """Return (summary_meta, list_of_result_dicts)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    return data, results


# ---------------------------------------------------------------------------
# Convert a refinement result entry into a fake ArchiveDesign so we can
# reuse the existing render_design_triplet machinery directly.
# ---------------------------------------------------------------------------

def _result_to_design(res: dict, basin_label: int, rank: int) -> ArchiveDesign:
    theta = np.asarray(res["converged_theta"], dtype=np.float64)
    rms = float(res.get("converged_rms_mm", float("nan")))
    return ArchiveDesign(
        cell=(rank, basin_label),
        fitness=-rms,
        measures=(float(res["archive_measures"][0]), float(res["archive_measures"][1])),
        theta=theta,
        rms_spot=rms,
    )


# ---------------------------------------------------------------------------
# Render a single refined design panel
# ---------------------------------------------------------------------------

def _render_refined_panel(
    *,
    lens_template: Path,
    res: dict,
    basin_label: int,
    rank: int,
    theme: PlotTheme,
    ax,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: list[float],
    device: str,
    focal_length_target_mm: float | None,
    lens_type: str = _LENS_TRIPLET,
    use_start_theta: bool = False,
) -> float:
    """Draw one panel; returns the computed RMS spot (mm).

    When ``use_start_theta`` is True, renders the pre-refinement design from
    ``res['start_theta']`` and reports ``start_rms_mm`` in the title — same
    indexing as the post-refinement grid for 1:1 debug comparison.
    """
    from deeplens.optics import GeoLens  # type: ignore

    theta_key = "start_theta" if use_start_theta else "converged_theta"
    theta = np.asarray(res[theta_key], dtype=np.float64)
    lens = GeoLens(filename=str(lens_template), device=device)
    if lens_type == _LENS_EF50:
        _apply_theta_to_lens_ef50(
            lens, theta,
            wvln_um=wavelength_um,
            focal_length_target_mm=focal_length_target_mm,
        )
    else:
        _apply_theta_to_lens_triplet(
            lens, theta,
            wvln_um=wavelength_um,
            focal_length_target_mm=focal_length_target_mm,
        )

    ax.set_facecolor(theme.background)
    if theme.lens_fill is not None:
        for z_poly, x_poly in _lens_fill_polygons(lens):
            ax.fill(z_poly, x_poly, color=theme.lens_fill, alpha=1.0,
                    linewidth=0.0, zorder=1)

    lens.draw_lens_2d(ax=ax, fig=ax.figure, color=theme.lens_outline)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_values = [float(f) * rfov_deg for f in fov_fracs] or [0.0]

    for i, fov in enumerate(fov_values):
        ray = lens.sample_point_source_2D(
            fov=fov,
            depth=float(depth_mm),
            num_rays=int(num_rays),
            wvln=float(wavelength_um),
        )
        ray.prop_to(-1.0)
        _ray_out, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        lens.draw_ray_2d(
            ray_o_record, ax=ax, fig=ax.figure,
            color=theme.ray_colors[i % len(theme.ray_colors)],
        )

    foclen = float(getattr(lens, "foclen", float("nan")))

    if use_start_theta:
        spot_mm = float(res.get("start_rms_mm", float("nan")))
        ax.set_title(
            f"#{rank + 1}  basin {basin_label}   [BEFORE]\n"
            f"start rms {spot_mm:.4f} mm\n"
            f"f={foclen:.1f} mm",
            fontsize=6.5,
        )
    else:
        # Use the stored converged RMS; fall back to live compute if missing.
        spot_mm = res.get("converged_rms_mm")
        if spot_mm is None or not np.isfinite(spot_mm):
            spot_mm = _compute_rms_spot(
                lens=lens,
                depth_mm=float(depth_mm),
                wavelength_um=float(wavelength_um),
                num_rays=int(num_rays),
            )

        start_rms = res.get("start_rms_mm", float("nan"))
        converged = res.get("converged", False)
        n_iters = res.get("n_iters", "?")
        conv_str = "conv" if converged else "stop"

        ax.set_title(
            f"#{rank + 1}  basin {basin_label}\n"
            f"start {start_rms:.4f} → {spot_mm:.4f} mm  ({n_iters}it, {conv_str})\n"
            f"f={foclen:.1f} mm",
            fontsize=6.5,
        )
    ax.set_axis_off()

    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return float(spot_mm)


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def render_refinement_grid(
    *,
    results: list[dict],
    basin_labels: list[int],
    lens_template: Path,
    theme: PlotTheme,
    out_path: Path,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: list[float],
    cols: int,
    dpi: int,
    device: str,
    focal_length_target_mm: float | None,
    title: str | None = None,
    lens_type: str = _LENS_TRIPLET,
    use_start_theta: bool = False,
) -> list[dict]:
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        print("[render-refined] no results to render", flush=True)
        return []

    cols = max(1, int(cols))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4.2 * rows), dpi=int(dpi))
    fig.patch.set_facecolor(theme.background)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    panel_meta = []
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_facecolor(theme.background)
        ax.axis("off")
        if idx >= n:
            continue

        res = results[idx]
        basin = basin_labels[idx]
        if use_start_theta:
            print(
                f"[render-refined] {idx + 1}/{n}  basin={basin}  "
                f"[BEFORE]  start_rms={res.get('start_rms_mm', float('nan')):.5f} mm",
                flush=True,
            )
        else:
            print(
                f"[render-refined] {idx + 1}/{n}  basin={basin}  "
                f"converged_rms={res.get('converged_rms_mm', float('nan')):.5f} mm",
                flush=True,
            )

        spot_mm = _render_refined_panel(
            lens_template=lens_template,
            res=res,
            basin_label=basin,
            rank=idx,
            theme=theme,
            ax=ax,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
            fov_fracs=fov_fracs,
            device=device,
            focal_length_target_mm=focal_length_target_mm,
            lens_type=lens_type,
            use_start_theta=use_start_theta,
        )
        panel_meta.append({
            "rank": idx,
            "basin": basin,
            "start_rms_mm": res.get("start_rms_mm"),
            "converged_rms_mm": float(spot_mm),
            "n_iters": res.get("n_iters"),
            "converged": res.get("converged"),
            "theta": res.get("converged_theta"),
        })

    type_label = "EF50" if lens_type == _LENS_EF50 else "triplet"
    if title is None:
        if use_start_theta:
            auto_title = (
                f"Selected {type_label} designs — BEFORE refinement — "
                f"{n} elites  ({len(set(basin_labels))} basins)"
            )
        else:
            auto_title = (
                f"Refined {type_label} designs — "
                f"{n} elites  ({len(set(basin_labels))} basins)"
            )
    else:
        auto_title = title
    fig.suptitle(
        auto_title,
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor=theme.background, bbox_inches="tight")
    plt.close(fig)
    return panel_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render ray-diagram panels for converged refined designs (triplet or EF50)."
    )
    p.add_argument(
        "--results", default=None,
        help="Path to refinement_results.json (default: auto-discover latest across triplet & EF50)",
    )
    p.add_argument(
        "--lens-template", default=None,
        help="Path to lens template JSON (default: auto from config / archive path)",
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: next to results JSON)")
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9", help="Comma-separated FOV fractions")
    p.add_argument("--num-rays", type=int, default=32, help="Rays per bundle (default: 32)")
    p.add_argument("--grid-cols", type=int, default=5, help="Grid columns (default: 5)")
    p.add_argument("--depth-mm", type=float, default=None,
                   help="Object depth (default: from config, or -1000 triplet / -20000 EF50)")
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", default="cuda", help="Render device: cpu / cuda")
    p.add_argument("--individual-pngs", action="store_true",
                   help="Save each design as a separate PNG in addition to the grid")
    p.add_argument(
        "--render-start", action="store_true", default=False,
        help="Debug: also render a grid of the selected designs BEFORE refinement, "
             "using start_theta from each result. Same indexing as the main grid.",
    )
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F0EDE4")
    p.add_argument("--lens-fill-color", default="#7BAFD4")
    p.add_argument("--lens-outline-color", default="#1A4E8C")
    p.add_argument("--ray-color", default="#E07B39")
    return p.parse_args(argv)


def main(argv=None):
    import matplotlib.pyplot as plt

    args = _parse_args(argv)

    # --- locate results JSON ------------------------------------------------
    results_path = (
        Path(args.results) if args.results else _find_latest_refinement()
    )
    print(f"[render-refined] results: {results_path}", flush=True)

    data, results = load_refinement_results(results_path)
    basin_labels: list[int] = data.get("basin_labels", list(range(len(results))))

    if not results:
        print("[render-refined] no results found in JSON — nothing to render", flush=True)
        return

    # --- detect lens type ---------------------------------------------------
    archive_str = data.get("archive", "")
    lens_type = _detect_lens_type(archive_str)
    type_label = "EF50" if lens_type == _LENS_EF50 else "triplet"
    print(f"[render-refined] detected lens type: {type_label}", flush=True)

    # --- figure out lens template + focal length from the linked archive ----
    archive_path = Path(archive_str)
    qd_cfg: dict = {}
    if archive_path.exists():
        try:
            loader = _load_ef50_qd_config if lens_type == _LENS_EF50 else _load_triplet_qd_config
            qd_cfg = loader(None, archive_path)
        except Exception:
            pass

    _default_lens = {
        _LENS_EF50: str(PROJECT_ROOT / "DeepLens/datasets/lenses/camera/ef50mm_f1.8.json"),
        _LENS_TRIPLET: str(PROJECT_ROOT / "DeepLens/datasets/lenses/cooke_smaller_ap_1mm.json"),
    }
    _default_depth = {_LENS_EF50: -20000.0, _LENS_TRIPLET: -1000.0}

    lens_template = Path(
        args.lens_template
        if args.lens_template
        else qd_cfg.get("lens_path", _default_lens[lens_type])
    )
    focal_length_target_mm: float | None = (
        float(qd_cfg["focal_length_target_mm"])
        if qd_cfg.get("focal_length_target_mm") is not None
        else None
    )
    depth_mm = float(
        args.depth_mm
        if args.depth_mm is not None
        else qd_cfg.get("object_depth_mm", _default_depth[lens_type])
    )
    wavelength_um = float(qd_cfg.get("wavelength_um", args.wavelength_um))

    print(f"[render-refined] lens template:  {lens_template}", flush=True)
    if focal_length_target_mm is not None:
        print(f"[render-refined] fixed EFL: {focal_length_target_mm} mm", flush=True)
    print(f"[render-refined] depth: {depth_mm} mm", flush=True)
    print(f"[render-refined] {len(results)} designs, {len(set(basin_labels))} basins", flush=True)

    # --- output dir ---------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else results_path.parent / f"renders_refined_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- theme + FOV --------------------------------------------------------
    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    # --- render grid --------------------------------------------------------
    grid_path = out_dir / "refined_designs_grid.png"
    panel_meta = render_refinement_grid(
        results=results,
        basin_labels=basin_labels,
        lens_template=lens_template,
        theme=theme,
        out_path=grid_path,
        depth_mm=float(depth_mm),
        wavelength_um=float(wavelength_um),
        num_rays=int(args.num_rays),
        fov_fracs=fov_fracs,
        cols=int(args.grid_cols),
        dpi=int(args.dpi),
        device=str(args.device),
        focal_length_target_mm=focal_length_target_mm,
        lens_type=lens_type,
    )
    print(f"[render-refined] grid saved → {grid_path}", flush=True)

    # --- debug: BEFORE-refinement grid (same indexing as main grid) ---------
    if args.render_start:
        before_path = out_dir / "selected_designs_BEFORE_grid.png"
        print(
            f"[render-refined] rendering BEFORE-refinement grid "
            f"({len(results)} designs) → {before_path.name}",
            flush=True,
        )
        render_refinement_grid(
            results=results,
            basin_labels=basin_labels,
            lens_template=lens_template,
            theme=theme,
            out_path=before_path,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(args.num_rays),
            fov_fracs=fov_fracs,
            cols=int(args.grid_cols),
            dpi=int(args.dpi),
            device=str(args.device),
            focal_length_target_mm=focal_length_target_mm,
            lens_type=lens_type,
            use_start_theta=True,
        )
        print(f"[render-refined] BEFORE grid saved → {before_path}", flush=True)

    # --- per-basin grids -----------------------------------------------------
    from collections import defaultdict
    basin_groups: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    for res, basin in zip(results, basin_labels):
        basin_groups[basin].append((res, basin))

    if len(basin_groups) > 1:
        for basin_id in sorted(basin_groups.keys()):
            group_results = [r for r, _ in basin_groups[basin_id]]
            group_labels = [basin_id] * len(group_results)
            basin_path = out_dir / f"basin_{basin_id}_grid.png"
            print(
                f"[render-refined] rendering basin {basin_id} "
                f"({len(group_results)} designs) → {basin_path.name}",
                flush=True,
            )
            render_refinement_grid(
                results=group_results,
                basin_labels=group_labels,
                lens_template=lens_template,
                theme=theme,
                out_path=basin_path,
                depth_mm=float(depth_mm),
                wavelength_um=float(wavelength_um),
                num_rays=int(args.num_rays),
                fov_fracs=fov_fracs,
                cols=int(args.grid_cols),
                dpi=int(args.dpi),
                device=str(args.device),
                focal_length_target_mm=focal_length_target_mm,
                title=f"Basin {basin_id} — {len(group_results)} designs",
                lens_type=lens_type,
            )
            print(f"[render-refined] basin {basin_id} saved → {basin_path}", flush=True)

    # --- individual PNGs (one per design) ------------------------------------
    if args.individual_pngs:
        indiv_dir = out_dir / "individual"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        n = len(results)
        for idx, (res, basin) in enumerate(zip(results, basin_labels)):
            fig_i, ax_i = plt.subplots(1, 1, figsize=(5.3, 4.2), dpi=int(args.dpi))
            fig_i.patch.set_facecolor(theme.background)
            print(
                f"[render-refined] individual {idx + 1}/{n}  basin={basin}  "
                f"converged_rms={res.get('converged_rms_mm', float('nan')):.5f} mm",
                flush=True,
            )
            _render_refined_panel(
                lens_template=lens_template,
                res=res,
                basin_label=basin,
                rank=idx,
                theme=theme,
                ax=ax_i,
                depth_mm=float(depth_mm),
                wavelength_um=float(wavelength_um),
                num_rays=int(args.num_rays),
                fov_fracs=fov_fracs,
                device=str(args.device),
                focal_length_target_mm=focal_length_target_mm,
                lens_type=lens_type,
            )
            fig_i.tight_layout()
            png_name = f"rank_{idx + 1:02d}_basin_{basin}.png"
            fig_i.savefig(
                indiv_dir / png_name, dpi=int(args.dpi),
                facecolor=theme.background, bbox_inches="tight",
            )
            plt.close(fig_i)
        print(f"[render-refined] {n} individual PNGs saved → {indiv_dir}", flush=True)

    # --- save panel summary -------------------------------------------------
    summary_path = out_dir / "render_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "refinement_results": str(results_path),
                "lens_type": lens_type,
                "lens_template": str(lens_template),
                "depth_mm": float(depth_mm),
                "fov_fracs": [float(f) for f in fov_fracs],
                "num_rays": int(args.num_rays),
                "device": str(args.device),
                "grid_image": grid_path.name,
                "individual_pngs": bool(args.individual_pngs),
                "panels": panel_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[render-refined] summary  → {summary_path}", flush=True)
    print(f"[render-refined] done.", flush=True)


if __name__ == "__main__":
    main()
