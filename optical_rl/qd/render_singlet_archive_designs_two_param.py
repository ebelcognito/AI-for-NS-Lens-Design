"""Render diverse high-quality designs from a two-parameter singlet CMA-MEGA archive.

Identical to render_singlet_archive_designs.py except:
  - theta = [c1, c2] (both curvatures free)
  - d_sensor is derived via the paraxial back-focal solve, NOT the EFL-constrained solve
  - Auto-discovery searches for singlet_cmamega_two_param_* run directories
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse everything from the 1-param renderer — archive loading, selection,
# rendering pipeline, themes, CLI parsing, grid layout.
from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    ArchiveDesign,
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
    _parse_fov_fracs,
    load_archive,
    render_archive_grid,
    resolve_theme,
    select_diverse_top,
)


# ──────────────────────────────────────────────────────────────────────────────
# Two-param theta → lens (back-focal solve; no EFL constraint)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_theta_to_lens_two_param(lens, theta: np.ndarray, wvln_um: float = 0.587):
    """Apply theta = [c1, c2] to the lens; d_sensor placed at paraxial back focus.

    Unlike the 1-param renderer, the target EFL is not enforced.  The sensor
    is placed wherever the thick-lens paraxial back focal point is for this
    particular (c1, c2) pair.

    Thick-lens BFL:
        phi_1 = (n-1)*c1
        phi_2 = -(n-1)*c2   (rear surface: glass → air)
        P     = phi_1 + phi_2 - t*phi_1*phi_2/n
        BFL   = (1 - t*phi_1/n) / P
    """
    import torch

    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    _eps = 1e-3

    c1 = float(arr[0]) if float(arr[0]) != 0.0 else _eps
    c2 = float(arr[1]) if float(arr[1]) != 0.0 else _eps

    n = float(lens.surfaces[1].mat2.refractive_index(wvln_um))
    t_glass = float(lens.surfaces[2].d.item() - lens.surfaces[1].d.item())
    z_last = float(lens.surfaces[-1].d.item())

    phi_1 = (n - 1.0) * c1
    phi_2 = -(n - 1.0) * c2
    P = phi_1 + phi_2 - t_glass * phi_1 * phi_2 / n
    numerator = 1.0 - t_glass * phi_1 / n
    safe_P = max(P, 1e-4) if P > 0 else min(P, -1e-4)
    bfl = numerator / safe_P
    d_sensor = z_last + bfl

    dtype = lens.d_sensor.dtype
    lens.surfaces[1].c = torch.tensor(c1, dtype=dtype, device=lens.device)
    lens.surfaces[2].c = torch.tensor(c2, dtype=dtype, device=lens.device)
    lens.d_sensor = torch.tensor(d_sensor, dtype=dtype, device=lens.device)
    # #region agent log
    import json as _json, time as _time
    with open('/home/hisham26/Desktop/RL/.cursor/debug-cd9c89.log', 'a') as _lf:
        _lf.write(_json.dumps({'sessionId':'cd9c89','timestamp':int(_time.time()*1000),'location':'render_singlet_archive_designs_two_param.py:77','message':'pre-post_computation','data':{'c1':c1,'c2':c2,'d_sensor':d_sensor},'hypothesisId':'H1'}) + '\n')
    # #endregion
    try:
        lens.post_computation()
    except Exception as _exc:
        # #region agent log
        with open('/home/hisham26/Desktop/RL/.cursor/debug-cd9c89.log', 'a') as _lf:
            _lf.write(_json.dumps({'sessionId':'cd9c89','timestamp':int(_time.time()*1000),'location':'render_singlet_archive_designs_two_param.py:83','message':'post_computation CAUGHT exception','data':{'c1':c1,'c2':c2,'exc_type':type(_exc).__name__,'exc_msg':str(_exc)[:200]},'hypothesisId':'H3'}) + '\n')
        # #endregion


def _compute_d_sensor_for_theta(theta: np.ndarray, *, n: float, t_glass: float, z_last: float) -> float:
    """Compute d_sensor from theta=[c1,c2] without loading a lens object."""
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    _eps = 1e-3
    c1 = float(arr[0]) if float(arr[0]) != 0.0 else _eps
    c2 = float(arr[1]) if float(arr[1]) != 0.0 else _eps
    phi_1 = (n - 1.0) * c1
    phi_2 = -(n - 1.0) * c2
    P = phi_1 + phi_2 - t_glass * phi_1 * phi_2 / n
    numerator = 1.0 - t_glass * phi_1 / n
    safe_P = max(P, 1e-4) if P > 0 else min(P, -1e-4)
    return z_last + numerator / safe_P


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers (same as 1-param, but call _apply_theta_to_lens_two_param)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_deeplens_import():
    deeplens_root = PROJECT_ROOT / "DeepLens"
    if str(deeplens_root) not in sys.path:
        sys.path.insert(0, str(deeplens_root))
    from deeplens.optics import GeoLens  # type: ignore
    return GeoLens


def render_design_two_param(
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
    title: str | None = None,
) -> dict[str, float]:
    GeoLens = _ensure_deeplens_import()
    lens = GeoLens(filename=str(lens_template), device=device)
    _apply_theta_to_lens_two_param(lens, design.theta, wvln_um=wavelength_um)

    ax.set_facecolor(theme.background)

    if theme.lens_fill is not None:
        for z_poly, x_poly in _lens_fill_polygons(lens):
            ax.fill(z_poly, x_poly, color=theme.lens_fill, alpha=1.0, linewidth=0.0, zorder=1)

    fig = ax.figure
    ax, fig = lens.draw_lens_2d(ax=ax, fig=fig, color=theme.lens_outline)

    rfov_deg = float(np.rad2deg(float(lens.rfov)))
    fov_values = [float(frac) * rfov_deg for frac in fov_fracs]
    if not fov_values:
        fov_values = [0.0]

    for i, fov in enumerate(fov_values):
        ray = lens.sample_point_source_2D(
            fov=fov,
            depth=float(depth_mm),
            num_rays=int(num_rays),
            wvln=float(wavelength_um),
        )
        ray.prop_to(-1.0)
        _ray_out, ray_o_record = lens.trace2sensor(ray=ray, record=True)
        ray_color = theme.ray_colors[i % len(theme.ray_colors)]
        ax, fig = lens.draw_ray_2d(ray_o_record, ax=ax, fig=fig, color=ray_color)

    spot = design.rms_spot
    if spot is None:
        spot = _compute_rms_spot(
            lens=lens,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
        )

    foclen = float(getattr(lens, "foclen", float("nan")))
    c1 = float(design.theta[0])
    c2 = float(design.theta[1])
    if title is not None:
        ax.set_title(title, fontsize=8)
    else:
        ax.set_title(
            f"#{int(rank)+1}  spot={spot:.3f}mm  f={foclen:.1f}mm\nc1={c1:.4f}  c2={c2:.4f}",
            fontsize=7,
        )
    ax.set_axis_off()

    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"spot_size_mm": float(spot), "focal_length_mm": float(foclen)}


def render_archive_grid_two_param(
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
    panel_titles=None,
) -> list[dict[str, float]]:
    import math
    import matplotlib.pyplot as plt

    def _fmt_float(x, *, ndigits: int = 6) -> str:
        try:
            xf = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(xf):
            return str(xf)
        return f"{xf:.{ndigits}f}"

    def _print_individual_info(*, idx: int, design: ArchiveDesign) -> None:
        cell = getattr(design, "cell", None)
        fitness = getattr(design, "fitness", None)
        measures = getattr(design, "measures", None)
        theta = getattr(design, "theta", None)
        rms_spot = getattr(design, "rms_spot", None)

        try:
            cell_str = f"[{int(cell[0])}, {int(cell[1])}]" if cell is not None else "None"
        except Exception:
            cell_str = str(cell)

        try:
            meas_str = (
                f"[{_fmt_float(measures[0])}, {_fmt_float(measures[1])}]"
                if measures is not None
                else "None"
            )
        except Exception:
            meas_str = str(measures)

        try:
            theta_str = (
                f"[{_fmt_float(theta[0], ndigits=8)}, {_fmt_float(theta[1], ndigits=8)}]"
                if theta is not None
                else "None"
            )
        except Exception:
            theta_str = str(theta)

        spot_str = "None"
        if rms_spot is not None:
            try:
                spot_str = _fmt_float(rms_spot, ndigits=6) if np.isfinite(float(rms_spot)) else str(rms_spot)
            except Exception:
                spot_str = str(rms_spot)

        fit_str = _fmt_float(fitness, ndigits=6) if fitness is not None else "None"

        print(
            f"[render-2p] about to render rank={idx} cell={cell_str} "
            f"fitness={fit_str} measures={meas_str} theta={theta_str} rms_spot={spot_str}",
            flush=True,
        )

    n = len(selected)
    if n == 0:
        return []
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))
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
        _print_individual_info(idx=idx, design=selected[idx])
        ptitle = None
        if panel_titles is not None and idx < len(panel_titles):
            ptitle = panel_titles[idx]
        metrics = render_design_two_param(
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
            title=ptitle,
        )
        panel_metrics.append(metrics)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor=theme.background)
    plt.close(fig)
    return panel_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Archive auto-discovery (two-param run dirs)
# ──────────────────────────────────────────────────────────────────────────────

def _find_latest_archive(results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else PROJECT_ROOT / "results" / "models"
    candidates = list(root.glob("singlet_cmamega_two_param_*/archive.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No archive.json found under {root}/singlet_cmamega_two_param_*"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _default_out_dir(archive: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return archive.parent / f"renders_2p_diverse_top9_{ts}"


def _default_lens_template() -> Path:
    return PROJECT_ROOT / "DeepLens" / "datasets" / "lenses" / "singlet" / "example1.json"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Render diverse high-quality two-param singlet designs from CMA-MEGA archive."
    )
    p.add_argument("--archive", default=None,
                   help="Path to archive.json (default: latest singlet_cmamega_two_param run)")
    p.add_argument("--lens-template", default=str(_default_lens_template()))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=-1000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-fitness", type=float, default=-0.03,
                   help="Exclude designs with fitness below this value.")
    p.add_argument("--max-d-sensor", type=float, default=100.0,
                   help="Reject designs whose paraxial d_sensor exceeds this value (mm).")
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F8F5E9")
    p.add_argument("--lens-fill-color", default="#9DC08B")
    p.add_argument("--lens-outline-color", default="#3A7D44")
    p.add_argument("--ray-color", default="#DF6D14")
    return p.parse_args(argv)


def main(argv=None):
    import json

    args = parse_args(argv)
    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    lens_template = Path(args.lens_template)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(archive_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    designs = load_archive(archive_path)
    if args.min_fitness is not None:
        before = len(designs)
        designs = [d for d in designs if d.fitness >= float(args.min_fitness)]
        print(
            f"[render-2p] min-fitness filter: kept {len(designs)}/{before} designs "
            f"(fitness >= {args.min_fitness})",
            flush=True,
        )
        if not designs:
            print("[render-2p] no designs pass the min-fitness filter", flush=True)
            return

    if args.max_d_sensor is not None:
        GeoLens = _ensure_deeplens_import()
        _tpl = GeoLens(filename=str(lens_template), device="cpu")
        _n = float(_tpl.surfaces[1].mat2.refractive_index(float(args.wavelength_um)))
        _t_glass = float(_tpl.surfaces[2].d.item() - _tpl.surfaces[1].d.item())
        _z_last = float(_tpl.surfaces[-1].d.item())
        del _tpl
        before = len(designs)
        designs = [
            d for d in designs
            if abs(_compute_d_sensor_for_theta(d.theta, n=_n, t_glass=_t_glass, z_last=_z_last))
               <= float(args.max_d_sensor)
        ]
        print(
            f"[render-2p] max-d-sensor filter: kept {len(designs)}/{before} designs "
            f"(|d_sensor| <= {args.max_d_sensor} mm)",
            flush=True,
        )
        if not designs:
            print("[render-2p] no designs pass the max-d-sensor filter", flush=True)
            return

    selected = select_diverse_top(
        designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    if not selected:
        print("[render-2p] no valid designs found in archive", flush=True)
        return

    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot) if d.rms_spot is not None and np.isfinite(d.rms_spot)
            else float("inf"),
            -float(d.fitness),
        ),
    )

    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    grid_path = out_dir / "archive_diverse_top_grid.png"
    print("reached ")
    panel_metrics = render_archive_grid_two_param(
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
    )
    print(f"[render-2p] saved {grid_path.name}", flush=True)
    print("reached 2")

    selected_meta = []
    for rank, (design, panel) in enumerate(zip(selected, panel_metrics)):
        selected_meta.append(
            {
                "rank": rank,
                "cell": [int(design.cell[0]), int(design.cell[1])],
                "fitness": float(design.fitness),
                "rms_spot_from_archive": (
                    float(design.rms_spot)
                    if design.rms_spot is not None and np.isfinite(design.rms_spot)
                    else None
                ),
                "spot_size_mm": float(panel["spot_size_mm"]),
                "focal_length_mm": float(panel["focal_length_mm"]),
                "measures": [float(design.measures[0]), float(design.measures[1])],
                "theta": [float(x) for x in design.theta.tolist()],
            }
        )

    summary = {
        "archive": str(archive_path),
        "lens_template": str(lens_template),
        "selection_policy": (
            "Top+Farthest in measure space for diversity, then sorted by spot size."
        ),
        "use_default_theme": bool(args.use_default_theme),
        "theme": {
            "background": theme.background,
            "lens_fill": theme.lens_fill,
            "lens_outline": theme.lens_outline,
            "ray_colors": list(theme.ray_colors),
        },
        "count": len(selected),
        "top_frac": float(args.top_frac),
        "seed": int(args.seed),
        "depth_mm": float(args.depth_mm),
        "wavelength_um": float(args.wavelength_um),
        "num_rays": int(args.num_rays),
        "fov_fracs": [float(x) for x in fov_fracs],
        "grid_cols": int(args.grid_cols),
        "dpi": int(args.dpi),
        "device": str(args.device),
        "selected": selected_meta,
        "grid_image": grid_path.name,
    }
    (out_dir / "selected_designs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[render-2p] outputs at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
