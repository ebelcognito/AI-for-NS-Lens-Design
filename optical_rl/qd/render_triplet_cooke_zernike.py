"""Render diverse high-quality designs from a Cooke-triplet Zernike CMA-MEGA archive.

Mirrors render_singlet_archive_designs_two_param.py in structure.
theta = [c0, c1, c2, c3, c5, c6]  (6 surface curvatures)
d_sensor is derived from the paraxial ABCD back-focal-length solve.

Usage:
  python3 optical_rl/qd/render_triplet_cooke_zernike.py
  python3 optical_rl/qd/render_triplet_cooke_zernike.py --archive results/models/triplet_cooke_zernike_*/archive.json
  python3 optical_rl/qd/render_triplet_cooke_zernike.py --count 12 --grid-cols 4 --fov-fracs 0.0,0.5,0.9
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Optional YAML config reader (same helper used by main_* scripts)
from optical_rl.utils.config import load_config  # noqa: E402

# Reuse selection, theming, archive loading from singlet renderer.
from optical_rl.qd.render_singlet_archive_designs import (  # noqa: E402
    ArchiveDesign,
    PlotTheme,
    _compute_rms_spot,
    _lens_fill_polygons,
    _parse_fov_fracs,
    load_archive,
    resolve_theme,
    select_diverse_top,
)

# Refractive surface indices for the Cooke triplet (stop is surface 4, skipped).
_SURF_IDX = (0, 1, 2, 3, 5, 6)


# ---------------------------------------------------------------------------
# Paraxial ABCD BFL solve (pure Python / numpy — no PyTorch needed here)
# ---------------------------------------------------------------------------

def _paraxial_bfl_triplet(
    c0, c1, c2, c3, c5, c6,
    *,
    n1: float, n2: float, n3: float,
    d01: float, d12: float, d23: float, d35: float, d56: float,
) -> float:
    """Chain paraxial ABCD matrices → BFL (mm).

    Marginal ray: [y=1, nu=0] in object space.
    Returns back focal length (distance from surface 6 to focus).
    """
    y, nu = 1.0, 0.0
    nu -= (n1 - 1.0) * c0 * y;     y += (d01 / n1) * nu   # surf 0, gap 01
    nu -= (1.0 - n1) * c1 * y;     y += d12 * nu           # surf 1, gap 12
    nu -= (n2 - 1.0) * c2 * y;     y += (d23 / n2) * nu   # surf 2, gap 23
    nu -= (1.0 - n2) * c3 * y;     y += d35 * nu           # surf 3, gap 3→5
    nu -= (n3 - 1.0) * c5 * y;     y += (d56 / n3) * nu   # surf 5, gap 56
    nu -= (1.0 - n3) * c6 * y                               # surf 6
    safe_nu = nu if abs(nu) > 1e-9 else 1e-9
    return -y / safe_nu


def _derive_c6_from_efl(
    c0, c1, c2, c3, c5,
    *,
    n1: float, n2: float, n3: float,
    d01: float, d12: float, d23: float, d35: float, d56: float,
    f_target: float,
) -> tuple[float, float]:
    """Numpy mirror of TripletCookeZernikeProblem._paraxial_fixed_efl.

    Traces the marginal ray through surfaces 0-5, then solves for c6 such that
    EFL = f_target.  Returns (c6, bfl).
    """
    y, nu = 1.0, 0.0
    nu -= (n1 - 1.0) * c0 * y;  y += (d01 / n1) * nu
    nu -= (1.0 - n1) * c1 * y;  y += d12 * nu
    nu -= (n2 - 1.0) * c2 * y;  y += (d23 / n2) * nu
    nu -= (1.0 - n2) * c3 * y;  y += d35 * nu
    nu -= (n3 - 1.0) * c5 * y;  y += (d56 / n3) * nu
    # Now at exit of element-3 glass; surface 6 is glass n3 → air.
    P = 1.0 / f_target
    safe_y = y if abs(y) > 1e-9 else 1e-9
    phi_6 = (nu + P) / safe_y
    c6 = phi_6 / (1.0 - n3)
    bfl = y * f_target  # exact: -y / (-P)
    return c6, bfl


def _apply_theta_to_lens_triplet(
    lens,
    theta: np.ndarray,
    wvln_um: float = 0.587,
    focal_length_target_mm: float | None = None,
) -> None:
    """Set curvatures on *lens* and derive d_sensor from the paraxial BFL solve.

    Handles both modes:
      6-D theta [c0,c1,c2,c3,c5,c6]: all curvatures free; d_sensor from BFL.
      5-D theta [c0,c1,c2,c3,c5]:    c6 derived to enforce EFL = focal_length_target_mm.
    """
    import torch

    arr = np.asarray(theta, dtype=np.float64).reshape(-1)

    n1 = float(lens.surfaces[0].mat2.refractive_index(wvln_um))
    n2 = float(lens.surfaces[2].mat2.refractive_index(wvln_um))
    n3 = float(lens.surfaces[5].mat2.refractive_index(wvln_um))

    d = [float(lens.surfaces[i].d.item()) for i in range(7)]
    d01 = d[1] - d[0]
    d12 = d[2] - d[1]
    d23 = d[3] - d[2]
    d35 = d[5] - d[3]
    d56 = d[6] - d[5]
    z_s6 = d[6]

    c0, c1, c2, c3, c5 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])

    if len(arr) == 5:
        # Fixed-EFL mode: derive c6 from the paraxial solve.
        if focal_length_target_mm is None:
            raise ValueError(
                "theta is 5-D (fixed-EFL archive) but focal_length_target_mm was not supplied. "
                "Pass --config or --focal-length when rendering a fixed-EFL archive."
            )
        c6, bfl = _derive_c6_from_efl(
            c0, c1, c2, c3, c5,
            n1=n1, n2=n2, n3=n3,
            d01=d01, d12=d12, d23=d23, d35=d35, d56=d56,
            f_target=float(focal_length_target_mm),
        )
        d_sensor = z_s6 + max(bfl, 1.0)
    else:
        # Free mode: c6 is the 6th element of theta.
        c6 = float(arr[5])
        bfl = _paraxial_bfl_triplet(
            c0, c1, c2, c3, c5, c6,
            n1=n1, n2=n2, n3=n3,
            d01=d01, d12=d12, d23=d23, d35=d35, d56=d56,
        )
        d_sensor = z_s6 + max(bfl, 1.0)

    curvatures = [c0, c1, c2, c3, c5, c6]
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

def render_design_triplet(
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
    _apply_theta_to_lens_triplet(
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
        surf_labels = [0, 1, 2, 3, 5, 6]
        theta_arr = np.asarray(design.theta, dtype=np.float64).reshape(-1)
        theta_str = "  ".join(
            f"c{surf_labels[k]}={float(theta_arr[k]):.4f}" for k in range(len(theta_arr))
        )
        efl_note = (
            f"  EFL={focal_length_target_mm:.0f}mm(fixed)"
            if focal_length_target_mm is not None and len(theta_arr) == 5
            else f"  f={foclen:.1f}mm"
        )
        ax.set_title(
            f"#{rank+1}  spot={spot:.4f}mm{efl_note}\n{theta_str}",
            fontsize=6,
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

def render_archive_grid_triplet(
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
            f"[render-triplet-zernike] rendering #{idx+1}/{n} "
            f"fitness={selected[idx].fitness:.5f}",
            flush=True,
        )
        ptitle = panel_titles[idx] if panel_titles and idx < len(panel_titles) else None
        metrics = render_design_triplet(
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

def _find_latest_archive(results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else PROJECT_ROOT / "results" / "models"
    candidates = list(root.glob("triplet_cooke_zernike_*/archive.json"))
    if not candidates:
        raise FileNotFoundError(f"No archive.json found under {root}/triplet_cooke_zernike_*")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Render diverse Cooke-triplet Zernike designs from CMA-MEGA archive."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config; used to supply defaults like qd.render_min_fitness.",
    )
    p.add_argument("--archive", default=None,
                   help="Path to archive.json (default: latest triplet_cooke_zernike_* run)")
    p.add_argument("--lens-template",
                   default=str(PROJECT_ROOT / "DeepLens/datasets/lenses/cooke_smaller_ap_1mm.json"))
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
    p.add_argument("--min-fitness", type=float, default=None,
                   help="Exclude designs with fitness below this threshold.")
    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F0EDE4")
    p.add_argument("--lens-fill-color", default="#7BAFD4")
    p.add_argument("--lens-outline-color", default="#1A4E8C")
    p.add_argument("--ray-color", default="#E07B39")
    return p.parse_args(argv)


def _load_qd_config(config_path: str | None, archive_path: Path) -> dict:
    """Load the qd section from the YAML config.

    Priority:
      1. Explicit --config argument.
      2. Any *.yaml file saved in the archive's run directory.
    """
    paths_to_try: list[Path] = []
    if config_path:
        paths_to_try.append(Path(config_path))
    for p in sorted(archive_path.parent.glob("*.yaml")):
        paths_to_try.append(p)
    for p in paths_to_try:
        try:
            cfg = load_config(str(p)) or {}
            if isinstance(cfg, dict) and "qd" in cfg:
                return cfg["qd"]
        except Exception:
            pass
    return {}


def main(argv=None):
    import json

    args = _parse_args(argv)

    archive_path = Path(args.archive) if args.archive else _find_latest_archive()

    # Read QD config from YAML (--config or the YAML saved in the run dir).
    qd_cfg = _load_qd_config(args.config, archive_path)

    focal_length_target_mm: float | None = None
    if qd_cfg.get("focal_length_target_mm") is not None:
        focal_length_target_mm = float(qd_cfg["focal_length_target_mm"])

    # YAML-driven default min-fitness (only if --min-fitness not explicitly set)
    if args.min_fitness is None and qd_cfg.get("render_min_fitness") is not None:
        args.min_fitness = float(qd_cfg["render_min_fitness"])
    elif args.min_fitness is None and args.config:
        try:
            cfg = load_config(args.config) or {}
            qd = cfg.get("qd", {}) if isinstance(cfg, dict) else {}
            if "render_min_fitness" in qd and qd["render_min_fitness"] is not None:
                args.min_fitness = float(qd["render_min_fitness"])
        except Exception:
            pass
    lens_template = Path(args.lens_template)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else archive_path.parent / f"renders_zernike_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-triplet-zernike] archive:      {archive_path}", flush=True)
    print(f"[render-triplet-zernike] lens template: {lens_template}", flush=True)
    print(f"[render-triplet-zernike] output dir:    {out_dir}", flush=True)
    if focal_length_target_mm is not None:
        print(f"[render-triplet-zernike] EFL constraint: {focal_length_target_mm} mm (5-D archive)", flush=True)
    else:
        print(f"[render-triplet-zernike] EFL constraint: none (6-D archive)", flush=True)

    designs = load_archive(archive_path)
    print(f"[render-triplet-zernike] loaded {len(designs)} designs", flush=True)

    if args.min_fitness is not None:
        before = len(designs)
        designs = [d for d in designs if d.fitness >= float(args.min_fitness)]
        print(f"[render-triplet-zernike] min-fitness filter: kept {len(designs)}/{before}", flush=True)

    if not designs:
        print("[render-triplet-zernike] no designs to render", flush=True)
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
    print(f"[render-triplet-zernike] selected {len(selected)} diverse designs", flush=True)

    theme = resolve_theme(
        use_default_theme=bool(args.use_default_theme),
        bg_color=str(args.bg_color),
        lens_fill_color=str(args.lens_fill_color),
        lens_outline_color=str(args.lens_outline_color),
        ray_color=str(args.ray_color),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid_triplet(
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
    print(f"[render-triplet-zernike] grid saved: {grid_path}", flush=True)

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
    print(f"[render-triplet-zernike] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
