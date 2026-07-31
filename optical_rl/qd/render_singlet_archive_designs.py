from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ArchiveDesign:
    cell: tuple[int, int]
    fitness: float
    measures: tuple[float, float]
    theta: np.ndarray
    rms_spot: float | None = None


@dataclass(frozen=True)
class PlotTheme:
    background: str
    lens_fill: str | None
    lens_outline: str
    ray_colors: list[str]


def resolve_theme(
    *,
    use_default_theme: bool,
    bg_color: str,
    lens_fill_color: str,
    lens_outline_color: str,
    ray_color: str,
) -> PlotTheme:
    if use_default_theme:
        return PlotTheme(
            background="#FFFFFF",
            lens_fill=None,
            lens_outline="black",
            ray_colors=["#CC0000", "#006600", "#0066CC"],
        )
    return PlotTheme(
        background=bg_color,
        lens_fill=lens_fill_color,
        lens_outline=lens_outline_color,
        ray_colors=[ray_color],
    )


def load_archive(path: Path) -> list[ArchiveDesign]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[ArchiveDesign] = []
    for row in rows:
        try:
            cell = tuple(int(x) for x in row["cell"])
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


def _normalized_measure_points(designs: Sequence[ArchiveDesign]) -> np.ndarray:
    if not designs:
        return np.zeros((0, 2), dtype=np.float64)
    m = np.array([d.measures for d in designs], dtype=np.float64)
    lo = m.min(axis=0)
    hi = m.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return (m - lo) / span


def select_diverse_top(
    designs: Sequence[ArchiveDesign],
    *,
    count: int,
    top_frac: float,
    seed: int,
) -> list[ArchiveDesign]:
    if count <= 0 or not designs:
        return []
    ordered = sorted(designs, key=lambda d: d.fitness, reverse=True)
    pool_n = max(int(count) * 4, int(math.ceil(float(top_frac) * len(ordered))))
    pool_n = int(np.clip(pool_n, 1, len(ordered)))
    pool = ordered[:pool_n]
    points = _normalized_measure_points(pool)
    rng = np.random.default_rng(int(seed))

    selected_idx: list[int] = [0]
    available = set(range(1, len(pool)))

    while len(selected_idx) < min(int(count), len(pool)):
        best_idx = None
        best_score = -float("inf")
        for idx in available:
            p = points[idx]
            dists = [float(np.linalg.norm(p - points[j])) for j in selected_idx]
            min_dist = min(dists) if dists else 0.0
            score = min_dist
            if score > best_score:
                best_score = score
                best_idx = idx
            elif score == best_score and best_idx is not None:
                # tie-break: higher fitness, then random stable jitter
                if pool[idx].fitness > pool[best_idx].fitness:
                    best_idx = idx
                elif pool[idx].fitness == pool[best_idx].fitness:
                    if rng.random() < 0.5:
                        best_idx = idx
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        available.remove(best_idx)

    return [pool[i] for i in selected_idx]


def _ensure_deeplens_import():
    deeplens_root = PROJECT_ROOT / "DeepLens"
    if str(deeplens_root) not in sys.path:
        sys.path.insert(0, str(deeplens_root))
    from deeplens.optics import GeoLens  # type: ignore

    return GeoLens


def _lens_fill_polygons(lens, *, num_samples: int = 128):
    polys = []
    for i in range(len(lens.surfaces) - 1):
        s0 = lens.surfaces[i]
        s1 = lens.surfaces[i + 1]
        if float(s0.mat2.n) <= 1.1:
            continue

        r0 = float(s0.draw_r()) * 0.999
        r1 = float(s1.draw_r()) * 0.999
        r_eff = min(r0, r1)
        if r_eff <= 1e-8:
            continue

        rs = np.linspace(-r_eff, r_eff, num_samples, dtype=np.float64)
        import torch

        rt = torch.tensor(rs, dtype=torch.float32, device=lens.device)
        z0 = s0.surface_with_offset(rt, torch.zeros_like(rt), valid_check=False).detach().cpu().numpy()
        z1 = s1.surface_with_offset(rt, torch.zeros_like(rt), valid_check=False).detach().cpu().numpy()

        z_poly = np.concatenate([z0, z1[::-1]], axis=0)
        x_poly = np.concatenate([rs, rs[::-1]], axis=0)
        polys.append((z_poly, x_poly))
    return polys


def _apply_theta_to_lens(lens, theta: np.ndarray, wvln_um: float = 0.587):
    """Apply a 1D theta = [c1] to the lens, deriving c2 and d_sensor via paraxial solve."""
    import torch

    c1 = float(np.asarray(theta, dtype=np.float64).reshape(-1)[0])

    # Read paraxial constants from the template lens (before modification).
    n = float(lens.surfaces[1].mat2.refractive_index(wvln_um))
    t_glass = float(lens.surfaces[2].d.item() - lens.surfaces[1].d.item())
    z_last = float(lens.surfaces[-1].d.item())
    f_target = float(lens.foclen) if hasattr(lens, "foclen") else float(lens.efl)

    P = 1.0 / f_target
    phi_1 = (n - 1.0) * c1
    raw_denom = 1.0 - t_glass * phi_1 / n
    safe_denom = max(raw_denom, 1e-4) if raw_denom > 0 else min(raw_denom, -1e-4)
    phi_2 = (P - phi_1) / safe_denom
    c2 = -phi_2 / (n - 1.0)
    bfl = safe_denom / P
    d_sensor = z_last + bfl

    dtype = lens.d_sensor.dtype
    lens.surfaces[1].c = torch.tensor(c1, dtype=dtype, device=lens.device)
    lens.surfaces[2].c = torch.tensor(c2, dtype=dtype, device=lens.device)
    lens.d_sensor = torch.tensor(d_sensor, dtype=dtype, device=lens.device)
    try:
        lens.post_computation()
    except Exception:
        pass


def _compute_rms_spot(
    *,
    lens,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
) -> float:
    import torch

    ray = lens.sample_point_source_2D(
        fov=0.0,
        depth=float(depth_mm),
        num_rays=int(num_rays),
        wvln=float(wavelength_um),
    )
    ray.prop_to(-1.0)
    ray_out = lens.trace2sensor(ray=ray)
    valid = ray_out.is_valid
    denom = valid.sum().clamp_min(1e-8)
    xy = ray_out.o[..., :2]
    centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
    err2 = ((xy - centroid) ** 2).sum(dim=-1)
    rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
    return float(rms.detach().cpu().item())


def render_design(
    *,
    lens_template: Path,
    design: ArchiveDesign,
    theme: PlotTheme,
    ax,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Iterable[float],
    rank: int,
    device: str,
    title: str | None = None,
) -> dict[str, float]:
    GeoLens = _ensure_deeplens_import()
    lens = GeoLens(filename=str(lens_template), device=device)
    _apply_theta_to_lens(lens, design.theta, wvln_um=wavelength_um)

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
    if title is not None:
        ax.set_title(title, fontsize=8)
    else:
        ax.set_title(f"#{int(rank) + 1}  spot={spot:.3f} mm  f={foclen:.2f} mm", fontsize=8)
    ax.set_axis_off()

    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"spot_size_mm": float(spot), "focal_length_mm": float(foclen)}


def render_archive_grid(
    *,
    selected: Sequence[ArchiveDesign],
    lens_template: Path,
    theme: PlotTheme,
    out_path: Path,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Iterable[float],
    cols: int,
    dpi: int,
    device: str,
    panel_titles: Sequence[str | None] | None = None,
) -> list[dict[str, float]]:
    import matplotlib.pyplot as plt

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
        ptitle = None
        if panel_titles is not None and idx < len(panel_titles):
            ptitle = panel_titles[idx]
        metrics = render_design(
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


def _parse_fov_fracs(s: str) -> list[float]:
    vals = []
    for part in str(s).split(","):
        t = part.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _default_out_dir(archive: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return archive.parent / f"renders_diverse_top9_{ts}"


def _default_lens_template() -> Path:
    return PROJECT_ROOT / "DeepLens" / "datasets" / "lenses" / "singlet" / "example1.json"


def _find_latest_archive(results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else PROJECT_ROOT / "results" / "models"
    candidates = list(root.glob("singlet_cmamega_*/archive.json"))
    if not candidates:
        raise FileNotFoundError(f"No archive.json found under {root}/singlet_cmamega_*")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render diverse high-quality singlet designs from CMA-MEGA archive.")
    p.add_argument("--archive", default=None, help="Path to archive.json (default: latest singlet_cmamega run)")
    p.add_argument(
        "--lens-template",
        default=str(_default_lens_template()),
        help="Path to lens template json",
    )
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=-1000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--grid-cols", type=int, default=9)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu", help="Rendering device: cpu, cuda, mps, or auto")

    p.add_argument("--min-fitness", type=float, default=-10,
                   help="Exclude designs with fitness below this value (default: -0.002). "
                        "Fitness is negative; higher (less negative) = smaller spot.")

    p.add_argument("--use-default-theme", action="store_true")
    p.add_argument("--bg-color", default="#F8F5E9")
    p.add_argument("--lens-fill-color", default="#9DC08B")
    p.add_argument("--lens-outline-color", default="#3A7D44")
    p.add_argument("--ray-color", default="#DF6D14")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    lens_template = Path(args.lens_template)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(archive_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    designs = load_archive(archive_path)
    if args.min_fitness is not None:
        before = len(designs)
        designs = [d for d in designs if d.fitness >= float(args.min_fitness)]
        print(f"[render-singlet] min-fitness filter: kept {len(designs)}/{before} designs "
              f"(fitness >= {args.min_fitness})", flush=True)
        if not designs:
            print("[render-singlet] no designs pass the min-fitness filter", flush=True)
            return
    selected = select_diverse_top(
        designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    if not selected:
        print("[render-singlet] no valid designs found in archive", flush=True)
        return

    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot) if d.rms_spot is not None and np.isfinite(d.rms_spot) else float("inf"),
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

    selected_meta = []
    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid(
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
    print(f"[render-singlet] saved {grid_path.name}", flush=True)

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
            "Top+Farthest in measure space for diversity, then sorted by spot size for render order."
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
    (out_dir / "selected_designs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[render-singlet] outputs at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
