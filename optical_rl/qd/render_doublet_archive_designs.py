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
class _DoubletLayout:
    refractive_idx: tuple[int, int, int, int]
    stop_idx: int
    thickness_1: float
    thickness_2: float
    stop_d: float


def load_archive(path: Path) -> list[ArchiveDesign]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[ArchiveDesign] = []
    for row in rows:
        try:
            cell = tuple(int(x) for x in row["cell"])
            fitness = float(row["fitness"])
            measures = tuple(float(x) for x in row["measures"])
            theta = np.asarray(row["theta"], dtype=np.float64).reshape(6)
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
                if pool[idx].fitness > pool[best_idx].fitness:
                    best_idx = idx
                elif pool[idx].fitness == pool[best_idx].fitness and rng.random() < 0.5:
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


def _prepare_doublet_lens(lens) -> _DoubletLayout:
    import torch

    if len(lens.surfaces) >= 7:
        keep = [2, 3, 4, 5, 6]
    else:
        keep = list(range(len(lens.surfaces)))
    kept = [lens.surfaces[i] for i in keep]
    if len(kept) != 5:
        raise ValueError(f"Expected 5 surfaces in starter block, got {len(kept)}")
    d0 = float(kept[0].d.item())
    for s in kept:
        s.d = torch.tensor(float(s.d.item()) - d0, dtype=s.d.dtype, device=lens.device)
    lens.surfaces = kept
    lens.d_sensor = torch.tensor(float(lens.d_sensor.item()) - d0, dtype=lens.d_sensor.dtype, device=lens.device)

    refractive_idx = (0, 1, 3, 4)
    stop_idx = 2
    s0 = lens.surfaces[refractive_idx[0]]
    s1 = lens.surfaces[refractive_idx[1]]
    s2 = lens.surfaces[refractive_idx[2]]
    s3 = lens.surfaces[refractive_idx[3]]
    stop = lens.surfaces[stop_idx]
    return _DoubletLayout(
        refractive_idx=refractive_idx,
        stop_idx=stop_idx,
        thickness_1=float(s1.d.item() - s0.d.item()),
        thickness_2=float(s3.d.item() - s2.d.item()),
        stop_d=float(stop.d.item()),
    )


def _apply_theta_to_doublet_lens(lens, layout: _DoubletLayout, theta: np.ndarray):
    import torch

    t = np.asarray(theta, dtype=np.float64).reshape(6)
    s0, s1, s2, s3 = layout.refractive_idx
    stop = layout.stop_idx

    lens.surfaces[s0].c = torch.tensor(float(t[0]), dtype=lens.surfaces[s0].c.dtype, device=lens.device)
    lens.surfaces[s1].c = torch.tensor(float(t[1]), dtype=lens.surfaces[s1].c.dtype, device=lens.device)
    lens.surfaces[s2].c = torch.tensor(float(t[2]), dtype=lens.surfaces[s2].c.dtype, device=lens.device)
    lens.surfaces[s3].c = torch.tensor(float(t[3]), dtype=lens.surfaces[s3].c.dtype, device=lens.device)

    d0 = float(lens.surfaces[s0].d.item())
    dtype = lens.surfaces[s0].d.dtype
    lens.surfaces[s1].d = lens.surfaces[s0].d + torch.tensor(float(layout.thickness_1), dtype=dtype, device=lens.device)
    lens.surfaces[stop].d = torch.tensor(d0 + float(layout.stop_d), dtype=dtype, device=lens.device)
    lens.surfaces[s2].d = lens.surfaces[s1].d + torch.tensor(float(t[4]), dtype=dtype, device=lens.device)
    lens.surfaces[s3].d = lens.surfaces[s2].d + torch.tensor(float(layout.thickness_2), dtype=dtype, device=lens.device)
    lens.d_sensor = torch.tensor(float(t[5]), dtype=lens.d_sensor.dtype, device=lens.device)
    lens.post_computation()


def _compute_rms_spot(*, lens, depth_mm: float, wavelength_um: float, num_rays: int) -> float:
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
    ax,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Iterable[float],
    rank: int,
    device: str,
) -> dict[str, float]:
    GeoLens = _ensure_deeplens_import()
    lens = GeoLens(filename=str(lens_template), device=device)
    layout = _prepare_doublet_lens(lens)
    _apply_theta_to_doublet_lens(lens, layout, design.theta)

    fig = ax.figure
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax, fig = lens.draw_lens_2d(ax=ax, fig=fig, color="black")

    ray_colors = ["#CC0000", "#006600", "#0066CC"]
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
        ax, fig = lens.draw_ray_2d(ray_o_record, ax=ax, fig=fig, color=ray_colors[i % len(ray_colors)])

    spot = design.rms_spot
    if spot is None:
        spot = _compute_rms_spot(
            lens=lens,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
        )
    foclen = float(getattr(lens, "foclen", float("nan")))
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
    out_path: Path,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_fracs: Iterable[float],
    cols: int,
    dpi: int,
    device: str,
) -> list[dict[str, float]]:
    import matplotlib.pyplot as plt

    n = len(selected)
    if n == 0:
        return []
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.9 * rows), dpi=int(dpi))
    fig.patch.set_facecolor("#FFFFFF")
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
        ax.set_facecolor("#FFFFFF")
        ax.axis("off")
        if idx >= n:
            continue
        metrics = render_design(
            lens_template=lens_template,
            design=selected[idx],
            ax=ax,
            depth_mm=float(depth_mm),
            wavelength_um=float(wavelength_um),
            num_rays=int(num_rays),
            fov_fracs=fov_fracs,
            rank=idx,
            device=device,
        )
        panel_metrics.append(metrics)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor="#FFFFFF")
    plt.close(fig)
    return panel_metrics


def _parse_fov_fracs(s: str) -> list[float]:
    vals = []
    for part in str(s).split(","):
        t = part.strip()
        if t:
            vals.append(float(t))
    return vals


def _default_out_dir(archive: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return archive.parent / f"renders_diverse_top9_{ts}"


def _default_lens_template() -> Path:
    return PROJECT_ROOT / "optical_rl" / "qd" / "doublet_starter.json"


def _find_latest_archive(results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else PROJECT_ROOT / "results" / "models"
    candidates = list(root.glob("doublet_cmamega_*/archive.json"))
    if not candidates:
        raise FileNotFoundError(f"No archive.json found under {root}/doublet_cmamega_*")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render diverse high-quality doublet designs from CMA-MEGA archive.")
    p.add_argument("--archive", default=None, help="Path to archive.json (default: latest doublet_cmamega run)")
    p.add_argument("--lens-template", default=str(_default_lens_template()), help="Path to cooke lens template json")
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=-1000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-fracs", default="0.0,0.5,0.9")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu", help="Rendering device: cpu, cuda, mps, or auto")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    lens_template = Path(args.lens_template)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(archive_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    designs = load_archive(archive_path)
    selected = select_diverse_top(
        designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    if not selected:
        print("[render-doublet] no valid designs found in archive", flush=True)
        return

    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot) if d.rms_spot is not None and np.isfinite(d.rms_spot) else float("inf"),
            -float(d.fitness),
        ),
    )
    fov_fracs = _parse_fov_fracs(str(args.fov_fracs))

    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid(
        selected=selected,
        lens_template=lens_template,
        out_path=grid_path,
        depth_mm=float(args.depth_mm),
        wavelength_um=float(args.wavelength_um),
        num_rays=int(args.num_rays),
        fov_fracs=fov_fracs,
        cols=int(args.grid_cols),
        dpi=int(args.dpi),
        device=str(args.device),
    )
    print(f"[render-doublet] saved {grid_path.name}", flush=True)

    selected_meta = []
    for rank, (design, panel) in enumerate(zip(selected, panel_metrics)):
        selected_meta.append(
            {
                "rank": rank,
                "cell": [int(design.cell[0]), int(design.cell[1])],
                "fitness": float(design.fitness),
                "rms_spot_from_archive": (
                    float(design.rms_spot) if design.rms_spot is not None and np.isfinite(design.rms_spot) else None
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
    print(f"[render-doublet] outputs at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
