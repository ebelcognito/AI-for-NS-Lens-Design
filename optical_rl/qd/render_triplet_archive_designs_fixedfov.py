from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.utils.config import load_config


@dataclass(frozen=True)
class ArchiveDesign:
    cell: tuple[int, int]
    fitness: float
    measures: tuple[float, float]
    theta: np.ndarray
    rms_spot: float | None = None


def load_archive(path: Path) -> list[ArchiveDesign]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[ArchiveDesign] = []
    for row in rows:
        try:
            cell = tuple(int(x) for x in row["cell"])
            fitness = float(row["fitness"])
            measures = tuple(float(x) for x in row["measures"])
            theta = np.asarray(row["theta"], dtype=np.float64).reshape(7)
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


def _apply_theta_to_triplet_lens(lens, theta: np.ndarray):
    import torch

    t = np.asarray(theta, dtype=np.float64).reshape(7)
    refractive_idx = (0, 1, 2, 3, 5, 6)
    for i, surf_idx in enumerate(refractive_idx):
        lens.surfaces[surf_idx].c = torch.tensor(
            float(t[i]),
            dtype=lens.surfaces[surf_idx].c.dtype,
            device=lens.device,
        )
    lens.d_sensor = torch.tensor(float(t[6]), dtype=lens.d_sensor.dtype, device=lens.device)
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


def _normalize_fov_degs(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        v = float(value)
        if not np.isfinite(v):
            continue
        if not any(abs(v - u) < 1e-12 for u in out):
            out.append(v)
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        token = part.strip()
        if token:
            out.append(float(token))
    return out


def _resolve_fov_degs(*, config_path: Path, fov_degs_override: str | None) -> list[float]:
    if fov_degs_override:
        values = _normalize_fov_degs(_parse_float_list(fov_degs_override))
        if values:
            return values

    cfg = load_config(str(config_path))
    qd_cfg = cfg.get("qd", {})
    cfg_list = qd_cfg.get("render_fov_degs")
    if isinstance(cfg_list, (list, tuple)):
        values = _normalize_fov_degs([float(x) for x in cfg_list])
        if values:
            return values

    alpha = float(qd_cfg.get("alpha_deg", 0.5))
    stage2 = float(qd_cfg.get("stage2_field_deg", 2.0))
    values = _normalize_fov_degs([0.0, alpha, stage2])
    return values if values else [0.0]


def render_design(
    *,
    lens_template: Path,
    design: ArchiveDesign,
    ax,
    depth_mm: float,
    wavelength_um: float,
    num_rays: int,
    fov_degs: Sequence[float],
    rank: int,
    device: str,
) -> dict[str, float]:
    GeoLens = _ensure_deeplens_import()
    lens = GeoLens(filename=str(lens_template), device=device)
    _apply_theta_to_triplet_lens(lens, design.theta)

    fig = ax.figure
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax, fig = lens.draw_lens_2d(ax=ax, fig=fig, color="black")

    ray_colors = ["#CC0000", "#006600", "#0066CC"]
    use_fovs = list(fov_degs) if fov_degs else [0.0]
    for i, fov in enumerate(use_fovs):
        ray = lens.sample_point_source_2D(
            fov=float(fov),
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
    fov_degs: Sequence[float],
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
            fov_degs=fov_degs,
            rank=idx,
            device=device,
        )
        panel_metrics.append(metrics)

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), facecolor="#FFFFFF")
    plt.close(fig)
    return panel_metrics


def _default_out_dir(archive: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return archive.parent / f"renders_diverse_top9_fixedfov_{ts}"


def _default_lens_template() -> Path:
    return PROJECT_ROOT / "DeepLens" / "datasets" / "lenses" / "cooke.json"


def _find_latest_archive(results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else PROJECT_ROOT / "results" / "models"
    candidates = list(root.glob("triplet_cmamega_*/archive.json"))
    if not candidates:
        raise FileNotFoundError(f"No archive.json found under {root}/triplet_cmamega_*")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render diverse high-quality triplet designs using fixed absolute FOV angles."
    )
    p.add_argument("--archive", default=None, help="Path to archive.json (default: latest triplet_cmamega run)")
    p.add_argument(
        "--config",
        default="optical_rl/configs/qd_triplet_cmamega.yaml",
        help="Triplet YAML config path used to resolve default fixed FOV angles.",
    )
    p.add_argument(
        "--lens-template",
        default=str(_default_lens_template()),
        help="Path to cooke lens template json",
    )
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-mm", type=float, default=-1000.0)
    p.add_argument("--wavelength-um", type=float, default=0.587)
    p.add_argument("--num-rays", type=int, default=11)
    p.add_argument("--fov-degs", default=None, help="Comma-separated absolute angles in degrees, e.g. '0.0,0.5,2.0'.")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--device", default="cpu", help="Rendering device: cpu, cuda, mps, or auto")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    archive_path = Path(args.archive) if args.archive else _find_latest_archive()
    lens_template = Path(args.lens_template)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(archive_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_degs = _resolve_fov_degs(config_path=config_path, fov_degs_override=args.fov_degs)

    designs = load_archive(archive_path)
    selected = select_diverse_top(
        designs,
        count=int(args.count),
        top_frac=float(args.top_frac),
        seed=int(args.seed),
    )
    if not selected:
        print("[render-triplet-fixedfov] no valid designs found in archive", flush=True)
        return

    selected = sorted(
        selected,
        key=lambda d: (
            float(d.rms_spot) if d.rms_spot is not None and np.isfinite(d.rms_spot) else float("inf"),
            -float(d.fitness),
        ),
    )

    grid_path = out_dir / "archive_diverse_top_grid.png"
    panel_metrics = render_archive_grid(
        selected=selected,
        lens_template=lens_template,
        out_path=grid_path,
        depth_mm=float(args.depth_mm),
        wavelength_um=float(args.wavelength_um),
        num_rays=int(args.num_rays),
        fov_degs=fov_degs,
        cols=int(args.grid_cols),
        dpi=int(args.dpi),
        device=str(args.device),
    )
    print(f"[render-triplet-fixedfov] saved {grid_path.name}", flush=True)

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
        "config": str(config_path),
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
        "fov_degs_used": [float(x) for x in fov_degs],
        "grid_cols": int(args.grid_cols),
        "dpi": int(args.dpi),
        "device": str(args.device),
        "selected": selected_meta,
        "grid_image": grid_path.name,
    }
    (out_dir / "selected_designs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[render-triplet-fixedfov] outputs at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
