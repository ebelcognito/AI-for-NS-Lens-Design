import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Allow running as a script: `python optical_rl/main_qd_quad_curve_glass_v2.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.map_elites import MapElites, MapElitesConfig
from optical_rl.qd.quad_curve_glass_descriptors import (
    descriptor_from_bounding_box,
    descriptor_from_irradiance_map,
)
from optical_rl.qd.quad_curve_glass_design import (
    QuadCurveGlassDesignConfig,
    build_path_points,
    clip_design,
    corners_sags_to_genotype,
    genotype_to_corners_sags,
)
from optical_rl.utils.config import load_config
from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_ray_bundle_with_custom_glass


def _ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def _new_model_run_dir(base_dir: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"qd_quad_curve_glass_v2_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass(frozen=True)
class _EvalCfg:
    env: Dict
    qd: Dict


def _build_ray_targets(env_cfg: Dict):
    ray_mode = str(env_cfg.get("ray_mode", "explicit")).lower().strip()
    if ray_mode not in ("explicit", "bundle"):
        raise ValueError(f"Unsupported ray_mode: {ray_mode}")

    if ray_mode == "bundle":
        ray_count = int(env_cfg["ray_count"])
        if ray_count <= 0:
            raise ValueError(f"ray_count must be > 0, got {ray_count}")
        target_x = float(env_cfg["ray_bundle_target_x"])
        y_min = float(env_cfg["ray_bundle_y_min"])
        y_max = float(env_cfg["ray_bundle_y_max"])
        ys = np.linspace(y_min, y_max, num=ray_count, dtype=np.float64)
        return [(target_x, float(y)) for y in ys]

    ray_targets = [
        (float(env_cfg["ray1_target_x"]), float(env_cfg["ray1_target_y"])),
        (float(env_cfg["ray2_target_x"]), float(env_cfg["ray2_target_y"])),
    ]
    if env_cfg.get("ray3_target_x") is not None and env_cfg.get("ray3_target_y") is not None:
        ray_targets.append((float(env_cfg["ray3_target_x"]), float(env_cfg["ray3_target_y"])))
    if env_cfg.get("ray4_target_x") is not None and env_cfg.get("ray4_target_y") is not None:
        ray_targets.append((float(env_cfg["ray4_target_x"]), float(env_cfg["ray4_target_y"])))
    return ray_targets


def _build_scene(env_cfg: Dict, points, detector_center_y: float, ray_targets):
    return scene_ray_bundle_with_custom_glass(
        source_x=float(env_cfg["source_x"]),
        source_y=float(env_cfg["source_y"]),
        ray_targets=ray_targets,
        detector_x=float(env_cfg["detector_x"]),
        detector_center_y=float(detector_center_y),
        detector_half_height=float(env_cfg["detector_height"]) / 2.0,
        points=points,
        ref_index=float(env_cfg.get("ref_index", 1.5)),
    )


def _save_archive_npz(out_path: Path, me: MapElites) -> None:
    keys = list(me.archive.keys())
    if not keys:
        np.savez_compressed(out_path, idx0=np.array([]), idx1=np.array([]))
        return

    idx0 = np.array([k[0] for k in keys], dtype=np.int32)
    idx1 = np.array([k[1] for k in keys], dtype=np.int32)
    genotypes = np.stack([me.archive[k][0] for k in keys], axis=0).astype(np.float64)
    fitness = np.array([me.archive[k][1] for k in keys], dtype=np.float64)
    desc = np.array([me.archive[k][2] for k in keys], dtype=np.float64)

    np.savez_compressed(
        out_path,
        idx0=idx0,
        idx1=idx1,
        genotype=genotypes,
        fitness=fitness,
        desc=desc,
    )


def _save_heatmap_png(out_path: Path, me: MapElites) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    n0, n1 = me.config.bins
    grid = np.full((n0, n1), np.nan, dtype=np.float64)
    for (i, j), (_g, f, _d) in me.archive.items():
        grid[i, j] = float(f)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.T, origin="lower", aspect="auto")
    ax.set_title("MAP-Elites fitness heatmap (power)")
    ax.set_xlabel("desc bin 0")
    ax.set_ylabel("desc bin 1")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    config_path = os.environ.get(
        "RL_CONFIG", "optical_rl/configs/qd_quad_curve_glass_v2.yaml"
    )
    cfg = load_config(config_path)

    logging_cfg = cfg["logging"]
    _ensure_dirs(
        logging_cfg["results_dir"],
        logging_cfg["logs_dir"],
        logging_cfg["models_dir"],
        logging_cfg["tensorboard_dir"],
        logging_cfg["visualizations_dir"],
    )
    model_run_dir = _new_model_run_dir(logging_cfg["models_dir"])

    config_src = Path(config_path)
    config_dst = model_run_dir / config_src.name
    shutil.copy2(config_src, config_dst)
    print(f"[qd] Config saved to: {config_dst}", flush=True)

    env_cfg = cfg["env"].copy()
    qd_cfg = cfg["qd"].copy()
    ray_targets = _build_ray_targets(env_cfg)

    design_cfg = QuadCurveGlassDesignConfig(
        world_width=float(env_cfg["world_width"]),
        world_height=float(env_cfg["world_height"]),
        center_x=float(env_cfg["center_x"]),
        center_y=float(env_cfg["center_y"]),
        quadrant_margin=float(env_cfg["quadrant_margin"]),
        right_side_x_max=float(env_cfg["right_side_x_max"]),
        sag_min=float(env_cfg["sag_min"]),
        sag_max=float(env_cfg["sag_max"]),
    )

    descriptor_kind = str(qd_cfg.get("descriptor", "bbox")).lower().strip()
    if descriptor_kind not in ("bbox", "irradiance"):
        raise ValueError(f"Unsupported descriptor: {descriptor_kind}")

    me_cfg = MapElitesConfig(
        bins=tuple(qd_cfg.get("bins", [20, 20])),
        desc_bounds=(
            tuple(qd_cfg["desc_bounds"][0]),
            tuple(qd_cfg["desc_bounds"][1]),
        ),
        sigma=float(qd_cfg.get("sigma", 0.1)),
        seed=int(qd_cfg.get("seed", 0)),
    )

    rng = np.random.default_rng(int(qd_cfg.get("seed", 0)))

    def sample_genotype(r: np.random.Generator) -> np.ndarray:
        base = np.array(
            [
                [130.0, 130.0],
                [170.0, 130.0],
                [170.0, 170.0],
                [130.0, 170.0],
            ],
            dtype=np.float64,
        )
        # defines the starting quad (which is a square)
        corners = base + r.normal(0.0, float(qd_cfg.get("init_corner_sigma", 10.0)), size=base.shape)
        # get the standard deviation of the corner v airation distribution, Now this 

        corners = [(float(x), float(y)) for x, y in corners]
        sags = r.uniform(float(design_cfg.sag_min), float(design_cfg.sag_max), size=(4,)).tolist()
        if bool(qd_cfg.get("ignore_top_bottom_sag", True)):
            sags[0] = 0.0
            sags[2] = 0.0
        corners, sags = clip_design(design_cfg, corners, sags)
        return corners_sags_to_genotype(corners, sags)

    def mutate(r: np.random.Generator, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64).copy()
        g[:8] += r.normal(0.0, float(qd_cfg.get("mut_corner_sigma", 2.0)), size=(8,))
        g[8:] += r.normal(0.0, float(qd_cfg.get("mut_sag_sigma", 0.02)), size=(4,))
        corners, sags = genotype_to_corners_sags(g)
        if bool(qd_cfg.get("ignore_top_bottom_sag", True)):
            sags[0] = 0.0
            sags[2] = 0.0
        corners, sags = clip_design(design_cfg, corners, sags)
        return corners_sags_to_genotype(corners, sags)

    use_persistent_runner = bool(qd_cfg.get("use_persistent_runner", True))
    runner = create_runner(
        persistent=use_persistent_runner,
        timeout_s=float(qd_cfg.get("timeout_s", 5.0)),
        verbose=bool(qd_cfg.get("verbose_timing", False)),
    )
    runner.start()

    def evaluate(g: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        corners, sags = genotype_to_corners_sags(g)
        if bool(qd_cfg.get("ignore_top_bottom_sag", True)):
            sags[0] = 0.0
            sags[2] = 0.0
        corners, sags = clip_design(design_cfg, corners, sags)
        points = build_path_points(design_cfg, corners, sags)

        if bool(qd_cfg.get("randomize_detector_center_y", True)):
            detector_center_y = float(
                rng.uniform(float(env_cfg["detector_y_min"]), float(env_cfg["detector_y_max"]))
            )
        else:
            detector_center_y = float(env_cfg["detector_center_y"])

        scene = _build_scene(
            env_cfg,
            points=points,
            detector_center_y=detector_center_y,
            ray_targets=ray_targets,
        )

        try:
            if use_persistent_runner:
                result, _timing = runner.simulate(scene, episode=0, step=0)
            else:
                result, _timing = run_simulation(scene, timeout_s=float(qd_cfg.get("timeout_s", 5.0)), verbose=False)
        except Exception:
            return 0.0, (0.0, 0.0)

        detectors = result.get("detectors") or []
        fitness = float(detectors[0].get("power", 0.0)) if detectors else 0.0

        if descriptor_kind == "irradiance" and detectors:
            desc_obj = descriptor_from_irradiance_map(detectors[0])
            if desc_obj is not None:
                return fitness, (float(desc_obj.centroid), float(desc_obj.spread))

        desc = descriptor_from_bounding_box(corners)
        return fitness, desc

    me = MapElites(me_cfg, sample_genotype=sample_genotype, evaluate=evaluate, mutate=mutate)

    try:
        init_samples = int(qd_cfg.get("init_samples", 100))
        total_evals = int(qd_cfg.get("total_evals", 5000))
        me.seed_archive(init_samples)
        stats = me.run(num_evals=total_evals, seed=0)
    finally:
        runner.stop()

    print(f"[qd] {stats}", flush=True)

    archive_path = model_run_dir / "archive.npz"
    _save_archive_npz(archive_path, me)
    print(f"[qd] Archive saved to: {archive_path}", flush=True)

    heatmap_path = model_run_dir / "heatmap.png"
    _save_heatmap_png(heatmap_path, me)
    if heatmap_path.exists():
        print(f"[qd] Heatmap saved to: {heatmap_path}", flush=True)


if __name__ == "__main__":
    main()
