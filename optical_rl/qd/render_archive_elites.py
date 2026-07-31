import argparse
import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.quad_curve_glass_design import (
    QuadCurveGlassDesignConfig,
    build_path_points,
    genotype_to_corners_sags,
)
from optical_rl.utils.config import load_config
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
from optical_rl.utils.scene_builder import scene_two_rays_with_custom_glass


@dataclass(frozen=True)
class Elite:
    idx0: int
    idx1: int
    genotype: np.ndarray
    fitness: float
    desc0: float
    desc1: float


def _save_frame_from_result(result: Dict[str, Any], frame_path: Path) -> None:
    images = result.get("images") or []
    if not images:
        raise RuntimeError(
            "No images returned. Ensure node-canvas is installed and a CropBox is present."
        )
    data_url = images[0].get("dataUrl", "")
    if "," not in data_url:
        raise RuntimeError("Malformed dataUrl in runner output")
    png_data = data_url.split(",", 1)[1]
    frame_path.write_bytes(base64.b64decode(png_data))


def _scene_with_cropbox(scene: Dict[str, Any], *, world_width: float, world_height: float, cropbox_width: int) -> Dict[str, Any]:
    scene = dict(scene)
    objs = list(scene.get("objs") or [])
    objs.append(
        {
            "type": "CropBox",
            "p1": {"x": 0, "y": 0},
            "p4": {"x": float(world_width) + 50.0, "y": float(world_height) + 50.0},
            "width": int(cropbox_width),
        }
    )
    scene["objs"] = objs
    return scene


def _build_scene(env_cfg: Dict[str, Any], points, detector_center_y: float) -> Dict[str, Any]:
    return scene_two_rays_with_custom_glass(
        source_x=env_cfg["source_x"],
        source_y=env_cfg["source_y"],
        ray1_target_x=env_cfg["ray1_target_x"],
        ray1_target_y=env_cfg["ray1_target_y"],
        ray2_target_x=env_cfg["ray2_target_x"],
        ray2_target_y=env_cfg["ray2_target_y"],
        ray3_target_x=env_cfg.get("ray3_target_x"),
        ray3_target_y=env_cfg.get("ray3_target_y"),
        ray4_target_x=env_cfg.get("ray4_target_x"),
        ray4_target_y=env_cfg.get("ray4_target_y"),
        detector_x=env_cfg["detector_x"],
        detector_center_y=float(detector_center_y),
        detector_half_height=float(env_cfg["detector_height"]) / 2.0,
        points=points,
        ref_index=float(env_cfg.get("ref_index", 1.5)),
    )


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Render MAP-Elites elites from archive.npz using ray-optics CropBox.")
    p.add_argument("--archive", required=True, help="Path to archive.npz produced by the QD run.")
    p.add_argument("--config", required=True, help="Path to the QD YAML config used for the run (for env params).")
    p.add_argument("--out-dir", default=f"results/visualizations/qd_elites_{timestamp}")
    p.add_argument("--min-power", type=float, default=3.0)
    p.add_argument("--max-count", type=int, default=0, help="0 = no limit")
    p.add_argument("--cropbox-width", type=int, default=1100)
    p.add_argument("--timeout-s", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    archive_path = Path(args.archive)
    cfg_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(cfg_path))
    env_cfg = dict(cfg["env"])

    data = np.load(str(archive_path))
    for key in ("idx0", "idx1", "genotype", "fitness", "desc"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in {archive_path}")

    idx0 = data["idx0"].astype(np.int32, copy=False)
    idx1 = data["idx1"].astype(np.int32, copy=False)
    genotype = data["genotype"].astype(np.float64, copy=False)
    fitness = data["fitness"].astype(np.float64, copy=False)
    desc = data["desc"].astype(np.float64, copy=False)

    elites: List[Elite] = []
    for i in range(int(fitness.shape[0])):
        f = float(fitness[i])
        if f <= float(args.min_power):
            continue
        elites.append(
            Elite(
                idx0=int(idx0[i]),
                idx1=int(idx1[i]),
                genotype=np.asarray(genotype[i], dtype=np.float64),
                fitness=f,
                desc0=float(desc[i, 0]),
                desc1=float(desc[i, 1]),
            )
        )

    elites.sort(key=lambda e: e.fitness, reverse=True)
    if args.max_count and len(elites) > int(args.max_count):
        elites = elites[: int(args.max_count)]

    (out_dir / "selected_elites.json").write_text(
        json.dumps(
            {
                "archive": str(archive_path),
                "config": str(cfg_path),
                "min_power": float(args.min_power),
                "count": len(elites),
                "elites": [
                    {
                        "rank": r,
                        "bin": [e.idx0, e.idx1],
                        "fitness": e.fitness,
                        "desc": [e.desc0, e.desc1],
                    }
                    for r, e in enumerate(elites)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not elites:
        print(f"[render] No elites with fitness > {args.min_power}", flush=True)
        return

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

    with PersistentRayOpticsRunner(timeout_s=float(args.timeout_s), verbose=False) as runner:
        for r, elite in enumerate(elites):
            corners, sags = genotype_to_corners_sags(elite.genotype)
            points = build_path_points(design_cfg, corners, sags)

            detector_center_y = float(env_cfg.get("detector_center_y", 150.0))
            scene = _build_scene(env_cfg, points=points, detector_center_y=detector_center_y)
            scene = _scene_with_cropbox(
                scene,
                world_width=float(env_cfg["world_width"]),
                world_height=float(env_cfg["world_height"]),
                cropbox_width=int(args.cropbox_width),
            )

            result, _timing = runner.simulate(scene, episode=0, step=r)

            stem = f"elite_rank{r:02d}_bin{elite.idx0:02d}-{elite.idx1:02d}_power{elite.fitness:.3f}"
            img_path = out_dir / f"{stem}.png"
            meta_path = out_dir / f"{stem}.json"

            _save_frame_from_result(result, img_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "rank": r,
                        "bin": [elite.idx0, elite.idx1],
                        "fitness": elite.fitness,
                        "desc": [elite.desc0, elite.desc1],
                        "corners": [[float(x), float(y)] for x, y in corners],
                        "sags": [float(s) for s in sags],
                        "genotype": [float(x) for x in elite.genotype.reshape(-1).tolist()],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"[render] saved {img_path.name}", flush=True)

    print(f"[render] outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()

