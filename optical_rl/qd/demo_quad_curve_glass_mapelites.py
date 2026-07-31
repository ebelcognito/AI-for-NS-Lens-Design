"""Tiny MAP-Elites demo for the quad-curve glass problem.

This is intentionally simple and is meant for early-stage experimentation, not
production performance. It evaluates *design vectors* (glass shape parameters),
not a step-based policy.

Usage (after installing requirements and having Node available):
  python3 -m optical_rl.qd.demo_quad_curve_glass_mapelites
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from optical_rl.qd.map_elites import MapElites, MapElitesConfig
from optical_rl.qd.quad_curve_glass_descriptors import descriptor_from_bounding_box


@dataclass(frozen=True)
class _DesignCfg:
    world_width: float = 300.0
    world_height: float = 300.0
    center_x: float = 150.0
    center_y: float = 150.0
    quadrant_margin: float = 5.0
    right_side_x_max: float = 270.0
    sag_min: float = -0.35
    sag_max: float = 0.35


def _clip_corners(cfg: _DesignCfg, corners: np.ndarray) -> np.ndarray:
    m = float(cfg.quadrant_margin)
    cx, cy = float(cfg.center_x), float(cfg.center_y)
    w, h = float(cfg.world_width), float(cfg.world_height)

    # corners: [TL, TR, BR, BL] each (x,y)
    clipped = corners.copy()
    # TL
    clipped[0, 0] = np.clip(clipped[0, 0], m, cx - m)
    clipped[0, 1] = np.clip(clipped[0, 1], m, cy - m)
    # TR
    clipped[1, 0] = np.clip(clipped[1, 0], cx + m, min(w - m, float(cfg.right_side_x_max)))
    clipped[1, 1] = np.clip(clipped[1, 1], m, cy - m)
    # BR
    clipped[2, 0] = np.clip(clipped[2, 0], cx + m, min(w - m, float(cfg.right_side_x_max)))
    clipped[2, 1] = np.clip(clipped[2, 1], cy + m, h - m)
    # BL
    clipped[3, 0] = np.clip(clipped[3, 0], m, cx - m)
    clipped[3, 1] = np.clip(clipped[3, 1], cy + m, h - m)
    return clipped


def _sample_design(rng: np.random.Generator, cfg: _DesignCfg) -> np.ndarray:
    base = np.array(
        [
            [130.0, 130.0],  # TL
            [170.0, 130.0],  # TR
            [170.0, 170.0],  # BR
            [130.0, 170.0],  # BL
        ],
        dtype=np.float64,
    )
    corners = base + rng.normal(0.0, 10.0, size=base.shape)
    corners = _clip_corners(cfg, corners)
    # sags: [top, right, bottom, left] but top/bottom ignored in env_v2
    sags = rng.uniform(float(cfg.sag_min), float(cfg.sag_max), size=(4,))
    sags[0] = 0.0
    sags[2] = 0.0
    return np.concatenate([corners.reshape(-1), sags], axis=0)


def _mutate_design(rng: np.random.Generator, g: np.ndarray, cfg: _DesignCfg) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64).copy()
    g[:8] += rng.normal(0.0, 2.0, size=(8,))
    g[8:] += rng.normal(0.0, 0.02, size=(4,))

    corners = _clip_corners(cfg, g[:8].reshape(4, 2))
    sags = np.clip(g[8:], float(cfg.sag_min), float(cfg.sag_max))
    sags[0] = 0.0
    sags[2] = 0.0
    return np.concatenate([corners.reshape(-1), sags], axis=0)


def _evaluate_design(genotype: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Returns (fitness, (desc0, desc1)).

    Fitness uses detector power (higher is better).
    Descriptor uses (width, height) of the current glass polygon.
    """
    try:
        from optical_rl.envs.quad_curve_glass_env_v2 import (
            QuadCurveGlassConfigV2,
            QuadCurveGlassEnvV2,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This demo requires `gymnasium` (and Node for ray-optics). "
            "Install requirements.txt in your venv and try again."
        ) from exc

    cfg = QuadCurveGlassConfigV2(max_steps=1, use_persistent_runner=False, verbose_timing=False)
    env = QuadCurveGlassEnvV2(cfg)
    env.reset(seed=0)

    g = np.asarray(genotype, dtype=np.float64).reshape(-1)
    corners = g[:8].reshape(4, 2)
    sags = g[8:12].reshape(4)
    env.corners = [(float(x), float(y)) for x, y in corners]
    env.sags = [float(s) for s in sags]
    env.points = env._build_path_points(env.corners, env.sags)

    ok, result, _timing = env._simulate_safe()
    env.close()

    if not ok:
        return 0.0, (0.0, 0.0)

    detectors = result.get("detectors") or []
    fitness = float(detectors[0].get("power", 0.0)) if detectors else 0.0
    desc = descriptor_from_bounding_box(env.corners)
    return fitness, desc


def main() -> None:  # pragma: no cover
    cfg = _DesignCfg()

    me_cfg = MapElitesConfig(
        bins=(20, 20),
        # width/height bounds in pixels (very rough; adjust as you learn)
        desc_bounds=((10.0, 200.0), (10.0, 200.0)),
        sigma=0.1,
        seed=0,
    )

    me = MapElites(
        me_cfg,
        sample_genotype=lambda rng: _sample_design(rng, cfg),
        evaluate=_evaluate_design,
        mutate=lambda rng, g: _mutate_design(rng, g, cfg),
    )

    me.seed_archive(50)
    stats = me.run(num_evals=500, seed=0)
    print(stats)


if __name__ == "__main__":  # pragma: no cover
    main()

