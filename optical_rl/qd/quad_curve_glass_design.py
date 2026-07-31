from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class QuadCurveGlassDesignConfig:
    world_width: float = 300.0
    world_height: float = 300.0
    center_x: float = 150.0
    center_y: float = 150.0
    quadrant_margin: float = 5.0
    right_side_x_max: float = 270.0
    sag_min: float = -0.35
    sag_max: float = 0.35


def clip_corner_to_quadrant(cfg: QuadCurveGlassDesignConfig, corner_idx: int, x: float, y: float) -> Point:
    m = float(cfg.quadrant_margin)
    cx = float(cfg.center_x)
    cy = float(cfg.center_y)
    w = float(cfg.world_width)
    h = float(cfg.world_height)
    if corner_idx == 0:  # TL
        xmin, xmax, ymin, ymax = m, cx - m, m, cy - m
    elif corner_idx == 1:  # TR
        xmin, xmax, ymin, ymax = cx + m, min(w - m, float(cfg.right_side_x_max)), m, cy - m
    elif corner_idx == 2:  # BR
        xmin, xmax, ymin, ymax = cx + m, min(w - m, float(cfg.right_side_x_max)), cy + m, h - m
    elif corner_idx == 3:  # BL
        xmin, xmax, ymin, ymax = m, cx - m, cy + m, h - m
    else:
        raise ValueError(f"Invalid corner index: {corner_idx}")
    return float(np.clip(x, xmin, xmax)), float(np.clip(y, ymin, ymax))


def genotype_to_corners_sags(genotype: Sequence[float]) -> Tuple[List[Point], List[float]]:
    g = np.asarray(genotype, dtype=np.float64).reshape(-1)
    if g.shape[0] != 12:
        raise ValueError(f"Expected genotype length 12, got {g.shape[0]}")
    corners = [(float(g[0]), float(g[1])), (float(g[2]), float(g[3])), (float(g[4]), float(g[5])), (float(g[6]), float(g[7]))]
    sags = [float(s) for s in g[8:12].tolist()]
    return corners, sags


def corners_sags_to_genotype(corners: Sequence[Point], sags: Sequence[float]) -> np.ndarray:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")
    if len(sags) != 4:
        raise ValueError(f"Expected 4 sags, got {len(sags)}")
    g = np.zeros((12,), dtype=np.float64)
    for i, (x, y) in enumerate(corners):
        g[2 * i] = float(x)
        g[2 * i + 1] = float(y)
    g[8:12] = np.asarray(sags, dtype=np.float64)
    return g


def clip_design(cfg: QuadCurveGlassDesignConfig, corners: Sequence[Point], sags: Sequence[float]) -> Tuple[List[Point], List[float]]:
    if len(corners) != 4 or len(sags) != 4:
        raise ValueError("Expected 4 corners and 4 sags")
    clipped_corners: List[Point] = []
    for i, (x, y) in enumerate(corners):
        clipped_corners.append(clip_corner_to_quadrant(cfg, i, float(x), float(y)))
    clipped_sags = [float(np.clip(float(s), float(cfg.sag_min), float(cfg.sag_max))) for s in sags]
    return clipped_corners, clipped_sags


def build_path_points(cfg: QuadCurveGlassDesignConfig, corners: Sequence[Point], sags: Sequence[float]) -> List[Point]:
    """Build the 8-point glass path (4 corners + 4 arc control points).

    Mirrors the logic used in `QuadCurveGlassEnvV2._build_path_points` but does not depend on gymnasium.
    """
    if len(corners) != 4 or len(sags) != 4:
        raise ValueError("Expected 4 corners and 4 sags")

    center = np.array([cfg.center_x, cfg.center_y], dtype=np.float64)
    path: List[Point] = []

    for edge_idx in range(4):
        p0 = np.array(corners[edge_idx], dtype=np.float64)
        p1 = np.array(corners[(edge_idx + 1) % 4], dtype=np.float64)
        edge = p1 - p0
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1e-6:
            edge_len = 1e-6

        midpoint = (p0 + p1) / 2.0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        normal_len = float(np.linalg.norm(normal))
        if normal_len < 1e-6:
            normal = np.array([0.0, 0.0], dtype=np.float64)
        else:
            normal /= normal_len

        to_center = center - midpoint
        if float(np.dot(normal, to_center)) < 0:
            normal = -normal

        sag = float(np.clip(float(sags[edge_idx]), float(cfg.sag_min), float(cfg.sag_max)))
        ctrl = midpoint + normal * (sag * edge_len)

        ctrl_x = float(np.clip(ctrl[0], 0.0, float(cfg.world_width)))
        ctrl_y = float(np.clip(ctrl[1], 0.0, float(cfg.world_height)))

        path.append((float(p0[0]), float(p0[1])))
        path.append((ctrl_x, ctrl_y))

    return path

