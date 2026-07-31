from __future__ import annotations

from typing import Any, Dict, Tuple, cast

from .problem_doublet_cooke import Bounds6D, DoubletCookeConfig, FixedCookeDoubletProblem


def build_doublet_problem_from_qd_cfg(
    qd_cfg: Dict[str, Any]
) -> Tuple[FixedCookeDoubletProblem, Tuple[Tuple[float, float], ...], Tuple[int, int]]:
    bounds = tuple(tuple(float(x) for x in pair) for pair in qd_cfg["theta_bounds"])
    active = tuple(int(x) for x in qd_cfg.get("active_descriptor_indices", [0, 1]))

    problem_cfg = DoubletCookeConfig(
        lens_path=str(
            qd_cfg.get("lens_path", "/home/hisham26/Desktop/RL/optical_rl/qd/doublet_starter.json")
        ),
        object_depth_mm=float(qd_cfg.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd_cfg.get("wavelength_um", 0.587)),
        theta_bounds=cast(Bounds6D, bounds),
        invalid_tol=float(qd_cfg.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd_cfg.get("invalid_penalty_weight", 5.0)),
        num_rays=int(qd_cfg.get("num_rays", 256)),
        alpha_deg=float(qd_cfg.get("alpha_deg", 0.5)),
        stage2_field_deg=float(qd_cfg.get("stage2_field_deg", 2.0)),
        device=str(qd_cfg.get("device", "auto")),
        active_descriptor_indices=active,
    )
    return FixedCookeDoubletProblem(problem_cfg), bounds, active
