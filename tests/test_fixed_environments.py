"""Fixed environment keeps conditions out of CMA's geometry genome."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.qd.fixed_environments import (GEOMETRY_PARAMETERS,
                                          problem_from_environment_spec)


def main() -> None:
    spec = {
        "environment_id": "test", "status": "accepted",
        "intended_sequence": "T@A -> TIR@T -> TIR@B -> TIR@A -> T@B",
        "fixed": {"ray_y": -6.5, "ang": -42.0, "n": 1.67,
                  "x_sensor": 120.0, "beam_width": 8.0,
                  "focus_window_half_width": 8.0},
        "variables": {
            "cx": [35.0, 38.0], "thick": [20.0, 24.0],
            "fsag": [-0.1, 0.0], "bsag": [0.04, 0.1],
            "tilt_a": [-25.0, -22.0], "tilt_b": [-9.0, -6.0],
        },
    }
    problem = problem_from_environment_spec(spec, n_rays=31)
    assert tuple(problem.param_names) == GEOMETRY_PARAMETERS
    assert len(problem.lo) == len(problem.hi) == 6
    assert problem.intended_sequence == spec["intended_sequence"]
    x = (problem.lo + problem.hi) / 2
    p = problem.vec_to_params(x)
    assert (p["ray_y"], p["ang"], p["n"]) == (-6.5, -42.0, 1.67)
    # Sag genes in an environment spec are relative; the engine receives physical sag.
    assert abs(p["fsag"] - x[2] * problem.half_h) < 1e-12
    layout = problem.layout(p, field_deg=3.0)
    assert layout.source.center == -6.5
    assert layout.source.aim_deg == -39.0
    assert layout.source.n_rays == 31
    assert layout.element.n == 1.67
    assert problem.is_valid(p)
    print("ALL FIXED-ENVIRONMENT CHECKS PASS")


if __name__ == "__main__":
    main()
