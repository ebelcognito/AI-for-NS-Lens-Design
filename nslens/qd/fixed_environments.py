"""Feasibility-first fold environments with fixed source and material settings.

The historical 9-gene problem lets CMA change the source position, source angle, and
glass index together with the optic.  That makes an archive look diverse while every
"environment" is actually the same loose global box.  A fixed environment instead owns
those three physical conditions and exposes only the six geometry genes to search.

Environment specs are generated and empirically screened by
``nslens.analysis.environment_pilot`` before they are admitted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .problems import FoldProblem

GEOMETRY_PARAMETERS = ("cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b")


@dataclass
class FixedFoldProblem(FoldProblem):
    """Six-DOF fold problem; source position/angle and refractive index are constants."""

    param_names: Sequence[str] = field(default_factory=lambda: list(GEOMETRY_PARAMETERS))
    lo: np.ndarray = field(default_factory=lambda: np.array(
        [35.0, 6.0, -0.5, -0.5, -25.0, -25.0]))
    hi: np.ndarray = field(default_factory=lambda: np.array(
        [85.0, 45.0, 0.5, 0.5, 25.0, 25.0]))
    fixed_ray_y: float = 0.0
    fixed_ang: float = 0.0
    fixed_n: float = 1.75
    intended_sequence: str = ""
    source_clearance: float = 2.0
    relative_sag: bool = True
    max_sag_ratio: float | None = 0.268
    det_clearance: float | None = 12.0

    def vec_to_params(self, x: Sequence[float]) -> dict[str, float]:
        if len(x) != len(self.param_names):
            raise ValueError(f"expected {len(self.param_names)} geometry values, got {len(x)}")
        p = {
            "ray_y": float(self.fixed_ray_y),
            "ang": float(self.fixed_ang),
            "n": float(self.fixed_n),
            **dict(zip(self.param_names, [float(value) for value in x])),
        }
        if self.relative_sag:
            chord = {"half_h": self.half_h, "half_thick": p["thick"] / 2.0}
            for key, which in self._SAG_CHORD.items():
                if key in p:
                    p[key] *= chord[which]
        return p

    def is_valid(self, p: Mapping[str, float]) -> bool:
        """Apply the hardened body, detector, and source-line clearance gates."""
        if not super().is_valid(dict(p)):
            return False
        near_source_x = min(float(vertex["x"]) for vertex in self.element(dict(p)).path())
        return near_source_x > float(self.source_clearance)


def problem_from_environment_spec(spec: Mapping[str, Any], *, n_rays: int = 17
                                  ) -> FixedFoldProblem:
    """Build a searchable problem from one accepted environment-pilot spec."""
    if spec.get("status") != "accepted":
        raise ValueError(f"environment {spec.get('environment_id')!r} was not accepted")
    fixed = spec["fixed"]
    variables = spec["variables"]
    missing = [name for name in GEOMETRY_PARAMETERS if name not in variables]
    if missing:
        raise ValueError(f"environment spec lacks geometry bounds for {missing}")
    lo = np.array([float(variables[name][0]) for name in GEOMETRY_PARAMETERS])
    hi = np.array([float(variables[name][1]) for name in GEOMETRY_PARAMETERS])
    if np.any(lo >= hi):
        raise ValueError("every geometry parameter needs a non-empty lower/upper interval")
    return FixedFoldProblem(
        lo=lo, hi=hi, n_rays=int(n_rays),
        fixed_ray_y=float(fixed["ray_y"]), fixed_ang=float(fixed["ang"]),
        fixed_n=float(fixed["n"]), x_sensor=float(fixed.get("x_sensor", 120.0)),
        intended_sequence=str(spec.get("intended_sequence", "")),
        source_clearance=float(fixed.get("source_clearance", 2.0)),
        beam_w=float(fixed.get("beam_width", 8.0)),
        focus_w=float(fixed.get("focus_window_half_width", 8.0)),
    )


__all__ = ["GEOMETRY_PARAMETERS", "FixedFoldProblem", "problem_from_environment_spec"]
