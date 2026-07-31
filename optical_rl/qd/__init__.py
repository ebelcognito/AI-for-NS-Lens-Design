"""Quality-Diversity (QD) utilities (introductory).

This package is intentionally light on dependencies so it can be imported without
`gymnasium` / `stable_baselines3` installed.
"""

from .map_elites import MapElites, MapElitesConfig, MapElitesStats
from .singlet_cmamega import (
    CMAMEGAArchive2D,
    FixedSingletCMAMEGARunner,
    FixedSingletProblem,
    GradientCoefficientCMA,
    SingletCMAMEGAConfig,
    SingletEvalResult,
    project_box,
    theta_to_unit,
    unit_to_theta,
)

__all__ = [
    "CMAMEGAArchive2D",
    "FixedSingletCMAMEGARunner",
    "FixedSingletProblem",
    "GradientCoefficientCMA",
    "MapElites",
    "MapElitesConfig",
    "MapElitesStats",
    "SingletCMAMEGAConfig",
    "SingletEvalResult",
    "project_box",
    "theta_to_unit",
    "unit_to_theta",
]

