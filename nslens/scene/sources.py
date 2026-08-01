"""Sources.

Two families cover every scenario in the project:

  ParallelBundle  a collimated bundle: N rays offset along a transverse axis, all
                  aimed the same way. fold_demo emits one along +x from the plane
                  x = 0; prism_fold emits one straight up from the line y = -75.
                  Those are the SAME source with a different axis, which is why the
                  perpendicular "prism fold" env needed no new source code.

  PointFan        several point emitters, each fanning N rays about an aim direction.
                  Used by the AR/VR env, where the display is an extended source.

Ray brightness is 1.0 per ray so that first-order Fresnel ghosts (~4-15% of a ray) stay
above the engine's absolute truncation threshold (0.01) and are actually traced rather
than merely booked as truncated. Every reported share is normalised by the ledger's
emitted total, so the absolute scale cancels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ParallelBundle:
    """A collimated bundle of `n_rays` spanning `width`, centred at `center` on the
    transverse axis, emitted from the line where the other coordinate equals `origin`.

    axis="y": rays start at (origin, center+d)  -- the collinear fold env
    axis="x": rays start at (center+d, origin)  -- the perpendicular prism-fold env
    """

    center: float
    aim_deg: float
    width: float = 8.0
    n_rays: int = 21
    axis: str = "y"
    origin: float = 0.0
    brightness: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")

    def rays(self) -> List[dict]:
        a = np.radians(self.aim_deg)
        out: List[dict] = []
        for d in np.linspace(-self.width / 2.0, self.width / 2.0, self.n_rays):
            if self.axis == "y":
                y = self.center + d
                p1 = {"x": self.origin, "y": float(y)}
                p2 = {"x": self.origin + float(np.cos(a)), "y": float(y + np.sin(a))}
            else:
                x = self.center + d
                p1 = {"x": float(x), "y": self.origin}
                p2 = {"x": float(x + np.cos(a)), "y": float(self.origin + np.sin(a))}
            out.append({"type": "SingleRay", "p1": p1, "p2": p2,
                        "brightness": float(self.brightness)})
        return out


@dataclass(frozen=True)
class PointFan:
    """Point emitters, each fanning `n_fan` rays uniformly over +-`halfangle_deg`
    about `aim_deg`."""

    points: Sequence[Tuple[float, float]]
    aim_deg: float
    halfangle_deg: float
    n_fan: int
    brightness: float = 1.0

    def rays(self) -> List[dict]:
        out: List[dict] = []
        angles = np.radians(self.aim_deg
                            + np.linspace(-self.halfangle_deg, self.halfangle_deg,
                                          self.n_fan))
        for (sx, sy) in self.points:
            for a in angles:
                out.append({"type": "SingleRay",
                            "p1": {"x": sx, "y": sy},
                            "p2": {"x": sx + float(np.cos(a)),
                                   "y": sy + float(np.sin(a))},
                            "brightness": float(self.brightness)})
        return out
