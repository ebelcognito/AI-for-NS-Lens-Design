from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .types import Descriptor2D, EvalResult
@dataclass(frozen=True)
class ArchiveStats:
    filled: float
    total_cells: float
    coverage: float
    best_fitness: float

    class ArchiveGrid2D:
        def __init__(
            self,
            bins: Tuple[int, int] # here its 2d, because our measure space is 2d..
            bounds: Tuple[Tuple[float, float], Tuple[float, float]],
            *,
            seed:int = 0,

        )
            self.bins = (int(bins[0]), int(bins[1]))
            self.bounds = bounds
            self._rng = np.random.default_rng(int(seed))
            self._cells: Dict[Tuple[int, int], EvalResult] = {}
            self._best_fitness = -float("inf")

        def _to_index(self, measures: Descriptor2D) -> Optional[Tuple[int, int]]:
            (x_lo, x_hi), (y_lo, y_hi) = self.bounds
            x, y = float(measures[0]), float(measures[1])
            if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
                return None