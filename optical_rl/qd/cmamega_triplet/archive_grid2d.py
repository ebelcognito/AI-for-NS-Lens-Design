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
        bins: Tuple[int, int],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        *,
        seed: int = 0,
    ):
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

        def to_bin(value: float, lo: float, hi: float, n: int) -> int:
            if hi <= lo:
                return 0
            frac = (value - lo) / (hi - lo)
            return int(np.clip(np.floor(frac * n), 0, n - 1))

        return (
            to_bin(x, x_lo, x_hi, self.bins[0]),
            to_bin(y, y_lo, y_hi, self.bins[1]),
        )

    def insert(self, result: EvalResult) -> int:
        """Insert a candidate. Returns INSERT_{NONE,REPLACED,NEW} from
        optical_rl.qd._checkpoint. Truthy if any kind of improvement
        occurred, so legacy `if archive.insert(...)` call sites still work.
        """
        from optical_rl.qd._checkpoint import INSERT_NONE, INSERT_REPLACED, INSERT_NEW
        idx = self._to_index(result.measures)
        if idx is None:
            return INSERT_NONE
        current = self._cells.get(idx)
        self._best_fitness = max(self._best_fitness, float(result.fitness))
        if current is None:
            self._cells[idx] = result
            return INSERT_NEW
        if float(result.fitness) > float(current.fitness):
            self._cells[idx] = result
            return INSERT_REPLACED
        return INSERT_NONE

    def sample_parent(self, seed: int | None = None) -> np.ndarray:
        if not self._cells:
            raise RuntimeError("archive is empty")
        rng = self._rng if seed is None else np.random.default_rng(int(seed))
        keys = list(self._cells.keys())
        key = keys[int(rng.integers(0, len(keys)))]
        return self._cells[key].theta.copy()

    def stats(self) -> ArchiveStats:
        total = self.bins[0] * self.bins[1]
        filled = len(self._cells)
        return ArchiveStats(
            filled=float(filled),
            total_cells=float(total),
            coverage=float(filled / max(total, 1)),
            best_fitness=float(self._best_fitness if filled else -float("inf")),
        )

    @property
    def cells(self) -> Dict[Tuple[int, int], EvalResult]:
        return dict(self._cells)
