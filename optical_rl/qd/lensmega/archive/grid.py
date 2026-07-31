from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from optical_rl.qd._checkpoint import INSERT_NEW, INSERT_NONE, INSERT_REPLACED


CellIndex = Tuple[int, ...]


class GridArchive:
    """N-dim MAP-Elites grid archive.

    Matches the ``insert``/``sample_parent``/``stats``/``cells`` API used by
    the existing CMA-MEGA runner, but generalizes from hardcoded 2D to
    arbitrary behavior dimensionality ``dim``.  Cells are keyed by a length-
    ``dim`` tuple of ints; ``insert`` returns the same INSERT_NONE /
    INSERT_REPLACED / INSERT_NEW sentinels the old archive used so runner
    code ports unchanged.
    """

    def __init__(
        self,
        dim: int,
        bins: Sequence[int],
        bounds: Optional[np.ndarray] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        assert len(bins) == self.dim, "bins length must match descriptor dim"
        self.bins: Tuple[int, ...] = tuple(int(b) for b in bins)
        self._bounds: Optional[np.ndarray] = None
        if bounds is not None:
            self.set_bounds(bounds)
        self._rng = np.random.default_rng(int(seed))
        self._cells: Dict[CellIndex, Any] = {}
        self._best_fitness: float = -float("inf")

    @property
    def bounds(self) -> Optional[np.ndarray]:
        return self._bounds

    def set_bounds(self, bounds: np.ndarray) -> None:
        arr = np.asarray(bounds, dtype=np.float64)
        assert arr.shape == (self.dim, 2), (
            f"bounds must have shape ({self.dim}, 2), got {arr.shape}"
        )
        assert np.all(arr[:, 0] < arr[:, 1]), "every axis must have lo < hi"
        self._bounds = arr

    def _to_index(self, measures: np.ndarray) -> Optional[CellIndex]:
        if self._bounds is None:
            raise RuntimeError("GridArchive.bounds is None; set_bounds() first")
        m = np.asarray(measures, dtype=np.float64).reshape(-1)
        assert m.shape[0] == self.dim, (
            f"measures length {m.shape[0]} does not match archive dim {self.dim}"
        )
        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        if np.any(m < lo) or np.any(m > hi):
            return None
        idx = np.floor((m - lo) / (hi - lo) * np.asarray(self.bins)).astype(np.int64)
        idx = np.clip(idx, 0, np.asarray(self.bins) - 1)
        return tuple(int(v) for v in idx)

    def insert(self, result: Any) -> int:
        """Insert a candidate result; returns an INSERT_* sentinel.

        ``result`` must be an object with ``.theta``, ``.fitness``, and
        ``.measures`` attributes -- intentionally duck-typed so this module
        does not import the concrete EvalResult class.
        """
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

    def sample_parent(self) -> np.ndarray:
        if not self._cells:
            raise RuntimeError("archive is empty")
        keys = list(self._cells.keys())
        key = keys[int(self._rng.integers(0, len(keys)))]
        return np.asarray(self._cells[key].theta, dtype=np.float64).copy()

    def stats(self) -> Dict[str, float]:
        total = 1
        for b in self.bins:
            total *= b
        filled = len(self._cells)
        return {
            "filled": float(filled),
            "total_cells": float(total),
            "coverage": float(filled / max(total, 1)),
            "best_fitness": float(
                self._best_fitness if filled else -float("inf")
            ),
        }

    @property
    def cells(self) -> Dict[CellIndex, Any]:
        return dict(self._cells)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "bins": self.bins,
            "bounds": self._bounds,
            "cells": self._cells,
            "best_fitness": self._best_fitness,
            "rng_state": self._rng.bit_generator.state,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.dim = state["dim"]
        self.bins = state["bins"]
        self._bounds = state["bounds"]
        self._cells = state["cells"]
        self._best_fitness = state["best_fitness"]
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng_state"]
