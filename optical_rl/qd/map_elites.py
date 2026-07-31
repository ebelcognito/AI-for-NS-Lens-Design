from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np


Descriptor2D = Tuple[float, float]
Index2D = Tuple[int, int]


@dataclass(frozen=True)
class MapElitesConfig:
    bins: Tuple[int, int]
    desc_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    sigma: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class MapElitesStats:
    num_evals: int
    filled: int
    total_cells: int
    coverage: float
    best_fitness: float


@dataclass
class _Cell:
    genotype: np.ndarray
    fitness: float
    desc: Descriptor2D


def _as_float_tuple_2(x: Iterable[float]) -> Descriptor2D:
    a, b = x
    return float(a), float(b)


class MapElites:
    """Minimal 2D MAP-Elites.

    The algorithm itself is generic: you provide
      - a genotype sampler
      - an evaluator returning (fitness, (d0, d1))
      - a mutation operator (optional; default is Gaussian noise)
    """

    def __init__(
        self,
        config: MapElitesConfig,
        *,
        sample_genotype: Callable[[np.random.Generator], np.ndarray],
        evaluate: Callable[[np.ndarray], Tuple[float, Descriptor2D]],
        mutate: Optional[
            Callable[[np.random.Generator, np.ndarray], np.ndarray]
        ] = None,
    ):
        self.config = config
        self._rng = np.random.default_rng(int(config.seed))
        self._sample_genotype = sample_genotype
        self._evaluate = evaluate
        self._mutate = mutate or self._default_mutate

        self._archive: Dict[Index2D, _Cell] = {}
        self._num_evals = 0
        self._best_fitness = -float("inf")

    @property
    def archive(self) -> Dict[Index2D, Tuple[np.ndarray, float, Descriptor2D]]:
        return {
            k: (v.genotype.copy(), float(v.fitness), (float(v.desc[0]), float(v.desc[1])))
            for k, v in self._archive.items()
        }

    def stats(self) -> MapElitesStats:
        total = int(self.config.bins[0]) * int(self.config.bins[1])
        filled = len(self._archive)
        coverage = float(filled / max(total, 1))
        best = float(self._best_fitness) if self._num_evals > 0 else -float("inf")
        return MapElitesStats(
            num_evals=int(self._num_evals),
            filled=int(filled),
            total_cells=int(total),
            coverage=coverage,
            best_fitness=best,
        )

    def _default_mutate(self, rng: np.random.Generator, genotype: np.ndarray) -> np.ndarray:
        g = np.asarray(genotype, dtype=np.float64).copy()
        g += rng.normal(0.0, float(self.config.sigma), size=g.shape)
        return g.astype(np.float64, copy=False)

    def _desc_to_index(self, desc: Descriptor2D) -> Optional[Index2D]:
        (min0, max0), (min1, max1) = self.config.desc_bounds
        d0, d1 = _as_float_tuple_2(desc)

        if not (min0 <= d0 <= max0 and min1 <= d1 <= max1):
            return None

        n0, n1 = int(self.config.bins[0]), int(self.config.bins[1])
        if n0 <= 0 or n1 <= 0:
            raise ValueError(f"Invalid bins: {self.config.bins}")

        def to_bin(d: float, lo: float, hi: float, n: int) -> int:
            if hi == lo:
                return 0
            t = (d - lo) / (hi - lo)
            idx = int(np.floor(t * n))
            return int(np.clip(idx, 0, n - 1))

        return to_bin(d0, min0, max0, n0), to_bin(d1, min1, max1, n1)

    def _insert(self, genotype: np.ndarray, fitness: float, desc: Descriptor2D) -> bool:
        idx = self._desc_to_index(desc)
        if idx is None:
            return False

        existing = self._archive.get(idx)
        if existing is None or float(fitness) > float(existing.fitness):
            self._archive[idx] = _Cell(
                genotype=np.asarray(genotype, dtype=np.float64).copy(),
                fitness=float(fitness),
                desc=_as_float_tuple_2(desc),
            )
            return True
        return False

    def ask(self) -> np.ndarray:
        """Propose a new candidate genotype."""
        if not self._archive:
            return np.asarray(self._sample_genotype(self._rng), dtype=np.float64)
        keys = list(self._archive.keys())
        idx = keys[int(self._rng.integers(0, len(keys)))]
        parent = self._archive[idx].genotype
        return np.asarray(self._mutate(self._rng, parent), dtype=np.float64)

    def tell(self, genotype: np.ndarray, fitness: float, desc: Descriptor2D) -> bool:
        """Record a candidate evaluation. Returns True if it improved a cell."""
        self._num_evals += 1
        self._best_fitness = max(self._best_fitness, float(fitness))
        return self._insert(genotype, fitness, desc)

    def seed_archive(self, n: int) -> MapElitesStats:
        for _ in range(int(n)):
            g = np.asarray(self._sample_genotype(self._rng), dtype=np.float64)
            f, d = self._evaluate(g)
            self.tell(g, float(f), _as_float_tuple_2(d))
        return self.stats()

    def run(self, *, num_evals: int, seed: int = 0) -> MapElitesStats:
        """Convenience runner: seed then mutate until `num_evals` total evals."""
        if self._num_evals == 0 and seed > 0:
            self.seed_archive(int(seed))

        target = int(num_evals)
        while self._num_evals < target:
            g = self.ask()
            f, d = self._evaluate(g)
            self.tell(g, float(f), _as_float_tuple_2(d))
        return self.stats()
