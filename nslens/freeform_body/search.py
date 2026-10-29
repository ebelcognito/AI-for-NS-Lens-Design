"""Search driver for Stage-1 freeform discovery.

Defining this driver does not run an experiment.  It connects the normalized
multiresolution codec, route-uniform archive, and the repository's existing CMA-ES
emitter.  Callers choose an evaluation budget explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nslens.search import CMAES

from .archive import DiverseDiscoveryArchive
from .problem import DiscoveryProblem


@dataclass(frozen=True)
class SearchProgress:
    evaluations: int
    invalid: int
    joint_niches: int
    route_families: int
    shape_families: int
    best_fitness: float


class FreeformDiscoverySearch:
    """CMA improvement emitters over normalized [-1, 1] freeform genes."""

    def __init__(self, problem: DiscoveryProblem, *,
                 archive: DiverseDiscoveryArchive | None = None,
                 emitters: int = 8, popsize: int = 10, sigma0: float = 0.15,
                 seed: int = 0):
        if not isinstance(emitters, (int, np.integer)) or not isinstance(popsize, (int, np.integer)):
            raise ValueError("emitters and popsize must be integers")
        if emitters < 1 or popsize < 2 or sigma0 <= 0:
            raise ValueError("invalid emitter configuration")
        self.problem = problem
        self.archive = archive or DiverseDiscoveryArchive()
        self.emitter_count = emitters
        self.popsize = popsize
        self.sigma0 = sigma0
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.invalid = 0
        self._emitters = [self._new_emitter(seed + i) for i in range(emitters)]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        size = self.problem.codec.gene_size
        return -np.ones(size), np.ones(size)

    def _new_emitter(self, seed: int) -> CMAES:
        start = self.archive.random_elite(self.rng)
        if start is None:
            start = self.rng.uniform(-1.0, 1.0, self.problem.codec.gene_size)
        return CMAES(start, self.sigma0, popsize=self.popsize,
                     seed=seed, bounds=self.bounds)

    def step(self, *, max_new_evaluations: int | None = None) -> SearchProgress:
        """Advance the emitters without exceeding an optional call allowance.

        A partial final population is admitted to the archive but is not supplied to
        CMA-ES, since fabricating ranks for unevaluated samples would corrupt its
        update.  A later call simply asks that emitter for a fresh population.
        """
        if max_new_evaluations is not None:
            if (not isinstance(max_new_evaluations, (int, np.integer))
                    or max_new_evaluations < 1):
                raise ValueError("max_new_evaluations must be a positive integer")
        remaining = max_new_evaluations
        for index, emitter in enumerate(self._emitters):
            if remaining == 0:
                break
            candidates = emitter.ask()
            evaluated_count = (len(candidates) if remaining is None
                               else min(len(candidates), remaining))
            ranks = np.empty(evaluated_count)
            added = False
            for k, genes in enumerate(candidates[:evaluated_count]):
                evaluation = self.problem.evaluate(genes)
                self.evaluations += 1
                if not evaluation.feasible:
                    self.invalid += 1
                    # Infeasible candidates stay strictly below every feasible one, but
                    # keep their relative order.  With a flat objective search_score is
                    # constant here and this reduces to the old single penalty; with a
                    # feasibility ladder it is the only gradient out of the dead region.
                    ranks[k] = -1e9 + float(evaluation.search_score)
                    continue
                status, old = self.archive.add(genes, evaluation)
                if status == "new":
                    ranks[k] = 2e12 + evaluation.fitness
                    added = True
                elif status == "improved":
                    ranks[k] = 1e12 + evaluation.fitness - float(old)
                    added = True
                else:
                    ranks[k] = evaluation.fitness
            if remaining is not None:
                remaining -= evaluated_count
            if evaluated_count == len(candidates):
                emitter.tell(np.argsort(-ranks))
                if not added or emitter.restart_needed:
                    self._emitters[index] = self._new_emitter(
                        int(self.rng.integers(0, 1 << 30)))
        summary = self.archive.summary()
        return SearchProgress(
            evaluations=self.evaluations,
            invalid=self.invalid,
            joint_niches=int(summary["joint_niches"]),
            route_families=int(summary["route_families"]),
            shape_families=int(summary["shape_families"]),
            best_fitness=float(summary["best_fitness"]),
        )

    def run(self, max_evaluations: int,
            callback: Callable[[SearchProgress], None] | None = None) -> SearchProgress:
        if (not isinstance(max_evaluations, (int, np.integer))
                or max_evaluations <= 0):
            raise ValueError("max_evaluations must be a positive integer")
        progress = self.progress()
        while self.evaluations < max_evaluations:
            progress = self.step(max_new_evaluations=max_evaluations - self.evaluations)
            if callback is not None:
                callback(progress)
        return progress

    def progress(self) -> SearchProgress:
        summary = self.archive.summary()
        return SearchProgress(
            evaluations=self.evaluations,
            invalid=self.invalid,
            joint_niches=int(summary["joint_niches"]),
            route_families=int(summary["route_families"]),
            shape_families=int(summary["shape_families"]),
            best_fitness=float(summary["best_fitness"]),
        )
