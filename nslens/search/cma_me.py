"""
Self-contained CMA-ME (Covariance Matrix Adaptation MAP-Elites) — no pyribs/pycma.

CMA-ME = MAP-Elites filled by a pool of CMA-ES "improvement emitters" (Fontaine et al.
2020).  Each emitter is a standard (mu/mu_w, lambda) CMA-ES whose solutions are ranked by
*archive improvement* (filled a new cell > improved an existing cell > added nothing), and
which restarts from a random elite when a batch adds nothing to the archive.

This is the gradient-free QD solver the probe calls for (not RL, not DQD).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------------
# Standard CMA-ES (rank-based; tell() takes an order, not raw objective values)
# ----------------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float, *, popsize: Optional[int] = None,
                 seed: int = 0, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 boundary_handling: str = "clip"):
        if boundary_handling not in ("clip", "reflect"):
            raise ValueError("boundary_handling must be clip or reflect")
        self.boundary_handling = boundary_handling
        self.N = int(len(x0))
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.sigma0 = float(sigma0)
        self.bounds = bounds

        self.lam = int(popsize) if popsize else (4 + int(np.floor(3 * np.log(self.N))))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        N = self.N
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.C = np.eye(N)
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.invsqrtC = np.eye(N)
        self.gen = 0
        self.restart_needed = False
        self._last_X: Optional[np.ndarray] = None
        self._last_Y: Optional[np.ndarray] = None
        self._last_Z: Optional[np.ndarray] = None
        self._last_raw_X: Optional[np.ndarray] = None
        self._last_clipping_mask = np.zeros((self.lam, self.N), dtype=bool)
        self._ask_counter = 0

    def ask(self) -> np.ndarray:
        if self._last_X is not None:
            raise RuntimeError("ask() called twice without tell()")
        z = self.rng.standard_normal((self.lam, self.N))
        y = z @ (self.B * self.D).T            # y = B D z   (rows are samples)
        raw_X = self.mean[None, :] + self.sigma * y
        X = raw_X
        if self.bounds is not None:
            lo, hi = self.bounds
            if self.boundary_handling == "reflect":
                span = hi - lo
                if np.any(span <= 0) or not np.all(np.isfinite(span)):
                    raise ValueError("reflection requires finite positive bound spans")
                folded = np.remainder(X - lo, 2 * span)
                X = lo + np.minimum(folded, 2 * span - folded)
            else:
                X = np.clip(X, lo, hi)
            y = (X - self.mean[None, :]) / self.sigma   # recompute y after clipping
            clipping_mask = (raw_X < lo) | (raw_X > hi) | ~np.isfinite(raw_X)
        else:
            clipping_mask = ~np.isfinite(raw_X)
        self._last_X, self._last_Y = X, y
        self._last_Z = z
        self._last_raw_X = raw_X
        self._last_clipping_mask = clipping_mask
        self._ask_counter += 1
        return X

    def tell(self, order: np.ndarray, observations: Optional[Sequence[dict]] = None) -> None:
        """order: indices of the lambda samples from best to worst."""
        del observations
        if self._last_Y is None:
            raise RuntimeError("tell() called before ask()")
        order = np.asarray(order)
        if order.shape != (self.lam,) or not np.array_equal(
                np.sort(order), np.arange(self.lam)):
            raise ValueError(f"order must be a permutation of range({self.lam})")
        Y = self._last_Y[order]                # (lam, N), best-first
        yw = self.weights @ Y[: self.mu]       # weighted mean step
        self.mean = self.mean + self.sigma * yw

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ yw)
        psn = np.linalg.norm(self.ps)
        hsig = psn / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < (1.4 + 2 / (self.N + 1))
        hsig = 1.0 if hsig else 0.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw

        Ymu = Y[: self.mu]
        rank_mu = (Ymu * self.weights[:, None]).T @ Ymu
        rank_one = np.outer(self.pc, self.pc)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (rank_one + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)

        self.sigma *= np.exp((self.cs / self.damps) * (psn / self.chiN - 1))
        self.gen += 1
        self._update_eigen()
        self._last_X = None
        self._last_Y = None
        self._last_Z = None
        self._last_raw_X = None
        if self.sigma < 1e-11 or self.sigma > 1e7 or not np.isfinite(self.sigma):
            self.restart_needed = True

    def _update_eigen(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T   # enforce symmetry
        try:
            vals, vecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            self.restart_needed = True
            return
        if not np.all(np.isfinite(vals)) or np.min(vals) <= 0:
            self.restart_needed = True
            return
        if np.max(vals) / max(np.min(vals), 1e-30) > 1e14:
            self.restart_needed = True
        self.D = np.sqrt(vals)
        self.B = vecs
        self.invsqrtC = (vecs * (1.0 / self.D)) @ vecs.T

    @property
    def last_clipping_mask(self) -> np.ndarray:
        return self._last_clipping_mask.copy()

    def state_snapshot(self) -> Dict[str, Any]:
        """Lossless, JSON-compatible CMA state for trajectory replay/distillation."""
        return {
            "algorithm": "full_covariance_cma_es",
            "boundary_handling": self.boundary_handling,
            "seed": self.seed,
            "population_size": int(self.lam),
            "generation": int(self.gen),
            "ask_counter": int(self._ask_counter),
            "mean": self.mean.tolist(),
            "sigma": float(self.sigma),
            "sigma0": float(self.sigma0),
            "covariance": self.C.tolist(),
            "evolution_path_c": self.pc.tolist(),
            "evolution_path_sigma": self.ps.tolist(),
            "eigenvectors": self.B.tolist(),
            "axis_scales": self.D.tolist(),
            "inverse_sqrt_covariance": self.invsqrtC.tolist(),
            "restart_needed": bool(self.restart_needed),
            "rng_state": _json_value(copy.deepcopy(self.rng.bit_generator.state)),
        }

    @classmethod
    def from_state_snapshot(
        cls,
        snapshot: Dict[str, Any],
        *,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "CMAES":
        """Restore a snapshot so the next ``ask`` exactly replays the RNG stream."""
        required = (
            "mean", "sigma", "sigma0", "covariance", "evolution_path_c",
            "evolution_path_sigma", "eigenvectors", "axis_scales",
            "inverse_sqrt_covariance", "rng_state", "population_size",
        )
        missing = [key for key in required if key not in snapshot]
        if missing:
            raise ValueError(f"CMA snapshot missing fields: {missing}")
        restored = cls(
            np.asarray(snapshot["mean"], dtype=np.float64),
            float(snapshot["sigma0"]),
            popsize=int(snapshot["population_size"]),
            seed=int(snapshot.get("seed", 0)),
            bounds=bounds,
            boundary_handling=snapshot.get("boundary_handling", "clip"),
        )
        restored.mean = np.asarray(snapshot["mean"], dtype=np.float64)
        restored.sigma = float(snapshot["sigma"])
        restored.C = np.asarray(snapshot["covariance"], dtype=np.float64)
        restored.pc = np.asarray(snapshot["evolution_path_c"], dtype=np.float64)
        restored.ps = np.asarray(snapshot["evolution_path_sigma"], dtype=np.float64)
        restored.B = np.asarray(snapshot["eigenvectors"], dtype=np.float64)
        restored.D = np.asarray(snapshot["axis_scales"], dtype=np.float64)
        restored.invsqrtC = np.asarray(
            snapshot["inverse_sqrt_covariance"], dtype=np.float64)
        restored.gen = int(snapshot.get("generation", 0))
        restored._ask_counter = int(snapshot.get("ask_counter", 0))
        restored.restart_needed = bool(snapshot.get("restart_needed", False))
        restored.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        return restored

    def sampling_snapshot(self) -> Dict[str, Any]:
        """Return the most recent ask's exact samples and bound intervention."""
        if self._last_X is None or self._last_Y is None or self._last_Z is None:
            raise RuntimeError("sampling_snapshot() requires a preceding ask()")
        return {
            "sample_id": f"seed{self.seed}:ask{self._ask_counter - 1}",
            "standard_normal": self._last_Z.tolist(),
            "transformed_steps": self._last_Y.tolist(),
            "raw_population": self._last_raw_X.tolist(),
            "population": self._last_X.tolist(),
            "clipping_mask": self._last_clipping_mask.astype(int).tolist(),
        }


def _json_value(value: Any) -> Any:
    """Recursively convert NumPy RNG state to lossless JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


# ----------------------------------------------------------------------------------
# Grid archive (2D)
# ----------------------------------------------------------------------------------
@dataclass
class Cell:
    genotype: np.ndarray
    fitness: float
    bc: Tuple[float, float]
    info: dict = field(default_factory=dict)


class GridArchive:
    def __init__(self, bins: Tuple[int, int],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        self.bins = bins
        self.bounds = bounds
        self.cells: Dict[Tuple[int, int], Cell] = {}

    def index(self, bc: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        (lo0, hi0), (lo1, hi1) = self.bounds
        d0, d1 = float(bc[0]), float(bc[1])
        if not (lo0 <= d0 <= hi0 and lo1 <= d1 <= hi1):
            # clamp into range rather than discard (keeps edge designs)
            d0 = min(max(d0, lo0), hi0)
            d1 = min(max(d1, lo1), hi1)
        i0 = min(int((d0 - lo0) / (hi0 - lo0) * self.bins[0]), self.bins[0] - 1)
        i1 = min(int((d1 - lo1) / (hi1 - lo1) * self.bins[1]), self.bins[1] - 1)
        return (max(i0, 0), max(i1, 0))

    def add(self, genotype, fitness, bc, info=None):
        """Returns ('new'|'improved'|'none', old_fitness)."""
        if bc is None:
            return ("none", None)
        idx = self.index(bc)
        if idx is None:
            return ("none", None)
        cur = self.cells.get(idx)
        if cur is None:
            self.cells[idx] = Cell(np.asarray(genotype).copy(), float(fitness), tuple(bc), info or {})
            return ("new", None)
        if fitness > cur.fitness:
            old = cur.fitness
            self.cells[idx] = Cell(np.asarray(genotype).copy(), float(fitness), tuple(bc), info or {})
            return ("improved", old)
        return ("none", cur.fitness)

    def random_elite(self, rng) -> Optional[np.ndarray]:
        if not self.cells:
            return None
        keys = list(self.cells.keys())
        return self.cells[keys[rng.integers(0, len(keys))]].genotype

    @property
    def coverage(self) -> float:
        return len(self.cells) / (self.bins[0] * self.bins[1])

    @property
    def best_fitness(self) -> float:
        return max((c.fitness for c in self.cells.values()), default=float("-inf"))


# ----------------------------------------------------------------------------------
# CMA-ME driver
# ----------------------------------------------------------------------------------
class CMA_ME:
    def __init__(self, archive: GridArchive, evaluate: Callable, *,
                 sample_genotype: Callable, bounds, n_emitters: int = 5,
                 sigma0: float = 0.15, popsize: int = 10, seed: int = 0,
                 param_scaling: str = "isotropic"):
        self.archive = archive
        self.evaluate = evaluate
        self.sample_genotype = sample_genotype
        self.bounds = bounds            # (lo, hi) arrays, world units
        self.lo, self.hi = bounds
        self.scale = (self.hi - self.lo)
        self.sigma0 = sigma0
        self.popsize = popsize
        self.rng = np.random.default_rng(seed)
        self.n_emitters = n_emitters
        if param_scaling not in ("isotropic", "normalized"):
            raise ValueError(f"param_scaling must be 'isotropic' or 'normalized', got {param_scaling!r}")
        self.param_scaling = param_scaling
        # "isotropic" (legacy): one absolute step, sigma0 * mean(range), shared by every
        # dimension. Dimensions whose own range is far below the mean are then sampled
        # far outside their bounds and land on the clip -- e.g. for the 2-element probe
        # the step is 360% of nidx's range but 3.4% of cx's.
        # "normalized": the emitter searches the unit cube, so sigma0 means "this fraction
        # of each parameter's own range" in every dimension.
        self._sigma_abs = self.sigma0 * float(np.mean(self.scale))
        self._unit_lo = np.zeros_like(self.lo)
        self._unit_hi = np.ones_like(self.hi)
        self.emitters: List[CMAES] = [self._new_emitter(seed + i) for i in range(n_emitters)]
        self.n_evals = 0
        self.history: List[dict] = []

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.lo) / self.scale

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u, dtype=np.float64) * self.scale

    def _new_emitter(self, seed: int) -> CMAES:
        x0 = self.archive.random_elite(self.rng)
        if x0 is None:
            x0 = self.sample_genotype(self.rng)
        x0 = np.asarray(x0, dtype=np.float64)
        if self.param_scaling == "normalized":
            return CMAES(self._to_unit(x0), self.sigma0, popsize=self.popsize,
                         seed=seed, bounds=(self._unit_lo, self._unit_hi))
        return CMAES(x0, self._sigma_abs, popsize=self.popsize,
                     seed=seed, bounds=(self.lo, self.hi))

    def _step_emitter(self, emitter: CMAES) -> int:
        X = emitter.ask()
        # The emitter searches the unit cube under "normalized"; the evaluator and the
        # archive always speak world units.
        Xw = self._from_unit(X) if self.param_scaling == "normalized" else X
        keys = np.empty(len(X))
        added_any = False
        for k, x in enumerate(Xw):
            ev = self.evaluate(x)
            self.n_evals += 1
            status, old = self.archive.add(
                x, ev.fitness, ev.bc,
                info=dict(flux=ev.flux, spot_rms=ev.spot_rms, folded_frac=ev.folded_frac,
                          mean_reversals=ev.mean_reversals, device_depth=ev.device_depth,
                          opl_ratio=ev.opl_ratio, detector_power=ev.detector_power,
                          n_forward=ev.n_forward),
            )
            if status == "new":
                keys[k] = 2e12 + ev.fitness
                added_any = True
            elif status == "improved":
                keys[k] = 1e12 + (ev.fitness - (old if old is not None else 0.0))
                added_any = True
            else:
                keys[k] = ev.fitness   # not added: rank by raw fitness (lowest tier)
        order = np.argsort(-keys)
        emitter.tell(order)
        return added_any

    def run(self, max_evals: int, *, log_every: int = 1000, logger=None) -> None:
        gen = 0
        while self.n_evals < max_evals:
            for e_idx in range(self.n_emitters):
                if self.n_evals >= max_evals:
                    break
                added = self._step_emitter(self.emitters[e_idx])
                if (not added) or self.emitters[e_idx].restart_needed:
                    self.emitters[e_idx] = self._new_emitter(int(self.rng.integers(0, 1 << 30)))
            gen += 1
            if logger is not None and (self.n_evals // log_every) > (len(self.history)):
                rec = dict(evals=self.n_evals, coverage=self.archive.coverage,
                           filled=len(self.archive.cells), best=self.archive.best_fitness)
                self.history.append(rec)
                logger(rec)
