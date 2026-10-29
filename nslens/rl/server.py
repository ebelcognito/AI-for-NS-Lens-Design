"""The reward service: a long-lived pool of engine workers behind a batch call.

Three things here are not incidental, and getting any of them wrong throws away the result
that the whole plan rests on -- that the verifier is free relative to LLM generation
(doc 18 §0.1: 291 nine-field designs/s at 20 workers, against 30-60 s of generation per
GRPO step).

  * **The pool is long-lived.** `PersistentRayOpticsRunner` exists to avoid paying Node
    startup per evaluation; spawning a pool per training step pays it per step instead and
    gives back the near-linear scaling. Build one `RewardPool` and keep it.

  * **The cache stores VERDICTS, not rewards.** A verdict is what the engine measured and
    depends only on (env, genotype); a reward additionally depends on the spec that asked
    for it. Caching the verdict makes the same design free across every spec that mentions
    it, which matters because rollouts repeat heavily once the policy sharpens.

  * **A timeout is a reward, not an error.** A design that hangs the tracer must score
    quickly and let the step continue. It comes back as `engine_error`, which is the same
    stage as unbuildable geometry, and the step is unblocked.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import env_config, reward as reward_mod
from .reward import Reward, Spec, Verdict

#: genotype rounding for the cache key. Six decimals is what `driver._trace` already
#: rounds to when it writes trace.jsonl, so a cached verdict and a traced one agree.
CACHE_DIGITS = 6

_RUNNER = None          # per-worker engine process
_TIMEOUT = 20.0


def _init_worker(timeout_s: float) -> None:
    global _RUNNER, _TIMEOUT
    from ..engine import Runner
    _TIMEOUT = timeout_s
    _RUNNER = Runner(timeout_s=timeout_s)
    _RUNNER.start()


def _work(item: Tuple[Tuple[str, float, float], Tuple[float, ...]]) -> Verdict:
    """One genotype -> one Verdict, in a worker. Never raises: an engine death has to come
    back as a stage, or one bad design stalls the whole batch."""
    (env_name, fov, focus_w), x = item
    env = env_config.get(env_name, fov=fov, focus_w=focus_w)
    try:
        return reward_mod.evaluate(_RUNNER, env, np.asarray(x, float))
    except Exception as exc:                                # includes tracer timeouts
        try:
            _RUNNER.restart()
        except Exception:
            pass
        return Verdict("engine_error", f"{type(exc).__name__}: {exc}")


@dataclass
class PoolStats:
    calls: int = 0
    cache_hits: int = 0
    designs: int = 0
    engine_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.calls, 1)

    @property
    def designs_per_second(self) -> float:
        return self.designs / max(self.engine_seconds, 1e-9)


class RewardPool:
    """Batch genotype -> (Reward, Verdict), across `workers` engine processes.

    Usage:

        with RewardPool(workers=20) as pool:
            rewards = pool.score([(spec, genotype), ...])
    """

    def __init__(self, workers: Optional[int] = None, *, timeout_s: float = 20.0,
                 cache_size: int = 200_000, chunksize: int = 1):
        self.workers = int(workers or min(20, os.cpu_count() or 4))
        self.timeout_s = float(timeout_s)
        self.cache_size = int(cache_size)
        self.chunksize = int(chunksize)
        self._pool: Optional[mp.pool.Pool] = None
        self._cache: "OrderedDict[tuple, Verdict]" = OrderedDict()
        self.stats = PoolStats()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "RewardPool":
        if self._pool is None:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(self.workers, initializer=_init_worker,
                                  initargs=(self.timeout_s,))
        return self

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "RewardPool":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ cache
    def _key(self, variant: Tuple[str, float, float], x: Sequence[float]) -> tuple:
        # the VARIANT, not the env name: the same genotype at a different FOV or window
        # is a different measurement, and sharing a cache entry between them would hand
        # the training loop a score for a spec it never evaluated
        return tuple(variant) + tuple(round(float(v), CACHE_DIGITS) for v in x)

    def _remember(self, key: tuple, v: Verdict) -> None:
        self._cache[key] = v
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------ the call
    def verdicts(self, items: Sequence[Tuple[Tuple[str, float, float], Sequence[float]]]
                 ) -> List[Verdict]:
        """(variant, genotype) pairs -> Verdicts, cached and batched. A variant is the
        `(env_name, fov, focus_w)` triple a `Spec` carries.

        Duplicates WITHIN a batch are collapsed too, not only across batches: a sharpened
        policy emits the same vector many times in one group.
        """
        if self._pool is None:
            self.start()
        items = [(tuple(v), x) for v, x in items]
        keys = [self._key(v, x) for v, x in items]
        todo: "OrderedDict[tuple, Tuple[tuple, tuple]]" = OrderedDict()
        for key, (v, x) in zip(keys, items):
            if key not in self._cache and key not in todo:
                todo[key] = (v, tuple(float(val) for val in x))

        if todo:
            t0 = time.time()
            fresh = self._pool.map(_work, list(todo.values()),
                                   chunksize=self.chunksize)
            self.stats.engine_seconds += time.time() - t0
            self.stats.designs += len(todo)
            for key, v in zip(todo.keys(), fresh):
                self._remember(key, v)

        self.stats.calls += len(items)
        self.stats.cache_hits += len(items) - len(todo)
        out = []
        for key in keys:
            v = self._cache[key]
            self._cache.move_to_end(key)
            out.append(v)
        return out

    def score(self, items: Sequence[Tuple[Spec, Sequence[float]]]
              ) -> List[Tuple[Reward, Verdict]]:
        """(spec, genotype) pairs -> (Reward, Verdict). The training-loop entry point."""
        vs = self.verdicts([(s.variant, x) for s, x in items])
        return [(reward_mod.reward_of(
                    v, s, env_config.get(s.env, fov=s.fov, focus_w=s.focus_w)), v)
                for (s, _x), v in zip(items, vs)]


# ------------------------------------------------------------------ milestone 1 gate
def bench(n: int = 600, workers: Optional[int] = None, env: str = "fold",
          seed: int = 0, log=print) -> Dict[str, float]:
    """Sustained designs/s through the full reward path, on uniform random genotypes.

    Uniform noise, not archive elites, and the difference matters: a random genotype
    usually fails a geometry test before it reaches the engine, so a throughput number
    measured on noise flatters the service. The `engine_frac` column below reports how many
    of the sampled designs actually got traced, so the number can be read honestly.
    """
    e = env_config.get(env)
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(e.problem.lo, float), np.asarray(e.problem.hi, float)
    X = rng.uniform(lo, hi, size=(n, len(lo)))
    with RewardPool(workers=workers) as pool:
        # a warm batch first: the first call pays Node startup in every worker
        var = (env, e.fov, e.problem.focus_w)
        pool.verdicts([(var, x) for x in X[: pool.workers]])
        pool.stats = PoolStats()
        t0 = time.time()
        vs = pool.verdicts([(var, x) for x in X])
        dt = time.time() - t0
        w = pool.workers
    traced = sum(1 for v in vs if v.buildable)
    out = dict(designs=float(len(X)), workers=float(w), seconds=dt,
               designs_per_second=len(X) / dt,
               traced=float(traced), engine_frac=traced / max(len(X), 1))
    log(f"[bench] {len(X)} designs, {w} workers, {dt:.2f}s -> "
        f"{out['designs_per_second']:.1f} designs/s "
        f"({traced} reached the engine, {out['engine_frac']*100:.0f}%)")
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--env", default="fold")
    a = ap.parse_args()
    bench(n=a.n, workers=a.workers, env=a.env)
