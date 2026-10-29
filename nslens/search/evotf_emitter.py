"""Official EvoTF-ES behind the repository's small emitter interface.

The dependency is deliberately optional.  Importing :mod:`nslens` does not import JAX
or evosax; constructing :class:`EvoTFEmitter` does.  This module contains no learned
optimizer implementation -- all sampling, features, attention blocks, and update heads
come from ``evosax.algorithms.EvoTF_ES`` and its bundled released checkpoint.

Pinned upstream provenance
--------------------------
evosax 0.3.1, upstream commit ``d4b4127708cb2cd0cc906b910ddcd091013153a9``
bundles ``2024_03_SNES_small.pkl`` with SHA-256
``a802c7d2d312e42eb78f40fc7d8e9b245419f04ba9a2a0aa32f9933b254459ec``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from importlib import metadata, resources
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


EVOSAX_VERSION = "0.3.1"
JAX_VERSION = "0.6.2"
JAXLIB_VERSION = "0.6.2"
EVOTF_CHECKPOINT = "2024_03_SNES_small.pkl"
EVOTF_CHECKPOINT_SHA256 = (
    "a802c7d2d312e42eb78f40fc7d8e9b245419f04ba9a2a0aa32f9933b254459ec"
)
EVOSAX_UPSTREAM_COMMIT = "d4b4127708cb2cd0cc906b910ddcd091013153a9"


class EvoTFDependencyError(RuntimeError):
    """The optional, pinned official EvoTF runtime is unavailable or incompatible."""


def stable_rank_fitness(
    order: Sequence[int], population_size: Optional[int] = None
) -> np.ndarray:
    """Translate a best-first candidate order to minimization rank percentiles.

    The CMA-ME driver uses very large tier offsets solely to construct ``order``.  Those
    offsets have no physical scale and must never enter EvoTF.  A complete order also
    provides a deterministic tie convention: whichever tied candidate appears first in
    the driver's order receives the better rank.  The best and worst values are 0 and 1
    respectively (a singleton population receives 0).
    """
    order_array = np.asarray(order)
    if order_array.ndim != 1:
        raise ValueError(f"order must be one-dimensional, got shape {order_array.shape}")
    if not np.issubdtype(order_array.dtype, np.integer):
        if not np.all(np.equal(order_array, np.floor(order_array))):
            raise ValueError("order must contain integer candidate indices")
        order_array = order_array.astype(np.int64)
    else:
        order_array = order_array.astype(np.int64, copy=False)
    size = len(order_array) if population_size is None else int(population_size)
    if size <= 0:
        raise ValueError("population_size must be positive")
    if len(order_array) != size or not np.array_equal(
        np.sort(order_array), np.arange(size)
    ):
        raise ValueError(f"order must be a permutation of range({size})")

    percentiles = np.arange(size, dtype=np.float32) / max(size - 1, 1)
    fitness = np.empty(size, dtype=np.float32)
    fitness[order_array] = percentiles
    return fitness


@dataclass
class _Telemetry:
    ask_calls: int = 0
    tell_calls: int = 0
    ask_seconds: float = 0.0
    inference_seconds: float = 0.0
    clipped_candidates: int = 0
    clipped_values: int = 0
    clipped_by_gene: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    fallback_events: int = 0
    fallback_candidates: int = 0
    nonfinite_events: int = 0
    restart_events: int = 0
    restart_reason: Optional[str] = None
    numerical_failure: Optional[str] = None


class _OfficialBackend:
    """One immutable upstream strategy shared by same-shaped emitter lifecycles.

    ``evosax`` JIT-compiles methods with the strategy object as a static argument.
    Reusing this object prevents every CMA-ME restart from recompiling the identical
    transformer; optimizer state and PRNG keys remain private to each emitter.
    """

    def __init__(self, population_size: int, num_dims: int, max_context_len: int):
        jax, jnp, strategy_class = _load_official_runtime()
        self.jax = jax
        self.jnp = jnp
        self.strategy = strategy_class(
            population_size=population_size,
            solution=jnp.zeros(num_dims, dtype=jnp.float32),
            max_context_len=max_context_len,
        )


_BACKENDS: dict[tuple[int, int, int], _OfficialBackend] = {}


def _load_official_runtime():
    try:
        installed = metadata.version("evosax")
        installed_jax = metadata.version("jax")
        installed_jaxlib = metadata.version("jaxlib")
        import jax
        import jax.numpy as jnp
        from evosax.algorithms import EvoTF_ES
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise EvoTFDependencyError(
            "EvoTF requires the optional official runtime; install it with "
            "`python -m pip install -e '.[evotf]'` (evosax==0.3.1)."
        ) from exc
    if installed != EVOSAX_VERSION:
        raise EvoTFDependencyError(
            f"EvoTF integration is pinned to evosax=={EVOSAX_VERSION}; found {installed}."
        )
    if installed_jax != JAX_VERSION or installed_jaxlib != JAXLIB_VERSION:
        raise EvoTFDependencyError(
            "EvoTF integration requires "
            f"jax=={JAX_VERSION} and jaxlib=={JAXLIB_VERSION}; found "
            f"jax=={installed_jax}, jaxlib=={installed_jaxlib}."
        )
    try:
        checkpoint = (
            resources.files("evosax.algorithms")
            .joinpath("ckpt", "evotf", EVOTF_CHECKPOINT)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise EvoTFDependencyError(
            f"evosax=={EVOSAX_VERSION} does not contain {EVOTF_CHECKPOINT}"
        ) from exc
    digest = hashlib.sha256(checkpoint).hexdigest()
    if digest != EVOTF_CHECKPOINT_SHA256:
        raise EvoTFDependencyError(
            f"unexpected {EVOTF_CHECKPOINT} SHA-256 {digest}; "
            f"expected {EVOTF_CHECKPOINT_SHA256}"
        )
    return jax, jnp, EvoTF_ES


def verify_official_runtime() -> None:
    """Fail fast unless the pinned official implementation/checkpoint is present."""
    _load_official_runtime()


def _backend(population_size: int, num_dims: int, max_context_len: int) -> _OfficialBackend:
    key = (int(population_size), int(num_dims), int(max_context_len))
    if key not in _BACKENDS:
        _BACKENDS[key] = _OfficialBackend(*key)
    return _BACKENDS[key]


class EvoTFEmitter:
    """Thin ``ask``/``tell`` adapter for the released, diagonal EvoTF-ES.

    ``observations`` is accepted for the common future Optical-EvoTF interface but is
    intentionally ignored here: the released checkpoint receives only its official
    numeric solution, rank-fitness, and diagonal-distribution history features.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        *,
        popsize: Optional[int] = None,
        seed: int = 0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_context_len: int = 100,
    ):
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.ndim != 1 or len(x0) == 0 or not np.all(np.isfinite(x0)):
            raise ValueError("x0 must be a non-empty, finite one-dimensional array")
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            raise ValueError("sigma0 must be finite and positive")
        if popsize is None:
            popsize = 4 + int(np.floor(3 * np.log(len(x0))))
        if int(popsize) <= 0:
            raise ValueError("popsize must be positive")
        if int(max_context_len) <= 0:
            raise ValueError("max_context_len must be positive")

        self.N = len(x0)
        self.lam = int(popsize)
        self.bounds = self._validate_bounds(bounds, x0.shape)
        self.max_context_len = int(max_context_len)
        self._fallback_mean = self._clip_numpy(x0)
        self._backend = _backend(self.lam, self.N, self.max_context_len)
        self._strategy = self._backend.strategy
        self._jax, self._jnp = self._backend.jax, self._backend.jnp
        self._params = self._strategy.default_params.replace(std_init=float(sigma0))
        self._key = self._jax.random.key(int(seed))
        self._key, init_key = self._jax.random.split(self._key)
        self._state = self._strategy.init(
            init_key, self._jnp.asarray(x0, dtype=self._jnp.float32), self._params
        )
        self._telemetry = _Telemetry(clipped_by_gene=np.zeros(self.N, dtype=np.int64))
        self._restart_needed = False
        self._last_population = None
        self._last_clipping_mask = np.zeros((self.lam, self.N), dtype=bool)
        self._last_fallback_mask = np.zeros(self.lam, dtype=bool)

    @staticmethod
    def _validate_bounds(bounds, shape):
        if bounds is None:
            return None
        lo, hi = (np.asarray(part, dtype=np.float64) for part in bounds)
        if lo.shape != shape or hi.shape != shape:
            raise ValueError(f"bounds must both have shape {shape}")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo >= hi):
            raise ValueError("bounds must be finite with lo < hi in every dimension")
        return lo.copy(), hi.copy()

    def _clip_numpy(self, values: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return np.asarray(values, dtype=np.float64).copy()
        return np.clip(values, self.bounds[0], self.bounds[1])

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self._state.mean, dtype=np.float64)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self._state.std, dtype=np.float64)

    @property
    def sigma(self) -> float:
        """Scalar trace-compatible summary; EvoTF's actual scale is diagonal."""
        std = self.std
        if np.all(np.isfinite(std)) and np.all(std > 0.0):
            return float(np.exp(np.mean(np.log(std))))
        return float("nan")

    @property
    def gen(self) -> int:
        return int(np.asarray(self._state.generation_counter))

    @property
    def restart_needed(self) -> bool:
        return self._restart_needed

    @property
    def last_clipping_mask(self) -> np.ndarray:
        return self._last_clipping_mask.copy()

    @property
    def last_fallback_mask(self) -> np.ndarray:
        return self._last_fallback_mask.copy()

    def ask(self) -> np.ndarray:
        if self._last_population is not None:
            raise RuntimeError("ask() called twice without tell()")
        started = time.perf_counter()
        self._key, ask_key = self._jax.random.split(self._key)
        raw, state = self._strategy.ask(ask_key, self._state, self._params)

        if self.bounds is None:
            clipped = raw
            clipping_mask = ~self._jnp.isfinite(raw)
        else:
            lo = self._jnp.asarray(self.bounds[0], dtype=raw.dtype)
            hi = self._jnp.asarray(self.bounds[1], dtype=raw.dtype)
            clipping_mask = (~self._jnp.isfinite(raw)) | (raw < lo) | (raw > hi)
            clipped = self._jnp.clip(raw, lo, hi)

        finite_rows = self._jnp.all(self._jnp.isfinite(clipped), axis=1)
        fallback = self._jnp.asarray(self._fallback_mean, dtype=raw.dtype)
        population = self._jnp.where(finite_rows[:, None], clipped, fallback[None, :])
        self._jax.block_until_ready(population)
        elapsed = time.perf_counter() - started

        mask_np = np.asarray(clipping_mask, dtype=bool)
        fallback_np = np.asarray(~finite_rows, dtype=bool)
        self._last_clipping_mask = mask_np
        self._last_fallback_mask = fallback_np
        self._telemetry.ask_calls += 1
        self._telemetry.ask_seconds += elapsed
        self._telemetry.clipped_candidates += int(
            np.count_nonzero(np.any(mask_np, axis=1))
        )
        self._telemetry.clipped_values += int(np.count_nonzero(mask_np))
        self._telemetry.clipped_by_gene += np.count_nonzero(mask_np, axis=0)
        if np.any(fallback_np):
            self._telemetry.fallback_events += 1
            self._telemetry.fallback_candidates += int(np.count_nonzero(fallback_np))
            self._mark_numerical_failure("non-finite candidate population")

        self._state = state
        self._last_population = population
        return np.asarray(population, dtype=np.float64)

    def tell(
        self,
        order: Sequence[int],
        observations: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        del observations  # released EvoTF has no optical-context input
        if self._last_population is None:
            raise RuntimeError("tell() called before ask()")
        rank_fitness = stable_rank_fitness(order, self.lam)
        started = time.perf_counter()
        self._key, tell_key = self._jax.random.split(self._key)
        state, _ = self._strategy.tell(
            tell_key,
            self._last_population,
            self._jnp.asarray(rank_fitness),
            self._state,
            self._params,
        )
        self._jax.block_until_ready(state.mean)
        elapsed = time.perf_counter() - started
        self._state = state
        self._last_population = None
        self._telemetry.tell_calls += 1
        self._telemetry.inference_seconds += elapsed
        if not self._state_is_finite():
            self._mark_numerical_failure("non-finite EvoTF distribution state")

    def _state_is_finite(self) -> bool:
        leaves = self._jax.tree.leaves(self._state)
        return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves)

    def _mark_numerical_failure(self, reason: str) -> None:
        self._restart_needed = True
        self._telemetry.nonfinite_events += 1
        self._telemetry.numerical_failure = reason

    def note_restart(self, reason: str) -> None:
        """Record the driver-level lifecycle termination before replacement."""
        self._telemetry.restart_events += 1
        self._telemetry.restart_reason = str(reason)

    def state_summary(self) -> dict[str, Any]:
        """Return JSON-serializable distribution state, provenance, and telemetry."""
        state_finite = self._state_is_finite()

        def serializable(values):
            return [float(value) if np.isfinite(value) else None for value in values]

        return {
            "algorithm": "released_evotf_es",
            "evosax_version": EVOSAX_VERSION,
            "jax_version": JAX_VERSION,
            "jaxlib_version": JAXLIB_VERSION,
            "upstream_commit": EVOSAX_UPSTREAM_COMMIT,
            "checkpoint": EVOTF_CHECKPOINT,
            "checkpoint_sha256": EVOTF_CHECKPOINT_SHA256,
            "population_size": self.lam,
            "dimensions": self.N,
            "max_context_len": self.max_context_len,
            "generation": self.gen,
            "mean": serializable(self.mean),
            "diagonal_std": serializable(self.std),
            "state_finite": bool(state_finite),
            "restart_needed": bool(self.restart_needed),
            "ask_calls": self._telemetry.ask_calls,
            "tell_calls": self._telemetry.tell_calls,
            "ask_seconds": float(self._telemetry.ask_seconds),
            "inference_seconds": float(self._telemetry.inference_seconds),
            "clipped_candidates": self._telemetry.clipped_candidates,
            "clipped_values": self._telemetry.clipped_values,
            "clipped_by_gene": [int(value) for value in self._telemetry.clipped_by_gene],
            "fallback_events": self._telemetry.fallback_events,
            "fallback_candidates": self._telemetry.fallback_candidates,
            "nonfinite_events": self._telemetry.nonfinite_events,
            "numerical_failure": self._telemetry.numerical_failure,
            "restart_events": self._telemetry.restart_events,
            "restart_reason": self._telemetry.restart_reason,
        }


__all__ = [
    "EVOSAX_UPSTREAM_COMMIT",
    "EVOSAX_VERSION",
    "EVOTF_CHECKPOINT",
    "EVOTF_CHECKPOINT_SHA256",
    "JAXLIB_VERSION",
    "JAX_VERSION",
    "EvoTFDependencyError",
    "EvoTFEmitter",
    "stable_rank_fitness",
    "verify_official_runtime",
]
