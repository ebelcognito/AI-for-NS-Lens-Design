"""Canonical (mu/mu_w, lambda)-CMA-ES used as the inner engine of CMA-MEGA.

Replaces the previous EMA-tracked covariance heuristic with the canonical
CMA-ES update rules from Hansen 2016 (arXiv:1604.00772):

  * positive log weights w_i = log(mu+1/2) - log(i)
  * evolution paths p_sigma and p_c
  * rank-1 covariance update via p_c
  * rank-mu covariance update via weighted elite outer products
  * step-size adaptation via the cumulative path length heuristic
  * eigendecomposition of C for sampling and C^(-1/2) computation
  * Hansen's standard termination criteria (TolSigma, ConditionCov,
    NoEffectAxis, NoEffectCoord, TolX) exposed via should_restart()

The class keeps the historical (ask, tell, sigma, elite_frac) API used by
every CMA-MEGA runner in optical_rl/qd/, so existing call sites do not
need to change. New methods (reset, should_restart, get_state, set_state)
are additive.

Conventions:
  * dim         = coefficient dimensionality (1 fitness + k descriptors).
  * lambda_     = population_size; mu = lambda_ // 2.
  * scores passed to tell() are HIGHER-IS-BETTER. Sorting is descending,
    so the top-mu coefficients are the best-improving ones.
  * mean is non-zero in general - it drifts via the canonical update.
    Restart resets mean to 0 along with C, sigma, and the evolution paths.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


_CMA_STATE_SCHEMA: int = 1


class GradientCoefficientCMA:
    """Canonical (mu/mu_w, lambda)-CMA-ES over a (k+1)-dim coefficient space.

    Constructor args (kept for backwards compatibility with the old emitter):
      dim:             coefficient dimensionality.
      seed:            seed for the internal numpy Generator.
      sigma:           initial step size; clipped only by very loose
                       numerical sanity rails (1e-10, 1e10) - canonical
                       CMA-ES is self-limiting via the path-length update.
      elite_frac:      fraction of population_size used as mu. Default 0.5.
      population_size: lambda. If None, the per-call ask(n) value is used
                       to derive mu, weights, and the learning constants
                       on the first call (lazy init for compatibility).
    """

    def __init__(
        self,
        *,
        dim: int = 3,
        seed: int = 0,
        sigma: float = 0.75,
        elite_frac: float = 0.5,
        population_size: Optional[int] = None,
    ):
        self.dim = int(dim)
        self.elite_frac = float(elite_frac)
        self.sigma_initial = float(sigma)

        # State
        self._rng = np.random.default_rng(int(seed))
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.C = np.eye(self.dim, dtype=np.float64)
        self.sigma = float(sigma)
        self.p_sigma = np.zeros((self.dim,), dtype=np.float64)
        self.p_c = np.zeros((self.dim,), dtype=np.float64)
        self.B = np.eye(self.dim, dtype=np.float64)
        self.D = np.ones((self.dim,), dtype=np.float64)
        self._eigen_dirty = False
        self.generation = 0

        # Learning-rate constants. If population_size is given up front,
        # initialize now; otherwise lazy-initialize on first tell().
        self._constants_ready = False
        self.lambda_: int = 0
        self.mu: int = 0
        self.weights: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.mu_eff: float = 0.0
        self.c_sigma: float = 0.0
        self.d_sigma: float = 0.0
        self.c_c: float = 0.0
        self.c_1: float = 0.0
        self.c_mu: float = 0.0
        self.chi_n: float = 0.0
        if population_size is not None:
            self._init_constants(int(population_size))

    # ------------------------------------------------------------------
    # Hansen learning-rate constants (Eq. 49 of Hansen 2016 tutorial)
    # ------------------------------------------------------------------

    def _init_constants(self, lambda_: int) -> None:
        lambda_ = max(2, int(lambda_))
        mu = max(1, int(math.floor(self.elite_frac * lambda_)))
        # Hansen recommends mu = lambda // 2 for the default elite_frac
        # of 0.5, which matches floor(0.5 * lambda) for even lambda.
        # For odd lambda, floor(0.5 * lambda) gives the smaller half.

        # Positive log weights, normalized to sum to 1.
        raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        weights = raw / raw.sum()
        mu_eff = float(1.0 / np.sum(weights ** 2))

        n = float(self.dim)
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu_raw = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, c_mu_raw)
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.lambda_ = lambda_
        self.mu = mu
        self.weights = weights
        self.mu_eff = mu_eff
        self.c_sigma = c_sigma
        self.d_sigma = d_sigma
        self.c_c = c_c
        self.c_1 = c_1
        self.c_mu = c_mu
        self.chi_n = chi_n
        self._constants_ready = True

    # ------------------------------------------------------------------
    # Eigendecomposition cache
    # ------------------------------------------------------------------

    def _refresh_eigen(self) -> None:
        # Symmetrize defensively before decomposition.
        C_sym = (self.C + self.C.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(C_sym)
        # Floor tiny / negative eigenvalues to keep sqrt finite.
        eigvals = np.clip(eigvals, 1e-20, None)
        self.D = np.sqrt(eigvals).astype(np.float64, copy=False)
        self.B = eigvecs.astype(np.float64, copy=False)
        self.C = C_sym
        self._eigen_dirty = False

    def _ensure_eigen(self) -> None:
        if self._eigen_dirty:
            self._refresh_eigen()

    # ------------------------------------------------------------------
    # ask: sample n full coefficient vectors c = mean + sigma * B*D*z
    # ------------------------------------------------------------------

    def ask(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        self._ensure_eigen()
        # M[j, k] = B[j, k] * D[k]
        M = self.B * self.D[None, :]
        z = self._rng.standard_normal(size=(n, self.dim))
        y = z @ M.T  # (n, dim) ~ N(0, C)
        x = self.mean[None, :] + self.sigma * y
        return x.astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # tell: canonical CMA-ES update from sorted elites
    # ------------------------------------------------------------------

    def tell(self, coeffs: np.ndarray, scores: np.ndarray) -> None:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if coeffs.ndim != 2 or coeffs.shape[0] != scores.shape[0]:
            raise ValueError("coeffs and scores must align")
        n_samples = coeffs.shape[0]
        if n_samples == 0:
            return

        # Lazy-init constants on the first call (or whenever lambda_ has
        # never been set explicitly). We use the actual sample count.
        if not self._constants_ready or self.lambda_ <= 0:
            self._init_constants(n_samples)
        # If the runner is finishing up with a smaller batch than lambda_,
        # transparently fall back to the smaller mu without re-deriving
        # the global constants - this preserves adaptation across the run.
        mu_call = max(1, min(self.mu, n_samples // 2)) if n_samples >= 2 else 1
        if mu_call != self.mu:
            raw = np.log(mu_call + 0.5) - np.log(np.arange(1, mu_call + 1, dtype=np.float64))
            weights_call = raw / raw.sum()
            mu_eff_call = float(1.0 / np.sum(weights_call ** 2))
        else:
            weights_call = self.weights
            mu_eff_call = self.mu_eff

        # Descending sort: best (highest score) first.
        order = np.argsort(-scores, kind="stable")
        selected = order[:mu_call]
        elites_x = coeffs[selected]                       # (mu, dim)
        # y_i = (x_i - old_mean) / sigma  ~ N(0, C)
        elites_y = (elites_x - self.mean[None, :]) / max(self.sigma, 1e-20)

        # Weighted recombination in y-space
        y_w = weights_call @ elites_y                     # (dim,)

        # Update mean: m <- m + sigma * y_w  (canonical drift)
        old_mean = self.mean
        self.mean = old_mean + self.sigma * y_w

        # ------- evolution path for sigma -------
        self._ensure_eigen()
        # C^{-1/2} y_w  via  B diag(1/D) B^T y_w
        Bt_y = self.B.T @ y_w
        invD_Bt_y = Bt_y / self.D
        C_inv_sqrt_y_w = self.B @ invD_Bt_y
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * mu_eff_call
        ) * C_inv_sqrt_y_w

        norm_p_sigma = float(np.linalg.norm(self.p_sigma))
        # h_sigma indicator (Hansen Eq. 50)
        h_threshold = (
            (1.4 + 2.0 / (self.dim + 1.0))
            * self.chi_n
            * math.sqrt(max(0.0, 1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1))))
        )
        h_sigma = 1.0 if norm_p_sigma < h_threshold else 0.0

        # ------- evolution path for C -------
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2.0 - self.c_c) * mu_eff_call
        ) * y_w

        # ------- covariance update (rank-1 + rank-mu) -------
        artmp = elites_y                                   # (mu, dim)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        rank_one = np.outer(self.p_c, self.p_c) + delta_h * self.C
        # rank-mu = sum_i w_i * y_i y_i^T  =  (artmp.T * w) @ artmp
        rank_mu = (artmp.T * weights_call) @ artmp
        self.C = (1.0 - self.c_1 - self.c_mu) * self.C + self.c_1 * rank_one + self.c_mu * rank_mu

        # ------- step-size update -------
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1.0)
        )
        # Loose numerical sanity rail.
        self.sigma = float(np.clip(self.sigma, 1e-10, 1e10))

        # Symmetrize C and mark eigen cache dirty.
        self.C = (self.C + self.C.T) * 0.5
        self._eigen_dirty = True
        self.generation += 1

    # ------------------------------------------------------------------
    # Restart support
    # ------------------------------------------------------------------

    def reset(self, *, new_mean: Optional[np.ndarray] = None, new_sigma: Optional[float] = None) -> None:
        """Reset CMA-ES state. Used by the QD outer loop on no-improvement."""
        if new_mean is None:
            self.mean = np.zeros((self.dim,), dtype=np.float64)
        else:
            self.mean = np.asarray(new_mean, dtype=np.float64).reshape(self.dim).copy()
        self.C = np.eye(self.dim, dtype=np.float64)
        self.sigma = float(self.sigma_initial if new_sigma is None else new_sigma)
        self.p_sigma = np.zeros((self.dim,), dtype=np.float64)
        self.p_c = np.zeros((self.dim,), dtype=np.float64)
        self.B = np.eye(self.dim, dtype=np.float64)
        self.D = np.ones((self.dim,), dtype=np.float64)
        self._eigen_dirty = False
        # generation must reset because h_sigma's threshold depends on it
        # via (1 - (1-c_sigma)^(2*(g+1))).
        self.generation = 0

    def should_restart(self) -> Tuple[bool, Optional[str]]:
        """Hansen's standard CMA-ES termination criteria.

        Returns (True, reason_str) when the distribution has degenerated
        in a way that justifies a fresh restart from a new base point.
        """
        if not self._constants_ready:
            return False, None
        self._ensure_eigen()

        # TolSigma: sigma collapsed
        if self.sigma < self.sigma_initial * 1e-10:
            return True, "tol_sigma"

        # ConditionCov: ratio of largest to smallest eigenvalue of C
        eig_max = float(self.D.max() ** 2)
        eig_min = float(self.D.min() ** 2)
        if eig_min <= 0.0 or eig_max / eig_min > 1e14:
            return True, "condition_cov"

        # NoEffectAxis: cycling axis (Hansen heuristic)
        i_axis = self.generation % self.dim
        delta_axis = 0.1 * self.sigma * float(self.D[i_axis]) * self.B[:, i_axis]
        if np.allclose(self.mean, self.mean + delta_axis, rtol=0.0, atol=0.0):
            return True, "no_effect_axis"

        # NoEffectCoord: per-coordinate
        delta_coord = 0.2 * self.sigma * np.sqrt(np.diag(self.C))
        if np.any(self.mean == (self.mean + delta_coord)):
            return True, "no_effect_coord"

        # TolX
        tol_x = 1e-12 * self.sigma_initial
        max_dev = self.sigma * float(np.max(np.sqrt(np.diag(self.C))))
        max_p_c = self.sigma * float(np.max(np.abs(self.p_c))) if self.p_c.size else 0.0
        if max_dev < tol_x and max_p_c < tol_x:
            return True, "tol_x"

        return False, None

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Serializable snapshot of the full CMA-ES state."""
        return {
            "schema_version": _CMA_STATE_SCHEMA,
            "dim": int(self.dim),
            "elite_frac": float(self.elite_frac),
            "sigma_initial": float(self.sigma_initial),
            "constants_ready": bool(self._constants_ready),
            "lambda_": int(self.lambda_),
            "mu": int(self.mu),
            "weights": self.weights.tolist(),
            "mu_eff": float(self.mu_eff),
            "c_sigma": float(self.c_sigma),
            "d_sigma": float(self.d_sigma),
            "c_c": float(self.c_c),
            "c_1": float(self.c_1),
            "c_mu": float(self.c_mu),
            "chi_n": float(self.chi_n),
            "mean": self.mean.tolist(),
            "C": self.C.tolist(),
            "sigma": float(self.sigma),
            "p_sigma": self.p_sigma.tolist(),
            "p_c": self.p_c.tolist(),
            "B": self.B.tolist(),
            "D": self.D.tolist(),
            "eigen_dirty": bool(self._eigen_dirty),
            "generation": int(self.generation),
            "rng_state": self._rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from a snapshot produced by get_state()."""
        if int(state.get("schema_version", -1)) != _CMA_STATE_SCHEMA:
            raise ValueError(
                f"CMA state schema mismatch: file is "
                f"v{state.get('schema_version')}, runtime is v{_CMA_STATE_SCHEMA}"
            )
        if int(state["dim"]) != self.dim:
            raise ValueError(
                f"CMA state dim mismatch: file has dim={state['dim']}, "
                f"runtime emitter has dim={self.dim}"
            )
        self.elite_frac = float(state.get("elite_frac", self.elite_frac))
        self.sigma_initial = float(state.get("sigma_initial", self.sigma_initial))
        self._constants_ready = bool(state.get("constants_ready", False))
        self.lambda_ = int(state.get("lambda_", 0))
        self.mu = int(state.get("mu", 0))
        self.weights = np.asarray(state.get("weights", []), dtype=np.float64)
        self.mu_eff = float(state.get("mu_eff", 0.0))
        self.c_sigma = float(state.get("c_sigma", 0.0))
        self.d_sigma = float(state.get("d_sigma", 0.0))
        self.c_c = float(state.get("c_c", 0.0))
        self.c_1 = float(state.get("c_1", 0.0))
        self.c_mu = float(state.get("c_mu", 0.0))
        self.chi_n = float(state.get("chi_n", 0.0))
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.C = np.asarray(state["C"], dtype=np.float64)
        self.sigma = float(state["sigma"])
        self.p_sigma = np.asarray(state["p_sigma"], dtype=np.float64)
        self.p_c = np.asarray(state["p_c"], dtype=np.float64)
        self.B = np.asarray(state.get("B", np.eye(self.dim)), dtype=np.float64)
        self.D = np.asarray(state.get("D", np.ones(self.dim)), dtype=np.float64)
        self._eigen_dirty = bool(state.get("eigen_dirty", True))
        self.generation = int(state.get("generation", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state
