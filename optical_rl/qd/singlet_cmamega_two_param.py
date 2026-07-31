"""Two-parameter singlet CMA-MEGA: free c1 and c2, d_sensor derived via paraxial BFL solve.

This module extends singlet_cmamega.py by freeing both surface curvatures while
constraining the sensor position to the paraxial back focus.  With two free
parameters the archive can fill a 2D region in descriptor space rather than the
1D arc produced by the single-parameter (c1-only) formulation.

The only new physics is the `_paraxial_solve_bfl` method on TwoParamSingletProblem.
Everything else — archive, emitter, fitness, Zernike measures — is reused unchanged.

Also fixes the hardcoded "z4"/"z11" gradient keys in the runner so that any
descriptor_zernike_indices pair in the config works correctly.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np

from .singlet_cmamega import (
    CMAMEGAArchive2D,
    FixedSingletCMAMEGARunner,
    FixedSingletProblem,
    GradientCoefficientCMA,
    SingletCMAMEGAConfig,
    SingletEvalResult,
    theta_to_unit,
    unit_to_theta,
)


class TwoParamSingletProblem(FixedSingletProblem):
    """Singlet problem with theta = [c1, c2]; d_sensor derived from paraxial back-focal solve."""

    # ──────────────────────────────────────────────────────────────────────────
    # New physics: back-focal solve for free (c1, c2)
    # ──────────────────────────────────────────────────────────────────────────

    def _paraxial_solve_bfl(self, c1_t, c2_t):
        """Derive d_sensor by placing the sensor at the paraxial back focal point.

        Both c1 and c2 are free; the target EFL is not enforced — the system
        takes whatever focal length (c1, c2) implies and focuses there.

        Thick-lens back-focal distance (ABCD formulation):
            phi_1 = (n-1)*c1
            phi_2 = -(n-1)*c2     (rear surface: glass → air, normal into air)
            P     = phi_1 + phi_2 - t*phi_1*phi_2/n
            BFL   = (1 - t*phi_1/n) / P

        Singularity guard: P is clamped away from zero (diverging lens at threshold).
        """
        n = self._n_glass
        t = self._t_glass
        phi_1 = (n - 1.0) * c1_t
        phi_2 = -(n - 1.0) * c2_t
        P = phi_1 + phi_2 - t * phi_1 * phi_2 / n
        numerator = 1.0 - t * phi_1 / n
        safe_P = P.clamp(min=1e-4) if float(P.detach().item()) > 0 else P.clamp(max=-1e-4)
        bfl = numerator / safe_P
        return self._z_last + bfl

    # ──────────────────────────────────────────────────────────────────────────
    # Overrides
    # ──────────────────────────────────────────────────────────────────────────

    def _build_lens_from_theta_tensor(self, theta_t):
        """Build a GeoLens from theta_t = [c1, c2]; d_sensor via paraxial BFL solve."""
        c1, c2 = theta_t[0], theta_t[1]
        d_sensor = self._paraxial_solve_bfl(c1, c2)
        lens = copy.deepcopy(self._template_lens)
        dtype = lens.surfaces[1].c.dtype
        lens.surfaces[1].c = c1.to(dtype=dtype)
        lens.surfaces[2].c = c2.to(dtype=dtype)
        lens.d_sensor = d_sensor.to(dtype=dtype)
        return lens

    def theta_to_lens(self, theta: np.ndarray):
        """Public API: build a GeoLens from a 2D numpy theta = [c1, c2]."""
        self._ensure_backend()
        torch = self._torch
        theta = self.project_theta(theta)
        c1_t = torch.tensor(float(theta[0]), dtype=torch.float32, device=self._device)
        c2_t = torch.tensor(float(theta[1]), dtype=torch.float32, device=self._device)
        d_sensor_t = self._paraxial_solve_bfl(c1_t, c2_t)
        lens = copy.deepcopy(self._template_lens)
        dtype = lens.surfaces[1].c.dtype
        lens.surfaces[1].c = c1_t.to(dtype=dtype)
        lens.surfaces[2].c = c2_t.to(dtype=dtype)
        lens.d_sensor = d_sensor_t.to(dtype=dtype)
        return lens

    def initial_theta(self) -> np.ndarray:
        self._ensure_backend()
        return np.zeros(2, dtype=np.float64)

    def _fitness_and_metrics(self, theta_t):
        """Same as parent but applies a large penalty when invalid_fraction > invalid_tol
        or when the lens is diverging (negative focal length / concave lens).

        The penalty is added to the existing fitness (not replacing it) so the
        gradient graph through theta_t → fitness remains connected, allowing
        autograd to compute gradients even for invalid designs.
        """
        torch = self._torch

        # Reject diverging lenses (P <= 0 means negative focal length).
        n, t = self._n_glass, self._t_glass
        c1, c2 = theta_t[0], theta_t[1]
        phi_1 = (n - 1.0) * c1
        phi_2 = -(n - 1.0) * c2
        P = phi_1 + phi_2 - t * phi_1 * phi_2 / n
        if float(P.detach().item()) <= 0.0:
            bad = theta_t.sum() * 0.0 - 1e9
            return bad, theta_t[0] * 0.0, theta_t[1] * 0.0, {"invalid_fraction": 1.0}

        fitness, m0, m1, metrics = super()._fitness_and_metrics(theta_t)

        invalid_frac = metrics.get("invalid_fraction", 0.0)
        if invalid_frac > self.config.invalid_tol:
            # Subtract a large scalar while keeping the computation graph intact.
            fitness = fitness - float(self.config.invalid_penalty_weight) * 1e5

        return fitness, m0, m1, metrics


class TwoParamSingletCMAMEGARunner(FixedSingletCMAMEGARunner):
    """CMA-MEGA runner for the two-parameter singlet problem.

    Uses TwoParamSingletProblem and fixes the hardcoded gradient key issue in
    the parent runner so that any descriptor_zernike_indices pair works.
    """

    def __init__(self, config: SingletCMAMEGAConfig):
        # Call grandparent __init__ fields directly to avoid parent creating
        # a FixedSingletProblem — we substitute TwoParamSingletProblem instead.
        self.config = config
        self.problem = TwoParamSingletProblem(config)
        self.archive: Optional[CMAMEGAArchive2D] = None
        self.emitter = GradientCoefficientCMA(
            dim=1 + len(config.descriptor_zernike_indices),
            seed=int(config.cma_seed),
            sigma=float(config.initial_sigma),
        )

    def _normalized_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Dynamic gradient keys derived from descriptor_zernike_indices."""
        half = 0.5 * np.array(
            [hi - lo for lo, hi in self.config.theta_bounds], dtype=np.float64
        )

        def norm_grad(name: str) -> np.ndarray:
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        j0, j1 = self.config.descriptor_zernike_indices
        k0, k1 = f"z{j0}", f"z{j1}"
        return {"fitness": norm_grad("fitness"), k0: norm_grad(k0), k1: norm_grad(k1)}

    def run(self, iterations: int) -> Dict[str, float]:
        if self.archive is None or not self.archive.cells:
            self.seed_archive()

        j0, j1 = self.config.descriptor_zernike_indices
        k0, k1 = f"z{j0}", f"z{j1}"

        theta_dim = len(self.config.theta_bounds)
        scores = []
        eval_budget = int(iterations)
        completed = 0
        generation = 0

        while completed < eval_budget:
            parent_theta = self.archive.sample_parent()
            parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
            assert parent_eval.grads is not None

            unit_parent = theta_to_unit(parent_theta, self.config.theta_bounds)
            grads = self._normalized_gradients(parent_eval.grads)
            batch_n = min(int(self.config.population_size), eval_budget - completed)
            coeffs = self.emitter.ask(batch_n)
            batch_scores = []
            batch_improved = []

            for local_idx, coeff in enumerate(coeffs):
                direction = (
                    abs(coeff[0]) * grads["fitness"]
                    + coeff[1] * grads[k0]
                    + coeff[2] * grads[k1]
                )
                dir_norm = np.linalg.norm(direction)
                if not np.isfinite(dir_norm) or dir_norm < 1e-12:
                    rng = np.random.default_rng(self.config.cma_seed + completed + local_idx)
                    direction = rng.normal(0.0, 1.0, size=(theta_dim,))
                    direction /= np.linalg.norm(direction) + 1e-12

                candidate_unit = np.clip(
                    unit_parent + float(self.config.cma_eta) * direction, -1.0, 1.0
                )
                candidate_theta = unit_to_theta(candidate_unit, self.config.theta_bounds)
                candidate_eval = self.problem.evaluate(candidate_theta, with_grads=False)
                improved = self.archive.insert(candidate_eval)
                scores.append(candidate_eval.fitness)
                batch_scores.append(candidate_eval.fitness if improved else -np.inf)
                batch_improved.append(improved)

            self.emitter.tell(coeffs, np.asarray(batch_scores, dtype=np.float64))
            completed += batch_n
            generation += 1

            if generation % max(int(self.config.log_every), 1) == 0:
                stats = self.archive.stats()
                n_improved = sum(batch_improved)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted_elites = sum(1 for i in elite_idx if not batch_improved[i])
                print(
                    f"[singlet-2p-cmamega] gen={generation} evals={completed} "
                    f"coverage={stats['coverage']:.4f} best={stats['best_fitness']:.6f} | "
                    f"archive_improved={n_improved}/{batch_n} "
                    f"wasted_elites={n_wasted_elites}/{elite_n} "
                    f"sigma={self.emitter.sigma:.4f}",
                    flush=True,
                )

        stats = self.archive.stats()
        stats["iterations"] = float(iterations)
        stats["generations"] = float(generation)
        stats["mean_score"] = float(np.mean(scores)) if scores else float("nan")
        return stats
