from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .archive_grid2d import ArchiveGrid2D
from .coeff_cma import GradientCoefficientCMA
from .problem_triplet_cooke import theta_to_unit, unit_to_theta
from .types import Problem, RunnerConfig


class TripletCMAMEGARunner:
    def __init__(self, config: RunnerConfig, problem: Problem):
        self.config = config
        self.problem = problem
        self.archive: Optional[ArchiveGrid2D] = None
        coeff_dim = 1 + len(tuple(config.active_descriptor_indices))
        self.emitter = GradientCoefficientCMA(
            dim=coeff_dim,
            seed=int(config.cma_seed),
            sigma=float(config.initial_sigma),
            population_size=int(config.population_size),
        )
        self._clip_hits_total = 0.0
        self._eval_counter = 0.0
        # Persistent runner state — survives checkpoint/resume.
        self.completed: int = 0
        self.generation: int = 0
        self.restart_count: int = 0
        self.wallclock_seconds: float = 0.0
        self.result_dir = None

    def calibrate_archive_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        active = tuple(int(i) for i in self.config.active_descriptor_indices)
        samples = self.problem.sample_theta(self.config.calibration_count, self.config.cma_seed + 17)
        vals0, vals1 = [], []
        for theta in samples:
            result = self.problem.evaluate(theta, with_grads=False)
            vals0.append(float(result.descriptor_vector[active[0]]))
            vals1.append(float(result.descriptor_vector[active[1]]))

        arr0 = np.asarray(vals0, dtype=np.float64)
        arr1 = np.asarray(vals1, dtype=np.float64)
        lo0 = float(np.percentile(arr0, 2.0))
        hi0 = float(np.percentile(arr0, 98.0))
        span0 = max(hi0 - lo0, 1e-6)
        lo1 = float(np.percentile(arr1, 2.0))
        hi1 = float(np.percentile(arr1, 98.0))
        span1 = max(hi1 - lo1, 1e-6)

        bounds = (
            (lo0 - 0.05 * span0, hi0 + 0.05 * span0),
            (lo1 - 0.05 * span1, hi1 + 0.05 * span1),
        )
        self.archive = ArchiveGrid2D(self.config.archive_bins, bounds, seed=self.config.cma_seed)
        return bounds

    def seed_archive(self) -> Dict[str, float]:
        if self.archive is None:
            if self.config.archive_bounds is not None:
                self.archive = ArchiveGrid2D(
                    self.config.archive_bins,
                    self.config.archive_bounds,
                    seed=self.config.cma_seed,
                )
            else:
                self.calibrate_archive_bounds()

        flat = self.problem.initial_theta()
        first = self.problem.evaluate(flat, with_grads=False)
        self.archive.insert(first)
        self._clip_hits_total += float(first.metrics.get("clip_hits_total", 0.0))
        self._eval_counter += 1.0

        seed_thetas = self.problem.sample_theta(self.config.seed_count, self.config.cma_seed + 31)
        for theta in seed_thetas:
            result = self.problem.evaluate(theta, with_grads=False)
            self.archive.insert(result)
            self._clip_hits_total += float(result.metrics.get("clip_hits_total", 0.0))
            self._eval_counter += 1.0
        stats = self.archive.stats()
        return {
            "filled": stats.filled,
            "total_cells": stats.total_cells,
            "coverage": stats.coverage,
            "best_fitness": stats.best_fitness,
        }

    def _normalized_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        half = 0.5 * np.array([hi - lo for lo, hi in self.config.theta_bounds], dtype=np.float64)

        def norm(name: str) -> np.ndarray:
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        out: Dict[str, np.ndarray] = {"fitness": norm("fitness")}
        for idx in range(len(tuple(self.config.active_descriptor_indices))):
            out[f"d{idx}"] = norm(f"d{tuple(self.config.active_descriptor_indices)[idx]}")
        return out

    def run(self, iterations: int, *, resume: bool = False) -> Dict[str, float]:
        """Canonical CMA-MEGA loop. See ef50_zernike_parallel.EF50ZernikeRunner.run
        for the full docstring — this runner shares all checkpoint/restart/
        telemetry semantics.
        """
        import signal
        import time
        from pathlib import Path
        from optical_rl.qd._checkpoint import (
            CMAMEGACheckpoint,
            INSERT_NEW,
            INSERT_REPLACED,
            TrajectoryLogger,
            hash_config,
            now_iso,
            save_checkpoint,
        )

        if not resume and (self.archive is None or not self.archive.cells):
            self.seed_archive()

        eval_budget = int(iterations)
        if not resume:
            self.completed = 0
            self.generation = 0
            self.restart_count = 0
            self.wallclock_seconds = 0.0
        target_completed = self.completed + eval_budget
        scores: list = []
        active = tuple(self.config.active_descriptor_indices)

        result_dir = self.result_dir
        ckpt_path = (Path(result_dir) / "checkpoint.pkl") if result_dir is not None else None
        traj_logger = (
            TrajectoryLogger(Path(result_dir) / "trajectory.csv")
            if result_dir is not None else None
        )
        cfg_hash = hash_config(self.config)
        checkpoint_every = max(0, int(self.config.checkpoint_every))
        new_cell_bonus = float(self.config.new_cell_bonus)

        def _build_ckpt() -> "CMAMEGACheckpoint":
            return CMAMEGACheckpoint(
                archive=self.archive,
                emitter_state=self.emitter.get_state(),
                completed_evaluations=int(self.completed),
                generation=int(self.generation),
                restart_count=int(self.restart_count),
                best_fitness=float(self.archive.stats().best_fitness),
                config_hash=cfg_hash,
                timestamp=now_iso(),
                wallclock_seconds=float(self.wallclock_seconds),
            )

        def _flush_checkpoint() -> None:
            if ckpt_path is None:
                return
            save_checkpoint(_build_ckpt(), ckpt_path)

        sigint_received = {"flag": False}
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):  # noqa: ARG001
            sigint_received["flag"] = True
            print("[triplet-cmamega] SIGINT received — checkpointing then exiting.", flush=True)

        signal.signal(signal.SIGINT, _handle_sigint)
        loop_t0 = time.perf_counter()

        try:
            while self.completed < target_completed:
                if sigint_received["flag"]:
                    break
                parent_theta = self.archive.sample_parent()
                parent_eval = self.problem.evaluate(parent_theta, with_grads=True)
                assert parent_eval.grads is not None
                grads = self._normalized_gradients(parent_eval.grads)
                unit_parent = theta_to_unit(parent_theta, self.config.theta_bounds)

                batch_n = min(int(self.config.population_size), target_completed - self.completed)
                coeffs = self.emitter.ask(batch_n)

                batch_scores: list = []
                batch_status: list = []
                for local_idx, coeff in enumerate(coeffs):
                    direction = abs(coeff[0]) * grads["fitness"]
                    for j in range(len(active)):
                        direction = direction + coeff[j + 1] * grads[f"d{j}"]
                    dir_norm = np.linalg.norm(direction)
                    if not np.isfinite(dir_norm) or dir_norm < 1e-12:
                        rng = np.random.default_rng(self.config.cma_seed + self.completed + local_idx)
                        direction = rng.normal(0.0, 1.0, size=(len(self.config.theta_bounds),))
                        direction /= np.linalg.norm(direction) + 1e-12

                    candidate_unit = np.clip(unit_parent + float(self.config.cma_eta) * direction, -1.0, 1.0)
                    candidate_theta = unit_to_theta(candidate_unit, self.config.theta_bounds)
                    candidate_eval = self.problem.evaluate(candidate_theta, with_grads=False)
                    status = int(self.archive.insert(candidate_eval))
                    self._clip_hits_total += float(candidate_eval.metrics.get("clip_hits_total", 0.0))
                    self._eval_counter += 1.0
                    scores.append(candidate_eval.fitness)
                    if status == INSERT_NEW:
                        batch_scores.append(candidate_eval.fitness + new_cell_bonus)
                    elif status == INSERT_REPLACED:
                        batch_scores.append(candidate_eval.fitness)
                    else:
                        batch_scores.append(-np.inf)
                    batch_status.append(status)

                self.emitter.tell(coeffs, np.asarray(batch_scores, dtype=np.float64))
                self.completed += batch_n
                self.generation += 1

                n_new = sum(1 for s in batch_status if s == INSERT_NEW)
                n_replaced = sum(1 for s in batch_status if s == INSERT_REPLACED)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted = sum(1 for i in elite_idx if batch_status[i] == 0)

                restart_reason = None
                if (n_new + n_replaced) == 0:
                    restart_reason = "no_improvement"
                else:
                    restart_flag, hansen_reason = self.emitter.should_restart()
                    if restart_flag:
                        restart_reason = hansen_reason or "hansen_term"
                if restart_reason is not None:
                    sigma_pre = float(self.emitter.sigma)
                    self.emitter.reset()
                    self.restart_count += 1
                    print(
                        f"[triplet-cmamega] [restart] gen={self.generation} "
                        f"reason={restart_reason} sigma_pre={sigma_pre:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                self.wallclock_seconds += (time.perf_counter() - loop_t0)
                loop_t0 = time.perf_counter()

                if traj_logger is not None:
                    s = self.archive.stats()
                    traj_logger.log(
                        gen=self.generation,
                        evals=self.completed,
                        coverage=float(s.coverage),
                        best_fitness=float(s.best_fitness),
                        new_cells=n_new,
                        replaced_elites=n_replaced,
                        wasted_elites=n_wasted,
                        sigma=float(self.emitter.sigma),
                        restarts=self.restart_count,
                        wallclock_s=float(self.wallclock_seconds),
                    )

                if self.generation % max(int(self.config.log_every), 1) == 0:
                    s = self.archive.stats()
                    print(
                        f"[triplet-cmamega] gen={self.generation} evals={self.completed} "
                        f"coverage={s.coverage:.4f} best={s.best_fitness:.6f} | "
                        f"new={n_new}/{batch_n} replaced={n_replaced}/{batch_n} "
                        f"wasted={n_wasted}/{elite_n} "
                        f"sigma={self.emitter.sigma:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                if checkpoint_every > 0 and self.generation % checkpoint_every == 0:
                    _flush_checkpoint()
        finally:
            try:
                _flush_checkpoint()
            except Exception as e:  # pragma: no cover
                print(f"[triplet-cmamega] WARNING: final checkpoint failed: {e}", flush=True)
            signal.signal(signal.SIGINT, prev_handler)

        s = self.archive.stats()
        return {
            "filled": s.filled,
            "total_cells": s.total_cells,
            "coverage": s.coverage,
            "best_fitness": s.best_fitness,
            "iterations": float(eval_budget),
            "generations": float(self.generation),
            "restart_count": float(self.restart_count),
            "mean_score": float(np.mean(scores)) if scores else float("nan"),
            "clip_hits_mean": float(self._clip_hits_total / max(self._eval_counter, 1.0)),
        }
