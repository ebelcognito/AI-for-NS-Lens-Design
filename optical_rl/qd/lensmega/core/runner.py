from __future__ import annotations

import csv
import json
import math
import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from optical_rl.qd._checkpoint import (
    CMAMEGACheckpoint,
    INSERT_NEW,
    INSERT_REPLACED,
    now_iso,
)
from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA

from ..archive.grid import GridArchive
from ..config.schema import AlgorithmCfg, RunConfig
from ..descriptors.base import Descriptor
from ..problem.base import EvalResult
from ..problem.lens_problem import LensProblem
from ..problem.tracing import project_box, theta_to_unit, unit_to_theta
from .cpu_pool import CPUWorkerPool
from .lineage import LineageLogger, theta_hash
from .output_manager import OutputManager
from .seeder import Seeder


class _EliteRenderWorker:
    """Background thread that renders elite lens designs without blocking."""

    def __init__(self, elite_dir: Path, cfg: "RunConfig") -> None:
        self._dir = elite_dir
        self._cfg = cfg
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, tag: str, theta: np.ndarray, info: dict) -> None:
        self._q.put((tag, theta.copy(), dict(info)))

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            tag, theta, info = item
            try:
                self._render(tag, theta, info)
            except Exception as e:
                print(f"[elite-render] failed {tag}: {e}", flush=True)

    def _render(self, tag: str, theta: np.ndarray, info: dict) -> None:
        import torch

        p = self._cfg.problem

        try:
            from deeplens.optics import GeoLens
            from optical_rl.qd.render_lensmega_archive import _apply_theta_generic
            from optical_rl.qd.lensmega.problem.lens_template import load_lens_template
            from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
            from optical_rl.qd.lensmega.descriptors.seidel import compute_per_element_seidel
        except Exception as e:
            print(f"[elite-render] import failed: {e}", flush=True)
            return

        def _load_lens_and_template():
            template, lens = load_lens_template(
                p.lens_path,
                wavelength_um=p.wavelength_um,
                stop_surface_index=p.stop_surface_index,
                free_curvature_indices=tuple(p.free_curvature_indices),
                derived_curvature_index=p.derived_curvature_index,
                device="cpu",
                cemented_interfaces=tuple(p.cemented_interfaces),
            )
            solver = GeneralizedParaxialSolver(template)
            return template, lens, solver

        def _apply_theta(lens, th, template, solver):
            _apply_theta_generic(
                lens, th,
                template=template, solver=solver,
                focal_length_target_mm=p.focal_length_target_mm,
                device="cpu",
                parameterization=p.parameterization,
                free_curvature_indices=p.free_curvature_indices,
                free_gap_indices=p.free_gap_indices,
                variable_flange=p.variable_flange,
                variable_materials=p.variable_materials,
                free_material_indices=p.free_material_indices,
            )

        def _compute_fnum(lens):
            """Paraxial F# read from the built lens. Returns None if it fails
            so the caller can fall back to '?' in the title."""
            try:
                import torch as _torch
                with _torch.no_grad():
                    _, pupilr = lens.calc_entrance_pupil(paraxial=True)
                pr = float(pupilr)
                if pr <= 1e-6:
                    return None
                efl = float(lens.foclen)
                return efl / (2.0 * pr)
            except Exception:
                return None

        def _render_png(lens, filepath, title):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            lens.draw_lens_2d(ax=ax, fig=fig, color="black")
            rfov = float(np.rad2deg(float(lens.rfov))) if hasattr(lens, "rfov") else 14.0
            for fov_frac, color in [(0.0, "#2196F3"), (0.5, "#4CAF50"), (1.0, "#F44336")]:
                try:
                    ray = lens.sample_point_source_2D(
                        fov=fov_frac * rfov, depth=-20000.0, num_rays=11, wvln=0.587,
                    )
                    ray.prop_to(-1.0)
                    _, rec = lens.trace2sensor(ray=ray, record=True)
                    lens.draw_ray_2d(rec, ax=ax, fig=fig, color=color)
                except Exception:
                    pass
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("z (mm)")
            ax.set_ylabel("y (mm)")
            fig.tight_layout()
            fig.savefig(filepath, dpi=150)
            plt.close(fig)

        # ---- 1. Per-element Seidel fingerprint on the RAW prior ----
        template, lens, solver = _load_lens_and_template()
        _apply_theta(lens, theta, template, solver)

        hfov = 14.4  # approximate; exact doesn't matter for fingerprint
        try:
            import math
            efl = float(info.get("efl_mm", 50.0))
            half_sensor = template.sensor_size_mm[0] / 2.0
            hfov = math.degrees(math.atan(half_sensor / max(abs(efl), 1.0)))
        except Exception:
            pass

        try:
            fingerprint = compute_per_element_seidel(
                lens, template,
                field_y_deg=0.5 * hfov,
                wavelength_um=p.wavelength_um,
                device="cpu",
            )
            info["fingerprint_raw"] = fingerprint
        except Exception as e:
            info["fingerprint_raw"] = {"error": str(e)}

        # ---- 2. Render raw prior PNG ----
        rms_um = info.get("rms_um", "?")
        efl_s = info.get("efl_mm", "?")
        signs = info.get("sign_pattern", "?")
        try:
            raw_fit = info.get("fitness", "?")
            fnum_raw = _compute_fnum(lens)
            fnum_str = f"F/{fnum_raw:.2f}" if fnum_raw is not None else "F/?"
            _render_png(
                lens,
                self._dir / f"{tag}_raw.png",
                f"{tag} RAW | fitness={raw_fit:.6f}  RMS={rms_um}μm  "
                f"EFL={efl_s}mm  {fnum_str}  [{signs}]",
            )
        except Exception as e:
            print(f"[elite-render] raw PNG failed {tag}: {e}", flush=True)

        # ---- 3. L-BFGS refinement ----
        #
        # The refiner uses a heavier EFL penalty weight (2x the config value)
        # so that the optimizer aggressively locks EFL while polishing spot
        # size.  The spot-size gradient dominates the step direction.
        try:
            theta_bounds = list(p.theta_bounds)
            lo = np.array([b[0] for b in theta_bounds], dtype=np.float64)
            hi = np.array([b[1] for b in theta_bounds], dtype=np.float64)

            # Build a fresh problem for refinement (thread-safe).
            # Override penalty weights from config so the refiner can
            # prioritise spot size over constraints.
            from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
            import copy as _copy
            refine_cfg = _copy.deepcopy(self._cfg)
            io = self._cfg.io
            if io.refine_efl_penalty_weight is not None:
                refine_cfg.problem.efl_penalty_weight = io.refine_efl_penalty_weight
            if io.refine_invalid_penalty_weight is not None:
                refine_cfg.problem.invalid_penalty_weight = io.refine_invalid_penalty_weight
            if io.refine_intersect_penalty_weight is not None:
                refine_cfg.problem.intersect_penalty_weight = io.refine_intersect_penalty_weight
            if io.refine_field_weights is not None:
                refine_cfg.problem.field_weights = tuple(io.refine_field_weights)
            # Heavier F# penalty during the polishing stage. Only takes
            # effect when problem.enforce_fnum is True; otherwise the
            # penalty is skipped and this override is a no-op.
            if (refine_cfg.problem.enforce_fnum
                    and io.refine_fnum_penalty_weight is not None):
                refine_cfg.problem.fnum_penalty_weight = io.refine_fnum_penalty_weight
            desc = _build_descriptor(refine_cfg)
            prob = _build_problem(refine_cfg, desc)
            prob._ensure_backend()

            theta_t = torch.tensor(
                theta.copy(), dtype=torch.float32, device="cpu",
                requires_grad=True,
            )

            optimizer = torch.optim.LBFGS(
                [theta_t], lr=io.refine_lbfgs_lr, max_iter=10,
                line_search_fn="strong_wolfe",
            )
            lo_t = torch.tensor(lo, dtype=torch.float32)
            hi_t = torch.tensor(hi, dtype=torch.float32)
            converge_tol = io.refine_converge_tol

            best_loss = float("inf")
            best_theta = theta.copy()
            lbfgs_steps = 0
            converged = False
            final_gnorm = float("inf")
            no_improve_count = 0

            for step in range(io.refine_lbfgs_max_steps):
                def closure():
                    optimizer.zero_grad()
                    with torch.no_grad():
                        theta_t.clamp_(lo_t, hi_t)
                    fitness, _, _ = prob._fitness_and_metrics(theta_t)
                    loss = -fitness
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                loss_val = optimizer.step(closure)
                lbfgs_steps = step + 1

                if loss_val is not None:
                    lv = float(loss_val)
                    if lv < best_loss - 1e-8:
                        best_loss = lv
                        no_improve_count = 0
                        with torch.no_grad():
                            best_theta = theta_t.detach().cpu().numpy().copy()
                    else:
                        no_improve_count += 1

                if theta_t.grad is not None:
                    final_gnorm = float(theta_t.grad.norm().item())
                    if final_gnorm < converge_tol:
                        converged = True
                        break

                # Plateau detection
                if no_improve_count >= 5:
                    converged = True  # plateaued = effectively converged
                    break

            best_theta = np.clip(best_theta, lo, hi)
            refined_fitness = -best_loss
            raw_fitness = info.get("fitness", -999)
            improvement = (
                raw_fitness / refined_fitness
                if abs(refined_fitness) > 1e-12 else 0.0
            )

            # Evaluate refined design to get actual metrics
            refined_eval = prob.evaluate(best_theta, with_grads=False)
            ref_rms = refined_eval.metrics.get("rms_spot", abs(refined_fitness))
            ref_efl = refined_eval.metrics.get("efl_actual", 0.0)

            # ---- Basin stability analysis ----
            # Perturb the refined design in random directions, measure
            # how much fitness degrades.  Flat basin = stable prior.
            n_stab = io.refine_stability_samples
            perturb_scale = io.refine_stability_perturbation
            bounds_range = hi - lo
            rng = np.random.RandomState(42)
            fitness_deltas = []
            for _ in range(n_stab):
                noise = rng.randn(len(best_theta)) * perturb_scale * bounds_range
                perturbed = np.clip(best_theta + noise, lo, hi)
                pr = prob.evaluate(perturbed, with_grads=False)
                fitness_deltas.append(pr.fitness - refined_fitness)

            fitness_deltas = np.array(fitness_deltas)
            # basin_width: mean fitness held under perturbation (less negative = wider)
            # sensitivity: std of fitness change (lower = more stable)
            stability = {
                "perturbation_scale": perturb_scale,
                "n_samples": n_stab,
                "fitness_delta_mean": round(float(fitness_deltas.mean()), 8),
                "fitness_delta_std": round(float(fitness_deltas.std()), 8),
                "fitness_delta_worst": round(float(fitness_deltas.min()), 8),
                "pct_survived": round(
                    float((fitness_deltas > -abs(refined_fitness)).mean()) * 100, 1
                ),
                "sensitivity": round(float(fitness_deltas.std()), 8),
            }

            info["refined"] = {
                "theta": [float(v) for v in best_theta],
                "fitness": float(refined_fitness),
                "rms_mm": float(ref_rms),
                "rms_um": f"{ref_rms * 1000:.1f}",
                "efl_mm": f"{ref_efl:.2f}",
                "lbfgs_steps": lbfgs_steps,
                "converged": converged,
                "final_grad_norm": round(float(final_gnorm), 8),
                "improvement_ratio": round(float(improvement), 4),
                "raw_fitness": float(raw_fitness),
                "stability": stability,
                "metrics": {
                    k: float(v) for k, v in refined_eval.metrics.items()
                },
            }

            # F# from the refined design (paraxial, read from the rebuilt lens
            # below — also stashed in info for the JSON / PNG label).
            ref_fnum = refined_eval.metrics.get("fnum_actual")
            if ref_fnum is None:
                # F# penalty disabled in this run; compute on demand for logging.
                _tmpl_fn, _lens_fn, _solv_fn = _load_lens_and_template()
                _apply_theta(_lens_fn, best_theta, _tmpl_fn, _solv_fn)
                ref_fnum = _compute_fnum(_lens_fn)
            fnum_str_ref = f"F/{ref_fnum:.2f}" if ref_fnum is not None else "F/?"

            conv_str = "CONVERGED" if converged else "max_steps"
            print(
                f"[elite-refine] {tag}: "
                f"raw={raw_fitness:.6f} → refined={refined_fitness:.6f} "
                f"RMS={ref_rms*1000:.1f}μm EFL={ref_efl:.1f}mm {fnum_str_ref} "
                f"({lbfgs_steps} steps, {conv_str}, "
                f"gnorm={final_gnorm:.2e}, {improvement:.2f}x) "
                f"stability: sensitivity={stability['sensitivity']:.6f} "
                f"survived={stability['pct_survived']:.0f}%",
                flush=True,
            )

            # ---- 4. Per-element fingerprint on REFINED design ----
            template2, lens2, solver2 = _load_lens_and_template()
            _apply_theta(lens2, best_theta, template2, solver2)

            try:
                fp_refined = compute_per_element_seidel(
                    lens2, template2,
                    field_y_deg=0.5 * hfov,
                    wavelength_um=p.wavelength_um,
                    device="cpu",
                )
                info["fingerprint_refined"] = fp_refined
            except Exception as e:
                info["fingerprint_refined"] = {"error": str(e)}

            # ---- 5. Render refined PNG ----
            ref_signs = "".join(
                "+" if v > 0 else "-" for v in np.sign(best_theta[:6])
            )
            try:
                sens = stability['sensitivity']
                surv = stability['pct_survived']
                fnum_str_png = (f"F/{ref_fnum:.2f}"
                                if ref_fnum is not None else "F/?")
                _render_png(
                    lens2,
                    self._dir / f"{tag}_refined.png",
                    f"{tag} REFINED | fitness={refined_fitness:.6f}  "
                    f"RMS={ref_rms*1000:.1f}μm  EFL={ref_efl:.1f}mm  "
                    f"{fnum_str_png}  [{ref_signs}]  "
                    f"stability={sens:.6f}  survived={surv:.0f}%",
                )
            except Exception as e:
                print(f"[elite-render] refined PNG failed {tag}: {e}", flush=True)

        except Exception as e:
            info["refined"] = {"error": str(e)}
            print(f"[elite-refine] LBFGS failed {tag}: {e}", flush=True)

        # ---- 6. Save comprehensive JSON ----
        json_path = self._dir / f"{tag}.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)

    def shutdown(self) -> None:
        self._q.put(None)


class EliteTracker:
    """Lightweight tracker for top-performing designs.

    Maintains a CSV of every design above ``fitness_threshold``, appended
    incrementally.  Renders are offloaded to a background thread.
    """

    COOKE_C = np.array([0.0454, -0.0023, -0.045, 0.0493, 0.0125, -0.0544])
    COOKE_SIGNS = np.sign(COOKE_C)
    COOKE_NORM = float(np.linalg.norm(COOKE_C))

    def __init__(
        self, path: Path, fitness_threshold: float = -0.05,
        cfg: Optional["RunConfig"] = None,
    ) -> None:
        self.path = path
        self.threshold = fitness_threshold
        self._count = 0
        self._best_fitness = -float("inf")
        # Elite renders directory
        self._elite_dir = path.parent / "elite_designs"
        self._elite_dir.mkdir(exist_ok=True)
        # Background render worker
        self._renderer = (
            _EliteRenderWorker(self._elite_dir, cfg) if cfg is not None else None
        )
        # Write CSV header
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "eval", "fitness", "rms_mm", "rms_um", "efl_mm",
                "invalid_frac", "sign_pattern", "sign_match",
                "cosine_sim", "c0", "c1", "c2", "c3", "c5", "c6",
                "g0", "g1", "g2", "g3", "g4", "g5", "d_sensor",
                "m0", "m1",
            ])

    def check(self, r: "EvalResult", evals: int) -> None:
        """Check a result; append to CSV and render if above threshold."""
        if r is None:
            return
        if r.fitness < self.threshold:
            return

        rms = r.metrics.get("rms_spot", abs(r.fitness))
        inv = r.metrics.get("invalid_fraction", 1.0)
        c = r.theta[:6]
        signs = np.sign(c)
        sign_str = "".join("+" if s > 0 else "-" for s in signs)
        sign_match = int((signs == self.COOKE_SIGNS).sum())
        cos_sim = float(np.dot(c, self.COOKE_C) / (
            np.linalg.norm(c) * self.COOKE_NORM + 1e-12
        ))

        self._count += 1
        is_new_best = r.fitness > self._best_fitness
        if is_new_best:
            self._best_fitness = r.fitness
            fnum_val = r.metrics.get("fnum_actual")
            fnum_str = f"F/{fnum_val:.2f} " if fnum_val is not None else ""
            print(
                f"[elite] NEW BEST fitness={r.fitness:.6f} "
                f"RMS={rms*1000:.1f}μm "
                f"signs=[{sign_str}] sim={cos_sim:+.3f} "
                f"EFL={r.metrics.get('efl_actual',0):.1f}mm "
                f"{fnum_str}"
                f"@ {evals} evals",
                flush=True,
            )

        # Append to CSV
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                evals, f"{r.fitness:.6f}",
                f"{rms:.6f}", f"{rms*1000:.1f}",
                f"{r.metrics.get('efl_actual', 0):.2f}",
                f"{inv:.3f}", sign_str, sign_match,
                f"{cos_sim:.4f}",
                *[f"{v:.6f}" for v in r.theta[:6]],
                *[f"{v:.3f}" for v in r.theta[6:12]],
                f"{r.theta[12]:.2f}",
                *[f"{v:.6f}" for v in r.measures],
            ])

        # Submit render to background thread
        if self._renderer is not None:
            tag = f"elite_{self._count:04d}_{evals}k"
            if is_new_best:
                tag = f"BEST_{self._count:04d}_{evals}k"
            from datetime import datetime
            info = {
                "eval": evals,
                "discovered_at": datetime.now().isoformat(timespec="seconds"),
                "elite_index": self._count,
                "is_new_best": is_new_best,
                "fitness": float(r.fitness),
                "rms_mm": float(rms),
                "rms_um": f"{rms*1000:.1f}",
                "efl_mm": f"{r.metrics.get('efl_actual', 0):.2f}",
                "invalid_fraction": float(inv),
                "sign_pattern": sign_str,
                "sign_match": sign_match,
                "cosine_sim": float(cos_sim),
                "theta": [float(v) for v in r.theta],
                "measures": [float(v) for v in r.measures],
                "metrics": {k: float(v) for k, v in r.metrics.items()},
            }
            self._renderer.submit(tag, r.theta, info)

    def shutdown(self) -> None:
        if self._renderer is not None:
            self._renderer.shutdown()

    @property
    def count(self) -> int:
        return self._count

    @property
    def best_fitness(self) -> float:
        return self._best_fitness


class GradientLogger:
    """Append-only logger of raw parent gradients per CMA-MEGA generation.

    Used by the §3.2 gradient-alignment experiment: captures ``∇_θ L`` (via
    fitness, where L = -fitness) and each ``∇_θ M_j`` *before* the
    ``_normalized_gradients`` call so cosine similarity can be computed
    directly on the un-rescaled vectors.
    """

    def __init__(self, path: Path, theta_dim: int, descriptor_dim: int) -> None:
        self.path = path
        self.theta_dim = int(theta_dim)
        self.K = int(descriptor_dim)
        self._iters: List[int] = []
        self._gens: List[int] = []
        self._thetas: List[np.ndarray] = []
        self._grad_fit: List[np.ndarray] = []
        self._grad_m: List[np.ndarray] = []  # each (K, theta_dim)
        self._fitness: List[float] = []
        self._flush_every = 200  # generations between disk flushes

    def log(
        self,
        *,
        evals: int,
        generation: int,
        theta: np.ndarray,
        grads: Dict[str, np.ndarray],
        fitness: float,
    ) -> None:
        self._iters.append(int(evals))
        self._gens.append(int(generation))
        self._thetas.append(np.asarray(theta, dtype=np.float64).copy())
        self._grad_fit.append(
            np.asarray(grads["fitness"], dtype=np.float64).copy()
        )
        gm = np.empty((self.K, self.theta_dim), dtype=np.float64)
        for k in range(self.K):
            gm[k] = np.asarray(grads[f"m{k}"], dtype=np.float64)
        self._grad_m.append(gm)
        self._fitness.append(float(fitness))
        if len(self._iters) % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self._iters:
            return
        np.savez(
            self.path,
            evals=np.asarray(self._iters, dtype=np.int64),
            generations=np.asarray(self._gens, dtype=np.int64),
            thetas=np.asarray(self._thetas, dtype=np.float64),
            grad_fitness=np.asarray(self._grad_fit, dtype=np.float64),
            grad_m=np.asarray(self._grad_m, dtype=np.float64),
            fitness=np.asarray(self._fitness, dtype=np.float64),
        )


class CMAMEGARunner:
    """Lens-agnostic CMA-MEGA loop.

    Imports only the protocol-level types from the new package + the shared
    emitter / checkpoint plumbing.  The only lens-specific knowledge is
    encapsulated inside the ``LensProblem`` instance that gets passed in.
    """

    def __init__(
        self,
        *,
        problem: LensProblem,
        descriptor: Descriptor,
        archive: GridArchive,
        emitter: GradientCoefficientCMA,
        seeder: Seeder,
        outputs: OutputManager,
        cfg: RunConfig,
        pool: Optional[CPUWorkerPool] = None,
        lineage: Optional[LineageLogger] = None,
    ) -> None:
        self.problem = problem
        self.descriptor = descriptor
        self.archive = archive
        self.emitter = emitter
        self.seeder = seeder
        self.outputs = outputs
        self.cfg = cfg
        self.pool = pool
        self.lineage = lineage
        self.alg: AlgorithmCfg = cfg.algorithm
        self.completed = 0
        self.generation = 0
        self.restart_count = 0
        self.wallclock_seconds = 0.0

    # ------------------------------------------------------------------

    def _normalized_gradients(
        self, grads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        bounds = np.asarray(self.cfg.problem.theta_bounds, dtype=np.float64)
        half = 0.5 * (bounds[:, 1] - bounds[:, 0])

        def norm(name: str) -> np.ndarray:
            g = np.asarray(grads[name], dtype=np.float64) * half
            n = np.linalg.norm(g)
            if not np.isfinite(n) or n < 1e-12:
                return np.zeros_like(g)
            return g / n

        out = {"fitness": norm("fitness")}
        for k in range(self.descriptor.dim):
            out[f"m{k}"] = norm(f"m{k}")
        return out

    # ------------------------------------------------------------------

    def run(self, *, resume: bool = False) -> Dict[str, float]:
        K = self.descriptor.dim
        theta_bounds = list(self.cfg.problem.theta_bounds)
        theta_dim = len(theta_bounds)

        if not resume and not self.archive.cells:
            self.seeder.seed_archive(self.problem, self.archive)

        traj_logger = self.outputs.trajectory_logger()
        elite_threshold = self.cfg.io.elite_fitness_threshold
        if elite_threshold is not None:
            elite_tracker = EliteTracker(
                self.outputs.paths.root / "elite_tracker.csv",
                fitness_threshold=elite_threshold,
                cfg=self.cfg,
            )
        else:
            elite_tracker = None
        grad_logger: Optional[GradientLogger] = None
        if bool(getattr(self.cfg.io, "log_gradients", False)):
            grad_logger = GradientLogger(
                path=self.outputs.paths.root / "gradient_log.npz",
                theta_dim=theta_dim,
                descriptor_dim=K,
            )
        eval_budget = int(self.alg.iterations)
        target_completed = self.completed + eval_budget
        new_cell_bonus = float(self.alg.new_cell_bonus)
        scale_bonus = bool(self.alg.scale_bonus_by_fitness)
        bonus_fit_thr = float(self.alg.bonus_fitness_threshold)
        bonus_fit_scale = max(float(self.alg.bonus_fitness_scale), 1e-12)
        cfg_hash = self.outputs.config_hash(self.cfg)
        checkpoint_every = max(0, int(self.cfg.io.checkpoint_every))

        scores_all: List[float] = []
        _no_improve_run = 0
        _no_fitness_improve_run = 0
        _last_best_fitness = float(self.archive.stats()["best_fitness"])
        loop_t0 = time.perf_counter()
        window_t0 = time.perf_counter()
        window_evals = 0

        def _build_ckpt() -> CMAMEGACheckpoint:
            return CMAMEGACheckpoint(
                archive=self.archive,
                emitter_state=self.emitter.get_state(),
                completed_evaluations=int(self.completed),
                generation=int(self.generation),
                restart_count=int(self.restart_count),
                best_fitness=float(self.archive.stats()["best_fitness"]),
                config_hash=cfg_hash,
                timestamp=now_iso(),
                wallclock_seconds=float(self.wallclock_seconds),
            )

        try:
            while self.completed < target_completed:
                # Parent: reject rejected-designs up to 10 retries.
                parent_eval: EvalResult = None
                for _ in range(10):
                    parent_theta = self.archive.sample_parent()
                    parent_eval = self.problem.evaluate(
                        parent_theta, with_grads=True
                    )
                    if parent_eval.metrics.get("rejected", 0.0) <= 0.0:
                        break
                    parent_eval = None
                if parent_eval is None:
                    print(
                        "[lensmega] archive contains only rejected designs; "
                        "aborting CMA-MEGA loop"
                    )
                    break
                assert parent_eval.grads is not None

                if grad_logger is not None:
                    grad_logger.log(
                        evals=int(self.completed),
                        generation=int(self.generation),
                        theta=parent_theta,
                        grads=parent_eval.grads,
                        fitness=float(parent_eval.fitness),
                    )

                unit_parent = theta_to_unit(parent_theta, theta_bounds)
                grads = self._normalized_gradients(parent_eval.grads)

                batch_n = min(
                    int(self.alg.population_size),
                    target_completed - self.completed,
                )
                coeffs = self.emitter.ask(batch_n)

                candidate_thetas = np.empty((batch_n, theta_dim), dtype=np.float64)
                ablate = bool(getattr(self.alg, "ablate_gradients", False))
                for local_idx, coeff in enumerate(coeffs):
                    if ablate:
                        # Ablation: ignore the physics gradients. Sample a
                        # direction in theta-space from a standard Gaussian,
                        # normalise. This makes the emitter equivalent to
                        # Gaussian-mutation MAP-Elites (no Seidel steering).
                        rng = np.random.default_rng(
                            self.alg.cma_seed + self.completed + local_idx
                        )
                        direction = rng.normal(0.0, 1.0, size=(theta_dim,))
                    else:
                        direction = abs(coeff[0]) * grads["fitness"]
                        for k in range(K):
                            direction = direction + coeff[1 + k] * grads[f"m{k}"]
                    dir_norm = np.linalg.norm(direction)
                    if not np.isfinite(dir_norm) or dir_norm < 1e-12:
                        rng = np.random.default_rng(
                            self.alg.cma_seed + self.completed + local_idx
                        )
                        direction = rng.normal(0.0, 1.0, size=(theta_dim,))
                        direction /= np.linalg.norm(direction) + 1e-12
                    candidate_unit = np.clip(
                        unit_parent + float(self.alg.cma_eta) * direction, -1.0, 1.0
                    )
                    candidate_thetas[local_idx] = unit_to_theta(
                        candidate_unit, theta_bounds
                    )

                # Population evaluation: parallel CPU pool if available,
                # else serial in-process fallback.
                if self.pool is not None:
                    pool_results = self.pool.evaluate(candidate_thetas[:batch_n])
                else:
                    pool_results = [
                        self.problem.evaluate(
                            candidate_thetas[i], with_grads=False
                        )
                        for i in range(batch_n)
                    ]

                parent_hash_for_gen = theta_hash(parent_theta)

                batch_scores: List[float] = []
                batch_status: List[int] = []
                n_rejected = 0
                n_rejected_pupil = 0
                n_rejected_bfl = 0
                for r in pool_results:
                    if elite_tracker is not None:
                        elite_tracker.check(r, self.completed)
                    if r is None:
                        # Worker error: penalise hard but don't insert.
                        scores_all.append(-1e9)
                        batch_scores.append(-np.inf)
                        batch_status.append(0)
                        n_rejected += 1
                        continue
                    if r.metrics.get("rejected", 0.0) > 0.0 or not np.isfinite(
                        r.fitness
                    ):
                        status = 0
                        n_rejected += 1
                        if r.metrics.get("rejected_pupil", 0.0) > 0.0:
                            n_rejected_pupil += 1
                        if r.metrics.get("rejected_bfl", 0.0) > 0.0:
                            n_rejected_bfl += 1
                    else:
                        status = int(self.archive.insert(r))
                    scores_all.append(float(r.fitness))
                    if status == INSERT_NEW:
                        if scale_bonus:
                            # Sigmoid-weighted bonus: garbage-fitness new cells
                            # get a tiny fraction of the bonus; high-fitness new
                            # cells get the full bonus. Stops CMA from learning
                            # "any new cell wins" when most novelty is in bad
                            # regions of BC space.
                            z = (float(r.fitness) - bonus_fit_thr) / bonus_fit_scale
                            # Numerically stable sigmoid (avoid overflow on z<<0).
                            if z >= 0:
                                quality = 1.0 / (1.0 + math.exp(-z))
                            else:
                                ez = math.exp(z)
                                quality = ez / (1.0 + ez)
                            batch_scores.append(
                                float(r.fitness) + new_cell_bonus * quality
                            )
                        else:
                            batch_scores.append(float(r.fitness) + new_cell_bonus)
                    elif status == INSERT_REPLACED:
                        batch_scores.append(float(r.fitness))
                    else:
                        batch_scores.append(-np.inf)
                    batch_status.append(status)
                    if self.lineage is not None:
                        self.lineage.log(
                            source="offspring",
                            gen=int(self.generation) + 1,
                            parent_theta_hash=parent_hash_for_gen,
                            result=r,
                            insert_status=int(status),
                            archive=self.archive,
                        )

                self.emitter.tell(
                    coeffs, np.asarray(batch_scores, dtype=np.float64)
                )

                self.completed += batch_n
                self.generation += 1
                window_evals += batch_n

                n_new = sum(1 for s in batch_status if s == INSERT_NEW)
                n_replaced = sum(1 for s in batch_status if s == INSERT_REPLACED)
                elite_n = max(1, int(np.ceil(batch_n * self.emitter.elite_frac)))
                elite_idx = np.argsort(batch_scores)[-elite_n:]
                n_wasted = sum(1 for i in elite_idx if batch_status[i] == 0)

                # Restart logic — require `restart_patience` consecutive
                # no-improvement generations before resetting CMA.
                if (n_new + n_replaced) == 0:
                    _no_improve_run += 1
                else:
                    _no_improve_run = 0

                # Fitness-improvement tracker. Independent of insertion-based
                # no_improve: corner-saturated CMA can keep inserting novel BC
                # cells while best_fitness stays frozen, which is exactly the
                # sigma-runaway equilibrium we want to break.
                current_best = float(self.archive.stats()["best_fitness"])
                if current_best > _last_best_fitness + float(
                    self.alg.fitness_improve_eps
                ):
                    _no_fitness_improve_run = 0
                    _last_best_fitness = current_best
                else:
                    _no_fitness_improve_run += 1

                restart_reason = None
                if _no_improve_run >= int(self.alg.restart_patience):
                    restart_reason = "no_improvement"
                    _no_improve_run = 0
                elif (
                    int(self.alg.fitness_patience) > 0
                    and _no_fitness_improve_run >= int(self.alg.fitness_patience)
                ):
                    restart_reason = "no_fitness_improvement"
                    _no_fitness_improve_run = 0
                else:
                    flag, hansen_reason = self.emitter.should_restart()
                    if flag:
                        restart_reason = hansen_reason or "hansen_term"
                        _no_improve_run = 0
                        _no_fitness_improve_run = 0
                if restart_reason is not None:
                    sigma_pre = float(self.emitter.sigma)
                    self.emitter.reset()
                    self.restart_count += 1
                    _no_fitness_improve_run = 0
                    print(
                        f"[lensmega] restart gen={self.generation} "
                        f"reason={restart_reason} sigma_pre={sigma_pre:.4f} "
                        f"restarts={self.restart_count}",
                        flush=True,
                    )

                self.wallclock_seconds += time.perf_counter() - loop_t0
                loop_t0 = time.perf_counter()

                s = self.archive.stats()
                traj_logger.log(
                    gen=self.generation,
                    evals=self.completed,
                    coverage=float(s["coverage"]),
                    best_fitness=float(s["best_fitness"]),
                    new_cells=n_new,
                    replaced_elites=n_replaced,
                    wasted_elites=n_wasted,
                    rejected=n_rejected,
                    rejected_pupil=n_rejected_pupil,
                    rejected_bfl=n_rejected_bfl,
                    sigma=float(self.emitter.sigma),
                    restarts=self.restart_count,
                    wallclock_s=float(self.wallclock_seconds),
                )

                if self.generation % max(int(self.alg.log_every), 1) == 0:
                    now = time.perf_counter()
                    window_secs = max(1e-9, now - window_t0)
                    evals_per_s = window_evals / window_secs
                    total_wall = self.wallclock_seconds
                    mm, ss = divmod(int(total_wall), 60)
                    hh, mm = divmod(mm, 60)
                    if hh > 0:
                        wall_str = f"{hh}h{mm:02d}m{ss:02d}s"
                    else:
                        wall_str = f"{mm}m{ss:02d}s"
                    if elite_tracker is not None and elite_tracker.count > 0:
                        elite_str = (
                            f"elites={elite_tracker.count} "
                            f"best_fit={elite_tracker.best_fitness:.6f}"
                        )
                    else:
                        elite_str = ""
                    print(
                        f"[lensmega] gen={self.generation} "
                        f"evals={self.completed} coverage={s['coverage']:.4f} "
                        f"best={s['best_fitness']:.6f} new={n_new}/{batch_n} "
                        f"replaced={n_replaced}/{batch_n} "
                        f"rej={n_rejected}/{batch_n}"
                        f"(pupil={n_rejected_pupil},bfl={n_rejected_bfl}) "
                        f"sigma={self.emitter.sigma:.4f} | "
                        f"rate={evals_per_s:6.1f} evals/s "
                        f"wall={wall_str} "
                        f"| {elite_str}",
                        flush=True,
                    )
                    window_t0 = now
                    window_evals = 0

                if checkpoint_every > 0 and self.generation % checkpoint_every == 0:
                    self.outputs.save_checkpoint(_build_ckpt())

                # Periodic archive plot every ~10K evaluations
                if self.completed % 10000 < batch_n:
                    try:
                        self.outputs.save_archive_plot(
                            self.archive,
                            coord_names=self.descriptor.coord_names,
                        )
                        self.outputs.save_archive_plot_filtered(
                            self.archive,
                            coord_names=self.descriptor.coord_names,
                            fitness_threshold=elite_threshold if elite_threshold is not None else -0.05,
                            evals=self.completed,
                        )
                    except Exception:
                        pass
        finally:
            try:
                self.outputs.save_checkpoint(_build_ckpt())
            except Exception as e:  # pragma: no cover
                print(f"[lensmega] final checkpoint failed: {e}")
            try:
                self.outputs.save_archive_pickle(self.archive)
                self.outputs.save_archive_cells(
                    self.archive,
                    descriptor_kind=self.cfg.descriptor.kind,
                    coord_names=self.descriptor.coord_names,
                )
                self.outputs.save_archive_plot(
                    self.archive, coord_names=self.descriptor.coord_names
                )
            except Exception as e:  # pragma: no cover
                print(f"[lensmega] final archive dump failed: {e}")
            # Render a grid of the top geometrically diverse designs.
            try:
                self._render_diverse_grid()
            except Exception as e:
                print(f"[lensmega] diverse grid render failed: {e}")
            if elite_tracker is not None:
                elite_tracker.shutdown()
            if grad_logger is not None:
                try:
                    grad_logger.flush()
                    print(
                        f"[lensmega] gradient log saved: {grad_logger.path} "
                        f"({len(grad_logger._iters)} entries)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[lensmega] gradient log flush failed: {e}")

        s = self.archive.stats()
        s["iterations"] = float(eval_budget)
        s["generations"] = float(self.generation)
        s["restart_count"] = float(self.restart_count)
        s["mean_score"] = (
            float(np.mean(scores_all)) if scores_all else float("nan")
        )
        return s

    def _render_diverse_grid(
        self, top_n: int = 50, pick_k: int = 9, cols: int = 3
    ) -> None:
        """Render a grid of the top geometrically diverse archive designs.

        1. Take the *top_n* designs by fitness.
        2. Normalize theta (curvatures + gaps only) to [0, 1].
        3. Greedy farthest-point sampling to pick *pick_k* diverse designs.
        4. Render in a grid.
        """
        from optical_rl.qd.render_lensmega_archive import (
            ArchiveDesign,
            render_archive_grid_lensmega,
        )
        from optical_rl.qd.render_singlet_archive_designs import PlotTheme
        from ..problem.lens_template import load_lens_template
        from ..problem.paraxial import GeneralizedParaxialSolver

        cells = self.archive.cells
        if len(cells) < 2:
            return

        p = self.cfg.problem

        # 1. Collect all archive designs, sort by fitness.
        all_designs = sorted(
            ((k, r) for k, r in cells.items()),
            key=lambda kr: kr[1].fitness, reverse=True,
        )
        candidates = all_designs[: min(top_n, len(all_designs))]
        if len(candidates) < 2:
            return

        # 2. Normalize geometric theta (curvatures + gaps, exclude flange + materials).
        nc = len(p.free_curvature_indices)
        ng = len(p.free_gap_indices)
        geo_dim = nc + ng
        bounds = np.array(p.theta_bounds[:geo_dim], dtype=np.float64)
        lo, hi = bounds[:, 0], bounds[:, 1]
        span = hi - lo
        span[span < 1e-12] = 1.0

        geo_vecs = np.array([r.theta[:geo_dim] for _, r in candidates])
        geo_norm = (geo_vecs - lo) / span  # (N, geo_dim) in [0, 1]

        # 3. Greedy farthest-point sampling.
        k = min(pick_k, len(candidates))
        selected_idx = [0]  # start with the best design
        min_dists = np.full(len(candidates), np.inf)

        for _ in range(k - 1):
            last = geo_norm[selected_idx[-1]]
            dists_to_last = np.linalg.norm(geo_norm - last, axis=1)
            min_dists = np.minimum(min_dists, dists_to_last)
            # Exclude already selected
            min_dists_copy = min_dists.copy()
            for si in selected_idx:
                min_dists_copy[si] = -1.0
            selected_idx.append(int(np.argmax(min_dists_copy)))

        selected = [candidates[i] for i in selected_idx]
        print(
            f"[lensmega] rendering {len(selected)} diverse designs "
            f"(from top {len(candidates)} by fitness)",
            flush=True,
        )
        for i, (cell, r) in enumerate(selected):
            print(
                f"  #{i+1} fitness={r.fitness:.4f}  "
                f"S2={r.measures[0]:.6f}  S3={r.measures[1]:.6f}",
                flush=True,
            )

        # 4. Render grid.
        template, lens = load_lens_template(
            p.lens_path,
            wavelength_um=p.wavelength_um,
            stop_surface_index=p.stop_surface_index,
            free_curvature_indices=tuple(p.free_curvature_indices),
            device="cpu",
            cemented_interfaces=tuple(p.cemented_interfaces),
        )
        solver = GeneralizedParaxialSolver(template)

        archive_designs = [
            ArchiveDesign(
                cell=cell, theta=r.theta,
                fitness=r.fitness, measures=tuple(r.measures),
            )
            for cell, r in selected
        ]
        titles = [
            f"#{i+1}  f={r.fitness:.3f}"
            for i, (_, r) in enumerate(selected)
        ]

        out_path = self.outputs.paths.root / "diverse_elites.png"
        render_archive_grid_lensmega(
            selected=archive_designs,
            lens_json_path=Path(p.lens_path),
            template=template,
            solver=solver,
            focal_length_target_mm=p.focal_length_target_mm,
            theme=PlotTheme(
                background="white",
                lens_fill="#d0e0f0",
                lens_outline="black",
                ray_colors=["#2196F3", "#4CAF50", "#F44336"],
            ),
            out_path=out_path,
            depth_mm=float(p.object_depth_mm),
            wavelength_um=p.wavelength_um,
            num_rays=3,
            fov_fracs=[0.0, 0.5, 1.0],
            cols=cols,
            dpi=150,
            device="cpu",
            free_curvature_indices=p.free_curvature_indices,
            panel_titles=titles,
            parameterization=p.parameterization,
            free_gap_indices=p.free_gap_indices,
            variable_flange=p.variable_flange,
            variable_materials=p.variable_materials,
            free_material_indices=p.free_material_indices,
        )
        print(f"[lensmega] diverse elites grid saved: {out_path}", flush=True)
