from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..archive.calibration import percentile_bounds
from ..archive.grid import GridArchive
from ..config.schema import AlgorithmCfg, ArchiveCfg, IOCfg
from .cpu_pool import CPUWorkerPool
from .lineage import LineageLogger


class Seeder:
    """Seeding + calibration.  Uses a ``CPUWorkerPool`` when supplied,
    else falls back to single-process serial evaluation."""

    def __init__(
        self,
        alg_cfg: AlgorithmCfg,
        archive_cfg: ArchiveCfg,
        io_cfg: IOCfg,
        pool: Optional[CPUWorkerPool] = None,
        lineage: Optional[LineageLogger] = None,
    ) -> None:
        self.alg = alg_cfg
        self.archive_cfg = archive_cfg
        self.io = io_cfg
        self.pool = pool
        self.lineage = lineage

    # ------------------------------------------------------------------

    def _evaluate_many(self, problem, thetas: np.ndarray) -> List[Any]:
        if self.pool is not None:
            return [r for r in self.pool.evaluate(thetas) if r is not None]
        results = []
        for row in thetas:
            r = problem.evaluate(row, with_grads=False)
            results.append(r)
        return results

    def calibrate(self, problem, descriptor_dim: int) -> np.ndarray:
        n = int(self.alg.calibration_count)
        thetas = problem.sample_theta(n, seed=self.alg.cma_seed + 9001)
        results = self._evaluate_many(problem, thetas)
        if self.lineage is not None:
            for r in results:
                if r is None:
                    continue
                self.lineage.log(
                    source="calibration",
                    gen=0,
                    parent_theta_hash=None,
                    result=r,
                    insert_status=0,
                    archive=None,
                )
        # IMPORTANT: filter out pathological designs before computing BC
        # bounds.  BFL-negative and invalid-ray-heavy designs produce
        # pathological Seidel values (orders of magnitude larger than any
        # real lens), and letting them into the percentile calculation
        # inflates the archive bounds so badly that good lenses collapse
        # into a sub-pixel region of the grid.  See paper §"Behaviour-
        # space calibration must be physics-aware".
        valid = [
            r for r in results
            if r.metrics.get("rejected", 0.0) == 0.0
            and r.metrics.get("rejected_bfl", 0.0) == 0.0
            and r.metrics.get("rejected_pupil", 0.0) == 0.0
            and r.metrics.get("invalid_fraction", 1.0) < 0.2
            and np.isfinite(r.fitness)
            and np.all(np.isfinite(np.asarray(r.measures)))
        ]
        if len(valid) < max(16, n // 8):
            raise RuntimeError(
                f"calibration produced too few valid designs: "
                f"{len(valid)}/{n}.  (Filter requires rejected=0, "
                f"rejected_bfl=0, rejected_pupil=0, invalid_fraction<0.2, "
                f"finite fitness and measures.)"
            )
        measures_nd = np.stack([np.asarray(r.measures) for r in valid])
        bounds = percentile_bounds(
            measures_nd,
            lo=self.archive_cfg.bound_calibration_pct[0],
            hi=self.archive_cfg.bound_calibration_pct[1],
            pad=self.archive_cfg.bound_calibration_pad,
        )
        for k in range(descriptor_dim):
            span = bounds[k, 1] - bounds[k, 0]
            if span < 1e-6:
                print(
                    f"[warn] calibration axis {k} is degenerate: span={span:.3e}"
                )
        return bounds

    def seed_archive(
        self, problem, archive: GridArchive
    ) -> dict:
        n = int(self.alg.seed_count)
        thetas = problem.sample_theta(n, seed=self.alg.cma_seed + 1)
        results = self._evaluate_many(problem, thetas)
        inserted_new = 0
        inserted_rep = 0
        rejected = 0
        rejected_pupil = 0
        rejected_bfl = 0
        for r in results:
            if r.metrics.get("rejected", 0.0) > 0.0:
                rejected += 1
                if r.metrics.get("rejected_pupil", 0.0) > 0.0:
                    rejected_pupil += 1
                if r.metrics.get("rejected_bfl", 0.0) > 0.0:
                    rejected_bfl += 1
                if self.lineage is not None:
                    self.lineage.log(
                        source="seed",
                        gen=0,
                        parent_theta_hash=None,
                        result=r,
                        insert_status=0,
                        archive=archive,
                    )
                continue
            if not np.isfinite(r.fitness):
                rejected += 1
                if self.lineage is not None:
                    self.lineage.log(
                        source="seed",
                        gen=0,
                        parent_theta_hash=None,
                        result=r,
                        insert_status=0,
                        archive=archive,
                    )
                continue
            code = archive.insert(r)
            if code == 2:
                inserted_new += 1
            elif code == 1:
                inserted_rep += 1
            if self.lineage is not None:
                self.lineage.log(
                    source="seed",
                    gen=0,
                    parent_theta_hash=None,
                    result=r,
                    insert_status=int(code),
                    archive=archive,
                )
            if self.io.log_seed_accepts and code > 0:
                print(
                    f"[seed] insert code={code} fitness={r.fitness:.4g} "
                    f"measures={[float(v) for v in r.measures]}"
                )
        pct = 100 * rejected / max(1, len(results))
        pct_pupil = 100 * rejected_pupil / max(1, len(results))
        pct_bfl = 100 * rejected_bfl / max(1, len(results))
        print(
            f"[seed] rejected {rejected}/{len(results)} ({pct:.1f}%) — "
            f"pupil={rejected_pupil} ({pct_pupil:.1f}%), "
            f"bfl={rejected_bfl} ({pct_bfl:.1f}%)",
            flush=True,
        )
        return {
            "seeded": int(n),
            "inserted_new": inserted_new,
            "inserted_replaced": inserted_rep,
            "rejected": rejected,
            "rejected_pupil": rejected_pupil,
            "rejected_bfl": rejected_bfl,
            "coverage_after_seed": archive.stats()["coverage"],
        }
