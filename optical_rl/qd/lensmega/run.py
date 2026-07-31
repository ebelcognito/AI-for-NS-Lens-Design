from __future__ import annotations

# Pin intra-op thread counts BEFORE importing torch/numpy/deeplens.  Parallel
# CPU evaluation is done at the process level (multiple worker processes);
# each process doing its own 16-wide torch/BLAS threadpool oversubscribes
# cores and wrecks scaling.  Setting these env vars here propagates to every
# spawned worker because they inherit the parent's environment.
import os as _os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)

from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA

from .archive.grid import GridArchive
from .config.loader import load_config
from .config.schema import RunConfig
from .core.cpu_pool import CPUWorkerPool, auto_worker_count
from .core.lineage import LineageLogger
from .core.output_manager import OutputManager
from .core.runner import CMAMEGARunner
from .core.seeder import Seeder
from .descriptors.base import Descriptor
from .descriptors.random_proj import RandomProjectionDescriptor
from .descriptors.seidel import SeidelDescriptor
from .descriptors.zernike import ZernikeDescriptor
from .problem.lens_problem import LensProblem, LensProblemConfig
from .problem.power_budget import PowerBudgetLensProblem
from .problem.full_dof import FullDOFLensProblem


def _build_descriptor(cfg: RunConfig) -> Descriptor:
    kind = cfg.descriptor.kind
    params = dict(cfg.descriptor.params)
    if kind == "zernike":
        return ZernikeDescriptor(
            noll_indices=params["noll_indices"],
            num_rays=int(params.get("num_rays", 512)),
        )
    if kind == "seidel":
        return SeidelDescriptor(seidel_terms=params["terms"])
    if kind == "random":
        return RandomProjectionDescriptor(
            dim_out=int(params.get("dim", 2)),
            seed=int(params.get("seed", 12345)),
            theta_dim=len(cfg.problem.theta_bounds),
            theta_bounds=list(cfg.problem.theta_bounds),
        )
    raise ValueError(f"unknown descriptor kind {kind!r}")


def _build_problem(cfg: RunConfig, descriptor: Descriptor) -> LensProblem:
    p = cfg.problem
    device = "cpu" if cfg.io.device == "auto" else cfg.io.device
    prob_cfg = LensProblemConfig(
        lens_path=p.lens_path,
        wavelength_um=p.wavelength_um,
        fitness_wavelengths_um=tuple(p.fitness_wavelengths_um),
        object_depth_mm=p.object_depth_mm,
        stop_surface_index=p.stop_surface_index,
        free_curvature_indices=tuple(p.free_curvature_indices),
        derived_curvature_index=p.derived_curvature_index,
        focal_length_target_mm=p.focal_length_target_mm,
        theta_bounds=tuple(tuple(b) for b in p.theta_bounds),
        stage=p.stage,
        num_rays=p.num_rays,
        invalid_tol=p.invalid_tol,
        invalid_penalty_weight=p.invalid_penalty_weight,
        intersect_penalty_weight=p.intersect_penalty_weight,
        air_min_mm=p.air_min_mm,
        thick_min_mm=p.thick_min_mm,
        strict_entrance_pupil=p.strict_entrance_pupil,
        descriptor_field_index=p.descriptor_field_index,
        parameterization=p.parameterization,
        efl_penalty_weight=p.efl_penalty_weight,
        free_gap_indices=tuple(p.free_gap_indices),
        variable_flange=p.variable_flange,
        flange_bounds=tuple(p.flange_bounds) if p.flange_bounds else None,
        variable_materials=p.variable_materials,
        free_material_indices=tuple(p.free_material_indices),
        cemented_interfaces=tuple(p.cemented_interfaces),
        field_weights=tuple(p.field_weights) if p.field_weights else None,
        device=device,
        enforce_fnum=p.enforce_fnum,
        f_number_target=p.f_number_target,
        fnum_penalty_weight=p.fnum_penalty_weight,
    )
    if p.parameterization == "full_dof":
        return FullDOFLensProblem(prob_cfg, descriptor)
    if p.parameterization == "power_budget":
        return PowerBudgetLensProblem(prob_cfg, descriptor)
    return LensProblem(prob_cfg, descriptor)


def _build_archive(cfg: RunConfig) -> GridArchive:
    bounds = None
    if cfg.archive.bounds is not None:
        bounds = np.asarray(cfg.archive.bounds, dtype=np.float64)
    return GridArchive(
        dim=len(cfg.archive.bins),
        bins=cfg.archive.bins,
        bounds=bounds,
        seed=cfg.algorithm.cma_seed,
    )


def _build_emitter(cfg: RunConfig, descriptor_dim: int) -> GradientCoefficientCMA:
    return GradientCoefficientCMA(
        dim=1 + descriptor_dim,
        sigma=cfg.algorithm.initial_sigma,
        population_size=cfg.algorithm.population_size,
        seed=cfg.algorithm.cma_seed,
    )


def build_all(cfg: RunConfig) -> Tuple[
    LensProblem, Descriptor, GridArchive, GradientCoefficientCMA, Seeder
]:
    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor)
    archive = _build_archive(cfg)
    emitter = _build_emitter(cfg, descriptor.dim)
    seeder = Seeder(cfg.algorithm, cfg.archive, cfg.io)
    return problem, descriptor, archive, emitter, seeder


def main() -> None:
    p = argparse.ArgumentParser(
        description="Modular CMA-MEGA lens-design runner"
    )
    p.add_argument("--config", required=True, help="path to run YAML")
    p.add_argument(
        "--resume", type=str, default=None, help="path to existing run dir"
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="override algorithm.iterations",
    )
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.iterations is not None:
        cfg.algorithm.iterations = args.iterations

    problem, descriptor, archive, emitter, seeder = build_all(cfg)
    problem._ensure_backend()

    # Spin up the CPU worker pool up-front so seeding, calibration, and the
    # CMA-MEGA loop all share the same workers.  `parallel_workers: 1` (or
    # 0) forces the serial fallback; `null` auto-detects.
    num_workers = auto_worker_count(cfg.io.parallel_workers)
    pool: Optional[CPUWorkerPool] = None
    if num_workers > 1:
        print(
            f"[lensmega] starting CPU worker pool: {num_workers} workers",
            flush=True,
        )
        pool = CPUWorkerPool(
            num_workers=num_workers,
            config_dict=cfg.model_dump(mode="python"),
        )
        seeder.pool = pool

    if args.resume:
        outputs = OutputManager.for_resume(Path(args.resume))
        ckpt = outputs.load_checkpoint()
        archive = ckpt.archive
        emitter = _build_emitter(cfg, descriptor.dim)
        emitter.set_state(ckpt.emitter_state)
        completed = int(ckpt.completed_evaluations)
        generation = int(ckpt.generation)
        restart_count = int(ckpt.restart_count)
        wallclock = float(ckpt.wallclock_seconds)
        lineage = LineageLogger(outputs.paths.lineage)
        seeder.lineage = lineage
    else:
        outputs = OutputManager.for_new_run(cfg.io)
        outputs.stamp(cfg, source_yaml=config_path)
        lineage = LineageLogger(outputs.paths.lineage)
        seeder.lineage = lineage
        if cfg.archive.bounds is None:
            print(
                f"[lensmega] calibrating archive bounds "
                f"(n={cfg.algorithm.calibration_count})",
                flush=True,
            )
            bounds = seeder.calibrate(problem, descriptor.dim)
            archive.set_bounds(bounds)
            print(f"[lensmega] calibrated bounds:\n{bounds}", flush=True)
        completed = 0
        generation = 0
        restart_count = 0
        wallclock = 0.0

    runner = CMAMEGARunner(
        problem=problem,
        descriptor=descriptor,
        archive=archive,
        emitter=emitter,
        seeder=seeder,
        outputs=outputs,
        cfg=cfg,
        pool=pool,
        lineage=lineage,
    )
    runner.completed = completed
    runner.generation = generation
    runner.restart_count = restart_count
    runner.wallclock_seconds = wallclock

    print(
        f"[lensmega] run_dir={outputs.run_dir}",
        flush=True,
    )
    try:
        stats = runner.run(resume=bool(args.resume))
    finally:
        if pool is not None:
            pool.shutdown()
        if lineage is not None:
            lineage.close()
    print(f"[lensmega] final stats: {stats}")


if __name__ == "__main__":
    main()
