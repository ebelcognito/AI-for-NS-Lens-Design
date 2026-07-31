"""Entry point for focal-length constraint experiments.

Usage:
    python -m optical_rl.experiments.focal_length.run_experiment \
        --config optical_rl/experiments/focal_length/configs/triplet_soft_penalty.yaml

The ``experiment`` section in the YAML selects which LensProblem subclass to
use.  If absent, falls back to the standard lensmega ``LensProblem`` (baseline).
"""
from __future__ import annotations

import os

# Pin intra-op thread counts BEFORE importing torch/numpy.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

from optical_rl.qd.singlet_cmamega import GradientCoefficientCMA

from optical_rl.qd.lensmega.archive.grid import GridArchive
from optical_rl.qd.lensmega.config.schema import RunConfig
from optical_rl.qd.lensmega.core.cpu_pool import CPUWorkerPool, auto_worker_count
from optical_rl.qd.lensmega.core.output_manager import OutputManager
from optical_rl.qd.lensmega.core.runner import CMAMEGARunner
from optical_rl.qd.lensmega.core.seeder import Seeder
from optical_rl.qd.lensmega.descriptors.base import Descriptor
from optical_rl.qd.lensmega.descriptors.zernike import ZernikeDescriptor
from optical_rl.qd.lensmega.descriptors.seidel import SeidelDescriptor
from optical_rl.qd.lensmega.problem.lens_problem import LensProblem, LensProblemConfig

from .exp_a_mid_solve import MidSolveLensProblem
from .exp_b_soft_penalty import SoftPenaltyLensProblem
from .exp_c_image_height import ImageHeightLensProblem
from .exp_d_power_budget import PowerBudgetLensProblem


def _load_config_with_experiment(config_path: Path):
    """Load YAML, extract the custom ``experiment`` section, then parse
    the rest as a standard ``RunConfig``.
    """
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    experiment_cfg = raw.pop("experiment", None)
    run_cfg = RunConfig.model_validate(raw)
    return run_cfg, experiment_cfg


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
    raise ValueError(f"unknown descriptor kind {kind!r}")


def _build_problem(
    cfg: RunConfig,
    descriptor: Descriptor,
    experiment_cfg: Optional[dict],
) -> LensProblem:
    """Build the appropriate LensProblem subclass based on experiment type."""
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
        device=device,
    )

    if experiment_cfg is None:
        # Baseline: standard LensProblem.
        return LensProblem(prob_cfg, descriptor)

    exp_type = experiment_cfg.get("type", "baseline")

    if exp_type == "baseline":
        return LensProblem(prob_cfg, descriptor)

    elif exp_type == "mid_solve":
        mid_idx = experiment_cfg["mid_solve_surface_index"]
        efl_target = experiment_cfg["efl_target_mm"]
        return MidSolveLensProblem(
            prob_cfg, descriptor,
            mid_solve_index=mid_idx,
            efl_target_mm=efl_target,
        )

    elif exp_type == "soft_penalty":
        return SoftPenaltyLensProblem(
            prob_cfg, descriptor,
            efl_target_mm=experiment_cfg["efl_target_mm"],
            efl_penalty_weight=experiment_cfg.get("efl_penalty_weight", 10.0),
        )

    elif exp_type == "image_height":
        return ImageHeightLensProblem(
            prob_cfg, descriptor,
            efl_target_mm=experiment_cfg["efl_target_mm"],
            focal_loss_weight=experiment_cfg.get("focal_loss_weight", 5.0),
            efl_penalty_weight=experiment_cfg.get("efl_penalty_weight", 1.0),
        )

    elif exp_type == "power_budget":
        return PowerBudgetLensProblem(
            prob_cfg, descriptor,
            efl_target_mm=experiment_cfg["efl_target_mm"],
            efl_penalty_weight=experiment_cfg.get("efl_penalty_weight", 5.0),
        )

    else:
        raise ValueError(f"unknown experiment type: {exp_type!r}")


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Focal-length constraint experiments for LensMEGA"
    )
    p.add_argument("--config", required=True, help="path to experiment YAML")
    p.add_argument(
        "--iterations", type=int, default=None,
        help="override algorithm.iterations",
    )
    args = p.parse_args()

    config_path = Path(args.config)
    cfg, experiment_cfg = _load_config_with_experiment(config_path)

    if args.iterations is not None:
        cfg.algorithm.iterations = args.iterations

    exp_type = (experiment_cfg or {}).get("type", "baseline")
    print(f"[focal_exp] experiment type: {exp_type}", flush=True)

    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor, experiment_cfg)
    archive = _build_archive(cfg)
    emitter = _build_emitter(cfg, descriptor.dim)
    seeder = Seeder(cfg.algorithm, cfg.archive, cfg.io)

    problem._ensure_backend()

    num_workers = auto_worker_count(cfg.io.parallel_workers)
    pool: Optional[CPUWorkerPool] = None
    if num_workers > 1:
        print(
            f"[focal_exp] starting CPU worker pool: {num_workers} workers",
            flush=True,
        )
        pool = CPUWorkerPool(
            num_workers=num_workers,
            config_dict=cfg.model_dump(mode="python"),
        )
        seeder.pool = pool

    outputs = OutputManager.for_new_run(cfg.io)
    outputs.stamp(cfg, source_yaml=config_path)

    if cfg.archive.bounds is None:
        print(
            f"[focal_exp] calibrating archive bounds "
            f"(n={cfg.algorithm.calibration_count})",
            flush=True,
        )
        bounds = seeder.calibrate(problem, descriptor.dim)
        archive.set_bounds(bounds)
        print(f"[focal_exp] calibrated bounds:\n{bounds}", flush=True)

    runner = CMAMEGARunner(
        problem=problem,
        descriptor=descriptor,
        archive=archive,
        emitter=emitter,
        seeder=seeder,
        outputs=outputs,
        cfg=cfg,
        pool=pool,
    )

    print(f"[focal_exp] run_dir={outputs.run_dir}", flush=True)
    try:
        stats = runner.run()
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"[focal_exp] final stats: {stats}")


if __name__ == "__main__":
    main()
