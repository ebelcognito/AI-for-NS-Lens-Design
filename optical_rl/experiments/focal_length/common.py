"""Shared utilities for focal-length constraint experiments.

All experiments import from the existing lensmega infrastructure (read-only)
and define custom ``LensProblem`` subclasses that override ``_build_lens``
and/or ``_fitness_and_metrics``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from optical_rl.qd.lensmega.problem.lens_problem import LensProblem, LensProblemConfig
from optical_rl.qd.lensmega.problem.lens_template import LensTemplate, load_lens_template
from optical_rl.qd.lensmega.problem.paraxial import GeneralizedParaxialSolver
from optical_rl.qd.lensmega.problem.penalties import intersection_penalty, invalid_fraction_penalty
from optical_rl.qd.lensmega.problem.tracing import (
    disk_samples,
    latin_hypercube,
    project_box,
    rms_spot_batch,
    trace_fields_batch,
)
from optical_rl.qd.lensmega.problem.base import EvalResult


@dataclass
class ExperimentConfig:
    """Extra config fields for focal-length experiments, beyond the base RunConfig."""
    experiment_type: str = "baseline"
    efl_penalty_weight: float = 10.0
    focal_loss_weight: float = 5.0
    focal_length_target_mm: float = 50.0
    mid_solve_surface_index: Optional[int] = None


def rms_spot_batch_with_centroids(ray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``rms_spot_batch`` but also returns per-field centroids.

    Returns ``(rms, invalid_fraction, centroids)`` where centroids has shape
    ``(F, 2)`` — the (x, y) centroid of the valid rays at each field point.
    """
    valid = ray.is_valid
    xy = ray.o[..., :2]
    denom = valid.sum(dim=-1).clamp_min(1e-8)
    centroid = (xy * valid.unsqueeze(-1)).sum(dim=-2) / denom.unsqueeze(-1)
    err2 = ((xy - centroid.unsqueeze(-2)) ** 2).sum(dim=-1)
    rms = ((err2 * valid).sum(dim=-1) / denom + 1e-12).sqrt()
    inv_frac = 1.0 - valid.sum(dim=-1) / valid.shape[-1]
    return rms, inv_frac, centroid
