from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


Descriptor2D = Tuple[float, float]
Bounds = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class EvalResult:
    theta: np.ndarray
    fitness: float
    measures: Descriptor2D
    descriptor_vector: np.ndarray
    metrics: Dict[str, float]
    grads: Optional[Dict[str, np.ndarray]] = None


@dataclass(frozen=True)
class RunnerConfig:
    theta_bounds: Bounds
    archive_bins: Tuple[int, int]
    archive_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    invalid_tol: float = 0.05
    invalid_penalty_weight: float = 5.0
    cma_eta: float = 0.05
    seed_count: int = 128
    calibration_count: int = 512
    initial_sigma: float = 0.75
    population_size: int = 16
    cma_seed: int = 0
    log_every: int = 50
    active_descriptor_indices: Tuple[int, int] = (0, 1)
    # Stop-and-go checkpointing — see optical_rl/qd/_checkpoint.py.
    checkpoint_every: int = 50
    # Two-tier improvement-rank bonus for new cells.
    new_cell_bonus: float = 1.0e6


class Problem(Protocol):
    def initial_theta(self) -> np.ndarray:
        ...

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        ...

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        ...

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> EvalResult:
        ...

    def descriptor_names(self) -> Sequence[str]:
        ...
