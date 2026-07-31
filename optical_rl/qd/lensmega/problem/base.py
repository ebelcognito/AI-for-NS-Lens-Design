from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


@dataclass
class EvalResult:
    theta: np.ndarray
    fitness: float
    measures: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    grads: Optional[Dict[str, np.ndarray]] = None


class Problem(Protocol):
    @property
    def theta_dim(self) -> int: ...

    @property
    def theta_bounds(self) -> np.ndarray: ...

    @property
    def descriptor_dim(self) -> int: ...

    def initial_theta(self) -> np.ndarray: ...

    def sample_theta(self, n: int, seed: int) -> np.ndarray: ...

    def project_theta(self, theta: np.ndarray) -> np.ndarray: ...

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> EvalResult: ...
