"""Feature contract and Torch model for fixed-environment feasibility/quality."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..engine.signatures import parse_target

FEATURE_NAMES = (
    "cx_norm", "thick_norm", "fsag_norm", "bsag_norm", "tilt_a_norm",
    "tilt_b_norm", "ray_y_scaled", "angle_scaled", "index_scaled",
    "first_fold_is_bottom", "return_at_a_is_tir",
)


def feature_vector(environment: Mapping[str, Any], genotype: Sequence[float]) -> np.ndarray:
    """Encode geometry plus compositional path/environment context in 11 numbers."""
    bounds = np.asarray(environment["parameter_bounds"], dtype=np.float32)
    geometry = np.asarray(genotype, dtype=np.float32)
    if bounds.shape != (6, 2) or geometry.shape != (6,):
        raise ValueError("fixed surrogate expects six geometry values and six bounds")
    normalized = (geometry - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    target = parse_target(environment["intended_sequence"].replace("->", " "))
    if len(target) < 4:
        raise ValueError("intended sequence is too short for fixed-fold encoding")
    first_fold_is_bottom = 1.0 if target[1][1] == "D" else -1.0
    return_events = [interaction for interaction, face in target if face == "A"]
    return_at_a_is_tir = 1.0 if "tir" in return_events else -1.0
    fixed = environment["fixed"]
    context = np.asarray([
        float(fixed["ray_y"]) / 25.0,
        float(fixed["ang"]) / 60.0,
        (float(fixed["n"]) - 1.5) / 0.5,
        first_fold_is_bottom,
        return_at_a_is_tir,
    ], dtype=np.float32)
    return np.concatenate([normalized, context]).astype(np.float32)


def build_model(hidden: int = 64):
    """Construct the optional-Torch two-head surrogate."""
    import torch.nn as nn

    class FixedSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(len(FEATURE_NAMES), hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden // 2), nn.SiLU(),
            )
            self.feasibility = nn.Linear(hidden // 2, 1)
            self.quality = nn.Linear(hidden // 2, 1)

        def forward(self, x):
            latent = self.trunk(x)
            return self.feasibility(latent).squeeze(-1), \
                self.quality(latent).squeeze(-1).sigmoid()

    return FixedSurrogate()


def load_checkpoint(path, *, device="cpu"):
    """Load a checkpoint written by ``train_fixed_surrogate``."""
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(int(payload["config"]["hidden"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = ["FEATURE_NAMES", "build_model", "feature_vector", "load_checkpoint"]
