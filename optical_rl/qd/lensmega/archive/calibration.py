from __future__ import annotations

import numpy as np


def percentile_bounds(
    measures_nd: np.ndarray,
    *,
    lo: float = 2.0,
    hi: float = 98.0,
    pad: float = 0.05,
) -> np.ndarray:
    """Per-axis percentile bounds with a fractional pad, for ND descriptors.

    ``measures_nd`` has shape ``(N, K)``.  Returns shape ``(K, 2)``.  Axes
    where the calibrated spread collapses are left with ``(hi - lo) < 1e-6``;
    the caller (ParallelSeeder) is expected to warn on that.
    """
    arr = np.asarray(measures_nd, dtype=np.float64)
    assert arr.ndim == 2, "expected (N, K) array of measures"
    lo_p = np.percentile(arr, lo, axis=0)
    hi_p = np.percentile(arr, hi, axis=0)
    span = hi_p - lo_p
    margin = pad * np.maximum(span, np.abs(hi_p) * 1e-3 + 1e-12)
    return np.stack([lo_p - margin, hi_p + margin], axis=1)
