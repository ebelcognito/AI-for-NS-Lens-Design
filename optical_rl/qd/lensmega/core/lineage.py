"""Per-emission lineage logging for LensMEGA runs.

Writes one JSONL row per evaluation (seed, calibration, or CMA-MEGA
offspring).  The log lets downstream analysis reconstruct the parent →
child chain for every archive elite by matching child's
``parent_theta_hash`` to an earlier row's ``theta_hash``.

Schema (one dict per line)::

    {
      "source":    "seed" | "calibration" | "offspring",
      "gen":       int,           # 0 for seed/calibration
      "emission":  int,           # monotonic counter across the whole run
      "parent_theta_hash": str | null,
      "theta_hash": str,
      "theta":     [float, ...],  # full projected theta
      "fitness":   float,
      "rejected":  bool,
      "insert_status": 0 | 1 | 2,  # 0=none, 1=replaced, 2=new cell
      "bc_cell":   [int, int] | null,
      "measures":  [float, ...],  # descriptor output (e.g. S2, S3)
      "seidel_all": [s1, s2, s3, s4, s5]  # may be absent if descriptor!=seidel
    }
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


_SEIDEL_LABELS = ("S1", "S2", "S3", "S4", "S5")


def theta_hash(theta: np.ndarray) -> str:
    """Stable 16-hex-char content hash of a theta vector (bit-exact)."""
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


class LineageLogger:
    """Append-only JSONL writer. Thread-safe only via single-writer use."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fh = self._path.open("a", buffering=1)  # line-buffered
        self._counter = 0

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_seidel(metrics: dict) -> Optional[list]:
        vals = []
        for lbl in _SEIDEL_LABELS:
            k = f"seidel_{lbl}"
            if k not in metrics:
                return None
            vals.append(float(metrics[k]))
        return vals

    @staticmethod
    def _bc_cell(archive, measures: Sequence[float]) -> Optional[list]:
        """Best-effort projection of a measures vector to the archive cell
        index.  Returns None if the archive has no bounds yet or projection
        fails (e.g. measures out of range)."""
        try:
            idx = archive._to_index(np.asarray(measures, dtype=np.float64))
            if idx is None:
                return None
            return [int(i) for i in idx]
        except Exception:
            return None

    # ------------------------------------------------------------------

    def log(
        self,
        *,
        source: str,
        gen: int,
        parent_theta_hash: Optional[str],
        result,
        insert_status: int,
        archive=None,
    ) -> int:
        """Log one evaluation result.  Returns the emission counter used."""
        if result is None:
            # Worker error — skip; no theta/measures to record.
            return -1
        self._counter += 1
        theta_np = np.asarray(result.theta, dtype=np.float64)
        measures_np = np.asarray(result.measures, dtype=np.float64)
        bc_cell = None
        if (
            archive is not None
            and insert_status > 0
            and measures_np.size == archive.dim
        ):
            bc_cell = self._bc_cell(archive, measures_np)
        row = {
            "source": source,
            "gen": int(gen),
            "emission": int(self._counter),
            "parent_theta_hash": parent_theta_hash,
            "theta_hash": theta_hash(theta_np),
            "theta": [float(v) for v in theta_np],
            "fitness": (
                float(result.fitness) if np.isfinite(result.fitness) else None
            ),
            "rejected": bool(result.metrics.get("rejected", 0.0) > 0.0),
            "insert_status": int(insert_status),
            "bc_cell": bc_cell,
            "measures": [float(v) for v in measures_np],
        }
        seidel_all = self._extract_seidel(result.metrics)
        if seidel_all is not None:
            row["seidel_all"] = seidel_all
        # Additional per-eval scalars useful for downstream analysis.  Kept
        # optional so the log stays valid when a metric is missing.
        for k in ("rms_spot", "invalid_fraction", "efl_actual",
                  "efl_error_pct", "efl_penalty", "bfl", "rejected_bfl",
                  "rejected_pupil"):
            v = result.metrics.get(k)
            if v is not None:
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        self._fh.write(json.dumps(row) + "\n")
        return self._counter

    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "LineageLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
