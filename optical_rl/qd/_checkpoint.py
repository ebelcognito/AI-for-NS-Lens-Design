"""Stop-and-go checkpoint format shared by all CMA-MEGA runners.

A checkpoint bundles three things into one pickled file:
  1. The full archive (cells dict + bounds + bins + RNG state).
  2. The full CMA-ES emitter state from GradientCoefficientCMA.get_state().
  3. Runner-level counters (completed evaluations, generation index,
     restart count, best fitness, wallclock).

Runners write checkpoints atomically (via .tmp + os.replace) so that
SIGINT mid-write does not leave a half-flushed file. The companion
trajectory.csv is appended one row per generation and survives kills
without needing the checkpoint at all.

Checkpoints are pickled (not JSON) because the archive cells contain
NumPy arrays and runner-specific dataclasses (EF50EvalResult,
BatchedEvalResult, ...) that don't have a clean JSON representation.
"""

from __future__ import annotations

import csv
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


_CHECKPOINT_SCHEMA: int = 1


# ---------------------------------------------------------------------------
# Archive insert status (canonical CMA-MEGA improvement ranking)
# ---------------------------------------------------------------------------
#
# Archive.insert() previously returned a single bool that conflated
# "filled an empty cell" with "beat the existing elite". Canonical
# CMA-MEGA needs to distinguish these two cases so that new cells can
# rank above replaced cells in the improvement-sorted scores fed to
# CMA-ES.tell(). The values are chosen so existing call sites that do
# `if archive.insert(...):` keep working — 0 is falsy, 1 and 2 are
# truthy.
INSERT_NONE: int = 0      # rejected, out-of-bounds, or beaten by current elite
INSERT_REPLACED: int = 1  # replaced an existing elite (positive delta-fitness)
INSERT_NEW: int = 2       # filled a previously-empty cell


@dataclass
class CMAMEGACheckpoint:
    """One serialisable snapshot of a CMA-MEGA run.

    Attributes:
        archive: The full archive object (e.g. EF50Archive2D, BatchedArchive2D,
            TripletArchive2D, ArchiveGrid2D). Pickled directly. The class
            must be importable at resume time, which is true for all
            five archives in optical_rl/qd/.
        emitter_state: The dict from GradientCoefficientCMA.get_state().
        completed_evaluations: How many fitness evaluations have been spent
            in the CMA-MEGA loop (does not include seeding).
        generation: How many CMA-MEGA generations have been run.
        restart_count: How many emitter restarts have fired.
        best_fitness: archive.stats()['best_fitness'] at checkpoint time.
        config_hash: SHA256 of the runner config dict; resume validates
            this to detect drift.
        timestamp: ISO-8601 string for human inspection.
        wallclock_seconds: Total seconds spent in the CMA-MEGA loop.
        schema_version: Bumped on incompatible format changes.
    """

    archive: Any
    emitter_state: Dict[str, Any]
    completed_evaluations: int
    generation: int
    restart_count: int
    best_fitness: float
    config_hash: str
    timestamp: str
    wallclock_seconds: float
    schema_version: int = _CHECKPOINT_SCHEMA
    extra: Dict[str, Any] = field(default_factory=dict)


def save_checkpoint(ckpt: CMAMEGACheckpoint, path: os.PathLike) -> None:
    """Atomically pickle a checkpoint to `path`.

    Writes to `path.tmp` then renames over `path` so a SIGINT mid-write
    cannot corrupt an existing checkpoint.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, p)


def load_checkpoint(path: os.PathLike) -> CMAMEGACheckpoint:
    """Load a checkpoint and validate its schema version."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, CMAMEGACheckpoint):
        raise TypeError(
            f"{path}: expected CMAMEGACheckpoint, got {type(obj).__name__}"
        )
    if obj.schema_version != _CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{path}: schema version {obj.schema_version} is not supported "
            f"by this runtime (v{_CHECKPOINT_SCHEMA})."
        )
    return obj


def hash_config(config_obj: Any) -> str:
    """SHA256 of a config dataclass for drift detection on resume.

    Uses repr(asdict(config_obj)) to be robust against frozen dataclasses
    and tuple/list normalization. Drift detection is best-effort; the
    purpose is to catch obvious mismatches like "ran with EF50, resumed
    with triplet config" rather than to verify byte-for-byte identity.
    """
    import dataclasses
    import hashlib

    if dataclasses.is_dataclass(config_obj):
        as_dict = dataclasses.asdict(config_obj)
    elif isinstance(config_obj, dict):
        as_dict = dict(config_obj)
    else:
        as_dict = {"repr": repr(config_obj)}
    payload = repr(sorted(as_dict.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Per-generation trajectory CSV (free telemetry, written even without ckpt)
# ---------------------------------------------------------------------------

TRAJECTORY_FIELDS = (
    "gen",
    "evals",
    "coverage",
    "best_fitness",
    "new_cells",
    "replaced_elites",
    "wasted_elites",
    "rejected",
    "rejected_pupil",
    "rejected_bfl",
    "sigma",
    "restarts",
    "wallclock_s",
)


class TrajectoryLogger:
    """Append-only per-generation telemetry. One row per generation.

    File is opened in append mode each call so a kill mid-run leaves a
    valid (truncated to last completed line) CSV behind. Header is
    written on the first call only.
    """

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(
        self,
        *,
        gen: int,
        evals: int,
        coverage: float,
        best_fitness: float,
        new_cells: int,
        replaced_elites: int,
        wasted_elites: int,
        rejected: int = 0,
        rejected_pupil: int = 0,
        rejected_bfl: int = 0,
        sigma: float,
        restarts: int,
        wallclock_s: float,
    ) -> None:
        row = {
            "gen": int(gen),
            "evals": int(evals),
            "coverage": float(coverage),
            "best_fitness": float(best_fitness),
            "new_cells": int(new_cells),
            "replaced_elites": int(replaced_elites),
            "wasted_elites": int(wasted_elites),
            "rejected": int(rejected),
            "rejected_pupil": int(rejected_pupil),
            "rejected_bfl": int(rejected_bfl),
            "sigma": float(sigma),
            "restarts": int(restarts),
            "wallclock_s": float(wallclock_s),
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)
