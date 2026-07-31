from __future__ import annotations

import json
import pickle
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from optical_rl.qd._checkpoint import (
    CMAMEGACheckpoint,
    TrajectoryLogger,
    hash_config,
    load_checkpoint,
    save_checkpoint,
)

from ..archive.grid import GridArchive
from ..config.schema import IOCfg, RunConfig
from ..io.layout import RunPaths


class OutputManager:
    """Owns a single run directory and every file inside it.

    Responsibilities:
      - Construct the timestamped run dir on new runs.
      - Re-point to an existing dir on ``--resume``.
      - Write/read the config copy, resolved config JSON, metadata.
      - Save/load checkpoints atomically.
      - Dump the archive (pickle + JSON cells).
      - Serve a TrajectoryLogger.
    """

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    @classmethod
    def for_new_run(cls, io_cfg: IOCfg) -> "OutputManager":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(io_cfg.results_dir) / f"{io_cfg.run_name}_{ts}"
        root.mkdir(parents=True, exist_ok=True)
        return cls(RunPaths(root=root))

    @classmethod
    def for_resume(cls, run_dir: Path) -> "OutputManager":
        if not run_dir.exists():
            raise FileNotFoundError(f"resume target does not exist: {run_dir}")
        return cls(RunPaths(root=run_dir))

    @property
    def run_dir(self) -> Path:
        return self.paths.root

    def stamp(self, cfg: RunConfig, source_yaml: Optional[Path] = None) -> None:
        """Write config.yaml, config_resolved.json, and metadata.json once at start."""
        if source_yaml is not None and source_yaml.exists():
            shutil.copy2(source_yaml, self.paths.config)
        with self.paths.config_resolved.open("w") as f:
            json.dump(cfg.model_dump(mode="json"), f, indent=2, default=str)
        meta = {
            "host": socket.gethostname(),
            "run_dir": str(self.paths.root),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": cfg.io.run_name,
            "package": "optical_rl.qd.lensmega",
        }
        with self.paths.metadata.open("w") as f:
            json.dump(meta, f, indent=2, default=str)

    def save_checkpoint(self, ckpt: CMAMEGACheckpoint) -> None:
        save_checkpoint(ckpt, self.paths.checkpoint)

    def load_checkpoint(self) -> CMAMEGACheckpoint:
        return load_checkpoint(self.paths.checkpoint)

    def save_archive_pickle(self, archive: GridArchive) -> None:
        with self.paths.archive_pickle.open("wb") as f:
            pickle.dump(archive, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_archive_cells(
        self, archive: GridArchive, *, descriptor_kind: str, coord_names: tuple
    ) -> None:
        cells_out = []
        for idx, r in archive.cells.items():
            cells_out.append(
                {
                    "cell_index": list(idx),
                    "theta": [float(v) for v in r.theta],
                    "fitness": float(r.fitness),
                    "measures": [float(v) for v in r.measures],
                    "metrics": {k: float(v) for k, v in r.metrics.items()},
                }
            )
        payload: Dict[str, Any] = {
            "descriptor_kind": descriptor_kind,
            "descriptor_coord_names": list(coord_names),
            "archive_bins": list(archive.bins),
            "archive_bounds": (
                None
                if archive.bounds is None
                else [[float(a), float(b)] for a, b in archive.bounds.tolist()]
            ),
            "cells": cells_out,
            "stats": archive.stats(),
        }
        with self.paths.archive_cells.open("w") as f:
            json.dump(payload, f, indent=2)

    def save_archive_plot(
        self, archive: GridArchive, *, coord_names: tuple
    ) -> None:
        """Two-panel archive figure: bin heatmap + (m0, m1) scatter.

        Only meaningful for ``archive.dim == 2``.  Higher-dim descriptors
        skip with a warning instead of crashing -- a 3D+ visualization
        belongs in a separate post-processing tool, not in the runner.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except Exception as e:
            print(f"[lensmega] matplotlib unavailable, skipping plot: {e}")
            return

        if archive.dim != 2:
            print(
                f"[lensmega] archive.dim={archive.dim}, skipping archive_map.png "
                f"(only 2D descriptors are plotted by the runner)"
            )
            return

        import numpy as np

        bins_x, bins_y = archive.bins
        neg_grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
        xs: list = []
        ys: list = []
        neg_cs: list = []
        for idx, r in archive.cells.items():
            i, j = idx
            f = float(r.fitness)
            neg_grid[i, j] = -f
            xs.append(float(r.measures[0]))
            ys.append(float(r.measures[1]))
            neg_cs.append(-f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        valid = neg_grid[np.isfinite(neg_grid) & (neg_grid > 0)]
        norm = (
            LogNorm(vmin=float(valid.min()), vmax=float(valid.max()))
            if valid.size > 0
            else None
        )
        im = axes[0].imshow(neg_grid.T, origin="lower", aspect="auto", norm=norm)
        axes[0].set_title("Archive Fitness Heatmap")
        axes[0].set_xlabel(f"{coord_names[0]} bin")
        axes[0].set_ylabel(f"{coord_names[1]} bin")
        fig.colorbar(im, ax=axes[0], shrink=0.85).set_label("-fitness (log)")

        if xs:
            pos_cs = [v for v in neg_cs if v > 0]
            scatter_norm = (
                LogNorm(vmin=min(pos_cs), vmax=max(pos_cs)) if pos_cs else None
            )
            sc = axes[1].scatter(xs, ys, c=neg_cs, s=24, norm=scatter_norm)
            if archive.bounds is not None:
                (x_lo, x_hi), (y_lo, y_hi) = archive.bounds.tolist()
                axes[1].set_xlim(x_lo, x_hi)
                axes[1].set_ylim(y_lo, y_hi)
            fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("-fitness (log)")
        axes[1].set_title("Archive elites")
        axes[1].set_xlabel(coord_names[0])
        axes[1].set_ylabel(coord_names[1])

        fig.tight_layout()
        fig.savefig(self.paths.archive_plot, dpi=120)
        plt.close(fig)
        print(f"[lensmega] archive plot saved: {self.paths.archive_plot}")

    def save_archive_plot_filtered(
        self,
        archive: GridArchive,
        *,
        coord_names: tuple,
        fitness_threshold: float = -0.05,
        evals: int = 0,
    ) -> None:
        """Archive scatter showing only designs above a fitness threshold.

        Saved as ``archive_map_filtered.png`` (overwritten each call) and
        a timestamped copy in ``archive_snapshots/``.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        if archive.dim != 2:
            return

        import numpy as np

        xs, ys, fit_vals = [], [], []
        for _, r in archive.cells.items():
            if r.fitness >= fitness_threshold:
                xs.append(float(r.measures[0]))
                ys.append(float(r.measures[1]))
                fit_vals.append(float(r.fitness))

        fig, ax = plt.subplots(figsize=(8, 7))
        if xs:
            fit_arr = np.array(fit_vals)
            # Color: less negative = better → use raw fitness, higher is better
            sc = ax.scatter(
                xs, ys, c=fit_arr, s=18, cmap="viridis",
                vmin=fitness_threshold, vmax=0.0,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
            cbar.set_label("fitness (higher = better)")
        if archive.bounds is not None:
            (x_lo, x_hi), (y_lo, y_hi) = archive.bounds.tolist()
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(coord_names[0])
        ax.set_ylabel(coord_names[1])
        evals_k = evals // 1000
        ax.set_title(
            f"Designs with fitness > {fitness_threshold}  "
            f"({len(xs)} / {len(archive.cells)} elites)  [{evals_k}k evals]"
        )
        fig.tight_layout()

        path_latest = self.paths.root / "archive_map_filtered.png"
        fig.savefig(path_latest, dpi=150)
        snap_dir = self.paths.root / "archive_snapshots"
        snap_dir.mkdir(exist_ok=True)
        path_snap = snap_dir / f"archive_filtered_{evals_k:04d}k.png"
        fig.savefig(path_snap, dpi=150)
        plt.close(fig)

    def trajectory_logger(self) -> TrajectoryLogger:
        return TrajectoryLogger(self.paths.trajectory)

    @staticmethod
    def config_hash(cfg: RunConfig) -> str:
        return hash_config(cfg.model_dump(mode="json"))
