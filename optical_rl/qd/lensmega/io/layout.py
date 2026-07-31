from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Single source of truth for every file written into a run directory.

    Every writer in the new package goes through one of these paths; that way
    a schema change (rename, add a file) touches exactly one place.
    """

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def config_resolved(self) -> Path:
        return self.root / "config_resolved.json"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.pkl"

    @property
    def archive_pickle(self) -> Path:
        return self.root / "archive.pkl"

    @property
    def archive_cells(self) -> Path:
        return self.root / "archive_cells.json"

    @property
    def archive_plot(self) -> Path:
        return self.root / "archive_map.png"

    @property
    def trajectory(self) -> Path:
        return self.root / "trajectory.csv"

    @property
    def lineage(self) -> Path:
        return self.root / "lineage.jsonl"

    @property
    def log(self) -> Path:
        return self.root / "log.txt"
