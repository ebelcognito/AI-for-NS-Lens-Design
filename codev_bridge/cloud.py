"""Cloud-side queue operations.  This module has no third-party dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .protocol import canonical_json_bytes, make_job, sha256_file, utc_now_text


QUEUE_DIRS = (
    "jobs/pending",
    "jobs/claimed",
    "results/ready",
    "archive",
    "rejected",
)


def initialize_root(root: Path) -> None:
    """Create a development queue tree.

    Production ownership and modes should be created with the reviewed sudo
    commands in README.md; this helper never elevates privileges.
    """

    for relative in QUEUE_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)


def submit_job(
    root: Path,
    *,
    case_id: str,
    case_file: Path,
    codev_release: str = "2026",
    captures: list[str] | None = None,
    timeout_seconds: int = 300,
    max_result_bytes: int = 256 * 1024,
    max_artifact_bytes: int = 16 * 1024 * 1024,
) -> tuple[dict[str, Any], Path]:
    """Publish one job by same-directory atomic rename."""

    pending = root / "jobs" / "pending"
    if not pending.is_dir():
        raise FileNotFoundError(
            f"queue directory does not exist: {pending}; initialize the bridge first"
        )
    if not case_file.is_file():
        raise FileNotFoundError(f"staged case file does not exist: {case_file}")

    job = make_job(
        job_id=str(uuid4()),
        created_at=utc_now_text(),
        case_id=case_id,
        case_sha256=sha256_file(str(case_file)),
        codev_release=codev_release,
        captures=captures
        or ["error_log", "lens_data", "global_coordinates", "trace_summary"],
        timeout_seconds=timeout_seconds,
        max_result_bytes=max_result_bytes,
        max_artifact_bytes=max_artifact_bytes,
    )

    final_path = pending / f"{job['job_id']}.json"
    temp_path = pending / f".{job['job_id']}.json.part"
    _exclusive_atomic_write(temp_path, final_path, canonical_json_bytes(job))
    return job, final_path


def job_status(root: Path, job_id: str) -> dict[str, Any]:
    result_path = root / "results" / "ready" / f"{job_id}.result.json"
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as stream:
            return {"state": "result_ready", "result": json.load(stream)}

    candidates = (
        ("pending", root / "jobs" / "pending" / f"{job_id}.json"),
        ("claimed", root / "jobs" / "claimed" / f"{job_id}.json"),
        ("archived", root / "archive" / f"{job_id}.json"),
        ("rejected", root / "rejected" / f"{job_id}.json"),
    )
    for state, path in candidates:
        if path.is_file():
            return {"state": state, "path": str(path)}
    return {"state": "unknown"}


def _exclusive_atomic_write(temp_path: Path, final_path: Path, data: bytes) -> None:
    descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

