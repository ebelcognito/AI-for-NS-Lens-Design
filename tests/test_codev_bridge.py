from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from uuid import uuid4

from codev_bridge.adapter import MockAdapter
from codev_bridge.cloud import initialize_root, job_status, submit_job
from codev_bridge.protocol import (
    ProtocolError,
    canonical_json_bytes,
    decode_job,
    make_job,
    utc_now_text,
)
from codev_bridge.worker import Worker


def valid_job() -> dict:
    return make_job(
        job_id=str(uuid4()),
        created_at=utc_now_text(),
        case_id="tier_a_v2c",
        case_sha256="a" * 64,
        codev_release="2026",
        captures=["error_log", "global_coordinates"],
        timeout_seconds=300,
        max_result_bytes=262144,
        max_artifact_bytes=16777216,
    )


class CodeVBridgeTests(unittest.TestCase):
    def test_job_round_trip_is_strict_and_canonical(self) -> None:
        job = valid_job()
        self.assertEqual(decode_job(canonical_json_bytes(job)), job)

        injected = deepcopy(job)
        injected["command"] = "powershell.exe"
        with self.assertRaisesRegex(ProtocolError, "unknown fields: command"):
            decode_job(canonical_json_bytes(injected))

    def test_dangerous_or_unbounded_job_values_are_rejected(self) -> None:
        cases = [
            (("operation",), "shell", "not allowlisted"),
            (("prescription", "case_id"), "..\\evil.seq", "invalid format"),
            (
                ("environment", "captures"),
                ["error_log", "run_command"],
                "non-allowlisted",
            ),
            (("limits", "timeout_seconds"), 99999, "must be between"),
        ]
        for path, value, message in cases:
            with self.subTest(path=path):
                job = valid_job()
                target = job
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                with self.assertRaisesRegex(ProtocolError, message):
                    decode_job(canonical_json_bytes(job))

    def test_submit_uses_final_name_and_status_finds_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            root = tmp_path / "queue"
            initialize_root(root)
            case_file = tmp_path / "benchmark.seq"
            case_file.write_text("trusted staged case\n", encoding="utf-8")

            job, final_path = submit_job(
                root, case_id="tier_a_v2c", case_file=case_file
            )

            self.assertEqual(final_path.name, f"{job['job_id']}.json")
            self.assertFalse(list(final_path.parent.glob("*.part")))
            self.assertEqual(
                json.loads(final_path.read_text(encoding="utf-8")), job
            )
            self.assertEqual(job_status(root, job["job_id"])["state"], "pending")

    def test_mock_adapter_never_executes_a_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = MockAdapter().run(
                valid_job(), Path(temporary), Path(temporary) / "staged.seq"
            )
        self.assertIn("CODE V was not started", payload["summary"]["message"])
        self.assertEqual(payload["artifacts"], [])

    def test_worker_claims_runs_and_archives_a_mock_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            queue = base / "remote" / "bridge"
            initialize_root(queue)
            case_file = base / "benchmark.seq"
            case_file.write_text("trusted staged case\n", encoding="utf-8")
            job, _ = submit_job(
                queue, case_id="tier_a_v2c", case_file=case_file
            )
            digest = job["prescription"]["case_sha256"]
            config = _worker_config(base, case_file, digest)
            worker = Worker(config)

            processed = worker.run_cycle(FakeSFTP(base / "remote"))

            self.assertEqual(processed, 1)
            self.assertFalse((queue / "jobs" / "pending" / f"{job['job_id']}.json").exists())
            self.assertTrue((queue / "archive" / f"{job['job_id']}.json").is_file())
            result_path = queue / "results" / "ready" / f"{job['job_id']}.result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "succeeded")
            self.assertIn("connectivity test", result["summary"]["message"])

    def test_worker_publishes_artifacts_before_result_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            queue = base / "remote" / "bridge"
            initialize_root(queue)
            case_file = base / "benchmark.seq"
            case_file.write_text("trusted staged case\n", encoding="utf-8")
            job, _ = submit_job(
                queue, case_id="tier_a_v2c", case_file=case_file
            )
            worker = Worker(
                _worker_config(
                    base, case_file, job["prescription"]["case_sha256"]
                )
            )
            worker.adapter = ArtifactAdapter()

            self.assertEqual(worker.run_cycle(FakeSFTP(base / "remote")), 1)

            artifact_path = (
                queue
                / "results"
                / "ready"
                / f"{job['job_id']}.artifacts"
                / "error-log.txt"
            )
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), "no errors\n")
            result_path = queue / "results" / "ready" / f"{job['job_id']}.result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["artifacts"][0]["name"], "error-log.txt")
            self.assertEqual(result["artifacts"][0]["size_bytes"], 10)
def _worker_config(base: Path, case_file: Path, digest: str) -> dict:
    return {
        "worker_id": "test-worker",
        "server": {
            "host": "100.64.0.1",
            "port": 22,
            "username": "codevbridge",
            "private_key_file": str(base / "unused-key"),
            "known_hosts_file": str(base / "unused-known-hosts"),
        },
        "remote_root": "/bridge",
        "poll_seconds": 5,
        "max_job_bytes": 65536,
        "max_job_age_seconds": 86400,
        "max_timeout_seconds": 600,
        "max_result_bytes": 262144,
        "max_artifact_bytes": 16777216,
        "local_state_directory": str(base / "state"),
        "allowed_cases": {
            "tier_a_v2c": {"path": str(case_file), "sha256": digest}
        },
        "adapter": {"type": "mock"},
    }


class FakeSFTP:
    """Filesystem-backed subset of Paramiko SFTP used by the queue integration test."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    def listdir(self, remote: str) -> list[str]:
        return os.listdir(self._path(remote))

    def stat(self, remote: str) -> SimpleNamespace:
        value = self._path(remote).stat()
        return SimpleNamespace(st_size=value.st_size, st_mode=value.st_mode)

    def open(self, remote: str, mode: str):
        return self._path(remote).open(mode)

    def posix_rename(self, source: str, destination: str) -> None:
        os.replace(self._path(source), self._path(destination))

    def rename(self, source: str, destination: str) -> None:
        os.rename(self._path(source), self._path(destination))

    def mkdir(self, remote: str, mode: int = 0o777) -> None:
        self._path(remote).mkdir(mode=mode)


class ArtifactAdapter:
    def run(self, job: dict, work_dir: Path, case_file: Path) -> dict:
        artifact_dir = work_dir / "artifacts"
        artifact_dir.mkdir()
        (artifact_dir / "error-log.txt").write_text("no errors\n", encoding="utf-8")
        return {
            "summary": {"loaded_without_errors": True},
            "artifacts": [
                {"name": "error-log.txt", "media_type": "text/plain"}
            ],
        }


if __name__ == "__main__":
    unittest.main()
