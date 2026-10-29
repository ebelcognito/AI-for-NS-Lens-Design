"""Persistent outbound-only SFTP polling worker for the Windows CODE V machine."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any, Iterator, Mapping

from .adapter import AdapterError, make_adapter
from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    canonical_json_bytes,
    decode_job,
    parse_utc_timestamp,
    sha256_file,
    utc_now_text,
)


LOG = logging.getLogger("codev_bridge.worker")
_JOB_FILE = re.compile(
    r"^(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkerConfigError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerConfigError(f"cannot read worker config {path}: {exc}") from exc
    expected = {
        "worker_id",
        "server",
        "remote_root",
        "poll_seconds",
        "max_job_bytes",
        "max_job_age_seconds",
        "max_timeout_seconds",
        "max_result_bytes",
        "max_artifact_bytes",
        "local_state_directory",
        "allowed_cases",
        "adapter",
    }
    _exact_keys(config, expected, "worker config")

    if not isinstance(config["worker_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", config["worker_id"]
    ):
        raise WorkerConfigError("worker_id has an invalid format")
    server = config["server"]
    if not isinstance(server, dict):
        raise WorkerConfigError("server must be an object")
    _exact_keys(
        server,
        {"host", "port", "username", "private_key_file", "known_hosts_file"},
        "server",
    )
    for key in ("host", "username", "private_key_file", "known_hosts_file"):
        if not isinstance(server[key], str) or not server[key]:
            raise WorkerConfigError(f"server.{key} must be a non-empty string")
    _integer_range(server["port"], "server.port", 1, 65535)
    if not Path(server["private_key_file"]).is_absolute():
        raise WorkerConfigError("server.private_key_file must be absolute")
    if not Path(server["known_hosts_file"]).is_absolute():
        raise WorkerConfigError("server.known_hosts_file must be absolute")

    remote_root = config["remote_root"]
    if not isinstance(remote_root, str) or not remote_root.startswith("/"):
        raise WorkerConfigError("remote_root must be an absolute SFTP path")
    if ".." in PurePosixPath(remote_root).parts:
        raise WorkerConfigError("remote_root may not contain '..'")

    _number_range(config["poll_seconds"], "poll_seconds", 1, 300)
    _integer_range(config["max_job_bytes"], "max_job_bytes", 1024, 1024 * 1024)
    _integer_range(
        config["max_job_age_seconds"], "max_job_age_seconds", 60, 30 * 86400
    )
    _integer_range(config["max_timeout_seconds"], "max_timeout_seconds", 1, 1800)
    _integer_range(
        config["max_result_bytes"], "max_result_bytes", 1024, 4 * 1024 * 1024
    )
    _integer_range(
        config["max_artifact_bytes"],
        "max_artifact_bytes",
        0,
        64 * 1024 * 1024,
    )
    state_dir = Path(config["local_state_directory"])
    if not state_dir.is_absolute():
        raise WorkerConfigError("local_state_directory must be absolute")

    allowed_cases = config["allowed_cases"]
    if not isinstance(allowed_cases, dict) or not allowed_cases:
        raise WorkerConfigError("allowed_cases must be a non-empty object")
    for case_id, case in allowed_cases.items():
        if not isinstance(case_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{0,63}", case_id
        ):
            raise WorkerConfigError(f"invalid allowed case ID: {case_id!r}")
        if not isinstance(case, dict):
            raise WorkerConfigError(f"allowed_cases[{case_id!r}] must be an object")
        _exact_keys(case, {"path", "sha256"}, f"allowed_cases[{case_id!r}]")
        if not isinstance(case["path"], str) or not Path(case["path"]).is_absolute():
            raise WorkerConfigError(
                f"allowed_cases[{case_id!r}].path must be an absolute local path"
            )
        if not Path(case["path"]).is_file():
            raise WorkerConfigError(
                f"allowed_cases[{case_id!r}].path does not exist: {case['path']}"
            )
        digest = case["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise WorkerConfigError(
                f"allowed_cases[{case_id!r}].sha256 must be a lowercase SHA-256 digest"
            )

    if not isinstance(config["adapter"], dict):
        raise WorkerConfigError("adapter must be an object")
    # Construct once during validation so unsafe command templates fail at startup.
    make_adapter(config["adapter"])
    return config


class Worker:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.adapter = make_adapter(config["adapter"])
        self.remote_root = PurePosixPath(config["remote_root"])
        self.state_dir = Path(config["local_state_directory"])
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run_forever(self) -> None:
        LOG.info("worker %s started", self.config["worker_id"])
        while True:
            try:
                with self._connect() as sftp:
                    while True:
                        processed = self.run_cycle(sftp)
                        if not processed:
                            time.sleep(self.config["poll_seconds"])
            except KeyboardInterrupt:
                LOG.info("worker stopped")
                return
            except Exception:
                LOG.exception("SFTP cycle failed; reconnecting after poll interval")
                time.sleep(self.config["poll_seconds"])

    def run_once(self) -> int:
        with self._connect() as sftp:
            return self.run_cycle(sftp)

    def run_cycle(self, sftp: Any) -> int:
        processed = 0
        # Recover jobs left claimed after a worker or connection interruption before
        # claiming new work.  CODE V evaluations must therefore be side-effect free.
        for name in self._job_names(sftp, self._remote("jobs", "claimed")):
            self._process_claimed(sftp, name)
            processed += 1
        for name in self._job_names(sftp, self._remote("jobs", "pending")):
            source = self._remote("jobs", "pending", name)
            claimed = self._remote("jobs", "claimed", name)
            try:
                _rename(sftp, source, claimed)
            except OSError:
                continue
            self._process_claimed(sftp, name)
            processed += 1
        return processed

    def _process_claimed(self, sftp: Any, name: str) -> None:
        match = _JOB_FILE.fullmatch(name)
        if not match:
            return
        filename_job_id = match.group("id")
        claimed = self._remote("jobs", "claimed", name)
        result_path = self._remote(
            "results", "ready", f"{filename_job_id}.result.json"
        )
        if _exists(sftp, result_path):
            self._finish_job(sftp, claimed, filename_job_id, rejected=False)
            shutil.rmtree(
                self.state_dir / "outgoing" / filename_job_id, ignore_errors=True
            )
            return

        rejected = False
        operation = "unknown"
        result_limit = int(self.config["max_result_bytes"])
        outgoing_directory: Path | None = None
        try:
            data = _read_remote_limited(
                sftp, claimed, int(self.config["max_job_bytes"])
            )
            job = decode_job(data, max_bytes=int(self.config["max_job_bytes"]))
            operation = job["operation"]
            if job["job_id"] != filename_job_id:
                raise ProtocolError("job_id does not match its queue filename")
            self._enforce_local_policy(job)
            result_limit = min(result_limit, job["limits"]["max_result_bytes"])
            payload, artifacts = self._execute(job)
            outgoing_directory = self.state_dir / "outgoing" / job["job_id"]
            result = self._make_result(job, "succeeded", payload, artifacts=artifacts)
            if len(canonical_json_bytes(result)) > result_limit:
                raise AdapterError("trusted CODE V adapter result exceeded the result limit")
            self._publish_artifacts(sftp, job["job_id"], artifacts)
        except (ProtocolError, WorkerConfigError) as exc:
            rejected = True
            result = self._make_failure_result(
                filename_job_id, operation, "rejected", str(exc)
            )
        except (AdapterError, OSError, TypeError, ValueError) as exc:
            result = self._make_failure_result(
                filename_job_id, operation, "failed", str(exc)
            )

        result_data = canonical_json_bytes(result)
        if len(result_data) > result_limit:
            result = self._make_failure_result(
                filename_job_id,
                operation,
                "failed",
                "worker result exceeded its local max_result_bytes policy",
            )
            result_data = canonical_json_bytes(result)
        _upload_atomic(sftp, result_path, result_data)
        self._finish_job(sftp, claimed, filename_job_id, rejected=rejected)
        if outgoing_directory is not None:
            shutil.rmtree(outgoing_directory, ignore_errors=True)
        LOG.info("job %s -> %s", filename_job_id, result["status"])

    def _enforce_local_policy(self, job: Mapping[str, Any]) -> None:
        created_at = parse_utc_timestamp(job["created_at"])
        age = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age < -300:
            raise ProtocolError("job timestamp is more than five minutes in the future")
        if age > int(self.config["max_job_age_seconds"]):
            raise ProtocolError("job is older than the worker's max_job_age_seconds")

        case_id = job["prescription"]["case_id"]
        approved_case = self.config["allowed_cases"].get(case_id)
        if approved_case is None:
            raise ProtocolError(f"case_id is not in the local allowlist: {case_id}")
        expected_digest = approved_case["sha256"]
        if job["prescription"]["case_sha256"] != expected_digest:
            raise ProtocolError(
                "case_sha256 does not match the locally approved case revision"
            )
        if sha256_file(approved_case["path"]) != expected_digest:
            raise ProtocolError(
                "the locally staged case file no longer matches its approved SHA-256"
            )
        limits = job["limits"]
        if limits["timeout_seconds"] > int(self.config["max_timeout_seconds"]):
            raise ProtocolError("job timeout exceeds local worker policy")
        if limits["max_result_bytes"] > int(self.config["max_result_bytes"]):
            raise ProtocolError("job max_result_bytes exceeds local worker policy")
        if limits["max_artifact_bytes"] > int(self.config["max_artifact_bytes"]):
            raise ProtocolError("job max_artifact_bytes exceeds local worker policy")

    def _execute(
        self, job: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        job_work_dir = Path(
            tempfile.mkdtemp(prefix=f"job-{job['job_id']}-", dir=self.state_dir)
        )
        try:
            approved_case = self.config["allowed_cases"][job["prescription"]["case_id"]]
            adapter_payload = self.adapter.run(
                job, job_work_dir, Path(approved_case["path"])
            )
            artifact_records = self._collect_artifacts(job, adapter_payload, job_work_dir)
            summary = adapter_payload["summary"]
            # Ensure the adapter didn't smuggle a non-JSON object into the result.
            json.dumps(summary, allow_nan=False)
            return summary, artifact_records
        finally:
            shutil.rmtree(job_work_dir, ignore_errors=True)

    def _collect_artifacts(
        self,
        job: Mapping[str, Any],
        adapter_payload: Mapping[str, Any],
        job_work_dir: Path,
    ) -> list[dict[str, Any]]:
        # Artifact publication happens in _process_claimed through this in-memory
        # marker.  Copy into persistent state first because job_work_dir is deleted.
        artifacts = adapter_payload["artifacts"]
        if not artifacts:
            return []
        persistent = self.state_dir / "outgoing" / job["job_id"]
        shutil.rmtree(persistent, ignore_errors=True)
        persistent.mkdir(parents=True, exist_ok=False)
        total = 0
        records: list[dict[str, Any]] = []
        try:
            for artifact in artifacts:
                name = artifact["name"]
                if not _ARTIFACT_NAME.fullmatch(name):
                    raise AdapterError(f"unsafe artifact name: {name!r}")
                source = job_work_dir / "artifacts" / name
                size = source.stat().st_size
                total += size
                if total > min(
                    job["limits"]["max_artifact_bytes"],
                    int(self.config["max_artifact_bytes"]),
                ):
                    raise AdapterError("adapter artifacts exceed max_artifact_bytes")
                destination = persistent / name
                shutil.copyfile(source, destination, follow_symlinks=False)
                records.append(
                    {
                        "name": name,
                        "media_type": artifact["media_type"],
                        "size_bytes": size,
                        "sha256": _sha256_path(destination),
                        "_local_path": str(destination),
                    }
                )
            return records
        except BaseException:
            shutil.rmtree(persistent, ignore_errors=True)
            raise

    def _publish_artifacts(
        self, sftp: Any, job_id: str, artifacts: list[dict[str, Any]]
    ) -> None:
        if not artifacts:
            return
        remote_directory = self._remote("results", "ready", f"{job_id}.artifacts")
        try:
            sftp.mkdir(remote_directory, mode=0o770)
        except OSError:
            # A retry after a lost connection may find the directory already there.
            if not _is_directory(sftp, remote_directory):
                raise
        for artifact in artifacts:
            destination = str(PurePosixPath(remote_directory) / artifact["name"])
            with open(artifact["_local_path"], "rb") as stream:
                data = stream.read(int(self.config["max_artifact_bytes"]) + 1)
            if len(data) != artifact["size_bytes"]:
                raise AdapterError("staged artifact changed before publication")
            _upload_atomic(sftp, destination, data)

    def _make_result(
        self,
        job: Mapping[str, Any],
        status: str,
        summary: Mapping[str, Any],
        *,
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        public_artifacts = [
            {key: value for key, value in artifact.items() if key != "_local_path"}
            for artifact in artifacts
        ]
        return {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job["job_id"],
            "operation": job["operation"],
            "finished_at": utc_now_text(),
            "status": status,
            "worker_id": self.config["worker_id"],
            "summary": summary,
            "artifacts": public_artifacts,
        }

    def _make_failure_result(
        self, job_id: str, operation: str, status: str, message: str
    ) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "operation": operation,
            "finished_at": utc_now_text(),
            "status": status,
            "worker_id": self.config["worker_id"],
            "summary": {"error": message[:4096]},
            "artifacts": [],
        }

    def _finish_job(
        self, sftp: Any, claimed: str, job_id: str, *, rejected: bool
    ) -> None:
        destination = self._remote(
            "rejected" if rejected else "archive", f"{job_id}.json"
        )
        _rename(sftp, claimed, destination)

    def _job_names(self, sftp: Any, directory: str) -> list[str]:
        return sorted(
            name for name in sftp.listdir(directory) if _JOB_FILE.fullmatch(name)
        )

    def _remote(self, *parts: str) -> str:
        path = self.remote_root
        for part in parts:
            path /= part
        return str(path)

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        try:
            import paramiko
        except ImportError as exc:
            raise WorkerConfigError(
                "Paramiko is required on the Windows worker; install the bridge extra"
            ) from exc
        server = self.config["server"]
        client = paramiko.SSHClient()
        client.load_host_keys(server["known_hosts_file"])
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=server["host"],
            port=server["port"],
            username=server["username"],
            key_filename=server["private_key_file"],
            look_for_keys=False,
            allow_agent=False,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
        )
        sftp = client.open_sftp()
        try:
            yield sftp
        finally:
            sftp.close()
            client.close()


def _read_remote_limited(sftp: Any, path: str, max_bytes: int) -> bytes:
    size = sftp.stat(path).st_size
    if size > max_bytes:
        raise ProtocolError(f"remote job is {size} bytes; limit is {max_bytes}")
    with sftp.open(path, "rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ProtocolError(f"remote job exceeds the {max_bytes}-byte limit")
    return data


def _upload_atomic(sftp: Any, final_path: str, data: bytes) -> None:
    final = PurePosixPath(final_path)
    temp = str(final.parent / f".{final.name}.part")
    # Dot-part files are never eligible results.  Rewriting a stale part left by
    # this worker is safe and lets an interrupted upload recover automatically.
    with sftp.open(temp, "wb") as stream:
        stream.write(data)
        stream.flush()
    _rename(sftp, temp, final_path)


def _rename(sftp: Any, source: str, destination: str) -> None:
    try:
        sftp.posix_rename(source, destination)
    except (AttributeError, OSError):
        sftp.rename(source, destination)


def _exists(sftp: Any, path: str) -> bool:
    try:
        sftp.stat(path)
    except OSError:
        return False
    return True


def _is_directory(sftp: Any, path: str) -> bool:
    import stat

    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except OSError:
        return False


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise WorkerConfigError(
            f"{label} fields must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _integer_range(value: Any, label: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerConfigError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise WorkerConfigError(f"{label} must be between {minimum} and {maximum}")


def _number_range(value: Any, label: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerConfigError(f"{label} must be a number")
    if not minimum <= value <= maximum:
        raise WorkerConfigError(f"{label} must be between {minimum} and {maximum}")
