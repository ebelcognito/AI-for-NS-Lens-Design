"""Local-only execution adapters.

The command adapter is configured on the Windows worker and never accepts a
program, working directory, environment variable, or arbitrary argument from a job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .protocol import canonical_json_bytes


class AdapterError(RuntimeError):
    pass


class Adapter:
    def run(
        self, job: Mapping[str, Any], work_dir: Path, case_file: Path
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockAdapter(Adapter):
    """Connectivity test adapter.  It never starts CODE V or another process."""

    def run(
        self, job: Mapping[str, Any], work_dir: Path, case_file: Path
    ) -> dict[str, Any]:
        return {
            "summary": {
                "message": "bridge connectivity test completed; CODE V was not started",
                "case_id": job["prescription"]["case_id"],
                "captures_requested": job["environment"]["captures"],
            },
            "artifacts": [],
        }


class FixedCommandAdapter(Adapter):
    """Run one trusted local adapter executable with a closed argument template."""

    _PLACEHOLDERS = frozenset(
        {"{job_file}", "{case_file}", "{result_file}", "{artifact_dir}"}
    )

    def __init__(self, config: Mapping[str, Any]) -> None:
        expected = {"type", "executable", "working_directory", "arguments"}
        _exact_keys(config, expected, "adapter")
        executable = Path(_require_string(config, "executable"))
        working_directory = Path(_require_string(config, "working_directory"))
        arguments = config["arguments"]
        if not executable.is_absolute() or not working_directory.is_absolute():
            raise AdapterError("adapter executable and working_directory must be absolute")
        if not executable.is_file():
            raise AdapterError(f"trusted adapter executable does not exist: {executable}")
        if not working_directory.is_dir():
            raise AdapterError(
                f"trusted adapter working_directory does not exist: {working_directory}"
            )
        if not isinstance(arguments, list) or not all(
            isinstance(item, str) for item in arguments
        ):
            raise AdapterError("adapter.arguments must be a list of strings")
        for item in arguments:
            if "{" in item or "}" in item:
                if item not in self._PLACEHOLDERS:
                    raise AdapterError(
                        "adapter arguments may use only the complete placeholders "
                        "{job_file}, {case_file}, {result_file}, and {artifact_dir}"
                    )
        required_placeholders = {"{job_file}", "{case_file}", "{result_file}"}
        if not required_placeholders.issubset(arguments):
            raise AdapterError(
                "adapter.arguments must include {job_file}, {case_file}, "
                "and {result_file}"
            )
        self.executable = executable
        self.working_directory = working_directory
        self.arguments = list(arguments)

    def run(
        self, job: Mapping[str, Any], work_dir: Path, case_file: Path
    ) -> dict[str, Any]:
        job_file = work_dir / "normalized-job.json"
        result_file = work_dir / "adapter-result.json"
        artifact_dir = work_dir / "artifacts"
        artifact_dir.mkdir(mode=0o700, exist_ok=False)
        job_file.write_bytes(canonical_json_bytes(job))

        replacements = {
            "{job_file}": str(job_file),
            "{case_file}": str(case_file),
            "{result_file}": str(result_file),
            "{artifact_dir}": str(artifact_dir),
        }
        command = [str(self.executable)] + [
            replacements.get(argument, argument) for argument in self.arguments
        ]

        stdout_path = work_dir / "adapter.stdout.log"
        stderr_path = work_dir / "adapter.stderr.log"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=self.working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    timeout=job["limits"]["timeout_seconds"],
                    check=False,
                    env=os.environ.copy(),
                )
        except subprocess.TimeoutExpired as exc:
            raise AdapterError("trusted CODE V adapter exceeded its time limit") from exc

        if completed.returncode != 0:
            stderr_tail = _read_tail(stderr_path, 8192)
            raise AdapterError(
                f"trusted CODE V adapter exited {completed.returncode}: {stderr_tail}"
            )
        if not result_file.is_file():
            raise AdapterError("trusted CODE V adapter did not create adapter-result.json")
        if result_file.stat().st_size > job["limits"]["max_result_bytes"]:
            raise AdapterError("trusted CODE V adapter result exceeded max_result_bytes")
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("trusted CODE V adapter returned invalid JSON") from exc
        return _validate_adapter_payload(payload, artifact_dir)


def make_adapter(config: Mapping[str, Any]) -> Adapter:
    adapter_type = config.get("type")
    if adapter_type == "mock":
        _exact_keys(config, {"type"}, "adapter")
        return MockAdapter()
    if adapter_type == "fixed_command":
        return FixedCommandAdapter(config)
    raise AdapterError(f"unsupported local adapter type: {adapter_type!r}")


def _validate_adapter_payload(value: Any, artifact_dir: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError("adapter result must be a JSON object")
    _exact_keys(value, {"summary", "artifacts"}, "adapter result")
    if not isinstance(value["summary"], dict):
        raise AdapterError("adapter result summary must be a JSON object")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list):
        raise AdapterError("adapter result artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise AdapterError("each adapter artifact must be an object")
        _exact_keys(artifact, {"name", "media_type"}, "adapter artifact")
        name = artifact["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or len(name) > 128
        ):
            raise AdapterError("adapter artifact name must be a safe basename")
        media_type = artifact["media_type"]
        if not isinstance(media_type, str) or len(media_type) > 80:
            raise AdapterError("adapter artifact media_type is invalid")
        artifact_path = artifact_dir / name
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise AdapterError(f"adapter artifact is missing or unsafe: {name}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AdapterError(
            f"{label} fields must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _require_string(value: Mapping[str, Any], key: str) -> str:
    candidate = value[key]
    if not isinstance(candidate, str) or not candidate:
        raise AdapterError(f"adapter.{key} must be a non-empty string")
    return candidate


def _read_tail(path: Path, limit: int) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit), os.SEEK_SET)
        return stream.read(limit).decode("utf-8", errors="replace").strip()
