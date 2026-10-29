"""Strict data contracts for the CODE V bridge.

The job format intentionally cannot express a program, path, shell command, CODE V
macro, or arbitrary argument list.  A local allowlist maps ``case_id`` values to
trusted, already-staged CODE V inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping
from uuid import UUID


PROTOCOL_VERSION = "1.0"
RUN_STAGED_CASE = "codev.run_staged_case"
ALLOWED_OPERATIONS = frozenset({RUN_STAGED_CASE})
ALLOWED_CAPTURES = frozenset(
    {
        "error_log",
        "lens_data",
        "global_coordinates",
        "layout_image",
        "trace_summary",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "job_id",
        "created_at",
        "operation",
        "prescription",
        "environment",
        "limits",
    }
)
_PRESCRIPTION_KEYS = frozenset({"case_id", "case_sha256"})
_ENVIRONMENT_KEYS = frozenset({"codev_release", "captures"})
_LIMIT_KEYS = frozenset(
    {"timeout_seconds", "max_result_bytes", "max_artifact_bytes"}
)
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,31}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProtocolError(ValueError):
    """A job failed the bridge's closed, declarative schema."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON suitable for atomic queue files."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProtocolError("created_at must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProtocolError("created_at is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProtocolError("created_at must be in UTC")
    return parsed


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_job(value: Any) -> dict[str, Any]:
    """Validate and return a normalized copy of a bridge job.

    Unknown fields are rejected at every level.  This is deliberate: accepting a
    future field before the worker understands it would weaken the security model.
    """

    job = _require_object(value, "job")
    _require_exact_keys(job, _TOP_LEVEL_KEYS, "job")

    if job["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol_version {job['protocol_version']!r}; "
            f"expected {PROTOCOL_VERSION!r}"
        )

    job_id = job["job_id"]
    if not isinstance(job_id, str):
        raise ProtocolError("job_id must be a canonical UUID string")
    try:
        parsed_id = UUID(job_id)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("job_id must be a canonical UUID string") from exc
    if parsed_id.version != 4 or str(parsed_id) != job_id:
        raise ProtocolError("job_id must be a lowercase canonical UUIDv4")

    parse_utc_timestamp(job["created_at"])

    operation = job["operation"]
    if operation not in ALLOWED_OPERATIONS:
        raise ProtocolError(f"operation is not allowlisted: {operation!r}")

    prescription = _require_object(job["prescription"], "prescription")
    _require_exact_keys(prescription, _PRESCRIPTION_KEYS, "prescription")
    case_id = prescription["case_id"]
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise ProtocolError("prescription.case_id has an invalid format")
    case_sha256 = prescription["case_sha256"]
    if not isinstance(case_sha256, str) or not _SHA256.fullmatch(case_sha256):
        raise ProtocolError("prescription.case_sha256 must be 64 lowercase hex digits")

    environment = _require_object(job["environment"], "environment")
    _require_exact_keys(environment, _ENVIRONMENT_KEYS, "environment")
    release = environment["codev_release"]
    if not isinstance(release, str) or not _RELEASE.fullmatch(release):
        raise ProtocolError("environment.codev_release has an invalid format")
    captures = environment["captures"]
    if not isinstance(captures, list) or not captures:
        raise ProtocolError("environment.captures must be a non-empty list")
    if len(captures) != len(set(captures)):
        raise ProtocolError("environment.captures contains duplicates")
    unknown_captures = set(captures) - ALLOWED_CAPTURES
    if unknown_captures:
        raise ProtocolError(
            "environment.captures contains non-allowlisted values: "
            + ", ".join(sorted(map(str, unknown_captures)))
        )

    limits = _require_object(job["limits"], "limits")
    _require_exact_keys(limits, _LIMIT_KEYS, "limits")
    _require_int_range(limits, "timeout_seconds", 1, 1800)
    _require_int_range(limits, "max_result_bytes", 1024, 4 * 1024 * 1024)
    _require_int_range(limits, "max_artifact_bytes", 0, 64 * 1024 * 1024)

    # JSON round-tripping also strips Mapping subclasses and guarantees that the
    # returned value contains only ordinary JSON data.
    return json.loads(canonical_json_bytes(job))


def decode_job(data: bytes, *, max_bytes: int = 64 * 1024) -> dict[str, Any]:
    if len(data) > max_bytes:
        raise ProtocolError(f"job exceeds the {max_bytes}-byte worker limit")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("job is not valid UTF-8 JSON") from exc
    return validate_job(value)


def make_job(
    *,
    job_id: str,
    created_at: str,
    case_id: str,
    case_sha256: str,
    codev_release: str,
    captures: list[str],
    timeout_seconds: int,
    max_result_bytes: int,
    max_artifact_bytes: int,
) -> dict[str, Any]:
    return validate_job(
        {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "created_at": created_at,
            "operation": RUN_STAGED_CASE,
            "prescription": {
                "case_id": case_id,
                "case_sha256": case_sha256,
            },
            "environment": {
                "codev_release": codev_release,
                "captures": captures,
            },
            "limits": {
                "timeout_seconds": timeout_seconds,
                "max_result_bytes": max_result_bytes,
                "max_artifact_bytes": max_artifact_bytes,
            },
        }
    )


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ProtocolError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ProtocolError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _require_int_range(
    value: Mapping[str, Any], key: str, minimum: int, maximum: int
) -> None:
    candidate = value[key]
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ProtocolError(f"limits.{key} must be an integer")
    if not minimum <= candidate <= maximum:
        raise ProtocolError(f"limits.{key} must be between {minimum} and {maximum}")

