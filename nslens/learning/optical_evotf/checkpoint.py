"""Portable Optical-EvoTF checkpoint bundles with explicit upstream provenance."""
from __future__ import annotations

import hashlib
import json
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from flax import serialization

from ...search.evotf_emitter import (EVOSAX_UPSTREAM_COMMIT, EVOSAX_VERSION,
                                     EVOTF_CHECKPOINT, JAXLIB_VERSION, JAX_VERSION,
                                     EVOTF_CHECKPOINT_SHA256)


CHECKPOINT_SCHEMA = "nslens.optical_evotf_checkpoint.v1"
RUNTIME_VERSIONS = {
    "jax": JAX_VERSION,
    "jaxlib": JAXLIB_VERSION,
    "flax": "0.10.7",
    "optax": "0.2.5",
}


def _installed_runtime() -> dict[str, str]:
    return {package: metadata.version(package) for package in RUNTIME_VERSIONS}


def verify_runtime() -> dict[str, str]:
    installed = _installed_runtime()
    if installed != RUNTIME_VERSIONS:
        expected = ", ".join(f"{key}=={value}" for key, value in RUNTIME_VERSIONS.items())
        found = ", ".join(f"{key}=={value}" for key, value in installed.items())
        raise RuntimeError(
            f"Optical-EvoTF requires the pinned runtime ({expected}); found {found}")
    return installed


def _encode(tree) -> bytes:
    return serialization.msgpack_serialize(
        serialization.to_state_dict(tree), in_place=False)


def _decode(payload: bytes):
    return serialization.msgpack_restore(payload)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    adapter_variables,
    enabled: tuple[str, ...],
    hidden_dim: int,
    categorical_mode: str,
    max_context_len: int,
    training: Mapping[str, Any],
    core_variables=None,
) -> dict:
    runtime = verify_runtime()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    adapter_payload = _encode(adapter_variables)
    (path / "adapter.msgpack").write_bytes(adapter_payload)
    files = {"adapter.msgpack": _sha(adapter_payload)}
    if core_variables is not None:
        core_payload = _encode(core_variables)
        (path / "core.msgpack").write_bytes(core_payload)
        files["core.msgpack"] = _sha(core_payload)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "name": "Optical-EvoTF" if enabled else "numeric-only-EvoTF",
            "enabled_adapters": list(enabled),
            "hidden_dim": int(hidden_dim),
            "categorical_mode": categorical_mode,
            "max_context_len": int(max_context_len),
            "core": "fine_tuned" if core_variables is not None else "released_frozen",
        },
        "upstream": {
            "evosax_version": EVOSAX_VERSION,
            "commit": EVOSAX_UPSTREAM_COMMIT,
            "checkpoint": EVOTF_CHECKPOINT,
            "checkpoint_sha256": EVOTF_CHECKPOINT_SHA256,
        },
        "runtime": runtime,
        "training": dict(training),
        "files": files,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_checkpoint(path: Path) -> dict:
    verify_runtime()
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported Optical-EvoTF checkpoint schema in {path}")
    upstream = manifest.get("upstream") or {}
    expected = {
        "evosax_version": EVOSAX_VERSION,
        "commit": EVOSAX_UPSTREAM_COMMIT,
        "checkpoint": EVOTF_CHECKPOINT,
        "checkpoint_sha256": EVOTF_CHECKPOINT_SHA256,
    }
    if upstream != expected:
        raise ValueError("checkpoint upstream provenance does not match the pinned runtime")
    runtime = manifest.get("runtime")
    if runtime is not None and runtime != RUNTIME_VERSIONS:
        raise ValueError("checkpoint JAX/Flax runtime does not match the pinned runtime")
    decoded = {"manifest": manifest}
    for filename, expected_sha in manifest.get("files", {}).items():
        payload = (path / filename).read_bytes()
        if _sha(payload) != expected_sha:
            raise ValueError(f"checkpoint file hash mismatch: {filename}")
        decoded[filename.removesuffix(".msgpack")] = _decode(payload)
    if "adapter" not in decoded:
        raise ValueError("checkpoint is missing adapter.msgpack")
    return decoded


__all__ = [
    "CHECKPOINT_SCHEMA", "RUNTIME_VERSIONS", "load_checkpoint", "save_checkpoint",
    "verify_runtime"]
