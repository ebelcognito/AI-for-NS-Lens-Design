"""Design and screen diverse fixed-condition search environments.

Seeds come only from Experiment-1 candidates labelled ``intentional``.  Each proposed
environment fixes source position, source angle, and refractive index, then exposes a
local six-dimensional geometry box.  Random trials measure the useful-path yield at
several box widths; the widest box meeting the declared yield floor is accepted.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..engine import Runner, terminal_ledger_residuals
from ..engine.signatures import parse_target
from ..qd.fixed_environments import GEOMETRY_PARAMETERS
from ..rl import env_config
from ..rl.reward import geometry_verdict
from .interaction_audit import _optical_vertices, _root_ids

SCHEMA = "nslens.fixed_environment_pilot.v1"
SPEC_SCHEMA = "nslens.fixed_fold_environment.v1"
DEFAULT_SCALES = (0.01, 0.025, 0.05)
ACCEPTANCE = {
    "minimum_geometry_valid_rate": 0.50,
    "minimum_path_feasible_rate": 0.20,
    "minimum_path_feasible_trials": 6,
    "minimum_per_field_root_support": 0.80,
    "minimum_mean_delivered_energy_share": 0.05,
    "required_dominant_field_fraction": 1.0,
    "ledger_tolerance": 1e-9,
}

_RUNNER = None
_ENV = None
_PROBLEM = None
_FIELDS: tuple[float, ...] = ()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(text: str) -> tuple[tuple[str, str], ...]:
    return parse_target(text.replace("->", " "))


def _worker_close() -> None:
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.__exit__(None, None, None)
        _RUNNER = None


def _worker_init(fields, rays) -> None:
    global _RUNNER, _ENV, _PROBLEM, _FIELDS
    _ENV = env_config.get("fold", fov=8.0, focus_w=8.0)
    _PROBLEM = replace(_ENV.problem, n_rays=int(rays))
    _FIELDS = tuple(float(field) for field in fields)
    _RUNNER = Runner(timeout_s=60)
    _RUNNER.__enter__()
    atexit.register(_worker_close)


def _field_evidence(result: Mapping[str, Any], params: Mapping[str, float], target,
                    n_rays: int) -> dict:
    terminal = list(result.get("rayTerminations", []))
    exact = [record for record in terminal
             if record.get("historyComplete") is True and not record.get("aggregate")]
    roots = _root_ids(terminal)
    face_of = _PROBLEM.face_of(params)
    rows = []
    delivered_energy = Counter()
    emitted = float(result["energyLedger"]["emitted"])
    for record in exact:
        signature = tuple(event for event, _ in _optical_vertices(record, face_of))
        root = roots.get(record.get("rayId"))
        rows.append((record, signature, root))
        if record.get("outcome") == "delivered":
            delivered_energy[signature] += float(record.get("b", 0.0))
    dominant = (max(delivered_energy, key=delivered_energy.get)
                if delivered_energy else None)
    matching = [(record, root) for record, signature, root in rows
                if signature == tuple(target) and record.get("outcome") == "delivered"]
    matching_roots = {root for _, root in matching if root is not None}
    target_energy = sum(float(record.get("b", 0.0)) for record, _ in matching)
    residuals = terminal_ledger_residuals(result)
    closure = max(abs(value) for value in residuals.values())
    return {
        "root_support": min(1.0, len(matching_roots) / n_rays),
        "delivered_energy_share": target_energy / emitted if emitted else 0.0,
        "target_is_dominant": dominant == tuple(target),
        "delivered_share": float(result["energyLedger"]["delivered"]) / emitted,
        "history_exact_energy_share": (sum(float(record.get("b", 0.0)) for record in exact)
                                       / emitted if emitted else 0.0),
        "ledger_residual_max": closure,
    }


def _pilot_one(task: Mapping[str, Any]) -> dict:
    x = np.asarray(task["genotype"], dtype=float)
    trial = {
        "schema_version": SCHEMA, "environment_id": task["environment_id"],
        "seed_design_id": task["seed_design_id"], "scale": task["scale"],
        "sample": task["sample"],
        "design": {name: float(value) for name, value in zip(_ENV.param_names, x)},
        "geometry_valid": False, "path_feasible": False, "fields": [],
    }
    bad = geometry_verdict(_ENV, x)
    if bad is not None:
        trial["failure"] = f"{bad.stage}: {bad.reason}"
        return trial
    trial["geometry_valid"] = True
    target = _target(task["historical_sequence"])
    params = _PROBLEM.vec_to_params(x)
    try:
        for field in _FIELDS:
            result, _ = _RUNNER.simulate(
                _PROBLEM.layout(params, field_deg=field).to_scene_dict())
            evidence = _field_evidence(result, params, target, _PROBLEM.n_rays)
            evidence["field_deg"] = field
            trial["fields"].append(evidence)
    except Exception as exc:
        trial["failure"] = f"{type(exc).__name__}: {exc}"
        return trial
    fields = trial["fields"]
    dominant_fraction = sum(field["target_is_dominant"] for field in fields) / len(fields)
    mean_energy = sum(field["delivered_energy_share"] for field in fields) / len(fields)
    trial["minimum_root_support"] = min(field["root_support"] for field in fields)
    trial["mean_delivered_energy_share"] = mean_energy
    trial["dominant_field_fraction"] = dominant_fraction
    trial["path_feasible"] = bool(
        trial["minimum_root_support"] >= ACCEPTANCE["minimum_per_field_root_support"]
        and mean_energy >= ACCEPTANCE["minimum_mean_delivered_energy_share"]
        and dominant_fraction >= ACCEPTANCE["required_dominant_field_fraction"]
        and max(field["ledger_residual_max"] for field in fields)
        <= ACCEPTANCE["ledger_tolerance"])
    if not trial["path_feasible"]:
        trial["failure"] = "interaction path failed the declared pilot thresholds"
    return trial


def _load_intentional(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("audit_label") == "intentional" and \
                    (record.get("source_hardened_verdict") or {}).get("training_eligible"):
                rows.append(record)
    if not rows:
        raise ValueError(f"no intentional, training-eligible records in {path}")
    return rows


def _feature(record: Mapping[str, Any], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    names = env_config.get("fold").param_names
    x = np.array([record["design"][name] for name in names], dtype=float)
    return (x - lo) / (hi - lo)


def _select_anchors(records: Sequence[dict], count: int) -> list[dict]:
    """Cover every verified sequence first, then farthest-point sample the prescription box."""
    env = env_config.get("fold")
    lo, hi = np.asarray(env.problem.lo, float), np.asarray(env.problem.hi, float)
    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record["historical_sequence"]].append(record)
    selected = [max(group, key=lambda record: record["source_hardened_verdict"]["fitness"])
                for _, group in sorted(by_sequence.items())]
    selected = selected[:count]
    remaining = [record for record in records if record not in selected]
    while len(selected) < min(count, len(records)):
        def score(record):
            point = _feature(record, lo, hi)
            distance = min(float(np.linalg.norm(point - _feature(chosen, lo, hi)))
                           for chosen in selected)
            return distance + 0.05 * float(record["source_hardened_verdict"]["fitness"])
        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _bounds(anchor: Mapping[str, Any], scale: float, env) -> dict[str, tuple[float, float]]:
    global_bounds = env.bounds()
    result = {}
    for name in GEOMETRY_PARAMETERS:
        lo, hi = global_bounds[name]
        center = float(anchor["design"][name])
        half_width = scale * (hi - lo)
        result[name] = (max(lo, center - half_width), min(hi, center + half_width))
    return result


def _tasks(anchors: Sequence[dict], scales: Sequence[float], samples: int,
           seed: int) -> tuple[list[dict], dict[tuple[str, float], dict]]:
    env = env_config.get("fold")
    tasks = []
    bounds_by_key = {}
    for anchor_index, anchor in enumerate(anchors):
        env_id = f"fixed-fold-{anchor_index:02d}-{anchor['design_id'].split(':')[-1][:8]}"
        for scale in scales:
            bounds = _bounds(anchor, scale, env)
            bounds_by_key[(env_id, scale)] = bounds
            seed_text = f"{seed}|{env_id}|{scale}".encode()
            local_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
            rng = np.random.default_rng(local_seed)
            for sample in range(samples):
                design = dict(anchor["design"])
                for name, (lo, hi) in bounds.items():
                    design[name] = float(rng.uniform(lo, hi))
                tasks.append({
                    "environment_id": env_id, "seed_design_id": anchor["design_id"],
                    "historical_sequence": anchor["historical_sequence"],
                    "scale": float(scale), "sample": sample,
                    "genotype": [design[name] for name in env.param_names],
                })
    return tasks, bounds_by_key


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def run(args) -> dict:
    audit_path = Path(args.audit_records).resolve()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)
    intentional = _load_intentional(audit_path)
    anchors = _select_anchors(intentional, args.environments)
    scales = tuple(sorted(float(value) for value in args.scales.split(",")))
    tasks, bounds_by_key = _tasks(anchors, scales, args.samples, args.seed)
    fields = (-float(args.fov), 0.0, float(args.fov))
    workers = max(1, min(args.workers, len(tasks)))
    started = time.time()
    trials = []
    if workers == 1:
        _worker_init(fields, args.rays)
        try:
            for index, task in enumerate(tasks, 1):
                trials.append(_pilot_one(task))
                if index % 50 == 0 or index == len(tasks):
                    print(f"[environment-pilot] {index}/{len(tasks)}", flush=True)
        finally:
            _worker_close()
    else:
        with Pool(workers, initializer=_worker_init, initargs=(fields, args.rays)) as pool:
            for index, trial in enumerate(pool.imap(_pilot_one, tasks, chunksize=1), 1):
                trials.append(trial)
                if index % 50 == 0 or index == len(tasks):
                    print(f"[environment-pilot] {index}/{len(tasks)}", flush=True)

    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial["environment_id"], trial["scale"])].append(trial)
    anchor_of = {}
    for index, anchor in enumerate(anchors):
        env_id = f"fixed-fold-{index:02d}-{anchor['design_id'].split(':')[-1][:8]}"
        anchor_of[env_id] = anchor

    summaries = {}
    specs = []
    for env_id, anchor in anchor_of.items():
        accepted_scale = None
        contiguous = True
        scale_summaries = []
        for scale in scales:
            group = grouped[(env_id, scale)]
            geometry_count = sum(trial["geometry_valid"] for trial in group)
            feasible_count = sum(trial["path_feasible"] for trial in group)
            summary = {
                "scale_fraction_of_global_gene_range": scale,
                "trials": len(group), "geometry_valid": geometry_count,
                "path_feasible": feasible_count,
                "geometry_valid_rate": geometry_count / len(group),
                "path_feasible_rate": feasible_count / len(group),
                "median_feasible_delivered_energy_share": (
                    float(np.median([trial["mean_delivered_energy_share"] for trial in group
                                     if trial["path_feasible"]]))
                    if feasible_count else 0.0),
            }
            summary["accepted"] = bool(
                summary["geometry_valid_rate"] >= ACCEPTANCE["minimum_geometry_valid_rate"]
                and summary["path_feasible_rate"] >= ACCEPTANCE["minimum_path_feasible_rate"]
                and feasible_count >= ACCEPTANCE["minimum_path_feasible_trials"])
            summary["accepted_in_contiguous_width_sweep"] = bool(
                contiguous and summary["accepted"])
            if summary["accepted_in_contiguous_width_sweep"]:
                accepted_scale = scale
            else:
                # A larger axis-aligned box contains the failed smaller neighborhood.
                # Never admit it merely because a finite random sample fluctuated upward.
                contiguous = False
            scale_summaries.append(summary)
        summaries[env_id] = scale_summaries
        fixed = {name: float(anchor["design"][name]) for name in ("ray_y", "ang", "n")}
        fixed.update({"x_sensor": 120.0, "beam_width": 8.0,
                      "focus_window_half_width": 8.0, "source_clearance": 2.0})
        spec = {
            "schema_version": SPEC_SCHEMA, "environment_id": env_id,
            "status": "accepted" if accepted_scale is not None else "rejected",
            "seed_design_id": anchor["design_id"],
            "intended_sequence": anchor["historical_sequence"],
            "fixed": fixed,
            "variables": ({name: list(interval) for name, interval in
                           bounds_by_key[(env_id, accepted_scale)].items()}
                          if accepted_scale is not None else {}),
            "field_curriculum_half_angle_deg": [2.0, 4.0, 6.0, 8.0],
            "pilot": {"selected_scale": accepted_scale,
                      "scale_summaries": scale_summaries},
        }
        specs.append(spec)

    trial_path = output / "trials.jsonl"
    _atomic_text(trial_path, "".join(_canonical_json(trial) + "\n" for trial in trials))
    spec_path = output / "environment_specs.json"
    _atomic_text(spec_path, json.dumps({"schema_version": SPEC_SCHEMA,
                                       "environments": specs}, indent=2) + "\n")
    status = Counter(spec["status"] for spec in specs)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_audit_records": str(audit_path), "source_sha256": _sha256_file(audit_path),
        "intentional_training_eligible_pool": len(intentional),
        "anchor_count": len(anchors), "environment_status_counts": dict(status),
        "config": {"scales": scales, "samples_per_scale": args.samples,
                   "pilot_fields_deg": fields, "rays_per_field": args.rays,
                   "seed": args.seed, "workers": workers},
        "acceptance": ACCEPTANCE,
        "outputs": {
            "environment_specs": {"path": str(spec_path), "sha256": _sha256_file(spec_path)},
            "trials": {"path": str(trial_path), "sha256": _sha256_file(trial_path),
                       "rows": len(trials)},
        },
        "wall_seconds": time.time() - started,
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_text(output / "README.md", (
        "# Fixed-environment feasibility pilot\n\n"
        "Each environment fixes source position, source angle, and refractive index. Only "
        "six geometry parameters vary. Bounds are admitted only after random trials retain "
        "the seed's Experiment-1-verified interaction path at the rates declared in "
        "`manifest.json`. Rejected trials are retained in `trials.jsonl`.\n"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--scales", default=",".join(str(value) for value in DEFAULT_SCALES))
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--fov", type=float, default=8.0)
    parser.add_argument("--rays", type=int, default=33)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"output": str(Path(args.out).resolve()),
                      "environment_status_counts": manifest["environment_status_counts"],
                      "wall_seconds": round(manifest["wall_seconds"], 3)}, indent=2))


if __name__ == "__main__":
    main()
