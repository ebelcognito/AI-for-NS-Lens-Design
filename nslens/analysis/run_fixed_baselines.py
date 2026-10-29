"""Run equal-budget CMA-ME and uniform-random fixed-environment baselines.

Each run owns one persistent ray-engine process.  Parallelism is across independent
environment/seed/algorithm runs, which keeps the established single-run CMA RNG protocol
intact while using the workstation's cores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..engine import Runner
from ..qd import fitness as fitness_registry
from ..qd.driver import _evaluate, run as run_cma
from ..qd.experiments import (DEFAULT_FIXED_ENVIRONMENT_SPECS,
                              fixed_environment, load_fixed_environment_specs)
from ..search import GridArchive

SCHEMA = "nslens.fixed_baselines.v1"
ALGORITHMS = ("cma", "random")
TRACE_INFO_KEYS = (
    "tir_count", "spot", "efficiency", "dom_signature", "thinness_T",
    "per_field", "worst_focthru", "intended_sequence",
    "intended_root_support_min", "intended_energy_share_mean",
    "intended_dominant_all", "intended_critical_margin_min",
    "intended_per_field",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _random_trace_record(index: int, x: np.ndarray, result: dict | None,
                         status: str, old: float | None,
                         diagnostics: Mapping[str, Any] | None = None) -> dict:
    record = {
        "eval": index,
        "algorithm": "uniform_random",
        "status": status,
        "genotype": [round(float(value), 6) for value in x],
    }
    if diagnostics is not None and "engine_calls" in diagnostics:
        record["engine_calls"] = int(diagnostics["engine_calls"])
    if result is None and diagnostics:
        record["rejection"] = dict(diagnostics)
    if result is not None:
        record.update(
            fitness=round(float(result["fitness"]), 6),
            bc=[round(float(value), 4) for value in result["bc"]],
            delta=(round(float(result["fitness"] - old), 6)
                   if status == "improved" and old is not None else None),
            info={key: result["info"][key] for key in TRACE_INFO_KEYS
                  if key in result["info"]},
        )
    return record


def _run_random(cfg, out_dir: Path) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {out_dir}")
    out_dir.mkdir(parents=True)
    problem = cfg.problem
    lo = np.asarray(problem.lo, dtype=float)
    hi = np.asarray(problem.hi, dtype=float)
    rng = np.random.default_rng(cfg.seed)
    archive = GridArchive(tuple(cfg.bc_bins), tuple(tuple(v) for v in cfg.bc_bounds))
    trace_path = out_dir / "trace.jsonl"
    invalid = 0
    started = time.time()
    with trace_path.open("w") as trace, Runner(timeout_s=20) as runner:
        for index in range(1, cfg.evals + 1):
            x = rng.uniform(lo, hi)
            diagnostics = {}
            result = _evaluate(runner, problem, cfg, x,
                               diagnostics=diagnostics)
            if result is None:
                invalid += 1
                status, old = "invalid", None
            else:
                status, old = archive.add(
                    x, result["fitness"], result["bc"], info=result["info"])
            trace.write(json.dumps(_random_trace_record(
                index, x, result, status, old, diagnostics),
                allow_nan=False) + "\n")

    k0, k1 = cfg.cell_bc_keys
    cells = []
    for (i0, i1), cell in archive.cells.items():
        row = {
            "i0": int(i0), "i1": int(i1),
            k0: float(cell.bc[0]), k1: float(cell.bc[1]),
            "fitness": float(cell.fitness),
            "genotype": [float(value) for value in cell.genotype],
        }
        row.update(cell.info)
        cells.append(row)
    payload = {
        "meta": {
            "schema_version": SCHEMA,
            "algorithm": "uniform_random",
            "experiment": cfg.label,
            "evals": cfg.evals,
            "seed": cfg.seed,
            "invalid": invalid,
            "bc_bins": list(cfg.bc_bins),
            "bc_bounds": [list(value) for value in cfg.bc_bounds],
            "fitness": fitness_registry.META_DESCRIPTION[cfg.fitness],
            "param_names": list(problem.param_names),
            "fields": list(cfg.fields or ()),
            "intended_sequence": cfg.intended_sequence,
            "intended_path_gates": {
                "min_root_support": cfg.min_intended_root_support,
                "min_mean_delivered_energy_share":
                    cfg.min_intended_delivered_energy_share,
                "require_dominant_all_fields":
                    cfg.require_intended_dominant_all_fields,
                "min_critical_margin_deg":
                    cfg.min_intended_critical_margin_deg,
            },
            "wall_seconds": time.time() - started,
            **cfg.meta_extra,
        },
        "history": [],
        "cells": cells,
    }
    _atomic_json(out_dir / "archive.json", payload)
    return payload


def _run_one(task: Mapping[str, Any]) -> dict:
    environment_id = str(task["environment_id"])
    algorithm = str(task["algorithm"])
    seed = int(task["seed"])
    specs_path = Path(task["specs_path"])
    out_dir = Path(task["out_dir"])
    cfg = fixed_environment(environment_id, specs_path)
    cfg.seed = seed
    cfg.evals = int(task["evals"])
    if algorithm == "cma":
        payload = run_cma(cfg, out_dir, log=lambda _message: None)
        payload["meta"]["algorithm"] = "cma_me"
        _atomic_json(out_dir / "archive.json", payload)
    elif algorithm == "random":
        payload = _run_random(cfg, out_dir)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    archive_path = out_dir / "archive.json"
    trace_path = out_dir / "trace.jsonl"
    return {
        "environment_id": environment_id,
        "algorithm": algorithm,
        "seed": seed,
        "evals": int(payload["meta"]["evals"]),
        "invalid": int(payload["meta"]["invalid"]),
        "cells": len(payload["cells"]),
        "best_fitness": max((float(row["fitness"]) for row in payload["cells"]),
                            default=None),
        "wall_seconds": float(payload["meta"]["wall_seconds"]),
        "archive": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path),
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
    }


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-specs",
                        default=str(DEFAULT_FIXED_ENVIRONMENT_SPECS))
    parser.add_argument("--environment-id", action="append",
                        help="repeat to run a subset; default is every accepted environment")
    parser.add_argument("--algorithms", default="cma,random",
                        help="comma-separated subset of cma,random")
    parser.add_argument("--seeds", default="0", help="comma-separated integer seeds")
    parser.add_argument("--evals", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.evals <= 0 or args.workers <= 0:
        parser.error("--evals and --workers must be positive")
    algorithms = _csv(args.algorithms)
    unknown = set(algorithms) - set(ALGORITHMS)
    if not algorithms or unknown:
        parser.error(f"--algorithms must contain only {','.join(ALGORITHMS)}")
    try:
        seeds = [int(value) for value in _csv(args.seeds)]
    except ValueError as exc:
        parser.error(f"invalid --seeds: {exc}")
    if not seeds:
        parser.error("--seeds must not be empty")

    specs_path = Path(args.environment_specs).resolve()
    available = load_fixed_environment_specs(specs_path)
    known_ids = [row["environment_id"] for row in available]
    environment_ids = args.environment_id or known_ids
    missing = set(environment_ids) - set(known_ids)
    if missing:
        parser.error(f"unknown --environment-id values: {', '.join(sorted(missing))}")

    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    tasks = []
    for algorithm in algorithms:
        for environment_id in environment_ids:
            for seed in seeds:
                run_dir = output / algorithm / environment_id / f"seed{seed}"
                tasks.append({
                    "environment_id": environment_id,
                    "algorithm": algorithm,
                    "seed": seed,
                    "evals": args.evals,
                    "specs_path": str(specs_path),
                    "out_dir": str(run_dir),
                })

    started = time.time()
    results = []
    workers = min(args.workers, len(tasks))
    with Pool(workers) as pool:
        for index, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results.append(result)
            best = result["best_fitness"]
            best_text = "none" if best is None else f"{best:.4f}"
            print(f"[fixed-baseline] {index}/{len(tasks)} "
                  f"{result['algorithm']} {result['environment_id']} "
                  f"seed={result['seed']} cells={result['cells']} best={best_text}",
                  flush=True)

    results.sort(key=lambda row: (row["algorithm"], row["environment_id"], row["seed"]))
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_specs": str(specs_path),
        "environment_specs_sha256": _sha256(specs_path),
        "algorithms": algorithms,
        "seeds": seeds,
        "evals_per_run": args.evals,
        "workers": workers,
        "runs": results,
        "wall_seconds": time.time() - started,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps({
        "runs": len(results),
        "wall_seconds": manifest["wall_seconds"],
        "manifest": str((output / "manifest.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
