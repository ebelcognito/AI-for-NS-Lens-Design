"""Fresh-seed equal-ray-engine-call benchmark for fixed-environment policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from ..engine import Runner
from ..learning.fixed_surrogate import feature_vector
from ..learning.train_fixed_pairwise_policy import pair_features
from ..qd.driver import _evaluate
from ..qd.experiments import (DEFAULT_FIXED_ENVIRONMENT_SPECS, fixed_environment,
                              load_fixed_environment_specs)
from ..search import CMAES, GridArchive

SCHEMA = "nslens.fixed_call_budget_benchmark.v1"
ALGORITHMS = ("cma", "random", "learned")
PARAMETERS = ("cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b")


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


def _environment_payload(spec: Mapping[str, Any]) -> dict:
    fixed = spec["fixed"]
    return {
        "environment_id": spec["environment_id"],
        "fixed": {key: fixed[key] for key in (
            "ray_y", "ang", "n", "x_sensor", "beam_width",
            "focus_window_half_width")},
        "intended_sequence": spec["intended_sequence"],
        "fields_deg": [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
        "parameter_names": list(PARAMETERS),
        "parameter_bounds": [spec["variables"][name] for name in PARAMETERS],
    }


def _global_scores(bundle: Mapping[str, Any], environment: dict,
                   candidates: np.ndarray) -> np.ndarray:
    x = np.stack([feature_vector(environment, candidate) for candidate in candidates])
    probability = bundle["classifier"].predict_proba(x)[:, 1]
    mode = bundle["score_mode"]
    if mode == "feasibility":
        return probability
    quality = bundle["regressor"].predict(x)
    if mode == "feasibility_x_quality":
        return probability * quality
    if mode == "threshold_quality":
        return (probability >= 0.5) * quality
    raise ValueError(f"unknown global score mode {mode!r}")


def _valid_geometry(problem, candidates: np.ndarray) -> np.ndarray:
    return np.asarray([candidate for candidate in candidates
                       if problem.is_valid(problem.vec_to_params(candidate))])


def _global_proposal(rng, problem, environment, bundle, pool_size,
                     lo, hi) -> tuple[np.ndarray, int]:
    candidates = []
    draws = 0
    while len(candidates) < pool_size:
        batch = rng.uniform(lo, hi, size=(pool_size * 2, len(lo)))
        draws += len(batch)
        candidates.extend(_valid_geometry(problem, batch))
    pool = np.asarray(candidates[:pool_size])
    return pool[int(np.argmax(_global_scores(bundle, environment, pool)))], draws


def _pairwise_proposal(rng, problem, environment, model, global_bundle,
                       parent, sigma, pool_size, lo, hi) -> tuple[np.ndarray, int]:
    candidates = []
    draws = 0
    scale = hi - lo
    while len(candidates) < pool_size:
        batch = np.clip(parent + rng.normal(size=(pool_size * 2, len(lo)))
                        * sigma * scale, lo, hi)
        draws += len(batch)
        candidates.extend(_valid_geometry(problem, batch))
    pool = np.asarray(candidates[:pool_size])
    state = {
        "environment": environment,
        "parent_normalized": ((parent - lo) / scale).tolist(),
        "sigma": float(sigma),
    }
    features = []
    pairs = []
    for first in range(pool_size):
        for second in range(first + 1, pool_size):
            features.append(pair_features(
                state, {"genotype": pool[first]}, {"genotype": pool[second]}))
            pairs.append((first, second))
    probability = model.predict_proba(np.stack(features))[:, 1]
    wins = np.zeros(pool_size)
    for (first, second), value in zip(pairs, probability):
        wins[first] += value
        wins[second] += 1.0 - value
    global_score = _global_scores(global_bundle, environment, pool)
    ranking = np.lexsort((-global_score, -wins))
    return pool[int(ranking[0])], draws


def _record(index: int, algorithm: str, proposal_kind: str, genotype,
            result, status, old, diagnostics, cumulative_calls: int,
            model_candidates: int, sigma: float | None) -> dict:
    row = {
        "eval": index, "algorithm": algorithm, "proposal_kind": proposal_kind,
        "genotype": [round(float(value), 6) for value in genotype],
        "status": status, "engine_calls": int(diagnostics["engine_calls"]),
        "cumulative_engine_calls": cumulative_calls,
        "model_candidates_considered": model_candidates,
    }
    if sigma is not None:
        row["sigma"] = sigma
    if result is None:
        row["rejection"] = {key: diagnostics[key] for key in ("stage", "reason")}
    else:
        row.update(fitness=float(result["fitness"]),
                   bc=[float(value) for value in result["bc"]],
                   delta=(float(result["fitness"] - old)
                          if status == "improved" and old is not None else None),
                   info={key: result["info"].get(key) for key in (
                       "intended_root_support_min", "intended_energy_share_mean",
                       "intended_dominant_all", "intended_critical_margin_min",
                       "worst_focthru")})
    return row


def _archive_payload(cfg, archive, algorithm, seed, budget, used, proposals,
                     actual_model_candidates, wall_seconds) -> dict:
    k0, k1 = cfg.cell_bc_keys
    cells = []
    for (i0, i1), cell in archive.cells.items():
        row = {"i0": int(i0), "i1": int(i1), k0: float(cell.bc[0]),
               k1: float(cell.bc[1]), "fitness": float(cell.fitness),
               "genotype": [float(value) for value in cell.genotype]}
        row.update(cell.info)
        cells.append(row)
    return {
        "meta": {"schema_version": SCHEMA, "algorithm": algorithm,
                 "environment_id": cfg.meta_extra["fixed_environment_id"],
                 "seed": seed, "engine_call_budget": budget,
                 "engine_calls_used": used, "unused_engine_calls": budget - used,
                 "proposals_evaluated": proposals,
                 "model_candidates_considered": actual_model_candidates,
                 "param_names": list(cfg.problem.param_names),
                 "fields": list(cfg.fields), "intended_sequence": cfg.intended_sequence,
                 "wall_seconds": wall_seconds},
        "cells": cells,
    }


def _run_random_or_learned(task: Mapping[str, Any], cfg, spec, output: Path) -> dict:
    algorithm = task["algorithm"]
    seed = int(task["seed"])
    budget = int(task["engine_call_budget"])
    rng = np.random.default_rng(seed)
    problem = cfg.problem
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    archive = GridArchive(tuple(cfg.bc_bins), tuple(tuple(value) for value in cfg.bc_bounds))
    global_bundle = pairwise_bundle = None
    if algorithm == "learned":
        global_bundle = joblib.load(task["global_model"])
        pairwise_bundle = joblib.load(task["pairwise_model"])
        # The outer benchmark already parallelizes independent runs. Avoid nested joblib
        # pools (and their warning/launch overhead) inside every proposal decision.
        global_bundle["classifier"].n_jobs = 1
        global_bundle["regressor"].n_jobs = 1
        pairwise_bundle["model"].n_jobs = 1
    incumbent = None
    incumbent_fitness = None
    sigma = 0.12
    stagnation = 0
    calls = proposals = model_candidates = 0
    started = time.time()
    environment = _environment_payload(spec)
    trace_path = output / "trace.jsonl"
    with trace_path.open("w") as trace, Runner(timeout_s=20) as runner:
        while budget - calls >= 9:
            proposals += 1
            if algorithm == "random":
                genotype = rng.uniform(lo, hi)
                proposal_kind, considered, used_sigma = "uniform", 1, None
            elif incumbent is None or proposals % 5 == 0:
                genotype, considered = _global_proposal(
                    rng, problem, environment, global_bundle,
                    int(task["global_pool_size"]), lo, hi)
                proposal_kind, used_sigma = "model_global", None
            else:
                genotype, considered = _pairwise_proposal(
                    rng, problem, environment, pairwise_bundle["model"], global_bundle,
                    incumbent, sigma, int(task["local_pool_size"]), lo, hi)
                proposal_kind, used_sigma = "model_pairwise", sigma
            model_candidates += considered
            diagnostics = {}
            result = _evaluate(runner, problem, cfg, genotype,
                               diagnostics=diagnostics)
            cost = int(diagnostics["engine_calls"])
            calls += cost
            if result is None:
                status, old = "invalid", None
                stagnation += 1
            else:
                status, old = archive.add(
                    genotype, result["fitness"], result["bc"], info=result["info"])
                if incumbent_fitness is None or result["fitness"] > incumbent_fitness:
                    incumbent = np.asarray(genotype, float).copy()
                    incumbent_fitness = float(result["fitness"])
                    sigma = max(0.025, sigma * 0.90)
                    stagnation = 0
                else:
                    stagnation += 1
            if stagnation >= 25:
                sigma = min(0.30, sigma * 1.4)
                stagnation = 0
            trace.write(json.dumps(_record(
                proposals, algorithm, proposal_kind, genotype, result, status, old,
                diagnostics, calls, considered, used_sigma), allow_nan=False) + "\n")
            if proposals > budget * 20:
                raise RuntimeError("too many zero-call proposals; aborting")
    payload = _archive_payload(cfg, archive, algorithm, seed, budget, calls, proposals,
                               model_candidates, time.time() - started)
    _atomic_json(output / "archive.json", payload)
    return payload


def _run_cma(task: Mapping[str, Any], cfg, output: Path) -> dict:
    seed = int(task["seed"])
    budget = int(task["engine_call_budget"])
    problem = cfg.problem
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    scale = hi - lo
    rng = np.random.default_rng(seed)
    archive = GridArchive(tuple(cfg.bc_bins), tuple(tuple(value) for value in cfg.bc_bounds))

    def new_emitter(emitter_seed):
        initial = archive.random_elite(rng)
        if initial is None:
            initial = np.asarray(cfg.initial_genotype, float)
        return CMAES((np.asarray(initial) - lo) / scale, cfg.sigma0,
                     popsize=cfg.popsize, seed=emitter_seed,
                     bounds=(np.zeros_like(lo), np.ones_like(hi)))

    emitters = [new_emitter(seed + index) for index in range(cfg.emitters)]
    calls = proposals = 0
    stop = False
    started = time.time()
    trace_path = output / "trace.jsonl"
    with trace_path.open("w") as trace, Runner(timeout_s=20) as runner:
        while not stop:
            for emitter_index, emitter in enumerate(emitters):
                population = lo + emitter.ask() * scale
                parent = lo + emitter.mean * scale
                keys = np.empty(len(population))
                added = False
                completed = 0
                for index, genotype in enumerate(population):
                    if budget - calls < 9:
                        stop = True
                        break
                    proposals += 1
                    diagnostics = {}
                    result = _evaluate(runner, problem, cfg, genotype,
                                       diagnostics=diagnostics)
                    calls += int(diagnostics["engine_calls"])
                    if result is None:
                        status, old, keys[index] = "invalid", None, -1e9
                    else:
                        status, old = archive.add(
                            genotype, result["fitness"], result["bc"], info=result["info"])
                        if status == "new":
                            keys[index], added = 2e12 + result["fitness"], True
                        elif status == "improved":
                            keys[index], added = 1e12 + result["fitness"] - (old or 0), True
                        else:
                            keys[index] = result["fitness"]
                    row = _record(proposals, "cma", "cma_population", genotype,
                                  result, status, old, diagnostics, calls, 1,
                                  float(emitter.sigma))
                    row.update(emitter=emitter_index, generation=int(emitter.gen),
                               parent=[round(float(value), 6) for value in parent])
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    completed += 1
                if completed == len(population):
                    emitter.tell(np.argsort(-keys))
                    if (not added) or emitter.restart_needed:
                        emitters[emitter_index] = new_emitter(
                            int(rng.integers(0, 1 << 30)))
                if stop:
                    break
    payload = _archive_payload(cfg, archive, "cma", seed, budget, calls, proposals,
                               proposals, time.time() - started)
    _atomic_json(output / "archive.json", payload)
    return payload


def _run_one(task: Mapping[str, Any]) -> dict:
    output = Path(task["out_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    specs_path = Path(task["environment_specs"])
    environment_id = task["environment_id"]
    cfg = fixed_environment(environment_id, specs_path)
    specs = load_fixed_environment_specs(specs_path)
    spec = next(row for row in specs if row["environment_id"] == environment_id)
    payload = (_run_cma(task, cfg, output) if task["algorithm"] == "cma"
               else _run_random_or_learned(task, cfg, spec, output))
    archive_path, trace_path = output / "archive.json", output / "trace.jsonl"
    return {
        "environment_id": environment_id, "algorithm": task["algorithm"],
        "seed": int(task["seed"]),
        "engine_call_budget": int(task["engine_call_budget"]),
        "engine_calls_used": int(payload["meta"]["engine_calls_used"]),
        "proposals_evaluated": int(payload["meta"]["proposals_evaluated"]),
        "cells": len(payload["cells"]),
        "best_fitness": max((row["fitness"] for row in payload["cells"]), default=None),
        "wall_seconds": payload["meta"]["wall_seconds"],
        "archive": str(archive_path.resolve()), "archive_sha256": _sha256(archive_path),
        "trace": str(trace_path.resolve()), "trace_sha256": _sha256(trace_path),
    }


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-specs",
                        default=str(DEFAULT_FIXED_ENVIRONMENT_SPECS))
    parser.add_argument("--environment-ids", required=True)
    parser.add_argument("--algorithms", default="cma,random,learned")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--engine-call-budget", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--pairwise-model", required=True)
    parser.add_argument("--global-pool-size", type=int, default=20)
    parser.add_argument("--local-pool-size", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    algorithms = _csv(args.algorithms)
    if not algorithms or set(algorithms) - set(ALGORITHMS):
        parser.error(f"algorithms must be a subset of {ALGORITHMS}")
    seeds = [int(value) for value in _csv(args.seeds)]
    environment_ids = _csv(args.environment_ids)
    if min(args.engine_call_budget, args.workers, args.global_pool_size,
           args.local_pool_size) <= 0:
        parser.error("budgets, workers, and pool sizes must be positive")
    specs_path = Path(args.environment_specs).resolve()
    known = {row["environment_id"] for row in load_fixed_environment_specs(specs_path)}
    if set(environment_ids) - known:
        parser.error(f"unknown environments: {sorted(set(environment_ids) - known)}")
    global_model = Path(args.global_model).resolve()
    pairwise_model = Path(args.pairwise_model).resolve()
    for path in (global_model, pairwise_model):
        if not path.exists():
            parser.error(f"missing model: {path}")
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    tasks = []
    for algorithm in algorithms:
        for environment_id in environment_ids:
            for seed in seeds:
                tasks.append({
                    "algorithm": algorithm, "environment_id": environment_id,
                    "seed": seed, "engine_call_budget": args.engine_call_budget,
                    "environment_specs": str(specs_path),
                    "global_model": str(global_model),
                    "pairwise_model": str(pairwise_model),
                    "global_pool_size": args.global_pool_size,
                    "local_pool_size": args.local_pool_size,
                    "out_dir": str(output / algorithm / environment_id / f"seed{seed}"),
                })
    started = time.time()
    results = []
    with Pool(min(args.workers, len(tasks))) as pool:
        for index, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results.append(result)
            print(f"[call-budget] {index}/{len(tasks)} {result['algorithm']} "
                  f"{result['environment_id']} seed={result['seed']} "
                  f"calls={result['engine_calls_used']} best={result['best_fitness']}",
                  flush=True)
    results.sort(key=lambda row: (row["algorithm"], row["environment_id"], row["seed"]))
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment_specs": str(specs_path),
        "environment_specs_sha256": _sha256(specs_path),
        "environment_ids": environment_ids, "algorithms": algorithms, "seeds": seeds,
        "engine_call_budget": args.engine_call_budget,
        "budget_rule": "do not start an evaluation when fewer than 9 calls remain",
        "models": {"global": {"path": str(global_model), "sha256": _sha256(global_model)},
                   "pairwise": {"path": str(pairwise_model),
                                "sha256": _sha256(pairwise_model)}},
        "proposal_pools": {"global": args.global_pool_size,
                           "local_pairwise": args.local_pool_size},
        "runs": results, "wall_seconds": time.time() - started,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str((output / "manifest.json").resolve()),
                      "runs": len(results), "wall_seconds": manifest["wall_seconds"]},
                     indent=2))


if __name__ == "__main__":
    main()
