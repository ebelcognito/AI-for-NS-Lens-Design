"""Run paired held-task CMA/EvoTF/Optical-EvoTF online comparisons and ladders."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from nslens.learning.optical_evotf.checkpoint import load_checkpoint
from nslens.learning.optical_evotf.emitter import OpticalEvoTFEmitter
from nslens.qd import experiments
from nslens.qd.driver import run
from nslens.search.evotf_emitter import EvoTFEmitter
from nslens.search.optimizer_transition import optimizer_fingerprints


LADDER = {
    "role": ("parameter_role",),
    "role_environment": ("parameter_role", "environment"),
    "role_environment_status": (
        "parameter_role", "environment", "status_failure"),
    "role_environment_status_route": (
        "parameter_role", "environment", "status_failure", "route"),
    "full": (
        "parameter_role", "environment", "status_failure", "route", "archive"),
}


def _checkpoint_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in {*LADDER, "one_hot_full"}:
        raise argparse.ArgumentTypeError(
            "checkpoint must be NAME=PATH; NAME is a ladder rung or one_hot_full")
    return name, Path(path)


def _validate_checkpoint(
    path: Path,
    expected_adapters: tuple[str, ...],
    context_len: int,
    *,
    categorical_mode: str,
    population_size: int,
):
    loaded = load_checkpoint(path)
    model = loaded["manifest"]["model"]
    training = loaded["manifest"].get("training") or {}
    if tuple(model["enabled_adapters"]) != expected_adapters:
        raise ValueError(f"{path} adapter ladder does not match {expected_adapters}")
    if int(model["max_context_len"]) != context_len:
        raise ValueError(f"{path} context length does not match --context-len")
    if model.get("categorical_mode") != categorical_mode:
        raise ValueError(f"{path} categorical mode does not match {categorical_mode}")
    if int(training.get("population_size", -1)) != population_size:
        raise ValueError(f"{path} population size does not match --popsize")
    if int(training.get("dimensions", -1)) != 9:
        raise ValueError(f"{path} was not trained for the nine-gene representation")
    if training.get("phase") != "hindsight":
        raise ValueError(f"{path} is not a final hindsight-trained checkpoint")
    if training.get("test_metrics_read") is not False:
        raise ValueError(f"{path} does not certify test-blind model selection")
    return loaded["manifest"]


def _data_hashes(manifest: dict) -> dict:
    data = (manifest.get("training") or {}).get("data") or {}
    return {
        split: [row.get("sha256") for row in data.get(split, [])]
        for split in ("train", "validation")
    }


def _configure(base_name, *, arm, seed, evals, emitters, popsize, context_len,
               checkpoint=None, adapters=(), categorical_mode="learned"):
    cfg = experiments.get(base_name)
    if cfg.param_scaling != "normalized" or len(cfg.problem.param_names) != 9 \
            or not getattr(cfg.problem, "relative_sag", False):
        raise ValueError("ablation requires a normalized nine-gene relative-sag task")
    cfg.emitter = arm
    cfg.seed = seed
    cfg.evals = evals
    cfg.emitters = emitters
    cfg.popsize = popsize
    cfg.evotf_context_len = context_len
    cfg.evotf_checkpoint = str(checkpoint) if checkpoint else None
    cfg.optical_adapters = adapters
    cfg.optical_categorical_mode = categorical_mode
    cfg.trace = True
    cfg.optimizer_transitions = False
    return cfg


def _warm(cfg) -> float:
    if cfg.emitter == "cma":
        return 0.0
    started = time.perf_counter()
    dimensions = len(cfg.problem.param_names)
    bounds = (np.zeros(dimensions), np.ones(dimensions))
    if cfg.emitter == "evotf":
        emitter = EvoTFEmitter(
            np.full(dimensions, 0.5), cfg.sigma0, popsize=cfg.popsize,
            seed=2_147_483_647, bounds=bounds,
            max_context_len=cfg.evotf_context_len)
    else:
        fingerprints, context = optimizer_fingerprints(cfg, cfg.problem)
        emitter = OpticalEvoTFEmitter(
            np.full(dimensions, 0.5), cfg.sigma0, popsize=cfg.popsize,
            seed=2_147_483_647, bounds=bounds,
            max_context_len=cfg.evotf_context_len,
            parameter_names=cfg.problem.param_names,
            environment_context={**context, "fingerprints": fingerprints},
            checkpoint_path=Path(cfg.evotf_checkpoint),
            enabled_adapters=(() if cfg.emitter == "numeric_evotf"
                              else tuple(cfg.optical_adapters)),
            categorical_mode=cfg.optical_categorical_mode)
    emitter.ask()
    observations = [{
        "status": "invalid", "rank_percentile": index / max(cfg.popsize - 1, 1),
        "clipping_mask": [0] * dimensions,
        "failure": {"stage": "invalid_geometry", "reason": "warmup"},
    } for index in range(cfg.popsize)]
    emitter.tell(np.arange(cfg.popsize), observations)
    return time.perf_counter() - started


def _trace_metrics(path: Path) -> dict:
    first_valid = None
    first_improving = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if first_valid is None and row["status"] != "invalid":
                first_valid = row["eval"]
            if first_improving is None and row["status"] in ("new", "improved"):
                first_improving = row["eval"]
    return {"calls_to_first_valid": first_valid,
            "calls_to_first_improving": first_improving}


def _metrics(payload: dict, run_dir: Path, compile_seconds: float) -> dict:
    cells = payload["cells"]
    routes = {cell.get("dom_signature") for cell in cells if cell.get("dom_signature")}
    deep_routes = {route for route in routes if route.count("TIR@") >= 4}
    telemetry = payload["meta"].get("emitter_telemetry") or {}
    route_persistence = [cell.get("intended_dominant_all") for cell in cells
                         if cell.get("intended_dominant_all") is not None]
    result = {
        "candidate_evaluations": payload["meta"]["evals"],
        "engine_calls": payload["meta"]["engine_calls"],
        "coverage": len(cells) / float(np.prod(payload["meta"]["bc_bins"])),
        "filled": len(cells),
        "qd_score": float(sum(float(cell["fitness"]) for cell in cells)),
        "best_fitness": max((float(cell["fitness"]) for cell in cells), default=None),
        "route_families": len(routes),
        "deep_fold_route_families": len(deep_routes),
        "invalid_rate": payload["meta"]["invalid"] / max(payload["meta"]["evals"], 1),
        "clipped_candidate_rate": telemetry.get("clipped_candidate_rate"),
        "route_persistence": (float(np.mean(route_persistence))
                              if route_persistence else None),
        "optimizer_seconds": (float(telemetry.get("ask_seconds", 0.0))
                              + float(telemetry.get("inference_seconds", 0.0))),
        "wall_seconds": payload["meta"]["wall_seconds"],
        "compile_seconds": compile_seconds,
        "history": payload["history"],
        "archive": str((run_dir / "archive.json").resolve()),
        "trace": str((run_dir / "trace.jsonl").resolve()),
    }
    result.update(_trace_metrics(run_dir / "trace.jsonl"))
    return result


def _run_arm_seed(task: dict) -> dict:
    """Run one complete paired unit in an isolated worker process."""
    spec = task["spec"]
    cfg = _configure(
        task["experiment"], arm=spec["arm"], seed=task["seed"],
        evals=task["evals"], emitters=task["emitters"],
        popsize=task["popsize"], context_len=task["context_len"],
        checkpoint=spec.get("checkpoint"), adapters=spec.get("adapters", ()),
        categorical_mode=spec.get("categorical_mode", "learned"))
    compile_seconds = _warm(cfg)
    run_dir = task["output"] / "raw" / task["name"] / f"seed{task['seed']}"
    payload = run(cfg, run_dir)
    return {
        "arm": task["name"], "seed": task["seed"],
        "metrics": _metrics(payload, run_dir, compile_seconds),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="fold24_clean_gated")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evals", type=int, default=6000)
    parser.add_argument("--emitters", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="isolated arm/seed worker processes (default: 1, sequential)")
    parser.add_argument("--numeric-checkpoint", type=Path, required=True)
    parser.add_argument("--optical-checkpoint", type=Path, required=True)
    parser.add_argument("--ladder-checkpoint", action="append", type=_checkpoint_spec,
                        default=[])
    parser.add_argument("--output", type=Path,
                        default=Path("results/fold_evotf_ablation"))
    args = parser.parse_args(argv)
    if args.evals <= 0 or args.evals % args.popsize:
        parser.error("evals must be positive and divisible by popsize")
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    try:
        manifests = {
            "trained_numeric": _validate_checkpoint(
                args.numeric_checkpoint, (), args.context_len,
                categorical_mode="learned", population_size=args.popsize),
            "optical_full": _validate_checkpoint(
                args.optical_checkpoint, LADDER["full"], args.context_len,
                categorical_mode="learned", population_size=args.popsize),
        }
        ladder_paths = dict(args.ladder_checkpoint)
        for name, path in ladder_paths.items():
            adapters = LADDER["full"] if name == "one_hot_full" else LADDER[name]
            manifests[name] = _validate_checkpoint(
                path, adapters, args.context_len,
                categorical_mode=("one_hot" if name == "one_hot_full" else "learned"),
                population_size=args.popsize)
        expected_data = _data_hashes(manifests["trained_numeric"])
        if not expected_data["train"] or not expected_data["validation"]:
            raise ValueError("checkpoints must record training and validation data hashes")
        for name, manifest in manifests.items():
            if _data_hashes(manifest) != expected_data:
                raise ValueError(
                    f"{name} was not trained on the same train/validation corpus")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    arm_specs = {
        "cma": {"arm": "cma"},
        "released_evotf": {"arm": "evotf"},
        "trained_numeric": {
            "arm": "numeric_evotf", "checkpoint": args.numeric_checkpoint},
        "optical_full": {
            "arm": "optical_evotf", "checkpoint": args.optical_checkpoint,
            "adapters": LADDER["full"]},
    }
    for name, path in ladder_paths.items():
        arm_specs[f"ablation_{name}"] = {
            "arm": "optical_evotf", "checkpoint": path,
            "adapters": LADDER["full"] if name == "one_hot_full" else LADDER[name],
            "categorical_mode": "one_hot" if name == "one_hot_full" else "learned",
        }

    tasks = [
        {
            "experiment": args.experiment, "seed": seed, "evals": args.evals,
            "emitters": args.emitters, "popsize": args.popsize,
            "context_len": args.context_len, "output": args.output,
            "name": name, "spec": spec,
        }
        for seed in args.seeds
        for name, spec in arm_specs.items()
    ]
    if args.jobs == 1:
        records = [_run_arm_seed(task) for task in tasks]
    else:
        # Spawn is deliberate: forking a process after JAX has initialized its thread
        # pools can deadlock. Each complete search remains an independent paired unit.
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.jobs, len(tasks)), mp_context=context) as pool:
            records = list(pool.map(_run_arm_seed, tasks))
    records.sort(key=lambda record: (int(record["seed"]), record["arm"]))
    base_cfg = experiments.get(args.experiment)
    fingerprints, context = optimizer_fingerprints(base_cfg, base_cfg.problem)
    result = {
        "schema_version": "nslens.fold_evotf_ablation.v1",
        "experiment": args.experiment,
        "paired": {
            "seeds": args.seeds, "candidate_budget": args.evals,
            "population_size": args.popsize, "emitter_count": args.emitters,
            "worker_processes": min(args.jobs, len(tasks)),
            "same_archive_seed_and_verifier": True,
        },
        "fingerprints": fingerprints,
        "environment": context,
        "checkpoint_manifests": manifests,
        "arms": {
            name: {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in spec.items()}
            for name, spec in arm_specs.items()
        },
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
