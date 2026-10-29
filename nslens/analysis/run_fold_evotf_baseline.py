"""Run a small equal-candidate folded-optics CMA versus released-EvoTF benchmark.

The default is the current hardened, normalized, relative-sag nine-gene environment.
This is the zero-shot arm only: EvoTF comes directly from evosax 0.3.1 and receives no
optical metadata.  JAX compilation is warmed with a throwaway emitter and reported
separately from the timed optical searches.

Example::

    python -m nslens.analysis.run_fold_evotf_baseline \
        --evals 20 --seed 0 --output results/fold_evotf_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nslens.qd import experiments
from nslens.qd.driver import run
from nslens.search.evotf_emitter import (
    EVOSAX_UPSTREAM_COMMIT,
    EVOSAX_VERSION,
    EVOTF_CHECKPOINT,
    EVOTF_CHECKPOINT_SHA256,
    EvoTFEmitter,
)


def _configure(name: str, *, emitter: str, evals: int, seed: int,
               emitter_count: int, popsize: int, context_len: int):
    cfg = experiments.get(name)
    if cfg.param_scaling != "normalized":
        raise ValueError(f"{name} is not a normalized-coordinate experiment")
    if len(cfg.problem.param_names) != 9:
        raise ValueError(f"{name} has {len(cfg.problem.param_names)} genes, expected 9")
    if not getattr(cfg.problem, "relative_sag", False):
        raise ValueError(f"{name} does not use the relative-sag convention")
    cfg.emitter = emitter
    cfg.evals = evals
    cfg.seed = seed
    cfg.emitters = emitter_count
    cfg.popsize = popsize
    cfg.evotf_context_len = context_len
    cfg.trace = True
    cfg.label = f"{cfg.label}_{emitter}_zero_shot"
    return cfg


def _warm_evotf(popsize: int, dimension: int, sigma0: float,
                 context_len: int) -> dict:
    started = time.perf_counter()
    emitter = EvoTFEmitter(
        np.full(dimension, 0.5), sigma0, popsize=popsize, seed=2_147_483_647,
        bounds=(np.zeros(dimension), np.ones(dimension)),
        max_context_len=context_len,
    )
    emitter.ask()
    emitter.tell(np.arange(popsize))
    summary = emitter.state_summary()
    return {
        "seconds": time.perf_counter() - started,
        "ask_seconds": summary["ask_seconds"],
        "tell_inference_seconds": summary["inference_seconds"],
    }


def _result_summary(payload: dict) -> dict:
    meta = payload["meta"]
    return {
        "candidate_evaluations": meta["evals"],
        "engine_calls": meta["engine_calls"],
        "filled": len(payload["cells"]),
        "coverage": (len(payload["cells"])
                     / float(np.prod(np.asarray(meta["bc_bins"], dtype=int)))),
        "best_fitness": max(
            (float(cell["fitness"]) for cell in payload["cells"]),
            default=None,
        ),
        "invalid": meta["invalid"],
        "wall_seconds": meta["wall_seconds"],
        "emitter_telemetry": meta.get("emitter_telemetry"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="fold24_clean_gated")
    parser.add_argument("--evals", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emitters", type=int, default=1)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--context-len", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("results/fold_evotf_smoke"))
    args = parser.parse_args(argv)
    if args.evals <= 0 or args.emitters <= 0 or args.popsize <= 0:
        parser.error("evals, emitters, and popsize must be positive")
    if args.context_len <= 0:
        parser.error("context-len must be positive")
    if args.evals % args.popsize:
        parser.error("evals must be divisible by popsize for exact call accounting")

    try:
        cma_cfg = _configure(
            args.experiment, emitter="cma", evals=args.evals, seed=args.seed,
            emitter_count=args.emitters, popsize=args.popsize,
            context_len=args.context_len)
        evotf_cfg = _configure(
            args.experiment, emitter="evotf", evals=args.evals, seed=args.seed,
            emitter_count=args.emitters, popsize=args.popsize,
            context_len=args.context_len)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    warmup = _warm_evotf(
        args.popsize, len(evotf_cfg.problem.param_names), evotf_cfg.sigma0,
        args.context_len)
    cma_payload = run(cma_cfg, args.output / "cma")
    evotf_payload = run(evotf_cfg, args.output / "evotf")
    cma_summary = _result_summary(cma_payload)
    evotf_summary = _result_summary(evotf_payload)
    if cma_summary["candidate_evaluations"] != evotf_summary["candidate_evaluations"]:
        raise RuntimeError("paired arms used different candidate-evaluation counts")

    comparison = {
        "schema": "nslens.fold_evotf_zero_shot.v1",
        "experiment": args.experiment,
        "seed": args.seed,
        "paired_candidate_budget": cma_summary["candidate_evaluations"],
        "population_size": args.popsize,
        "emitter_count": args.emitters,
        "provenance": {
            "evosax_version": EVOSAX_VERSION,
            "evosax_upstream_commit": EVOSAX_UPSTREAM_COMMIT,
            "checkpoint": EVOTF_CHECKPOINT,
            "checkpoint_sha256": EVOTF_CHECKPOINT_SHA256,
        },
        "jax_warmup": warmup,
        "arms": {"cma": cma_summary, "released_evotf": evotf_summary},
        "archive_paths": {
            "cma": str((args.output / "cma" / "archive.json").resolve()),
            "released_evotf": str(
                (args.output / "evotf" / "archive.json").resolve()),
        },
        "call_accounting_note": (
            "Candidate verifier calls are exactly paired. engine_calls reports actual "
            "ray-engine invocations, which can differ because unchanged hardened gates "
            "reject candidates at different stages."
        ),
    }
    output_path = args.output / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
