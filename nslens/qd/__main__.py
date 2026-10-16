"""CLI for QD experiments.

    python -m nslens.qd --list
    python -m nslens.qd fold_qd --evals 14000 --seed 0 --out runs/fold_qd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import descriptors as descriptor_registry
from . import experiments
from . import fitness as fitness_registry
from .driver import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nslens.qd")
    ap.add_argument("experiment", nargs="?", help="see --list")
    ap.add_argument("--list", action="store_true", help="list experiments and registries")
    ap.add_argument("--environment-id",
                    help="accepted environment id when experiment=fixed_fold")
    ap.add_argument("--environment-specs",
                    default=str(experiments.DEFAULT_FIXED_ENVIRONMENT_SPECS),
                    help="fixed environment_specs.json catalogue")
    ap.add_argument("--evals", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--emitters", type=int)
    ap.add_argument("--emitter",
                    choices=("cma", "evotf", "numeric_evotf", "optical_evotf"),
                    help="emitter update policy (default: cma); EvoTF arms require .[evotf]")
    ap.add_argument("--popsize", type=int)
    ap.add_argument("--sigma0", type=float)
    ap.add_argument("--evotf-context-len", dest="evotf_context_len", type=int,
                    help="released EvoTF generation-history window (default: 100)")
    ap.add_argument("--evotf-checkpoint", dest="evotf_checkpoint",
                    help="trained numeric-only or Optical-EvoTF checkpoint directory")
    ap.add_argument("--optical-adapters", nargs="+", choices=(
        "parameter_role", "environment", "status_failure", "route", "archive"))
    ap.add_argument("--optical-categorical-mode", choices=("learned", "one_hot"))
    ap.add_argument("--param-scaling", dest="param_scaling",
                    choices=("isotropic", "normalized"),
                    help="isotropic = one absolute CMA step for every dimension "
                         "(frozen default); normalized = sigma0 is a fraction of each "
                         "parameter's own range")
    ap.add_argument("--convergence-gate", dest="convergence_gate",
                    choices=("packing", "clean_pack"),
                    help="which beam verdict the convergence gate reads: packing = "
                         "width envelope only; clean_pack = also no internal focus")
    ap.add_argument("--exit-convergence", dest="require_exit_convergence",
                    action="store_true", default=None,
                    help="also require the bundle to be converging on the exit->detector "
                         "leg, which the convergence gate structurally cannot see: waist "
                         "ahead of the exit face and landing inside the scoring window")
    ap.add_argument("--exit-window-mult", dest="exit_window_mult", type=float,
                    help="fraction of the 2*focus_w scoring window the bundle may fill "
                         "at the detector (default 1.0; 0.5 costs top designs)")
    ap.add_argument("--trace", action="store_true", default=None,
                    help="write trace.jsonl: every evaluation with its parent and step")
    ap.add_argument("--optimizer-transitions", dest="optimizer_transitions",
                    action="store_true", default=None,
                    help="write one replayable full-CMA record per emitter generation")
    ap.add_argument("--resume-from", dest="resume_from",
                    help="warm-start the archive from a previous run's directory")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.list or not args.experiment:
        print("experiments:")
        for n in experiments.names():
            c = experiments.get(n)
            print(f"  {n:16s} fitness={c.fitness:32s} bc={'+'.join(c.descriptors)}"
                  + ("  [lock_tilt]" if c.lock_tilt else "")
                  + (f"  fields={list(c.fields)}" if c.fields else ""))
        try:
            fixed_ids = [row["environment_id"] for row in
                         experiments.load_fixed_environment_specs(args.environment_specs)]
            print("  fixed_fold       interaction-aware fixed environment; "
                  "use --environment-id")
            print(f"                   available: {', '.join(fixed_ids)}")
        except (OSError, ValueError) as exc:
            print(f"  fixed_fold       unavailable: {exc}")
        print(f"\nfitness:     {', '.join(fitness_registry.names())}")
        print(f"descriptors: {', '.join(descriptor_registry.names())}")
        return 0 if args.list else 1

    if args.experiment == "fixed_fold":
        if not args.environment_id:
            ap.error("fixed_fold requires --environment-id; use --list to see ids")
        try:
            cfg = experiments.fixed_environment(
                args.environment_id, args.environment_specs)
        except (OSError, ValueError, KeyError) as exc:
            ap.error(str(exc))
    else:
        if args.environment_id:
            ap.error("--environment-id is only valid with experiment=fixed_fold")
        cfg = experiments.get(args.experiment)
    for k in ("evals", "seed", "emitters", "emitter", "popsize", "sigma0",
              "evotf_context_len", "param_scaling",
              "evotf_checkpoint", "optical_adapters", "optical_categorical_mode",
              "convergence_gate", "require_exit_convergence", "exit_window_mult",
              "trace", "optimizer_transitions", "resume_from"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    out = Path(args.out) if args.out else Path("runs") / cfg.name
    run(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
