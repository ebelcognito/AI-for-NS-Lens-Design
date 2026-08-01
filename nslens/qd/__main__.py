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
    ap.add_argument("--evals", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--emitters", type=int)
    ap.add_argument("--popsize", type=int)
    ap.add_argument("--sigma0", type=float)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.list or not args.experiment:
        print("experiments:")
        for n in experiments.names():
            c = experiments.get(n)
            print(f"  {n:16s} fitness={c.fitness:32s} bc={'+'.join(c.descriptors)}"
                  + ("  [lock_tilt]" if c.lock_tilt else "")
                  + (f"  fields={list(c.fields)}" if c.fields else ""))
        print(f"\nfitness:     {', '.join(fitness_registry.names())}")
        print(f"descriptors: {', '.join(descriptor_registry.names())}")
        return 0 if args.list else 1

    cfg = experiments.get(args.experiment)
    for k in ("evals", "seed", "emitters", "popsize", "sigma0"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    out = Path(args.out) if args.out else Path("runs") / cfg.name
    run(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
