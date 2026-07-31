"""M5 baseline: derivative-free CMA-ES with restarts (LDG-EA-style).

The closest prior "global search on the geometric design space" method class
(e.g. LDG-EA) runs CMA-ES directly on the raw lens parameters, driven purely
by the objective function, with restarts to escape local minima.  This is the
honest matched-compute proxy for that class: it uses NO gradients and NO
optical descriptors — just black-box CMA-ES over the normalised theta box, with
IPOP-style restarts (random restart + growing population) until the evaluation
budget is spent.

Every objective evaluation counts as one forward eval; the run stops when the
cumulative budget is reached.  Because CMA-ES is derivative-free, the gradient
budget is exactly zero (reported for the matched-compute table).

The full evaluation history is dumped as the candidate pool; downstream the
above-threshold subset is refined through the *identical* L-BFGS pipeline used
for every other method, then compared on epsilon-net diversity.

Usage::

    python m5_cmaes_restarts.py \
        --config optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml \
        --budget 100000 --seed 42 --out /tmp/m5.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)

import cma


def _build_problem(config_path: Path):
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
    cfg = load_config(config_path)
    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor)
    problem._ensure_backend()
    return cfg, problem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=100000,
                    help="total forward evaluations across all restarts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma0", type=float, default=0.25,
                    help="initial CMA step size in normalised [0,1] coords")
    ap.add_argument("--popsize", type=int, default=None,
                    help="base population (default: CMA rule 4+3ln(D))")
    ap.add_argument("--popsize-cap", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method-name", default="cmaes_restarts")
    args = ap.parse_args()

    cfg, problem = _build_problem(args.config)
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng_span = np.clip(hi - lo, 1e-12, None)
    D = lo.size

    def denorm(x: np.ndarray) -> np.ndarray:
        return lo + np.clip(x, 0.0, 1.0) * rng_span

    def objective(x: np.ndarray):
        """Return (cost=-fitness, theta, metrics). Cost minimised by CMA."""
        theta = denorm(np.asarray(x, dtype=np.float64))
        try:
            r = problem.evaluate(theta, with_grads=False)
            fit = float(r.fitness)
            metrics = {k: float(v) for k, v in r.metrics.items()}
        except Exception:
            fit, metrics = -1e9, {}
        if not np.isfinite(fit):
            fit = -1e9
        return -fit, theta, metrics

    base_pop = args.popsize or (4 + int(3 * np.log(D)))
    print(f"[m5] problem={args.config.name} theta_dim={D} budget={args.budget} "
          f"base_pop={base_pop} sigma0={args.sigma0}", flush=True)

    designs: List[dict] = []
    total_evals = 0
    restart = 0
    popsize = base_pop
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    while total_evals < args.budget:
        x0 = rng.random(D)  # random restart in the normalised box
        es = cma.CMAEvolutionStrategy(
            list(x0), args.sigma0,
            {
                "bounds": [0.0, 1.0],
                "popsize": int(popsize),
                "seed": int(args.seed + 1 + restart),
                "maxfevals": int(args.budget - total_evals),
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )
        gen = 0
        while not es.stop() and total_evals < args.budget:
            X = es.ask()
            costs = []
            for x in X:
                cost, theta, metrics = objective(x)
                costs.append(cost)
                designs.append({
                    "theta": [float(v) for v in theta],
                    "fitness": float(-cost),
                    "metrics": metrics,
                    "restart": restart,
                })
                total_evals += 1
            es.tell(X, costs)
            gen += 1
        best_fit = max((d["fitness"] for d in designs), default=float("-inf"))
        elapsed = time.perf_counter() - t0
        print(f"  [restart {restart}] pop={popsize} gens={gen} "
              f"evals={total_evals}/{args.budget} best_fit={best_fit:+.4f} "
              f"wall={elapsed:.0f}s", flush=True)
        restart += 1
        popsize = min(popsize * 2, args.popsize_cap)  # IPOP growth

    wall = time.perf_counter() - t0
    # Truncate any overshoot beyond budget for a clean matched-compute count.
    if len(designs) > args.budget:
        designs = designs[: args.budget]

    out = {
        "name": args.method_name,
        "n_evaluations": len(designs),
        "n_restarts": restart,
        "gradient_evals": 0,  # derivative-free
        "wall_seconds": float(wall),
        "config": str(args.config),
        "designs": designs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[m5] wrote {args.out}  n_evals={len(designs)}  restarts={restart}  "
          f"wall={wall:.0f}s", flush=True)
    if designs:
        fits = np.array([d["fitness"] for d in designs])
        print(f"[m5] best fitness: {fits.max():+.4f}")
        print(f"[m5] fit >= -0.2: {int((fits >= -0.2).sum())} / {len(fits)}")
        print(f"[m5] fit >= -1.0: {int((fits >= -1.0).sum())} / {len(fits)}")


if __name__ == "__main__":
    main()
