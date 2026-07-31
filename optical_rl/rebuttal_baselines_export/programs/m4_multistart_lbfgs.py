"""M4 baseline: multi-start L-BFGS on diverse θ initializations.

This is the "LensNet-style" baseline — many diverse starting designs,
each locally refined via L-BFGS.  Our version uses uniform-random θ as
starts (a strictly *more* diverse set than a trained NN could generate,
so this is a conservative upper bound on what NN-seeded refinement
could achieve).

Every L-BFGS closure call counts as one evaluation; the script stops
when the cumulative evaluation budget is spent.

Usage::

    python m4_multistart_lbfgs.py \
        --config optical_rl/qd/lensmega/config/examples/cooke_triplet_materials.yaml \
        --budget 50000 --n-starts 500 --max-steps 100 --out /tmp/m4.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

# Pin threads before importing torch — matches lensmega's pattern.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
torch.set_num_threads(1)


def _build_problem(config_path: Path):
    from optical_rl.qd.lensmega.config.loader import load_config
    from optical_rl.qd.lensmega.run import _build_descriptor, _build_problem
    cfg = load_config(config_path)
    descriptor = _build_descriptor(cfg)
    problem = _build_problem(cfg, descriptor)
    problem._ensure_backend()
    return cfg, problem


def _sample_starts(n: int, bounds: List[Tuple[float, float]], seed: int
                   ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return rng.uniform(lo, hi, size=(n, lo.size))


def refine_one(
    problem,
    theta_np: np.ndarray,
    bounds: List[Tuple[float, float]],
    *,
    max_steps: int,
    lr: float,
) -> Tuple[np.ndarray, float, dict, int]:
    """Run L-BFGS from this starting point; return (theta, fitness, metrics, n_evals)."""
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    theta_t = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    optimizer = torch.optim.LBFGS(
        [theta_t], lr=lr, max_iter=4,
        line_search_fn="strong_wolfe",
    )
    n_evals = [0]

    state = {
        "best_theta": theta_np.copy(),
        "best_fitness": -np.inf,
        "best_metrics": {},
    }

    def evaluate_tensor(t):
        n_evals[0] += 1
        fitness, _, metrics = problem._fitness_and_metrics(t)
        return fitness, metrics

    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            theta_t.data = torch.maximum(torch.minimum(theta_t.data, hi_t), lo_t)
        try:
            fitness, metrics = evaluate_tensor(theta_t)
        except Exception:
            return torch.tensor(1e9, dtype=torch.float32, requires_grad=True)
        loss = -fitness
        if torch.isfinite(loss) and loss.requires_grad:
            loss.backward()
        try:
            fit_val = float(fitness.detach().cpu().item())
            if np.isfinite(fit_val) and fit_val > state["best_fitness"]:
                state["best_fitness"] = fit_val
                state["best_metrics"] = {k: float(v) for k, v in metrics.items()}
                state["best_theta"] = theta_t.detach().cpu().numpy().copy()
        except Exception:
            pass
        return loss

    last_fit = -np.inf
    stagnation = 0
    for step in range(max_steps):
        try:
            optimizer.step(closure)
        except Exception:
            break
        if state["best_fitness"] > last_fit + 1e-6:
            last_fit = state["best_fitness"]
            stagnation = 0
        else:
            stagnation += 1
        if stagnation >= 5:
            break
    return (state["best_theta"], state["best_fitness"],
            state["best_metrics"], n_evals[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=50000,
                    help="total fitness evaluations across all starts")
    ap.add_argument("--n-starts", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=100,
                    help="max L-BFGS outer steps per start")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method-name", default="multi_start_lbfgs")
    args = ap.parse_args()

    cfg, problem = _build_problem(args.config)
    bounds = [tuple(b) for b in cfg.problem.theta_bounds]
    starts = _sample_starts(args.n_starts, bounds, args.seed)
    print(f"[m4] problem={args.config.name} n_starts={args.n_starts} "
          f"budget={args.budget} theta_dim={starts.shape[1]}", flush=True)

    designs: List[dict] = []
    total_evals = 0
    t0 = time.perf_counter()
    for idx, start in enumerate(starts):
        if total_evals >= args.budget:
            break
        try:
            theta_best, fit_best, metrics, n_used = refine_one(
                problem, start, bounds,
                max_steps=args.max_steps, lr=args.lr,
            )
        except Exception as e:
            print(f"  start {idx}: failed ({type(e).__name__}: {e})",
                  flush=True)
            continue
        total_evals += n_used
        if np.isfinite(fit_best):
            designs.append({
                "theta": theta_best.tolist(),
                "fitness": float(fit_best),
                "metrics": metrics,
                "n_evals_this_start": int(n_used),
            })
        if (idx + 1) % 25 == 0:
            best_so_far = max((d["fitness"] for d in designs),
                              default=float("-inf"))
            elapsed = time.perf_counter() - t0
            rate = total_evals / max(elapsed, 1e-9)
            print(f"  [{idx+1}/{args.n_starts}] evals={total_evals}/{args.budget}  "
                  f"best_fit={best_so_far:+.4f}  "
                  f"rate={rate:.0f} eval/s  wall={elapsed:.0f}s",
                  flush=True)

    wall = time.perf_counter() - t0
    out = {
        "name": args.method_name,
        "n_evaluations": int(total_evals),
        "wall_seconds": float(wall),
        "config": str(args.config),
        "n_starts_used": len(designs),
        "designs": designs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[m4] wrote {args.out}")
    if designs:
        fits = np.array([d["fitness"] for d in designs])
        print(f"[m4] best fitness: {fits.max():+.4f}")
        print(f"[m4] fit ≥ -0.2: {int((fits >= -0.2).sum())} / {len(fits)}")
        print(f"[m4] fit ≥ -0.1: {int((fits >= -0.1).sum())} / {len(fits)}")


if __name__ == "__main__":
    main()
