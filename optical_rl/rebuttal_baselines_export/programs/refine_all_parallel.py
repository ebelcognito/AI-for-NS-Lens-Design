#!/usr/bin/env python3
"""Parallel refinement of an entire pool of designs through L-BFGS-B.

Reads a list of (theta, raw_fitness) entries from either:
  - HL archive.pkl (filtered at --threshold)
  - SA+GA unrefined_elites_sa_ga.json (filtered at --threshold)

Distributes refinement across N worker processes (each with its own
LensProblem instance). Each worker runs the same L-BFGS-B + stability
pipeline as runner.py / sa_ga_refine_baseline.py.

Usage:
  python tools/refine_all_parallel.py \\
      --pool-from hl --hl-run <run_dir> --threshold -0.6 \\
      --workers 30 --out <output_json>
  python tools/refine_all_parallel.py \\
      --pool-from sa_ga --sa-ga-run <run_dir> --threshold -0.6 \\
      --workers 30 --out <output_json>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------- Worker ----------------
_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(cfg_dict: Dict[str, Any]) -> None:
    """Each worker builds its own LensProblem once, then reuses it."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from optical_rl.qd.lensmega.config.schema import RunConfig
    cfg = RunConfig.model_validate(cfg_dict)
    cfg.io.device = "cpu"
    _WORKER_STATE["cfg"] = cfg

    from sa_ga_refine_baseline import lbfgs_refine_one, stability_basin, BudgetCounter
    _WORKER_STATE["lbfgs_refine_one"] = lbfgs_refine_one
    _WORKER_STATE["stability_basin"] = stability_basin
    _WORKER_STATE["BudgetCounter"] = BudgetCounter

    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    _WORKER_STATE["lo"] = bounds[:, 0]
    _WORKER_STATE["hi"] = bounds[:, 1]


def _worker_refine(args: Tuple[int, List[float], float]) -> Dict[str, Any]:
    idx, theta_list, raw_fit = args
    cfg = _WORKER_STATE["cfg"]
    io = cfg.io
    lo = _WORKER_STATE["lo"]
    hi = _WORKER_STATE["hi"]
    counter = _WORKER_STATE["BudgetCounter"]()
    theta = np.asarray(theta_list, dtype=np.float64)
    try:
        rec = _WORKER_STATE["lbfgs_refine_one"](
            cfg, theta, lo, hi,
            refine_efl_penalty_weight=io.refine_efl_penalty_weight,
            refine_invalid_penalty_weight=io.refine_invalid_penalty_weight,
            refine_intersect_penalty_weight=io.refine_intersect_penalty_weight,
            refine_fnum_penalty_weight=io.refine_fnum_penalty_weight,
            refine_field_weights=(tuple(io.refine_field_weights)
                                  if io.refine_field_weights else None),
            refine_lbfgs_lr=io.refine_lbfgs_lr,
            refine_lbfgs_max_steps=io.refine_lbfgs_max_steps,
            refine_converge_tol=io.refine_converge_tol,
            counter=counter,
        )
        rec["raw_fitness"] = float(raw_fit)
        rec["raw_rms_um"] = float(-raw_fit * 1000)
        stab = _WORKER_STATE["stability_basin"](
            cfg, rec["theta"], lo, hi,
            rec["refined_fitness"],
            int(io.refine_stability_samples),
            float(io.refine_stability_perturbation),
            counter,
        )
        rec["stability"] = stab
        rec["theta"] = [float(x) for x in rec["theta"]]
        rec["pool_idx"] = idx
        return rec
    except Exception as e:
        return {
            "pool_idx": idx,
            "error": repr(e),
            "raw_fitness": float(raw_fit),
        }


# ---------------- Pool loaders ----------------

def load_hl_pool(run_dir: Path, threshold: float) -> List[Tuple[List[float], float]]:
    arch_path = run_dir / "archive.pkl"
    with arch_path.open("rb") as f:
        a = pickle.load(f)
    out = []
    for v in a.cells.values():
        if v is None: continue
        if float(v.fitness) < threshold: continue
        out.append((list(map(float, v.theta)), float(v.fitness)))
    return out


def load_sa_ga_pool(run_dir: Path, threshold: float) -> List[Tuple[List[float], float]]:
    p = run_dir / "unrefined_elites_sa_ga.json"
    if not p.exists():
        p = run_dir / "refined_designs_sa_ga.json"
    designs = json.loads(p.read_text())
    out = []
    for d in designs:
        raw = d.get("raw_fitness")
        if raw is None or raw < threshold: continue
        out.append((list(map(float, d["theta"])), float(raw)))
    return out


def load_json_pool(json_path: Path, threshold: float) -> List[Tuple[List[float], float]]:
    """Load a flat {designs:[{theta, fitness}]} pool (m4/m5 driver output)."""
    blob = json.loads(json_path.read_text())
    designs = blob.get("designs", blob if isinstance(blob, list) else [])
    out = []
    for d in designs:
        fit = d.get("fitness", d.get("raw_fitness"))
        if fit is None or float(fit) < threshold: continue
        out.append((list(map(float, d["theta"])), float(fit)))
    return out


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-from", choices=["hl", "sa_ga", "json"], required=True)
    ap.add_argument("--hl-run", default=None)
    ap.add_argument("--sa-ga-run", default=None)
    ap.add_argument("--json-pool", default=None,
                    help="flat {designs:[{theta,fitness}]} JSON (m4/m5 output)")
    ap.add_argument("--problem-config", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fitness floor for designs to refine")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Refine the top-N highest-fitness designs above "
                         "--threshold. If omitted, refine all above threshold.")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    if args.threshold is None and args.top_n is None:
        ap.error("must provide --threshold or --top-n (or both)")
    threshold = args.threshold if args.threshold is not None else -1e9

    run_dir = None
    if args.pool_from == "hl":
        run_dir = Path(args.hl_run).resolve()
        pool = load_hl_pool(run_dir, threshold)
    elif args.pool_from == "sa_ga":
        run_dir = Path(args.sa_ga_run).resolve()
        pool = load_sa_ga_pool(run_dir, threshold)
    else:  # json
        if not args.json_pool:
            ap.error("--pool-from json requires --json-pool")
        if not args.problem_config:
            ap.error("--pool-from json requires --problem-config")
        pool = load_json_pool(Path(args.json_pool).resolve(), threshold)

    if args.top_n is not None and args.top_n < len(pool):
        pool.sort(key=lambda x: -x[1])  # sort by fitness desc
        pool = pool[:args.top_n]
        print(f"[main] capped pool at top-{args.top_n} by fitness "
              f"(threshold {threshold})", flush=True)

    cfg_path = (args.problem_config or str(run_dir / "config.yaml"))
    from optical_rl.qd.lensmega.config.loader import load_config
    cfg = load_config(cfg_path)
    cfg_dict = cfg.model_dump(mode="json")

    print(f"[main] pool size: {len(pool)} designs at fitness >= "
          f"{args.threshold}", flush=True)
    print(f"[main] spawning {args.workers} workers...", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(cfg_dict,)) as p:
        tasks = [(i, theta, raw) for i, (theta, raw) in enumerate(pool)]
        results = []
        for i, rec in enumerate(p.imap_unordered(_worker_refine, tasks)):
            results.append(rec)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                print(f"[main] {i+1}/{len(tasks)} done "
                      f"(wall={time.time()-t0:.1f}s)", flush=True)

    # Sort by raw_fitness desc
    results.sort(key=lambda r: -r.get("raw_fitness", -1e9))

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    print(f"[main] success={n_ok}, errors={n_err}, "
          f"total_wall={time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pool_from": args.pool_from,
        "threshold": args.threshold,
        "n_pool": len(pool),
        "n_refined_ok": n_ok,
        "n_refined_err": n_err,
        "designs": [r for r in results if "error" not in r],
        "errors": [r for r in results if "error" in r],
    }, indent=2))
    print(f"[main] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
