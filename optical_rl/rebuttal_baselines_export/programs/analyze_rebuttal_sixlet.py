"""Rebuttal sixlet baseline analysis: HyperLens vs the reviewer-requested baselines.

Reads a manifest of refined-pool JSONs (one per method x seed), all produced by
the *identical* L-BFGS refinement pipeline (tools/refine_all_parallel.py), and
computes the head-to-head comparison on:

  * epsilon-net diversity at eps=0.20 (normalised L2 / sqrt(D)) at refined-RMS
    thresholds {<30um, <100um, <1mm}  -- the paper's structural-diversity metric
  * refined-RMS distribution (best / median / mean)
  * pool size and sub-30um count

Every metric is method-agnostic (computed from theta + refined_fitness alone),
so it means the same thing for a QD archive, a multi-start L-BFGS pool, or a
CMA-ES restart pool.

Manifest schema (JSON):
{
  "sixlet_config": "optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml",
  "methods": {
     "seidel_s2s3":      {"42": "<path>", "43": "<path>"},   # reference (the method)
     "sa_ga":            {"42": "<path>", "43": "<path>"},
     "multistart_lbfgs": {"42": "<path>", "43": "<path>"},
     "cmaes_restarts":   {"42": "<path>", "43": "<path>"},
     "randbc":           {"42": "<path>", "43": "<path>"},
     "nograd":           {"42": "<path>", "43": "<path>"},
     "paramspace":       {"42": "<path>", "43": "<path>"}
  }
}

Usage:
  python analyze_rebuttal_sixlet.py --manifest <manifest.json> [--eps 0.20] [--ref seidel_s2s3]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RMS_THRESHOLDS = [(-0.030, "<30um"), (-0.100, "<100um"), (-1.000, "<1mm")]


def epsilon_net(designs: List[Tuple[float, list]], eps: float,
                lo: np.ndarray, hi: np.ndarray) -> int:
    """Greedy eps-net cardinality; designs = [(fitness, theta)]."""
    rng = np.where((hi - lo) < 1e-12, 1e-12, hi - lo)
    sqrtD = np.sqrt(len(lo))
    accepted: List[np.ndarray] = []
    for _fit, theta in sorted(designs, key=lambda x: -x[0]):
        u = (np.asarray(theta, dtype=np.float64) - lo) / rng
        if not accepted:
            accepted.append(u)
            continue
        if min(np.linalg.norm(u - a) / sqrtD for a in accepted) > eps:
            accepted.append(u)
    return len(accepted)


def load_pool(path: str):
    """Return list of design dicts with keys refined_fitness, theta, rms_um."""
    if not path or not os.path.exists(path):
        return None
    blob = json.loads(Path(path).read_text())
    return blob.get("designs", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--ref", default="seidel_s2s3",
                    help="method name used as the reference row for ratios")
    ap.add_argument("--out", default=None, help="optional JSON dump of the table")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    from optical_rl.qd.lensmega.config.loader import load_config
    cfg = load_config(man["sixlet_config"])
    bounds = np.asarray(cfg.problem.theta_bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]

    methods: Dict[str, Dict[str, str]] = man["methods"]

    # rows[method][thresh_label] = list over seeds of eps-net counts
    rows: Dict[str, dict] = {}
    print(f"\n{'='*104}")
    print(f"REBUTTAL SIXLET HEAD-TO-HEAD  (eps={args.eps}, "
          f"D={len(lo)}, ref={args.ref})")
    print(f"{'='*104}")

    for method, seedmap in methods.items():
        rows[method] = {lbl: [] for _, lbl in RMS_THRESHOLDS}
        rows[method]["_pool"] = []
        rows[method]["_sub30"] = []
        rows[method]["_best_um"] = []
        rows[method]["_median_um"] = []
        rows[method]["_mean_um"] = []
        print(f"\n##### {method} #####")
        for seed, path in seedmap.items():
            designs = load_pool(path)
            if designs is None:
                print(f"  seed={seed}: MISSING pool ({path})")
                continue
            rms = np.array([d["rms_um"] for d in designs], dtype=np.float64)
            n_sub30 = int((rms < 30).sum())
            rows[method]["_pool"].append(len(designs))
            rows[method]["_sub30"].append(n_sub30)
            if rms.size:
                rows[method]["_best_um"].append(float(rms.min()))
                rows[method]["_median_um"].append(float(np.median(rms)))
                rows[method]["_mean_um"].append(float(rms.mean()))
            line = [f"  seed={seed}: pool={len(designs):>4d} sub30={n_sub30:>3d}"]
            for thr, lbl in RMS_THRESHOLDS:
                pool = [(d["refined_fitness"], d["theta"]) for d in designs
                        if d["refined_fitness"] >= thr]
                e = epsilon_net(pool, args.eps, lo, hi) if pool else 0
                rows[method][lbl].append(e)
                line.append(f"{lbl}:eps={e:>3d}(n={len(pool):>4d})")
            if rms.size:
                line.append(f"best={rms.min():.1f}um med={np.median(rms):.1f}um")
            print("  ".join(line))

    # ---- aggregated table (mean over seeds) ----
    def agg(vals):
        a = np.array(vals, dtype=np.float64)
        return (a.mean(), a.std()) if a.size else (float("nan"), float("nan"))

    ref = args.ref
    print(f"\n{'='*104}")
    print(f"AGGREGATED (mean over seeds) -- eps-net @ eps={args.eps}, "
          f"ratio vs {ref}")
    print(f"{'='*104}")
    hdr = f"{'method':<20}"
    for _, lbl in RMS_THRESHOLDS:
        hdr += f"{lbl+' eps':>16}{'ratio':>8}"
    hdr += f"{'best um':>10}{'med um':>10}{'pool':>7}"
    print(hdr)
    ref_eps = {lbl: agg(rows.get(ref, {}).get(lbl, []))[0]
               for _, lbl in RMS_THRESHOLDS}
    table_out = {}
    for method in methods:
        r = rows[method]
        s = f"{method:<20}"
        table_out[method] = {}
        for _, lbl in RMS_THRESHOLDS:
            m, sd = agg(r[lbl])
            ratio = (m / ref_eps[lbl]) if ref_eps.get(lbl) else float("nan")
            s += f"{m:>10.1f}±{sd:<4.1f}{ratio:>7.2f}x"
            table_out[method][lbl] = {"eps_mean": m, "eps_std": sd, "ratio": ratio}
        bm, _ = agg(r["_best_um"]); md, _ = agg(r["_median_um"]); pl, _ = agg(r["_pool"])
        s += f"{bm:>10.1f}{md:>10.1f}{pl:>7.0f}"
        table_out[method].update({"best_um": bm, "median_um": md, "pool": pl})
        print(s)

    print(f"\n(reference = {ref}; ratio>1 means that method finds MORE eps-distinct "
          f"refined designs than {ref}.)")
    print("Compute: all methods matched at 100k forward evals. Gradient budget -- "
          "seidel/randbc/nograd/paramspace: CMA-MEGA ~1 grad/gen; multistart_lbfgs: "
          "~1 grad/eval; cmaes_restarts: 0 (derivative-free).")

    if args.out:
        Path(args.out).write_text(json.dumps(table_out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
