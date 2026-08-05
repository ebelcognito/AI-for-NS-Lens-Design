"""Experiment: does per-parameter CMA step scaling find more good designs?

Motivation. `CMA_ME` historically gave every genotype dimension the SAME absolute
initial step, `sigma0 * mean(hi - lo)`. With the phase-0 doublet bounds that step is
3.24 world units, which is 3.4% of cx's range but 360% of nidx's and 324% of refl's.
Since `CMAES.ask()` clips to bounds, those three dimensions come out of every generation
pinned to their limits -- sampled bang-bang rather than searched. `param_scaling=
"normalized"` moves the emitter into the unit cube so sigma0 means "this fraction of
each parameter's own range" in every dimension.

Design. Two arms (isotropic = frozen baseline, normalized = the fix) x N seeds, equal
evaluation budget. Everything else -- evaluator, bounds, archive, flux floor, emitter
count, popsize -- is identical, so the only free variable is the step scaling.

Run:  python3 ns_probe/run_sigma_scaling.py --seeds 5 --evals 8000
Plot: python3 ns_probe/analyze_sigma_scaling.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = ("isotropic", "normalized")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--evals", type=int, default=8000)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", type=str, default=str(REPO / "ns_probe" / "results_sigma_scaling"))
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs = []

    for arm in ARMS:
        for seed in range(args.seeds):
            out = root / f"{arm}_seed{seed}"
            if (out / "archive.json").exists():
                print(f"[sweep] skip {arm} seed{seed} (already done)", flush=True)
                runs.append(dict(arm=arm, seed=seed, out=str(out)))
                continue
            cmd = [sys.executable, str(REPO / "ns_probe" / "run.py"),
                   "--evals", str(args.evals), "--seed", str(seed),
                   "--sigma0", str(args.sigma0), "--param-scaling", arm,
                   "--out", str(out)]
            print(f"[sweep] {arm} seed{seed} ...", flush=True)
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[sweep] FAILED {arm} seed{seed}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}",
                      flush=True)
                continue
            print(f"[sweep]   done in {time.time() - t:.0f}s  |  {r.stdout.strip().splitlines()[-1]}",
                  flush=True)
            runs.append(dict(arm=arm, seed=seed, out=str(out)))

    (root / "manifest.json").write_text(json.dumps(
        dict(arms=list(ARMS), seeds=args.seeds, evals=args.evals, sigma0=args.sigma0,
             wall_seconds=time.time() - t0, runs=runs), indent=2))
    print(f"[sweep] {len(runs)} runs in {time.time() - t0:.0f}s -> {root}", flush=True)


if __name__ == "__main__":
    main()
