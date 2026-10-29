"""Experiment: does per-parameter CMA step scaling find better PRISM designs?

Motivation. The QD driver gives every genotype dimension the same absolute initial step,
`sigma0 * mean(hi - lo)`. In both prism environments that step is ~1100% of the
refractive index's own range, so `CMAES.ask()`'s clip pins n to 1.50 or 2.00 on nearly
every draw -- 94% of the elites in the committed arvr_qd archive sit on an index rail.
`--param-scaling normalized` moves the emitter into the unit cube, so sigma0 means "this
fraction of each parameter's own range" in every dimension.

This is the sigma-scaling sweep (ns_probe/run_sigma_scaling.py) applied to the two prism
envs rather than the phase-0 doublet probe: same two arms, same matched budget, only the
step scaling differs.

    PYTHONPATH=.:ns_probe python ns_probe/run_prism_sigma_sweep.py --seeds 3
    PYTHONPATH=.:ns_probe python ns_probe/analyze_prism_sigma_sweep.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = ("isotropic", "normalized")
#: env -> evals, matching the committed budget for that experiment
ENVS = {"prism_fold_qd": 14000, "arvr_qd": 40000}


#: the driver's default step. An arm at this sigma0 keeps the plain `<arm>` label so the
#: first sweep's directories stay valid.
DEFAULT_SIGMA0 = 0.12


def label_of(arm: str, sigma0: float) -> str:
    return arm if sigma0 == DEFAULT_SIGMA0 else f"{arm}-sig{sigma0:g}"


def one(env: str, arm: str, sigma0: float, seed: int, evals: int, root: Path) -> dict:
    label = label_of(arm, sigma0)
    out = root / f"{env}__{label}__seed{seed}"
    rec = dict(env=env, arm=arm, sigma0=sigma0, label=label, seed=seed, out=str(out))
    if (out / "archive.json").exists():
        print(f"[sweep] skip {env} {label} seed{seed} (already done)", flush=True)
        return dict(rec, skipped=True)
    cmd = [sys.executable, "-m", "nslens.qd", env, "--evals", str(evals),
           "--seed", str(seed), "--param-scaling", arm, "--sigma0", str(sigma0),
           "--out", str(out)]
    t = time.time()
    print(f"[sweep] start {env} {label} seed{seed}", flush=True)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[sweep] FAILED {env} {label} seed{seed}\n{r.stdout[-1500:]}\n"
              f"{r.stderr[-1500:]}", flush=True)
        return dict(rec, failed=True)
    print(f"[sweep] done  {env} {label} seed{seed}  ({time.time()-t:.0f}s)", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4, help="runs in parallel")
    ap.add_argument("--envs", nargs="*", default=list(ENVS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--sigma0", type=float, nargs="*", default=[DEFAULT_SIGMA0],
                    help="CMA sigma0 values to try (one arm per value)")
    ap.add_argument("--evals", type=int, default=None, help="override both budgets")
    ap.add_argument("--out", default=str(REPO / "runs" / "prism_sigma_sweep"))
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(env, arm, sig, seed, args.evals or ENVS[env])
            for env in args.envs for arm in args.arms for sig in args.sigma0
            for seed in range(args.seeds)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(lambda j: one(*j, root), jobs))

    (root / f"sweep_{int(t0)}.json").write_text(json.dumps(
        dict(arms=args.arms, sigma0=args.sigma0, seeds=args.seeds, envs=args.envs,
             runs=runs, wall_seconds=time.time() - t0), indent=2))
    n_bad = sum(1 for r in runs if r.get("failed"))
    print(f"\n[sweep] {len(runs) - n_bad}/{len(runs)} runs ok "
          f"({time.time()-t0:.0f}s) -> {root}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
