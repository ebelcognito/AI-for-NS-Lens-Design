"""QD SEARCH with freeform surfaces — AR/VR PRISM config, on the validated BRANCHING tracer.

Uses prism_branch.trace_branching_ff (full Fresnel ray-tree + parametric aspheric faces), which
matches the JS oracle on clean designs and is conservative elsewhere (never over-credits). Same
problem/metric/archive as ns_probe/run_arvr_qd.py: 3 display sources -> 4-freeform-face TIR prism
-> eye pupil, collimation fitness, (FOV x prism-depth) archive.

Genotype = [ 9 base prism DOF | a2 a4 radial-bump per face x4 = 8 aspheric coeffs ].

  PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
  $PY -m ns_autodiff.search_prism_freeform --evals 20000 --seed 0
  $PY -m ns_autodiff.search_prism_freeform --evals 20000 --seed 0 --lock-aspheric   # arc baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from cma_me import CMAES, GridArchive                                                 # noqa: E402
import arvr_prism as AV                                                               # noqa: E402
from .prism_branch import trace_branching_ff, metrics                                 # noqa: E402

LO = np.concatenate([AV.LO, np.full(8, -2.0)])
HI = np.concatenate([AV.HI, np.full(8, 2.0)])
BASE_NAMES = list(AV.PRISM_NAMES)
PARAM_NAMES = BASE_NAMES + ["a2_A", "a4_A", "a2_B", "a4_B", "a2_T", "a4_T", "a2_D", "a4_D"]
BC_BINS = (20, 28)
BC_BOUNDS = ((0.0, 40.0), (4.0, 88.0))


def evaluate(x):
    base = x[:9]
    if not AV.geometry_valid_arvr(AV.prism_dict(base)):
        return None
    try:
        ex = trace_branching_ff(x)
    except Exception:
        return None
    m = metrics(ex)
    if not m["ordering_ok"] or m["fitness"] <= 0:
        return None
    xs = [v for r in ex if r["b"] > 0 for v in r["xs"]]
    depth = (max(xs) - min(xs)) if xs else 0.0
    if m["fov"] <= 0 or depth <= 0:
        return None
    return dict(fitness=float(m["fitness"]), bc=(float(m["fov"]), float(depth)),
                info=dict(fov=float(m["fov"]), depth=float(depth), fitness=float(m["fitness"]),
                          T_min=float(m["T_min"]), sigma_mean=float(m["sigma_mean"]),
                          thetas=[f["theta"] for f in m["fields"]],
                          sigmas=[f["sigma"] for f in m["fields"]],
                          aspheric=[float(v) for v in x[9:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--lock-aspheric", action="store_true")
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_search_prism_freeform"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = LO.copy(), HI.copy()
    if args.lock_aspheric:
        lo[9:] = 0.0; hi[9:] = 0.0
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed); scale = hi - lo; t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(lo, hi)
        sig = args.sigma0 * float(np.mean(scale[scale > 0]) if (scale > 0).any() else 1.0)
        return CMAES(np.asarray(x0, float), sig, popsize=args.popsize, seed=s, bounds=(lo, hi))

    emitters = [new_emitter(args.seed + i) for i in range(args.emitters)]
    n = 0; n_invalid = 0; history = []
    while n < args.evals:
        for ei in range(args.emitters):
            if n >= args.evals:
                break
            em = emitters[ei]; X = em.ask(); keys = np.empty(len(X)); added = False
            for k, xx in enumerate(X):
                ev = evaluate(xx); n += 1
                if ev is None:
                    keys[k] = -1e9; n_invalid += 1; continue
                st, old = archive.add(xx, ev["fitness"], ev["bc"], info=ev["info"])
                if st == "new":
                    keys[k] = 2e12 + ev["fitness"]; added = True
                elif st == "improved":
                    keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                else:
                    keys[k] = ev["fitness"]
            em.tell(np.argsort(-keys))
            if (not added) or em.restart_needed:
                emitters[ei] = new_emitter(int(rng.integers(0, 1 << 30)))
        if n // 1000 > len(history):
            rec = dict(evals=n, filled=len(archive.cells), coverage=archive.coverage,
                       best=archive.best_fitness, invalid=n_invalid)
            history.append(rec)
            print(f"[prism-ff] evals={n:6d} filled={rec['filled']:3d} "
                  f"cov={rec['coverage']*100:5.1f}% best={rec['best']:.4f} "
                  f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), fov=float(c.bc[0]), depth=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info); cells.append(rec)
    payload = dict(meta=dict(experiment=("arvr_prism_ARC_baseline" if args.lock_aspheric
                                         else "arvr_prism_FREEFORM_branching"),
                             lock_aspheric=bool(args.lock_aspheric), evals=int(n),
                             seed=int(args.seed), invalid=int(n_invalid), bc_bins=BC_BINS,
                             bc_bounds=BC_BOUNDS, bc_axes=["FOV_deg", "prism_depth"],
                             param_names=PARAM_NAMES, wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    base = out / ("archive_prism_arc.json" if args.lock_aspheric else "archive_prism_freeform.json")
    rep, k = base, 0
    while rep.exists():
        k += 1; rep = base.with_name(base.stem + f"_{k}.json")
    rep.write_text(json.dumps(payload, indent=2))
    print(f"[prism-ff] wrote {rep} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
