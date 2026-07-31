"""QD SEARCH for DIVERSE MULTI-FOLD AR/VR prisms — fold count as a STRUCTURAL archive axis.

Why this exists: search_prism_freeform uses BC = (FOV x depth) -- performance axes that fill the
archive with one shape (mostly 1-fold). To get a controllable spread of 1..N-fold relays, this
search makes the archive axis 1 the CHIEF-RAY FOLD COUNT (a structural descriptor, per doc 03 /
12_descriptor_ablation), so each fold count lands in its own niche. Two control knobs steer the
search into the multi-fold regime (doc 02 sec.6 lever):

  --tilt-range R   : narrow the face-tilt bounds to +-R deg. A single ~45deg surface can turn the
                     light 90deg in ONE bounce; narrowing the tilt means no single surface can, so
                     the light must ZIG-ZAG through 2+ TIRs. R~25-30 biases hard toward multi-fold.
  --min-folds K    : reject designs with chief-ray fold count < K (focus the whole budget on >=K).
  --max-folds M    : fold-count archive axis spans 0..M.

Everything else (genotype, branching freeform tracer, collimation fitness) is identical to
search_prism_freeform, so the multi-fold archive is directly comparable + refinable by
refine_prism_ff (use --allow-order-change for the multi-fold cells).

  PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
  # ~10-13 min, multi-fold-biased:
  $PY -m ns_autodiff.search_prism_multifold --evals 6000 --tilt-range 28 --min-folds 2 --max-seconds 780
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from cma_me import CMAES, GridArchive                                                 # noqa: E402
import arvr_prism as AV                                                               # noqa: E402
from .prism_branch import (trace_branching_ff, metrics, build_arcs_ff, nearest_hit_ff,  # noqa: E402
                           SOURCE_PTS, SOURCE_AIM, MAX_DEPTH)

LO = np.concatenate([AV.LO, np.full(8, -2.0)])
HI = np.concatenate([AV.HI, np.full(8, 2.0)])
PARAM_NAMES = list(AV.PRISM_NAMES) + ["a2_A", "a4_A", "a2_B", "a4_B", "a2_T", "a4_T", "a2_D", "a4_D"]


def fold_count(x):
    """Chief-ray (central source) TIR-bounce count -- the structural fold descriptor."""
    arcs, n = build_arcs_ff(np.asarray(x, float))
    sx, sy = SOURCE_PTS[1]
    O = np.array([sx, sy]); D = np.array([math.cos(math.radians(SOURCE_AIM)),
                                          math.sin(math.radians(SOURCE_AIM))])
    inside = False; ntir = 0
    for _ in range(MAX_DEPTH):
        hit = nearest_hit_ff(O, D, arcs)
        if hit is None:
            break
        _t, P, normal, _name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal; ntir += 1
        else:
            cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal; inside = not inside
        D = D / np.linalg.norm(D); O = P
    return ntir


def make_evaluate(min_folds, max_folds, min_throughput=0.0):
    def evaluate(x):
        base = x[:9]
        if not AV.geometry_valid_arvr(AV.prism_dict(base)):
            return None
        try:
            ex = trace_branching_ff(x)
        except Exception:
            return None
        if ex is None:
            return None
        m = metrics(ex)
        if not m["ordering_ok"] or m["fitness"] <= 0 or m["fov"] <= 0:
            return None
        if m["T_min"] < min_throughput:          # quality floor: keep refinable (energy on dominant path)
            return None
        folds = fold_count(x)
        if folds < min_folds or folds > max_folds:
            return None
        xs = [v for r in ex if r["b"] > 0 for v in r["xs"]]
        depth = (max(xs) - min(xs)) if xs else 0.0
        return dict(fitness=float(m["fitness"]), bc=(float(folds), float(m["fov"])),
                    info=dict(folds=int(folds), fov=float(m["fov"]), depth=float(depth),
                              fitness=float(m["fitness"]), T_min=float(m["T_min"]),
                              sigma_mean=float(m["sigma_mean"]),
                              thetas=[f["theta"] for f in m["fields"]],
                              sigmas=[f["sigma"] for f in m["fields"]],
                              aspheric=[float(v) for v in x[9:]]))
    return evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--tilt-range", type=float, default=28.0, help="narrow face tilt to +-R (multi-fold lever)")
    ap.add_argument("--min-folds", type=int, default=2)
    ap.add_argument("--max-folds", type=int, default=6)
    ap.add_argument("--min-throughput", type=float, default=0.0,
                    help="reject designs whose worst-source eyebox throughput is below this (quality floor)")
    ap.add_argument("--fov-bins", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=780.0)
    ap.add_argument("--lock-aspheric", action="store_true")
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_search_prism_multifold"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = LO.copy(), HI.copy()
    lo[6:8] = -args.tilt_range; hi[6:8] = args.tilt_range          # narrow tilt -> zig-zag
    if args.lock_aspheric:
        lo[9:] = 0.0; hi[9:] = 0.0
    BC_BINS = (args.max_folds + 1, args.fov_bins)
    BC_BOUNDS = ((0.0, float(args.max_folds + 1)), (0.0, 40.0))
    evaluate = make_evaluate(args.min_folds, args.max_folds, args.min_throughput)
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
    while n < args.evals and (time.time() - t0) < args.max_seconds:
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
            byfold = {}
            for (i0, _i1), c in archive.cells.items():
                byfold[int(i0)] = max(byfold.get(int(i0), 0.0), c.fitness)
            print(f"[multifold] evals={n:6d} filled={rec['filled']:3d} best={rec['best']:.4f} "
                  f"foldbest={ {k:round(v,3) for k,v in sorted(byfold.items())} } "
                  f"inval={n_invalid} ({time.time()-t0:5.0f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), fold_bin=int(c.bc[0]), fov=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info); cells.append(rec)
    payload = dict(meta=dict(experiment="arvr_prism_MULTIFOLD_foldcount",
                             tilt_range=args.tilt_range, min_folds=args.min_folds,
                             max_folds=args.max_folds, min_throughput=args.min_throughput,
                             evals=int(n), seed=int(args.seed),
                             invalid=int(n_invalid), bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                             bc_axes=["fold_count", "FOV_deg"], param_names=PARAM_NAMES,
                             wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    base = out / "archive_prism_multifold.json"; rep, k = base, 0
    while rep.exists():
        k += 1; rep = base.with_name(base.stem + f"_{k}.json")
    rep.write_text(json.dumps(payload, indent=2))
    foldhist = {}
    for c in cells:
        foldhist[c["folds"]] = foldhist.get(c["folds"], 0) + 1
    print(f"[multifold] wrote {rep} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)  fold histogram={dict(sorted(foldhist.items()))}", flush=True)


if __name__ == "__main__":
    main()
