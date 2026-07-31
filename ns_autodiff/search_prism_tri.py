"""QD SEARCH — 3-EDGE triangular AR/VR prism, archived by (FOLD COUNT x FOV) for diversity.

Two changes from search_prism_freeform.py, both to fix the "everything looks the same" problem:
  * geometry = a 3-edge TRIANGLE (build_triangle) instead of the 4-face block -- drops the
    redundant extreme-curvature 4th face;
  * the archive's first axis is the NUMBER OF TIR FOLDS (chief-ray bounce count), so 1-fold,
    2-fold, 3-fold... relays are forced into SEPARATE niches -- structural diversity, not just
    performance variations of one shape. Second axis stays FOV.

Faces are circular arcs (freeform DOF didn't help the search and just enlarge the space). Fitness
is the same 3-field collimation metric on the validated branching tracer.

  PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
  $PY -m ns_autodiff.search_prism_tri --smoke
  $PY -m ns_autodiff.search_prism_tri --evals 20000 --seed 0
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
from .prism_branch import (build_triangle, geometry_valid_tri, trace_faces, metrics,  # noqa: E402
                           chief_ray_folds, SOURCE_PTS, PUPIL_X)

#               cx     w   apexx  apexy  s_top s_left s_right  n
LO = np.array([40.0, 10.0, 20.0, -50.0, -12.0, -12.0, -12.0, 1.50])
HI = np.array([70.0, 45.0, 110.0, 20.0,  12.0,  12.0,  12.0, 2.00])
PARAM_NAMES = ["cx", "w", "apex_x", "apex_y", "s_top", "s_left", "s_right", "n"]
BC_BINS = (9, 20)
BC_BOUNDS = ((0.0, 9.0), (0.0, 40.0))      # (fold count, FOV deg)


def evaluate(x):
    if not geometry_valid_tri(x):
        return None
    try:
        faces, n = build_triangle(x)
        ex = trace_faces(faces, n)
    except Exception:
        return None
    if ex is None:                                       # trapping cavity hit the segment cap
        return None
    m = metrics(ex)
    if not m["ordering_ok"] or m["fitness"] <= 0 or m["fov"] <= 0:
        return None
    folds = chief_ray_folds(faces, n)
    xs = [v for r in ex if r["b"] > 0 for v in r["xs"]]
    depth = (max(xs) - min(xs)) if xs else 0.0
    return dict(fitness=float(m["fitness"]), bc=(float(folds), float(m["fov"])),
                info=dict(folds=int(folds), fov=float(m["fov"]), depth=float(depth),
                          fitness=float(m["fitness"]), T_min=float(m["T_min"]),
                          sigma_mean=float(m["sigma_mean"]),
                          thetas=[f["theta"] for f in m["fields"]]))


def smoke():
    """A few hand triangles: confirm rays relay to the eyebox and fold counts vary."""
    import numpy as np
    tries = [
        [55, 24, 95, -30, 0, 0, 0, 1.8],     # apex low-right
        [55, 24, 30, -35, 0, -6, 6, 1.7],
        [52, 30, 100, -10, -4, 4, -8, 1.9],
        [58, 18, 80, -45, 6, 0, -4, 1.6],
    ]
    print("smoke: hand triangles ->")
    for t in tries:
        x = np.array(t, float)
        ev = evaluate(x)
        if ev is None:
            print(f"  {t}: invalid"); continue
        i = ev["info"]
        print(f"  folds={i['folds']} FOV={i['fov']:.1f} fit={ev['fitness']:.3f} "
              f"sig={i['sigma_mean']:.2f} depth={i['depth']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_search_prism_tri"))
    args = ap.parse_args()
    if args.smoke:
        smoke(); return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed); scale = HI - LO; t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(LO, HI)
        return CMAES(np.asarray(x0, float), args.sigma0 * float(np.mean(scale)),
                     popsize=args.popsize, seed=s, bounds=(LO, HI))

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
            folds_filled = sorted(set(int(c.bc[0]) for c in archive.cells.values()))
            rec = dict(evals=n, filled=len(archive.cells), coverage=archive.coverage,
                       best=archive.best_fitness, invalid=n_invalid, fold_axes=folds_filled)
            history.append(rec)
            print(f"[tri] evals={n:6d} filled={rec['filled']:3d} cov={rec['coverage']*100:5.1f}% "
                  f"best={rec['best']:.4f} folds_seen={folds_filled} inv={n_invalid} "
                  f"({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), folds=float(c.bc[0]), fov=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info); cells.append(rec)
    payload = dict(meta=dict(experiment="arvr_prism_TRIANGLE_foldcount", evals=int(n),
                             seed=int(args.seed), invalid=int(n_invalid), bc_bins=BC_BINS,
                             bc_bounds=BC_BOUNDS, bc_axes=["fold_count", "FOV_deg"],
                             param_names=PARAM_NAMES, wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    rep = out / "archive_tri.json"; k = 0
    while rep.exists():
        k += 1; rep = out / f"archive_tri_{k}.json"
    rep.write_text(json.dumps(payload, indent=2))
    print(f"[tri] wrote {rep} ({len(cells)} cells, {n} evals, {payload['meta']['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
