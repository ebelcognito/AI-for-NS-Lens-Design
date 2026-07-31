"""
Multi-field fold QD: the same CMA-ME illumination, but each design is scored on its ability
to FOLD and FOCUS a *range* of input field angles (not just one). Fitness = mean focused
throughput over fields {-3, 0, +3} degrees. This addresses the biggest caveat of the
single-field runs: are the folds real imaging elements, or single-field flukes?

    python3 ns_probe/run_multifield_qd.py --evals 12000 --out ns_probe/results_multifield_qd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
from cma_me import CMAES, GridArchive
import fold_demo as FD
from sequence_tools import (aggregate_by_signature, format_signature,
                            make_single_element_face_classifier, signature_of)

X_SENSOR = FD.X_SENSOR
ELEM_H = FD.ELEM_H
FOCUS_W = 8.0
N_RAYS = 17
BEAM_W = 8.0
FIELDS = [-3.0, 0.0, 3.0]            # input field angles (deg)
BC_BINS = (9, 16)
BC_BOUNDS = ((0.0, 9.0), (10.0, 120.0))


def field_metrics(runner, p, field_deg):
    """Focused throughput + spot + (central) dominant sequence for one input field."""
    pf = dict(p); pf["ang"] = p["ang"] + field_deg
    try:
        res, _ = runner.simulate(FD.scene(pf, n_rays=N_RAYS, beam_w=BEAM_W, record=True))
    except Exception:
        return None
    if res.get("error"):
        return None
    led = res.get("energyLedger") or {}
    emitted = float(led.get("emitted", 0.0))
    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0 and signature_of(r, face_of) is not None]
    if emitted <= 0 or not recs:
        return dict(focthru=0.0, spot=99.0, dom_sig=None, tir=0)
    ys = np.array([r["y"] for r in recs]); bs = np.array([r["b"] for r in recs])
    cen = float(np.average(ys, weights=bs))
    focthru = float(bs[np.abs(ys - cen) <= FOCUS_W].sum()) / emitted
    spot = float(np.sqrt(np.average((ys - cen) ** 2, weights=bs)))
    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    dom_sig = max(by_sig, key=by_sig.get) if by_sig else None
    tir = sum(1 for (t, _f) in (dom_sig or ()) if t == "tir")
    optic_xs = [v["x"] for r in recs for v in r["seq"] if 0.5 < v["x"] < X_SENSOR - 0.5]
    T = X_SENSOR - (min(optic_xs) if optic_xs else 0.0)
    return dict(focthru=focthru, spot=spot, dom_sig=dom_sig, tir=tir, T=T)


def evaluate(runner, x):
    p = FD.vec_to_params(x)
    if not FD.geometry_valid(p):
        return None
    ms = [field_metrics(runner, p, f) for f in FIELDS]
    if any(m is None for m in ms):
        return None
    central = ms[FIELDS.index(0.0)]
    if central["dom_sig"] is None:
        return None
    foc = [m["focthru"] for m in ms]
    fitness = float(np.mean(foc))
    return dict(fitness=fitness, bc=(float(central["tir"]), float(central["T"])),
                info=dict(mean_focthru=fitness, worst_focthru=float(min(foc)),
                          per_field=[round(f, 4) for f in foc],
                          central_spot=central["spot"], tir_count=int(central["tir"]),
                          thinness_T=float(central["T"]),
                          dom_signature=format_signature(central["dom_sig"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_multifield_qd"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    scale = FD.HI - FD.LO
    t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(FD.LO, FD.HI)
        return CMAES(np.asarray(x0, float), args.sigma0 * float(np.mean(scale)),
                     popsize=args.popsize, seed=s, bounds=(FD.LO, FD.HI))

    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        emitters = [new_emitter(args.seed + i) for i in range(args.emitters)]
        n = 0; n_invalid = 0; history = []
        while n < args.evals:
            for ei in range(args.emitters):
                if n >= args.evals:
                    break
                em = emitters[ei]; X = em.ask(); keys = np.empty(len(X)); added = False
                for k, xx in enumerate(X):
                    ev = evaluate(runner, xx); n += 1
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
                print(f"[mf-qd] evals={n:6d} filled={rec['filled']:3d} cov={rec['coverage']*100:5.1f}% "
                      f"best_mean_focthru={rec['best']:.3f} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        r = dict(i0=int(i0), i1=int(i1), tir_count=float(c.bc[0]), T=float(c.bc[1]),
                 fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        r.update(c.info); cells.append(r)
    payload = dict(meta=dict(experiment="multifield_qd", fields=FIELDS, evals=int(n),
                             seed=int(args.seed), invalid=int(n_invalid), bc_bins=BC_BINS,
                             bc_bounds=BC_BOUNDS, fitness="mean focused throughput over fields",
                             param_names=FD.PARAM_NAMES, wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    print(f"[mf-qd] wrote {out/'archive.json'} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
