"""
QD illumination of efficient non-sequential folds, on Fable's typed/ledgered engine.

Genotype: the 9-DOF *tilt-enabled* single element from fold_demo (prismatic tilt is what
lets a whole bundle exceed the critical angle -> real TIR folds). Uncoated: every surface
event is honest Fresnel + TIR (the only loss-free reflection is geometric TIR).

Behaviour axes (DESCRIPTIVE, measured from the engine's TYPED interaction events):
  BC1 = number of TIR bounces on the dominant-energy interaction sequence  (fold depth:
        0 = straight transmission, 2 = double-TIR fold, 4 = quad, ...)
  BC2 = thinness T = X_SENSOR - frontmost optic surface used               (thin = small)

Fitness = FOCUSED THROUGHPUT: fraction of INPUT energy (from the exact energy ledger)
delivered to the sensor within a tight window around the energy centroid. Rewards designs
that route a lot of energy AND concentrate it -- never names folding/TIR/thinness, so any
fold that appears in a folded niche emerged, it was not prescribed.

    python3 ns_probe/run_fold_qd.py --evals 14000 --seed 0 --out ns_probe/results_fold_qd
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
FOCUS_W = 8.0                       # half-width of the focus window (world units)
N_RAYS = 21
BEAM_W = 8.0

BC_BINS = (9, 16)
BC_BOUNDS = ((0.0, 9.0), (10.0, 120.0))   # (TIR-bounce count, thinness T)

LOCK_TILT = False   # ablation: force face tilts to 0 (rotationally-symmetric-ish element)


def evaluate(runner, x):
    p = FD.vec_to_params(x)
    if not FD.geometry_valid(p):
        return None
    try:
        res, _ = runner.simulate(FD.scene(p, n_rays=N_RAYS, beam_w=BEAM_W, record=True))
    except Exception:
        return None
    if res.get("error"):
        return None
    led = res.get("energyLedger") or {}
    emitted = float(led.get("emitted", 0.0))
    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    # Count each sensor-arriving ray's energy ONCE: keep only records with a valid
    # first-crossing signature (signature_of returns None for re-crossings), so focused
    # throughput cannot exceed the input.
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, face_of) is not None]
    if emitted <= 0 or not recs:
        return None

    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    if not by_sig:
        return None
    dom_sig = max(by_sig, key=by_sig.get)
    dom_energy = by_sig[dom_sig]
    tir_count = sum(1 for (t, _f) in dom_sig if t == "tir")

    ys = np.array([r["y"] for r in recs], dtype=float)
    bs = np.array([r["b"] for r in recs], dtype=float)
    centroid = float(np.average(ys, weights=bs))
    window_energy = float(bs[np.abs(ys - centroid) <= FOCUS_W].sum())
    fitness = window_energy / emitted
    spot = float(np.sqrt(np.average((ys - centroid) ** 2, weights=bs)))
    efficiency = float(led.get("delivered", bs.sum())) / emitted

    optic_xs = [v["x"] for r in recs for v in r["seq"] if 0.5 < v["x"] < X_SENSOR - 0.5]
    T = X_SENSOR - (min(optic_xs) if optic_xs else 0.0)

    return dict(
        fitness=float(fitness),
        bc=(float(tir_count), float(T)),
        info=dict(efficiency=efficiency, spot=spot, fitness=float(fitness),
                  tir_count=int(tir_count), thinness_T=float(T),
                  dom_energy_share=float(dom_energy / emitted),
                  dom_signature=format_signature(dom_sig), centroid=centroid,
                  n_forward=len(recs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_fold_qd"))
    ap.add_argument("--lock-tilt", action="store_true",
                    help="ablation: force face tilts to 0 (symmetric element)")
    args = ap.parse_args()

    global LOCK_TILT
    LOCK_TILT = args.lock_tilt
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
        n = 0
        n_invalid = 0
        history = []
        while n < args.evals:
            for ei in range(args.emitters):
                if n >= args.evals:
                    break
                em = emitters[ei]
                X = em.ask()
                keys = np.empty(len(X))
                added = False
                for k, xx in enumerate(X):
                    if LOCK_TILT:
                        xx = xx.copy(); xx[6] = 0.0; xx[7] = 0.0   # zero tilt_a, tilt_b
                    ev = evaluate(runner, xx)
                    n += 1
                    if ev is None:
                        keys[k] = -1e9
                        n_invalid += 1
                        continue
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
                print(f"[fold-qd] evals={n:6d} filled={rec['filled']:3d} "
                      f"cov={rec['coverage']*100:5.1f}% best_focthru={rec['best']:.3f} "
                      f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), tir_count=float(c.bc[0]), T=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info)
        cells.append(rec)
    payload = dict(
        meta=dict(experiment="fold_qd_tilt_uncoated", evals=int(n), seed=int(args.seed),
                  invalid=int(n_invalid), bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                  bc_axes=["dominant_TIR_bounce_count", "thinness_T (thin=small)"],
                  fitness="focused_throughput (window energy / emitted)", focus_w=FOCUS_W,
                  param_names=FD.PARAM_NAMES, wall_seconds=float(time.time() - t0)),
        history=history, cells=cells)
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    print(f"[fold-qd] wrote {out/'archive.json'} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
