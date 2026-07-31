"""
AR/VR PRISM QD — Phase B: illuminate a (FOV x form-factor) archive of near-eye TIR
display relays.

Genotype: the 7-DOF prism (cx, thick, fsag, bsag, tilt_a, tilt_b, n). Uncoated -> TIR-only.
Sources/eyebox fixed (see arvr_prism.py). Imaging = COLLIMATED (virtual image at infinity).

Fitness (quality) = well-collimated throughput over the 3 display fields
    mean_s [ T_s * exp(-sigma_s/SIGMA0) ],  valid only if all 3 fields reach the eyebox
    with a strictly monotonic field map (a real, non-scrambled image). Never names folding.

Behaviour axes (DESCRIPTIVE):
    BC1 = FOV (deg)        = angular span of the two edge fields at the pupil
    BC2 = form factor      = prism DEPTH = x-extent of glass the light actually uses
                             (thin/compact = small). This is the axis that makes folds
                             worth discovering: to stay compact AND collimate a wide field,
                             the relay must fold (TIR) rather than be a long fat collimator.

    python3 ns_probe/run_arvr_qd.py --evals 9000 --seed 0
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
import arvr_prism as AV
import fold_demo as FD

PUPIL_X = AV.PUPIL_X
BC_BINS = (20, 28)                          # high-res archive (coverage is NOT the goal;
BC_BOUNDS = ((0.0, 40.0), (4.0, 88.0))      # fine FOV/form-factor resolution is) (FOV deg, depth)
MIN_TIR = 0                                 # if >0, only designs with >= MIN_TIR TIR bounces
                                            # on the dominant eyebox path enter the archive
                                            # (use 2 to force MULTI-fold relays)


def prism_depth(res):
    xs = [v["x"] for r in res.get("raySequences", []) if r.get("dirx", 0) > 0
          for v in r["seq"] if 1.0 < v["x"] < PUPIL_X - 1.0]
    return (max(xs) - min(xs)) if xs else 0.0


def tir_bounces(res):
    """TIR events on the densest eyebox-reaching path (for analysis, not a descriptor)."""
    best, bestb = 0, -1.0
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0 or abs(float(r["y"]) - AV.EYEBOX_Y) > AV.EYEBOX_HALF:
            continue
        if r["b"] > bestb:
            bestb = r["b"]
            best = sum(1 for v in r["seq"] if v.get("t") == "tir")
    return best


def evaluate(runner, x):
    p = AV.prism_dict(x)
    if not AV.geometry_valid_arvr(p):
        return None
    try:
        res, _ = runner.simulate(AV.scene_arvr(x))
    except Exception:
        return None
    if res.get("error"):
        return None
    m = AV.arvr_metrics(res)
    if not m["ordering_ok"] or m["fitness"] <= 0:      # only real 3-field relays in archive
        return None
    depth = prism_depth(res)
    fov = m["fov"]
    if fov <= 0:
        return None
    tc = tir_bounces(res)
    if tc < MIN_TIR:                        # force multi-fold when MIN_TIR>0
        return None
    return dict(
        fitness=float(m["fitness"]), bc=(float(fov), float(depth)),
        info=dict(fov=float(fov), depth=float(depth), fitness=float(m["fitness"]),
                  T_min=float(m["T_min"]), sigma_mean=float(m["sigma_mean"]),
                  tir=int(tc),
                  thetas=[f["theta"] for f in m["fields"]],
                  sigmas=[f["sigma"] for f in m["fields"]],
                  Ts=[f["T"] for f in m["fields"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_arvr_qd"))
    ap.add_argument("--tilt-range", type=float, default=None,
                    help="override face tilt bound to +-this (deg). Narrow (~28) so one "
                         "surface can't do the 90deg turn -> forces multi-bounce folds.")
    ap.add_argument("--source-tilt", type=float, default=0.0,
                    help="tilt the display emission this many deg LEFT of straight-down.")
    ap.add_argument("--min-tir", type=int, default=0,
                    help="only archive designs with >= this many TIR bounces (2 = multi-fold).")
    args = ap.parse_args()

    global MIN_TIR
    MIN_TIR = args.min_tir
    AV.SOURCE_AIM = -90.0 - args.source_tilt
    if args.tilt_range is not None:
        for nm in ("tilt_a", "tilt_b"):
            j = AV.PRISM_NAMES.index(nm)
            AV.LO[j], AV.HI[j] = -args.tilt_range, args.tilt_range
    print(f"[config] source_aim={AV.SOURCE_AIM:.0f}deg  tilt_range=+-{args.tilt_range}  "
          f"min_tir={MIN_TIR}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    scale = AV.HI - AV.LO
    t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(AV.LO, AV.HI)
        return CMAES(np.asarray(x0, float), args.sigma0 * float(np.mean(scale)),
                     popsize=args.popsize, seed=s, bounds=(AV.LO, AV.HI))

    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        emitters = [new_emitter(args.seed + i) for i in range(args.emitters)]
        n, n_invalid, history = 0, 0, []
        while n < args.evals:
            for ei in range(args.emitters):
                if n >= args.evals:
                    break
                em = emitters[ei]
                X = em.ask()
                keys = np.empty(len(X)); added = False
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
                print(f"[arvr-qd] evals={n:6d} filled={rec['filled']:3d} "
                      f"cov={rec['coverage']*100:5.1f}% best={rec['best']:.4f} "
                      f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), fov=float(c.bc[0]), depth=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info)
        cells.append(rec)
    payload = dict(
        meta=dict(experiment="arvr_prism_qd_tir", evals=int(n), seed=int(args.seed),
                  invalid=int(n_invalid), bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                  bc_axes=["FOV_deg", "prism_depth (compact=small)"],
                  fitness="well_collimated_throughput_3field", sigma0=AV.SIGMA0,
                  geometry=dict(display_y=AV.DISPLAY_Y, display_cx=AV.DISPLAY_CX,
                                display_half_w=AV.DISPLAY_HALF_W, source_aim=AV.SOURCE_AIM,
                                n_fan=AV.N_FAN, fan_halfangle=AV.FAN_HALFANGLE,
                                pupil_x=AV.PUPIL_X, eyebox_half=AV.EYEBOX_HALF),
                  param_names=AV.PRISM_NAMES, wall_seconds=float(time.time() - t0)),
        history=history, cells=cells)
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    print(f"[arvr-qd] wrote {out/'archive.json'} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
