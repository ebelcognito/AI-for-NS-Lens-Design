"""
Descriptor experiment: same fold-QD setup, SWAP ONLY the diversity axis (BC1).

This is the controlled test of "can a better descriptor surface more good non-sequential
designs?" Everything that defines QUALITY and the SEARCH is held identical to run_fold_qd.py:
  - genotype: the 9-DOF tilt element (fold_demo)
  - fitness:  FOCUSED THROUGHPUT (window energy / emitted)  -- unchanged, fold-blind
  - engine:   uncoated honest Fresnel + TIR
  - solver/budget: CMA-ME, same emitters/popsize/sigma0/evals/seed
Only BC1 (the first behaviour axis) changes; BC2 stays = thinness T so the comparison
isolates the effect of the diversity axis. Every candidate descriptor is ALSO measured and
stored in `info` for every elite, so compare_descriptors.py can score all runs uniformly.

BC1 axes (`--bc`):
  baseline    # of TIR bounces on dominant-energy path        (the current axis; integers 0..9)
  nsfrac      fraction of forward sensor energy that took >=1 TIR  (CONTINUOUS non-sequentialness)
  tilt        prismatic tilt magnitude sqrt(tilt_a^2+tilt_b^2)     (the causal symmetry-break knob)
  critmargin  min incidence margin past critical on the fold path  (how deep into the TIR regime)
  locus       fraction of TIR events on the FLAT EDGES (T/D)       (edge-TIR light-guiding vs curved-face fold)
  robust      mean focused-throughput retained under small geometry perturbation (tolerance)

    python3 ns_probe/run_fold_qd_descriptors.py --bc nsfrac --evals 14000 --seed 0 \
        --out ns_probe/results_desc_nsfrac
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
N_RAYS = 21
BEAM_W = 8.0

# BC2 (thinness) is shared across all variants so the only change vs baseline is BC1.
THIN_BINS, THIN_BOUNDS = 16, (10.0, 120.0)
CRIT_NOTIR = -5.0   # sentinel margin for designs with no TIR (lands in bottom critmargin bin)

# BC1 axis registry: name -> (key into the measured descriptor dict, n_bins, (lo, hi), label)
BC1 = {
    "baseline":   ("tir_count",  9, (0.0, 9.0),   "dominant_TIR_bounce_count"),
    "nsfrac":     ("ns_frac",   12, (0.0, 1.0),   "TIR_energy_fraction (continuous NS-ness)"),
    "tilt":       ("tilt_mag",  12, (0.0, 36.0),  "prismatic_tilt_magnitude_deg"),
    "critmargin": ("crit_margin", 12, (-6.0, 42.0), "min_incidence_margin_past_critical_deg"),
    "locus":      ("fold_locus", 11, (0.0, 1.0),  "edge_TIR_fraction (locus: 0=curved-face, 1=edge-guide)"),
    "robust":     ("robustness", 12, (0.0, 1.0),  "tolerance_retained_focused_throughput"),
}

# perturbation for the robustness descriptor: manufacturing-like jitter on the physical
# surface DOFs (cx, thick, fsag, bsag, tilt_a, tilt_b); NOT the beam (ray_y, ang) or index n.
PERT_IDX = np.array([2, 3, 4, 5, 6, 7])
PERT_K = 3
LOCK_TILT = False


def incidence_deg(seq, k):
    """Angle of incidence (from normal) at vertex k, inferred from the ray's turn."""
    a = np.array([seq[k]["x"] - seq[k - 1]["x"], seq[k]["y"] - seq[k - 1]["y"]])
    b = np.array([seq[k + 1]["x"] - seq[k]["x"], seq[k + 1]["y"] - seq[k]["y"]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return 90.0 - math.degrees(math.acos(c)) / 2.0


def _focthru(runner, x):
    """Just the focused-throughput fitness (used by the robustness descriptor)."""
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
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, face_of) is not None]
    if emitted <= 0 or not recs:
        return None
    ys = np.array([r["y"] for r in recs], dtype=float)
    bs = np.array([r["b"] for r in recs], dtype=float)
    centroid = float(np.average(ys, weights=bs))
    return float(bs[np.abs(ys - centroid) <= FOCUS_W].sum()) / emitted


def evaluate(runner, x, *, with_robust=False, rng=None):
    """Identical fitness + thinness to run_fold_qd.py, plus ALL candidate descriptors."""
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

    # --- candidate descriptors (all measured for every design) ---
    # (1) continuous NS-ness: forward sensor energy that underwent >=1 TIR / all forward energy
    fwd_total = float(bs.sum())
    tir_energy = 0.0
    for r in recs:
        sig_r = signature_of(r, face_of)
        if sig_r and any(t == "tir" for (t, _f) in sig_r):
            tir_energy += r["b"]
    ns_frac = tir_energy / fwd_total if fwd_total > 0 else 0.0

    # (2) prismatic tilt magnitude (genotype-level, causal symmetry break)
    tilt_mag = float(math.hypot(p["tilt_a"], p["tilt_b"]))

    # (3) fold locus: fraction of the dominant path's TIR events on the FLAT EDGES (T/D)
    edge_tir = sum(1 for (t, f) in dom_sig if t == "tir" and f in ("T", "D"))
    fold_locus = (edge_tir / tir_count) if tir_count > 0 else 0.0

    # (4) min incidence margin past critical on the dominant ray's TIR events
    theta_c = math.degrees(math.asin(min(1.0 / p["n"], 1.0)))
    dom_ray = max(recs, key=lambda r: r["b"])
    seq = dom_ray["seq"]
    margins = []
    for k in range(1, len(seq) - 1):
        if seq[k].get("t") == "tir":
            ti = incidence_deg(seq, k)
            if ti is not None:
                margins.append(ti - theta_c)
    crit_margin = float(min(margins)) if margins else CRIT_NOTIR

    # (5) tolerance robustness (optional; 3 perturbed re-evals)
    robustness = 0.0
    if with_robust:
        sig_p = 0.02 * (FD.HI - FD.LO)
        keep = []
        for _ in range(PERT_K):
            xp = np.array(x, float).copy()
            xp[PERT_IDX] += rng.normal(0.0, 1.0, size=len(PERT_IDX)) * sig_p[PERT_IDX]
            xp = np.clip(xp, FD.LO, FD.HI)
            f = _focthru(runner, xp)
            keep.append(0.0 if f is None else f)
        # retained fraction of base fitness, clipped to [0,1]
        robustness = float(np.clip(np.mean(keep) / max(fitness, 1e-3), 0.0, 1.0))

    desc = dict(tir_count=float(tir_count), ns_frac=float(ns_frac), tilt_mag=tilt_mag,
                crit_margin=crit_margin, fold_locus=float(fold_locus),
                robustness=float(robustness), thinness_T=float(T))

    info = dict(efficiency=efficiency, spot=spot, fitness=float(fitness),
                tir_count=int(tir_count), thinness_T=float(T),
                ns_frac=float(ns_frac), tilt_mag=tilt_mag, crit_margin=crit_margin,
                fold_locus=float(fold_locus), robustness=float(robustness),
                dom_energy_share=float(dom_energy / emitted),
                dom_signature=format_signature(dom_sig), centroid=centroid,
                n_forward=len(recs))
    return dict(fitness=float(fitness), desc=desc, info=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", choices=list(BC1), default="baseline",
                    help="which descriptor defines BC1 (the diversity axis)")
    ap.add_argument("--bc2", choices=list(BC1) + ["thinness"], default="thinness",
                    help="second axis; 'thinness' (default) reproduces the original; or a fold "
                         "descriptor, e.g. --bc baseline --bc2 locus for a depth x structure grid")
    ap.add_argument("--evals", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lock-tilt", action="store_true")
    args = ap.parse_args()

    global LOCK_TILT
    LOCK_TILT = args.lock_tilt
    bc_key, bc_bins, bc_bounds, bc_label = BC1[args.bc]
    if args.bc2 == "thinness":
        bc2_key, bc2_bins, bc2_bounds, bc2_label = "thinness_T", THIN_BINS, THIN_BOUNDS, "thinness_T (thin=small)"
    else:
        bc2_key, bc2_bins, bc2_bounds, bc2_label = BC1[args.bc2]
    with_robust = (args.bc == "robust" or args.bc2 == "robust")
    default_name = f"results_desc_{args.bc}" + ("" if args.bc2 == "thinness" else f"_x_{args.bc2}")
    out = Path(args.out or (REPO / "ns_probe" / default_name))
    out.mkdir(parents=True, exist_ok=True)

    BC_BINS = (bc_bins, bc2_bins)
    BC_BOUNDS = (bc_bounds, bc2_bounds)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    pert_rng = np.random.default_rng(args.seed + 9999)
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
                        xx = xx.copy(); xx[6] = 0.0; xx[7] = 0.0
                    ev = evaluate(runner, xx, with_robust=with_robust, rng=pert_rng)
                    n += 1
                    if ev is None:
                        keys[k] = -1e9
                        n_invalid += 1
                        continue
                    bc = (float(ev["desc"][bc_key]), float(ev["desc"][bc2_key]))
                    st, old = archive.add(xx, ev["fitness"], bc, info=ev["info"])
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
                print(f"[desc:{args.bc}] evals={n:6d} filled={rec['filled']:3d} "
                      f"cov={rec['coverage']*100:5.1f}% best_focthru={rec['best']:.3f} "
                      f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), bc1=float(c.bc[0]), T=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info)
        cells.append(rec)
    payload = dict(
        meta=dict(experiment=f"fold_qd_desc_{args.bc}_x_{args.bc2}", bc1_axis=args.bc,
                  bc2_axis=args.bc2, bc1_label=bc_label,
                  evals=int(n), seed=int(args.seed), invalid=int(n_invalid),
                  bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                  bc_axes=[bc_label, bc2_label],
                  fitness="focused_throughput (window energy / emitted)", focus_w=FOCUS_W,
                  param_names=FD.PARAM_NAMES, wall_seconds=float(time.time() - t0)),
        history=history, cells=cells)
    (out / "archive.json").write_text(json.dumps(payload, indent=2))
    print(f"[desc:{args.bc}] wrote {out/'archive.json'} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
