"""
THINNING CONTINUATION: take a discovered focusing fold and make the element THINNER
without giving up the focus.

WHY A CONTINUATION AND NOT A RE-SEARCH. `thick` is not an independent knob: shortening
the element shortens every internal leg of the fold, so the same tilts/curvatures no
longer land the bundle on the same faces at the same incidence, and the focus (and often
the whole TIR route) collapses. So we march thickness DOWN in small steps and, at every
step, re-optimize the remaining ELEMENT-SHAPE DOF (cx, fsag, bsag, tilt_a, tilt_b, n)
from the previous step's solution -- a continuation / homotopy in thickness. Each step is
a bounded CMA-ES polish on the honest Fresnel+TIR engine; nothing is linearized.

WHAT IS HELD FIXED. The input beam (ray_y, ang) is the FIELD the design was found for and
is FROZEN, exactly as in refine_lm.py: "thinner and still focusing" must mean a better
element, not a re-aimed source. Thickness itself is frozen at each step's target value.

OBJECTIVE (same metrics the QD archive records, so numbers are comparable):
    maximize  focused_throughput  -  spot_penalty * max(0, spot - spot_tol)
    subject to (hard reject) dominant path keeps >= --min-tir TIR bounces
`focused_throughput` = energy landing within +-8 of the sensor centroid / emitted;
`spot` = energy-weighted RMS sensor spread. Both at the archive's 21-ray density; the
picks are then RE-VALIDATED at a much denser bundle so a tight 21-ray spot cannot be an
aliasing artifact.

OUTPUT: a never-overwritten sweep JSON (full thickness/spot/throughput frontier + every
step's genotype), a trade-off figure, and high-res renders of the baseline and of each
recommended thinner design.

    python3 ns_probe/thin_design.py \
        --design ns_probe/results_gallery_focus2/focus_01_spot0p08_tir4.json
    python3 ns_probe/thin_design.py --design <json> --min-thick 8 --step 2.0 --gens 40
"""
from __future__ import annotations

import argparse
import base64
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
from cma_me import CMAES
import fold_demo as FD
from run_fold_qd_descriptors import incidence_deg, FOCUS_W, N_RAYS, BEAM_W
from sequence_tools import (aggregate_by_signature, format_signature,
                            make_single_element_face_classifier, signature_of)

PARAM_NAMES = FD.PARAM_NAMES          # [ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n]
LO, HI = FD.LO.copy(), FD.HI.copy()
I_THICK = PARAM_NAMES.index("thick")

# Element shape is free to compensate; the field (ray_y, ang) and thickness are frozen.
FREE_DOF = ["cx", "fsag", "bsag", "tilt_a", "tilt_b", "n"]

OUTDIR = REPO / "ns_probe" / "results_thin"


def uniq(path: Path) -> Path:
    k = 0
    out = path
    while out.exists():
        k += 1
        out = path.with_name(f"{path.stem}_{k}{path.suffix}")
    return out


# --------------------------------------------------------------------------- #
#  measurement (archive-identical metrics, arbitrary bundle density)
# --------------------------------------------------------------------------- #
def measure(runner, x, *, n_rays=N_RAYS, beam_w=BEAM_W):
    """The archive's own metrics (focused throughput / spot / efficiency / dominant
    signature / min TIR margin) for genotype x. None if the geometry is invalid."""
    p = FD.vec_to_params(x)
    if not FD.geometry_valid(p):
        return None
    try:
        res, _ = runner.simulate(FD.scene(p, n_rays=n_rays, beam_w=beam_w, record=True))
    except Exception:
        return None
    if res.get("error"):
        return None
    led = res.get("energyLedger") or {}
    emitted = float(led.get("emitted", 0.0))
    face_of = make_single_element_face_classifier(p["cx"], FD.ELEM_H)
    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, face_of) is not None]
    if emitted <= 0 or not recs:
        return None
    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    if not by_sig:
        return None
    dom_sig = max(by_sig, key=by_sig.get)
    tir_count = sum(1 for (t, _f) in dom_sig if t == "tir")

    ys = np.array([r["y"] for r in recs], dtype=float)
    bs = np.array([r["b"] for r in recs], dtype=float)
    centroid = float(np.average(ys, weights=bs))
    focthru = float(bs[np.abs(ys - centroid) <= FOCUS_W].sum()) / emitted
    spot = float(np.sqrt(np.average((ys - centroid) ** 2, weights=bs)))
    efficiency = float(led.get("delivered", bs.sum())) / emitted

    # min incidence margin past critical on the dominant (highest-energy) ray's TIRs
    theta_c = math.degrees(math.asin(min(1.0 / p["n"], 1.0)))
    seq = max(recs, key=lambda r: r["b"])["seq"]
    margins = [incidence_deg(seq, k) - theta_c for k in range(1, len(seq) - 1)
               if seq[k].get("t") == "tir" and incidence_deg(seq, k) is not None]
    crit_margin = float(min(margins)) if margins else -5.0

    return dict(focthru=focthru, spot=spot, efficiency=efficiency,
                dom_share=float(by_sig[dom_sig] / emitted), tir_count=int(tir_count),
                dom_signature=format_signature(dom_sig), crit_margin=crit_margin,
                centroid=centroid, n_rays=int(n_rays))


def score_of(m, *, spot_tol, spot_penalty, min_tir, min_margin=0.0):
    """Scalar the per-thickness polish maximizes. Focus is a soft constraint (penalty
    above spot_tol); keeping a real fold -- enough TIR bounces, and enough incidence
    margin past critical that the fold is not a knife-edge -- is a hard one."""
    if m is None or m["tir_count"] < min_tir or m["crit_margin"] < min_margin:
        return -1e9
    return m["focthru"] - spot_penalty * max(0.0, m["spot"] - spot_tol)


# --------------------------------------------------------------------------- #
#  manufacturing tolerance (same jitter model as the QD 'robust' descriptor)
# --------------------------------------------------------------------------- #
PERT_IDX = np.array([PARAM_NAMES.index(n) for n in
                     ("cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b")])


def tolerance(runner, x, *, k=24, sigma_frac=0.02, seed=1234, min_tir=2):
    """Perturb the PHYSICAL surface DOF by sigma_frac of their search range (the QD
    'robustness' descriptor's jitter; beam and index untouched) and report what survives.
    A thin design that only works at knife-edge incidence shows up here as a collapse."""
    rng = np.random.default_rng(seed)
    sig = sigma_frac * (HI - LO)
    thru, spots, kept = [], [], 0
    for _ in range(k):
        xp = np.array(x, float).copy()
        xp[PERT_IDX] += rng.normal(0.0, 1.0, size=len(PERT_IDX)) * sig[PERT_IDX]
        m = measure(runner, np.clip(xp, LO, HI))
        if m is None:
            thru.append(0.0)
            continue
        thru.append(m["focthru"])
        spots.append(m["spot"])
        kept += int(m["tir_count"] >= min_tir)
    base = measure(runner, x)
    return dict(k=int(k), sigma_frac=float(sigma_frac),
                focthru_mean=float(np.mean(thru)),
                focthru_retained=float(np.mean(thru) / max(base["focthru"], 1e-9)) if base else None,
                spot_median=float(np.median(spots)) if spots else float("nan"),
                spot_p90=float(np.percentile(spots, 90)) if spots else float("nan"),
                fold_kept_frac=float(kept / k))


# --------------------------------------------------------------------------- #
#  one thickness: bounded CMA-ES polish of the free shape DOF
# --------------------------------------------------------------------------- #
def polish(runner, x_start, thick, free_idx, *, gens, popsize, sigma0, seed,
           spot_tol, spot_penalty, min_tir, min_margin):
    lo, hi = LO[free_idx], HI[free_idx]
    span = hi - lo

    def full(xn):
        x = np.array(x_start, float).copy()
        x[free_idx] = lo + np.clip(xn, 0, 1) * span
        x[I_THICK] = thick
        return x

    def sc(xn):
        m = measure(runner, full(xn))
        return score_of(m, spot_tol=spot_tol, spot_penalty=spot_penalty,
                        min_tir=min_tir, min_margin=min_margin), m

    xn0 = (np.asarray(x_start, float)[free_idx] - lo) / span
    best_s, best_m = sc(xn0)
    best_x = full(xn0)
    n_ev = 1

    es = CMAES(np.clip(xn0, 0, 1), sigma0, popsize=popsize, seed=seed,
               bounds=(np.zeros(len(free_idx)), np.ones(len(free_idx))))
    for _g in range(gens):
        X = es.ask()
        S = np.empty(len(X))
        for k, xn in enumerate(X):
            S[k], m = sc(xn)
            n_ev += 1
            if S[k] > best_s:
                best_s, best_m, best_x = S[k], m, full(xn)
        es.tell(np.argsort(-S))
        if es.restart_needed:                      # re-seed from the best so far
            es = CMAES((best_x[free_idx] - lo) / span, sigma0, popsize=popsize,
                       seed=seed + 7919, bounds=(np.zeros(len(free_idx)),
                                                 np.ones(len(free_idx))))
    return best_x, best_s, best_m, n_ev


# --------------------------------------------------------------------------- #
#  rendering (same high-res look as fold_gallery)
# --------------------------------------------------------------------------- #
HQ_W, HQ_RAYS, HQ_DEPTH = 1900, 15, 11
CROP_X, CROP_Y = (-4.0, 145.0), (-40.0, 40.0)


def render(runner, p, path, meta):
    a = np.radians(p["ang"])
    objs = [{"type": "SingleRay", "p1": {"x": 0.0, "y": float(p["ray_y"] + dy)},
             "p2": {"x": float(np.cos(a)), "y": float(p["ray_y"] + dy + np.sin(a))},
             "brightness": 0.85} for dy in np.linspace(-4.0, 4.0, HQ_RAYS)]
    objs += [FD.element(p),
             {"type": "Detector", "p1": {"x": FD.X_SENSOR, "y": -120},
              "p2": {"x": FD.X_SENSOR, "y": 120}},
             {"type": "CropBox", "p1": {"x": CROP_X[0], "y": CROP_Y[0]},
              "p4": {"x": CROP_X[1], "y": CROP_Y[1]}, "width": HQ_W}]
    res, _ = runner.simulate({"version": 5, "maxRayDepth": HQ_DEPTH, "objs": objs})
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        print(f"  ! render failed for {path.name}: {res.get('error')}")
        return None
    out = uniq(path)
    out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
    out.with_suffix(".json").write_text(json.dumps({"geometry": p, **meta}, indent=2))
    return out


def figure(path, rows, base, spot_tol, picks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure] skipped ({e})")
        return None
    th = [r["thick"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(th, [r["spot"] for r in rows], "o-", color="tab:blue")
    ax[0].axhline(base["spot"], ls="--", color="grey", label=f"baseline {base['spot']:.3f}")
    ax[0].axhline(spot_tol, ls=":", color="tab:red", label=f"spot tol {spot_tol:.3f}")
    ax[0].set(title="focus vs thickness", xlabel="element thickness (mm)",
              ylabel="RMS spot (21 rays)", yscale="log")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(th, [r["focthru"] for r in rows], "s-", color="tab:green",
               label="focused throughput")
    ax[1].plot(th, [r["efficiency"] for r in rows], "^-", color="tab:olive",
               label="efficiency (delivered)")
    ax[1].axhline(base["focthru"], ls="--", color="grey",
                  label=f"baseline {base['focthru']:.3f}")
    ax[1].set(title="energy vs thickness", xlabel="element thickness (mm)",
              ylabel="fraction of emitted", ylim=(0, 1))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(th, [r["tir_count"] for r in rows], "d-", color="tab:orange",
               label="TIR bounces (dominant)")
    ax[2].plot(th, [r["crit_margin"] for r in rows], "v-", color="tab:purple",
               label="min margin past critical (deg)")
    ax[2].axhline(0, ls=":", color="red")
    ax[2].set(title="fold structure vs thickness", xlabel="element thickness (mm)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    for a in ax:
        for pk in picks:
            a.axvline(pk, color="k", alpha=.25, lw=1)
        a.invert_xaxis()
    fig.suptitle(f"Thinning continuation: baseline thick={base['thick']:.1f} mm, "
                 f"spot={base['spot']:.3f}, focused throughput={base['focthru']:.3f}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = uniq(path)
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, help="geometry sidecar JSON (with a 'geometry' key)")
    ap.add_argument("--min-thick", type=float, default=8.0, help="thinnest thickness to try (mm)")
    ap.add_argument("--step", type=float, default=2.5, help="thickness decrement per step (mm)")
    ap.add_argument("--gens", type=int, default=30, help="CMA-ES generations per thickness")
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.05, help="CMA sigma in normalized DOF units")
    ap.add_argument("--seeds", default="0,1", help="comma list of polish seeds per thickness")
    ap.add_argument("--params", default=",".join(FREE_DOF),
                    help=f"DOF free to compensate (subset of {PARAM_NAMES}, never 'thick')")
    ap.add_argument("--spot-tol", type=float, default=None,
                    help="spot allowed before penalty; default max(1.3*baseline, 0.10)")
    ap.add_argument("--spot-penalty", type=float, default=2.0)
    ap.add_argument("--min-tir", type=int, default=2, help="dominant path must keep >= this many TIRs")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="hard floor (deg) on the dominant ray's minimum incidence margin "
                         "past critical. 0 = off, which lets the continuation buy thinness "
                         "with knife-edge TIR; a few deg keeps the fold manufacturable.")
    ap.add_argument("--tol-k", type=int, default=24,
                    help="perturbations for the manufacturing-tolerance check (0 to skip)")
    ap.add_argument("--tol-sigma", type=float, default=0.02,
                    help="jitter as a fraction of each physical DOF's search range")
    ap.add_argument("--keep-frac", type=float, default=0.97,
                    help="a pick must keep >= this fraction of the baseline focused throughput")
    ap.add_argument("--validate-rays", type=int, default=81,
                    help="dense re-check of baseline and picks (0 to skip)")
    ap.add_argument("--out", default=str(OUTDIR))
    args = ap.parse_args()

    design_path = Path(args.design)
    d = json.loads(design_path.read_text())
    p0 = d["geometry"] if "geometry" in d else d
    x0 = np.array([float(p0[n]) for n in PARAM_NAMES])
    free = [s.strip() for s in args.params.split(",") if s.strip()]
    if "thick" in free:
        raise SystemExit("'thick' is the continuation variable -- it cannot also be free")
    bad = [n for n in free if n not in PARAM_NAMES]
    if bad:
        raise SystemExit(f"unknown DOF {bad}; choose from {PARAM_NAMES}")
    free_idx = np.array([PARAM_NAMES.index(n) for n in free])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = design_path.stem

    t_start = float(x0[I_THICK])
    schedule = []
    t = t_start - args.step
    while t >= args.min_thick - 1e-9:
        schedule.append(round(float(max(t, args.min_thick)), 4))
        t -= args.step

    print("=" * 78)
    print("THINNING CONTINUATION (thickness homotopy with per-step CMA-ES polish)")
    print("=" * 78)
    print(f"design      : {design_path}")
    print(f"free DOF    : {free}   (frozen: {[n for n in PARAM_NAMES if n not in free]})")
    print(f"schedule    : {t_start:.1f} -> {schedule[-1]:.1f} mm in {len(schedule)} steps "
          f"of {args.step} mm, {len(seeds)} seed(s) x {args.gens}x{args.popsize} evals each")

    t0 = time.time()
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        base = measure(runner, x0)
        if base is None:
            raise SystemExit("baseline geometry is invalid under the engine")
        spot_tol = args.spot_tol if args.spot_tol is not None else max(1.3 * base["spot"], 0.10)
        print(f"baseline    : thick={t_start:.2f} spot={base['spot']:.4f} "
              f"focthru={base['focthru']:.4f} eff={base['efficiency']:.4f} "
              f"tir={base['tir_count']} margin={base['crit_margin']:.2f} deg")
        print(f"              {base['dom_signature']}")
        print(f"objective   : focthru - {args.spot_penalty} * max(0, spot - {spot_tol:.4f}), "
              f"reject tir < {args.min_tir}"
              + (f" or margin < {args.min_margin:.2f} deg" if args.min_margin > 0 else
                 " (no margin floor)"))
        print("-" * 78)

        rows = [dict(thick=t_start, **{k: base[k] for k in
                     ("spot", "focthru", "efficiency", "dom_share", "tir_count",
                      "crit_margin", "dom_signature")},
                     score=score_of(base, spot_tol=spot_tol, spot_penalty=args.spot_penalty,
                                    min_tir=args.min_tir, min_margin=args.min_margin),
                     genotype=[float(v) for v in x0], evals=0, seed=None)]

        x_prev = x0.copy()          # continuation state: previous thickness' solution
        n_ev_total = 0
        for tk in schedule:
            best = None
            for sd in seeds:
                # warm starts: previous step's solution, and the original design
                for x_st in ([x_prev, x0] if not np.allclose(x_prev, x0) else [x0]):
                    x_b, s_b, m_b, n_ev = polish(
                        runner, x_st, tk, free_idx, gens=args.gens, popsize=args.popsize,
                        sigma0=args.sigma0, seed=sd, spot_tol=spot_tol,
                        spot_penalty=args.spot_penalty, min_tir=args.min_tir,
                        min_margin=args.min_margin)
                    n_ev_total += n_ev
                    if m_b is not None and (best is None or s_b > best[1]):
                        best = (x_b, s_b, m_b, sd)
            if best is None:
                print(f"  thick={tk:5.2f}  NO valid fold found (>= {args.min_tir} TIR)")
                rows.append(dict(thick=tk, spot=float("nan"), focthru=0.0, efficiency=0.0,
                                 dom_share=0.0, tir_count=0, crit_margin=float("nan"),
                                 dom_signature="", score=-1e9, genotype=None,
                                 evals=n_ev_total, seed=None))
                continue
            x_b, s_b, m_b, sd = best
            x_prev = x_b.copy()
            rows.append(dict(thick=tk, **{k: m_b[k] for k in
                             ("spot", "focthru", "efficiency", "dom_share", "tir_count",
                              "crit_margin", "dom_signature")},
                             score=float(s_b), genotype=[float(v) for v in x_b],
                             evals=n_ev, seed=int(sd)))
            keep = m_b["focthru"] >= args.keep_frac * base["focthru"]
            flag = "OK " if (keep and m_b["spot"] <= spot_tol) else "   "
            print(f"  {flag}thick={tk:5.2f}  spot={m_b['spot']:.4f}  "
                  f"focthru={m_b['focthru']:.4f}  eff={m_b['efficiency']:.4f}  "
                  f"tir={m_b['tir_count']}  margin={m_b['crit_margin']:5.2f}  "
                  f"{m_b['dom_signature']}", flush=True)

        # ---- picks: thinnest that keeps the focus, plus a conservative mid pick ----
        good = [r for r in rows[1:] if r["genotype"] and r["spot"] <= spot_tol
                and r["focthru"] >= args.keep_frac * base["focthru"]]
        picks = []
        if good:
            thinnest = min(good, key=lambda r: r["thick"])
            picks.append(("thinnest", thinnest))
            mid = [r for r in good if r["thick"] > thinnest["thick"]]
            if mid:
                # most conservative pick that is still a real reduction: best focus among
                # the thicker survivors
                picks.append(("safest", min(mid, key=lambda r: r["spot"])))

        # ---- dense re-validation + tolerance of baseline + picks ----
        dense, tol = {}, {}
        if args.validate_rays:
            dense["baseline"] = measure(runner, x0, n_rays=args.validate_rays)
            for tag, r in picks:
                dense[tag] = measure(runner, np.array(r["genotype"]),
                                     n_rays=args.validate_rays)
        if args.tol_k:
            tol["baseline"] = tolerance(runner, x0, k=args.tol_k,
                                        sigma_frac=args.tol_sigma, min_tir=args.min_tir)
            for tag, r in picks:
                tol[tag] = tolerance(runner, np.array(r["genotype"]), k=args.tol_k,
                                     sigma_frac=args.tol_sigma, min_tir=args.min_tir)

        # ---- renders ----
        renders = {}
        rb = render(runner, FD.vec_to_params(x0), out / f"{stem}_baseline.png",
                    dict(role="baseline", **{k: base[k] for k in
                         ("spot", "focthru", "efficiency", "tir_count", "dom_signature")}))
        if rb:
            renders["baseline"] = str(rb)
        for tag, r in picks:
            name = f"{stem}_thin{r['thick']:.1f}".replace(".", "p") + ".png"
            pr = render(runner, FD.vec_to_params(np.array(r["genotype"])), out / name,
                        dict(role=tag, thick=r["thick"], **{k: r[k] for k in
                             ("spot", "focthru", "efficiency", "tir_count", "dom_signature")}))
            if pr:
                renders[tag] = str(pr)

    dt = time.time() - t0
    print("-" * 78)
    if not picks:
        print("RESULT: no thinner design met the focus + throughput bars. "
              "Widen --spot-tol / lower --keep-frac, or use a finer --step.")
    for tag, r in picks:
        print(f"RESULT [{tag}] thick {t_start:.1f} -> {r['thick']:.1f} mm "
              f"({100*(1-r['thick']/t_start):.0f}% thinner)")
        print(f"    spot     {base['spot']:.4f} -> {r['spot']:.4f}")
        print(f"    focthru  {base['focthru']:.4f} -> {r['focthru']:.4f}")
        print(f"    eff      {base['efficiency']:.4f} -> {r['efficiency']:.4f}")
        print(f"    fold     {base['tir_count']} TIR -> {r['tir_count']} TIR   {r['dom_signature']}")
        print(f"    margin   {base['crit_margin']:.2f} -> {r['crit_margin']:.2f} deg past critical")
        if dense.get(tag):
            print(f"    dense re-check ({args.validate_rays} rays): spot "
                  f"{dense['baseline']['spot']:.4f} -> {dense[tag]['spot']:.4f}, focthru "
                  f"{dense['baseline']['focthru']:.4f} -> {dense[tag]['focthru']:.4f}")
        if tol.get(tag):
            tb, tt = tol["baseline"], tol[tag]
            print(f"    tolerance ({args.tol_k} x {100*args.tol_sigma:.0f}% DOF jitter): "
                  f"focthru retained {tb['focthru_retained']:.2f} -> {tt['focthru_retained']:.2f}, "
                  f"median spot {tb['spot_median']:.3f} -> {tt['spot_median']:.3f}, "
                  f"fold kept {tb['fold_kept_frac']:.0%} -> {tt['fold_kept_frac']:.0%}")
        print(f"    geometry {json.dumps({n: round(v, 6) for n, v in zip(PARAM_NAMES, r['genotype'])})}")

    fig = figure(out / f"{stem}_thinning.png", [r for r in rows if r["genotype"]],
                 dict(thick=t_start, **base), spot_tol, [r["thick"] for _t, r in picks])
    rep = uniq(out / f"{stem}_thinning.json")
    rep.write_text(json.dumps(dict(
        meta=dict(design=str(design_path), free_dof=free, frozen=[n for n in PARAM_NAMES
                                                                 if n not in free],
                  schedule=schedule, gens=args.gens, popsize=args.popsize,
                  sigma0=args.sigma0, seeds=seeds, spot_tol=spot_tol,
                  spot_penalty=args.spot_penalty, min_tir=args.min_tir,
                  min_margin=args.min_margin, keep_frac=args.keep_frac,
                  validate_rays=args.validate_rays, tol_k=args.tol_k,
                  tol_sigma=args.tol_sigma,
                  evals=int(n_ev_total), wall_seconds=dt,
                  metrics="focused_throughput/spot/efficiency as in run_fold_qd_descriptors"),
        baseline=dict(thick=t_start, genotype=[float(v) for v in x0], **base),
        sweep=rows, picks={tag: r for tag, r in picks}, dense_validation=dense,
        tolerance=tol, renders=renders), indent=2, default=float))
    print("-" * 78)
    print(f"[sweep]  {rep}")
    if fig:
        print(f"[figure] {fig}")
    for k, v in renders.items():
        print(f"[render] {k}: {v}")
    print(f"[done]   {n_ev_total} engine evaluations in {dt:.0f}s")


if __name__ == "__main__":
    main()
