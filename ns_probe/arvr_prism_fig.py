"""
AR/VR FREEFORM PRISM -- FIGURE-FAITHFUL geometry (Cheng-Hua wedge eyepiece layout).

This re-lays the near-eye relay to match the canonical freeform-prism figure instead of
the rotated 90deg rig in arvr_prism.py:

      MICRODISPLAY  (top-left, panel tilted ~45deg to horizontal, 3 point sources)
            \\  emits down-and-right into the prism
             \\
        +-----------------+        FREEFORM PRISM  (all 4 edges curved freeform;
        |   freeform      |        the body is the proven 9-DOF element from
        |   wedge prism   |        arvr_prism.prism_element -- A/B faces tiltable,
        +-----------------+        T/D edges curved, uncoated -> TIR-only)
                |  collimated beam leaves the bottom face travelling DOWN
                v
        =================            EXIT PUPIL  (flat / horizontal, at the bottom)
              (eye looks UP)         <- the "sensor": eye sits below, looking +y

WHY THIS GEOMETRY FORCES A WRAP (and lets 3-fold compete):
  The display is ABOVE and the eyebox is BELOW, both with light travelling broadly
  DOWNWARD -- but the eye needs a COLLIMATED beam exiting the BOTTOM face. A steep
  input ray hitting the bottom face lands BEYOND the critical angle -> it cannot exit
  there, it TIRs back UP, reflects off the upper (back) face, and only returns to the
  bottom face at a shallow enough angle to refract OUT toward the eye. The bottom face
  does double duty (TIR, then refraction), exactly as in the real freeform prism. The
  wrap is not prescribed; it is the only way to satisfy collimation + reach the bottom
  eyebox, so the search must discover it.

IMAGING METRIC = COLLIMATION (virtual image at infinity), identical in spirit to
arvr_prism.arvr_metrics but read on a HORIZONTAL bottom pupil (eyebox in x, rays must
exit travelling -y). Fitness = mean_s[ T_s * exp(-sigma_s/SIGMA0) ] * ordering_ok.
Fitness NEVER names or rewards folding.

QD ARCHIVE (DESCRIPTIVE axes):
  BC1 = fold_count  = # TIR bounces on the dominant eyebox-reaching path (0..MAX_FOLD)
  BC2 = FOV (deg)   = angular span of the two edge fields at the pupil
  -> the archive shows WHICH fold counts can collimate WHICH fields-of-view. Whether a
     clean 3-fold relay exists is then read directly off the fold_count=3 row.

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY ns_probe/arvr_prism_fig.py --hand          # plumbing: do rays reach the pupil?
    $PY ns_probe/arvr_prism_fig.py --evals 8000     # QD search + renders
    $PY ns_probe/arvr_prism_fig.py --evals 8000 --tilt-range 28   # optional fold nudge
"""
from __future__ import annotations

import argparse
import base64
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
import arvr_prism as AV          # reuse the proven 9-DOF freeform body + genotype bounds
import fold_demo as FD

# ---------------------------------------------------------------------------
# FIGURE-FAITHFUL geometry. World frame: x = horizontal (right +), y = vertical (up +).
# The prism body (AV.prism_element) is centred on y=0, spanning y in [-ELEM_H, +ELEM_H]
# and x around `cx`. We place the display above-left and the eyebox flat below.
# ---------------------------------------------------------------------------
ELEM_H = AV.ELEM_H                   # 30 (prism half-height)

# Microdisplay: 3 point sources on a panel tilted LINE_DEG to horizontal, up at top-left.
DISP_CX = 36.0                       # display centre x (left of / over the prism's left)
DISP_CY = 56.0                       # display centre y (above the prism top, with clearance)
DISP_HALF = 6.0                      # half-length of the source line (3 sources span this)
LINE_DEG = 45.0                      # panel tilt: source line at ~45deg to horizontal
SOURCE_AIM = -45.0                   # central emission dir (deg): down-and-right into prism
N_FAN = 17                           # rays per source (a pixel radiates a diverging fan)
FAN_HALFANGLE = 12.0                 # deg, fan half-cone about SOURCE_AIM
SIGMA0 = AV.SIGMA0                    # deg, angular tolerance in the fitness (0.7)

# Exit pupil: FLAT (horizontal) detector at the bottom; eye is below, looking UP (+y),
# so imaging rays must leave the prism travelling DOWN (diry < 0). Eyebox is generous and
# its centre is fixed; the search discovers where the folded beam wants to exit.
PUPIL_Y = -52.0                      # bottom pupil plane (rays cross travelling -y)
PUPIL_CX = 66.0                      # eyebox centre x (under the prism)
EYEBOX_HALF = 11.0                   # generous eyebox half-width (in x)
PUPIL_SPAN = 110.0                   # half-length of the physical detector segment
MAX_DEPTH = 28                       # ray-depth budget: enough for >=3 internal bounces


def _source_line():
    """3 sources along the tilted display panel, ordered along the line (param t)."""
    u = np.array([np.cos(np.radians(LINE_DEG)), np.sin(np.radians(LINE_DEG))])
    ts = np.linspace(-DISP_HALF, DISP_HALF, 3)
    return [(float(DISP_CX + t * u[0]), float(DISP_CY + t * u[1])) for t in ts]


SOURCE_PTS = _source_line()


def _source_index(px, py):
    return int(np.argmin([(px - sx) ** 2 + (py - sy) ** 2 for sx, sy in SOURCE_PTS]))


def geometry_valid_fig(p):
    """Non-degenerate body that clears the display above and the pupil below."""
    if not FD.geometry_valid(p):
        return False
    top = ELEM_H + max(p["tsag"], 0.0)
    bot = -(ELEM_H + max(p["dsag"], 0.0))
    return (top < DISP_CY - 6.0) and (bot > PUPIL_Y + 4.0)


def scene_fig(x, *, n_fan=N_FAN, fan_halfangle=FAN_HALFANGLE, brightness=1.0,
              record=True, crop=False, max_depth=MAX_DEPTH):
    p = AV.prism_dict(x)
    objs = []
    for (sx, sy) in SOURCE_PTS:
        for a in np.radians(SOURCE_AIM + np.linspace(-fan_halfangle, fan_halfangle, n_fan)):
            objs.append({"type": "SingleRay", "p1": {"x": sx, "y": sy},
                         "p2": {"x": sx + float(np.cos(a)), "y": sy + float(np.sin(a))},
                         "brightness": float(brightness)})
    objs.append(AV.prism_element(p))                              # 4-curved-face TIR prism
    objs.append({"type": "Detector",                             # FLAT bottom pupil
                 "p1": {"x": PUPIL_CX - PUPIL_SPAN, "y": PUPIL_Y},
                 "p2": {"x": PUPIL_CX + PUPIL_SPAN, "y": PUPIL_Y}})
    sc = {"version": 5, "maxRayDepth": max_depth, "objs": objs}
    if record:
        sc["recordInteractions"] = True
    if crop:
        objs.append({"type": "CropBox",
                     "p1": {"x": -12, "y": PUPIL_Y - 8},
                     "p4": {"x": PUPIL_CX + 90, "y": DISP_CY + 10},
                     "width": 1800})
    return sc


def arvr_metrics_fig(res, *, n_fan=N_FAN):
    """Per-source angular metrics at the FLAT bottom eyebox + scalar fitness.
    A ray counts if it leaves travelling DOWN (diry<0), crosses the pupil plane, and
    lands inside the eyebox (|x - PUPIL_CX| <= EYEBOX_HALF). Attributed to its source by
    the emission vertex."""
    per = {i: {"ang": [], "w": []} for i in range(len(SOURCE_PTS))}
    for r in res.get("raySequences", []):
        if r.get("diry", 0) >= 0 or r.get("b", 0) <= 0:          # must exit toward the eye (-y)
            continue
        if abs(float(r["y"]) - PUPIL_Y) > 3.0:                   # actually at the pupil plane
            continue
        if abs(float(r["x"]) - PUPIL_CX) > EYEBOX_HALF:          # inside the eyebox?
            continue
        sx0, sy0 = float(r["seq"][0]["x"]), float(r["seq"][0]["y"])
        src = _source_index(sx0, sy0)
        per[src]["ang"].append(np.degrees(np.arctan2(r["diry"], r["dirx"])))
        per[src]["w"].append(float(r["b"]))

    fields = []
    for i in range(len(SOURCE_PTS)):
        a = np.array(per[i]["ang"]); w = np.array(per[i]["w"])
        if w.sum() <= 0:
            fields.append(dict(present=False, T=0.0, theta=None, sigma=None, n=0))
            continue
        theta = float(np.average(a, weights=w))
        sigma = float(np.sqrt(np.average((a - theta) ** 2, weights=w)))
        fields.append(dict(present=True, T=float(w.sum() / n_fan), theta=theta,
                           sigma=sigma, n=int(len(a))))

    present = [f for f in fields if f["present"]]
    thetas = [f["theta"] for f in fields]
    ordering_ok = (len(present) == 3 and
                   (thetas[0] < thetas[1] < thetas[2] or thetas[0] > thetas[1] > thetas[2]))
    q = [f["T"] * np.exp(-f["sigma"] / SIGMA0) if f["present"] else 0.0 for f in fields]
    fitness = float(np.mean(q)) * (1.0 if ordering_ok else 0.0)
    fov = (abs(fields[2]["theta"] - fields[0]["theta"])
           if fields[0]["present"] and fields[2]["present"] else 0.0)
    return dict(fields=fields, fitness=fitness, fov=float(fov),
                ordering_ok=bool(ordering_ok),
                T_min=float(min(f["T"] for f in fields)),
                sigma_mean=float(np.mean([f["sigma"] for f in present])) if present else None)


def fold_count(res):
    """# TIR bounces on the densest eyebox-reaching path (DESCRIPTIVE; not in fitness)."""
    best, bestb = 0, -1.0
    for r in res.get("raySequences", []):
        if r.get("diry", 0) >= 0 or abs(float(r["y"]) - PUPIL_Y) > 3.0:
            continue
        if abs(float(r["x"]) - PUPIL_CX) > EYEBOX_HALF:
            continue
        if r.get("b", 0) > bestb:
            bestb = r["b"]
            best = sum(1 for v in r["seq"] if v.get("t") == "tir")
    return best


def report(m):
    print(f"  fitness={m['fitness']:.4f}  FOV={m['fov']:.2f} deg  "
          f"ordering_ok={m['ordering_ok']}  min throughput={m['T_min']:.2f}")
    print(f"  {'src':>4} {'(x,y)':>15} {'present':>7} {'T':>6} {'theta':>9} {'sigma':>8} {'n':>4}")
    for i, f in enumerate(m["fields"]):
        ty = f"{f['theta']:.3f}" if f["present"] else "-"
        sg = f"{f['sigma']:.3f}" if f["present"] else "-"
        px, py = SOURCE_PTS[i]
        print(f"  {i:>4} {f'({px:.1f},{py:.1f})':>15} {str(f['present']):>7} "
              f"{f['T']:>6.2f} {ty:>9} {sg:>8} {f['n']:>4}")


# ---------------------------------------------------------------------------
# QD search (CMA-ME over the 9-DOF freeform body; descriptive (fold, FOV) archive)
# ---------------------------------------------------------------------------
MAX_FOLD = 5
BC_BINS = (MAX_FOLD + 1, 24)
BC_BOUNDS = ((0.0, MAX_FOLD + 1.0), (0.0, 40.0))
MIN_TIR = 0                              # optional hard gate (default off = descriptive)


def evaluate(runner, x):
    p = AV.prism_dict(x)
    if not geometry_valid_fig(p):
        return None
    try:
        res, _ = runner.simulate(scene_fig(x))
    except Exception:
        return None
    if res.get("error"):
        return None
    m = arvr_metrics_fig(res)
    if not m["ordering_ok"] or m["fitness"] <= 0:        # only real 3-field relays
        return None
    fc = fold_count(res)
    if fc < MIN_TIR:
        return None
    fc_bc = float(min(fc, MAX_FOLD))
    return dict(fitness=float(m["fitness"]), bc=(fc_bc, float(m["fov"])),
                info=dict(fov=float(m["fov"]), folds=int(fc),
                          fitness=float(m["fitness"]), T_min=float(m["T_min"]),
                          sigma_mean=float(m["sigma_mean"]),
                          thetas=[f["theta"] for f in m["fields"]],
                          sigmas=[f["sigma"] for f in m["fields"]],
                          Ts=[f["T"] for f in m["fields"]]))


def _render(runner, x, outdir, stem, m, folds):
    """Render the relay diagram (never overwrite) + a geometry/metric sidecar JSON."""
    rres, _ = runner.simulate(scene_fig(x, n_fan=7, brightness=0.6, record=False, crop=True))
    imgs = rres.get("images") or []
    if not (imgs and imgs[0].get("dataUrl")):
        return None
    out = outdir / f"{stem}.png"
    k = 0
    while out.exists():
        k += 1; out = outdir / f"{stem}_{k}.png"
    out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
    out.with_suffix(".json").write_text(json.dumps(dict(
        prism=AV.prism_dict(x), metrics=m, folds=int(folds),
        geometry=dict(layout="figure_faithful_cheng_hua",
                      disp_cx=DISP_CX, disp_cy=DISP_CY, disp_half=DISP_HALF,
                      line_deg=LINE_DEG, source_aim=SOURCE_AIM, n_fan=N_FAN,
                      fan_halfangle=FAN_HALFANGLE, pupil_y=PUPIL_Y, pupil_cx=PUPIL_CX,
                      eyebox_half=EYEBOX_HALF, max_depth=MAX_DEPTH, sigma0=SIGMA0,
                      source_pts=SOURCE_PTS)),
        indent=2, default=float))
    print(f"[render] {out.name} (folds={folds}, fit={m['fitness']:.3f}, FOV={m['fov']:.1f}) + sidecar")
    return out


def run_search(runner, args, outdir):
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
            byfold = {}
            for c in archive.cells.values():
                byfold.setdefault(int(c.info["folds"]), 0)
                byfold[int(c.info["folds"])] += 1
            rec = dict(evals=n, filled=len(archive.cells), best=archive.best_fitness,
                       invalid=n_invalid, byfold=dict(sorted(byfold.items())))
            history.append(rec)
            print(f"[fig-qd] evals={n:6d} filled={rec['filled']:3d} best={rec['best']:.4f} "
                  f"folds={rec['byfold']} invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)
    return archive, n, n_invalid, history, t0


def main():
    global MIN_TIR, SOURCE_AIM, EYEBOX_HALF, PUPIL_Y
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", action="store_true",
                    help="plumbing check: trace a few prisms, count rays reaching the pupil")
    ap.add_argument("--evals", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--tilt-range", type=float, default=None,
                    help="optional: cap face tilt to +-this (deg) so no single face does the "
                         "whole turn (geometry nudge toward multi-fold). Default off.")
    ap.add_argument("--source-aim", type=float, default=None,
                    help="override central emission direction (deg). Default %.0f." % SOURCE_AIM)
    ap.add_argument("--min-tir", type=int, default=0,
                    help="optional hard gate: reject designs with < this many TIR bounces.")
    ap.add_argument("--eyebox-half", type=float, default=None,
                    help="override eyebox half-width (default %.0f)." % EYEBOX_HALF)
    ap.add_argument("--pupil-y", type=float, default=None,
                    help="override flat-pupil plane y (default %.0f)." % PUPIL_Y)
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_arvr_fig"))
    args = ap.parse_args()

    MIN_TIR = args.min_tir
    if args.source_aim is not None:
        SOURCE_AIM = args.source_aim
    if args.eyebox_half is not None:
        EYEBOX_HALF = args.eyebox_half
    if args.pupil_y is not None:
        PUPIL_Y = args.pupil_y
    if args.tilt_range is not None:
        for nm in ("tilt_a", "tilt_b"):
            j = AV.PRISM_NAMES.index(nm)
            AV.LO[j], AV.HI[j] = -args.tilt_range, args.tilt_range

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[config] layout=figure  source_aim={SOURCE_AIM:.0f}deg  line={LINE_DEG:.0f}deg  "
          f"pupil=(flat y={PUPIL_Y}, cx={PUPIL_CX}, half={EYEBOX_HALF})  "
          f"tilt_range={args.tilt_range}  min_tir={MIN_TIR}  max_depth={MAX_DEPTH}")

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        if args.hand:
            print("\n[hand] plumbing: do rays reach the flat bottom pupil under this layout?")
            rng = np.random.default_rng(args.seed)
            best = None
            for k in range(400):
                x = rng.uniform(AV.LO, AV.HI)
                if not geometry_valid_fig(AV.prism_dict(x)):
                    continue
                res, _ = runner.simulate(scene_fig(x))
                if res.get("error"):
                    continue
                m = arvr_metrics_fig(res)
                hit = sum(f["n"] for f in m["fields"])
                if best is None or hit > best[0]:
                    best = (hit, x, m, res)
            if best is None:
                print("[hand] no rays reached the pupil at all -- geometry needs recalibration")
                return
            hit, x, m, res = best
            print(f"[hand] best-of-400 random: {hit} rays reached the eyebox")
            report(m)
            print(f"[hand] folds on dominant path = {fold_count(res)}")
            _render(runner, x, outdir, "fig_hand_plumbing", m, fold_count(res))
            return

        print(f"\n[search] CMA-ME, {args.evals} evals, descriptive (fold,FOV) archive...")
        archive, n, n_invalid, history, t0 = run_search(runner, args, outdir)

        # ---- summarise + archive JSON ----
        cells = []
        for (i0, i1), c in archive.cells.items():
            rec = dict(i0=int(i0), i1=int(i1), folds=int(c.bc[0]), fov=float(c.bc[1]),
                       fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
            rec.update(c.info)
            cells.append(rec)
        payload = dict(
            meta=dict(experiment="arvr_prism_fig_qd", layout="figure_faithful_cheng_hua",
                      evals=int(n), seed=int(args.seed), invalid=int(n_invalid),
                      bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                      bc_axes=["fold_count", "FOV_deg"],
                      fitness="well_collimated_throughput_3field (folding NOT rewarded)",
                      sigma0=SIGMA0, source_aim=SOURCE_AIM, line_deg=LINE_DEG,
                      tilt_range=args.tilt_range, min_tir=MIN_TIR,
                      geometry=dict(disp_cx=DISP_CX, disp_cy=DISP_CY, disp_half=DISP_HALF,
                                    pupil_y=PUPIL_Y, pupil_cx=PUPIL_CX,
                                    eyebox_half=EYEBOX_HALF, max_depth=MAX_DEPTH,
                                    source_pts=SOURCE_PTS),
                      param_names=AV.PRISM_NAMES, wall_seconds=float(time.time() - t0)),
            history=history, cells=cells)
        (outdir / "archive.json").write_text(json.dumps(payload, indent=2))

        # per-fold summary
        byfold = {}
        for c in cells:
            byfold.setdefault(c["folds"], []).append(c)
        print(f"\n[fig-qd] {len(cells)} cells, {n} evals, {payload['meta']['wall_seconds']:.0f}s")
        print(f"  {'folds':>5} {'cells':>6} {'best_fit':>9} {'best_FOV':>9} {'min_sigma':>10}")
        for fc in sorted(byfold):
            cs = byfold[fc]
            bf = max(cs, key=lambda c: c["fitness"])
            print(f"  {fc:>5} {len(cs):>6} {bf['fitness']:>9.4f} "
                  f"{max(c['fov'] for c in cs):>9.2f} {min(c['sigma_mean'] for c in cs):>10.3f}")

        # ---- renders: best overall, best multi-fold (>=3 if any), widest FOV ----
        def _x(c): return np.array(c["genotype"])
        best = max(cells, key=lambda c: c["fitness"])
        res, _ = runner.simulate(scene_fig(_x(best)))
        _render(runner, _x(best), outdir, "fig_best", arvr_metrics_fig(res), best["folds"])

        multi = [c for c in cells if c["folds"] >= 3]
        if multi:
            bm = max(multi, key=lambda c: c["fitness"])
            res, _ = runner.simulate(scene_fig(_x(bm)))
            _render(runner, _x(bm), outdir, "fig_best_3fold", arvr_metrics_fig(res), bm["folds"])
        else:
            print("[fig-qd] no >=3-fold cell found; rendering best 2-fold instead")
            two = [c for c in cells if c["folds"] >= 2]
            if two:
                bm = max(two, key=lambda c: c["fitness"])
                res, _ = runner.simulate(scene_fig(_x(bm)))
                _render(runner, _x(bm), outdir, "fig_best_2fold", arvr_metrics_fig(res), bm["folds"])

        wide = max(cells, key=lambda c: c["fov"])
        res, _ = runner.simulate(scene_fig(_x(wide)))
        _render(runner, _x(wide), outdir, "fig_widest_fov", arvr_metrics_fig(res), wide["folds"])
        print(f"\n[fig-qd] done. archive + renders in {outdir}")


if __name__ == "__main__":
    main()
