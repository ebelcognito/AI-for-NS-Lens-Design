"""
AR/VR DISPLAY PRISM — Phase A: scene, angular metric, and a sanity search+render.

Repurposes the non-sequential TIR engine for a near-eye display relay. The forward model:

  * a MICRODISPLAY sampled by 3 POINT SOURCES (center + two edges), each emitting a
    DIVERGING FAN of rays (a pixel radiates);
  * a searched TIR PRISM (the 7-DOF tilt element from fold_demo, uncoated -> TIR-only,
    loss-free) folds the light;
  * an EYE-PUPIL PLANE with an EYEBOX aperture (the aperture stop) catches it.

Imaging condition = COLLIMATED (virtual image at infinity): a good relay turns each
display point into a BUNDLE OF PARALLEL rays at the pupil. So the imaging metric is the
ANGULAR spread of each source's eyebox rays (not a spatial spot):

  for each source s (rays landing inside the eyebox):
    T_s     = energy reaching the eyebox / energy emitted          (pupil throughput)
    theta_s = energy-weighted mean ray DIRECTION                   (output field angle)
    sigma_s = energy-weighted std of ray directions (deg)          (collimation error)

  FOV       = theta(+edge) - theta(-edge)                          (field of view)
  fitness   = mean_s [ T_s * exp(-sigma_s / SIGMA0) ] * ordering_ok
              (well-collimated throughput over the 3 fields; ordering_ok requires the 3
               field angles to be strictly monotonic in display position -> no scrambling)

Phase-A goals: confirm the plumbing + metric, and find ONE prism that relays the 3 fields
through the eyebox, then render it. (FOV x form-factor QD archive is Phase B.)

    python3 ns_probe/arvr_prism.py            # search a working prism + render
    python3 ns_probe/arvr_prism.py --hand     # plumbing check on a fixed prism
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import fold_demo as FD
from cma_me import CMAES

# ---- OFF-AXIS near-eye geometry (the real AR-combiner layout) ----
# The eye looks forward (+x) at the pupil on the right. The microdisplay is mounted ABOVE
# the optical axis and emits DOWNWARD, so its light must be FOLDED ~90 deg into the eye's
# line of sight. A straight transmission path is geometrically impossible -> a TIR fold is
# REQUIRED. (This is exactly why real freeform AR prisms fold; the collinear version did not.)
PUPIL_X = 120.0                      # eye-pupil plane (light enters the eye travelling +x)
EYEBOX_Y = 0.0                       # eyebox center (on the forward axis)
EYEBOX_HALF = 9.0                    # eyebox half-height (catchable pupil for the folded beam)
DISPLAY_Y = 50.0                     # microdisplay above the element (clearance for top curvature)
DISPLAY_CX = 55.0                    # centered over the prism region
DISPLAY_HALF_W = 6.0                 # 3 sources at x = 49, 55, 61
SOURCE_XS = np.linspace(DISPLAY_CX - DISPLAY_HALF_W, DISPLAY_CX + DISPLAY_HALF_W, 3)
SOURCE_PTS = [(float(sx), DISPLAY_Y) for sx in SOURCE_XS]
SOURCE_AIM = -90.0                   # central emission direction = straight down (deg)
N_FAN = 17                           # rays per source
FAN_HALFANGLE = 12.0                 # deg, fan half-cone about SOURCE_AIM
SIGMA0 = 0.7                         # deg, angular tolerance in the fitness

# prism genotype = the element DOF only (display + eyebox are fixed). ALL FOUR edges are
# curved freeform surfaces (4 sags) -- in the off-axis fold, the top (entry) and bottom (TIR)
# edges carry imaging light, so curving them adds power exactly where it is used. This makes
# the element a real 4-surface freeform prism rather than a 2-curved-face block.
ELEM_H = FD.ELEM_H                                # 30 (element half-height)
SAG = 15.0                                        # defined sag range: each face bulge in [-SAG, SAG]
TILT = 50.0                                       # face tilt range: a face can reach ~45deg,
PRISM_NAMES = ["cx", "thick", "fsag", "bsag", "tsag", "dsag", "tilt_a", "tilt_b", "n"]
#                left  right  top    bottom            face tilts (need ~45deg to fold 90deg)
LO = np.array([35.0,  6.0,  -SAG, -SAG, -SAG, -SAG, -TILT, -TILT, 1.50])
HI = np.array([85.0, 45.0,   SAG,  SAG,  SAG,  SAG,  TILT,  TILT,  2.00])


def prism_dict(x):
    return dict(zip(PRISM_NAMES, [float(v) for v in x]))


def prism_element(p):
    """Glass prism with ALL FOUR edges curved + tiltable left/right faces (uncoated -> TIR).
    Faces: A=left, B=right (sag in x); T=top, D=bottom (sag in y). sag>0 = convex outward."""
    lx, rx = p["cx"] - p["thick"] / 2, p["cx"] + p["thick"] / 2
    dA = np.tan(np.radians(p["tilt_a"])) * ELEM_H
    dB = np.tan(np.radians(p["tilt_b"])) * ELEM_H
    bl, br = (lx + dA, -ELEM_H), (rx + dB, -ELEM_H)          # bottom corners
    tr, tl = (rx - dB, ELEM_H), (lx - dA, ELEM_H)            # top corners
    path = [                                                  # corner, arc-mid, corner, ...
        {"x": bl[0], "y": bl[1], "arc": False},
        {"x": (bl[0] + br[0]) / 2, "y": -ELEM_H - p["dsag"], "arc": True},   # bottom edge
        {"x": br[0], "y": br[1], "arc": False},
        {"x": rx + p["bsag"], "y": 0.0, "arc": True},                        # right face B
        {"x": tr[0], "y": tr[1], "arc": False},
        {"x": (tr[0] + tl[0]) / 2, "y": ELEM_H + p["tsag"], "arc": True},    # top edge
        {"x": tl[0], "y": tl[1], "arc": False},
        {"x": lx - p["fsag"], "y": 0.0, "arc": True},                        # left face A
    ]
    return {"type": "Glass", "refIndex": float(p["n"]),
            "path": [{"x": float(v["x"]), "y": float(v["y"]), "arc": v["arc"]} for v in path]}


def geometry_valid_arvr(p):
    """Non-degenerate body + the curved top stays clear below the display."""
    return FD.geometry_valid(p) and (ELEM_H + max(p["tsag"], 0.0) < DISPLAY_Y - 2.0)


def scene_arvr(x, *, n_fan=N_FAN, fan_halfangle=FAN_HALFANGLE, brightness=1.0,
               record=True, crop=False, max_depth=16):
    p = prism_dict(x)
    objs = []
    for (sx, sy) in SOURCE_PTS:
        for a in np.radians(SOURCE_AIM + np.linspace(-fan_halfangle, fan_halfangle, n_fan)):
            objs.append({"type": "SingleRay", "p1": {"x": sx, "y": sy},
                         "p2": {"x": sx + float(np.cos(a)), "y": sy + float(np.sin(a))},
                         "brightness": float(brightness)})
    objs.append(prism_element(p))                               # 4-curved-face TIR prism
    objs.append({"type": "Detector", "p1": {"x": PUPIL_X, "y": -120},
                 "p2": {"x": PUPIL_X, "y": 120}})
    sc = {"version": 5, "maxRayDepth": max_depth, "objs": objs}
    if record:
        sc["recordInteractions"] = True
    if crop:
        # frame the FULL off-axis layout: display at y~DISPLAY_Y (top), prism down to
        # ~-(ELEM_H+SAG), eye plane at PUPIL_X. Generous margins so nothing is clipped.
        objs.append({"type": "CropBox",
                     "p1": {"x": -10, "y": -(ELEM_H + SAG) - 12},
                     "p4": {"x": PUPIL_X + 14, "y": DISPLAY_Y + 12},
                     "width": 1800})
    return sc


def _source_index(seq_x):
    return int(np.argmin([abs(seq_x - sx) for sx, _ in SOURCE_PTS]))


def arvr_metrics(res, *, n_fan=N_FAN):
    """Per-source angular metrics at the eyebox + scalar fitness. Sources are mounted at
    distinct x positions (top-firing display), so a ray is attributed to its source by the
    x of its emission vertex."""
    per = {i: {"ang": [], "w": []} for i in range(len(SOURCE_PTS))}
    for r in res.get("raySequences", []):
        if r.get("dirx", 0) <= 0 or r.get("b", 0) <= 0:         # must exit toward the eye (+x)
            continue
        if abs(float(r["y"]) - EYEBOX_Y) > EYEBOX_HALF:          # inside the eyebox?
            continue
        src = _source_index(float(r["seq"][0]["x"]))
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


def evaluate(runner, x):
    p = prism_dict(x)
    if not geometry_valid_arvr(p):
        return -1.0, None
    try:
        res, _ = runner.simulate(scene_arvr(x))
    except Exception:
        return -1.0, None
    if res.get("error"):
        return -1.0, None
    m = arvr_metrics(res)
    return m["fitness"], m


def search(runner, *, n_random=2500, cma_gens=60, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, -np.inf
    for k in range(n_random):
        x = rng.uniform(LO, HI)
        f, _ = evaluate(runner, x)
        if f > best_f:
            best_f, best_x = f, x
            if verbose and f > 0:
                print(f"  [random {k}] fitness={f:.4f}")
    if best_x is None or best_f <= 0:
        return best_x, best_f
    es = CMAES((best_x - LO) / (HI - LO), 0.08, popsize=12, seed=seed,
               bounds=(np.zeros(len(LO)), np.ones(len(LO))))
    for g in range(cma_gens):
        X = es.ask()
        F = np.array([evaluate(runner, LO + xn * (HI - LO))[0] for xn in X])
        es.tell(np.argsort(-F))
        gi = int(np.argmax(F))
        if F[gi] > best_f:
            best_f, best_x = F[gi], LO + X[gi] * (HI - LO)
            if verbose:
                print(f"  [cma {g}] fitness={best_f:.4f}")
        if es.restart_needed:
            break
    return best_x, best_f


def report(m):
    print(f"  fitness={m['fitness']:.4f}  FOV={m['fov']:.2f} deg  "
          f"ordering_ok={m['ordering_ok']}  min throughput={m['T_min']:.2f}")
    print(f"  {'src':>4} {'disp_x':>7} {'present':>7} {'T':>6} {'theta(deg)':>11} {'sigma(deg)':>11} {'n':>4}")
    for i, f in enumerate(m["fields"]):
        ty = f"{f['theta']:.3f}" if f["present"] else "-"
        sg = f"{f['sigma']:.3f}" if f["present"] else "-"
        print(f"  {i:>4} {SOURCE_XS[i]:>7.1f} {str(f['present']):>7} "
              f"{f['T']:>6.2f} {ty:>11} {sg:>11} {f['n']:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", action="store_true",
                    help="plumbing check on a fixed prism instead of searching")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=2500)
    args = ap.parse_args()

    outdir = REPO / "ns_probe" / "results_arvr"
    outdir.mkdir(parents=True, exist_ok=True)

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        if args.hand:
            x = np.array([FD.DEMO.get(n, 0.0) for n in PRISM_NAMES])  # tsag=dsag=0 -> flat top/bottom
            print("[hand] plumbing check (fold_demo prism, flat top/bottom)")
        else:
            print(f"[search] looking for a prism that relays the 3 display points "
                  f"to the eyebox ({args.evals} random + CMA)...")
            x, f = search(runner, n_random=args.evals, seed=args.seed)
            if x is None or f <= 0:
                print("[search] FAILED to find any working relay -- geometry needs recalibration")
                # fall back to plumbing geometry so we still render something
                x = np.array([FD.DEMO[n] for n in PRISM_NAMES])

        res, _ = runner.simulate(scene_arvr(x))
        m = arvr_metrics(res)
        print(f"\n[verify] near-eye relay metrics (display 3 pts -> eyebox @ x={PUPIL_X}, "
              f"|y|<={EYEBOX_HALF}, collimated target):")
        report(m)
        led = res.get("energyLedger") or {}
        print(f"  ledger: emitted={led.get('emitted',0):.1f} delivered={led.get('delivered',0):.2f} "
              f"escaped={led.get('escaped',0):.2f}")
        print(f"  prism: {prism_dict(x)}")

        # render relay diagram (fewer rays, never overwrite)
        rres, _ = runner.simulate(scene_arvr(x, n_fan=7, brightness=0.6,
                                             record=False, crop=True))
        imgs = rres.get("images") or []
        if imgs and imgs[0].get("dataUrl"):
            out = outdir / "arvr_relay.png"
            k = 0
            while out.exists():
                k += 1; out = outdir / f"arvr_relay_{k}.png"
            out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
            out.with_suffix(".json").write_text(json.dumps(
                dict(prism=prism_dict(x), metrics=m,
                     geometry=dict(display_y=DISPLAY_Y, display_cx=DISPLAY_CX,
                                   display_half_w=DISPLAY_HALF_W, source_aim=SOURCE_AIM,
                                   n_fan=N_FAN, fan_halfangle=FAN_HALFANGLE, pupil_x=PUPIL_X,
                                   eyebox_y=EYEBOX_Y, eyebox_half=EYEBOX_HALF,
                                   sigma0=SIGMA0)), indent=2, default=float))
            print(f"\n[render] {out} (+ sidecar)")


if __name__ == "__main__":
    main()
