"""
DEMONSTRATOR (legacy; superseded by fold_demo.py): a single optical element in which the
*dominant energy* undergoes  transmit@A -> TIR@B -> TIR@A -> ... -> transmit@B  (a
loss-free internal fold), simulated by the instrumented NS engine.

CORRECTION (2026-06-12): this file originally claimed "partialReflect is OFF, so every
bounce must be TIR". That was wrong — `Glass` defaults to partialReflect: true (full
Fresnel), so this scene always ran under honest Fresnel physics and the inference was a
lucky one (it held only because the dominant ray's energy, 0.76, is impossible for a
chain of Fresnel back-reflections). The engine now records interaction TYPES directly
(see ns_probe/sequence_tools.py), so the TIR claim is verified per event rather than
inferred; the energy ledger additionally shows input = delivered + escaped + truncated.
For the first-class, bundle-level, sequence-as-input demonstrator use:

    python3 ns_probe/fold_demo.py

Run:
    python3 ns_probe/tir_demo.py            # render + print the interaction sequence
    python3 ns_probe/tir_demo.py --search   # re-discover a geometry from scratch
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

ELEM_H = 30.0
X_SENSOR = 120.0

# A geometry (found by --search) where a single ray TIR-folds several times and exits forward.
DEMO = dict(ray_y=10.9, ang=16.3, cx=44.2, thick=7.3, fsag=-14.2, bsag=-17.5, n=1.99)


def element(cx, thick, fsag, bsag, n):
    """Single high-index element, NO coating, NO partialReflect -> reflections are TIR only."""
    lx, rx = cx - thick / 2, cx + thick / 2
    ty, by = -ELEM_H, ELEM_H
    return {"type": "Glass", "refIndex": float(n),
            "path": [{"x": lx, "y": ty, "arc": False}, {"x": rx, "y": ty, "arc": False},
                     {"x": rx + bsag, "y": 0.0, "arc": True}, {"x": rx, "y": by, "arc": False},
                     {"x": lx, "y": by, "arc": False}, {"x": lx - fsag, "y": 0.0, "arc": True}]}


def scene(p, *, rays=(0.0,), brightness=1.0, record=True, crop=False):
    a = np.radians(p["ang"])
    objs = []
    for dy in rays:
        y = p["ray_y"] + dy
        objs.append({"type": "SingleRay", "p1": {"x": 0.0, "y": float(y)},
                     "p2": {"x": float(np.cos(a)), "y": float(y + np.sin(a))},
                     "brightness": float(brightness)})
    objs.append(element(p["cx"], p["thick"], p["fsag"], p["bsag"], p["n"]))
    objs.append({"type": "Detector", "p1": {"x": X_SENSOR, "y": -80}, "p2": {"x": X_SENSOR, "y": 80}})
    sc = {"version": 5, "maxRayDepth": 16, "objs": objs}
    if record:
        sc["recordInteractions"] = True
    if crop:
        sc["objs"].append({"type": "CropBox", "p1": {"x": -6, "y": -34},
                           "p4": {"x": 128, "y": 34}, "width": 1280})
    return sc


def describe_sequence(seq, cx):
    """Label each vertex. Uses the engine's typed events (vertex field 't') when present;
    falls back to the old geometric inference for pre-instrumentation records."""
    pretty = {"refract": "transmit", "tir": "TIR", "partial_reflect": "partial-reflect"}
    out = []
    for k, v in enumerate(seq):
        x = v["x"]
        if v.get("t") in ("source", None) and k == 0:
            out.append("source")
        elif v.get("t") == "sensor" or x >= X_SENSOR - 1:
            out.append("sensor")
        else:
            face = "A(left)" if x < cx else "B(right)"
            if v.get("t"):
                out.append(f"{pretty.get(v['t'], v['t'])}@{face}")
            else:
                interior = 0 < k < len(seq) - 1 and not (seq[k + 1]["x"] >= X_SENSOR - 1)
                kind = "TIR" if (interior and k > 1) else "transmit"
                out.append(f"{kind}@{face}")
    return " -> ".join(out)


def search(runner, n_trials=4000):
    rng = np.random.default_rng(0)
    best = None
    for _ in range(n_trials):
        p = dict(ray_y=rng.uniform(-18, 18), ang=rng.uniform(-55, 55), cx=rng.uniform(35, 80),
                 thick=rng.uniform(6, 40), fsag=rng.uniform(-18, 18), bsag=rng.uniform(-18, 18),
                 n=rng.uniform(1.6, 2.0))
        try:
            res, _ = runner.simulate(scene(p))
        except Exception:
            continue
        for s in res.get("raySequences", []):
            if s.get("dirx", 0) <= 0 or s["b"] < 0.5:
                continue
            xs = [v["x"] for v in s["seq"]]
            rev = sum(1 for k in range(1, len(xs) - 1) if (xs[k + 1] - xs[k]) * (xs[k] - xs[k - 1]) < 0)
            if rev >= 2 and (best is None or (rev, s["b"]) > (best[0], best[1])):
                best = (rev, s["b"], p, s["seq"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()
    out = REPO / "ns_probe" / "results_single" / "renders" / "TIR_demo_single_element.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        p = DEMO
        if args.search:
            b = search(runner)
            if b:
                p = b[2]
                print(f"[search] best: reversals={b[0]} energy={b[1]:.3f}  geometry={p}")

        # report the dominant detector-reaching ray's interaction sequence
        res, _ = runner.simulate(scene(p))
        fwd = [s for s in res.get("raySequences", []) if s.get("dirx", 0) > 0]
        fwd.sort(key=lambda s: s["b"], reverse=True)
        if fwd:
            s = fwd[0]
            n_tir = sum(1 for k in range(1, len(s["seq"]) - 1)
                        if (s["seq"][k + 1]["x"] - s["seq"][k]["x"]) * (s["seq"][k]["x"] - s["seq"][k - 1]["x"]) < 0)
            print(f"[demo] dominant detector ray: energy={s['b']:.3f} of 1.0  ({n_tir} internal TIR bounces)")
            print("[demo] interaction sequence (typed by the engine recorder):")
            print("       " + describe_sequence(s["seq"], p["cx"]))

        # render a few rays so the bounce path is visible
        rres, _ = runner.simulate(scene(p, rays=(-0.6, 0.0, 0.6), brightness=3.0, record=False, crop=True))
        imgs = rres.get("images") or []
        if imgs and imgs[0].get("dataUrl"):
            out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
            print(f"[demo] wrote {out}")


if __name__ == "__main__":
    main()
