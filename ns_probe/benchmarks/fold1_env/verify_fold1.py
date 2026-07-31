"""Grader / verifier for the SINGLE-FOLD FOCUSING PRISM benchmark (fold_demo env).

Takes a candidate 9-parameter genotype, traces it through the real engine, and prints a
PASS/FAIL table for every requirement in benchmark_fold1_prompt.md. Exits 0 only if all
required checks pass.

    python3 ns_probe/benchmarks/fold1_env/verify_fold1.py design.json
    python3 ns_probe/benchmarks/fold1_env/verify_fold1.py design.json --render spot.png

design.json = {"ray_y":..,"ang":..,"cx":..,"thick":..,"fsag":..,"bsag":..,
               "tilt_a":..,"tilt_b":..,"n":..}

This script reveals no reference solution: it only measures what the candidate does.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import fold_demo as FD  # noqa: E402
from sequence_tools import (aggregate_by_signature, format_signature,  # noqa: E402
                            make_single_element_face_classifier, signature_of)

ELEM_H = FD.ELEM_H          # 30.0   flat edges at y = +-30 (D = top, T = bottom)
X_SENSOR = FD.X_SENSOR      # 120.0  detector plane
N_RAYS = 21
BEAM_W = 8.0

# ---- acceptance thresholds (commit-then-verify variant) ----------------------------
MIN_TIR = 1                 # at least one genuine total internal reflection
MIN_GLASS_LEFT = 3.0        # mm of air between the source plane (x=0) and the glass
MIN_SHARE = 0.80            # dominant-path energy / emitted energy
MAX_RMS = 1.0               # mm, RMS spot radius on the detector
MAX_DEFOCUS = 10.0          # mm, |best-focus plane - detector plane|
MIN_DEVIATION = 20.0        # deg, total input->output direction change
MIN_RAYS_ON_PATH = N_RAYS - 1


def glass_extent(p):
    xs = [v["x"] for v in FD.element(p)["path"]]
    return min(xs), max(xs)


def rms_at(ys, dirx, diry, x_hit, x_plane, weights):
    """Propagate the exiting rays from their detector crossing to an arbitrary plane."""
    t = (x_plane - x_hit) / dirx
    y = ys + t * diry
    c = np.average(y, weights=weights)
    return float(np.sqrt(np.average((y - c) ** 2, weights=weights))), float(c)


def evaluate(runner, p, *, render=None):
    out = dict(fatal=None)

    in_bounds = all(FD.LO[i] - 1e-9 <= p[k] <= FD.HI[i] + 1e-9
                    for i, k in enumerate(FD.PARAM_NAMES))
    out["in_bounds"] = bool(in_bounds)
    out["nondegenerate"] = bool(FD.geometry_valid(p))
    gl, gr = glass_extent(p)
    out["glass_left"], out["glass_right"] = gl, gr
    out["source_in_air"] = bool(gl >= MIN_GLASS_LEFT)
    out["detector_clear"] = bool(gr < X_SENSOR - 1.0)

    if not out["nondegenerate"]:
        out["fatal"] = "element geometry is degenerate (faces cross)"
        return out

    res, _ = runner.simulate(FD.scene(p, n_rays=N_RAYS, beam_w=BEAM_W, record=True))
    if res.get("error"):
        out["fatal"] = f"engine error: {res['error']}"
        return out

    led = res.get("energyLedger") or {}
    emitted = float(led.get("emitted", 0.0))
    if emitted <= 0:
        out["fatal"] = "no energy emitted"
        return out
    out["ledger"] = {k: float(v) for k, v in led.items()}

    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    by_sig = aggregate_by_signature(res.get("raySequences", []), face_of)
    if not by_sig:
        out["fatal"] = "no ray reached the detector"
        return out

    dom_sig = max(by_sig, key=by_sig.get)
    out["signature"] = format_signature(dom_sig) or "(empty: ray never touched the glass)"
    out["share"] = float(by_sig[dom_sig] / emitted)
    out["tir_count"] = sum(1 for (t, _f) in dom_sig if t == "tir")
    out["n_signatures"] = len(by_sig)
    out["enters_by_refraction"] = bool(dom_sig and dom_sig[0][0] == "refract")
    out["exits_by_refraction"] = bool(dom_sig and dom_sig[-1][0] == "refract")
    out["no_lossy_reflections"] = not any(
        t in ("partial_reflect", "mirror", "beam_split") for (t, _f) in dom_sig)
    out["faces_used"] = sorted({f for (_t, f) in dom_sig})

    recs = [r for r in res.get("raySequences", [])
            if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
            and signature_of(r, face_of) == dom_sig]
    out["rays_on_path"] = len(recs)
    if not recs:
        out["fatal"] = "dominant signature carries no forward rays"
        return out

    ys = np.array([r["y"] for r in recs], float)
    bs = np.array([r["b"] for r in recs], float)
    dx = np.array([r["dirx"] for r in recs], float)
    dy = np.array([r["diry"] for r in recs], float)
    xs = np.array([r["x"] for r in recs], float)

    out["centroid"] = float(np.average(ys, weights=bs))
    out["rms"] = float(np.sqrt(np.average((ys - out["centroid"]) ** 2, weights=bs)))
    out["span"] = float(ys.max() - ys.min())

    # where the beam actually comes to best focus (analytic propagation of exit rays)
    grid = np.arange(X_SENSOR - 60.0, X_SENSOR + 60.01, 0.25)
    curve = [rms_at(ys, dx, dy, xs, xp, bs)[0] for xp in grid]
    k = int(np.argmin(curve))
    out["x_focus"] = float(grid[k])
    out["rms_at_focus"] = float(curve[k])
    out["defocus"] = abs(out["x_focus"] - X_SENSOR)

    # is the fold doing real work? total input -> output direction change
    ang_in = float(p["ang"])
    ang_out = float(np.degrees(np.arctan2(np.average(dy, weights=bs),
                                          np.average(dx, weights=bs))))
    out["ang_in"], out["ang_out"] = ang_in, ang_out
    out["deviation"] = abs(ang_out - ang_in)

    # per-event detail of the strongest ray on the dominant path
    strongest = max(recs, key=lambda r: r["b"])
    rows = []
    for v in strongest["seq"]:
        t = v.get("t")
        if t == "source":
            continue
        rows.append(dict(event=t, mech=v.get("mech") or "",
                         face="-" if t == "sensor" else face_of(v),
                         x=float(v["x"]), y=float(v["y"]),
                         b_in=float(v["bIn"]), b_out=float(v["bOut"])))
    out["events"] = rows

    if render:
        rres, _ = runner.simulate(FD.scene(p, n_rays=9, beam_w=BEAM_W,
                                          ray_brightness=0.8, record=False, crop=True))
        imgs = rres.get("images") or []
        if imgs and imgs[0].get("dataUrl"):
            Path(render).write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
            out["render"] = str(render)
    return out


def checks_of(r):
    """The graded checks as (name, passed, detail) triples. Empty if the run was fatal."""
    if r.get("fatal"):
        return []
    return [
        ("1  parameters inside the declared bounds", r["in_bounds"], ""),
        ("2  element geometry non-degenerate", r["nondegenerate"], ""),
        (f"3  source plane x=0 in air (glass_left >= {MIN_GLASS_LEFT})",
         r["source_in_air"], f"glass_left = {r['glass_left']:.3f}"),
        ("4  detector plane in air, outside the glass", r["detector_clear"], ""),
        (f"5  dominant path has >= {MIN_TIR} genuine TIR", r["tir_count"] >= MIN_TIR,
         f"tir_count = {r['tir_count']}"),
        ("6  path enters glass by refraction", r["enters_by_refraction"], ""),
        ("7  path exits to air by refraction", r["exits_by_refraction"], ""),
        ("8  no partial-mirror / coating reflections on the path",
         r["no_lossy_reflections"], ""),
        (f"9  >= {MIN_RAYS_ON_PATH}/{N_RAYS} rays take the same path",
         r["rays_on_path"] >= MIN_RAYS_ON_PATH, f"{r['rays_on_path']}/{N_RAYS}"),
        (f"10 throughput share >= {MIN_SHARE}", r["share"] >= MIN_SHARE,
         f"share = {r['share']:.4f}"),
        (f"11 RMS spot on the detector <= {MAX_RMS} mm", r["rms"] <= MAX_RMS,
         f"RMS = {r['rms']:.4f} mm"),
        (f"12 focus lands on the detector (defocus <= {MAX_DEFOCUS} mm)",
         r["defocus"] <= MAX_DEFOCUS, f"defocus = {r['defocus']:.2f} mm"),
        (f"13 fold deviates the beam >= {MIN_DEVIATION} deg", r["deviation"] >= MIN_DEVIATION,
         f"deviation = {r['deviation']:.2f} deg"),
    ]


def passes(r):
    ch = checks_of(r)
    return bool(ch) and all(ok for _n, ok, _d in ch)


def report(p, r):
    print("=" * 74)
    print("SINGLE-FOLD FOCUSING PRISM — verification")
    print("=" * 74)
    print("genotype:")
    for k in FD.PARAM_NAMES:
        print(f"    {k:<8} = {p[k]:.6f}")
    if r.get("fatal"):
        print(f"\nFATAL: {r['fatal']}")
        print("\nVERDICT: FAIL")
        return False

    print(f"\ndominant path : {r['signature']}")
    print(f"faces used    : {', '.join(r['faces_used'])}"
          f"   (A=left curved, B=right curved, D=top flat, T=bottom flat)")
    print(f"glass spans x : {r['glass_left']:.3f} .. {r['glass_right']:.3f}")
    print(f"input angle   : {r['ang_in']:+.3f} deg     output angle: {r['ang_out']:+.3f} deg")
    print(f"detector spot : centroid y = {r['centroid']:+.4f}  RMS = {r['rms']:.4f} mm  "
          f"full span = {r['span']:.4f} mm")
    print(f"best focus at : x = {r['x_focus']:.2f}  (RMS {r['rms_at_focus']:.4f} mm)  "
          f"-> defocus {r['defocus']:.2f} mm from the detector at x={X_SENSOR:.0f}")

    print("\nray path of the strongest ray:")
    print(f"    {'event':<16} {'face':<5} {'x':>9} {'y':>9} {'E_in':>9} {'E_out':>9}")
    for e in r["events"]:
        ev = e["event"] + (f" ({e['mech']})" if e["mech"] else "")
        print(f"    {ev:<16} {e['face']:<5} {e['x']:9.3f} {e['y']:9.3f} "
              f"{e['b_in']:9.6f} {e['b_out']:9.6f}")

    checks = checks_of(r)
    print()
    n_pass = 0
    for name, ok, detail in checks:
        n_pass += bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    allok = n_pass == len(checks)
    print(f"\n{n_pass}/{len(checks)} checks passed")
    print(f"VERDICT: {'PASS' if allok else 'FAIL'}")
    if allok:
        print(f"\nquality (for ranking, not pass/fail): RMS {r['rms']:.4f} mm, "
              f"throughput {r['share']:.4f}, defocus {r['defocus']:.2f} mm, "
              f"{r['tir_count']} TIR fold(s)")
    if r.get("render"):
        print(f"render: {r['render']}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", help="JSON file with the 9 genotype parameters")
    ap.add_argument("--render", default=None, help="also write a PNG of the traced design")
    ap.add_argument("--json", default=None, help="dump the full measurement dict here")
    args = ap.parse_args()

    raw = json.loads(Path(args.design).read_text())
    if "genotype" in raw:
        raw = raw["genotype"]
    missing = [k for k in FD.PARAM_NAMES if k not in raw]
    if missing:
        sys.exit(f"design is missing parameters: {missing}")
    p = {k: float(raw[k]) for k in FD.PARAM_NAMES}

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        r = evaluate(runner, p, render=args.render)
    ok = report(p, r)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(genotype=p, measurement=r), indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
