"""
DEMONSTRATOR: a single optical element in which the DOMINANT energy of an input BEAM
undergoes a REQUESTED ray-surface interaction sequence — by default the loss-free
internal double bounce

    transmit@A -> TIR@B -> TIR@A -> transmit@B        ("T@A TIR@B TIR@A T@B")

simulated under full, honest Fresnel physics (uncoated surfaces, partialReflect ON:
every surface event splits energy by the Fresnel equations; the only loss-free
reflection available is true geometric TIR).

Unlike the older tir_demo.py (single lucky ray, sequence inferred from geometry), this
demonstrator:
  * traces a finite-width BEAM (default 21 rays, total input energy 1.0),
  * takes the target sequence as an INPUT (--sequence), not a hardcoded outcome,
  * uses the engine's typed interaction events (refract / tir / partial_reflect / sensor,
    with per-event energies) so the sequence claim is verified, not inferred,
  * audits the full energy ledger: input = delivered + reflected-away + absorbed +
    truncated, printed every run,
  * declares PASS only if the energy delivered via EXACTLY the target sequence exceeds
    50% of the input AND every competing sequence class.

Run:
    python3 ns_probe/fold_demo.py                  # verify + render the baked-in design
    python3 ns_probe/fold_demo.py --search         # re-discover a geometry from scratch
    python3 ns_probe/fold_demo.py --sequence "T@A TIR@B TIR@A TIR@B TIR@A T@B" --search
                                                   # any other target sequence
    python3 ns_probe/fold_demo.py --coating 0.5 --sequence "T@A PR@B PR@A T@B" --search
                                                   # the lossy partial-mirror route, for
                                                   # contrast (geometric max: (1-r)^2 r^2)
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
from cma_me import CMAES
from sequence_tools import (dominance_report, format_signature,
                            make_single_element_face_classifier, matches,
                            parse_target, print_report, signature_of)

ELEM_H = 30.0
X_SENSOR = 120.0

# Genotype: [ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n]
PARAM_NAMES = ["ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]
LO = np.array([-20.0, -55.0, 35.0, 6.0, -18.0, -18.0, -25.0, -25.0, 1.50])
HI = np.array([20.0, 55.0, 85.0, 45.0, 18.0, 18.0, 25.0, 25.0, 2.00])

# Found by `--search --seed 0` (random stage + CMA-ES polish) for the default target
# "T@A TIR@B TIR@A T@B"; baked in for reproducibility. Re-derive with --search.
# A wedge-like element (tilt_b ~ -20deg): the prismatic tilt is what lets the WHOLE
# bundle exceed the critical angle at B and A, then drop below it at the second B hit.
DEMO = {"ray_y": -19.690363547321468, "ang": -0.4386787509197916,
        "cx": 57.369992451615424, "thick": 8.45096749613329,
        "fsag": 10.315268741694716, "bsag": 8.746257001118732,
        "tilt_a": -7.95840713180154, "tilt_b": -20.238568440396122,
        "n": 1.7076175934459101}


def element(p, coating=None):
    """Single glass element: two (tiltable, curved) optical faces A=left, B=right,
    flat top/bottom edges. Uncoated => full Fresnel + TIR physics."""
    lx, rx = p["cx"] - p["thick"] / 2, p["cx"] + p["thick"] / 2
    dA = np.tan(np.radians(p["tilt_a"])) * ELEM_H
    dB = np.tan(np.radians(p["tilt_b"])) * ELEM_H
    obj = {"type": "Glass", "refIndex": float(p["n"]), "path": [
        {"x": lx + dA, "y": -ELEM_H, "arc": False},
        {"x": rx + dB, "y": -ELEM_H, "arc": False},
        {"x": rx + p["bsag"], "y": 0.0, "arc": True},
        {"x": rx - dB, "y": ELEM_H, "arc": False},
        {"x": lx - dA, "y": ELEM_H, "arc": False},
        {"x": lx - p["fsag"], "y": 0.0, "arc": True},
    ]}
    if coating is not None:
        obj["reflectivity"] = float(coating)  # forced partial mirror (the lossy route)
    return obj


def geometry_valid(p):
    lx, rx = p["cx"] - p["thick"] / 2, p["cx"] + p["thick"] / 2
    dA = np.tan(np.radians(p["tilt_a"])) * ELEM_H
    dB = np.tan(np.radians(p["tilt_b"])) * ELEM_H
    return (rx + dB) - (lx + dA) > 1.0 and (rx - dB) - (lx - dA) > 1.0


def scene(p, *, n_rays=21, beam_w=8.0, ray_brightness=1.0, record=True, crop=False,
          coating=None, max_depth=16):
    # Each ray is emitted at brightness 1.0 (input = n_rays) so that first-order Fresnel
    # ghosts (~4-15% of a ray) stay above the engine's absolute truncation threshold
    # (0.01) and are actually traced, not just booked as truncated. All reported shares
    # are normalized by the ledger's emitted total, so the input scale cancels.
    a = np.radians(p["ang"])
    objs = []
    for dy in np.linspace(-beam_w / 2, beam_w / 2, n_rays):
        y = p["ray_y"] + dy
        objs.append({"type": "SingleRay", "p1": {"x": 0.0, "y": float(y)},
                     "p2": {"x": float(np.cos(a)), "y": float(y + np.sin(a))},
                     "brightness": float(ray_brightness)})
    objs.append(element(p, coating=coating))
    objs.append({"type": "Detector", "p1": {"x": X_SENSOR, "y": -120},
                 "p2": {"x": X_SENSOR, "y": 120}})
    sc = {"version": 5, "maxRayDepth": max_depth, "objs": objs}
    if record:
        sc["recordInteractions"] = True
    if crop:
        objs.append({"type": "CropBox", "p1": {"x": -6, "y": -40},
                     "p4": {"x": 130, "y": 40}, "width": 1360})
    return sc


def vec_to_params(x):
    return dict(zip(PARAM_NAMES, [float(v) for v in x]))


def evaluate(runner, x, target, *, n_rays, beam_w, coating):
    """Objective: fraction of the INPUT energy delivered via exactly the target
    sequence. Invalid/erroring geometries rank below everything."""
    p = vec_to_params(x)
    if not geometry_valid(p):
        return -1.0, None
    try:
        res, _ = runner.simulate(scene(p, n_rays=n_rays, beam_w=beam_w, coating=coating))
    except Exception:
        return -1.0, None
    if res.get("error"):
        return -1.0, None
    face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
    rep = dominance_report(res, face_of, target)
    return rep["target_share_of_input"], rep


def search(runner, target, *, n_rays, beam_w, coating, n_random=2500, cma_gens=50,
           seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, -np.inf

    for k in range(n_random):
        x = rng.uniform(LO, HI)
        f, _ = evaluate(runner, x, target, n_rays=n_rays, beam_w=beam_w, coating=coating)
        if f > best_f:
            best_f, best_x = f, x
            if verbose:
                print(f"  [random {k}] target-share={f:.3f}  {vec_to_params(x)}")

    if best_x is None or best_f <= 0:
        return best_x, best_f

    # CMA-ES polish in normalized [0,1] coordinates
    x0 = (best_x - LO) / (HI - LO)
    es = CMAES(x0, 0.06, popsize=12, seed=seed,
               bounds=(np.zeros(len(LO)), np.ones(len(LO))))
    for g in range(cma_gens):
        X = es.ask()
        F = []
        for xn in X:
            f, _ = evaluate(runner, LO + xn * (HI - LO), target,
                            n_rays=n_rays, beam_w=beam_w, coating=coating)
            F.append(f)
        F = np.asarray(F)
        es.tell(np.argsort(-F))
        gi = int(np.argmax(F))
        if F[gi] > best_f:
            best_f = F[gi]
            best_x = LO + X[gi] * (HI - LO)
            if verbose:
                print(f"  [cma gen {g}] target-share={best_f:.3f}")
        if es.restart_needed:
            break
    return best_x, best_f


def event_table(record, face_of):
    rows = ["    step  type             face   x        y        E_in      E_out"]
    step = 0
    for v in record["seq"]:
        t = v.get("t")
        if t == "source":
            rows.append(f"    {step:>4}  source           -      "
                        f"{v['x']:<8.2f} {v['y']:<8.2f} -         -")
        else:
            face = "-" if t == "sensor" else face_of(v)
            mech = f" ({v['mech']})" if v.get("mech") else ""
            rows.append(f"    {step:>4}  {t + mech:<16} {face:<6} "
                        f"{v['x']:<8.2f} {v['y']:<8.2f} {v['bIn']:<9.6f} {v['bOut']:<9.6f}")
        step += 1
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="T@A TIR@B TIR@A T@B",
                    help="target interaction sequence, e.g. 'T@A TIR@B TIR@A T@B'")
    ap.add_argument("--search", action="store_true", help="re-discover the geometry")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rays", type=int, default=21)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--coating", type=float, default=None,
                    help="forced uniform reflectivity (the lossy partial-mirror route)")
    ap.add_argument("--evals", type=int, default=2500, help="random-stage evaluations")
    args = ap.parse_args()

    target = parse_target(args.sequence)
    suffix = "_coating" if args.coating is not None else ""
    out = REPO / "ns_probe" / "results_single" / "renders" / f"fold_demo{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # NEVER overwrite an existing render: a found design may be one of a kind. If the
    # name is taken, write fold_demo_1.png, _2.png, ... and a geometry sidecar JSON so
    # any render can be reproduced exactly even if the PNG is lost.
    k = 0
    while out.exists():
        k += 1
        out = out.with_name(f"fold_demo{suffix}_{k}.png")

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        if args.search or DEMO is None:
            print(f"[search] target sequence: {format_signature(target)}")
            x, f = search(runner, target, n_rays=args.rays, beam_w=args.beam_w,
                          coating=args.coating, n_random=args.evals, seed=args.seed)
            if x is None or f <= 0:
                print("[search] FAILED to find any geometry realizing the target sequence")
                sys.exit(1)
            p = vec_to_params(x)
            print(f"[search] best target-share={f:.3f}")
            print(f"[search] geometry: {p}")
        else:
            p = dict(DEMO)

        res, _ = runner.simulate(scene(p, n_rays=args.rays, beam_w=args.beam_w,
                                       coating=args.coating))
        face_of = make_single_element_face_classifier(p["cx"], ELEM_H)
        rep = dominance_report(res, face_of, target)

        print()
        print(f"[verify] requested sequence : {format_signature(target)}")
        print(f"[verify] beam: {args.rays} rays, width {args.beam_w}, total input 1.0, "
              f"physics: {'forced coating r=' + str(args.coating) if args.coating is not None else 'uncoated Fresnel + TIR only'}")
        print_report(rep)

        # Per-event energy audit of the strongest target-matching ray
        target_recs = [r for r in res["raySequences"] if r.get("dirx", 0) > 0
                       and signature_of(r, face_of) is not None
                       and matches(signature_of(r, face_of), target)]
        if target_recs:
            strongest = max(target_recs, key=lambda r: r["b"])
            print(f"\n[verify] strongest target ray (energy {strongest['b']:.6f} "
                  f"of 1.0 emitted for this ray):")
            print(event_table(strongest, face_of))

        # Render
        rres, _ = runner.simulate(scene(p, n_rays=9, beam_w=args.beam_w,
                                        ray_brightness=0.8, record=False, crop=True,
                                        coating=args.coating))
        imgs = rres.get("images") or []
        if imgs and imgs[0].get("dataUrl"):
            out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
            sidecar = out.with_suffix(".json")
            sidecar.write_text(json.dumps({
                "geometry": p, "sequence": args.sequence, "coating": args.coating,
                "rays": args.rays, "beam_w": args.beam_w, "seed": args.seed,
                "target_share_of_input": rep["target_share_of_input"],
                "ledger": rep["ledger"],
            }, indent=2))
            print(f"\n[render] wrote {out}  (+ geometry sidecar {sidecar.name})")

        if not rep["dominant"]:
            sys.exit(2)


if __name__ == "__main__":
    main()
