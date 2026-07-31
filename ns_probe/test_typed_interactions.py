"""
Physics validation for the typed-interaction recorder + energy ledger.

Run:  python3 ns_probe/test_typed_interactions.py

Checks (each is a measured assertion, not an eyeball):
  1. TIR events are typed 'tir' and are EXACTLY loss-free (bIn == bOut bitwise).
  2. Uncoated Fresnel partial reflection at near-normal incidence matches
     R = ((n-1)/(n+1))^2 at both the event energies and the spawned child,
     and the child is tagged mech='fresnel'.
  3. The forced-reflectivity knob is tagged mech='forced' and splits energy as set.
  4. The energy ledger closes: emitted == delivered + escaped + absorbed + truncated
     (+ unfinished) to float precision, on a multi-bounce Fresnel scene.
  5. Depth-budget truncation is booked in the ledger and closure still holds.
  6. With recordInteractions off, no instrumentation fields appear and detector
     readings agree with the flagged run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import tir_demo
from sequence_tools import (audit_ray_energy, ledger_closure,
                            make_single_element_face_classifier, signature_of)


def slab_scene(n=1.5, partial=True, reflectivity=None, record=True, max_depth=16):
    """A flat slab (faces at x=40 and x=50), near-normal ray, detectors fore and aft."""
    glass = {"type": "Glass", "refIndex": float(n), "partialReflect": bool(partial),
             "path": [{"x": 40, "y": -20, "arc": False}, {"x": 50, "y": -20, "arc": False},
                      {"x": 50, "y": 20, "arc": False}, {"x": 40, "y": 20, "arc": False}]}
    if reflectivity is not None:
        glass["reflectivity"] = float(reflectivity)
    sc = {"version": 5, "maxRayDepth": max_depth, "objs": [
        {"type": "SingleRay", "p1": {"x": 0, "y": 0.5}, "p2": {"x": 1, "y": 0.501},
         "brightness": 1.0},
        glass,
        {"type": "Detector", "p1": {"x": 100, "y": -40}, "p2": {"x": 100, "y": 40}},
        {"type": "Detector", "p1": {"x": -20, "y": -40}, "p2": {"x": -20, "y": 40}},
    ]}
    if record:
        sc["recordInteractions"] = True
    return sc


def main():
    failures = []

    def check(name, cond, detail=""):
        status = "ok " if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            failures.append(name)

    with PersistentRayOpticsRunner(timeout_s=30) as runner:

        # ---- 1. TIR typed + exactly loss-free, on the known fold geometry ----
        print("[1] TIR typing and loss-free energy")
        res, _ = runner.simulate(tir_demo.scene(tir_demo.DEMO))
        fwd = sorted([s for s in res["raySequences"] if s.get("dirx", 0) > 0],
                     key=lambda s: -s["b"])
        dom = fwd[0]
        tirs = [v for v in dom["seq"] if v.get("t") == "tir"]
        check("dominant ray has TIR events", len(tirs) >= 2, f"{len(tirs)} TIRs")
        check("every TIR is exactly loss-free",
              all(v["bIn"] == v["bOut"] for v in tirs))
        check("per-ray energy audit clean", audit_ray_energy(dom) == [],
              "; ".join(audit_ray_energy(dom)))
        face_of = make_single_element_face_classifier(tir_demo.DEMO["cx"], tir_demo.ELEM_H)
        sig = signature_of(dom, face_of)
        check("typed signature is T,TIR...,T", sig[0][0] == "refract"
              and sig[-1][0] == "refract" and all(t == "tir" for t, _ in sig[1:-1]),
              " ".join(f"{t}@{f}" for t, f in sig))

        # ---- 2. Fresnel partial reflection at near-normal incidence ----
        print("[2] Fresnel partial reflection (uncoated, physical)")
        n = 1.5
        R = ((n - 1) / (n + 1)) ** 2  # 0.04
        res, _ = runner.simulate(slab_scene(n=n))
        recs = res["raySequences"]
        main_fwd = max([r for r in recs if r["dirx"] > 0], key=lambda r: r["b"])
        entry = next(v for v in main_fwd["seq"] if v.get("t") == "refract")
        check("entry transmission = 1-R", abs(entry["bOut"] / entry["bIn"] - (1 - R)) < 1e-4,
              f"T={entry['bOut']/entry['bIn']:.5f} vs {1-R:.5f}")
        check("through-slab energy = (1-R)^2",
              abs(main_fwd["b"] - (1 - R) ** 2) < 1e-4, f"b={main_fwd['b']:.5f}")
        back = [r for r in recs if r["dirx"] < 0]
        first_bounce = [r for r in back
                        if signature_of(r, lambda v: "S")
                        and signature_of(r, lambda v: "S")[0][0] == "partial_reflect"]
        check("Fresnel child exists, energy = R",
              any(abs(r["b"] - R) < 1e-4 for r in first_bounce),
              f"back energies: {[round(r['b'],5) for r in back]}")
        pr = next(v for v in first_bounce[0]["seq"] if v.get("t") == "partial_reflect")
        check("Fresnel child tagged mech='fresnel'", pr.get("mech") == "fresnel")
        check("ledger closes (Fresnel slab)", abs(ledger_closure(res["energyLedger"])) < 1e-9,
              f"residual={ledger_closure(res['energyLedger']):.2e}")

        # ---- 3. Forced reflectivity knob ----
        print("[3] Forced-reflectivity coating (the artificial knob)")
        res, _ = runner.simulate(slab_scene(n=1.5, reflectivity=0.3))
        recs = res["raySequences"]
        back = [r for r in recs if r["dirx"] < 0]
        forced = [v for r in back for v in r["seq"] if v.get("t") == "partial_reflect"]
        check("forced child tagged mech='forced'",
              all(v.get("mech") == "forced" for v in forced) and forced != [])
        check("forced child energy = r", any(abs(r["b"] - 0.3) < 1e-6 for r in back),
              f"back energies: {[round(r['b'],5) for r in back]}")
        main_fwd = max([r for r in recs if r["dirx"] > 0], key=lambda r: r["b"])
        check("forced through-energy = (1-r)^2", abs(main_fwd["b"] - 0.49) < 1e-6,
              f"b={main_fwd['b']:.5f}")
        check("ledger closes (forced)", abs(ledger_closure(res["energyLedger"])) < 1e-9)

        # ---- 4. Ledger closure on the multi-bounce fold scene ----
        print("[4] Ledger closure, multi-bounce Fresnel fold")
        res, _ = runner.simulate(tir_demo.scene(tir_demo.DEMO))
        led = res["energyLedger"]
        check("emitted == 1", abs(led["emitted"] - 1.0) < 1e-12)
        check("ledger closes", abs(ledger_closure(led)) < 1e-9,
              f"residual={ledger_closure(led):.2e}")
        check("delivered matches sensor-record sum",
              abs(led["delivered"] - sum(r["b"] for r in res["raySequences"])) < 1e-9)

        # ---- 5. Depth-budget truncation is booked ----
        print("[5] Depth-budget truncation accounting")
        sc = tir_demo.scene(tir_demo.DEMO)
        sc["maxRayDepth"] = 3
        res, _ = runner.simulate(sc)
        led = res["energyLedger"]
        check("truncation books the killed fold", led["truncated"] > 0.5,
              f"truncated={led['truncated']:.4f}")
        check("ledger still closes", abs(ledger_closure(led)) < 1e-9)

        # ---- 6. Flag off => no instrumentation, same physics ----
        print("[6] Default path unchanged when flag is off")
        res_on, _ = runner.simulate(slab_scene(n=1.5, partial=False, record=True))
        res_off, _ = runner.simulate(slab_scene(n=1.5, partial=False, record=False))
        check("no raySequences when off", "raySequences" not in res_off)
        check("no energyLedger when off", "energyLedger" not in res_off)
        check("identical detector power (TIR-only scene)",
              res_on["detectors"][0]["power"] == res_off["detectors"][0]["power"],
              f"{res_on['detectors'][0]['power']} vs {res_off['detectors'][0]['power']}")
        res_on_f, _ = runner.simulate(slab_scene(n=1.5, partial=True, record=True))
        res_off_f, _ = runner.simulate(slab_scene(n=1.5, partial=True, record=False))
        dp = abs(res_on_f["detectors"][0]["power"] - res_off_f["detectors"][0]["power"])
        check("Fresnel power agrees within ghost-resample tolerance", dp < 0.02,
              f"|diff|={dp:.2e} (flagged mode disables sub-threshold ghost resurrection)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
