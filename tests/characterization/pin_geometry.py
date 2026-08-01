"""
Characterization pins for the three element geometries that `nslens` will unify.

This is a SAFETY NET, not a unit test suite. It does not assert that the physics is
*correct* -- it asserts that the physics is *unchanged*. Every structural refactor
(Beck: a "tidying") must leave these bit-comparable outputs identical; any diff means
the refactor changed behaviour and must be reverted or re-justified.

What is pinned, per geometry, for a set of fixed genotypes:
  - every sensor-arriving ray  (x, y, direction, brightness)
  - the exact energy ledger    (emitted / delivered / truncated / absorbed)
  - the dominant interaction signature (the typed face/event sequence)

Those three together determine every downstream number in the project (fitness,
descriptors, spot size, throughput, fold depth), so pinning them pins everything.

    record:  python tests/characterization/pin_geometry.py --record
    check:   python tests/characterization/pin_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

GOLDEN = Path(__file__).with_name("golden_geometry.json")

# Round only at the last digits of double precision: tight enough to catch a real
# numerical change, loose enough to survive platform-level libm jitter.
NDIG = 9


def _cases():
    """(name, scene_fn, genotypes) for each geometry family, built lazily so that a
    broken import in one family does not hide the others."""
    import arvr_prism as AV
    import fold_demo as FD
    import prism_fold as PF

    # fold_demo: the baked-in DEMO wedge + the fold1 benchmark reference design.
    fold_gts = [
        [FD.DEMO[k] for k in FD.PARAM_NAMES],
        [-18.65635660138779, 37.483939982719576, 63.993227745392275,
         41.69415522629459, 9.34758117693124, 0.7439142577369253,
         20.4858703051664, -14.223046072754633, 1.6571399294573437],
    ]
    # deterministic interior samples, so the pin covers more than the two champions
    rng = np.random.default_rng(0)
    for _ in range(3):
        fold_gts.append(list(FD.LO + rng.random(len(FD.LO)) * (FD.HI - FD.LO)))

    prism_gts = []
    rng = np.random.default_rng(1)
    for _ in range(3):
        prism_gts.append(list(AV.LO + rng.random(len(AV.LO)) * (AV.HI - AV.LO)))

    pfold_gts = []
    rng = np.random.default_rng(2)
    for _ in range(3):
        pfold_gts.append(list(PF.LO + rng.random(len(PF.LO)) * (PF.HI - PF.LO)))

    from sequence_tools import make_single_element_face_classifier as face_cls

    # NOTE: the three families disagree on their own calling convention --
    # fold_demo.scene/prism_fold.scene take a PARAM DICT, arvr_prism.scene_arvr takes
    # the RAW VECTOR. Normalising that here is exactly the wart nslens removes.
    return [
        ("fold_demo", FD.PARAM_NAMES, FD.vec_to_params, FD.geometry_valid,
         lambda p: face_cls(p["cx"], FD.ELEM_H),
         lambda gt: FD.scene(FD.vec_to_params(gt), n_rays=21, beam_w=8.0, record=True),
         fold_gts),
        ("arvr_prism", AV.PRISM_NAMES, AV.prism_dict, AV.geometry_valid_arvr,
         lambda p: face_cls(p["cx"], AV.ELEM_H),
         lambda gt: AV.scene_arvr(gt, record=True), prism_gts),
        ("prism_fold", PF.PARAM_NAMES, PF.vec_to_params, PF.geometry_valid,
         lambda p: face_cls(p["cx"], PF.ELEM_H),
         lambda gt: PF.scene(PF.vec_to_params(gt), record=True), pfold_gts),
    ]


def _probe(runner, scene_fn, to_params, valid, face_of_fn, gt):
    from sequence_tools import aggregate_by_signature, format_signature

    p = to_params(gt)
    if not valid(p):
        return {"valid": False}

    # A raised exception must NOT be silently pinned as "expected behaviour" -- that
    # would make this file assert nothing. Let it propagate and fail the recording.
    res, _ = runner.simulate(scene_fn(gt))
    if res.get("error"):
        return {"valid": True, "engine_error": str(res["error"])}

    seqs = res.get("raySequences", []) or []
    led = res.get("energyLedger", {}) or {}
    face_of = face_of_fn(p)

    # every sensor-arriving ray, ordered canonically so the pin is stable
    rays = sorted(
        ({"y": round(float(r["y"]), NDIG), "b": round(float(r["b"]), NDIG),
          "dirx": round(float(r.get("dirx", 0.0)), NDIG)}
         for r in seqs if r.get("b", 0) > 0),
        key=lambda d: (d["y"], d["b"], d["dirx"]),
    )

    by_sig = aggregate_by_signature(seqs, face_of)
    dom = max(by_sig, key=by_sig.get) if by_sig else None

    return {
        "valid": True,
        "n_rays": len(rays),
        "rays": rays,
        "ledger": {k: round(float(v), NDIG) for k, v in sorted(led.items())
                   if isinstance(v, (int, float))},
        "dominant_signature": format_signature(dom) if dom else None,
        "dominant_energy": round(float(by_sig[dom]), NDIG) if dom else None,
        "n_signatures": len(by_sig),
    }


def collect():
    cases = _cases()
    out = {}
    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        for name, pnames, to_params, valid, face_of_fn, scene_fn, gts in cases:
            out[name] = {
                "param_names": list(pnames),
                "probes": [{"genotype": [round(float(v), NDIG) for v in gt],
                            "result": _probe(runner, scene_fn, to_params, valid,
                                             face_of_fn, gt)}
                           for gt in gts],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="write the golden file instead of checking against it")
    args = ap.parse_args()

    current = collect()

    if args.record:
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        n = sum(len(v["probes"]) for v in current.values())
        print(f"recorded {n} probes across {len(current)} geometries -> {GOLDEN.name}")
        return 0

    if not GOLDEN.exists():
        print(f"no golden file at {GOLDEN}; run with --record first", file=sys.stderr)
        return 2

    golden = json.loads(GOLDEN.read_text())
    if current == golden:
        n = sum(len(v["probes"]) for v in current.values())
        print(f"PASS  {n} probes across {len(current)} geometries are unchanged")
        return 0

    print("FAIL  behaviour changed vs golden:", file=sys.stderr)
    for geo in sorted(set(golden) | set(current)):
        g, c = golden.get(geo), current.get(geo)
        if g == c:
            continue
        if g is None or c is None:
            print(f"  {geo}: {'added' if g is None else 'REMOVED'}", file=sys.stderr)
            continue
        for i, (gp, cp) in enumerate(zip(g["probes"], c["probes"])):
            if gp != cp:
                gr, cr = gp["result"], cp["result"]
                print(f"  {geo}[{i}]: n_rays {gr.get('n_rays')} -> {cr.get('n_rays')}, "
                      f"ledger {gr.get('ledger')} -> {cr.get('ledger')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
