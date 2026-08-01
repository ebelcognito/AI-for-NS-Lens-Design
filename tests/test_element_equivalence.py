"""nslens.Element must be a DROP-IN replacement for the legacy element builders.

This is the gate for phase 1. The unified Element is only allowed to exist if it
reproduces fold_demo.element and arvr_prism.prism_element *exactly* -- identical vertex
count, identical vertex order, identical floats, identical arc flags -- across the whole
sampled genotype box, not just at the champions.

Anything less would silently move every archive, benchmark and CODE V export.

    python tests/test_element_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import arvr_prism as AV                      # noqa: E402
import fold_demo as FD                       # noqa: E402
import prism_fold as PF                      # noqa: E402

from nslens.geometry import Element          # noqa: E402

N_SAMPLES = 2000


def _same_scene(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    if a["type"] != b["type"] or float(a["refIndex"]) != float(b["refIndex"]):
        return False
    if a.get("reflectivity") != b.get("reflectivity"):
        return False
    pa, pb = a["path"], b["path"]
    if len(pa) != len(pb):
        return False
    for va, vb in zip(pa, pb):
        # exact float equality: the unified builder must not merely be close
        if float(va["x"]) != float(vb["x"]) or float(va["y"]) != float(vb["y"]):
            return False
        if bool(va["arc"]) != bool(vb["arc"]):
            return False
    return True


def _report(label: str, fails: list, checked: int) -> bool:
    if fails:
        print(f"  FAIL {label}: {len(fails)}/{checked} mismatched")
        gt, legacy, ours = fails[0]
        print(f"    first mismatch at genotype {np.round(gt, 6).tolist()}")
        print(f"    legacy path ({len(legacy['path'])} verts): "
              f"{[(round(v['x'], 6), round(v['y'], 6), v['arc']) for v in legacy['path']]}")
        print(f"    nslens path ({len(ours['path'])} verts): "
              f"{[(round(v['x'], 6), round(v['y'], 6), v['arc']) for v in ours['path']]}")
        return False
    print(f"  PASS {label}: {checked} genotypes, exact match")
    return True


def check_fold_demo() -> bool:
    rng = np.random.default_rng(0)
    fails, checked = [], 0
    for _ in range(N_SAMPLES):
        gt = FD.LO + rng.random(len(FD.LO)) * (FD.HI - FD.LO)
        p = FD.vec_to_params(gt)
        legacy = FD.element(p)
        ours = Element.from_legacy_params(p, half_h=FD.ELEM_H).to_scene_dict()
        checked += 1
        if not _same_scene(legacy, ours):
            fails.append((gt, legacy, ours))
    ok = _report("fold_demo.element", fails, checked)

    # the baked-in DEMO champion, and the coating (partial-mirror) variant
    legacy = FD.element(FD.DEMO, coating=0.5)
    ours = Element.from_legacy_params(FD.DEMO, half_h=FD.ELEM_H,
                                      reflectivity=0.5).to_scene_dict()
    if not _same_scene(legacy, ours):
        print("  FAIL fold_demo.element: DEMO champion with coating=0.5 mismatched")
        ok = False
    else:
        print("  PASS fold_demo.element: DEMO champion + coating=0.5")
    return ok


def check_arvr_prism() -> bool:
    rng = np.random.default_rng(1)
    fails, checked = [], 0
    for _ in range(N_SAMPLES):
        gt = AV.LO + rng.random(len(AV.LO)) * (AV.HI - AV.LO)
        p = AV.prism_dict(gt)
        legacy = AV.prism_element(p)
        ours = Element.from_legacy_params(p, half_h=AV.ELEM_H).to_scene_dict()
        checked += 1
        if not _same_scene(legacy, ours):
            fails.append((gt, legacy, ours))
    return _report("arvr_prism.prism_element", fails, checked)


def check_validity() -> bool:
    """The validity predicate must reject exactly the same genotypes as before -- a
    looser one would let degenerate geometry into archives, a tighter one would silently
    shrink the search space."""
    rng = np.random.default_rng(2)
    mism, checked, rejected = 0, 0, 0
    for _ in range(N_SAMPLES):
        gt = FD.LO + rng.random(len(FD.LO)) * (FD.HI - FD.LO)
        p = FD.vec_to_params(gt)
        legacy = bool(FD.geometry_valid(p))
        ours = Element.from_legacy_params(p, half_h=FD.ELEM_H).is_valid()
        checked += 1
        rejected += (not legacy)
        mism += (legacy != ours)
    if mism:
        print(f"  FAIL geometry_valid: {mism}/{checked} disagree")
        return False
    print(f"  PASS geometry_valid: {checked} genotypes agree "
          f"({rejected} rejected by both)")
    return True


def check_x_extent() -> bool:
    """prism_fold.element_x_extent drives the detector-clearance check, so it has to
    agree too -- note it measures the boundary vertices, control points included."""
    rng = np.random.default_rng(3)
    mism, checked = 0, 0
    for _ in range(N_SAMPLES):
        gt = PF.LO + rng.random(len(PF.LO)) * (PF.HI - PF.LO)
        p = PF.vec_to_params(gt)
        legacy = PF.element_x_extent(p)
        ours = Element.from_legacy_params(p, half_h=PF.ELEM_H).x_extent()
        checked += 1
        if float(legacy[0]) != float(ours[0]) or float(legacy[1]) != float(ours[1]):
            mism += 1
    if mism:
        print(f"  FAIL element_x_extent: {mism}/{checked} disagree")
        return False
    print(f"  PASS element_x_extent: {checked} genotypes agree")
    return True


def check_configuration_story() -> bool:
    """The claim that motivates the refactor: fold_demo and arvr_prism differ ONLY in
    which faces are curved. Verified structurally, not by eye."""
    fold = Element.from_legacy_params(FD.DEMO, half_h=FD.ELEM_H)
    curved_fold = {n for n in fold.faces if fold.faces[n].is_curved}
    prism_p = AV.prism_dict(AV.LO + 0.5 * (AV.HI - AV.LO))
    prism = Element.from_legacy_params(prism_p, half_h=AV.ELEM_H)
    curved_prism = {n for n in prism.faces if prism.faces[n].is_curved}

    ok = True
    if curved_fold != {"left", "right"}:
        print(f"  FAIL fold_demo should curve left/right, got {sorted(curved_fold)}")
        ok = False
    if curved_prism != {"left", "right", "bottom", "top"}:
        print(f"  FAIL arvr_prism should curve all four, got {sorted(curved_prism)}")
        ok = False
    if len(fold.path()) != 6 or len(prism.path()) != 8:
        print(f"  FAIL vertex counts: fold={len(fold.path())} (want 6), "
              f"prism={len(prism.path())} (want 8)")
        ok = False
    if ok:
        print("  PASS configuration story: fold=left/right curved (6 verts), "
              "prism=all four (8 verts) -- same class")
    return ok


def main() -> int:
    print("nslens.Element vs legacy element builders")
    results = [check_fold_demo(), check_arvr_prism(), check_validity(),
               check_x_extent(), check_configuration_story()]
    if all(results):
        print("ALL EQUIVALENCE CHECKS PASS")
        return 0
    print("EQUIVALENCE FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
