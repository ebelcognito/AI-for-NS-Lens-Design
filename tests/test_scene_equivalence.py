"""nslens.Layout must reproduce the legacy scene builders exactly.

Element equivalence (test_element_equivalence.py) proves the glass is identical. This
proves the whole SCENE is identical -- every ray, the detector, the depth cap and the
record flag -- for all three scenarios:

    fold_demo.scene       collinear: parallel bundle from x=0, detector at x=120
    prism_fold.scene      perpendicular: parallel bundle from y=-75, detector at x=170
    arvr_prism.scene_arvr AR/VR: point fans from the display, detector at the pupil

If this passes, a Layout can be swapped in anywhere a legacy scene builder is called
without moving a single number.

    python tests/test_scene_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import arvr_prism as AV                                   # noqa: E402
import fold_demo as FD                                    # noqa: E402
import prism_fold as PF                                   # noqa: E402

from nslens.geometry import Element                       # noqa: E402
from nslens.scene import Detector, Layout, ParallelBundle, PointFan   # noqa: E402

N_SAMPLES = 500


def _diff_scene(a: dict, b: dict) -> str | None:
    """None if identical, else a short human description of the first difference."""
    for key in ("version", "maxRayDepth", "recordInteractions"):
        if a.get(key) != b.get(key):
            return f"{key}: {a.get(key)!r} != {b.get(key)!r}"
    oa, ob = a["objs"], b["objs"]
    if len(oa) != len(ob):
        return f"object count {len(oa)} != {len(ob)}"
    for i, (x, y) in enumerate(zip(oa, ob)):
        if x.get("type") != y.get("type"):
            return f"objs[{i}].type {x.get('type')} != {y.get('type')}"
        if x != y:
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    return f"objs[{i}] ({x.get('type')}) field {k!r}: {x.get(k)!r} != {y.get(k)!r}"
            return f"objs[{i}] differs"
    return None


def _run(label, sampler, legacy_fn, layout_fn, n=N_SAMPLES) -> bool:
    rng = np.random.default_rng(7)
    for i in range(n):
        gt = sampler(rng)
        legacy = legacy_fn(gt)
        ours = layout_fn(gt).to_scene_dict()
        d = _diff_scene(legacy, ours)
        if d is not None:
            print(f"  FAIL {label}: sample {i} -> {d}")
            return False
    print(f"  PASS {label}: {n} scenes, exact match")
    return True


def check_fold_demo() -> bool:
    def sampler(rng):
        return FD.LO + rng.random(len(FD.LO)) * (FD.HI - FD.LO)

    def legacy(gt):
        return FD.scene(FD.vec_to_params(gt), n_rays=21, beam_w=8.0, record=True)

    def layout(gt):
        p = FD.vec_to_params(gt)
        return Layout(
            source=ParallelBundle(center=p["ray_y"], aim_deg=p["ang"],
                                  width=8.0, n_rays=21, axis="y", origin=0.0),
            element=Element.from_legacy_params(p, half_h=FD.ELEM_H),
            detector=Detector.vertical(FD.X_SENSOR),
        )
    return _run("fold_demo.scene", sampler, legacy, layout)


def check_prism_fold() -> bool:
    def sampler(rng):
        return PF.LO + rng.random(len(PF.LO)) * (PF.HI - PF.LO)

    def legacy(gt):
        return PF.scene(PF.vec_to_params(gt), record=True)

    def layout(gt):
        p = PF.vec_to_params(gt)
        return Layout(
            source=ParallelBundle(center=p["ray_x"], aim_deg=90.0,
                                  width=PF.BEAM_W, n_rays=PF.N_RAYS,
                                  axis="x", origin=PF.Y_SOURCE),
            element=Element.from_legacy_params(p, half_h=PF.ELEM_H),
            detector=Detector.vertical(PF.X_SENSOR),
        )
    return _run("prism_fold.scene", sampler, legacy, layout)


def check_arvr() -> bool:
    def sampler(rng):
        return AV.LO + rng.random(len(AV.LO)) * (AV.HI - AV.LO)

    def legacy(gt):
        return AV.scene_arvr(gt, record=True)

    def layout(gt):
        p = AV.prism_dict(gt)
        return Layout(
            source=PointFan(points=AV.SOURCE_PTS, aim_deg=AV.SOURCE_AIM,
                            halfangle_deg=AV.FAN_HALFANGLE, n_fan=AV.N_FAN),
            element=Element.from_legacy_params(p, half_h=AV.ELEM_H),
            detector=Detector.vertical(AV.PUPIL_X),
        )
    return _run("arvr_prism.scene_arvr", sampler, legacy, layout)


def check_source_reuse() -> bool:
    """The claim behind the refactor: the collinear and perpendicular envs use the SAME
    source type, differing only in axis and origin. Verified, not asserted."""
    a = ParallelBundle(center=0.0, aim_deg=0.0, axis="y", origin=0.0)
    b = ParallelBundle(center=0.0, aim_deg=90.0, axis="x", origin=-75.0)
    if type(a) is not type(b):
        print("  FAIL source reuse: different types")
        return False
    if len(a.rays()) != len(b.rays()):
        print("  FAIL source reuse: different ray counts")
        return False
    print(f"  PASS source reuse: collinear and perpendicular envs share ParallelBundle "
          f"({len(a.rays())} rays each)")
    return True


def main() -> int:
    print("nslens.Layout vs legacy scene builders")
    results = [check_fold_demo(), check_prism_fold(), check_arvr(),
               check_source_reuse()]
    if all(results):
        print("ALL SCENE EQUIVALENCE CHECKS PASS")
        return 0
    print("SCENE EQUIVALENCE FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
