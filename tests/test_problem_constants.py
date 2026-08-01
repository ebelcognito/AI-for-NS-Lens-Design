"""nslens Problems restate the legacy module constants. Assert they still agree.

nslens/qd/problems.py deliberately hardcodes the genome box and the source/detector
placement instead of importing fold_demo/prism_fold, so that nslens does not depend on
the legacy tree. That is only safe if the two stay in sync -- this test is the tripwire.

Fast: no engine, no tracing.

    python tests/test_problem_constants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import fold_demo as FD                                  # noqa: E402
import prism_fold as PF                                 # noqa: E402

from nslens.qd.problems import FoldProblem, PrismFoldProblem   # noqa: E402


def _check(label, pairs) -> bool:
    bad = [(n, a, b) for n, a, b in pairs
           if not (np.array_equal(np.asarray(a), np.asarray(b)))]
    if bad:
        print(f"  FAIL {label}")
        for n, a, b in bad:
            print(f"        {n}: legacy={a!r} nslens={b!r}")
        return False
    print(f"  PASS {label}: {len(pairs)} constants agree")
    return True


def main() -> int:
    print("nslens Problem constants vs legacy modules")
    fold = FoldProblem()
    prism = PrismFoldProblem()

    ok = [
        _check("FoldProblem vs fold_demo", [
            ("param_names", FD.PARAM_NAMES, list(fold.param_names)),
            ("LO", FD.LO, fold.lo),
            ("HI", FD.HI, fold.hi),
            ("ELEM_H", FD.ELEM_H, fold.half_h),
            ("X_SENSOR", FD.X_SENSOR, fold.x_sensor),
        ]),
        _check("PrismFoldProblem vs prism_fold", [
            ("param_names", PF.PARAM_NAMES, list(prism.param_names)),
            ("LO", PF.LO, prism.lo),
            ("HI", PF.HI, prism.hi),
            ("ELEM_H", PF.ELEM_H, prism.half_h),
            ("X_SENSOR", PF.X_SENSOR, prism.x_sensor),
            ("Y_SOURCE", PF.Y_SOURCE, prism.y_source),
            ("DET_GAP", PF.DET_GAP, prism.det_gap),
            ("FOCUS_W", PF.FOCUS_W, prism.focus_w),
            ("N_RAYS", PF.N_RAYS, prism.n_rays),
            ("BEAM_W", PF.BEAM_W, prism.beam_w),
        ]),
    ]
    if all(ok):
        print("ALL CONSTANTS AGREE")
        return 0
    print("CONSTANTS DIVERGED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
