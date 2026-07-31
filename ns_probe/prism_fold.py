"""
PRISM-FOLD geometry: the perpendicular sibling of the collinear fold (fold_demo.py).

SAME measurement model as the fold env — a PARALLEL RAY BUNDLE in, a FOCUSED SPOT out,
fitness = focused throughput, descriptor = (TIR-bounce count, thinness) — but the SOURCE is
BELOW (emitting straight UP) and the DETECTOR is PERPENDICULAR to it (on the right, +x axis).
A straight transmission path is therefore geometrically impossible: to reach the detector the
bundle must be TIR-folded ~90 deg. This is the opposite regime from the collinear fold, where
folding is optional.

Because a 90 deg turn needs a ~45 deg reflecting surface, the element is the AR-style prism
(arvr_prism.prism_element: 4 curved faces, +-50 deg tiltable left/right faces) rather than the
+-25 deg fold_demo element, which cannot fold 90 deg. The source is a parallel bundle (NOT the
AR point-source fans); only the *geometry* is perpendicular.

Genotype (10-DOF): [ray_x, cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n]
    ray_x       : x-center of the upward parallel bundle (the source's transverse position)
    cx..n       : the 9-DOF AR-style prism element (arvr_prism.PRISM_NAMES)
The bundle is aimed straight up (parallel); its transverse position is searched like the fold
env searches ray_y.
"""
from __future__ import annotations

import numpy as np

import arvr_prism as AV
import fold_demo as FD

ELEM_H = AV.ELEM_H          # 30
# The element can sprawl far in +x: a +-50 deg face tilt shifts a corner by tan(50)*ELEM_H ~= 36,
# so over the full sample box the element's rightmost vertex reaches ~143 (measured). The detector
# MUST sit in free space beyond ANY sampleable element -- otherwise the detector plane cuts through
# the glass. So X_SENSOR is placed past that max extent + a margin, and geometry_valid() hard-rejects
# any element whose rightmost point comes within DET_GAP of the detector (guarantees an air gap).
DET_GAP = 12.0             # min clearance between element rightmost point and the detector plane
X_SENSOR = 170.0           # detector on the RIGHT (perpendicular), OUTSIDE the element sample region
Y_SOURCE = -75.0           # parallel bundle emitted from BELOW, traveling +y (up)
FOCUS_W = 8.0              # half-width of the focus window (matches run_fold_qd)
N_RAYS = 21
BEAM_W = 8.0

# [ray_x] + AR element (cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n)
PARAM_NAMES = ["ray_x"] + list(AV.PRISM_NAMES)
_SAG, _TILT = AV.SAG, AV.TILT      # 15, 50
LO = np.concatenate([[15.0], AV.LO])
HI = np.concatenate([[105.0], AV.HI])


def vec_to_params(x):
    return dict(zip(PARAM_NAMES, [float(v) for v in x]))


def element_x_extent(p):
    """(min, max) x of the element's boundary vertices -- the horizontal footprint of the glass."""
    xs = [v["x"] for v in AV.prism_element(p)["path"]]
    return min(xs), max(xs)


def geometry_valid(p):
    # non-degenerate element body (reuse the AR/fold body check) + the element's bottom edge
    # stays above the source bundle so it does not enclose the emitter + the element stays clear
    # of the detector plane so the detector is always in free space (never inside the glass).
    if not FD.geometry_valid(p):
        return False
    bottom = -ELEM_H - max(p["dsag"], 0.0)
    if bottom <= Y_SOURCE + 5.0:
        return False
    _, xmax = element_x_extent(p)
    return xmax < X_SENSOR - DET_GAP


def scene(p, *, n_rays=N_RAYS, beam_w=BEAM_W, ray_brightness=1.0, record=True, crop=False,
          max_depth=16):
    """Parallel bundle from below aimed straight up; AR-style prism; detector on the right."""
    a = np.radians(90.0)                                   # straight up (parallel bundle)
    objs = []
    for dx in np.linspace(-beam_w / 2, beam_w / 2, n_rays):
        x = p["ray_x"] + dx
        objs.append({"type": "SingleRay", "p1": {"x": float(x), "y": Y_SOURCE},
                     "p2": {"x": float(x + np.cos(a)), "y": float(Y_SOURCE + np.sin(a))},
                     "brightness": float(ray_brightness)})
    objs.append(AV.prism_element(p))                        # 4-curved-face TIR prism
    objs.append({"type": "Detector", "p1": {"x": X_SENSOR, "y": -120},
                 "p2": {"x": X_SENSOR, "y": 120}})
    sc = {"version": 5, "maxRayDepth": max_depth, "objs": objs}
    if record:
        sc["recordInteractions"] = True
    if crop:
        objs.append({"type": "CropBox", "p1": {"x": -6, "y": Y_SOURCE - 10},
                     "p4": {"x": X_SENSOR + 10, "y": ELEM_H + _SAG + 12}, "width": 1600})
    return sc
