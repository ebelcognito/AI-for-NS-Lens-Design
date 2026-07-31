"""
SINGLE-ELEMENT variant of the NS folding-emergence probe.

This is the faithful version of the original existence proof (a *single* deformable element
where total internal reflection emerged). One glass element with two arc faces, each face
having an independent continuous reflectivity (Mangin / monolithic-pancake style), plus the
usual geometry DOFs. Folding here = internal multi-bounce / TIR within the single element
whose forward-exiting component reaches the sensor on a long (folded) optical path.

Reuses probe.py for descriptors / fitness / evaluator / the empty-scene calibration; only the
genotype schema and scene builder differ.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ns_probe import probe

# Single element, per-face reflectivity. Genotype = 7 dims:
#   cx     axial centre
#   thick  centre thickness
#   fsag   front-face sag (signed; + = convex front)
#   bsag   back-face sag  (signed; + = convex back)
#   nidx   refractive index
#   rfront front-face reflectivity in [0,1]
#   rback  back-face reflectivity  in [0,1]
PARAM_NAMES = ["cx", "thick", "fsag", "bsag", "nidx", "rfront", "rback"]
BOUNDS = np.array([
    [12.0, 108.0],   # cx
    [2.0,  18.0],    # thick (a bit thicker allowed: a single element needs room to fold)
    [-15.0, 15.0],   # fsag
    [-15.0, 15.0],   # bsag
    [1.0,  1.95],    # nidx
    [0.0,  1.0],     # rfront
    [0.0,  1.0],     # rback
], dtype=np.float64)
DIM = len(PARAM_NAMES)


def bounds_lo_hi() -> Tuple[np.ndarray, np.ndarray]:
    return BOUNDS[:, 0].copy(), BOUNDS[:, 1].copy()


def clip_genotype(g: np.ndarray) -> np.ndarray:
    lo, hi = bounds_lo_hi()
    return np.clip(np.asarray(g, dtype=np.float64), lo, hi)


def sample_genotype(rng: np.random.Generator) -> np.ndarray:
    lo, hi = bounds_lo_hi()
    return lo + rng.random(DIM) * (hi - lo)


def _single_element(cx, thick, fsag, bsag, n, rfront, rback) -> dict:
    lx, rx = cx - thick / 2.0, cx + thick / 2.0
    ty, by = -probe.ELEM_HALF_H, probe.ELEM_HALF_H
    return {
        "type": "Glass",
        "refIndex": float(n),
        "partialReflect": True,
        # per-face reflectivity, selected by side of elemCx in the instrumented engine
        "reflectivityFront": float(max(0.0, min(1.0, rfront))),
        "reflectivityBack": float(max(0.0, min(1.0, rback))),
        "elemCx": float(cx),
        "path": [
            {"x": lx, "y": ty, "arc": False},
            {"x": rx, "y": ty, "arc": False},
            {"x": rx + float(bsag), "y": 0.0, "arc": True},
            {"x": rx, "y": by, "arc": False},
            {"x": lx, "y": by, "arc": False},
            {"x": lx - float(fsag), "y": 0.0, "arc": True},
        ],
    }


def build_scene(genotype: np.ndarray, *, record: bool = True) -> dict:
    cx, thick, fsag, bsag, nidx, rfront, rback = clip_genotype(genotype)
    objs: List[dict] = [{
        "type": "Beam",
        "p1": {"x": 0.0, "y": -probe.BEAM_HALF_H},
        "p2": {"x": 0.0, "y": probe.BEAM_HALF_H},
        "brightness": probe.INPUT_BRIGHTNESS,
    }, _single_element(cx, thick, fsag, bsag, nidx, rfront, rback), {
        "type": "Detector",
        "p1": {"x": probe.X_SENSOR, "y": -probe.SENSOR_HALF_H},
        "p2": {"x": probe.X_SENSOR, "y": probe.SENSOR_HALF_H},
    }]
    scene = {"version": 5, "rayModeDensity": probe.RAY_DENSITY,
             "maxRayDepth": probe.MAX_RAY_DEPTH, "objs": objs}
    if record:
        scene["recordInteractions"] = True
    return scene
