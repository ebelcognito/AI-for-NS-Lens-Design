"""
NS folding-emergence probe — core library.

Phase-0 probe from agent_context/ns_rl_research/03_design_discussion.md §3:
run a gradient-free quality-diversity search over a 2-element, rotationally-symmetric-ish
(2D meridional) non-sequential optical scene, where each element has a *continuous
reflectivity* degree of freedom, and ask:

    Does any high-thinness, good-imaging archive cell get populated by a *folded*
    (multi-bounce) design, or does the archive's good region stay all-transmission?

Faithfulness to the emergence thesis (see 01_extracted_research_context.md):
  * The fitness is imaging quality only (focus + an honest flux floor). No term
    references reflection, bounce count, folding, or thinness.
  * Folding, if it appears, is read off the archive's *descriptive* behaviour axes,
    which are MEASURED from the tracked ray-surface interaction sequences, never targeted.

Engine: the instrumented ray-optics JS tracer (per-ray interaction sequences +
forced per-surface reflectivity), driven via the persistent Python bridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------------
# Fixed scene geometry (world units; the 2D meridional cross-section of a rot.-sym. system)
# ----------------------------------------------------------------------------------
X_SENSOR = 120.0            # detector plane (axial)
SENSOR_HALF_H = 60.0        # detector half-height (generous, to catch folded/stray energy)
BEAM_HALF_H = 16.0          # collimated input fan half-height at x=0
RAY_DENSITY = 1.6           # scene.rayModeDensity -> ~ 2*BEAM_HALF_H*RAY_DENSITY input rays
MAX_RAY_DEPTH = 12          # bounce budget (ample for 3-5 pass folds; faster than 16)
ELEM_HALF_H = 24.0          # element semi-aperture (fixed; > beam so folded rays are caught)
INPUT_BRIGHTNESS = 1.0

N_ELEM = 2

# Per-element design parameters and bounds (genotype = N_ELEM * 6 = 12 dims).
#   cx      axial centre of the element
#   thick   axial centre-thickness
#   fsag    front-face sag (signed; + bulges toward the source = convex front)
#   bsag    back-face sag  (signed; + bulges toward the sensor = convex back)
#   nidx    refractive index
#   refl    reflectivity in [0,1]  (the free DOF that makes folding physically reachable)
PARAM_NAMES = ["cx", "thick", "fsag", "bsag", "nidx", "refl"]
BOUNDS = np.array([
    [12.0, 108.0],   # cx
    [2.0,  14.0],    # thick
    [-13.0, 13.0],   # fsag
    [-13.0, 13.0],   # bsag
    [1.0,  1.9],     # nidx
    [0.0,  1.0],     # refl
], dtype=np.float64)
DIM = N_ELEM * len(PARAM_NAMES)


def bounds_lo_hi() -> Tuple[np.ndarray, np.ndarray]:
    lo = np.tile(BOUNDS[:, 0], N_ELEM)
    hi = np.tile(BOUNDS[:, 1], N_ELEM)
    return lo, hi


def clip_genotype(g: np.ndarray) -> np.ndarray:
    lo, hi = bounds_lo_hi()
    return np.clip(np.asarray(g, dtype=np.float64), lo, hi)


def sample_genotype(rng: np.random.Generator) -> np.ndarray:
    lo, hi = bounds_lo_hi()
    return lo + rng.random(DIM) * (hi - lo)


# ----------------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------------
def _arc_glass(cx, thick, fsag, bsag, n, r) -> dict:
    """A glass element with two circular-arc faces and a forced per-surface reflectivity.

    Path order (closed polygon, two arc faces): topLeft -> topRight -> backArcCtrl(arc)
    -> bottomRight -> bottomLeft -> frontArcCtrl(arc).  An `arc` control point bulges the
    face between its two polygon neighbours through itself; its signed offset is the sag.
    """
    lx, rx = cx - thick / 2.0, cx + thick / 2.0
    ty, by = -ELEM_HALF_H, ELEM_HALF_H
    return {
        "type": "Glass",
        "refIndex": float(n),
        "reflectivity": float(max(0.0, min(1.0, r))),
        "partialReflect": True,
        "path": [
            {"x": lx, "y": ty, "arc": False},
            {"x": rx, "y": ty, "arc": False},
            {"x": rx + float(bsag), "y": 0.0, "arc": True},   # back face arc
            {"x": rx, "y": by, "arc": False},
            {"x": lx, "y": by, "arc": False},
            {"x": lx - float(fsag), "y": 0.0, "arc": True},   # front face arc
        ],
    }


def build_scene(genotype: np.ndarray, *, record: bool = True) -> dict:
    g = clip_genotype(genotype).reshape(N_ELEM, len(PARAM_NAMES))
    objs: List[dict] = [{
        "type": "Beam",
        "p1": {"x": 0.0, "y": -BEAM_HALF_H},
        "p2": {"x": 0.0, "y": BEAM_HALF_H},
        "brightness": INPUT_BRIGHTNESS,
    }]
    for row in g:
        cx, thick, fsag, bsag, nidx, refl = row
        objs.append(_arc_glass(cx, thick, fsag, bsag, nidx, refl))
    objs.append({
        "type": "Detector",
        "p1": {"x": X_SENSOR, "y": -SENSOR_HALF_H},
        "p2": {"x": X_SENSOR, "y": SENSOR_HALF_H},
    })
    scene = {
        "version": 5,
        "rayModeDensity": RAY_DENSITY,
        "maxRayDepth": MAX_RAY_DEPTH,
        "objs": objs,
    }
    if record:
        scene["recordInteractions"] = True
    return scene


def empty_scene() -> dict:
    """Clear path (beam + detector only) for B_in flux calibration."""
    return {
        "version": 5,
        "rayModeDensity": RAY_DENSITY,
        "maxRayDepth": MAX_RAY_DEPTH,
        "recordInteractions": True,
        "objs": [
            {"type": "Beam", "p1": {"x": 0.0, "y": -BEAM_HALF_H},
             "p2": {"x": 0.0, "y": BEAM_HALF_H}, "brightness": INPUT_BRIGHTNESS},
            {"type": "Detector", "p1": {"x": X_SENSOR, "y": -SENSOR_HALF_H},
             "p2": {"x": X_SENSOR, "y": SENSOR_HALF_H}},
        ],
    }


# ----------------------------------------------------------------------------------
# Descriptors + fitness from the recorded interaction sequences
# ----------------------------------------------------------------------------------
@dataclass
class EvalResult:
    fitness: float
    bc: Optional[Tuple[float, float]]      # (thinness_depth T, folding mean-reversals) or None
    flux: float = 0.0
    spot_rms: float = float("nan")
    centroid: float = float("nan")
    folded_frac: float = 0.0               # fraction of forward energy with >=2 axial reversals
    opl_ratio: float = float("nan")        # energy-weighted optical-path / axial-distance
    mean_reversals: float = 0.0
    device_depth: float = float("nan")     # T = X_SENSOR - frontmost optic surface used
    n_forward: int = 0
    detector_power: float = 0.0
    error: Optional[str] = None


def _axial_reversals(xs: Sequence[float]) -> int:
    rev = 0
    for k in range(1, len(xs) - 1):
        d_prev = xs[k] - xs[k - 1]
        d_next = xs[k + 1] - xs[k]
        if d_prev * d_next < 0:
            rev += 1
    return rev


def _path_length(seq: Sequence[dict]) -> float:
    L = 0.0
    for k in range(1, len(seq)):
        dx = seq[k]["x"] - seq[k - 1]["x"]
        dy = seq[k]["y"] - seq[k - 1]["y"]
        L += math.hypot(dx, dy)
    return L


def analyse(result: dict, b_in: float, *, flux_floor: float = 0.10,
            flux_weight: float = 1.0) -> EvalResult:
    """Turn one raw simulation result into (fitness, descriptors)."""
    if result.get("error"):
        return EvalResult(fitness=-1e6, bc=None, error=str(result["error"]))

    recs = result.get("raySequences", []) or []
    det_power = float((result.get("detectors") or [{}])[0].get("power", 0.0)) if result.get("detectors") else 0.0

    # Forward crossings = image-forming candidates (moving +x through the sensor plane).
    fwd = [s for s in recs if s.get("dirx", 0.0) > 0.0 and s.get("b", 0.0) > 0.0]
    if not fwd:
        return EvalResult(fitness=-1000.0, bc=None, flux=0.0, n_forward=0,
                          detector_power=det_power)

    ys = np.array([s["y"] for s in fwd], dtype=np.float64)
    bs = np.array([s["b"] for s in fwd], dtype=np.float64)
    wsum = float(bs.sum())
    flux = wsum / max(b_in, 1e-12)

    centroid = float(np.average(ys, weights=bs))
    spot_rms = float(math.sqrt(np.average((ys - centroid) ** 2, weights=bs)))

    # Folding descriptors (measured, never targeted).
    revs = np.array([_axial_reversals([v["x"] for v in s["seq"]]) for s in fwd], dtype=np.float64)
    mean_reversals = float(np.average(revs, weights=bs))
    folded_frac = float(bs[revs >= 2].sum() / wsum)
    opl = np.array([_path_length(s["seq"]) / max(s["seq"][-1]["x"] - s["seq"][0]["x"], 1e-9)
                    for s in fwd], dtype=np.float64)
    opl_ratio = float(np.average(opl, weights=bs))

    # Device depth T = X_SENSOR - frontmost optic surface actually used by forward energy.
    optic_xs = [v["x"] for s in fwd for v in s["seq"] if 0.5 < v["x"] < X_SENSOR - 0.5]
    front_x = min(optic_xs) if optic_xs else 0.0
    device_depth = float(X_SENSOR - front_x)

    # Fitness: imaging quality with an honest flux floor. No reference to folding/reflection.
    if flux < flux_floor:
        fitness = -1000.0 + 500.0 * flux            # climb flux first
    else:
        fitness = -spot_rms + flux_weight * min(flux, 1.0)

    bc = (device_depth, mean_reversals)
    return EvalResult(
        fitness=float(fitness), bc=bc, flux=float(flux), spot_rms=spot_rms,
        centroid=centroid, folded_frac=folded_frac, opl_ratio=opl_ratio,
        mean_reversals=mean_reversals, device_depth=device_depth,
        n_forward=len(fwd), detector_power=det_power,
    )


# ----------------------------------------------------------------------------------
# Evaluator (genotype -> EvalResult), wrapping the persistent Node bridge
# ----------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, runner, b_in: float, *, flux_floor: float = 0.10,
                 flux_weight: float = 1.0, scene_builder=None):
        self.runner = runner
        self.b_in = float(b_in)
        self.flux_floor = flux_floor
        self.flux_weight = flux_weight
        self.scene_builder = scene_builder or build_scene
        self.n_calls = 0
        self.n_errors = 0

    def __call__(self, genotype: np.ndarray) -> EvalResult:
        scene = self.scene_builder(genotype, record=True)
        self.n_calls += 1
        try:
            res, _ = self.runner.simulate(scene)
        except Exception as exc:  # timeout / dead process -> treat as invalid design
            self.n_errors += 1
            return EvalResult(fitness=-1e6, bc=None, error=f"sim:{type(exc).__name__}")
        return analyse(res, self.b_in, flux_floor=self.flux_floor,
                       flux_weight=self.flux_weight)


def calibrate_b_in(runner) -> float:
    res, _ = runner.simulate(empty_scene())
    recs = res.get("raySequences", []) or []
    fwd = [s for s in recs if s.get("dirx", 0.0) > 0.0]
    b = sum(s.get("b", 0.0) for s in fwd)
    if b <= 0:  # fallback to detector power
        b = float((res.get("detectors") or [{}])[0].get("power", 1.0))
    return float(b)
