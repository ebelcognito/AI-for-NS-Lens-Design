"""The AR/VR prism relay: a genuinely different measurement model.

Every other QD experiment measures FOCUSED THROUGHPUT of one bundle at a detector. This
one measures a three-field RELAY: a microdisplay above the element emits from three
distinct x positions, and the design must deliver all three to the eyebox as well
collimated beams whose exit angles stay correctly ordered (no field crossing).

So its fitness is not focused throughput, its descriptors are FOV and prism depth rather
than fold depth and thinness, and it rejects designs the throughput evaluator would
happily score. Forcing it through `Observation` would be the wrong abstraction -- what is
genuinely shared is the CMA-ME loop, the RNG protocol, the archive and the payload
format, so ArvrProblem supplies its own `evaluate` and reuses everything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ..scene import Detector, Layout, PointFan
from .problems import Problem

# --- display / eyebox geometry -------------------------------------------------
PUPIL_X = 120.0            # eye-pupil plane (light enters the eye travelling +x)
EYEBOX_Y = 0.0             # eyebox centre, on the forward axis
EYEBOX_HALF = 9.0          # eyebox half-height
DISPLAY_Y = 50.0           # microdisplay above the element
DISPLAY_CX = 55.0
DISPLAY_HALF_W = 6.0       # 3 sources at x = 49, 55, 61
SOURCE_XS = np.linspace(DISPLAY_CX - DISPLAY_HALF_W, DISPLAY_CX + DISPLAY_HALF_W, 3)
SOURCE_PTS = [(float(sx), DISPLAY_Y) for sx in SOURCE_XS]
SOURCE_AIM = -90.0         # central emission direction: straight down
N_FAN = 17                 # rays per source
FAN_HALFANGLE = 12.0       # deg, fan half-cone about SOURCE_AIM
SIGMA0 = 0.7               # deg, angular tolerance in the fitness

SAG = 15.0
TILT = 50.0


def source_index(seq_x: float) -> int:
    """Attribute a ray to its source by the x of its emission vertex."""
    return int(np.argmin([abs(seq_x - sx) for sx, _ in SOURCE_PTS]))


def arvr_metrics(result: dict, *, n_fan: int = N_FAN) -> dict:
    """Per-source angular metrics at the eyebox, plus the scalar relay fitness."""
    per = {i: {"ang": [], "w": []} for i in range(len(SOURCE_PTS))}
    for r in result.get("raySequences", []) or []:
        if r.get("dirx", 0) <= 0 or r.get("b", 0) <= 0:      # must exit toward the eye
            continue
        if abs(float(r["y"]) - EYEBOX_Y) > EYEBOX_HALF:      # inside the eyebox?
            continue
        src = source_index(float(r["seq"][0]["x"]))
        per[src]["ang"].append(np.degrees(np.arctan2(r["diry"], r["dirx"])))
        per[src]["w"].append(float(r["b"]))

    fields = []
    for i in range(len(SOURCE_PTS)):
        a = np.array(per[i]["ang"]); w = np.array(per[i]["w"])
        if w.sum() <= 0:
            fields.append(dict(present=False, T=0.0, theta=None, sigma=None, n=0))
            continue
        theta = float(np.average(a, weights=w))
        sigma = float(np.sqrt(np.average((a - theta) ** 2, weights=w)))
        fields.append(dict(present=True, T=float(w.sum() / n_fan), theta=theta,
                           sigma=sigma, n=int(len(a))))

    present = [f for f in fields if f["present"]]
    thetas = [f["theta"] for f in fields]
    # exit angles must stay monotonic across the display: a crossing means the relay
    # scrambles the image, however bright it is
    ordering_ok = (len(present) == 3 and
                   (thetas[0] < thetas[1] < thetas[2] or thetas[0] > thetas[1] > thetas[2]))
    q = [f["T"] * np.exp(-f["sigma"] / SIGMA0) if f["present"] else 0.0 for f in fields]
    fitness = float(np.mean(q)) * (1.0 if ordering_ok else 0.0)
    fov = (abs(fields[2]["theta"] - fields[0]["theta"])
           if fields[0]["present"] and fields[2]["present"] else 0.0)
    return dict(fields=fields, fitness=fitness, fov=float(fov),
                ordering_ok=bool(ordering_ok),
                T_min=float(min(f["T"] for f in fields)),
                sigma_mean=float(np.mean([f["sigma"] for f in present])) if present else None)


def prism_depth(result: dict) -> float:
    """Form factor: x-extent of the glass actually traversed on eye-bound paths."""
    xs = [v["x"] for r in result.get("raySequences", []) or [] if r.get("dirx", 0) > 0
          for v in r["seq"] if 1.0 < v["x"] < PUPIL_X - 1.0]
    return (max(xs) - min(xs)) if xs else 0.0


def tir_bounces_on_brightest(result: dict) -> int:
    """TIR events on the densest eyebox-reaching path. Reported for analysis; NOT a
    descriptor, so folding is never rewarded here either."""
    best, bestb = 0, -1.0
    for r in result.get("raySequences", []) or []:
        if r.get("dirx", 0) <= 0 or abs(float(r["y"]) - EYEBOX_Y) > EYEBOX_HALF:
            continue
        if r["b"] > bestb:
            bestb = r["b"]
            best = sum(1 for v in r["seq"] if v.get("t") == "tir")
    return best


@dataclass
class ArvrProblem(Problem):
    """Microdisplay above, eye to the right, 4-curved-face prism between them."""

    param_names: Sequence[str] = field(default_factory=lambda: [
        "cx", "thick", "fsag", "bsag", "tsag", "dsag", "tilt_a", "tilt_b", "n"])
    lo: np.ndarray = field(default_factory=lambda: np.array(
        [35.0, 6.0, -SAG, -SAG, -SAG, -SAG, -TILT, -TILT, 1.50]))
    hi: np.ndarray = field(default_factory=lambda: np.array(
        [85.0, 45.0, SAG, SAG, SAG, SAG, TILT, TILT, 2.00]))
    half_h: float = 30.0
    x_sensor: float = PUPIL_X
    n_fan: int = N_FAN
    fan_halfangle: float = FAN_HALFANGLE
    min_tir: int = 0            # if > 0, only archive designs with at least this many

    def layout(self, p: Dict[str, float], *, field_deg: float = 0.0) -> Layout:
        return Layout(
            source=PointFan(points=SOURCE_PTS, aim_deg=SOURCE_AIM + field_deg,
                            halfangle_deg=self.fan_halfangle, n_fan=self.n_fan),
            element=self.element(p),
            detector=Detector.vertical(self.x_sensor),
        )

    def is_valid(self, p: Dict[str, float]) -> bool:
        """Non-degenerate body AND the curved top stays clear below the display."""
        if not self.element(p).is_valid():
            return False
        return self.half_h + max(p.get("tsag", 0.0), 0.0) < DISPLAY_Y - 2.0

    # -- the driver calls this instead of the throughput evaluator ---------------
    def evaluate(self, runner, cfg, x, pert_rng=None) -> Optional[dict]:
        p = self.vec_to_params(x)
        if not self.is_valid(p):
            return None
        try:
            res, _ = runner.simulate(self.layout(p).to_scene_dict())
        except Exception:
            return None
        if res.get("error"):
            return None

        m = arvr_metrics(res, n_fan=self.n_fan)
        # only real 3-field relays enter the archive
        if not m["ordering_ok"] or m["fitness"] <= 0:
            return None
        depth = prism_depth(res)
        fov = m["fov"]
        if fov <= 0:
            return None
        tc = tir_bounces_on_brightest(res)
        if tc < self.min_tir:
            return None

        return dict(
            fitness=float(m["fitness"]), bc=(float(fov), float(depth)),
            info=dict(fov=float(fov), depth=float(depth), fitness=float(m["fitness"]),
                      T_min=float(m["T_min"]), sigma_mean=float(m["sigma_mean"]),
                      tir=int(tc),
                      thetas=[f["theta"] for f in m["fields"]],
                      sigmas=[f["sigma"] for f in m["fields"]],
                      Ts=[f["T"] for f in m["fields"]]))
