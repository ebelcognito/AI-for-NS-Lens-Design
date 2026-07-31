"""Analytic chief-ray (beam-center) trace through the fold element, following the
dom_signature. Self-contained (no Node engine): reproduces the engine's dominant-ray
hit points so we can build a FOLDED-AXIS CODE V model (real thicknesses between hit
points + fold tilts). Validated against CONTEXT.md §6 for cell 4_11.

Geometry conventions match export_fold_for_codev.faces_of:
  A,B = curved arcs (circle center + radius); T (y=-H), D (y=+H) = horizontal flats.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_fold_for_codev import faces_of, parse_signature, ELEM_H, X_SENSOR  # noqa: E402


def _unit(v):
    m = math.hypot(v[0], v[1])
    return (v[0] / m, v[1] / m)


def _ray_circle(o, d, C, R):
    """Nearest forward intersection (t>eps) of ray o+t d with circle (C,|R|)."""
    fx, fy = o[0] - C[0], o[1] - C[1]
    b = 2.0 * (fx * d[0] + fy * d[1])
    c = fx * fx + fy * fy - R * R
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    ts = sorted(t for t in ((-b - sq) / 2.0, (-b + sq) / 2.0) if t > 1e-7)
    return ts[0] if ts else None


def _ray_hline(o, d, y):
    if abs(d[1]) < 1e-12:
        return None
    t = (y - o[1]) / d[1]
    return t if t > 1e-7 else None


def _normal_arc(hit, C):
    """Unit outward normal of the arc at hit (from center to hit)."""
    return _unit((hit[0] - C[0], hit[1] - C[1]))


def trace(p, sig):
    """Return list of hit dicts along the chief ray + the final sensor hit.
    Each hit: face, event, point (x,y), n_in (incoming unit dir), n_out (outgoing),
    normal (unit, oriented against incoming), incidence(deg)."""
    faces = faces_of(p)
    events = parse_signature(sig)
    n = p["n"]

    o = (0.0, p["ray_y"])
    d = _unit((math.cos(math.radians(p["ang"])), math.sin(math.radians(p["ang"]))))
    medium_n = 1.0  # start in air
    hits = []

    for etype, fl in events:
        f = faces[fl]
        if f["kind"] == "cyl":
            C = f["center"]
            R = abs(f["radius"])
            t = _ray_circle(o, d, C, R)
        else:  # flat horizontal edge at vertex y
            C = None
            t = _ray_hline(o, d, f["vertex"][1])
        if t is None:
            raise RuntimeError(f"chief ray missed face {fl} ({etype})")
        hit = (o[0] + t * d[0], o[1] + t * d[1])

        if C is not None:
            nrm = _normal_arc(hit, C)
        else:
            nrm = (0.0, 1.0)
        # orient normal against the incoming ray
        if (nrm[0] * d[0] + nrm[1] * d[1]) > 0:
            nrm = (-nrm[0], -nrm[1])
        cos_i = -(d[0] * nrm[0] + d[1] * nrm[1])
        inc = math.degrees(math.acos(max(-1.0, min(1.0, cos_i))))

        n_in = d
        if etype == "tir":  # reflect
            dot = d[0] * nrm[0] + d[1] * nrm[1]
            d = _unit((d[0] - 2 * dot * nrm[0], d[1] - 2 * dot * nrm[1]))
        else:  # refract (Snell, vector form)
            n1 = medium_n
            n2 = n if medium_n == 1.0 else 1.0
            r = n1 / n2
            cosi = -(d[0] * nrm[0] + d[1] * nrm[1])
            k = 1.0 - r * r * (1.0 - cosi * cosi)
            if k < 0:
                raise RuntimeError(f"TIR where refract expected @ {fl}")
            d = _unit((r * d[0] + (r * cosi - math.sqrt(k)) * nrm[0],
                       r * d[1] + (r * cosi - math.sqrt(k)) * nrm[1]))
            medium_n = n2

        hits.append(dict(face=fl, event=etype, point=hit, n_in=n_in, n_out=d,
                         normal=nrm, incidence=inc))
        o = hit

    # to the sensor plane at x = X_SENSOR
    t = (X_SENSOR - o[0]) / d[0]
    sensor = (o[0] + t * d[0], o[1] + t * d[1])
    return hits, sensor


if __name__ == "__main__":
    # validate vs CONTEXT §6 (cell 4_11)
    p = dict(ray_y=9.4592, ang=23.8848, cx=58.3601, thick=28.6574,
             fsag=-3.9085, bsag=2.5657, tilt_a=25.0, tilt_b=15.6382, n=1.5)
    sig = "T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B"
    ref = [("A", (34.021, 24.524)), ("D", (44.524, 30.000)), ("B", (69.784, 16.831)),
           ("A", (55.407, -20.004)), ("T", (69.378, -30.000)), ("B", (80.056, -22.360))]
    hits, sensor = trace(p, sig)
    print(f"{'face':>4} {'event':>8} {'x':>9} {'y':>9} | {'ref x':>8} {'ref y':>8} {'err':>7}")
    ok = True
    for h, (rf, rp) in zip(hits, ref):
        e = math.hypot(h["point"][0] - rp[0], h["point"][1] - rp[1])
        ok &= e < 0.01
        print(f"{h['face']:>4} {h['event']:>8} {h['point'][0]:9.3f} {h['point'][1]:9.3f} "
              f"| {rp[0]:8.3f} {rp[1]:8.3f} {e:7.4f}")
    print(f"sensor: ({sensor[0]:.3f}, {sensor[1]:.3f})  [ref (120.000, 27.545)]")
    print("VALIDATION:", "PASS" if ok else "FAIL")
