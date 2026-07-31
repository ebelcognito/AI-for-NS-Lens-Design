"""Validation gate for the differentiable parametric-arc prism primitive.

Mirrors validate_freeform.py's philosophy: prove the new torch primitive is a faithful
reimplementation of the search/oracle one (prism_branch.hit_ff), not an approximation.

 (1) REDUCE-TO-CIRCLE: with coeffs=0, ray_parametric_arc_roots must reproduce prism_branch's
     bumped-circle hit (t, hit point, normal) on a real prism face, over a fan of rays.
 (2) FORWARD-MATCH WITH BUMP: same, with the design's REAL aspheric coeffs (a2,a4 per face) ->
     proves the radial-bump branch matches the numpy Newton, not just the bare circle.
 (3) IFT GRADIENT: analytic d(hit_y)/d(coeff) from the single live Newton step vs a finite
     difference through the same root finder. rel.err ~ 1e-6.

Run:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.validate_prism_ff
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from . import prism_branch as PB  # noqa: E402  (build_arcs_ff, hit_ff)
from .geometry_prism_ff import (ParametricArcSurface, ray_parametric_arc_roots,  # noqa: E402
                                surface_from_arc_dict)

torch.set_default_dtype(torch.float64)

ARCHIVE = REPO / "ns_autodiff" / "results_search_prism_freeform" / "archive_prism_freeform.json"


def best_genotype():
    cells = json.loads(ARCHIVE.read_text())["cells"]
    return np.array(max(cells, key=lambda c: c["fitness"])["genotype"], float)


def face_rays(face, n=21, tilts_deg=(-12, -6, 0, 6, 12), dist=40.0):
    """A fan of rays that strike `face` near a spread of arc positions s in [-0.8,0.8]:
    aim each ray inward toward a target point on the base circle, at several tilt angles."""
    ce = np.asarray(face["center"], float); r = float(face["r"])
    cen = ce
    apex = np.asarray(face["apex"], float)
    c1 = np.asarray(face["c1"], float); c2 = np.asarray(face["c2"], float)
    th_ap = math.atan2(apex[1] - cen[1], apex[0] - cen[0])
    th1 = th_ap + ((math.atan2(c1[1] - cen[1], c1[0] - cen[0]) - th_ap + math.pi) % (2 * math.pi) - math.pi)
    th2 = th_ap + ((math.atan2(c2[1] - cen[1], c2[0] - cen[0]) - th_ap + math.pi) % (2 * math.pi) - math.pi)
    half = max(abs(th1 - th_ap), abs(th2 - th_ap), 1e-6)
    O, D = [], []
    for s in np.linspace(-0.8, 0.8, n):
        theta = th_ap + s * half
        outward = np.array([math.cos(theta), math.sin(theta)])
        target = ce + r * outward
        for td in tilts_deg:
            a = math.radians(td)
            # rotate the inward direction (-outward) by the tilt
            ca, sa = math.cos(a), math.sin(a)
            d = -outward
            d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
            O.append(target + dist * outward)   # start outside, aimed back at the arc
            D.append(d / np.hypot(*d))
    return np.array(O, float), np.array(D, float)


def numpy_hits(O, D, face):
    """prism_branch.hit_ff per ray -> (t[m], hit[m,2], nrm[m,2], valid[m])."""
    m = len(O)
    t = np.zeros(m); hit = np.zeros((m, 2)); nrm = np.zeros((m, 2)); val = np.zeros(m, bool)
    for i in range(m):
        h = PB.hit_ff(O[i], D[i], face)
        if h is None:
            continue
        t[i], hit[i], nrm[i] = h[0], h[1], h[2]; val[i] = True
    return t, hit, nrm, val


def forward_match(genotype, with_bump):
    tag = "WITH-BUMP (real coeffs)" if with_bump else "REDUCE-TO-CIRCLE (coeffs=0)"
    print("=" * 78); print(f"  FORWARD-MATCH  {tag}"); print("=" * 78)
    x = genotype.copy()
    if not with_bump:
        x = x.copy(); x[9:] = 0.0
    arcs, n = PB.build_arcs_ff(x)
    ok = True
    for name in ("A", "B", "T", "D"):
        face = arcs[name]
        coeffs = None if not with_bump else torch.as_tensor(np.asarray(face["coeffs"], float))
        O, D = face_rays(face)
        tn, hn, nn, vn = numpy_hits(O, D, face)
        surf = surface_from_arc_dict(face, coeffs=coeffs)
        Ot = torch.as_tensor(O); Dt = torch.as_tensor(D)
        tt, ht, nt, vt = ray_parametric_arc_roots(Ot, Dt, surf)
        vt = vt.numpy()
        both = vn & vt
        if both.sum() == 0:
            print(f"  face {name}: NO common valid rays"); ok = False; continue
        dt = np.abs(tn[both] - tt.numpy()[both]).max()
        dh = np.linalg.norm(hn[both] - ht.numpy()[both], axis=-1).max()
        ntn = nt.numpy()
        dn = np.linalg.norm(nn[both] - ntn[both], axis=-1).max()     # both oriented against D
        agree = int((vn == vt).sum()); total = vn.size
        good = dt < 1e-6 and dh < 1e-6 and dn < 1e-5 and agree >= total - 1
        ok = ok and good
        print(f"  face {name}: {int(both.sum()):3d} common | validity {agree}/{total} | "
              f"max|dt|={dt:.2e} max|dhit|={dh:.2e} max|dnrm|={dn:.2e}  {'OK' if good else 'MISMATCH'}")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def ift_gradient(genotype):
    print("=" * 78); print("  IFT GRADIENT  (one live Newton step vs finite difference)"); print("=" * 78)
    arcs, n = PB.build_arcs_ff(genotype)
    face = arcs["A"]
    O, D = face_rays(face, n=1, tilts_deg=(3.0,))
    Ot = torch.as_tensor(O[:1]); Dt = torch.as_tensor(D[:1])
    base = np.asarray(face["coeffs"], float)

    def hit_y(coeffs):
        surf = surface_from_arc_dict(face, coeffs=coeffs)
        _, hit, _, _ = ray_parametric_arc_roots(Ot, Dt, surf)
        return hit[0, 1]

    ok = True
    for j, lbl in ((0, "a2"), (1, "a4")):
        c0 = torch.as_tensor(base).clone().requires_grad_(True)
        y = hit_y(c0); y.backward()
        g_an = c0.grad[j].item()
        d = 1e-6
        with torch.no_grad():
            cp = torch.as_tensor(base).clone(); cp[j] += d
            cm = torch.as_tensor(base).clone(); cm[j] -= d
            g_fd = (hit_y(cp) - hit_y(cm)).item() / (2 * d)
        rel = abs(g_an - g_fd) / max(abs(g_fd), 1e-9)
        good = rel < 1e-4
        ok = ok and good
        print(f"  d(hit_y)/d{lbl}  analytic={g_an:+.6e}  FD={g_fd:+.6e}  rel.err={rel:.2e}  "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    g = best_genotype()
    print(f"Validating on best freeform-archive genotype (fitness-max cell).")
    r1 = forward_match(g, with_bump=False)
    r2 = forward_match(g, with_bump=True)
    r3 = ift_gradient(g)
    print("=" * 78)
    print(f"OVERALL: {'PASS' if (r1 and r2 and r3) else 'FAIL'}")
