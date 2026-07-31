"""Validation gate for the freeform surface primitive.

Two checks, mirroring ns_autodiff/validate.py's philosophy (prove the new machinery is a
faithful reimplementation, not an approximation, before trusting it):

 (1) REDUCE-TO-ARC: with c=-1/R, k=0, no polynomial terms, ray_freeform_roots must reproduce
     the closed-form ray_circle_roots + on_arc nearest-valid hit to ~1e-9 (t, hit, normal),
     over a fan of rays at several incidence angles, on BOTH curved faces of a real element.

 (2) IFT GRADIENT: the exact analytic gradient d(hit_y)/d(param) from the single live Newton
     step must match a finite-difference gradient through the same root finder -- for the
     curvature c AND for a polynomial coefficient a4 (the real freeform DOF). cosine ~ 1.

Run:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.validate_freeform
"""
from __future__ import annotations

import math

import torch

from . import geometry_torch as G
from .element_torch import build_surfaces
from .element_freeform import arc_to_freeform
from .geometry_freeform import FreeformSurface, ray_freeform_roots

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
#  closed-form circle reference: nearest forward on-arc hit per ray
# --------------------------------------------------------------------------- #
def circle_nearest(O, D, arc):
    troots, hit, nrm, fwd = G.ray_circle_roots(O, D, arc["center"], arc["r"])
    on = G.on_arc(hit, arc["end1"], arc["end2"], arc["apex"]) & fwd     # [2,m]
    inf = torch.full_like(troots, float("inf"))
    cost = torch.where(on, troots, inf)                                  # [2,m]
    sel = torch.argmin(cost, dim=0)                                      # [m]
    m = O.shape[0]
    t = torch.gather(troots, 0, sel.view(1, m))[0]
    h = torch.gather(hit, 0, sel.view(1, m, 1).expand(1, m, 2))[0]
    n = torch.gather(nrm, 0, sel.view(1, m, 1).expand(1, m, 2))[0]
    valid = torch.gather(on, 0, sel.view(1, m))[0]
    return t, h, n, valid


def make_rays(x0, y_lo, y_hi, n, angles_deg):
    """Fan of rays from x=x0 across y in [y_lo,y_hi], at several small tilt angles."""
    ys = torch.linspace(y_lo, y_hi, n)
    Os, Ds = [], []
    for ad in angles_deg:
        a = math.radians(ad)
        for y in ys:
            Os.append(torch.tensor([x0, float(y)]))
            Ds.append(torch.tensor([math.cos(a), math.sin(a)]))
    return torch.stack(Os), G.normalize(torch.stack(Ds))


def reduce_to_arc():
    print("=" * 74)
    print("(1) REDUCE-TO-ARC  (freeform conic == circular arc, c=-1/R, k=0, no poly)")
    print("=" * 74)
    p = torch.tensor([0.0, 0.0, 85.0, 18.0, 12.0, -14.0, 8.0, -6.0, 1.55])
    surfaces = build_surfaces(p)
    # rays that strike each cap from just left of its apex (interior of the aperture, so the
    # chord-side on_arc test and the transverse-projection aperture test agree ray-for-ray)
    fans = {"A": make_rays(30.0, -16.0, 16.0, 25, [-10, -5, 0, 5, 10]),
            "B": make_rays(50.0, -16.0, 16.0, 25, [-10, -5, 0, 5, 10])}
    ok = True
    for face in ("A", "B"):
        arc = surfaces[face]
        O, D = fans[face]
        ff = arc_to_freeform(arc)                       # pure conic -> should match the arc
        tc, hc, nc, vc = circle_nearest(O, D, arc)
        tf, hf, nf, vf = ray_freeform_roots(O, D, ff)
        both = vc & vf
        n_both = int(both.sum())
        # validity agreement
        agree = int((vc == vf).sum()); total = vc.numel()
        if n_both == 0:
            print(f"  face {face}: NO common valid rays -- check geometry"); ok = False; continue
        et = (tf - tc)[both].abs().max().item()
        eh = (hf - hc)[both].norm(dim=-1).max().item()
        # normals may differ by overall sign convention; compare as lines (|.| of cross)
        cross = (nf[:, 0] * nc[:, 1] - nf[:, 1] * nc[:, 0])[both].abs().max().item()
        dotsign = (nf * nc).sum(-1)[both].min().item()
        print(f"  face {face}: {n_both:3d} common rays | validity agree {agree}/{total} | "
              f"max|dt|={et:.2e}  max|dhit|={eh:.2e}  max|n x n|={cross:.2e}  min(n.n)={dotsign:+.3f}")
        ok = ok and (et < 1e-8 and eh < 1e-8 and cross < 1e-8 and agree == total)
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def ift_gradient():
    print("=" * 74)
    print("(2) IFT GRADIENT  (analytic one-live-Newton-step vs finite difference)")
    print("=" * 74)
    p = torch.tensor([0.0, 0.0, 85.0, 18.0, 12.0, -14.0, 8.0, -6.0, 1.55])
    arc = build_surfaces(p)["A"]
    O = torch.tensor([[30.0, 5.0]]); D = G.normalize(torch.tensor([[math.cos(math.radians(4.0)),
                                                                    math.sin(math.radians(4.0))]]))
    base = arc_to_freeform(arc)
    ok = True

    def hit_y(c=None, a4=None):
        c_t = (c if c is not None else base.c)
        powers = (4,) if a4 is not None else ()
        coeffs = (torch.stack([a4]) if a4 is not None else None)
        surf = FreeformSurface(v=base.v, axis=base.axis, tang=base.tang, c=c_t, k=base.k,
                               h_min=base.h_min, h_max=base.h_max, powers=powers, coeffs=coeffs)
        _, hit, _, _ = ray_freeform_roots(O, D, surf)
        return hit[0, 1]

    # --- curvature c ---
    c0 = base.c.clone().requires_grad_(True)
    y = hit_y(c=c0); y.backward()
    g_an = c0.grad.item()
    dc = 1e-6
    with torch.no_grad():
        g_fd = (hit_y(c=base.c + dc) - hit_y(c=base.c - dc)).item() / (2 * dc)
    rel = abs(g_an - g_fd) / max(abs(g_fd), 1e-12)
    print(f"  d(hit_y)/dc   analytic={g_an:+.6e}  FD={g_fd:+.6e}  rel.err={rel:.2e}")
    ok = ok and rel < 1e-4

    # --- polynomial coefficient a4 (the real freeform DOF), evaluated at a4=0 ---
    a0 = torch.zeros((), requires_grad=True)
    y = hit_y(a4=a0); y.backward()
    g_an = a0.grad.item()
    da = 1e-5                       # h^4 is a stiff lever; coarse FD truncates (cf. README bsag)
    with torch.no_grad():
        g_fd = (hit_y(a4=torch.tensor(da)) - hit_y(a4=torch.tensor(-da))).item() / (2 * da)
    rel = abs(g_an - g_fd) / max(abs(g_fd), 1e-12)
    print(f"  d(hit_y)/da4  analytic={g_an:+.6e}  FD={g_fd:+.6e}  rel.err={rel:.2e}")
    ok = ok and rel < 1e-4
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = reduce_to_arc()
    r2 = ift_gradient()
    print("=" * 74)
    print(f"OVERALL: {'PASS' if (r1 and r2) else 'FAIL'}")
