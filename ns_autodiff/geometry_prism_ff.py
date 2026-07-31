"""Differentiable PARAMETRIC-ARC surface intersection for the AR/VR prism twin.

This is the one new piece of math the differentiable PRISM refiner needs -- the analog of
geometry_freeform.ray_freeform_roots, but for the surface form the prism actually uses.

Why NOT reuse geometry_freeform here
------------------------------------
geometry_freeform models a conic+poly `sag(h)` in a tilted frame. That form is single-valued
only for arcs spanning < 90deg (the cap curves back past 90). The measured prism folds need
112-137deg arcs (doc 03 gotcha), so the search/oracle tracer (prism_branch.hit_ff) represents
a face as a FULL CIRCLE plus a RADIAL BUMP on normalized arc position:

    surface point at arc parameter s in [-1,1]:   |P - center| = r + Delta(s)
    Delta(s) = sum_i a_i * s^i           (powers, default (2,4))
    s(P)     = ( unwrap(atan2(P-center), th_apex) - th_apex ) / half_span
               (apex at s=0, the two corners at s=+-1; valid for any span <= 180deg)

This module is the differentiable torch twin of that primitive. It returns the SAME
(t, hit, normal, valid) tuple as ray_freeform_roots / ray_circle_roots, so forward_prism_ff
can drop it into the nearest-surface argmin exactly like the circle/freeform faces.

The gradient recipe is identical to geometry_freeform (detached root find -> one LIVE Newton
step -> exact IFT gradient dt/dp = -G_p / G_t at the root), only the residual changes:

    G(t) = ( |P(t) - center| - r ) - Delta( s(t) ) = 0

Validated forward + grad against prism_branch.hit_ff in validate_prism_ff.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from .geometry_torch import EPS, _orient_against, normalize

T_MIN = 1e-6
TANGENT_FLOOR = 1e-7    # |dG/dt| below this => grazing / near-tangent: flag invalid
PI = math.pi


@dataclass
class ParametricArcSurface:
    """A circle (center, r) plus a radial bump Delta(s) on normalized arc position s in [-1,1].
    coeffs=None / all-zero reproduces the bare circular arc (the reduce-to-circle gate)."""
    center: torch.Tensor     # [2]
    r: torch.Tensor          # []
    th_apex: torch.Tensor    # [] base angle (atan2) of the apex direction
    half_span: torch.Tensor  # [] half the angular span apex->corner (>0)
    powers: tuple = (2, 4)
    coeffs: torch.Tensor = None   # [len(powers)] radial-bump coefficients, or None


def _unwrap(th, ref):
    """Wrap th into (ref-pi, ref+pi]; differentiable (the 2*pi*k shift is locally constant)."""
    return ref + ((th - ref + PI) % (2.0 * PI) - PI)


def _delta(s, powers, coeffs):
    """Radial bump Delta(s) and dDelta/ds. coeffs None -> (0,0)."""
    if coeffs is None:
        z = torch.zeros_like(s)
        return z, z.clone()
    d = torch.zeros_like(s)
    dd = torch.zeros_like(s)
    for j, pw in enumerate(powers):
        a = coeffs[j]
        d = d + a * s ** pw
        dd = dd + a * pw * s ** (pw - 1)
    return d, dd


def _G_and_dG(t, O, D, surf: ParametricArcSurface):
    """Residual G(t)=(|P-center|-r)-Delta(s) and dG/dt for batched rays at parameter t[...].
    Also returns (s, rad, radial, tang) for the normal/validity. D assumed unit."""
    P = O + t.unsqueeze(-1) * D
    rel = P - surf.center
    rad = torch.sqrt((rel * rel).sum(-1) + EPS * EPS)         # |P-center|
    radial = rel / rad.unsqueeze(-1)
    tang = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)   # +90deg rotate
    th = _unwrap(torch.atan2(rel[..., 1], rel[..., 0]), surf.th_apex)
    s = (th - surf.th_apex) / surf.half_span
    delta, ddelta = _delta(s, surf.powers, surf.coeffs)
    G = (rad - surf.r) - delta
    # dG/dt:  d(rad)/dt = radial.D ;  ds/dt = (1/half_span) * dtheta/dt , dtheta/dt = (rel x D)/rad^2
    drad_dt = (radial * D).sum(-1)
    cross = rel[..., 0] * D[..., 1] - rel[..., 1] * D[..., 0]
    ds_dt = (cross / (rad * rad)) / surf.half_span
    dG = drad_dt - ddelta * ds_dt
    return G, dG, s, rad, radial, tang, ddelta


def _circle_roots(O, D, surf: ParametricArcSurface):
    """The two closed-form roots of the BARE circle |O+tD-center|=r (D unit). Used only to
    bracket/seed the Newton polish (the bump is a small perturbation of the circle)."""
    oc = O - surf.center
    b = (D * oc).sum(-1)
    c = (oc * oc).sum(-1) - surf.r * surf.r
    disc = b * b - c
    has = disc >= 0
    sq = torch.sqrt(disc.clamp_min(0.0))
    return (-b - sq), (-b + sq), has


def _polish(t0, O, D, surf, n_newton=12):
    """Detached Newton from a circle-root seed onto the true bumped root."""
    t = t0.clone()
    for _ in range(n_newton):
        G, dG, *_ = _G_and_dG(t, O, D, surf)
        dG_safe = torch.where(dG.abs() < EPS, torch.full_like(dG, EPS), dG)
        t = t - G / dG_safe
    return t


def ray_parametric_arc_roots(O, D, surf: ParametricArcSurface, *, t_min: float = T_MIN,
                             n_newton: int = 12):
    """Nearest forward valid intersection of rays O+tD (D unit) with a parametric-arc face.

    Returns (t[m], hit[m,2], normal[m,2], valid[m]) -- one candidate per surface per ray, to be
    competed across surfaces by the caller's global nearest argmin (same contract as
    ray_freeform_roots). The two circle roots are the enumeration; each is Newton-polished onto
    the bumped surface (detached), the nearest forward valid one is kept, then a single LIVE
    Newton step injects the exact IFT gradient. Validity = on the real arc (|s|<=1), forward,
    not near-tangent."""
    O = O if O.dim() == 2 else O.unsqueeze(0)
    D = D if D.dim() == 2 else D.unsqueeze(0)
    m = O.shape[0]
    dtype, device = O.dtype, O.device
    INF = torch.tensor(float("inf"), dtype=dtype, device=device)

    with torch.no_grad():
        r0, r1, has = _circle_roots(O, D, surf)
        cands = []
        for seed in (r0, r1):
            t_p = _polish(seed, O, D, surf, n_newton=n_newton)
            G, dG, s, _rad, _rl, _tg, _dd = _G_and_dG(t_p, O, D, surf)
            ok = (has & (t_p > t_min) & (s.abs() <= 1.0 + 1e-6)
                  & (G.abs() < 1e-6) & (dG.abs() >= TANGENT_FLOOR))
            cands.append((t_p, ok))
        # nearest forward valid among the (up to) two polished roots
        cost = INF.expand(m).clone()
        t_det = torch.zeros(m, dtype=dtype, device=device)
        anyok = torch.zeros(m, dtype=torch.bool, device=device)
        for t_p, ok in cands:
            better = ok & (t_p < cost)
            cost = torch.where(better, t_p, cost)
            t_det = torch.where(better, t_p, t_det)
            anyok = anyok | ok

    # --- one LIVE Newton step => exact IFT gradient -------------------------------------- #
    G, dG, s, rad, radial, tang, ddelta = _G_and_dG(t_det, O, D, surf)
    dG_safe = torch.where(dG.abs() < TANGENT_FLOOR,
                          torch.where(dG >= 0, torch.full_like(dG, TANGENT_FLOOR),
                                      torch.full_like(dG, -TANGENT_FLOOR)), dG)
    t = t_det - G / dG_safe
    hit = O + t.unsqueeze(-1) * D

    # normal = grad_P G = radial - (dDelta/ds)/(half_span*rad) * tang, oriented to oppose D
    n_geom = radial - (ddelta / (surf.half_span * rad)).unsqueeze(-1) * tang
    normal = _orient_against(normalize(n_geom), D)

    valid = anyok & (s.abs() <= 1.0 + 1e-6) & (dG.abs() >= TANGENT_FLOOR)
    return t, hit, normal, valid


def surface_from_arc_dict(a, coeffs=None, powers=(2, 4), *, dtype=torch.float64, device="cpu"):
    """Build a ParametricArcSurface from a prism_branch arc dict {center, r, c1, c2, apex}.
    Mirrors prism_branch._aspheric_fields: th_apex from the apex direction, half_span = max
    angular distance apex->corner. `coeffs` may be a grad-carrying tensor (the refine DOF)."""
    import numpy as np
    ce = torch.as_tensor(np.asarray(a["center"], float), dtype=dtype, device=device)
    r = torch.as_tensor(float(a["r"]), dtype=dtype, device=device)
    apex = np.asarray(a["apex"], float)
    c1 = np.asarray(a["c1"], float)
    c2 = np.asarray(a["c2"], float)
    cen = np.asarray(a["center"], float)
    th_ap = math.atan2(apex[1] - cen[1], apex[0] - cen[0])
    th1 = th_ap + ((math.atan2(c1[1] - cen[1], c1[0] - cen[0]) - th_ap + PI) % (2 * PI) - PI)
    th2 = th_ap + ((math.atan2(c2[1] - cen[1], c2[0] - cen[0]) - th_ap + PI) % (2 * PI) - PI)
    half = max(abs(th1 - th_ap), abs(th2 - th_ap), 1e-6)
    return ParametricArcSurface(
        center=ce, r=r,
        th_apex=torch.as_tensor(th_ap, dtype=dtype, device=device),
        half_span=torch.as_tensor(half, dtype=dtype, device=device),
        powers=powers,
        coeffs=(None if coeffs is None else coeffs),
    )
