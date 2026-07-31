"""Differentiable freeform (conic + polynomial) surface intersection for the 2-D twin.

This is the ONE new piece of machinery the freeform extension needs. Everything else in
the tracer (Snell/Fresnel/TIR/the nearest-surface argmin) is unchanged: a freeform surface
just has to hand back the same (t, hit, normal, valid) tuple that `ray_circle_roots` /
`ray_segment_intersect` do in geometry_torch.py.

Why this file exists
--------------------
A circle/line gives a closed-form root. A freeform sag curve does not -- the ray-surface
equation F(t)=0 is nonlinear. So we:

  1. ENUMERATE: scan the ray across the surface's aperture (h is linear in t, so the scan
     domain is bounded), detect sign changes of F, and take the NEAREST forward bracket.
     This is "per-surface root enumeration": a wiggly curve can be crossed several times;
     we find the first physical hit and occlude the rest.
  2. POLISH (detached): Newton on the chosen bracket under no_grad -> t*.
  3. DIFFERENTIATE (one live step): t = t*_detached - F(t*)/F'(t*) recomputed with the
     grad-carrying params. That single step makes t carry the EXACT first-order gradient
     via the implicit function theorem (dt/dp = -F_p / F_t at the root), with no unrolling
     of the Newton iterations into the autograd graph.

Surface model (local tilted frame)
----------------------------------
Each freeform surface stores a vertex `v` (apex), an outward axis `axis` = unit(apex-center),
a transverse `tang` = perp(axis), curvature `c`, conic `k`, and polynomial `(powers, coeffs)`.
A surface point at transverse height h sits at axis-coordinate
    sag(h) = c h^2 / (1 + sqrt(1 - (1+k) c^2 h^2))  +  sum_j coeffs[j] h^powers[j].
With c=-1/R, k=0, coeffs=[] this reproduces a circular arc's apex cap EXACTLY (see
validate_freeform.py), so a freeform surface initializes *at* the discovered arc.

The intersection condition for ray P(t)=O+tD is
    F(t) = axis . (P(t) - v) - sag( tang . (P(t) - v) ) = 0
and the (unnormalized) surface normal is grad_P F = axis - sag'(h) * tang.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from .geometry_torch import EPS, _orient_against, normalize

T_MIN = 1e-6
TANGENT_FLOOR = 1e-7   # |dF/dt| below this => grazing/near-tangent: treat as topology edge


@dataclass
class FreeformSurface:
    """A conic+polynomial sag curve in a local tilted frame. All tensors share dtype/device
    and may carry grad. `powers`/`coeffs` are the polynomial deviation from the conic base
    (empty => pure conic == arc)."""
    v: torch.Tensor          # vertex / apex [2]
    axis: torch.Tensor       # unit outward axis at the vertex [2]
    tang: torch.Tensor       # unit transverse direction [2]
    c: torch.Tensor          # curvature (=-1/R reproduces the arc cap) []
    k: torch.Tensor          # conic constant []
    h_min: torch.Tensor      # aperture span in transverse coord []
    h_max: torch.Tensor      # []
    powers: tuple = ()       # polynomial powers, e.g. (2,3,4)
    coeffs: torch.Tensor = None  # matching coefficients [len(powers)] or None
    kind: str = "freeform"


def sag(h, surf: FreeformSurface):
    """Conic + polynomial sag(h) and its derivative sag'(h), both differentiable. `h` is a
    tensor of transverse heights; returns (z[...], dz[...])."""
    c, k = surf.c, surf.k
    A = (1.0 + k) * c * c
    u = (1.0 - A * h * h).clamp_min(0.0)          # clamp only for a finite sqrt
    w = torch.sqrt(u + EPS)
    denom = 1.0 + w
    z = c * h * h / denom
    # dz/dh for the conic term (analytic; matches the clamp-protected w):
    wprime = (-A * h) / w.clamp_min(EPS)
    dz = (2.0 * c * h * denom - c * h * h * wprime) / (denom * denom)
    if surf.powers and surf.coeffs is not None:
        for j, pw in enumerate(surf.powers):
            a = surf.coeffs[j]
            z = z + a * h ** pw
            dz = dz + a * pw * h ** (pw - 1)
    return z, dz


def _F_and_dF(t, O, D, surf: FreeformSurface):
    """Residual F(t) and dF/dt for the batched rays at parameter t[...]. Also returns h(t)."""
    P = O + t.unsqueeze(-1) * D                    # [...,2]
    rel = P - surf.v
    s_axis = (rel * surf.axis).sum(-1)             # axis coord of the ray point
    h = (rel * surf.tang).sum(-1)                  # transverse coord
    z, dz = sag(h, surf)
    F = s_axis - z
    ad = (D * surf.axis).sum(-1)                   # d(s_axis)/dt
    hd = (D * surf.tang).sum(-1)                   # dh/dt
    dF = ad - dz * hd
    return F, dF, h


def ray_freeform_roots(O, D, surf: FreeformSurface, *, n_scan: int = 64, n_newton: int = 12,
                       t_min: float = T_MIN):
    """Nearest forward valid intersection of rays O+tD (unit D) with a freeform surface.

    Returns (t[m], hit[m,2], normal[m,2], valid[m]) -- the per-surface candidate that the
    caller's global nearest-surface argmin then competes across surfaces. Enumeration ->
    nearest is folded in here (one returned hit per surface per ray), which is equivalent to
    the circle's two-root + argmin path but generalizes to many crossings.

    The bracket/Newton polish runs detached (no_grad); a single live Newton step then injects
    the exact IFT gradient. `n_scan` sets how finely the aperture is swept to bracket roots
    (raise it for very wiggly surfaces); near-tangent roots are marked invalid (topology edge).
    """
    O = O if O.dim() == 2 else O.unsqueeze(0)
    D = D if D.dim() == 2 else D.unsqueeze(0)
    m = O.shape[0]
    dtype, device = O.dtype, O.device
    relO = O - surf.v
    h0 = (relO * surf.tang).sum(-1)                # [m]  h = h0 + t*hd
    hd = (D * surf.tang).sum(-1)                   # [m]

    # --- enumerate + polish, all detached --------------------------------------------- #
    with torch.no_grad():
        hmin = torch.minimum(surf.h_min, surf.h_max)        # scalar
        hmax = torch.maximum(surf.h_min, surf.h_max)
        hd_safe = torch.where(hd.abs() < EPS, torch.full_like(hd, EPS), hd)
        # Scan only the FORWARD slice of the aperture per ray: t>t_min <=> h past h_b=h(t_min).
        # (Doing this in h-space, where the grid is bounded, instead of filtering t afterwards,
        # is what makes near-grazing rays -- tiny hd, so one h-step spans a huge t-range that
        # straddles the origin -- bracket correctly.)
        h_b = h0 + t_min * hd                                # [m] transverse coord at t=t_min
        fwd_high = hd > 0                                    # forward => h on this side of h_b
        lo_f = torch.where(fwd_high, torch.maximum(hmin.expand(m), h_b), hmin.expand(m))
        hi_f = torch.where(fwd_high, hmax.expand(m), torch.minimum(hmax.expand(m), h_b))
        nonempty = hi_f > lo_f + 1e-12                       # [m] is there a forward aperture?
        grid = torch.linspace(0.0, 1.0, n_scan, dtype=dtype, device=device)
        Hg = lo_f.unsqueeze(0) + grid.unsqueeze(1) * (hi_f - lo_f).unsqueeze(0)  # [n_scan, m]
        Tg = (Hg - h0.unsqueeze(0)) / hd_safe.unsqueeze(0)   # all >= ~t_min by construction
        Fg, _, _ = _F_and_dF(Tg, O.unsqueeze(0).expand(n_scan, m, 2),
                             D.unsqueeze(0).expand(n_scan, m, 2), surf)
        sgn = torch.sign(Fg)
        cross = (sgn[:-1] * sgn[1:] < 0)                     # [n_scan-1, m]  (all forward)
        t_lo = Tg[:-1]; t_hi = Tg[1:]
        big = torch.full_like(Tg[:-1], float("inf"))
        # nearest forward bracket per ray = smallest t among crossings
        bracket_t = torch.where(cross, torch.minimum(t_lo, t_hi), big)   # [n_scan-1, m]
        has = cross.any(0) & nonempty                                    # [m]
        sel = torch.argmin(bracket_t, dim=0)                             # [m]
        a_t = torch.gather(torch.minimum(t_lo, t_hi), 0, sel.view(1, m))[0]
        b_t = torch.gather(torch.maximum(t_lo, t_hi), 0, sel.view(1, m))[0]
        # safeguarded bisection-Newton (rtsafe): orient [lo,hi] so F(lo)<0<F(hi), then take a
        # Newton step when it stays in-bracket and is reducing fast, else bisect. Robust to the
        # near-tangent / multi-root geometry a wiggly sag produces.
        Fa, _, _ = _F_and_dF(a_t, O, D, surf)
        Fb, _, _ = _F_and_dF(b_t, O, D, surf)
        swap = Fa > 0
        lo = torch.where(swap, b_t, a_t)        # F(lo) <= 0
        hi = torch.where(swap, a_t, b_t)        # F(hi) >= 0
        t = 0.5 * (lo + hi)
        for _ in range(n_newton):
            F, dF, _ = _F_and_dF(t, O, D, surf)
            dF_s = torch.where(dF.abs() < EPS, torch.full_like(dF, EPS), dF)
            t_newton = t - F / dF_s
            use_bisect = ((t_newton - hi) * (t_newton - lo) > 0)        # Newton left the bracket
            t = torch.where(use_bisect, 0.5 * (lo + hi), t_newton)
            Fnew, _, _ = _F_and_dF(t, O, D, surf)
            neg = Fnew < 0
            lo = torch.where(neg, t, lo)
            hi = torch.where(neg, hi, t)
        t_det = t

    # --- one LIVE Newton step => exact IFT gradient ----------------------------------- #
    F, dF, h = _F_and_dF(t_det, O, D, surf)
    # clamp |dF/dt| away from zero WITHOUT changing its sign, so the differentiable divide
    # stays finite at near-tangent roots (those rays are flagged invalid below anyway).
    dF_safe = torch.where(dF.abs() < TANGENT_FLOOR,
                          torch.where(dF >= 0, torch.full_like(dF, TANGENT_FLOOR),
                                      torch.full_like(dF, -TANGENT_FLOOR)), dF)
    t = t_det - F / dF_safe
    hit = O + t.unsqueeze(-1) * D

    # normal = grad_P F = axis - sag'(h) * tang, oriented to oppose D
    _, dz = sag(h, surf)
    n_geom = surf.axis.unsqueeze(0) - dz.unsqueeze(-1) * surf.tang.unsqueeze(0)
    normal = _orient_against(normalize(n_geom), D)

    # validity: a forward bracket was found, h within aperture, not grazing
    in_aper = (h >= torch.minimum(surf.h_min, surf.h_max) - 1e-6) & \
              (h <= torch.maximum(surf.h_min, surf.h_max) + 1e-6)
    valid = has & (t > t_min) & in_aper & (hd.abs() >= EPS) & (dF.abs() >= TANGENT_FLOOR)
    return t, hit, normal, valid
