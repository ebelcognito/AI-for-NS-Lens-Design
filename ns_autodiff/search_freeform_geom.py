"""Shared numpy freeform geometry for the discovering search tracers (fold + prism).

Vectorized-over-rays conic+polynomial surface intersection, a faithful numpy port of
geometry_freeform.ray_freeform_roots (validated there against the autodiff twin and the JS
oracle). Powers are stored per surface so different configs can use different aspheric orders.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def circumcenter(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
    ux = (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
    uy = (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
    c = np.array([ux, uy]); return c, float(np.hypot(*(c - np.asarray(p1))))


def sag(h, c, k, powers, coeffs):
    A = (1.0 + k) * c * c
    u = np.clip(1.0 - A * h * h, 0.0, None); w = np.sqrt(u + 1e-12)
    denom = 1.0 + w; z = c * h * h / denom
    wp = (-A * h) / np.maximum(w, 1e-12)
    dz = (2.0 * c * h * denom - c * h * h * wp) / (denom * denom)
    for pw, a in zip(powers, coeffs):
        z = z + a * h ** pw; dz = dz + a * pw * h ** (pw - 1)
    return z, dz


def freeform_surface(apex, center, r, end1, end2, coeffs, powers=(2, 4)):
    apex = np.asarray(apex, float); center = np.asarray(center, float)
    axis = apex - center; axis = axis / max(np.hypot(*axis), EPS)
    tang = np.array([-axis[1], axis[0]])
    c = -1.0 / max(r, EPS)
    h1 = (np.asarray(end1, float) - apex) @ tang
    h2 = (np.asarray(end2, float) - apex) @ tang
    return dict(kind="ff", v=apex, axis=axis, tang=tang, c=c, k=0.0,
                coeffs=np.asarray(coeffs, float), powers=tuple(powers),
                h_min=min(h1, h2), h_max=max(h1, h2))


def roots_ff(O, D, s, *, n_scan=64, n_bisect=40, t_min=1e-6):
    """Nearest forward valid intersection of rays O[m,2],D[m,2] with a freeform surface."""
    m = O.shape[0]
    v, axis, tang = s["v"], s["axis"], s["tang"]
    pw, cf = s["powers"], s["coeffs"]
    h0 = (O - v) @ tang; hd = D @ tang
    a0 = (O - v) @ axis; ad = D @ axis
    hd_safe = np.where(np.abs(hd) < EPS, EPS, hd)
    hmin, hmax = s["h_min"], s["h_max"]
    h_b = h0 + t_min * hd
    fwd_high = hd > 0
    lo_f = np.where(fwd_high, np.maximum(hmin, h_b), hmin)
    hi_f = np.where(fwd_high, hmax, np.minimum(hmax, h_b))
    nonempty = hi_f > lo_f + 1e-12
    g = np.linspace(0.0, 1.0, n_scan)[:, None]
    Hg = lo_f[None, :] + g * (hi_f - lo_f)[None, :]
    z, _ = sag(Hg, s["c"], s["k"], pw, cf)
    Tg = (Hg - h0[None, :]) / hd_safe[None, :]
    Fg = a0[None, :] + Tg * ad[None, :] - z
    sgn = np.sign(Fg)
    cross = (sgn[:-1] * sgn[1:] < 0)
    has = cross.any(0) & nonempty
    t_lo, t_hi = Tg[:-1], Tg[1:]
    cand = np.where(cross, np.minimum(t_lo, t_hi), np.inf)
    sel = np.argmin(cand, axis=0); ii = np.arange(m)
    a_t = np.minimum(t_lo, t_hi)[sel, ii]; b_t = np.maximum(t_lo, t_hi)[sel, ii]

    def F_at(t):
        h = h0 + t * hd
        z2, _ = sag(h, s["c"], s["k"], pw, cf)
        return a0 + t * ad - z2
    Fa = F_at(a_t)
    swap = Fa > 0
    lo = np.where(swap, b_t, a_t); hi = np.where(swap, a_t, b_t)
    t = 0.5 * (lo + hi)
    for _ in range(n_bisect):
        Ft = F_at(t); neg = Ft < 0
        lo = np.where(neg, t, lo); hi = np.where(neg, hi, t); t = 0.5 * (lo + hi)
    h = h0 + t * hd
    _, dz = sag(h, s["c"], s["k"], pw, cf)
    hit = O + t[:, None] * D
    n_geom = axis[None, :] - dz[:, None] * tang[None, :]
    n_geom = n_geom / np.maximum(np.linalg.norm(n_geom, axis=1, keepdims=True), EPS)
    sign = -np.sign((n_geom * D).sum(1)); sign = np.where(sign == 0, 1.0, sign)
    normal = n_geom * sign[:, None]
    in_aper = (h >= min(hmin, hmax) - 1e-6) & (h <= max(hmin, hmax) + 1e-6)
    valid = has & (t > t_min) & in_aper & (np.abs(hd) >= EPS)
    return t, hit, normal, valid


def roots_line(O, D, a, b, *, t_min=1e-6):
    e = b - a
    denom = D[:, 0] * (-e[1]) - D[:, 1] * (-e[0])
    denom_s = np.where(np.abs(denom) < EPS, EPS, denom)
    diff = a[None, :] - O
    t = (diff[:, 0] * (-e[1]) - diff[:, 1] * (-e[0])) / denom_s
    hit = O + t[:, None] * D
    sp = ((hit - a[None, :]) * e[None, :]).sum(1) / max(e @ e, EPS)
    n_geom = np.array([-e[1], e[0]]); n_geom = n_geom / max(np.hypot(*n_geom), EPS)
    n_geom = np.broadcast_to(n_geom, O.shape).copy()
    sign = -np.sign((n_geom * D).sum(1)); sign = np.where(sign == 0, 1.0, sign)
    normal = n_geom * sign[:, None]
    valid = (t > t_min) & (np.abs(denom) >= EPS) & (sp >= -1e-6) & (sp <= 1.0 + 1e-6)
    return t, hit, normal, valid
