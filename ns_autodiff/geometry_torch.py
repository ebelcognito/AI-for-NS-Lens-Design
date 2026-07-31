"""Differentiable 2-D geometric-optics primitives (PyTorch, float64).

Every function here is a faithful, autograd-friendly reimplementation of the
corresponding operation in the JS engine (ray-optics/src/core), with no Python/torch
control flow that depends on the *values* of the optimized parameters -- the only
branching is the frozen topology (which surface, refract vs reflect), decided once by
the champion and passed in as data. That is what makes the whole forward trace
differentiable: within a fixed topology the objective is smooth (see
agent_context/ns_differentiable_refinement/00_differentiable_refinement.md).

Conventions
-----------
* A ray is (origin O[...,2], direction D[...,2]) with D a unit vector.
* All angles in radians internally.
* Normals are returned ORIENTED TO OPPOSE the incoming ray (n . D < 0), exactly as the
  JS `refract`/`reflect` expect (cos1 = -n . D >= 0). Orientation uses a detached sign
  flip, which is locally constant and so leaves the gradient of the magnitude intact.

Matches (line refs into the vendored engine):
* Snell vector form + TIR test : BaseGlass.js:194-214, 290-291
* Fresnel R_s / R_p             : BaseGlass.js:234-235
* reflection direction         : BaseGlass.js:203, 246
"""
from __future__ import annotations

import torch

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  small vector helpers
# --------------------------------------------------------------------------- #
def _dot(a, b):
    return (a * b).sum(dim=-1)


def normalize(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(EPS)


def circumcenter(p1, p2, p3):
    """Center of the circle through three 2-D points (the JS engine builds an arc face
    as the circle through path[i], path[i+1] (the 'arc' control point) and path[i+2];
    see Glass.js:520-525 which intersects the two perpendicular bisectors -- this closed
    form is the same circumcenter, but differentiable). Returns (center[...,2], r[...]).
    """
    ax, ay = p1[..., 0], p1[..., 1]
    bx, by = p2[..., 0], p2[..., 1]
    cx, cy = p3[..., 0], p3[..., 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = torch.stack([ux, uy], dim=-1)
    r = torch.linalg.norm(center - p1, dim=-1)
    return center, r


# --------------------------------------------------------------------------- #
#  ray <-> surface intersection
# --------------------------------------------------------------------------- #
def ray_circle_roots(O, D, center, r, *, t_min=1e-6):
    """Both intersections of unit-direction rays O+tD with a circle. Returns
    (t[2,...], hit[2,...,2], normal[2,...,2], fwd[2,...]) -- all FINITE (no inf masking,
    so backward never hits 0*inf); `fwd` is the boolean mask t > t_min. The normal is the
    radial geometric normal ORIENTED to oppose D. The caller decides which root is the real
    arc hit (chord-side test) and which surface is nearest, mirroring the engine, which
    considers both circle intersections (Glass.js:527-561)."""
    oc = O - center
    b = _dot(D, oc)                       # D is unit -> a == 1
    c = _dot(oc, oc) - r * r
    disc_raw = b * b - c
    real = (disc_raw >= 0).unsqueeze(0)   # rays that actually intersect the circle
    disc = disc_raw.clamp_min(0.0)        # clamp only for a finite (differentiable) sqrt
    sq = torch.sqrt(disc + EPS)
    troots = torch.stack([-b - sq, -b + sq], dim=0)        # [2,...]  (finite)
    hit = O.unsqueeze(0) + troots.unsqueeze(-1) * D.unsqueeze(0)
    n_geom = normalize(hit - center)
    normal = _orient_against(n_geom, D.unsqueeze(0).expand_as(n_geom))
    # a clamped (disc<0) root is a spurious closest-approach point, NOT a real hit -> invalid
    return troots, hit, normal, (troots > t_min) & real


def on_arc(hit, end1, end2, apex):
    """Boolean mask: is the circle point `hit` on the actual arc, i.e. on the same side of
    the chord (end1->end2) as the apex (control) point? For the element's modest-sag arcs
    (< 180 deg) this is exactly the arc-vs-chord test the engine performs."""
    def cross(p):
        e = end2 - end1
        d = p - end1
        return e[..., 0] * d[..., 1] - e[..., 1] * d[..., 0]
    s_apex = torch.sign(cross(apex))
    return torch.sign(cross(hit)) == s_apex


def ray_segment_intersect(O, D, a, b, *, t_min=1e-6):
    """Intersect rays with the infinite line through points a,b (flat element edges are
    line segments; for a frozen topology the ray is known to land on the segment, so we
    use the line and let validation confirm). Returns (t, hit, normal, fwd) -- all FINITE;
    `fwd` is t > t_min AND the ray not parallel to the edge. Normal oriented to oppose D,
    matching the JS line-normal construction (Glass.js:610-614)."""
    e = b - a                              # edge direction (not unit)
    # Solve O + tD = a + s e  ->  cross form for t.
    denom_raw = D[..., 0] * (-e[..., 1]) - D[..., 1] * (-e[..., 0])
    denom = torch.where(denom_raw.abs() < EPS, torch.full_like(denom_raw, EPS), denom_raw)
    diff = a - O
    t = (diff[..., 0] * (-e[..., 1]) - diff[..., 1] * (-e[..., 0])) / denom
    hit = O + t.unsqueeze(-1) * D
    n_geom = normalize(torch.stack([-e[..., 1], e[..., 0]], dim=-1).expand_as(O))
    normal = _orient_against(n_geom, D)
    return t, hit, normal, (t > t_min) & (denom_raw.abs() >= EPS)


def _orient_against(n, D):
    """Flip the geometric normal so it opposes D (cos1 = -n.D >= 0). The sign is detached
    (locally constant) so the gradient flows through n's magnitude only."""
    s = -torch.sign(_dot(n, D)).detach()
    s = torch.where(s == 0, torch.ones_like(s), s)
    return n * s.unsqueeze(-1)


# --------------------------------------------------------------------------- #
#  interactions (reflection / refraction / Fresnel)
# --------------------------------------------------------------------------- #
def reflect(D, normal):
    """Specular reflection. With cos1 = -n.D, reflected = D + 2 cos1 n (JS BaseGlass.js:203)."""
    cos1 = -_dot(D, normal)
    return D + 2.0 * cos1.unsqueeze(-1) * normal


def snell_refract(D, normal, n1):
    """Vector-form Snell refraction (BaseGlass.js:196-291).

    n1 = n_incident / n_transmitted (so n1 = 1/n entering glass, n1 = n exiting). Returns
    (D_out[...,2], cos1[...], cos2[...], tir[...]) where tir is a boolean mask of rays
    that would total-internally-reflect (sq1 < 0). For a frozen-topology refract event
    tir should be all-False; for a tir event we ignore D_out and call reflect()."""
    cos1 = -_dot(D, normal)
    sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
    tir = sq1 < 0
    cos2 = torch.sqrt(sq1.clamp_min(0.0) + EPS)
    D_out = n1.unsqueeze(-1) * D + (n1 * cos1 - cos2).unsqueeze(-1) * normal
    return D_out, cos1, cos2, tir


def fresnel_RsRp(cos1, cos2, n1):
    """Unpolarized-input Fresnel power reflectances (BaseGlass.js:234-235). n1 is the
    index ratio used in the JS code; cos1/cos2 are the incidence/transmission cosines."""
    Rs = ((n1 * cos1 - cos2) / (n1 * cos1 + cos2)) ** 2
    Rp = ((n1 * cos2 - cos1) / (n1 * cos2 + cos1)) ** 2
    return Rs, Rp


def critical_angle(n):
    """TIR critical angle (rad) for glass index n into air."""
    return torch.asin((1.0 / n).clamp(max=1.0 - 1e-12))
