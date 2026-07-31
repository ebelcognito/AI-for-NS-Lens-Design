"""The differentiable forward trace.

A faithful, differentiable reimplementation of the non-sequential JS tracer for ONE frozen
topology. The key fidelity point (learned the hard way): the face labels A/B/T/D in a
recorded signature are a *positional* classification of the hit point (x<cx -> A, x>cx -> B,
y at the edges -> T/D), NOT stable identifiers of which physical surface was struck -- a
strongly meniscus element can transmit through the geometric "B" side twice. So the twin
does NOT trust the labels for geometry. Instead it does what the engine does:

  at each step, intersect the ray with ALL four element surfaces (two arcs + two edges),
  take the NEAREST forward hit on its valid span, and decide refract-vs-TIR from physics
  (TIR iff sq1 = 1 - n1^2 (1-cos1^2) < 0).

The champion only freezes (a) how many interaction steps occur and (b) -- for validation --
the expected type/label at each step. The nearest-surface argmin and the TIR test are
locally constant within a fixed topology, so the trace is differentiable: the gradient
flows through the selected surface's geometry and the smooth Snell/Fresnel arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from . import geometry_torch as G
from .element_torch import ELEM_H, PIDX, X_SENSOR, build_surfaces, input_ray

T_MIN = 1e-6


@dataclass
class TraceResult:
    y: torch.Tensor              # sensor y per ray [m]
    energy: torch.Tensor         # delivered energy per ray [m] (b_s + b_p)
    tir_incidence: torch.Tensor  # incidence angle (rad) at every TIR event [k, m]
    critical: torch.Tensor       # scalar critical angle (rad)
    reached: torch.Tensor        # bool [m]
    event_kinds: List[str]       # 'refract'/'tir' actually taken (physics-decided), for audit

    def min_tir_margin_deg(self) -> torch.Tensor:
        if self.tir_incidence.numel() == 0:
            return torch.tensor(float("nan"), dtype=self.y.dtype)
        return torch.rad2deg(self.tir_incidence.min() - self.critical)


def _all_surface_hits(surfaces, O, D, dtype, device):
    """Enumerate every intersection CANDIDATE the ray could take next: both circle roots
    for each arc (valid only if on the real arc, via the chord-side test) and one hit per
    flat edge (valid only if within the segment). Returns stacked, all-FINITE
    (t_raw[k,m], hit[k,m,2], normal[k,m,2], valid[k,m] bool). Inf is deliberately kept out
    of every differentiable tensor; the caller forms a detached inf-masked cost for argmin
    so backward never sees 0*inf."""
    ts, hits, normals, valids = [], [], [], []
    for f in ("A", "B"):
        s = surfaces[f]
        troots, hit, nrm, fwd = G.ray_circle_roots(O, D, s["center"], s["r"], t_min=T_MIN)
        on = G.on_arc(hit, s["end1"], s["end2"], s["apex"]) & fwd
        for k in (0, 1):
            ts.append(troots[k]); hits.append(hit[k]); normals.append(nrm[k]); valids.append(on[k])
    for f in ("T", "D"):
        s = surfaces[f]
        t, hit, nrm, fwd = G.ray_segment_intersect(O, D, s["a"], s["b"], t_min=T_MIN)
        e = s["b"] - s["a"]
        sp = ((hit - s["a"]) * e).sum(-1) / (e * e).sum().clamp_min(G.EPS)
        valid = (sp >= -1e-6) & (sp <= 1.0 + 1e-6) & fwd
        ts.append(t); hits.append(hit); normals.append(nrm); valids.append(valid)
    return (torch.stack(ts, 0), torch.stack(hits, 0),
            torch.stack(normals, 0), torch.stack(valids, 0))   # [k,m],[k,m,2],[k,m,2],[k,m]


def trace(p: torch.Tensor, topo) -> TraceResult:
    """p: tensor[9] (PARAM_NAMES order). topo: FrozenTopology. Runs len(topo.events)
    nearest-surface interactions, then propagates to the sensor plane."""
    surfaces = build_surfaces(p)
    n = p[PIDX["n"]]
    O, D = input_ray(p, topo.dys)
    D = G.normalize(D)
    m = O.shape[0]
    dtype, device = p.dtype, p.device
    half = torch.full((m,), 0.5, dtype=dtype, device=device)
    b_s, b_p = half.clone(), half.clone()
    inside = False
    tir_inc: List[torch.Tensor] = []
    kinds: List[str] = []
    reached = torch.ones(m, dtype=torch.bool, device=device)

    for _step in range(len(topo.events)):
        ts, hits, normals, valids = _all_surface_hits(surfaces, O, D, dtype, device)
        # detached, inf-masked cost: used ONLY to pick the nearest valid surface (argmin
        # has no backward), so no inf ever enters a differentiable tensor.
        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        cost = torch.where(valids, ts.detach(), inf)        # [k,m]
        sel = torch.argmin(cost, dim=0)                     # [m]
        gi = sel.view(1, m, 1)
        hit = torch.gather(hits, 0, gi.expand(1, m, 2))[0]  # [m,2] (finite)
        normal = torch.gather(normals, 0, gi.expand(1, m, 2))[0]
        sel_valid = torch.gather(valids, 0, sel.view(1, m))[0]
        reached = reached & sel_valid

        n1 = (1.0 / n) if not inside else n
        n1_t = n1.expand(m) if torch.is_tensor(n1) else torch.full((m,), float(n1),
                                                                   dtype=dtype, device=device)
        D_out, cos1, cos2, tir = G.snell_refract(D, normal, n1_t)
        is_tir = bool(tir.any())                            # frozen topology -> uniform across rays
        if is_tir:
            tir_inc.append(torch.acos(cos1.clamp(-1.0, 1.0)))
            D = G.normalize(G.reflect(D, normal))
            O = hit
            kinds.append("tir")
        else:
            Rs, Rp = G.fresnel_RsRp(cos1, cos2, n1_t)
            b_s = b_s * (1.0 - Rs)
            b_p = b_p * (1.0 - Rp)
            D = G.normalize(D_out)
            O = hit
            inside = not inside
            kinds.append("refract")

    xs = torch.as_tensor(X_SENSOR, dtype=dtype, device=device)
    dx = D[..., 0]
    t_s = (xs - O[..., 0]) / torch.where(dx.abs() < G.EPS, torch.full_like(dx, G.EPS), dx)
    reached = reached & (t_s > 0)
    y = O[..., 1] + t_s * D[..., 1]
    energy = b_s + b_p
    tir_stack = (torch.stack(tir_inc, 0) if tir_inc
                 else torch.empty(0, m, dtype=dtype, device=device))
    return TraceResult(y=y, energy=energy, tir_incidence=tir_stack,
                       critical=G.critical_angle(n), reached=reached, event_kinds=kinds)
