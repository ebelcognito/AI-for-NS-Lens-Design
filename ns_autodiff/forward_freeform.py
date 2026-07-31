"""The differentiable forward trace with FREEFORM (conic+polynomial) curved faces.

A drop-in sibling of forward.py: same frozen-topology, nearest-forward, physics-decided
trace, but the two curved faces A/B are FreeformSurfaces (geometry_freeform.ray_freeform_roots)
instead of circular arcs. With no polynomial terms (coeffs=None) it reproduces forward.trace
bit-for-bit -- that is the forward-match gate in validate_freeform_forward.py.

What changes vs forward.py
--------------------------
* Each curved face contributes ONE candidate (its nearest-forward valid crossing -- enumeration
  is folded into ray_freeform_roots) instead of the circle's two roots. Per-surface-nearest then
  the global argmin = global nearest, so the selection is identical.
* The trace also records, per step, the selected hit point and a positional face label (ported
  from sequence_tools.make_single_element_face_classifier), so the twin can compute the same
  typed signature the JS oracle does -- needed for drift detection / honest re-score once the
  oracle can no longer trace the (freeform) geometry.

Everything else -- Snell/Fresnel/TIR/the s,p energy channels/the sensor propagation -- is the
SAME code path as forward.py (imported from geometry_torch), unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from . import geometry_torch as G
from .element_torch import ELEM_H, PIDX, X_SENSOR, build_surfaces, input_ray
from .element_freeform import arc_to_freeform
from .geometry_freeform import ray_freeform_roots

T_MIN = 1e-6
FACES = ("A", "B", "T", "D")


@dataclass
class TraceResultFF:
    y: torch.Tensor
    energy: torch.Tensor
    tir_incidence: torch.Tensor
    critical: torch.Tensor
    reached: torch.Tensor
    event_kinds: List[str]
    hit_path: torch.Tensor          # [steps, m, 2] selected hit per step (for signatures)
    sel_face: torch.Tensor          # [steps, m] int index into FACES of the selected surface

    def min_tir_margin_deg(self) -> torch.Tensor:
        if self.tir_incidence.numel() == 0:
            return torch.tensor(float("nan"), dtype=self.y.dtype)
        return torch.rad2deg(self.tir_incidence.min() - self.critical)


def build_surfaces_freeform(p, coeffs=None, powers=()):
    """Element with the two curved faces upgraded to FreeformSurfaces (T/D stay flat lines).
    `coeffs` is None (pure conic == arc) or a dict {'A': tensor[d], 'B': tensor[d]} of the
    polynomial-deviation coefficients for `powers`."""
    base = build_surfaces(p)
    cA = None if coeffs is None else coeffs.get("A")
    cB = None if coeffs is None else coeffs.get("B")
    return {
        "A": arc_to_freeform(base["A"], powers=powers, coeffs=cA),
        "B": arc_to_freeform(base["B"], powers=powers, coeffs=cB),
        "T": base["T"], "D": base["D"],
    }


def _all_surface_hits_ff(surfaces, O, D):
    """4 candidates: freeform A, freeform B, line T, line D. Each returns its nearest-forward
    valid crossing -> (t[4,m], hit[4,m,2], normal[4,m,2], valid[4,m])."""
    ts, hits, normals, valids = [], [], [], []
    for f in ("A", "B"):
        t, hit, nrm, val = ray_freeform_roots(O, D, surfaces[f], t_min=T_MIN)
        ts.append(t); hits.append(hit); normals.append(nrm); valids.append(val)
    for f in ("T", "D"):
        s = surfaces[f]
        t, hit, nrm, fwd = G.ray_segment_intersect(O, D, s["a"], s["b"], t_min=T_MIN)
        e = s["b"] - s["a"]
        sp = ((hit - s["a"]) * e).sum(-1) / (e * e).sum().clamp_min(G.EPS)
        valid = (sp >= -1e-6) & (sp <= 1.0 + 1e-6) & fwd
        ts.append(t); hits.append(hit); normals.append(nrm); valids.append(valid)
    return (torch.stack(ts, 0), torch.stack(hits, 0),
            torch.stack(normals, 0), torch.stack(valids, 0))


def trace_ff(p: torch.Tensor, topo, *, coeffs=None, powers=()) -> TraceResultFF:
    """Freeform forward trace over the frozen topology (len(topo.events) steps). p: tensor[9].
    coeffs/powers add freeform DOF to faces A,B (None => exact arcs => matches forward.trace)."""
    surfaces = build_surfaces_freeform(p, coeffs=coeffs, powers=powers)
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
    hit_path: List[torch.Tensor] = []
    sel_face: List[torch.Tensor] = []
    reached = torch.ones(m, dtype=torch.bool, device=device)

    for _step in range(len(topo.events)):
        ts, hits, normals, valids = _all_surface_hits_ff(surfaces, O, D)
        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        cost = torch.where(valids, ts.detach(), inf)
        sel = torch.argmin(cost, dim=0)                     # [m]
        gi = sel.view(1, m, 1)
        hit = torch.gather(hits, 0, gi.expand(1, m, 2))[0]
        normal = torch.gather(normals, 0, gi.expand(1, m, 2))[0]
        sel_valid = torch.gather(valids, 0, sel.view(1, m))[0]
        reached = reached & sel_valid

        n1 = (1.0 / n) if not inside else n
        n1_t = n1.expand(m) if torch.is_tensor(n1) else torch.full((m,), float(n1),
                                                                   dtype=dtype, device=device)
        D_out, cos1, cos2, tir = G.snell_refract(D, normal, n1_t)
        is_tir = bool(tir.any())
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
        hit_path.append(hit)
        sel_face.append(sel)

    xs = torch.as_tensor(X_SENSOR, dtype=dtype, device=device)
    dx = D[..., 0]
    t_s = (xs - O[..., 0]) / torch.where(dx.abs() < G.EPS, torch.full_like(dx, G.EPS), dx)
    reached = reached & (t_s > 0)
    y = O[..., 1] + t_s * D[..., 1]
    energy = b_s + b_p
    tir_stack = (torch.stack(tir_inc, 0) if tir_inc
                 else torch.empty(0, m, dtype=dtype, device=device))
    return TraceResultFF(y=y, energy=energy, tir_incidence=tir_stack,
                         critical=G.critical_angle(n), reached=reached, event_kinds=kinds,
                         hit_path=torch.stack(hit_path, 0) if hit_path else torch.empty(0, m, 2),
                         sel_face=torch.stack(sel_face, 0) if sel_face
                         else torch.empty(0, m, dtype=torch.long))


# --------------------------------------------------------------------------- #
#  twin-side positional signature (faithful port of sequence_tools)
# --------------------------------------------------------------------------- #
def face_label(x, y, cx, elem_h=ELEM_H, eps=1e-3):
    """Positional face label of a hit point, matching make_single_element_face_classifier:
    y<=-H -> T, y>=H -> D, else x<cx -> A, x>cx -> B."""
    if y <= -elem_h + eps:
        return "T"
    if y >= elem_h - eps:
        return "D"
    return "A" if x < cx else "B"


def ray_signatures(tr: TraceResultFF, cx: float):
    """Per-ray typed signature [(kind, face), ...] from the recorded hit path -- the same
    object signature_of builds from the oracle's raySequences, but computed in the twin."""
    steps = tr.hit_path.shape[0]
    m = tr.y.shape[0]
    hp = tr.hit_path.detach().cpu().numpy()
    out = []
    for i in range(m):
        sig = tuple((tr.event_kinds[s], face_label(float(hp[s, i, 0]), float(hp[s, i, 1]), cx))
                    for s in range(steps))
        out.append(sig)
    return out


def dominant_signature_twin(tr: TraceResultFF, cx: float):
    """Energy-weighted dominant typed signature over rays that reached the sensor."""
    sigs = ray_signatures(tr, cx)
    e = tr.energy.detach().cpu().numpy()
    reached = tr.reached.detach().cpu().numpy()
    by = {}
    for i, sig in enumerate(sigs):
        if not reached[i]:
            continue
        by[sig] = by.get(sig, 0.0) + float(e[i])
    if not by:
        return None, sigs
    return max(by, key=by.get), sigs
