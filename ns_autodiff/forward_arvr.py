"""Differentiable forward trace for the AR/VR prism (collimation / angular output).

Models the BRIGHT, image-forming bundle: each display point-source fan ray enters the prism,
folds by TIR, and exits to the eye-pupil plane. Uses nearest-surface selection over the four
arc faces and decides refract-vs-TIR from physics (sq1<0 => TIR), following the high-energy
transmitted+TIR path. Low-energy Fresnel ghosts (the engine's partial-reflect branches) are
stray light and are intentionally not traced.

Path length is the champion-recorded BRIGHT event count per source (per_source_topology), so
the trace runs exactly `n_events` nearest-surface steps then free-flights to the pupil plane.
This is the same frozen-topology recipe as the spot-fold twin (forward.py) and reproduces both
the catadioptric refract->tir (2 events) and the enter-fold-exit refract->tir->refract (3
events) designs without over/under-shooting the fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from . import geometry_torch as G
from .element_arvr import PIDX, PLANE_X, build_surfaces_arvr

T_MIN = 1e-6


@dataclass
class ArvrTrace:
    exit_dir: torch.Tensor
    exit_angle: torch.Tensor     # [m] degrees atan2(diry,dirx)
    pupil_y: torch.Tensor        # [m] y at PLANE_X
    energy: torch.Tensor         # [m] delivered energy (b_s+b_p)
    tir_incidence: torch.Tensor  # [k,m] incidence (rad) at TIR events
    critical: torch.Tensor
    reached: torch.Tensor        # [m] bool

    def min_tir_margin_deg(self):
        if self.tir_incidence.numel() == 0:
            return torch.tensor(float("nan"), dtype=self.pupil_y.dtype)
        return torch.rad2deg(self.tir_incidence.min() - self.critical)


def _all_arc_hits(surfaces, O, D, dtype, device):
    ts, hits, normals, valids = [], [], [], []
    for f in ("bottom", "right", "top", "left"):
        s = surfaces[f]
        troots, hit, nrm, fwd = G.ray_circle_roots(O, D, s["center"], s["r"], t_min=T_MIN)
        on = G.on_arc(hit, s["end1"], s["end2"], s["apex"]) & fwd
        for k in (0, 1):
            ts.append(troots[k]); hits.append(hit[k]); normals.append(nrm[k]); valids.append(on[k])
    return (torch.stack(ts, 0), torch.stack(hits, 0),
            torch.stack(normals, 0), torch.stack(valids, 0))


def trace_arvr(p, origins, dirs, n_events, types=None):
    """p[9] genotype; origins[m,2], dirs[m,2]; run n_events nearest-arc steps then propagate
    to the pupil plane. Decisions are PER-RAY (per-ray medium parity, per-ray reflect/refract,
    per-ray energy): one batch may mix interaction kinds, which the spot-fold twin never had.

    `types` is optional. If None, the kind is decided by physics (sq1<0 => TIR, else refract)
    -- the image-forming bright path. If given as a LongTensor [n_events, m] of codes
    (0=refract, 1=tir, 2=partial_reflect), the recorded type drives the interaction (geometry
    still nearest-surface) -- the faithful frozen-topology mode that also reproduces Fresnel
    ghosts and removes near-critical TIR/refract ambiguity. This is the validation gate; the
    refiner uses the physics mode on the bright bundle."""
    surfaces = build_surfaces_arvr(p)
    n = p[PIDX["n"]]
    O = origins.to(p.dtype)
    D = G.normalize(dirs.to(p.dtype))
    m = O.shape[0]
    dtype, device = p.dtype, p.device
    half = torch.full((m,), 0.5, dtype=dtype, device=device)
    b_s, b_p = half.clone(), half.clone()
    inside = torch.zeros(m, dtype=torch.bool, device=device)   # per-ray medium
    tir_inc: List[torch.Tensor] = []
    reached = torch.ones(m, dtype=torch.bool, device=device)

    for step in range(n_events):
        ts, hits, normals, valids = _all_arc_hits(surfaces, O, D, dtype, device)
        inf = torch.tensor(float("inf"), dtype=dtype, device=device)
        cost = torch.where(valids, ts.detach(), inf)
        sel = torch.argmin(cost, dim=0)
        gi = sel.view(1, m, 1)
        hit = torch.gather(hits, 0, gi.expand(1, m, 2))[0]
        normal = torch.gather(normals, 0, gi.expand(1, m, 2))[0]
        reached = reached & torch.gather(valids, 0, sel.view(1, m))[0]

        n1 = torch.where(inside, n.expand(m), (1.0 / n).expand(m))   # per-ray index ratio
        D_out, cos1, cos2, tir_phys = G.snell_refract(D, normal, n1)
        Rs, Rp = G.fresnel_RsRp(cos1, cos2, n1)
        D_ref = G.normalize(G.reflect(D, normal))
        D_tr = G.normalize(D_out)

        if types is None:
            reflects = tir_phys                      # physics: TIR reflects, else transmit
            is_refract = ~tir_phys
            energy_factor_s = torch.where(tir_phys, torch.ones_like(Rs), 1.0 - Rs)
            energy_factor_p = torch.where(tir_phys, torch.ones_like(Rp), 1.0 - Rp)
        else:
            code = types[step]                        # 0 refract, 1 tir, 2 partial_reflect
            is_refract = (code == 0)
            reflects = (code != 0)                    # tir or partial_reflect reflect
            is_pr = (code == 2)
            energy_factor_s = torch.where(is_refract, 1.0 - Rs, torch.where(is_pr, Rs, torch.ones_like(Rs)))
            energy_factor_p = torch.where(is_refract, 1.0 - Rp, torch.where(is_pr, Rp, torch.ones_like(Rp)))

        D = torch.where(reflects.unsqueeze(-1), D_ref, D_tr)
        O = hit
        b_s = b_s * energy_factor_s
        b_p = b_p * energy_factor_p
        inside = torch.where(is_refract, ~inside, inside)   # medium toggles only on refraction
        tir_mask = tir_phys if types is None else (types[step] == 1)
        if bool(tir_mask.any()):
            tir_inc.append(torch.where(tir_mask, torch.acos(cos1.clamp(-1.0, 1.0)),
                                       torch.full((m,), 1e3, dtype=dtype, device=device)))

    xs = torch.as_tensor(PLANE_X, dtype=dtype, device=device)
    dx = D[..., 0]
    t_s = (xs - O[..., 0]) / torch.where(dx.abs() < G.EPS, torch.full_like(dx, G.EPS), dx)
    reached = reached & (t_s > 0)
    pupil_y = O[..., 1] + t_s * D[..., 1]
    exit_angle = torch.rad2deg(torch.atan2(D[..., 1], D[..., 0]))
    energy = b_s + b_p
    tir_stack = (torch.stack(tir_inc, 0) if tir_inc
                 else torch.empty(0, m, dtype=dtype, device=device))
    return ArvrTrace(exit_dir=D, exit_angle=exit_angle, pupil_y=pupil_y, energy=energy,
                     tir_incidence=tir_stack, critical=G.critical_angle(n), reached=reached)
