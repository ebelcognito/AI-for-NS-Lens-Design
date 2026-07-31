"""Differentiable bright-path forward trace for the AR/VR prism with FREEFORM faces.

This is forward_arvr's recipe (per-ray nearest-face argmin, per-ray refract/TIR + medium parity,
free-flight to the pupil, per-source angular metric, n_events grouping for heterogeneous fold
counts) but the four faces are differentiable PARAMETRIC-ARC surfaces (geometry_prism_ff) so the
8 aspheric coefficients (a2,a4 per face A/B/T/D) are real gradient DOF -- which doc 03 shows is
where the refinement win lives.

Scope (deliberate, matches refine_arvr): the BRIGHT image-forming bundle, dominant path. Fresnel
ghosts are stray light and are not traced here; the honest final score goes back through the
faithful branching tracer (prism_branch.metrics) in the refiner. Base geometry (cx, thick, sags,
tilts, n) is FROZEN; only the aspheric coeffs carry grad in v1 (the fold's "polish the aspheric
on a frozen topology" analog). The per-source bright-path event count (n_events) is frozen once
from prism_branch's deterministic single-path follow -- that IS the frozen topology.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import geometry_torch as G
from . import prism_branch as PB
from .geometry_prism_ff import ray_parametric_arc_roots, surface_from_arc_dict

DTYPE = torch.float64
FACE_ORDER = ("A", "B", "T", "D")

PUPIL_X = PB.PUPIL_X
EYEBOX_HALF = PB.EYEBOX_HALF
SIGMA0 = PB.SIGMA0
N_FAN = PB.N_FAN
SOURCE_PTS = PB.SOURCE_PTS
SOURCE_AIM = PB.SOURCE_AIM
FAN_HALFANGLE = PB.FAN_HALFANGLE
MAX_DEPTH = PB.MAX_DEPTH


# --------------------------------------------------------------------------- #
#  faces + frozen-topology bookkeeping
# --------------------------------------------------------------------------- #
def build_faces(base9, coeffs, *, dtype=DTYPE):
    """4 parametric-arc faces: base geometry from numpy build_arcs (FROZEN), coeffs[8] (grad).
    Returns (faces[list of 4 ParametricArcSurface], n_index float, arcs_np dict)."""
    arcs_np, n = PB.build_arcs(np.asarray(base9, float))
    faces = [surface_from_arc_dict(arcs_np[name], coeffs=coeffs[2 * fi:2 * fi + 2], dtype=dtype)
             for fi, name in enumerate(FACE_ORDER)]
    return faces, float(n), arcs_np


def make_fans(n_fan=N_FAN):
    """The 3 display point-source fans, exactly as prism_branch.trace_faces emits them
    (optionally denser: n_fan > N_FAN sharpens the collimation objective at refine time)."""
    O, D, sid = [], [], []
    for si, (sx, sy) in enumerate(SOURCE_PTS):
        for ang in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, n_fan)):
            O.append([sx, sy]); D.append([math.cos(ang), math.sin(ang)]); sid.append(si)
    return np.array(O, float), np.array(D, float), np.array(sid, int)


def bright_events(arcs_ff_np, n, O0, D0, max_depth=MAX_DEPTH):
    """Deterministic single-path (bright) follow of one ray; returns the interaction count.
    transmit on refraction, reflect on TIR -- the dominant image-forming path. arcs_ff_np must
    carry th_apex/half_span/coeffs (i.e. from prism_branch.build_arcs_ff)."""
    O = np.asarray(O0, float).copy(); D = np.asarray(D0, float).copy()
    inside = False; nev = 0
    for _ in range(max_depth):
        hit = PB.nearest_hit_ff(O, D, arcs_ff_np)
        if hit is None:
            break
        _t, P, normal, _name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal
        else:
            cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal; inside = not inside
        D = D / np.linalg.norm(D); O = P; nev += 1
    return nev


def freeze_topology(x17, n_fan=N_FAN, max_depth=MAX_DEPTH):
    """Freeze the bright-path topology of design x17 (9 base + 8 aspheric): build the fans and
    record each ray's bright-path event count. Returns (origins, dirs, sid, n_events) numpy."""
    arcs_ff, n = PB.build_arcs_ff(np.asarray(x17, float))
    O, D, sid = make_fans(n_fan)
    nev = np.array([bright_events(arcs_ff, n, O[i], D[i], max_depth) for i in range(len(O))], int)
    return O, D, sid, nev


# --------------------------------------------------------------------------- #
#  the frozen-topology differentiable trace (one shared n_events per batch)
# --------------------------------------------------------------------------- #
def _all_face_hits(faces, O, D):
    ts, hits, normals, valids = [], [], [], []
    for s in faces:
        t, hit, nrm, val = ray_parametric_arc_roots(O, D, s)
        ts.append(t); hits.append(hit); normals.append(nrm); valids.append(val)
    return (torch.stack(ts, 0), torch.stack(hits, 0),
            torch.stack(normals, 0), torch.stack(valids, 0))


def trace_prism_ff(faces, n_index, origins, dirs, n_events):
    """n_events nearest-face steps with per-ray refract/TIR + medium parity, then free-flight to
    the pupil plane. Returns (exit_angle[m] deg, energy[m], pupil_y[m], reached[m], tir_inc[k,m],
    critical[])."""
    dtype = faces[0].center.dtype
    O = torch.as_tensor(origins, dtype=dtype)
    D = G.normalize(torch.as_tensor(dirs, dtype=dtype))
    m = O.shape[0]
    n = torch.as_tensor(float(n_index), dtype=dtype)
    half = torch.full((m,), 0.5, dtype=dtype)
    b_s, b_p = half.clone(), half.clone()
    inside = torch.zeros(m, dtype=torch.bool)
    reached = torch.ones(m, dtype=torch.bool)
    tir_inc = []
    inf = torch.tensor(float("inf"), dtype=dtype)

    for _step in range(int(n_events)):
        ts, hits, normals, valids = _all_face_hits(faces, O, D)
        cost = torch.where(valids, ts.detach(), inf)
        sel = torch.argmin(cost, dim=0)
        gi = sel.view(1, m, 1)
        hit = torch.gather(hits, 0, gi.expand(1, m, 2))[0]
        normal = torch.gather(normals, 0, gi.expand(1, m, 2))[0]
        reached = reached & torch.gather(valids, 0, sel.view(1, m))[0]

        n1 = torch.where(inside, n.expand(m), (1.0 / n).expand(m))
        D_out, cos1, cos2, tir = G.snell_refract(D, normal, n1)
        Rs, Rp = G.fresnel_RsRp(cos1, cos2, n1)
        D_ref = G.normalize(G.reflect(D, normal))
        D_tr = G.normalize(D_out)
        is_refract = ~tir
        ef_s = torch.where(tir, torch.ones_like(Rs), 1.0 - Rs)
        ef_p = torch.where(tir, torch.ones_like(Rp), 1.0 - Rp)

        D = torch.where(tir.unsqueeze(-1), D_ref, D_tr)
        O = hit
        b_s = b_s * ef_s
        b_p = b_p * ef_p
        inside = torch.where(is_refract, ~inside, inside)
        if bool(tir.any()):
            tir_inc.append(torch.where(tir, torch.acos(cos1.clamp(-1.0, 1.0)),
                                       torch.full((m,), 1e3, dtype=dtype)))

    dx = D[..., 0]
    t_s = (PUPIL_X - O[..., 0]) / torch.where(dx.abs() < G.EPS, torch.full_like(dx, G.EPS), dx)
    reached = reached & (t_s > 0)
    pupil_y = O[..., 1] + t_s * D[..., 1]
    exit_angle = torch.rad2deg(torch.atan2(D[..., 1], D[..., 0]))
    energy = b_s + b_p
    tir_stack = torch.stack(tir_inc, 0) if tir_inc else torch.empty(0, m, dtype=dtype)
    return exit_angle, energy, pupil_y, reached, tir_stack, G.critical_angle(n)


def trace_grouped(faces, n_index, origins, dirs, n_events):
    """Trace rays that may differ in bright-path length by grouping on n_events; scatter back to
    original order. Returns (ang[m], energy[m], pupil_y[m], reached[m], min_tir_margin_deg)."""
    dtype = faces[0].center.dtype
    m = len(n_events)
    ang = torch.zeros(m, dtype=dtype); en = torch.zeros(m, dtype=dtype)
    py = torch.zeros(m, dtype=dtype); rc = torch.zeros(m, dtype=torch.bool)
    n_events = np.asarray(n_events, int)
    margins = []
    for ne in sorted(set(int(x) for x in n_events)):
        if ne <= 0:
            continue
        idx = np.where(n_events == ne)[0]
        ea, e, p, r, tir, crit = trace_prism_ff(faces, n_index, origins[idx], dirs[idx], ne)
        ii = torch.as_tensor(idx)
        ang = ang.index_copy(0, ii, ea); en = en.index_copy(0, ii, e)
        py = py.index_copy(0, ii, p); rc = rc.index_copy(0, ii, r)
        if tir.numel():
            mm = torch.rad2deg(tir.min() - crit)
            if torch.isfinite(mm):
                margins.append(float(mm))
    return ang, en, py, rc, (float(min(margins)) if margins else float("nan"))


# --------------------------------------------------------------------------- #
#  differentiable per-source collimation metric (matches prism_branch.metrics)
# --------------------------------------------------------------------------- #
def collimation(ang, energy, pupil_y, sid, reached, n_fan, n_src=3):
    """Differentiable per-source angular metric of the bright bundle at the eyebox. Returns a
    dict with torch scalars: fitness, fov, sigma_mean, T_min, ordering_ok(bool), and per-source
    lists sigmas/thetas/Ts/present. Mirrors prism_branch.metrics (energy-weighted theta/sigma,
    T = sum(w)/n_fan, fitness = mean_s T_s*exp(-sigma_s/SIGMA0) * ordering_ok)."""
    dtype = ang.dtype
    sid = np.asarray(sid, int)
    eyebox = reached & (energy > 0) & (pupil_y.abs() <= EYEBOX_HALF)
    sigmas, thetas, Ts, present, qs = [], [], [], [], []
    for si in range(n_src):
        mask = torch.as_tensor(sid == si) & eyebox
        w = torch.where(mask, energy, torch.zeros_like(energy))
        W = w.sum()
        if float(W) <= 0:
            present.append(False); sigmas.append(None); thetas.append(None); Ts.append(0.0)
            qs.append(torch.zeros((), dtype=dtype)); continue
        theta = (ang * w).sum() / W
        sigma = torch.sqrt(((ang - theta) ** 2 * w).sum() / W + 0.0)
        T = W / n_fan
        present.append(True); sigmas.append(sigma); thetas.append(theta); Ts.append(T)
        qs.append(T * torch.exp(-sigma / SIGMA0))
    th_vals = [float(t) if t is not None else None for t in thetas]
    ordering_ok = (sum(present) == n_src and
                   (th_vals[0] < th_vals[1] < th_vals[2] or th_vals[0] > th_vals[1] > th_vals[2]))
    fitness = torch.stack(qs).mean() * (1.0 if ordering_ok else 0.0)
    fov = (abs(th_vals[2] - th_vals[0]) if present[0] and present[2] else 0.0)
    sig_present = [s for s in sigmas if s is not None]
    sigma_mean = (torch.stack(sig_present).mean() if sig_present else torch.zeros((), dtype=dtype))
    T_min = min(float(t) for t in Ts)
    return dict(fitness=fitness, fov=float(fov), sigma_mean=sigma_mean, T_min=T_min,
                ordering_ok=bool(ordering_ok), sigmas=sigmas, thetas=thetas, Ts=Ts, present=present)
