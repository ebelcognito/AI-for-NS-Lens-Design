"""Differentiable reconstruction of the AR/VR 4-curved-face freeform prism.

A torch port of ns_probe/arvr_prism.py:prism_element() -- the SAME geometry (four circular
arc faces: left A, right B, top, bottom), built from the 9-DOF prism genotype, as
differentiable tensors. We copy the geometry (never import-and-mutate arvr_prism).

Genotype order (matches arvr_prism.PRISM_NAMES / export common.param_names):
    [cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n]
     left right top? -- fsag=left face, bsag=right face, tsag=top edge, dsag=bottom edge.
"""
from __future__ import annotations

import math

import torch

from .geometry_torch import circumcenter

ELEM_H = 30.0            # arvr_prism.ELEM_H
PLANE_X = 120.0          # arvr_prism.PUPIL_X (eye-pupil plane)
PARAM_NAMES = ["cx", "thick", "fsag", "bsag", "tsag", "dsag", "tilt_a", "tilt_b", "n"]
PIDX = {n: i for i, n in enumerate(PARAM_NAMES)}
FACES = ("bottom", "right", "top", "left")


def _deg2rad(x):
    return x * (math.pi / 180.0)


def build_surfaces_arvr(p):
    """From a param tensor p[9] build the four arc faces as a dict keyed by face label.
    Each face is {'kind':'arc','center':[2],'r':[],'end1':[2],'end2':[2],'apex':[2]} where
    the circle passes through (end1, apex, end2) -- identical convention to the export's
    design.surfaces and to element_torch.build_surfaces."""
    H = torch.as_tensor(ELEM_H, dtype=p.dtype, device=p.device)
    cx, thick = p[PIDX["cx"]], p[PIDX["thick"]]
    fsag, bsag = p[PIDX["fsag"]], p[PIDX["bsag"]]
    tsag, dsag = p[PIDX["tsag"]], p[PIDX["dsag"]]
    tilt_a, tilt_b = p[PIDX["tilt_a"]], p[PIDX["tilt_b"]]

    lx, rx = cx - thick / 2.0, cx + thick / 2.0
    dA = torch.tan(_deg2rad(tilt_a)) * H
    dB = torch.tan(_deg2rad(tilt_b)) * H
    zero = torch.zeros((), dtype=p.dtype, device=p.device)

    def pt(x, y):
        return torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x, dtype=p.dtype, device=p.device),
                            y if torch.is_tensor(y) else torch.as_tensor(y, dtype=p.dtype, device=p.device)])

    bl = pt(lx + dA, -H)
    br = pt(rx + dB, -H)
    tr = pt(rx - dB, H)
    tl = pt(lx - dA, H)
    apex_bottom = pt((bl[0] + br[0]) / 2.0, -H - dsag)
    apex_right = pt(rx + bsag, zero)
    apex_top = pt((tr[0] + tl[0]) / 2.0, H + tsag)
    apex_left = pt(lx - fsag, zero)

    def arc(end1, apex, end2):
        c, r = circumcenter(end1, apex, end2)
        return {"kind": "arc", "center": c, "r": r, "end1": end1, "end2": end2, "apex": apex}

    return {
        "bottom": arc(bl, apex_bottom, br),
        "right":  arc(br, apex_right, tr),
        "top":    arc(tr, apex_top, tl),
        "left":   arc(tl, apex_left, bl),
    }


def geometry_valid_arvr(p) -> bool:
    """Mirror arvr_prism.geometry_valid_arvr: non-degenerate body + curved top clears display."""
    cx = float(p[PIDX["cx"]]); thick = float(p[PIDX["thick"]])
    ta = float(p[PIDX["tilt_a"]]); tb = float(p[PIDX["tilt_b"]]); tsag = float(p[PIDX["tsag"]])
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(ta)) * ELEM_H
    dB = math.tan(math.radians(tb)) * ELEM_H
    base = ((rx + dB) - (lx + dA) > 1.0) and ((rx - dB) - (lx - dA) > 1.0)
    return base and (ELEM_H + max(tsag, 0.0) < 50.0 - 2.0)   # DISPLAY_Y = 50
