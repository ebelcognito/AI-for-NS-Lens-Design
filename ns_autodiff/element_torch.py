"""Differentiable reconstruction of the single hexagonal element used by the folds.

This is a torch port of ns_probe/fold_demo.py:element() / scene() -- the SAME geometry
(two tiltable curved optical faces A=left, B=right, two flat edges T=bottom, D=top),
built from the 9-parameter genotype, but as differentiable tensors. We do NOT import and
mutate fold_demo; we copy the geometry so this package never perturbs the existing code.

Genotype order (PARAM_NAMES, identical to fold_demo):
    [ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n]
"""
from __future__ import annotations

import math

import torch

from .geometry_torch import circumcenter

ELEM_H = 30.0          # fold_demo.ELEM_H
X_SENSOR = 120.0       # fold_demo.X_SENSOR
PARAM_NAMES = ["ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n"]
PIDX = {n: i for i, n in enumerate(PARAM_NAMES)}

# Face labels match sequence_tools.make_single_element_face_classifier:
#   T = bottom flat edge (y=-H),  D = top flat edge (y=+H),
#   A = left optical arc (x<cx),  B = right optical arc (x>cx).
FACES = ("A", "B", "T", "D")


def _deg2rad(x):
    return x * (math.pi / 180.0)


def build_surfaces(p):
    """From a param tensor p[9] (requires_grad ok) build the four element surfaces as a
    dict keyed by face label. Arc faces -> {'kind':'arc','center':[2],'r':[]}; flat edges
    -> {'kind':'line','a':[2],'b':[2]}. All tensors share p's dtype/device/graph."""
    H = torch.as_tensor(ELEM_H, dtype=p.dtype, device=p.device)
    cx = p[PIDX["cx"]]
    thick = p[PIDX["thick"]]
    fsag = p[PIDX["fsag"]]
    bsag = p[PIDX["bsag"]]
    tilt_a = p[PIDX["tilt_a"]]
    tilt_b = p[PIDX["tilt_b"]]

    lx = cx - thick / 2.0
    rx = cx + thick / 2.0
    dA = torch.tan(_deg2rad(tilt_a)) * H
    dB = torch.tan(_deg2rad(tilt_b)) * H

    def pt(x, y):
        return torch.stack([x, y if torch.is_tensor(y) else torch.as_tensor(
            y, dtype=p.dtype, device=p.device)])

    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    # fold_demo path indices:
    #   0:(lx+dA,-H) 1:(rx+dB,-H) 2:(rx+bsag,0)[arc] 3:(rx-dB,H) 4:(lx-dA,H) 5:(lx-fsag,0)[arc]
    v0 = pt(lx + dA, -H)
    v1 = pt(rx + dB, -H)
    v2 = pt(rx + bsag, zero)
    v3 = pt(rx - dB, H)
    v4 = pt(lx - dA, H)
    v5 = pt(lx - fsag, zero)

    # Right arc B: circle through v1, v2(control), v3
    cB, rB = circumcenter(v1, v2, v3)
    # Left arc A: circle through v4, v5(control), v0
    cA, rA = circumcenter(v4, v5, v0)

    return {
        # arcs carry their chord endpoints + apex (control point) for the on-arc test
        "A": {"kind": "arc", "center": cA, "r": rA, "end1": v4, "end2": v0, "apex": v5},
        "B": {"kind": "arc", "center": cB, "r": rB, "end1": v1, "end2": v3, "apex": v2},
        "T": {"kind": "line", "a": v0, "b": v1},   # bottom edge y=-H
        "D": {"kind": "line", "a": v3, "b": v4},   # top edge y=+H
    }


def input_ray(p, dy):
    """The input ray for beam offset dy (source_y - ray_y), matching fold_demo.scene:
    origin (0, ray_y+dy), direction (cos a, sin a), a = radians(ang). dy may be a tensor
    batch [m] -> returns (O[m,2], D[m,2])."""
    a = _deg2rad(p[PIDX["ang"]])
    ray_y = p[PIDX["ray_y"]]
    dyt = torch.as_tensor(dy, dtype=p.dtype, device=p.device)
    y0 = ray_y + dyt
    O = torch.stack([torch.zeros_like(y0), y0], dim=-1)
    d = torch.stack([torch.cos(a), torch.sin(a)])
    D = d.expand(O.shape)
    return O, D


def geometry_valid(p) -> bool:
    """Mirror of fold_demo.geometry_valid (a cheap non-self-intersection guard)."""
    cx = float(p[PIDX["cx"]]); thick = float(p[PIDX["thick"]])
    tilt_a = float(p[PIDX["tilt_a"]]); tilt_b = float(p[PIDX["tilt_b"]])
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * ELEM_H
    dB = math.tan(math.radians(tilt_b)) * ELEM_H
    return (rx + dB) - (lx + dA) > 1.0 and (rx - dB) - (lx - dA) > 1.0
