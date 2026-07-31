"""Upgrade a discovered circular arc face into a freeform (conic+polynomial) surface,
initialized to reproduce the arc exactly.

This is the "init from the champion" step: QD/discovery finds the fold topology with cheap
circular arcs (element_torch.build_surfaces); refinement then unlocks freeform DOF that start
at zero, so the freeform design begins life *at* the discovered arc and can only add freedom.

Isolation: this module reads the arc dicts produced by element_torch.build_surfaces; it does
not mutate element_torch or the oracle-validated geometry_torch/forward.
"""
from __future__ import annotations

import torch

from .geometry_freeform import FreeformSurface
from .geometry_torch import EPS, normalize


def arc_to_freeform(arc: dict, *, powers=(), coeffs=None) -> FreeformSurface:
    """Convert an arc dict {center,r,end1,end2,apex} (from build_surfaces) into a
    FreeformSurface whose conic base reproduces the arc's apex cap.

    Construction (see geometry_freeform header for the derivation):
      vertex v   = apex
      axis       = unit(apex - center)        (outward radial at the apex)
      tang       = perp(axis)
      c          = -1 / r                     (so conic sag == the circle's near cap)
      k          = 0
      aperture   = transverse projection of the two chord endpoints
    `powers`/`coeffs` add a polynomial deviation (default: none -> exact arc).
    """
    center, r = arc["center"], arc["r"]
    v = arc["apex"]
    axis = normalize(v - center)
    tang = torch.stack([-axis[1], axis[0]])           # perp(axis)
    c = -1.0 / r.clamp_min(EPS) if torch.is_tensor(r) else torch.as_tensor(-1.0 / r)
    if not torch.is_tensor(c):
        c = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    k = torch.zeros((), dtype=v.dtype, device=v.device)
    h_e1 = (arc["end1"] - v) @ tang
    h_e2 = (arc["end2"] - v) @ tang
    h_min = torch.minimum(h_e1, h_e2)
    h_max = torch.maximum(h_e1, h_e2)
    if coeffs is not None and not torch.is_tensor(coeffs):
        coeffs = torch.as_tensor(coeffs, dtype=v.dtype, device=v.device)
    return FreeformSurface(v=v, axis=axis, tang=tang, c=c, k=k,
                           h_min=h_min, h_max=h_max, powers=tuple(powers), coeffs=coeffs)
