"""Differentiable objective: the same energy-weighted RMS spot + throughput-floor penalty
that ns_probe/refine_lm.py minimizes, but as a torch graph so its gradient is exact.

* dense_spot       -- energy-weighted RMS of tracked sensor hits about their best-focus
                      centroid, with FROZEN per-ray weights (refine_lm.Objective uses
                      w0 = champion energies, sum-normalized, and a floating centroid).
* throughput_floor -- one-sided penalty if delivered energy drops below a floor (can't
                      sharpen by dumping energy out of the fold).
* objective        -- spot**2 + mu * penalty**2  (sum-of-squares, matching the LM cost).
"""
from __future__ import annotations

import torch


def _weights(topo, device, dtype):
    w = torch.as_tensor(topo.w0, dtype=dtype, device=device)
    return w / w.sum().clamp_min(1e-30)


def dense_spot(tr, topo) -> torch.Tensor:
    """Energy-weighted RMS spot (mm) of the tracked rays, frozen weights, floating centroid."""
    w = _weights(topo, tr.y.device, tr.y.dtype)
    ybar = (w * tr.y).sum()
    var = (w * (tr.y - ybar) ** 2).sum()
    return torch.sqrt(var.clamp_min(0.0) + 1e-30)


def throughput(tr, topo) -> torch.Tensor:
    """Mean delivered energy over tracked rays (each input ray carries 1.0), i.e. the
    fraction of tracked-bundle energy reaching the sensor on the dominant path."""
    return tr.energy.mean()


def throughput_floor_penalty(tr, topo, thru0: float, slack: float) -> torch.Tensor:
    floor = thru0 * (1.0 - slack)
    return torch.clamp(floor - throughput(tr, topo), min=0.0)


def objective(tr, topo, *, thru0: float, slack: float = 0.02, mu: float = 50.0):
    spot = dense_spot(tr, topo)
    pen = throughput_floor_penalty(tr, topo, thru0, slack)
    cost = spot ** 2 + mu * pen ** 2
    return cost, spot, pen
