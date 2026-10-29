"""Two independently editable optical regions on a closed 2-D spline element.

The entrance and exit side each support quadratic, cubic and quartic departures.
Ray propagation and acceptance use the existing verified repair evaluator unchanged.
This domain is deliberately explicit about its two-transmission imaging route.
"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from nslens.freeform_body.boundary import FreeformBody2D
from nslens.learning.freeform_repair import ImagingSpec, evaluate

PARAMETERS = tuple(f"{side}_p{power}" for side in ("entrance", "exit") for power in (2, 3, 4))
COEFFICIENT_BOUND_MM = 2.0


def surface_body(spec: ImagingSpec, coefficients):
    c = np.asarray(coefficients, dtype=float)
    if c.shape != (6,) or not np.isfinite(c).all() or np.max(np.abs(c)) > COEFFICIENT_BOUND_MM:
        raise ValueError("six finite surface coefficients in [-2, 2] mm required")
    theta = np.linspace(0., 2 * np.pi, 32, endpoint=False)
    y = spec.radius_y * np.sin(theta)
    x = spec.radius_x * np.cos(theta)
    z = y / (.4 * spec.radius_y)
    # Smooth endpoint taper joins the two optical regions at the top/bottom of
    # the element. The base is an ellipse; the result is a periodic cubic spline.
    taper = (1 - (y / spec.radius_y) ** 2) ** 2
    for side, active in enumerate((np.cos(theta) < 0, np.cos(theta) >= 0)):
        sag = sum(c[3 * side + j] * z ** power for j, power in enumerate((2, 3, 4))) * taper
        x[active] += sag[active]
    return FreeformBody2D(np.stack((60. + x, y), axis=1), n=spec.index)


class _FixedBody:
    def __init__(self, body):
        self.body = body

    def decode(self, genes):
        if np.any(genes):
            raise ValueError("fixed-body verifier adapter expects zero placeholder genes")
        return self.body


def measure(spec, coefficients, fidelity="coarse"):
    body = surface_body(spec, coefficients)
    # Reuse every ray, energy, route and acceptance check from the previous
    # evaluator. Only the body construction differs; no global monkeypatching.
    fixed = _FixedBody(body)
    adapter = SimpleNamespace(**asdict(spec), codec=lambda: fixed)
    return evaluate(adapter, np.zeros(4), fidelity)


def diagnostics(verdict):
    residual = np.asarray(verdict.residual).reshape(3, 5)
    return dict(
        rms_mm=verdict.rms, worst_field_rms_mm=verdict.worst_field_rms,
        field_centroid_error_mm=np.mean(residual, axis=1).tolist(),
        field_rms_error_mm=np.sqrt(np.mean(residual ** 2, axis=1)).tolist(),
        pupil_residual_mm=residual.tolist(),
        min_primary_ray_energy=verdict.min_ray_energy,
        primary_throughput=verdict.throughput,
        two_transmission_route_fraction=verdict.route_fraction,
        physical_valid=verdict.physical_valid, coarse_success=verdict.success,
        failure_reason=verdict.reason,
    )
