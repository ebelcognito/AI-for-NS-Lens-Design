"""Observation and protected-direction utilities for folded TIR designs.

``loss`` is a search surrogate. Acceptance must always use ``raw['feasible']`` from
the optical evaluator; a low surrogate value is not an optical feasibility test.
"""
from __future__ import annotations

import numpy as np

from nslens.discovery.folded import evaluate


_INVALID_LOSS = 1_000.0
FEATURE_NAMES = (
    "quality", "support", "folded_throughput", "normalized_worst_spot",
    "centroid_field_low", "centroid_field_mid", "centroid_field_high",
    "source_clearance_20mm", "detector_clearance_20mm",
    "geometry_center_x_100mm", "geometry_center_y_100mm",
    "geometry_width_100mm", "geometry_height_100mm", "geometry_radial_rms_100mm",
    "mean_field_quality", "mean_field_folded_throughput", "mean_field_root_support",
    "max_requirement_violation", "sum_requirement_violations",
)
_FEATURE_SIZE = len(FEATURE_NAMES)


def _invalid_raw(reason: str, genes) -> dict:
    return {"status": "invalid", "feasible": False, "reason": reason,
            "genes": np.asarray(genes, dtype=float).tolist(), "violations": {}}


def observe(x, corners=4, dense=False, spot_weight=1.0) -> dict:
    """Evaluate one design and return scalar merit, protected outputs, and features.

    The fixed-size feature vector permits a learner to share training across polygon
    corner counts. The three centroid entries correspond to the brief's ordered
    fields; designs with a different number of fields are linearly resampled.
    """
    from nslens.discovery.folded import RoutingBrief
    from nslens.discovery.polygon import PolygonCodec

    genes = np.asarray(x, dtype=float)
    if genes.ndim != 1 or not np.isfinite(genes).all():
        raw = _invalid_raw("genes must be a finite vector", genes)
        return dict(loss=_INVALID_LOSS, valid=False, protected=np.zeros(5),
                    features=np.zeros(_FEATURE_SIZE), raw=raw)
    try:
        codec = PolygonCodec(int(corners))
        brief = RoutingBrief(
            spot_weight=float(spot_weight),
            fields_deg=(-.5, -.25, 0., .25, .5) if dense else (-.5, 0., .5),
            rays_per_field=25 if dense else 9,
        )
        result = evaluate(codec, genes, brief, return_trace=True,
                          geometry_samples=1024 if dense else 256)
        row = result[0] if isinstance(result, tuple) else result
    except (ValueError, FloatingPointError, TypeError) as exc:
        row = _invalid_raw(str(exc), genes)
    if row.get("status") != "measured":
        raw = row if isinstance(row, dict) else _invalid_raw("evaluation failed", genes)
        return dict(loss=_INVALID_LOSS, valid=False, protected=np.zeros(5),
                    features=np.zeros(_FEATURE_SIZE), raw=raw)

    fields = row.get("fields", [])
    centroids = np.asarray([f.get("detector_centroid", 0.) for f in fields], dtype=float)
    if len(centroids):
        # Normalize around detector centre in units of full detector width.
        normalized = (centroids - brief.detector_center_y) / (2 * brief.detector_half_height)
        source = np.linspace(-1., 1., len(normalized))
        centroid3 = np.interp(np.linspace(-1., 1., 3), source, normalized)
    else:
        centroid3 = np.zeros(3)
    source_clearance = float(row.get("source_clearance", 0.)) / 20.
    detector_clearance = float(row.get("detector_clearance", 0.)) / 20.
    protected = np.concatenate((centroid3, (source_clearance, detector_clearance)))

    vertices = np.asarray(row.get("vertices", []), dtype=float)
    if vertices.ndim == 2 and vertices.shape[1] == 2 and len(vertices):
        center = vertices.mean(axis=0) / 100.
        spread = np.ptp(vertices, axis=0) / 100.
        radial_rms = float(np.sqrt(np.mean(np.sum((vertices - vertices.mean(axis=0))**2, axis=1)))) / 100.
    else:
        center, spread, radial_rms = np.zeros(2), np.zeros(2), 0.
    margins = np.asarray([v for k, v in row.get("violations", {}).items()], dtype=float)
    max_violation = float(np.max(margins)) if margins.size else 0.
    violations_sum = float(np.sum(margins)) if margins.size else 0.
    cent3 = np.asarray(centroid3, dtype=float)
    field_quality = np.asarray([f.get("imaging_quality", 0.) for f in fields], dtype=float)
    field_fold = np.asarray([f.get("folded_throughput", 0.) for f in fields], dtype=float)
    field_support = np.asarray([f.get("root_fraction", 0.) for f in fields], dtype=float)
    features = np.asarray([
        row.get("quality", 0.), row.get("support", 0.), row.get("worst_folded_throughput", 0.),
        row.get("worst_spot_rms", 0.) / brief.beam_width_mm,
        *cent3, source_clearance, detector_clearance,
        *center, *spread, radial_rms,
        float(np.mean(field_quality)) if len(field_quality) else 0.,
        float(np.mean(field_fold)) if len(field_fold) else 0.,
        float(np.mean(field_support)) if len(field_support) else 0.,
        max_violation, violations_sum,
    ], dtype=float)
    # Keep feature shape fixed even if this diagnostic list evolves.
    features = np.pad(features[:_FEATURE_SIZE], (0, max(0, _FEATURE_SIZE-len(features))))
    loss = max_violation + .1 * violations_sum - .01 * float(row.get("quality", 0.))
    meaningful = bool(fields) and all(
        float(f.get("folded_throughput", 0.)) > 1e-4 for f in fields
    )
    return dict(loss=float(loss), valid=meaningful,
                protected=protected if meaningful else np.zeros(5),
                features=features, raw=row)


def build_jacobian(x, obs, observe_fn, h=.003) -> np.ndarray:
    """Central finite differences of protected outputs with actual bounded steps."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or h <= 0:
        raise ValueError("x must be a finite vector and h positive")
    base = np.asarray(obs["protected"], dtype=float)
    jac = np.zeros((base.size, x.size), dtype=float)
    if not obs.get("valid", False):
        return jac
    for j in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[j] = min(1., x[j] + h)
        minus[j] = max(-1., x[j] - h)
        denominator = plus[j] - minus[j]
        if denominator <= 0:
            continue
        op, om = observe_fn(plus), observe_fn(minus)
        if not op.get("valid", False) or not om.get("valid", False):
            continue
        yp = np.asarray(op["protected"], dtype=float)
        ym = np.asarray(om["protected"], dtype=float)
        if yp.shape != base.shape or ym.shape != base.shape:
            raise ValueError("protected output shape changed during Jacobian probes")
        jac[:, j] = (yp - ym) / denominator
    return jac


def compensated_delta(x, delta, J, damping=1e-6, max_step=.08) -> np.ndarray:
    """Project a mutation into the damped nullspace of protected outputs.

    Rescaling retains the requested mutation norm when a useful nullspace component
    exists; coordinate caps and box bounds then keep the action feasible.
    """
    x, delta, J = np.asarray(x, float), np.asarray(delta, float), np.asarray(J, float)
    if x.ndim != 1 or delta.shape != x.shape or J.ndim != 2 or J.shape[1] != x.size:
        raise ValueError("expected x, delta shaped (n,) and J shaped (m,n)")
    if not (np.isfinite(x).all() and np.isfinite(delta).all() and np.isfinite(J).all()):
        raise ValueError("inputs must be finite")
    if damping < 0 or max_step <= 0:
        raise ValueError("damping must be nonnegative and max_step positive")
    target_norm = float(np.linalg.norm(delta))
    if target_norm == 0 or J.shape[0] == 0:
        projected = delta.copy()
    else:
        gram = J @ J.T
        correction = J.T @ np.linalg.solve(gram + damping * np.eye(J.shape[0]), J @ delta)
        projected = delta - correction
        norm = float(np.linalg.norm(projected))
        if norm > 1e-12:
            projected *= target_norm / norm
    scale = min(1., max_step / max(float(np.max(np.abs(projected))), 1e-30))
    projected *= scale
    # One common scale keeps the action direction and protected projection intact.
    hit = np.abs(projected) > 0
    if np.any(hit):
        available = np.where(projected > 0, 1. - x, -1. - x)
        box_scales = available[hit] / projected[hit]
        scale_box = min(1., max(0., float(np.min(box_scales))))
        projected *= scale_box
    return projected
