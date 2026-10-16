"""Physical observables and local protected-feature compensation for mirror search.

This is a 2D, ideal-mirror geometric model helper.  It does not model diffraction,
3D aberrations, coatings, or mechanical clearance.  The trace grid is deliberately
fixed at either 3 fields x 9 pupil samples or 11 fields x 41 pupil samples so ``protected`` always has three entries:
center-field image centroid / 50 mm, fitted field-image slope / 50 mm, and the
on-axis image-versus-pupil slope.  Lower-dimensional support is reported invalid
rather than changing the feature dimensions.

``build_jacobian`` performs exactly 18 calls to ``observe_fn`` (central differences
for genes 0..8); it reuses the caller's baseline observation. Aperture genes 9..11
are never probed or changed. ``compensated_delta`` uses a damped least-squares
projection and respects normalized gene bounds [-1, 1]. Experiments should draw
the same raw mutation distribution and radii for compensated and control arms.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from nslens.three_mirror.pilot import FOCAL, trace

GENE_COUNT = 12
ACTIVE_GENE_COUNT = 9
PROTECTED_DIM = 3
PROBE_STEP = 0.003
PROJECTION_DAMPING = 1.0e-6
MAX_CORRECTION_NORM = 0.08
MAX_CORRECTION_COORD = 0.08
MAX_STEP_COORD = 0.08
BOUNDS = 1.0
_INVALID = 1.0e3


def observe(x: np.ndarray, dense: bool = False) -> dict:
    """Trace one fixed pilot grid and return protected values and physical features.

    ``valid`` requires at least two received rays at each field and three on-axis
    rays, allowing stable centroid, field-slope, and pupil-slope estimates. The
    invalid sentinel is finite for optimizer/Jacobian safety. ``raw`` is the trace
    result, including support and spot/error statistics.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (GENE_COUNT,) or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite 12-vector")
    fields, pupils = (11, 41) if dense else (3, 9)
    result, geometry, _outlines, histories, good = trace(
        x, fields=fields, pupils=pupils, paths=True
    )
    # The trace summary has aggregate scores only; paths retain the detector
    # intersections needed to compute signed image positions, all from this one trace.
    support = np.array([row["support"] for row in result["fields"]], dtype=float)
    counts = np.rint(support * pupils).astype(int)

    # Recover field centroids and pupil slope with one trace exposing the signed image
    # coordinates. The pilot trace API currently only returns ``good``; reconstruct
    # signed image positions from the final path points and detector basis.
    flat_good = np.asarray(good, dtype=bool)
    pupil = np.tile(np.linspace(-27.78 / 2.0, 27.78 / 2.0, pupils), fields)
    # Histories include the detector intersection as their final point for good rays.
    ray_count = fields * pupils
    image_pos = np.full(ray_count, np.nan)
    for i in np.flatnonzero(flat_good):
        image_pos[i] = (np.asarray(histories[i][-1]) - geometry["image"]) @ geometry["image_tangent"]
    selected_fields = np.array([0, fields // 2, fields - 1], dtype=int)
    selected_centroids = np.array([
        float(np.mean(image_pos[f * pupils:(f + 1) * pupils][flat_good[f * pupils:(f + 1) * pupils]]))
        if counts[f] else np.nan for f in selected_fields
    ])
    center = fields // 2
    center_slice = slice(center * pupils, (center + 1) * pupils)
    on_axis = flat_good[center_slice]
    valid = bool(np.all(counts[selected_fields] >= 2) and int(on_axis.sum()) >= 3)
    if valid:
        selected_angles = np.array([result["fields"][f]["field_deg"] for f in selected_fields])
        x_field = np.tan(np.deg2rad(selected_angles))
        # Normalize the fitted mm-per-tan(field) slope by the nominal 50 mm focal.
        field_slope = float(np.polyfit(x_field, selected_centroids, 1)[0] / FOCAL)
        pupil_slope = float(np.polyfit(pupil[center_slice][on_axis], image_pos[center_slice][on_axis], 1)[0])
        protected = np.array([selected_centroids[1] / FOCAL, field_slope, pupil_slope])
        if not np.all(np.isfinite(protected)):
            valid = False
    if not valid:
        protected = np.full(PROTECTED_DIM, _INVALID, dtype=float)

    turns = np.asarray(geometry["angles"], dtype=float)
    principal_turns = (turns + 180.0) % 360.0 - 180.0
    incidence_deg = 90.0 - np.abs(principal_turns) / 2.0
    lengths = np.asarray(geometry["lengths"], dtype=float)
    curves = np.asarray(geometry["curves"], dtype=float)
    # Fixed compact feature order: three curvatures, three chief-ray incidence
    # angles, two spacings, field-centroid residual RMS, max field residual, total
    # and worst-field throughput, and trace loss.
    target = FOCAL * np.tan(np.deg2rad(np.array([-4.0, 0.0, 4.0])))
    parity = 1.0 if selected_centroids[-1] >= selected_centroids[0] else -1.0
    residual = (selected_centroids - parity * target) / FOCAL
    finite_resid = residual[np.isfinite(residual)]
    resid_rms = float(np.sqrt(np.mean(finite_resid ** 2))) if len(finite_resid) else _INVALID
    resid_max = float(np.max(np.abs(finite_resid))) if len(finite_resid) else _INVALID
    features = np.r_[curves, incidence_deg, lengths[:2] / FOCAL,
                     resid_rms, resid_max, float(flat_good.mean()),
                     float(support.min()), float(result["loss"]) / 10000.0]
    return dict(loss=float(result["loss"]), valid=valid,
                protected=np.asarray(protected, dtype=float),
                features=np.asarray(features, dtype=float), raw=result)


def build_jacobian(x: np.ndarray, obs: dict,
                   observe_fn: Callable[[np.ndarray], dict],
                   h: float = PROBE_STEP) -> np.ndarray:
    """Central-difference protected-feature Jacobian, shape (3, 9).

    Exactly 18 observations are requested. A column is zero if either probe is
    invalid, avoiding finite differences across ray-support discontinuities.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (GENE_COUNT,):
        raise ValueError("x must be a 12-vector")
    if h <= 0 or not np.isfinite(h):
        raise ValueError("h must be finite and positive")
    base = np.asarray(obs.get("protected", ()), dtype=float)
    baseline_valid = (obs.get("valid", False) and base.shape == (PROTECTED_DIM,)
                      and np.all(np.isfinite(base)))
    if not baseline_valid:
        base = np.full(PROTECTED_DIM, _INVALID)
    jac = np.zeros((PROTECTED_DIM, ACTIVE_GENE_COUNT), dtype=float)
    for j in range(ACTIVE_GENE_COUNT):
        plus, minus = x.copy(), x.copy()
        plus[j] = min(BOUNDS, x[j] + h)
        minus[j] = max(-BOUNDS, x[j] - h)
        step = plus[j] - minus[j]
        op, om = observe_fn(plus), observe_fn(minus)
        if step > 0 and op.get("valid", False) and om.get("valid", False):
            vp = np.asarray(op["protected"], dtype=float)
            vm = np.asarray(om["protected"], dtype=float)
            if vp.shape == vm.shape == (PROTECTED_DIM,) and np.all(np.isfinite(vp)) and np.all(np.isfinite(vm)):
                jac[:, j] = (vp - vm) / step
    return jac if baseline_valid else np.zeros_like(jac)


def compensated_delta(x: np.ndarray, delta: np.ndarray, jacobian: np.ndarray,
                      damping: float = PROJECTION_DAMPING) -> np.ndarray:
    """Project a proposed mutation away from protected-feature change.

    The correction is limited to 0.08 normalized units in both Euclidean norm and
    any coordinate. A nondegenerate projection is then rescaled to the raw active
    step norm and globally capped at 0.08 per coordinate. Only genes 0..8 are
    projected; aperture genes remain fixed. A common final scale respects [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    if x.shape != (GENE_COUNT,) or delta.shape != (GENE_COUNT,):
        raise ValueError("x and delta must be 12-vectors")
    if jacobian.shape != (PROTECTED_DIM, ACTIVE_GENE_COUNT):
        raise ValueError("jacobian must have shape (3, 9)")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(delta)) and np.all(np.isfinite(jacobian))):
        raise ValueError("x, delta, and jacobian must be finite")
    if damping < 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and nonnegative")
    out = delta.copy()
    j = jacobian
    gram = j @ j.T
    correction = -j.T @ np.linalg.solve(gram + damping * np.eye(PROTECTED_DIM), j @ delta[:ACTIVE_GENE_COUNT])
    correction = np.clip(correction, -MAX_CORRECTION_COORD, MAX_CORRECTION_COORD)
    norm = float(np.linalg.norm(correction))
    if norm > MAX_CORRECTION_NORM:
        correction *= MAX_CORRECTION_NORM / norm
    out[:ACTIVE_GENE_COUNT] += correction
    out[ACTIVE_GENE_COUNT:] = 0.0
    target_norm = float(np.linalg.norm(delta[:ACTIVE_GENE_COUNT]))
    projected_norm = float(np.linalg.norm(out[:ACTIVE_GENE_COUNT]))
    if target_norm > 0.0 and projected_norm > 1.0e-12:
        out[:ACTIVE_GENE_COUNT] *= target_norm / projected_norm
    coordmax = float(np.max(np.abs(out[:ACTIVE_GENE_COUNT])))
    if coordmax > MAX_STEP_COORD:
        out[:ACTIVE_GENE_COUNT] *= MAX_STEP_COORD / coordmax
    # Scale the whole active step only as needed to remain inside the normalized box.
    scale = 1.0
    for i in range(ACTIVE_GENE_COUNT):
        if out[i] > 0:
            scale = min(scale, max(0.0, (BOUNDS - x[i]) / out[i]))
        elif out[i] < 0:
            scale = min(scale, max(0.0, (-BOUNDS - x[i]) / out[i]))
    return out * scale
