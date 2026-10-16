"""Residual based local moves for the variable mirror tracer.

This module only measures and proposes moves. Geometry feasibility is checked
by the experiment driver after a proposal is made.
"""
import numpy as np

from . import variable_mirror


def observe(g, n, task, parity=None):
    """Trace one fixed 3-field by 9-pupil task and return residual features."""
    task = dict(task or {})
    task["fields"] = 3
    task["pupils"] = 9
    row, m, _outlines, histories, good = variable_mirror.trace(
        g, n, paths=True, **task)
    parity = row["parity"] if parity is None else parity
    focal = float(task.get("focal", variable_mirror.FOCAL))
    half_field = float(task.get("half_field_deg", 4.0))
    scale = max(focal * np.tan(np.deg2rad(half_field)), 1.0)
    fields = np.repeat(np.linspace(-half_field, half_field, 3), 9)
    target = parity * focal * np.tan(np.deg2rad(fields))
    detector = np.zeros(27, dtype=float)
    for i, path in enumerate(histories):
        if good[i] and path:
            detector[i] = (path[-1] - m["image"]) @ m["image_tangent"]
    residual = np.where(good, (detector - target) / scale, 0.0)

    features = []
    good = np.asarray(good, dtype=bool)
    for f in range(3):
        sl = slice(f * 9, (f + 1) * 9)
        vals = residual[sl][good[sl]]
        features.extend((float(np.mean(vals)) if vals.size else 0.0,
                         float(np.std(vals)) if vals.size else 0.0,
                         float(np.sqrt(np.mean(vals ** 2))) if vals.size else 0.0))
    features.extend(float(np.mean(good[f * 9:(f + 1) * 9])) for f in range(3))
    # Dimensionless task descriptors keep feature scales stable across tasks.
    features.extend((focal / variable_mirror.FOCAL,
                     float(task.get("aperture", variable_mirror.APERTURE)) /
                     variable_mirror.APERTURE,
                     half_field / 4.0))
    return dict(raw=row, residual=residual, mask=good, parity=parity,
                features=np.asarray(features, dtype=float))


def jacobian(x, parent, observe_callback, h=0.003):
    """Bounded central-difference Jacobian, requiring common valid rays."""
    x = np.asarray(x, dtype=float)
    base_mask = np.asarray(parent["mask"], dtype=bool)
    base_residual = np.asarray(parent["residual"], dtype=float)
    J = np.zeros((base_residual.size, x.size), dtype=float)
    valid = np.zeros_like(J, dtype=bool)
    for j in range(x.size):
        plus_x, minus_x = x.copy(), x.copy()
        plus_x[j] = min(1.0, x[j] + h)
        minus_x[j] = max(-1.0, x[j] - h)
        plus = observe_callback(plus_x)
        minus = observe_callback(minus_x)
        den = plus_x[j] - minus_x[j]
        if den <= 0:
            continue
        pmask = np.asarray(plus["mask"], dtype=bool)
        mmask = np.asarray(minus["mask"], dtype=bool)
        common = base_mask & pmask & mmask
        valid[:, j] = common
        J[common, j] = (np.asarray(plus["residual"])[common] -
                        np.asarray(minus["residual"])[common]) / den
    return J, valid


def action_groups(n):
    """Return gene indices for the three parameter families and all genes."""
    variable_mirror._check_n(n)
    return {"angles": np.arange(n), "lengths": np.arange(n, 2*n),
            "curves": np.arange(2*n, 3*n), "all": np.arange(3*n)}


def propose(x, residual, J, group, radius):
    """Compute a damped least-squares residual reducing step in one group."""
    x = np.asarray(x, dtype=float)
    residual = np.asarray(residual, dtype=float)
    J = np.asarray(J, dtype=float)
    indices = np.asarray(group, dtype=int)
    step = np.zeros_like(x)
    if not indices.size:
        return step
    A = J[:, indices]
    # Ridge regularization makes rank deficient ray families well behaved.
    lhs = A.T @ A + 1.0e-4 * np.eye(indices.size)
    rhs = -(A.T @ residual)
    try:
        local = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        local = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    step[indices] = local
    norm = np.linalg.norm(step)
    if norm > radius >= 0:
        step *= radius / norm
    # A common fraction preserves the proposed direction while staying feasible.
    fraction = 1.0
    for i, delta in enumerate(step):
        if delta > 0:
            fraction = min(fraction, (1.0 - x[i]) / delta)
        elif delta < 0:
            fraction = min(fraction, (-1.0 - x[i]) / delta)
    step *= max(0.0, min(1.0, fraction))
    return step
