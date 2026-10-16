"""Small finite-dimensional probes for genuine 3D final-surface correction.

The probes only fit final-surface slopes using the initial valid rays. Their
objective is fixed-target image mapping error, with explicit penalties for lost
rays; no post-fit outcome enters the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from nslens.monolithic_ir import model as mono
from nslens.thermal_camera import freeform_model as thermal
from nslens.thermal_camera.model import unit


def _environment_key(environment):
    name = environment if isinstance(environment, str) else getattr(environment, "__name__", "")
    if name in ("thermal", "thermal_camera", "nslens.thermal_camera.freeform_model"):
        return "thermal"
    if name in ("mono", "monolithic", "monolithic_ir", "nslens.monolithic_ir.model"):
        return "mono"
    raise ValueError(f"unsupported 3D probe environment: {environment!r}")


def _pad_genes(key, genes):
    g = np.asarray(genes, dtype=float).ravel()
    size = thermal.N_TOTAL if key == "thermal" else mono.N_TOTAL
    if not np.isfinite(g).all() or len(g) > size:
        raise ValueError(f"genes must be finite and contain at most {size} values")
    out = np.zeros(size, dtype=float)
    out[:len(g)] = g
    return out


def _thermal_active_indices():
    return np.asarray([thermal.BASE_DIM + i for i, (surface, _, _) in enumerate(thermal.PARAMETERS)
                       if surface == "secondary"], dtype=int)


def _mono_active_indices():
    return np.asarray([14, 15, *range(41, 48)], dtype=int)


def _thermal_gradient_jacobian(x, y):
    """Rows are d(sag_x,sag_y)/d(active gene) at secondary hits."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    u, v = x / thermal.NORMALIZATION_MM["secondary"], y / thermal.NORMALIZATION_MM["secondary"]
    dx, dy = [], []
    for surface, i, j in thermal.PARAMETERS:
        if surface != "secondary":
            continue
        amp = thermal.AMPLITUDES_MM["secondary"]
        dx.append(amp * i * u ** (i - 1) * v ** j / thermal.NORMALIZATION_MM["secondary"] if i else np.zeros_like(u))
        dy.append(amp * j * u ** i * v ** (j - 1) / thermal.NORMALIZATION_MM["secondary"] if j else np.zeros_like(u))
    return np.stack(dx, axis=-1), np.stack(dy, axis=-1)


def _mono_gradient_jacobian(x_local, y):
    """Rows are d(exit_sag_x,exit_sag_y)/d(exit gene) in genes 14:16,41:48."""
    x, y = np.broadcast_arrays(np.asarray(x_local, float), np.asarray(y, float))
    X, Y = x / 3.0, y / 3.0
    dx = [0.04 * x, np.zeros_like(y)]
    dy = [np.zeros_like(x), 0.04 * y]
    amps = np.asarray([.08, .08, .04, .04, .04, .025, .025])
    for amp, (i, j) in zip(amps, mono.TERMS):
        dx.append(amp * i * X ** (i - 1) * Y ** j / 3.0 if i else np.zeros_like(X))
        dy.append(amp * j * X ** i * Y ** (j - 1) / 3.0 if j else np.zeros_like(X))
    return np.stack(dx, axis=-1), np.stack(dy, axis=-1)


def _normal_slopes(normal):
    n = unit(np.asarray(normal, float))
    nz = n[..., 2]
    safe = np.where(np.abs(nz) > 1e-10, nz, np.nan)
    return -n[..., 0] / safe, -n[..., 1] / safe


def _reflection_required_slopes(incoming, outgoing):
    """Surface-gradient slopes for a specular reflection from incoming to outgoing."""
    return _normal_slopes(unit(np.asarray(incoming) - np.asarray(outgoing)))


def _refraction_required_slopes(incoming, outgoing, n_in):
    """Surface-gradient slopes for refraction from index n_in to air."""
    n = np.asarray(n_in)[..., None] * unit(incoming) - unit(outgoing)
    return _normal_slopes(n)


def _targets_for_trace(key, trace, parity):
    # Thermal target is defined around the detector's global optical axis;
    # mono trace.xy is already detector-relative.
    angle = np.deg2rad(np.asarray(trace["fields_deg"], dtype=float))
    if key == "thermal":
        return np.asarray(parity)[None, :] * thermal.radial.EFL_TARGET_MM * np.tan(angle)
    return np.asarray(parity)[None, :] * 6.0 * np.tan(angle)


def _parity(key, trace):
    xy = np.asarray(trace["xy"])
    good = np.asarray(trace["good"], bool)
    centroids = np.zeros((xy.shape[0], 2), float)
    for f in range(xy.shape[0]):
        mask = good[f]
        if np.any(mask):
            centroids[f] = np.mean(xy[f][mask], axis=0)
    fields = np.deg2rad(np.asarray(trace["fields_deg"], float))
    focal = thermal.radial.EFL_TARGET_MM if key == "thermal" else 6.0
    angular_target = focal * np.tan(fields)
    # Each axis is frozen independently from the valid baseline centroids.
    return np.where(np.sum(centroids * angular_target, axis=0) >= 0, 1.0, -1.0)


def _measure(key, trace, parity):
    pix = thermal.PIXEL_PITCH_MM if key == "thermal" else mono.PIXEL_MM
    xy = np.asarray(trace["xy"], float)
    good = np.asarray(trace["good"], bool)
    targets = _targets_for_trace(key, trace, parity)
    residual = xy - targets[:, None, None, :]
    finite = np.isfinite(residual).all(axis=-1)
    good = good & finite
    err2 = np.sum(np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0) ** 2, axis=-1)
    valid_counts = np.sum(good, axis=2)
    per_cell_mse = np.sum(np.where(good, err2, 0.0), axis=2) / np.maximum(valid_counts, 1)
    per_cell_mse = np.where(valid_counts > 0, per_cell_mse, np.nan)
    support = np.mean(good, axis=2)
    physical = bool(trace.get("physical_geometry", True))
    geometry_penalty = 0.0
    if key == "thermal":
        m = trace["model"]
        physical = bool(m["minimum_plate_thickness_mm"] > .05
                        and m["minimum_secondary_plate_gap_mm"] > .05
                        and m["minimum_primary_secondary_gap_mm"] > .05)
    else:
        model = trace['model']
        entrance, mirror1, mirror2, exit_surface = model.surfaces
        x, y = np.meshgrid(np.linspace(-10, 10, 17), np.linspace(-5, 5, 11))
        front, back = mono.body(model, x, y)
        front_gap = mirror2.center[0]-mirror2.radius-entrance.center[0]-entrance.radius
        back_gap = exit_surface.center[0]-exit_surface.radius-mirror1.center[0]-mirror1.radius
        depth = float(np.max(back)-np.min(front))
        thickness = float(np.min(back-front))
        physical = bool(front_gap > 0 and back_gap > 0 and depth <= 10 and thickness > .1)
        # Preserve the native body's soft clearance/depth/thickness penalties.
        violations = np.array([max(0., .2-front_gap), max(0., .2-back_gap),
                               max(0., depth-10), max(0., .1-thickness)])
        geometry_penalty = float(400*np.sum(violations**2))
    lost = 1.0 - float(np.mean(good))
    valid_mse = err2[good]
    mean_mse = float(np.mean(valid_mse)) if len(valid_mse) else float("nan")
    penalized_err2 = np.where(good, err2, 100.0 * pix * pix)
    merit = float(np.mean(penalized_err2) / (pix * pix) + 1000.0 * lost + geometry_penalty)
    if not physical:
        merit += 1e6
    return {"loss": merit,
            "worst_mapping_rms_mm": float(np.sqrt(np.max(per_cell_mse))) if np.all(np.isfinite(per_cell_mse)) else None,
            "mean_mapping_mse_mm2": mean_mse if np.isfinite(mean_mse) else None,
            "worst_support": float(np.min(support)),
            "throughput": float(np.mean(good)),
            "valid_rays": int(np.sum(good)),
            "launched_rays": int(good.size),
            "physical_geometry": physical}


def _trace(key, genes, fields, pupils):
    if key == "thermal":
        return thermal.trace(genes, fields=fields, waves=1, pupils=pupils)
    return mono.trace(genes, fields=fields, waves=3, pupils=pupils)


def prepare(environment, genes, fields=3, pupils=3):
    """Trace and freeze targets/parity once; retain this trace for the slope fit."""
    key = _environment_key(environment)
    g = _pad_genes(key, genes)
    tr = _trace(key, g, fields, pupils)
    active = _thermal_active_indices() if key == "thermal" else _mono_active_indices()
    parity = _parity(key, tr)
    initial = _measure(key, tr, parity)
    return {"environment": key, "genes": g, "active_indices": active,
            "trace": tr, "parity": parity, "fields": fields, "pupils": pupils,
            "initial": initial}


def observe(context, genes, fields=None, pupils=None):
    """Return fixed-target mapping metrics from exactly one genuine 3D trace."""
    key = context["environment"]
    g = _pad_genes(key, genes)
    f = context["fields"] if fields is None else fields
    p = context["pupils"] if pupils is None else pupils
    tr = _trace(key, g, f, p)
    return _measure(key, tr, context["parity"])


def _ray_fit_data(context):
    key, tr = context["environment"], context["trace"]
    good = np.asarray(tr["good"], bool).ravel()
    paths = np.asarray(tr["paths"])
    if key == "thermal":
        # Thermal path slots are [origin, front plane, plate, primary, secondary, hole, image].
        hit = paths[4]
        incoming = unit(paths[4] - paths[3])
        target = _targets_for_trace(key, tr, context["parity"])
        target_flat = np.broadcast_to(target[:, None, None, :], tr["xy"].shape).reshape(-1, 2)
        model = tr["model"]
        out_point = np.column_stack((target_flat, np.full(len(target_flat), model["detector_z"])))
        outgoing = unit(out_point - hit)
        desired_x, desired_y = _reflection_required_slopes(incoming, outgoing)
        surface = model["secondary"]
        _, current_x, current_y = thermal.radial.sag_xy(hit[:, 0], hit[:, 1], *surface[1:])
        jac_x, jac_y = _thermal_gradient_jacobian(hit[:, 0], hit[:, 1])
    else:
        # Mirror 2 -> exit; mono paths are [origin, entrance, mirror1, mirror2, exit, detector].
        hit = paths[4]
        incoming = unit(paths[4] - paths[3])
        target = _targets_for_trace(key, tr, context["parity"])
        target_flat = np.broadcast_to(target[:, None, None, :], tr["xy"].shape).reshape(-1, 2)
        model = tr["model"]
        surface = model.surfaces[3]
        out_point = np.column_stack((model.detector[:2] + target_flat,
                                     np.full(len(target_flat), model.detector[2])))
        outgoing = unit(out_point - hit)
        wavelength = np.broadcast_to(tr["wavelengths_mm"][None, :, None], tr["good"].shape).ravel()
        n_in = mono.index(wavelength)
        desired_x, desired_y = _refraction_required_slopes(incoming, outgoing, n_in)
        _, current_x, current_y = mono.sag(surface, hit[:, 0], hit[:, 1])
        jac_x, jac_y = _mono_gradient_jacobian(hit[:, 0] - surface.center[0], hit[:, 1])
    finite = (good & np.isfinite(desired_x) & np.isfinite(desired_y)
              & np.isfinite(current_x) & np.isfinite(current_y)
              & np.isfinite(jac_x).all(axis=1) & np.isfinite(jac_y).all(axis=1))
    required = np.column_stack((desired_x[finite] - current_x[finite],
                                desired_y[finite] - current_y[finite]))
    J = np.zeros((int(np.sum(finite)) * 2, jac_x.shape[-1]), float)
    J[0::2] = jac_x[finite]
    J[1::2] = jac_y[finite]
    return finite, required, J


def fit_probe(context):
    """Fit bounded final-surface gene changes from the stored valid baseline rays."""
    key, genes = context["environment"], context["genes"]
    active = context["active_indices"]
    finite, required, J = _ray_fit_data(context)
    delta = np.zeros_like(genes)
    if not len(required):
        return delta, {"usable_rays": 0, "bound_saturated": 0,
                       "rms_required_slope": None, "rms_unexplained_slope": None}
    rhs = required.reshape(-1)
    rms_required = float(np.sqrt(np.mean(rhs**2)))
    lower, upper = -1.0 - genes[active], 1.0 - genes[active]
    result = lsq_linear(J, rhs, bounds=(lower, upper), method="trf", lsmr_tol="auto")
    delta[active] = result.x
    residual = rhs - J @ result.x
    tol = 1e-7
    saturated = np.isclose(genes[active] + result.x, -1.0, atol=tol) | np.isclose(genes[active] + result.x, 1.0, atol=tol)
    diagnostics = {"usable_rays": int(np.sum(finite)),
                   "active_indices": active.tolist(),
                   "bound_saturated": int(np.sum(saturated)),
                   "bound_saturated_indices": active[saturated].tolist(),
                   "rms_required_slope": rms_required,
                   "rms_unexplained_slope": float(np.sqrt(np.mean(residual**2))),
                   "fit_cost": float(result.cost),
                   "fit_optimality": float(result.optimality),
                   "fit_success": bool(result.success)}
    return delta, diagnostics
