"""Dense independent ray and geometry audit for a monolithic IR result.

This evaluates geometric ray performance only. It does not model MTF,
straylight, coating behavior, thermal effects, or manufacturing tolerances.
The material index is extrapolated above 12 um; results from 12--14 um are
reported separately from the 8--12 um interval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import Polynomial

from . import model
from .search import _write


FIELDS = dict(fields=7, waves=7, pupils=9, body_samples=33)
BODY_GRID = (101, 51)
EXPECTED_FOCAL_MM = 6.0
EXPECTED_ENTRANCE_DIAMETER_MM = 6.0
RMS_LIMIT_MM = 0.017
SUPPORT_LIMIT = 0.95
BODY_THICKNESS_LIMIT_MM = 10.0


def _load_genes(source):
    data = json.loads(Path(source).read_text())
    genes = np.asarray(data["genes"], dtype=float).reshape(-1)
    if not np.isfinite(genes).all():
        raise ValueError('source genes must be finite')
    if np.any(np.abs(genes) > 1):
        raise ValueError("source genes must be normalized to [-1, 1]")
    snapshot = Path(source).parent / "model_snapshot.py"
    if snapshot.exists():
        module_name = f"_monolithic_ir_snapshot_{abs(hash(snapshot.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, snapshot)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load model snapshot: {snapshot}")
        snapshot_model = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = snapshot_model
        spec.loader.exec_module(snapshot_model)
        if genes.shape != (snapshot_model.N_TOTAL,):
            raise ValueError(f'source must contain {snapshot_model.N_TOTAL} genes for its snapshot')
        return data, genes, snapshot_model
    if genes.shape != (model.N_TOTAL,):
        raise ValueError(f'source must contain {model.N_TOTAL} genes')
    return data, genes, model


def _finite_float(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _waist_report(tr, center_field_index, wavelength_index):
    """Find the sampled minimum RMS plane between M1 and M2 for on-axis rays."""
    p1 = tr["paths"][2].reshape(49, 7, -1, 3)[center_field_index, wavelength_index]
    p2 = tr["paths"][3].reshape(49, 7, -1, 3)[center_field_index, wavelength_index]
    good = tr["good"].reshape(49, 7, -1)[center_field_index, wavelength_index]
    p1, p2 = p1[good], p2[good]
    if len(p1) < 3:
        return dict(rays=int(len(p1)), real=False, location_from_m1_mm=None,
                    mirror_separation_mm=None, rms_mm=None, endpoint_rms_mm=None)
    chief = int(np.argmin(np.linalg.norm(p1-p1.mean(axis=0), axis=1)))
    chief_delta = p2[chief]-p1[chief]
    distance = float(np.linalg.norm(chief_delta))
    axis = chief_delta / max(distance, 1e-30)
    directions = p2-p1
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(norms[:, None], 1e-30)
    along0 = (p1-p1[chief]) @ axis
    denom = directions @ axis
    if np.any(abs(denom) < 1e-8):
        return dict(rays=int(len(p1)), real=False, location_from_m1_mm=None,
                    mirror_separation_mm=_finite_float(distance), rms_mm=None,
                    endpoint_rms_mm=None)
    along = np.linspace(0, distance, 501)
    scores = []
    for s in along:
        t = (s-along0)/denom
        q = p1 + directions*t[:, None]
        transverse = q - np.outer(q @ axis, axis)
        transverse -= transverse.mean(axis=0)
        scores.append(np.sqrt(np.mean(np.sum(transverse*transverse, axis=1))))
    scores = np.asarray(scores)
    k = int(np.argmin(scores))
    end = max(float(scores[0]), float(scores[-1]))
    real = 0 < k < len(along)-1 and scores[k] < end
    return dict(rays=int(len(p1)), real=bool(real),
                location_from_m1_mm=_finite_float(along[k]),
                mirror_separation_mm=_finite_float(distance),
                location_fraction=_finite_float(along[k]/distance) if distance else None,
                rms_mm=_finite_float(scores[k]), endpoint_rms_mm=_finite_float(end),
                method="sampled planes normal to chief M1-to-M2 ray; rays linearly propagated from traced mirror hits")


def _surface_line_roots(origin, direction, surface, active_model):
    """Positive in-aperture intersections of a ray with one polynomial patch."""
    x = Polynomial([origin[0]-surface.center[0], direction[0]])
    y = Polynomial([origin[1], direction[1]])
    z = Polynomial([origin[2]-surface.center[2], direction[2]])
    sag = surface.tilt*x + surface.power[0]*x*x + surface.power[1]*y*y
    for coeff, (i, j) in zip(surface.higher, active_model.TERMS):
        sag = sag + coeff*(x/3)**i*(y/3)**j
    poly = z-sag
    coeff = np.trim_zeros(np.asarray(poly.coef, dtype=float), "b")
    if len(coeff) <= 1 or not np.isfinite(coeff).all():
        return []
    roots = np.polynomial.polynomial.polyroots(coeff)
    hits = []
    for root in roots:
        if abs(root.imag) > 2e-7:
            continue
        t = float(root.real)
        if t <= 1e-6:
            continue
        px = origin[0]+t*direction[0]-surface.center[0]
        py = origin[1]+t*direction[1]
        if np.hypot(px, py) <= surface.radius+1e-7:
            hits.append(t)
    return sorted(hits)


def _independent_root_check(genes, active_model):
    """Compare Newton segments with positive roots for sparse fully good rays."""
    sample = active_model.trace(genes, fields=3, waves=3, pupils=3)
    good = sample["good"].reshape(-1)
    available = np.flatnonzero(good)
    # Bound root-solver cost while spreading checked rays over the ordered
    # field/wavelength/pupil sequence.
    chosen = available[np.unique(np.linspace(0, len(available)-1,
                                              min(128, len(available))).round().astype(int))] if len(available) else available
    paths = sample["paths"]
    surfaces = sample["model"].surfaces
    examined = failed = mismatched = intended_no_root = 0
    max_newton_error = 0.0
    tol = 2e-4
    for ray in chosen:
        for destination in range(4):
            origin = paths[destination, ray]
            target = paths[destination+1, ray]
            delta = target-origin
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance) or distance <= 1e-8:
                failed += 1; mismatched += 1; examined += 1
                continue
            direction = delta/distance
            roots_by_patch = [_surface_line_roots(origin, direction, s, active_model)
                              for s in surfaces]
            all_hits = [(t, k) for k, roots in enumerate(roots_by_patch) for t in roots]
            first = min(all_hits, default=None)
            intended = roots_by_patch[destination]
            if not intended:
                intended_no_root += 1; mismatched += 1
            else:
                err = min(abs(t-distance) for t in intended)
                max_newton_error = max(max_newton_error, err)
                if err > tol: mismatched += 1
            if first is None or first[1] != destination or abs(first[0]-distance) > tol:
                failed += 1
            examined += 1
    passed = examined > 0 and failed == 0 and mismatched == 0
    return dict(method="ray line substituted into all four saved polynomial sag patches; real positive roots retained inside clear apertures",
                sampling=dict(fields=3, waves=3, pupils=3),
                fully_transmitted_rays_available=int(good.sum()), rays_examined=int(len(chosen)), segments_examined=int(examined),
                first_hit_failures=int(failed), newton_root_mismatches=int(mismatched),
                intended_patch_without_positive_root=int(intended_no_root),
                max_newton_hit_root_error_mm=_finite_float(max_newton_error),
                tolerance_mm=tol, passing_fraction=float((examined-failed)/examined) if examined else None,
                passed=bool(passed))


def audit(source):
    source_data, genes, active_model = _load_genes(source)
    tr = active_model.trace(genes, **FIELDS)
    good, xy = tr["good"], tr["xy"]
    # Dense sample dimensions are fixed by FIELDS and model.trace ordering.
    nf, nw, nr = good.shape
    counts = good.sum(axis=2)
    centroids = np.full((nf, nw, 2), np.nan)
    rms = np.full((nf, nw), np.nan)
    for fi in range(nf):
        for wi in range(nw):
            mask = good[fi, wi]
            if mask.any():
                pts = xy[fi, wi, mask]
                center = pts.mean(axis=0)
                centroids[fi, wi] = center
                rms[fi, wi] = np.sqrt(np.mean(np.sum((pts-center)**2, axis=1)))

    wavelengths_um = tr["wavelengths_mm"]*1000
    cells = []
    for fi, field in enumerate(tr["fields_deg"]):
        for wi, wave in enumerate(wavelengths_um):
            cells.append(dict(field_index=int(fi), field_deg=field.tolist(),
                              wavelength_um=float(wave), valid_rays=int(counts[fi, wi]),
                              launched_rays=int(nr), support=float(counts[fi, wi]/nr),
                              centroid_mm=centroids[fi, wi].tolist() if counts[fi, wi] else None,
                              rms_mm=_finite_float(rms[fi, wi])))
    supported = counts > 0
    maximum_rms = float(np.max(rms[supported])) if supported.any() else None
    support = float(good.mean())
    min_cell_support = float(np.min(counts/nr))

    # Fit image centroid slopes against tangent field angle, separately at each
    # wavelength. Invalid/empty cells remain absent rather than entering as zero.
    field_tan = np.tan(np.deg2rad(tr["fields_deg"]))
    slopes = np.full((nw, 2), np.nan)
    for wi in range(nw):
        for axis in range(2):
            sel = supported[:, wi]
            x, y = field_tan[sel, axis], centroids[sel, wi, axis]
            if len(x) and float(x@x) > 1e-15:
                slopes[wi, axis] = float(x@y/(x@x))
    finite_slopes = slopes[np.isfinite(slopes)]
    parity = np.ones(2)
    for axis in range(2):
        vals = slopes[:, axis][np.isfinite(slopes[:, axis])]
        if len(vals): parity[axis] = 1.0 if np.median(vals) >= 0 else -1.0
    focal_values = np.abs(finite_slopes)
    implied_focal = float(np.median(focal_values)) if len(focal_values) else None
    distortion = []
    for fi in range(nf):
        target = parity*EXPECTED_FOCAL_MM*field_tan[fi]
        for wi in range(nw):
            if supported[fi, wi]:
                distortion.append(float(np.linalg.norm(centroids[fi, wi]-target)))
    max_distortion = max(distortion) if distortion else None
    focal_by_axis = [np.abs(slopes[:, axis][np.isfinite(slopes[:, axis])]) for axis in range(2)]
    focal_axis_medians = [float(np.median(v)) if len(v) else None for v in focal_by_axis]
    focal_axis_ok = bool(np.all(np.isfinite(slopes)) and np.all(np.abs(np.abs(slopes)-EXPECTED_FOCAL_MM) <= .1*EXPECTED_FOCAL_MM))

    # Check material body over a dense, full nominal 20 x 10 mm footprint.
    gx, gy = np.meshgrid(np.linspace(-10, 10, BODY_GRID[0]),
                         np.linspace(-5, 5, BODY_GRID[1]))
    front, back = active_model.body(tr["model"], gx, gy)
    gap = back-front
    e, m1, m2, ex = tr["model"].surfaces
    body_metrics = dict(grid_x=BODY_GRID[0], grid_y=BODY_GRID[1],
                        footprint_x_mm=20.0, footprint_y_mm=10.0,
                        axial_envelope_mm=_finite_float(np.max(back)-np.min(front)),
                        max_local_thickness_mm=_finite_float(np.max(gap)),
                        min_local_gap_mm=_finite_float(np.min(gap)),
                        entrance_to_m2_clearance_mm=_finite_float(m2.center[0]-m2.radius-e.center[0]-e.radius),
                        m1_to_exit_clearance_mm=_finite_float(ex.center[0]-ex.radius-m1.center[0]-m1.radius))

    center_field_index = 3*7+3
    waist = [_waist_report(tr, center_field_index, wi) for wi in range(nw)]
    root_check = _independent_root_check(genes, active_model)
    core_idx = wavelengths_um <= 12.0+1e-9
    extra_idx = wavelengths_um > 12.0+1e-9
    def band_report(mask):
        band_counts = counts[:, mask]
        band_rms = rms[:, mask]
        return dict(wavelengths_um=wavelengths_um[mask].tolist(),
                    support=float(good[:, mask].mean()) if mask.any() else None,
                    min_field_wavelength_support=float(np.min(band_counts/nr)) if mask.any() else None,
                    empty_field_wavelength_cells=int(np.sum(band_counts == 0)),
                    max_supported_cell_rms_mm=_finite_float(np.max(band_rms[band_counts > 0]))
                    if np.any(band_counts > 0) else None)
    core, extrapolated = band_report(core_idx), band_report(extra_idx)

    # One field centroid across all wavelengths captures chromatic image shift.
    poly_rms = np.full(nf, np.nan)
    for fi in range(nf):
        pts = xy[fi][good[fi]]
        if len(pts):
            center = pts.mean(axis=0)
            poly_rms[fi] = np.sqrt(np.mean(np.sum((pts-center)**2, axis=1)))
    max_poly_rms = float(np.max(poly_rms[np.isfinite(poly_rms)])) if np.isfinite(poly_rms).any() else None

    entrance_diameter = 2*tr["model"].surfaces[0].radius
    f_number = implied_focal/entrance_diameter if implied_focal is not None else None
    gates = dict(max_rms_le_17um=maximum_rms is not None and maximum_rms <= RMS_LIMIT_MM,
                 max_polychromatic_rms_le_17um=max_poly_rms is not None and max_poly_rms <= RMS_LIMIT_MM,
                 total_support_ge_95pct=support >= SUPPORT_LIMIT,
                 minimum_cell_support_ge_95pct=min_cell_support >= SUPPORT_LIMIT,
                 every_field_wavelength_cell_has_support=bool(np.all(counts > 0)),
                 body_axial_envelope_le_10mm=body_metrics["axial_envelope_mm"] is not None and body_metrics["axial_envelope_mm"] <= BODY_THICKNESS_LIMIT_MM,
                 positive_local_body_gap=body_metrics["min_local_gap_mm"] is not None and body_metrics["min_local_gap_mm"] > 0,
                 focal_length_within_10pct=implied_focal is not None and abs(implied_focal-EXPECTED_FOCAL_MM) <= .1*EXPECTED_FOCAL_MM and focal_axis_ok,
                 real_intermediate_waist_all_wavelengths=all(w["real"] for w in waist),
                 sparse_independent_first_root_check_passed=root_check["passed"])
    return dict(source=str(Path(source)), source_seed=source_data.get("seed"),
                representation=source_data.get("representation"), sampling=dict(**FIELDS),
                launched_rays=int(good.size), valid_rays=int(good.sum()),
                support=support, minimum_cell_support=min_cell_support,
                empty_field_wavelength_cells=int(np.sum(counts == 0)),
                max_supported_cell_rms_mm=maximum_rms,
                optical_mapping=dict(expected_focal_mm=EXPECTED_FOCAL_MM,
                    fitted_implied_focal_length_mm=implied_focal,
                    fitted_focal_length_by_axis_mm=focal_axis_medians,
                    fitted_slopes_mm_per_tan_field=slopes.tolist(),
                    signed_image_parity=parity.tolist(),
                    entrance_diameter_mm=float(entrance_diameter),
                    f_number=f_number,
                    max_distortion_vs_signed_6mm_mapping_mm=max_distortion,
                    interpretation="centroid least-squares slopes versus tan(field); distortion compares only to signed 6 mm mapping, without refitting focal length"),
                body=body_metrics, intermediate_waist_by_wavelength=waist,
                spectral_bands=dict(core_8_to_12_um=core,
                                    extrapolated_above_12_to_14_um=extrapolated),
                max_polychromatic_field_rms_mm=max_poly_rms,
                polychromatic_field_rms_mm=[_finite_float(x) for x in poly_rms],
                sparse_independent_intersection_root_check=root_check,
                cells=cells, gates=gates,
                passes_all_gates=bool(all(gates.values())),
                independent_intersection_root_validation="sparse fully transmitted rays checked against all positive in-aperture roots on all four patches; not exhaustive",
                limitations=["geometric ray audit only; MTF, straylight, coatings, thermal behavior, and manufacturing tolerances are not tested",
                             "IG4 index values above 12 um are extrapolated and reported separately"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="search/polish result JSON containing genes")
    parser.add_argument("--out", required=True, help="audit JSON destination")
    args = parser.parse_args(argv)
    result = audit(args.source)
    _write(args.out, result)


if __name__ == "__main__":
    main()



