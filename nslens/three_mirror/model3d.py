"""Three-mirror layout/launch adapter for the existing spectrometer 3D physics.

Local surface coordinates are (u, v, sag), in mm. Surface coefficients multiply
(u/30)**i * (v/30)**j and therefore have units mm. No symmetry is imposed.
The finite mirrors are zero-thickness circular disks in local (u, v).
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
from nslens.spectrometer import model as optics
from nslens.spectrometer.intersections import intersect as fast_intersect

SCALE = 30.0
POWERS = tuple((degree-j, j) for degree in range(2, 7)
               for j in range(degree+1))


def unit(v):
    v = np.asarray(v, float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm <= 0) or not np.all(np.isfinite(norm)):
        raise ValueError("Directions must be finite and nonzero")
    return v/norm


def lift_sample(sample, sagittal="equal_curvature"):
    """Preserve the 2D slice; explicitly choose the unmeasured v² curvature.

    equal_curvature sets the v² coefficient equal to u². cylindrical sets it
    to zero. Neither is an optimized 3D prescription or inferred from 2D data.
    Existing u³ through u⁶ terms are copied to mirror 3 only.
    """
    if sagittal not in ("equal_curvature", "cylindrical"):
        raise ValueError("Unknown sagittal initialization")
    if sample['model'] == 'legacy_three_mirror':
        from .pilot import decode
        base = decode(sample['genes'])
    elif sample['model'] == 'variable_mirror_n3':
        from nslens.analysis.variable_mirror import decode
        base = decode(sample['genes'], 3)
    else:
        raise ValueError("Expected a saved three-mirror model")
    embed = lambda x: np.concatenate((np.asarray(x, float), [0.]))
    surfaces = []
    for j in range(3):
        coeff = np.zeros(len(POWERS))
        coeff[POWERS.index((2, 0))] = base['curves'][j]*SCALE**2
        if sagittal == 'equal_curvature':
            coeff[POWERS.index((0, 2))] = base['curves'][j]*SCALE**2
        if j == 2:
            for degree, value in enumerate(sample.get('freeform_coefficients_mm', [0.]*4), 3):
                coeff[POWERS.index((degree, 0))] = value
        surfaces.append(dict(vertex=embed(base['vertices'][j]),
                             basis=np.column_stack((embed(base['tangents'][j]),
                                                    [0., 0., 1.], embed(base['normals'][j]))),
                             coefficients=coeff, radius=float(base['widths'][j])))
    return dict(surfaces=surfaces, detector=dict(
        vertex=embed(base['image']), basis=np.column_stack((
            embed(base['image_tangent']), [0., 0., 1.], embed(base['image_normal'])))),
        initialization=sagittal, source_id=sample.get('id'), task=deepcopy(sample['task']))


def optical_surface(surface):
    """Convert normalized design coefficients into the existing engine's units."""
    return dict(vertex=surface['vertex'], basis=surface['basis'],
                aperture=surface['radius'], terms=POWERS,
                coeff=np.asarray(surface['coefficients']) /
                np.array([SCALE**(i+j) for i, j in POWERS]))


def surface_sag(surface, uv, gradient=False):
    uv = np.asarray(uv, float)
    s = optical_surface(surface)
    z, zx, zy = optics.sag(uv[..., 0], uv[..., 1], s['coeff'], s['terms'])
    return (z, np.stack((zx, zy), axis=-1)) if gradient else z


def surface_normal(surface, points):
    uv = ((np.asarray(points)-surface['vertex']) @ surface['basis'])[..., :2]
    _, slope = surface_sag(surface, uv, gradient=True)
    local = np.concatenate((-slope, np.ones(slope.shape[:-1]+(1,))), axis=-1)
    return optics.unit(local @ surface['basis'].T)


def intersect_surface(origins, directions, surface):
    """Use the shared batched all-root solver for nearest-hit routing."""
    _, _, t, valid = fast_intersect(origins, directions, optical_surface(surface))
    return np.where(valid, t, np.inf)


def pose_delta(layout, component, translation_mm=(0., 0., 0.), rotation_rad=(0., 0., 0.)):
    """Rigid world-axis motion about this component's vertex, without scaffolding."""
    result = deepcopy(layout)
    obj = result['detector'] if component == 'detector' else result['surfaces'][component]
    shift, rot = np.asarray(translation_mm, float), np.asarray(rotation_rad, float)
    if shift.shape != (3,) or rot.shape != (3,) or not np.isfinite([shift, rot]).all():
        raise ValueError("Pose changes require finite XYZ triples")
    obj['vertex'] = obj['vertex']+shift
    obj['basis'] = Rotation.from_rotvec(rot).as_matrix() @ obj['basis']
    return result


def trace_rays(layout, origins, directions, *, intersect_fn=None):
    """Trace nearest mirror/detector events; report every launched ray.

    Good rays visit each mirror exactly once, in any order, then the forward
    side of the infinite detector. Route is returned for fixed-route refinement.
    """
    o = np.array(origins, float, copy=True)
    d = unit(directions).copy()
    if o.ndim != 2 or o.shape[1] != 3 or o.shape != d.shape or not np.isfinite(o).all():
        raise ValueError("Expected matching finite Nx3 ray arrays")
    n = len(layout['surfaces'])
    if n != 3: raise ValueError("This model requires exactly three mirrors")
    count = len(o)
    active = np.ones(count, bool)
    hits = np.zeros(count, int)
    route = np.full((count, n+3), -1, int)
    image = np.full((count, 2), np.nan)
    detector = layout['detector']
    fn = fast_intersect if intersect_fn is None else intersect_fn
    surfaces = [optical_surface(s) for s in layout['surfaces']]
    for step in range(n+3):
        ids = np.flatnonzero(active)
        if not len(ids): break
        distances = []
        for surface in surfaces:
            _, _, t, valid = fn(o[ids], d[ids], surface)
            distances.append(np.where(valid, t, np.inf))
        distances = np.column_stack(distances)
        which = np.argmin(distances, axis=1)
        tm = distances[np.arange(len(ids)), which]
        den = d[ids] @ detector['basis'][:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = ((detector['vertex']-o[ids]) @ detector['basis'][:, 2])/den
        plane_hit = (ti > 1e-7) & (ti < tm)
        if 'bounds_mm' in detector:
            # A finite opaque detector blocks from either side. Only its
            # forward face can record signal; its footprint stays fixed.
            with np.errstate(invalid='ignore'):
                uv = (o[ids]+ti[:, None]*d[ids]-detector['vertex']) @ detector['basis'][:, :2]
            bounds = np.asarray(detector['bounds_mm'])
            intercepted = plane_hit & np.all(uv >= bounds[0], axis=1) & np.all(uv <= bounds[1], axis=1)
            sensor = intercepted & (den > 0)
        else:
            sensor = plane_hit & (den > 0)
            intercepted = sensor
        si = ids[sensor]
        points = o[si]+ti[sensor, None]*d[si]
        image[si] = (points-detector['vertex']) @ detector['basis'][:, :2]
        reflect = ~intercepted & np.isfinite(tm)
        active[ids] = reflect
        for j, surface in enumerate(layout['surfaces']):
            mask = reflect & (which == j)
            ri = ids[mask]
            if not len(ri): continue
            points = o[ri]+tm[mask, None]*d[ri]
            normal = surface_normal(surface, points)
            d[ri] = optics.reflect(d[ri], normal)
            o[ri] = points
            hits[ri] += 1
            route[ri, step] = j
    good = (np.isfinite(image).all(axis=1) & (hits == n)
            & np.all(np.sort(route[:, :n], axis=1) == np.arange(n), axis=1))
    return dict(image_mm=image, good=good, route=route, hits=hits,
                launched=count, received=int(good.sum()))


def launch_rays(layout, field_angles_deg, pupil_xy_mm):
    """Cartesian field/pupil product; incoming chief direction is world +X.

    A conservative sag bound keeps the launch plane upstream of all surfaces.
    Pupil coordinates denote the intercepts at world X=0.
    """
    f, p = np.asarray(field_angles_deg, float), np.asarray(pupil_xy_mm, float)
    if f.ndim != 2 or p.ndim != 2 or f.shape[1] != 2 or p.shape[1] != 2 or not len(f) or not len(p):
        raise ValueError("Nonempty field and pupil arrays must have two columns")
    if not np.isfinite(f).all() or not np.isfinite(p).all() or np.any(abs(f) >= 89):
        raise ValueError("Finite pupils and field angles below 89 degrees required")
    minima = []
    for s in layout['surfaces']:
        bound = sum(abs(c)*(s['radius']/SCALE)**(i+j) for (i, j), c in zip(POWERS, s['coefficients']))
        radial = s['radius']*np.linalg.norm(s['basis'][0, :2])
        minima.append(s['vertex'][0]-radial-abs(s['basis'][0, 2])*bound)
    x0 = min(minima)-20.
    angles = np.repeat(f, len(p), axis=0)
    pupils = np.tile(p, (len(f), 1))
    tangent = np.tan(np.deg2rad(angles))
    origins = np.column_stack((np.full(len(angles), x0), pupils+x0*tangent))
    directions = unit(np.column_stack((np.ones(len(angles)), tangent)))
    return origins, directions


def evaluate(layout, field_angles_deg, pupil_xy_mm, focal_mm=None):
    """Geometric radial RMS about each field centroid, plus mapping diagnostics."""
    fields, pupils = np.asarray(field_angles_deg), np.asarray(pupil_xy_mm)
    ray = trace_rays(layout, *launch_rays(layout, fields, pupils))
    xy = ray['image_mm'].reshape(len(fields), len(pupils), 2)
    good = ray['good'].reshape(len(fields), len(pupils))
    rows = []
    focal = layout['task']['focal'] if focal_mm is None else focal_mm
    target = focal*np.tan(np.deg2rad(fields))
    for k, mask in enumerate(good):
        values = xy[k, mask]
        centroid = values.mean(axis=0) if len(values) else None
        spread = np.std(values, axis=0) if len(values) >= 2 else None
        rows.append(dict(field_deg=fields[k].tolist(), support=float(mask.mean()),
                         centroid_mm=centroid.tolist() if centroid is not None else None,
                         rms_xy_mm=spread.tolist() if spread is not None else None,
                         rms_spot_mm=float(np.linalg.norm(spread)) if spread is not None else None))
    # One shared parity per detector axis, never a fit per individual field.
    parity = []
    for axis in range(2):
        costs = [np.sum(np.where(good, (xy[..., axis]-sign*target[:, None, axis])**2, 0)) for sign in (-1, 1)]
        parity.append((-1, 1)[int(np.argmin(costs))])
    for k, row in enumerate(rows):
        values = xy[k, good[k]]
        error = values-target[k]*parity
        row['mapping_rms_mm'] = float(np.sqrt(np.mean(np.sum(error**2, axis=1)))) if len(values) else None
    complete = all(row['rms_spot_mm'] is not None for row in rows)
    return dict(fields=rows, worst_support=min(r['support'] for r in rows),
                geometric_survival=float(good.mean()), launched_rays=ray['launched'],
                worst_rms_spot_mm=max(r['rms_spot_mm'] for r in rows) if complete else None,
                rms_spot_mm=float(np.sqrt(np.mean([r['rms_spot_mm']**2 for r in rows]))) if complete else None,
                worst_mapping_rms_mm=max(r['mapping_rms_mm'] for r in rows) if complete else None,
                image_parity=parity, ray_image_mm=ray['image_mm'], ray_good=ray['good'], ray_route=ray['route'])


def field_grid(half_angles_deg=(4., 3.), counts=(5, 5)):
    axes = []
    for half, count in zip(half_angles_deg, counts):
        if not isinstance(count, (int, np.integer)) or count < 1 or not np.isfinite(half) or half < 0:
            raise ValueError("Nonnegative half angles and positive integer counts required")
        axes.append(np.linspace(-half, half, count) if count > 1 else np.array([0.]))
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def disk_pupil(diameter_mm=27.78, grid=17):
    if not np.isfinite(diameter_mm) or diameter_mm <= 0 or not isinstance(grid, (int, np.integer)) or grid < 3:
        raise ValueError("Positive diameter and integer grid >=3 required")
    u = np.linspace(-1., 1., grid)
    xy = np.stack(np.meshgrid(u, u, indexing='ij'), axis=-1).reshape(-1, 2)
    return xy[np.sum(xy*xy, axis=1) <= 1+1e-12]*diameter_mm/2
