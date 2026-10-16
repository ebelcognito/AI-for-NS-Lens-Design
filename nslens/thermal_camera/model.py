"""3D geometric Schmidt-Cassegrain channel, lengths in mm.

Engineering scaffold, not a published prescription. Constant silicon index is
an approximation. Every launched disk sample enters the throughput denominator.
"""
from __future__ import annotations
import numpy as np

N_BASE, N_TOTAL = 8, 14
PIXEL_PITCH_MM = .012
EFL_TARGET_MM = 16 * PIXEL_PITCH_MM * 180 / np.pi
APERTURE_DIAMETER_MM = EFL_TARGET_MM / 1.1
FIELD_X_DEG, FIELD_Y_DEG = 4.5, 3.5
WAVELENGTHS_MM = (.008, .011, .014)
N_CORRECTOR = 3.42


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def reflect(d, n):
    n = unit(n)
    return unit(d - 2 * np.sum(d*n, axis=-1, keepdims=True)*n)


def snell(d, n, n1, n2):
    """Normals are automatically oriented along incidence; false means TIR."""
    d, n = unit(d), unit(n)
    n = np.where((np.sum(d*n, axis=-1) < 0)[..., None], -n, n)
    eta = np.asarray(n1) / np.asarray(n2)
    cosi = np.sum(d*n, axis=-1)
    k = 1 - eta**2 * (1-cosi**2)
    q = eta[..., None]*(d-cosi[..., None]*n) + np.sqrt(np.maximum(k, 0))[..., None]*n
    return unit(q), k >= 0


def _radial(r, curvature, conic, coeff):
    r = np.asarray(r, float)
    q = 1 - (1+conic)*curvature**2*r*r
    root = np.sqrt(np.maximum(q, 1e-12))
    z, dz = curvature*r*r/(1+root), curvature*r/root
    for j, a in enumerate(coeff):
        p = 4+2*j
        z, dz = z+a*r**p, dz+p*a*r**(p-1)
    return z, dz


def sag_xy(x, y, curvature, conic, coeff):
    """Conic/radial sag plus optional normalized Cartesian freeform terms.

    A dictionary coefficient descriptor carries radial coefficients, a fixed
    normalization radius, and explicit (x power, y power, sag amplitude in mm).
    Analytic derivatives are with respect to physical millimetres.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    radial = coeff['radial'] if isinstance(coeff, dict) else coeff
    r = np.hypot(x, y)
    z, dr = _radial(r, curvature, conic, radial)
    zx, zy = dr*x/np.maximum(r, 1e-30), dr*y/np.maximum(r, 1e-30)
    if isinstance(coeff, dict):
        scale = float(coeff['scale'])
        if scale <= 0:
            raise ValueError('freeform normalization scale must be positive')
        u, v = x/scale, y/scale
        for i, j, amplitude in coeff['xy']:
            if amplitude == 0:
                continue
            z = z + amplitude*u**i*v**j
            if i:
                zx = zx + amplitude*i*u**(i-1)*v**j/scale
            if j:
                zy = zy + amplitude*j*u**i*v**(j-1)/scale
    return z, zx, zy


def _surface(o, d, zv, curvature, conic, coeff, iterations=7):
    """Local Newton intersection; dense audit does not certify global roots."""
    with np.errstate(all='ignore'):
        t = (zv-o[:, 2]) / np.where(abs(d[:, 2]) > 1e-12, d[:, 2], 1e-12)
        for _ in range(iterations):
            p = o+t[:, None]*d
            r = np.hypot(p[:, 0], p[:, 1])
            z, zx, zy = sag_xy(p[:, 0], p[:, 1], curvature, conic, coeff)
            deriv = d[:, 2]-zx*d[:, 0]-zy*d[:, 1]
            t -= np.clip((p[:, 2]-zv-z)/np.where(abs(deriv) > 1e-12, deriv, 1e-12), -20, 20)
        p = o+t[:, None]*d
        r = np.hypot(p[:, 0], p[:, 1])
        z, zx, zy = sag_xy(p[:, 0], p[:, 1], curvature, conic, coeff)
        n = unit(np.column_stack([-zx, -zy, np.ones(len(r))]))
        valid = (np.isfinite(t) & (t > 1e-7) & (abs(p[:, 2]-zv-z) < 1e-7)
                 & (1-(1+conic)*curvature**2*r*r > 0))
    return p, n, t, valid, r


def decode(g):
    a = np.zeros(N_TOTAL)
    src = np.asarray(g, float).ravel()
    if len(src) > N_TOTAL or not np.isfinite(src).all():
        raise ValueError('genes must be finite and have at most N_TOTAL entries')
    a[:len(src)] = src
    L, s, back = 6.1+a[0], 1.4+.3*a[1], .5+.3*a[7]
    # Analytic paraxial scaffold, then independent curvature perturbations.
    separation = L-s
    v = separation+back
    f1 = EFL_TARGET_MM*separation/(EFL_TARGET_MM-v)
    u = f1-separation
    cp = -1/(2*f1)*(1+.18*a[2])
    cs = -(1/u-1/v)/2*(1+.3*a[3])
    radius = APERTURE_DIAMETER_MM/2
    secondary_radius = radius*u/f1+.45+.35*a[4]
    hole_radius = 1.20+.40*a[5]
    plate_t = .45+.15*a[6]
    primary = (L, cp, -1+2*a[8], np.zeros(3))
    secondary = (s, cs, -2+4*a[12], np.array([3e-4*a[13], 0., 0.]))
    plate = (plate_t, 0., 0., np.array([3e-5*a[9], 5e-7*a[10], 8e-9*a[11]]))
    rr = np.linspace(0, radius, 65)
    zp, _ = _radial(rr, *primary[1:])
    zc, dc = _radial(rr, *plate[1:])
    rs = np.linspace(0, secondary_radius, 65)
    zs, _ = _radial(rs, *secondary[1:])
    depth = max(L+back, float(np.max(L+zp))) - min(0., float(np.min(plate_t+zc)))
    return dict(primary=primary, secondary=secondary, plate=plate,
                primary_vertex_z=L, secondary_vertex_z=s, detector_z=L+back,
                primary_radius=radius, secondary_radius=secondary_radius,
                hole_radius=hole_radius, plate_thickness=plate_t, back_focus=back,
                efl_target_mm=EFL_TARGET_MM, aperture_diameter_mm=2*radius,
                index=N_CORRECTOR, nominal_f_number=1.1, optical_depth_mm=depth,
                minimum_plate_thickness_mm=float(np.min(plate_t+zc)),
                corrector_max_slope=float(np.max(abs(dc))),
                minimum_secondary_plate_gap_mm=float(np.min(s+zs)-np.max(plate_t+zc)),
                minimum_primary_secondary_gap_mm=float(np.min(L+zp)-np.max(s+zs)),
                assumption='paraxial two-mirror scaffold; constant-index silicon; unpublished prescription')


def _grid(count):
    """Equal-area annular-sector midpoint quadrature; no central/rim bias."""
    if int(count) != count or count < 1:
        raise ValueError('pupil sampling count must be a positive integer')
    radial = np.sqrt((np.arange(count)+.5)/count)
    theta = 2*np.pi*(np.arange(max(8, 4*count))+.5)/max(8, 4*count)
    return np.column_stack([(radial[:, None]*np.cos(theta)).ravel(),
                            (radial[:, None]*np.sin(theta)).ravel()])


def trace(g, fields=3, waves=3, pupils=5, field_xy=None, pupil_xy=None,
          entrance_o=None, entrance_d=None, decoded_model=None):
    m = decode(g) if decoded_model is None else decoded_model
    pup = _grid(pupils) if pupil_xy is None else np.asarray(pupil_xy, float)
    if pup.ndim != 2 or pup.shape[1] != 2 or not len(pup) or not np.isfinite(pup).all() or np.any(np.sum(pup*pup, axis=1) > 1+1e-10):
        raise ValueError('pupil_xy must contain points inside the unit disk')
    if field_xy is None:
        if fields < 1 or waves < 1:
            raise ValueError('sampling counts must be positive')
        fx = np.linspace(-FIELD_X_DEG, FIELD_X_DEG, fields) if fields > 1 else np.array([0.])
        fy = np.linspace(-FIELD_Y_DEG, FIELD_Y_DEG, fields) if fields > 1 else np.array([0.])
        x, y = np.meshgrid(fx, fy)
        field = np.column_stack([x.ravel(), y.ravel()])
    else:
        field = np.asarray(field_xy, float)
    wl = np.linspace(WAVELENGTHS_MM[0], WAVELENGTHS_MM[-1], waves) if waves > 1 else np.array([.011])
    nf, nw, npup = len(field), len(wl), len(pup)
    shape = (nf, nw, npup)
    N = nf*nw*npup
    fld = np.repeat(field, nw*npup, axis=0)
    d = unit(np.column_stack([np.tan(np.deg2rad(fld)), np.ones(N)]))
    xy = np.tile(pup*m['primary_radius'], (nf*nw, 1))
    o = np.column_stack([xy, np.zeros(N)]) - d/d[:, 2, None]
    if (entrance_o is None) != (entrance_d is None):
        raise ValueError('supply both entrance_o and entrance_d')
    if entrance_o is not None:
        o, d = np.asarray(entrance_o, float), unit(entrance_d)
        if o.shape != (N, 3) or d.shape != (N, 3):
            raise ValueError('injected entrance arrays must match sampling')
    good = np.ones(N, bool)
    paths = [o.copy()]
    obstruction = np.zeros(N, bool)
    overflow = np.zeros(N)
    for surface, ni, nt in [((0., 0., 0., []), 1., m['index']), (m['plate'], m['index'], 1.)]:
        p, n, t, valid, r = _surface(o, d, *surface)
        good &= valid & (r <= m['primary_radius'])
        overflow += np.maximum(r-m['primary_radius'], 0)
        d, refracted = snell(d, n, ni, nt)
        good &= refracted
        o = p
        paths.append(p.copy())
    _, _, ts, vs, rs = _surface(o, d, *m['secondary'])
    p, n, tp, valid, r = _surface(o, d, *m['primary'])
    blocked = vs & (ts < tp) & (rs <= m['secondary_radius'])
    blocked |= valid & (r < m['hole_radius'])
    obstruction |= blocked
    good &= valid & ~blocked & (r <= m['primary_radius'])
    overflow += np.maximum(r-m['primary_radius'], 0)
    d = reflect(d, n)
    o = p
    paths.append(p.copy())
    p, n, t, valid, r = _surface(o, d, *m['secondary'])
    good &= valid & (r <= m['secondary_radius'])
    overflow += np.maximum(r-m['secondary_radius'], 0)
    d = reflect(d, n)
    o = p
    paths.append(p.copy())
    p, n, t, valid, r = _surface(o, d, *m['primary'])
    blocked = valid & (r >= m['hole_radius'])
    good &= valid & ~blocked
    obstruction |= blocked
    overflow += np.maximum(r-m['hole_radius'], 0)
    o = p
    paths.append(p.copy())
    with np.errstate(all='ignore'):
        t = (m['detector_z']-o[:, 2])/d[:, 2]
        image = o+t[:, None]*d
    good &= np.isfinite(t) & (t > 0) & np.isfinite(image).all(axis=1)
    paths.append(image.copy())
    physical = (m['minimum_plate_thickness_mm'] > .05 and
                m['minimum_secondary_plate_gap_mm'] > .05 and
                m['minimum_primary_secondary_gap_mm'] > .05)
    good &= physical
    return dict(xy=image[:, :2].reshape(*shape, 2), good=good.reshape(shape),
                paths=np.asarray(paths), obstructed=obstruction.reshape(shape),
                fields_deg=field, wavelengths_mm=wl, model=m, launched=N,
                pupil_xy=pup, aperture_overflow_mm=overflow.reshape(shape),
                physical_geometry=physical)


def evaluate(g, fields=3, waves=3, pupils=5, **kwargs):
    with np.errstate(all='ignore'):
        r = trace(g, fields, waves, pupils, **kwargs)
        xy, good = r['xy'], r['good']
        count = good.sum(axis=2)
        cent = np.sum(np.where(good[..., None], xy, 0), axis=2)/np.maximum(count[..., None], 1)
        spot = np.sqrt(np.sum(np.where(good, np.sum((xy-cent[:, :, None])**2, axis=-1), 0), axis=2)/np.maximum(count, 1))
        ideal = np.tan(np.deg2rad(r['fields_deg']))[:, None, :]*EFL_TARGET_MM
        parity = 1 if np.sum((cent-ideal)**2) <= np.sum((cent+ideal)**2) else -1
        mapping = np.linalg.norm(cent-parity*ideal, axis=-1)
        throughput = float(good.mean())
        support = count/good.shape[2]
        loss = np.mean(np.minimum(spot/PIXEL_PITCH_MM, 100)**2)
        loss += .05*np.mean(np.minimum(mapping/PIXEL_PITCH_MM, 100)**2)
        loss += 80*np.mean(np.maximum(.55-support, 0)**2)
        loss += 200*np.mean(count == 0)+2*(1-throughput)
        loss += 5*np.mean(np.minimum(r['aperture_overflow_mm'], 10)**2)
        m = r['model']
        loss += 100*max(m['optical_depth_mm']-10, 0)**2
        for key in ('minimum_plate_thickness_mm', 'minimum_secondary_plate_gap_mm', 'minimum_primary_secondary_gap_mm'):
            loss += 1000*max(.05-m[key], 0)**2
        theta = np.broadcast_to(np.tan(np.deg2rad(r['fields_deg']))[:, None, :], cent.shape)
        selected = count > 0
        ef = abs(float(np.sum(theta[selected]*cent[selected])/max(np.sum(theta[selected]**2), 1e-30)))
    return dict(loss=float(loss) if np.isfinite(loss) else 1e8,
                throughput=throughput, min_throughput=float(support.min()),
                empty_field_wave_cells=int(np.sum(count == 0)), imaging_complete=bool(np.all(count > 0)),
                rms_spot_mm=float(np.sqrt(np.mean(spot**2))), max_spot_mm=float(np.max(spot)),
                rms_spot_pixels=float(np.sqrt(np.mean(spot**2))/PIXEL_PITCH_MM),
                mapping_rms_mm=float(np.sqrt(np.mean(mapping**2))),
                image_parity=parity, fitted_efl_mm=ef,
                fitted_f_number=ef/APERTURE_DIAMETER_MM,
                area_equivalent_f_number=1.1/np.sqrt(throughput) if throughput > 0 else None,
                obstruction_fraction=float(r['obstructed'].mean()),
                optical_depth_mm=m['optical_depth_mm'], physical_geometry=r['physical_geometry'],
                centroids_mm=cent.tolist(), per_field_spot_mm=spot.tolist(),
                per_field_throughput=support.tolist(), launched=r['launched'])

