"""Nested Cartesian freeform expansion of the frozen radial camera family.

The existing fourteen parameters remain active. XY polynomial blocks are added
by total degree, on primary, secondary and corrector. Zero extension preserves
the exact previous optical surfaces. Lengths and polynomial amplitudes are mm.
"""
from __future__ import annotations
import numpy as np
from . import model as radial

BASE_DIM = radial.N_TOTAL
SURFACES = ('primary', 'secondary', 'plate')
TERMS_BY_DEGREE = {degree: tuple((i, degree-i) for i in range(degree, -1, -1))
                   for degree in (2, 3, 4)}
# Order is degree, surface, monomial: every stage is an exact prefix.
PARAMETERS = tuple((name, i, j) for degree in (2, 3, 4) for name in SURFACES
                   for i, j in TERMS_BY_DEGREE[degree])
N_TOTAL = BASE_DIM + len(PARAMETERS)
STAGES = (('quadratic_xy', 23, 3, 4), ('cubic_xy', 35, 3, 6), ('quartic_xy', 50, 5, 6))
AMPLITUDES_MM = {'primary': .060, 'secondary': .030, 'plate': .030}
NORMALIZATION_MM = {'primary': radial.APERTURE_DIAMETER_MM/2,
                    'secondary': 3., 'plate': radial.APERTURE_DIAMETER_MM/2}


def pad(g):
    g = np.asarray(g, float)
    if g.ndim != 1 or len(g) < BASE_DIM or len(g) > N_TOTAL or not np.isfinite(g).all():
        raise ValueError('expected finite 14..50 element gene vector')
    return np.pad(g, (0, N_TOTAL-len(g)))


def sag_hessian(x, y, surface):
    """Analytic second derivatives for physical-surface curvature screening."""
    _, c, k, desc = surface
    x, y = np.asarray(x), np.asarray(y)
    r2 = x*x+y*y
    q = 1-(1+k)*c*c*r2
    root = np.sqrt(np.maximum(q, 1e-12))
    a = c/root
    b = (1+k)*c**3/root**3
    hxx, hyy, hxy = a+b*x*x, a+b*y*y, b*x*y
    radial_coeff = desc['radial'] if isinstance(desc, dict) else desc
    for j, coefficient in enumerate(radial_coeff):
        exponent = 2+j
        hxx = hxx + 2*exponent*coefficient*r2**(exponent-1) + 4*exponent*(exponent-1)*coefficient*x*x*r2**(exponent-2)
        hyy = hyy + 2*exponent*coefficient*r2**(exponent-1) + 4*exponent*(exponent-1)*coefficient*y*y*r2**(exponent-2)
        hxy = hxy + 4*exponent*(exponent-1)*coefficient*x*y*r2**(exponent-2)
    if isinstance(desc, dict):
        scale = desc['scale']
        u, v = x/scale, y/scale
        for i, j, coefficient in desc['xy']:
            if coefficient == 0:
                continue
            if i >= 2:
                hxx = hxx+coefficient*i*(i-1)*u**(i-2)*v**j/scale**2
            if j >= 2:
                hyy = hyy+coefficient*j*(j-1)*u**i*v**(j-2)/scale**2
            if i and j:
                hxy = hxy+coefficient*i*j*u**(i-1)*v**(j-1)/scale**2
    return hxx, hxy, hyy


def geometry_metrics(m, rings=9, azimuths=32):
    """Conservative sampled z-interval gaps, slopes and mirror curvature signs."""
    a = np.arange(azimuths)*2*np.pi/azimuths
    r = np.linspace(0, 1, rings)
    u, v = (r[:, None]*np.cos(a)).ravel(), (r[:, None]*np.sin(a)).ravel()
    metrics = {}
    for name in SURFACES:
        radius = m['secondary_radius'] if name == 'secondary' else m['primary_radius']
        x, y = u*radius, v*radius
        keep = np.hypot(x, y) >= m['hole_radius'] if name == 'primary' else np.ones(len(x), bool)
        x, y = x[keep], y[keep]
        z0, c, k, coeff = m[name]
        z, zx, zy = radial.sag_xy(x, y, c, k, coeff)
        hxx, hxy, hyy = sag_hessian(x, y, m[name])
        upper = (hxx+hyy)/2+np.sqrt(((hxx-hyy)/2)**2+hxy*hxy)
        desc = coeff if isinstance(coeff, dict) else None
        radial_z = radial._radial(np.hypot(x, y), c, k, desc['radial'] if desc else coeff)[0]
        metrics[name] = dict(z_min_mm=float(np.min(z0+z)), z_max_mm=float(np.max(z0+z)),
                             max_slope=float(np.max(np.hypot(zx, zy))),
                             max_hessian_eigenvalue_per_mm=float(np.max(upper)),
                             max_freeform_departure_mm=float(np.max(abs(z-radial_z))))
    gaps = dict(minimum_plate_thickness_mm=metrics['plate']['z_min_mm'],
                minimum_secondary_plate_gap_mm=metrics['secondary']['z_min_mm']-metrics['plate']['z_max_mm'],
                minimum_primary_secondary_gap_mm=metrics['primary']['z_min_mm']-metrics['secondary']['z_max_mm'])
    depths = [0., m['detector_z']] + [row[key] for row in metrics.values() for key in ('z_min_mm', 'z_max_mm')]
    return dict(surfaces=metrics, **gaps, optical_depth_mm=max(depths)-min(depths),
                sampled_only=True, rings=rings, azimuths=azimuths)


def decode(g):
    g = pad(g)
    m = radial.decode(g[:BASE_DIM])
    for name in SURFACES:
        z, c, k, coefficients = m[name]
        xy = [(i, j, float(amplitude)*AMPLITUDES_MM[name])
              for amplitude, (surface, i, j) in zip(g[BASE_DIM:], PARAMETERS) if surface == name]
        m[name] = (z, c, k, dict(radial=coefficients, xy=xy, scale=NORMALIZATION_MM[name]))
    geom = geometry_metrics(m)
    for key in ('minimum_plate_thickness_mm', 'minimum_secondary_plate_gap_mm',
                'minimum_primary_secondary_gap_mm', 'optical_depth_mm'):
        m[key] = geom[key]
    m['freeform_geometry'] = geom
    m['corrector_max_slope'] = geom['surfaces']['plate']['max_slope']
    m['representation'] = 'conic + radial aspheres + normalized XY monomials, degree 2 through 4'
    return m


def trace(g, fields=3, waves=1, pupils=5, **kwargs):
    g = pad(g)
    return radial.trace(g[:BASE_DIM], fields, waves, pupils, decoded_model=decode(g), **kwargs)


def evaluate(g, fields=3, waves=1, pupils=5, **kwargs):
    g = pad(g)
    m = decode(g)
    result = radial.evaluate(g[:BASE_DIM], fields, waves, pupils, decoded_model=m, **kwargs)
    support = np.asarray(result['per_field_throughput'])
    # Same merit for every freeform stage and the matched radial control.
    penalty = 150*np.mean(np.maximum(.50-support, 0)**2)
    penalty += .5*max(result['max_spot_mm']/radial.PIXEL_PITCH_MM-1, 0)**2
    geometry_penalty = 0.
    for name, row in m['freeform_geometry']['surfaces'].items():
        geometry_penalty += 100*max(row['max_slope']-(.3 if name == 'plate' else 1.), 0)**2
    result['radial_merit_component'] = result['loss']
    result['loss'] += float(penalty+geometry_penalty)
    result['geometry_penalty'] = float(geometry_penalty)
    result['freeform_geometry'] = m['freeform_geometry']
    return result


# Shared constants/utilities for the camera's audits and renderers.
PIXEL_PITCH_MM = radial.PIXEL_PITCH_MM
APERTURE_DIAMETER_MM = radial.APERTURE_DIAMETER_MM
N_CORRECTOR = radial.N_CORRECTOR
_grid = radial._grid
_radial = radial._radial
sag_xy = radial.sag_xy
snell = radial.snell
unit = radial.unit

