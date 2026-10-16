"""Independent polynomial-root check for radial and XY-freeform candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial as P
from . import model as radial
from . import freeform_model as freeform
from . import spectrometer_style_model as spectrometer_style
from .search import write_json


def roots(o, d, surface):
    """Return sorted positive ray parameters satisfying the complete sag model."""
    zv, c, k, coeff = surface
    x, y, z = [P([a, b]) for a, b in zip(o, d)]
    r2 = x*x+y*y
    q = z-zv
    if isinstance(coeff, dict):
        radial_coeff = coeff['radial']
        for j, a in enumerate(radial_coeff):
            q = q-a*r2**(2+j)
        scale = float(coeff['scale'])
        for i, j, a in coeff['xy']:
            q = q-a*(x/scale)**i*(y/scale)**j
    else:
        radial_coeff = coeff
        for j, a in enumerate(radial_coeff):
            q = q-a*r2**(2+j)
    equation = q if c == 0 else c*r2-2*q+(1+k)*c*q*q
    candidates = []
    for root in equation.roots():
        if abs(root.imag)>1e-6 or root.real<=1e-7:
            continue
        t = float(root.real)
        point = o+t*d
        r = np.linalg.norm(point[:2])
        if 1-(1+k)*c*c*r*r <= 0:
            continue
        sag, _, _ = radial.sag_xy(point[0], point[1], c, k, coeff)
        if abs(point[2]-zv-sag)<1e-6:
            candidates.append(t)
    return sorted(candidates)


def _backend(result, genes):
    representation = result.get('representation')
    if representation == 'spectrometer_style_xy':
        return spectrometer_style
    if representation == 'xy_freeform' or len(genes) > radial.N_TOTAL:
        return freeform
    return radial


def audit(path, max_rays=180):
    path = Path(path)
    result = json.loads(path.read_text())
    g = np.asarray(result['genes'], float).ravel()
    optical = _backend(result, g)
    r = optical.trace(g, fields=5, waves=1, pupils=9)
    m = r['model']
    surfaces = [(0., 0., 0., []), m['plate'], m['primary'], m['secondary'], m['primary']]
    ids = np.flatnonzero(r['good'].ravel())
    if len(ids)>max_rays:
        ids = ids[np.linspace(0, len(ids)-1, max_rays).astype(int)]
    errors, failures = [], []
    for idx in ids:
        for j, surface in enumerate(surfaces):
            origin, hit = r['paths'][j, idx], r['paths'][j+1, idx]
            segment = hit-origin
            length = np.linalg.norm(segment)
            candidates = roots(origin, segment/length, surface)
            if not candidates:
                failures.append(dict(ray=int(idx), surface=j, reason='no forward polynomial root'))
                continue
            error = abs(candidates[0]-length)
            errors.append(error)
            if error>1e-5:
                failures.append(dict(ray=int(idx), surface=j, reason='nearest root differs', error_mm=float(error)))
    summary = dict(method='independent polynomial roots with positive-real branch filtering',
                   backend=result.get('representation', 'xy_freeform' if optical is freeform else 'radial'),
                   sampled_valid_rays=len(ids), tested_intersections=len(errors),
                   max_hit_distance_error_mm=max(errors) if errors else None,
                   failures=failures, passed=bool(len(ids) and not failures),
                   freeform_geometry=m.get('freeform_geometry'),
                   limitation='Checks sampled surviving paths only; not an independent optical engine or mechanical certification')
    write_json(path.with_name('root_audit.json'), summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result'); p.add_argument('--max-rays',type=int,default=180)
    a=p.parse_args(); audit(a.result,a.max_rays)
