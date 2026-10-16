"""Batched nearest-root intersections for the shared 3D polynomial surfaces.

Analytic linear/quadratic paths; batched companion eigenvalues for higher
degrees. Keep physics_audit.exact_intersect as the independent scalar reference.
All roots are considered, preserving finite-aperture nearest-hit semantics.
"""
from math import comb

import numpy as np

from . import model


def _distances(a, v, surface):
    terms = surface.get('terms', model.TERMS)
    active = [(float(c), i, j) for c, (i, j) in zip(surface['coeff'], terms) if c != 0]
    degree = max(1, max((i+j for _, i, j in active), default=1))
    coefficients = np.zeros((len(a), degree+1))
    coefficients[:, 0], coefficients[:, 1] = a[:, 2], v[:, 2]
    # Cache repeated coordinate powers once per chunk, not once per monomial.
    powers = [[np.ones(len(a))] for _ in range(4)]
    for table, values in zip(powers, (a[:, 0], v[:, 0], a[:, 1], v[:, 1])):
        for _ in range(degree): table.append(table[-1]*values)
    for c, i, j in active:
        for k in range(i+1):
            for l in range(j+1):
                coefficients[:, k+l] -= (c*comb(i, k)*comb(j, l)
                    *powers[0][i-k]*powers[1][k]*powers[2][j-l]*powers[3][l])
    finite = np.isfinite(coefficients).all(axis=1)
    # Small physical coefficients must not silently be truncated.
    degrees = np.max(np.where(coefficients != 0, np.arange(degree+1), 0), axis=1)
    distance = np.full(len(a), np.inf)
    for n in range(1, degree+1):
        ids = np.flatnonzero(finite & (degrees == n))
        if not len(ids): continue
        p = coefficients[ids, :n+1]
        if n == 1:
            roots = (-p[:, 0]/p[:, 1])[:, None].astype(complex)
        elif n == 2:
            aa, bb, cc = p[:, 2], p[:, 1], p[:, 0]
            # Scale the quadratic to avoid overflow in its discriminant.
            scale = np.max(abs(p), axis=1)
            aa, bb, cc = aa/scale, bb/scale, cc/scale
            disc = bb*bb-4*aa*cc
            q = -.5*(bb+np.where(bb >= 0, 1., -1.)*np.sqrt(np.maximum(disc, 0)))
            with np.errstate(divide='ignore', invalid='ignore'):
                r1 = q/aa
                r2 = np.where(q != 0, cc/q, -bb/(2*aa))
            roots = np.column_stack((r1, r2)).astype(complex)
            roots[disc < 0] = np.nan
        else:
            companion = np.zeros((len(ids), n, n))
            companion[:, 0, :] = -p[:, -2::-1]/p[:, -1, None]
            companion[:, np.arange(1, n), np.arange(n-1)] = 1.
            try:
                roots = np.linalg.eigvals(companion)
            except np.linalg.LinAlgError:
                # Rare solver failure is isolated; do not discard other rays.
                roots = np.full((len(ids), n), np.nan, complex)
                for k, row in enumerate(p):
                    try: roots[k] = np.roots(row[::-1])
                    except np.linalg.LinAlgError: pass
        t = roots.real
        valid = np.isfinite(t) & (t > 1e-6) & (abs(roots.imag) < 1e-7*(1+abs(t)))
        with np.errstate(invalid='ignore', over='ignore'):
            hit = a[ids, None, :]+t[..., None]*v[ids, None, :]
            valid &= np.sum(hit[..., :2]**2, axis=-1) <= surface['aperture']**2+1e-9
            # Complex/backward/off-aperture roots need no sag evaluation.
            if np.any(valid):
                z, _, _ = model.sag(hit[..., 0][valid], hit[..., 1][valid], surface['coeff'], surface.get('terms'))
                valid[valid] = abs(hit[..., 2][valid]-z) < 1e-6
        distance[ids] = np.min(np.where(valid, t, np.inf), axis=1)
    return distance


def intersect(o, d, surface, *, chunk_size=4096):
    """Drop-in (points, normals, distances, good) intersection API, CPU float64.

    Chunking bounds eigenvalue/coordinate temporaries. Invalid distances are zero,
    matching the scalar audit contract; callers must always use the good mask.
    """
    o, d = np.asarray(o, float), np.asarray(d, float)
    if o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape:
        raise ValueError('Origins and directions must be matching Nx3 arrays')
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    terms = surface.get('terms', model.TERMS)
    if 'terms' in surface and len(surface['coeff']) != len(terms):
        raise ValueError('One coefficient per explicit polynomial term required')
    b = surface['basis']
    a, v = (o-surface['vertex']) @ b, d @ b
    distance = np.full(len(o), np.inf)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for start in range(0, len(o), chunk_size):
            end = start+chunk_size
            distance[start:end] = _distances(a[start:end], v[start:end], surface)
        good = np.isfinite(distance)
        ts = np.where(good, distance, 0.)
        local = a+ts[:, None]*v
        _, zx, zy = model.sag(local[:, 0], local[:, 1], surface['coeff'], surface.get('terms'))
        normals = model.unit(np.column_stack((-zx, -zy, np.ones(len(o)))) @ b.T)
    return o+ts[:, None]*d, normals, ts, good
