"""Variable-count (3–6) finite-parabola mirror tracing pilot.

Genes are grouped as n normalized turn angles, n leg lengths, then n curvatures.
Topology proposals preserve coarse layout only; they do not preserve exact paths.
"""
import numpy as np

FOCAL = 50.0
APERTURE = 27.78
BASELINE_ANGLES_DEG = np.array([-150., 180., -180., 180., -180., 180.])
ANGLE_RANGES_DEG = np.array([28., 58., 58., 58., 58., 58.])


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def baseline_genes(n):
    """Return a normalized baseline chromosome for n mirrors (3 <= n <= 6)."""
    _check_n(n)
    return np.concatenate((BASELINE_ANGLES_DEG[:n] / 180., np.zeros(2*n)))


def _check_n(n):
    if not isinstance(n, (int, np.integer)) or not 3 <= n <= 6:
        raise ValueError("n must be an integer from 3 through 6")


def decode(g, n):
    """Build mirror frames from [n turns, n lengths, n curvatures]."""
    _check_n(n)
    g = np.asarray(g, dtype=float)
    if g.shape != (3*n,):
        raise ValueError(f"expected {3*n} genes for n={n}; got shape {g.shape}")
    angles = 180. * np.clip(g[:n], -1., 1.)
    lengths = 65. + 40. * np.clip(g[n:2*n], -1., 1.)
    curves = .025 * np.clip(g[2*n:], -1., 1.)
    widths = np.full(n, 30.)
    vertices, normals, tangents = [], [], []
    p = np.zeros(2)
    d = np.array([1., 0.])
    for angle, length in zip(angles, lengths):
        a = np.deg2rad(angle)
        out = np.array([np.cos(a)*d[0]-np.sin(a)*d[1],
                        np.sin(a)*d[0]+np.cos(a)*d[1]])
        # Keep exact 0-degree turns numerically defined while retaining the
        # requested nearly straight representation.
        delta = d - out
        if np.linalg.norm(delta) < 1.e-12:
            delta = np.array([-d[1], d[0]]) * 1.e-12
        normal = unit(delta)
        tangent = np.array([-normal[1], normal[0]])
        vertices.append(p.copy()); normals.append(normal); tangents.append(tangent)
        p = p + length*out
        d = out
    return dict(vertices=np.array(vertices), normals=np.array(normals),
                tangents=np.array(tangents), curves=curves, widths=widths,
                image=p, image_normal=d, image_tangent=np.array([-d[1], d[0]]),
                angles=angles, lengths=lengths)


def intersections(o, d, m):
    """Exact forward intersections with finite parabolas; stable quadratic roots."""
    delta = o[:, None, :] - m['vertices'][None, :, :]
    u = np.einsum('rmi,mi->rm', delta, m['tangents'])
    v = np.einsum('rmi,mi->rm', delta, m['normals'])
    du = d @ m['tangents'].T
    dv = d @ m['normals'].T
    a = m['curves'][None, :] * du**2
    b = 2*m['curves'][None, :]*u*du - dv
    c = m['curves'][None, :]*u**2 - v
    disc = b*b - 4*a*c
    q = -.5*(b + np.where(b >= 0, 1., -1.)*np.sqrt(np.maximum(0., disc)))
    with np.errstate(divide='ignore', invalid='ignore'):
        r1, r2, linear = q/a, c/q, -c/b
    small = abs(a) < 1.e-14
    r1 = np.where(small, linear, r1)
    r2 = np.where(small, np.inf, r2)
    roots = np.stack((r1, r2), axis=-1)
    with np.errstate(invalid='ignore'):
        local = u[:, :, None] + roots*du[:, :, None]
    valid = (np.isfinite(roots) & (roots > 1.e-7)
             & (abs(local) <= m['widths'][None, :, None]))
    valid &= ((disc >= 0) | small)[:, :, None]
    return np.min(np.where(valid, roots, np.inf), axis=-1)


def trace(g, n, fields=3, pupils=9, paths=False, *, focal=50.0,
          aperture=27.78, half_field_deg=4.0):
    """Trace all rays to nearest surfaces. Paths mode returns original 5-tuple."""
    for name, value in (("focal", focal), ("aperture", aperture),
                        ("half_field_deg", half_field_deg)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("fields", fields), ("pupils", pupils)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be an integer >= 1")
    m = decode(g, n)
    field = np.repeat(np.linspace(-half_field_deg, half_field_deg, fields), pupils)
    theta = np.deg2rad(field)
    pupil = np.tile(np.linspace(-aperture/2, aperture/2, pupils), fields)
    samples = np.linspace(-1., 1., 65)
    outlines = [v + samples[:, None]*w*t + (a*(samples*w)**2)[:, None]*normal
                for v, t, normal, a, w in zip(m['vertices'], m['tangents'],
                                               m['normals'], m['curves'], m['widths'])]
    x0 = min(np.min(p[:, 0]) for p in outlines) - 20.
    o = np.column_stack((np.full(len(field), x0), pupil + x0*np.tan(theta)))
    d = np.column_stack((np.cos(theta), np.sin(theta)))
    active = np.ones(len(field), bool)
    hits = np.zeros(len(field), int)
    route = np.full((len(field), n+3), -1, int)
    image = np.full(len(field), np.nan)
    histories = [[p.copy()] for p in o] if paths else None
    for step in range(n+3):
        distances = intersections(o, d, m)
        which = np.argmin(distances, axis=1)
        tm = distances[np.arange(len(o)), which]
        den = d @ m['image_normal']
        with np.errstate(divide='ignore', invalid='ignore'):
            ti = ((m['image']-o) @ m['image_normal']) / den
        sensor = active & (ti > 1.e-7) & (ti < tm) & (den > 0)
        if np.any(sensor):
            p = o[sensor] + ti[sensor, None]*d[sensor]
            image[sensor] = (p-m['image']) @ m['image_tangent']
            if paths:
                for idx, point in zip(np.flatnonzero(sensor), p):
                    histories[idx].append(point.copy())
        reflect = active & ~sensor & np.isfinite(tm)
        active = reflect
        if not np.any(reflect):
            break
        ids = np.flatnonzero(reflect)
        j = which[ids]
        p = o[ids] + tm[ids, None]*d[ids]
        u = np.sum((p-m['vertices'][j])*m['tangents'][j], axis=1)
        normal = unit(m['normals'][j] - 2*(m['curves'][j]*u)[:, None]*m['tangents'][j])
        d[ids] -= 2*np.sum(d[ids]*normal, axis=1)[:, None]*normal
        o[ids] = p
        route[ids, step] = j
        hits[ids] += 1
        if paths:
            for idx, point in zip(ids, p):
                histories[idx].append(point.copy())
    good = np.isfinite(image) & (hits == n) & np.all(
        np.sort(route[:, :n], axis=1) == np.arange(n), axis=1)
    target = focal*np.tan(theta)
    errors = [np.where(good, np.minimum(abs(image-sign*target), 100.), 100.)
              for sign in (-1, 1)]
    parity = (-1, 1)[int(np.argmin([np.mean(e*e) for e in errors]))]
    err = errors[0 if parity == -1 else 1]
    support = good.reshape(fields, pupils).mean(axis=1)
    rmse = np.sqrt(np.mean(err.reshape(fields, pupils)**2, axis=1))
    progress = np.sum(np.cumprod(route[:, :n] == np.arange(n), axis=1), axis=1)
    loss = float(np.mean(err**2) + 10000*np.mean(~good)
                 + 1000*np.mean((n-progress)/n))
    rows = []
    field_mapping_rms = []
    for f in range(fields):
        mask = (np.arange(len(field))//pupils == f) & good
        values = image[mask]
        target_value = target[f*pupils]
        rows.append(dict(field_deg=float(field[f*pupils]), support=float(support[f]),
                         rms_spot_mm=float(np.std(values)) if len(values) else None,
                         centroid_error_mm=float(abs(np.mean(values)-parity*target_value))
                         if len(values) else None))
        if len(values):
            field_mapping_rms.append(float(np.sqrt(np.mean(
                (values-parity*target_value)**2))))
        else:
            field_mapping_rms.append(None)
    worst_rms_spot = (max(row["rms_spot_mm"] for row in rows)
                      if all(sum(good[f*pupils:(f+1)*pupils]) >= 2
                             for f in range(fields)) else None)
    worst_centroid_error = (max(row["centroid_error_mm"] for row in rows)
                            if all(sum(good[f*pupils:(f+1)*pupils]) >= 1
                                   for f in range(fields)) else None)
    worst_mapping_rms = (max(field_mapping_rms)
                         if all(value is not None for value in field_mapping_rms)
                         else None)
    result = dict(loss=loss, worst_support=float(min(support)),
                  target_rmse_with_loss_penalty_mm=float(max(rmse)), parity=parity,
                  fields=rows, genes=np.asarray(g).tolist(), n=n,
                  task=dict(focal=float(focal), aperture=float(aperture),
                            half_field_deg=float(half_field_deg), fields=int(fields),
                            pupils=int(pupils)),
                  worst_rms_spot_mm=worst_rms_spot,
                  worst_centroid_error_mm=worst_centroid_error,
                  worst_mapping_rms_mm=worst_mapping_rms)
    if paths:
        return result, m, outlines, histories, good
    return result


def propose(parent, n, new_n, rng):
    """Map a chromosome to an adjacent mirror count with a coarse transfer.

    Insertion splits one leg and adds a small turn; removal merges neighboring
    legs and adds the removed turn to the next turn (wrapped to [-180, 180]).
    These moves preserve approximate layout, not exact ray paths.
    """
    _check_n(n); _check_n(new_n)
    if abs(new_n-n) != 1:
        raise ValueError("new_n must differ from n by exactly one")
    parent = np.asarray(parent, dtype=float)
    if parent.shape != (3*n,):
        raise ValueError(f"expected {3*n} genes for n={n}; got shape {parent.shape}")
    angles = 180*np.clip(parent[:n], -1., 1.)
    lengths = 65+40*np.clip(parent[n:2*n], -1., 1.)
    curves = .025*np.clip(parent[2*n:], -1., 1.)
    if new_n == n+1:
        # Insert only on a leg between existing mirrors, not the terminal leg.
        i = int(rng.integers(n-1))
        leg = lengths[i]
        first = min(max(leg/2, 25.), 80.)
        second = min(max(leg-first, 25.), 80.)
        lengths[i] = first
        angles = np.insert(angles, i+1, float(rng.uniform(-10., 10.)))
        lengths = np.insert(lengths, i+1, second)
        curves = np.insert(curves, i+1, 0.)
    else:
        # Remove an internal mirror. Its adjacent legs merge and its turn is
        # transferred to the following mirror as a coarse layout approximation.
        j = int(rng.integers(1, n-1))
        lengths[j-1] = min(lengths[j-1] + lengths[j], 105.)
        angles[j+1] = ((angles[j+1] + angles[j] + 180.) % 360.) - 180.
        angles = np.delete(angles, j)
        lengths = np.delete(lengths, j)
        curves = np.delete(curves, j)
    genes = np.concatenate((angles/180., (lengths-65.)/40., curves/.025))
    return np.clip(genes, -1., 1.)


def smoke_check():
    """Small deterministic shape/finite and adjacent-topology sanity check."""
    rng = np.random.default_rng(1729)
    for n in range(3, 7):
        g = baseline_genes(n)
        m = decode(g, n)
        assert m['vertices'].shape == (n, 2) and m['widths'].shape == (n,)
        assert np.all(np.isfinite(m['vertices']))
        result = trace(g, n, fields=2, pupils=3)
        assert result['n'] == n and np.isfinite(result['loss'])
        for adjacent in (n-1, n+1):
            if 3 <= adjacent <= 6:
                mapped = propose(g, n, adjacent, rng)
                assert mapped.shape == (3*adjacent,) and np.all(np.isfinite(mapped))
                checked = trace(mapped, adjacent, fields=2, pupils=3)
                assert np.isfinite(checked['loss'])
    return True


if __name__ == '__main__':
    smoke_check()
    print('variable_mirror smoke check passed')
