"""Display channel of the freeform AR prism in Su & Romanova, Appl. Opt. 65, 5158 (2026), traced in
reverse (Sec. 2): eye pupil -> A (refract in) -> B (mirror) -> A (TIR) -> C (refract out) -> image plane.

Everything is batched over a population: X (P, 16 or 26) -> per-individual metrics, so one GA
generation is one numpy call. Global frame read off Fig. 7: mirror B's vertex is the origin (its
position is not a design variable), the eye is on the +X side with the pupil plane at x = 25 mm and
rays travelling -X, Z points up toward the display, Y is horizontal. A tilt t rotates a surface about
+Y (local x = (cos t, 0, -sin t)); B's sag axis is reversed so that R_B < 0 is concave toward the eye.
Conventions the paper leaves open are listed in README.md.
"""
import itertools

import numpy as np

N_GLASS = 1.53          # not stated in the paper; Zeonex E48R at d-line
PUPIL_X = 25.           # pupil plane (Fig. 7: the eye-box ray bundles converge at x ~ 25 mm)
EYEBOX = 5.             # Sec. 4: 5 x 5 mm eye box
HALF_FIELD = 12.5       # deg per axis: a 25 x 25 deg square field has the stated 35.35 deg diagonal
W_TARGET = 12.          # Eq. (12)
MIN_SEP = 1.            # Stage-2 minimum C -> image separation along the chief ray (value not given)
LEAK_PENALTY = 1.       # mm added to f1, f2 when < 99.9% of rays reach the image (value not given)

# Table 1: name, lower, upper, representative seed. Lengths mm, angles deg.
GEOM = [('RB', -300., -30., -56.), ('xA', 5., 25., 13.), ('apA', 20., 100., 40.), ('tB', -60., 10., -23.),
        ('tA', -10., 10., 0.), ('RA', 50., 300., 132.), ('xC', 0., 10., 5.), ('tC', 60., 110., 78.),
        ('zC', 20., 60., 35.), ('RC', 30., 150., 35.), ('KA', -1., 1., -.5), ('KB', -1., 1., -.5),
        ('xI', -30., 10., -10.), ('yI', -30., 30., 0.), ('zI', 10., 60., 35.), ('tI', -160., -100., -135.)]
POLY = ['y', 'z', 'yy', 'yz', 'zz']                        # Eq. (2): a1 y + a2 z + a3 y^2 + a4 yz + a5 z^2
FREEFORM = [(f'{s}_{p}', -5e-3, 5e-3, 0.) for s in 'AB' for p in POLY]
NAMES = [g[0] for g in GEOM + FREEFORM]
LO = np.array([g[1] for g in GEOM + FREEFORM])
HI = np.array([g[2] for g in GEOM + FREEFORM])
SEED = np.array([g[3] for g in GEOM + FREEFORM])
IDX = {n: i for i, n in enumerate(NAMES)}


def ray_grid():
    """Fig. 3: 3 x 3 eye-box points times 3 x 3 field directions = 81 rays, grouped field-major."""
    steps = (-1., 0., 1.)
    o, d = [], []
    for fy, fz in itertools.product(steps, steps):
        v = np.array([-1., np.tan(np.deg2rad(fy*HALF_FIELD)), np.tan(np.deg2rad(fz*HALF_FIELD))])
        for py, pz in itertools.product(steps, steps):
            o.append((PUPIL_X, py*EYEBOX/2, pz*EYEBOX/2))
            d.append(v/np.linalg.norm(v))
    return np.array(o), np.array(d)


RAY_O, RAY_D = ray_grid()
N_FIELDS, PER_FIELD = 9, 9
CENTER_FIELD, CHIEF = 4, 4*9 + 4


def frame(tilt_deg, sigma=1.):
    """(P,1) tilt -> (P,3,3) whose columns are the local x, y, z axes in global coordinates.

    Rotation about +Y; sigma = -1 reverses the sag axis (and z, keeping the frame right-handed)."""
    t = np.deg2rad(tilt_deg[:, 0])
    c, s, z = np.cos(t), np.sin(t), np.zeros_like(t)
    ex = sigma*np.stack([c, z, -s], -1)
    ey = np.stack([z, z + 1, z], -1)
    ez = sigma*np.stack([s, z, c], -1)
    return np.stack([ex, ey, ez], -1)


def to_local(p, vertex, R):
    return np.einsum('pni,pij->pnj', p - vertex[:, None, :], R)


def dir_to_local(d, R):
    return np.einsum('pni,pij->pnj', d, R)


def dir_to_global(d, R):
    return np.einsum('pij,pnj->pni', R, d)


def sag(c, K, a, y, z, clamp=False):
    """Eqs. (2)-(3). c = 1/R, K conic, a (P,5) polynomial; y, z (P,N). Returns s, ds/dy, ds/dz, valid."""
    r2 = y*y + z*z
    arg = 1. - (1. + K)*c*c*r2
    ok = arg >= 0.
    q = np.sqrt(np.where(ok, arg, 0.) if clamp else np.maximum(arg, 1e-12))
    a1, a2, a3, a4, a5 = (a[:, i:i+1] for i in range(5))
    s = c*r2/(1. + q) + a1*y + a2*z + a3*y*y + a4*y*z + a5*z*z
    sy = c*y/q + a1 + 2*a3*y + a4*z
    sz = c*z/q + a2 + a4*y + 2*a5*z
    return s, sy, sz, ok


def intersect(o, d, c, K, a, iters=12):
    """Eq. (4): Newton on F(t) = x(t) - s(y(t), z(t)) in the local frame, started on the tangent plane."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -o[..., 0]/d[..., 0]
        for _ in range(iters):
            p = o + t[..., None]*d
            s, sy, sz, _ = sag(c, K, a, p[..., 1], p[..., 2])
            F = p[..., 0] - s
            t = t - F/(d[..., 0] - sy*d[..., 1] - sz*d[..., 2])
        p = o + t[..., None]*d
        s, sy, sz, ok = sag(c, K, a, p[..., 1], p[..., 2])
        ok &= np.isfinite(t) & (np.abs(p[..., 0] - s) < 1e-7) & (t > 1e-6)
        n = np.stack([np.ones_like(sy), -sy, -sz], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return t, p, n, ok


def refract(d, n, n1, n2):
    """Vector Snell. Returns new direction and a mask that is False where refraction is impossible (TIR)."""
    n = np.where((np.sum(d*n, -1) > 0)[..., None], -n, n)
    ci = -np.sum(d*n, -1)
    eta = n1/n2
    k = 1. - eta*eta*(1. - ci*ci)
    ok = k >= 0.
    out = eta*d + (eta*ci - np.sqrt(np.maximum(k, 0.)))[..., None]*n
    return out, ok


def reflect(d, n):
    return d - 2.*np.sum(d*n, -1)[..., None]*n


class Surface:
    def __init__(self, vertex, tilt, radius, conic, poly, sigma=1., outside=1.):
        self.v, self.R, self.outside = vertex, frame(tilt, sigma), outside
        self.c = 1./radius
        self.K, self.a = conic, poly

    def hit(self, o, d):
        """Global ray -> global hit point, global unit normal, local hit point, validity."""
        _, pl, nl, ok = intersect(to_local(o, self.v, self.R), dir_to_local(d, self.R), self.c, self.K, self.a)
        return self.v[:, None, :] + dir_to_global(pl, self.R), dir_to_global(nl, self.R), pl, ok

    def beyond(self, p):
        """Signed distance (along local x) of global points past the surface; positive = outside the glass."""
        pl = to_local(p, self.v, self.R)
        with np.errstate(divide='ignore', invalid='ignore'):
            s, _, _, _ = sag(self.c, self.K, self.a, pl[..., 1], pl[..., 2], clamp=True)
        return self.outside*(pl[..., 0] - s)


def build(X):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] == len(GEOM):
        X = np.hstack([X, np.zeros((len(X), len(FREEFORM)))])
    g = {n: X[:, i:i+1] for i, n in enumerate(NAMES)}
    zero = np.zeros_like(g['xA'])
    A = Surface(np.hstack([g['xA'], zero, zero]), g['tA'], g['RA'], g['KA'], X[:, 16:21])
    # B: sag axis points away from the eye, glass on its -x side. C: sag axis points down into the glass.
    B = Surface(np.hstack([zero, zero, zero]), g['tB'], g['RB'], g['KB'], X[:, 21:26], sigma=-1.)
    C = Surface(np.hstack([g['xC'], zero, g['zC']]), g['tC'], g['RC'], zero, np.zeros((len(X), 5)), outside=-1.)
    Fi = frame(g['tI'])
    img = dict(p0=np.hstack([g['xI'], g['yI'], g['zI']]), n=Fi[:, :, 0],
               u=Fi[:, :, 1],                                         # image horizontal axis (Eq. 11) = global Y
               v=Fi[:, :, 2])
    return X, g, A, B, C, img


def trace(X):
    """Trace the 81-ray grid for every individual. Returns per-ray points and per-individual metrics."""
    X, g, A, B, C, img = build(X)
    P = len(X)
    o = np.broadcast_to(RAY_O, (P, 81, 3)).copy()
    d = np.broadcast_to(RAY_D, (P, 81, 3)).copy()
    nan = np.full((P, 81, 3), np.nan)
    with np.errstate(invalid='ignore'):
        pA1, nA1, lA1, ok = A.hit(o, d)
        ok &= np.sum(lA1[..., 1:]**2, -1) <= g['apA']**2
        d, okr = refract(d, nA1, 1., N_GLASS); ok &= okr
        pB, nB, _, okb = B.hit(pA1, d)
        aliveB = ok & okb
        d = reflect(d, nB)
        pA2, nA2, lA2, oka = A.hit(pB, d)
        oka &= np.sum(lA2[..., 1:]**2, -1) <= g['apA']**2
        _, escapes = refract(d, nA2, N_GLASS, 1.)                    # refraction possible => TIR violated
        aliveA2 = aliveB & oka & ~escapes
        d = reflect(d, nA2)
        pC, nC, _, okc = C.hit(pA2, d)
        d, okr = refract(d, nC, N_GLASS, 1.)
        aliveC = aliveA2 & okc & okr
        den = np.einsum('pni,pi->pn', d, img['n'])
        tI = np.einsum('pni,pi->pn', img['p0'][:, None, :] - pC, img['n'])/den
        pI = pC + tI[..., None]*d
        aliveI = aliveC & np.isfinite(tI) & (tI > 0)
    rel = pI - img['p0'][:, None, :]
    u = np.where(aliveI, np.einsum('pni,pi->pn', rel, img['u']), np.nan)
    v = np.where(aliveI, np.einsum('pni,pi->pn', rel, img['v']), np.nan)

    N = 81
    nB_, nA2_, nC_, nI_ = (m.sum(1) for m in (aliveB, aliveA2, aliveC, aliveI))
    uf, vf = u.reshape(P, N_FIELDS, PER_FIELD), v.reshape(P, N_FIELDS, PER_FIELD)
    with np.errstate(invalid='ignore'), _quiet():
        cu, cv = np.nanmean(uf, -1, keepdims=True), np.nanmean(vf, -1, keepdims=True)
        rms = np.sqrt(np.nanmean((uf - cu)**2 + (vf - cv)**2, -1))   # (P, 9); NaN where a field is empty
        rbar = np.nanmean(rms, -1)
        W = np.nanmax(u, -1) - np.nanmin(u, -1)
    # Eq. (7) status score
    S = np.full(P, 10000.)
    for frac, base in ((nB_/N, 5000.), (nA2_/N, 3000.), (nC_/N, 1000.)):
        S = np.where(frac > .1, base + 1000.*(1. - frac), S)
    S = np.where(nI_/N > .9, rbar, S)
    # Eqs. (5)/(6) wedge penalty: TIR hits on A must lie below C; C hits must lie inside B.
    with _quiet():
        sAC = np.nanmax(np.where(aliveA2, C.beyond(pA2), -np.inf), 1)   # signed worst clearance; < 0 is inside
        sBC = np.nanmax(np.where(aliveC, B.beyond(pC), -np.inf), 1)
    dAC, dBC = np.maximum(sAC, 0.), np.maximum(sBC, 0.)
    pw = 1e4*dAC + 1e4*dBC
    sep = np.where(aliveI[:, CHIEF], tI[:, CHIEF], np.nan)
    return dict(X=X, S=S, Pwedge=pw, dAC=dAC, dBC=dBC, sAC=sAC, sBC=sBC, rms=rms, rbar=rbar, W=W, frac=nI_/N, sep=sep,
                counts=np.stack([nB_, nA2_, nC_, nI_], 1), u=u, v=v,
                points=dict(A1=np.where(aliveB[..., None], pA1, nan), B=np.where(aliveB[..., None], pB, nan),
                            A2=np.where(aliveA2[..., None], pA2, nan), C=np.where(aliveC[..., None], pC, nan),
                            I=np.where(aliveI[..., None], pI, nan)))


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings(); self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def stage1_fitness(X):
    """Eq. (8)."""
    r = trace(X)
    return r['S'] + r['Pwedge'], r


def stage2_objectives(X):
    """Eqs. (9)-(12) with the Sec. 3.B penalties. Returns F (P,3), G (P,1) with G <= 0 feasible."""
    r = trace(X)
    with _quiet():
        f1, f2 = np.nanmean(r['rms'], 1), np.nanmax(r['rms'], 1)
    f3 = np.abs(r['W'] - W_TARGET)
    leak = r['frac'] < .999
    f1, f2 = f1 + LEAK_PENALTY*leak, f2 + LEAK_PENALTY*leak
    bad = (r['S'] >= 1000.) | (r['Pwedge'] > 0) | ~np.isfinite(f1) | ~np.isfinite(f3)
    big = 1e4 + r['S'] + r['Pwedge']
    F = np.stack([np.where(bad, big, f1), np.where(bad, big, f2), np.where(bad, big, f3)], 1)
    G = (MIN_SEP - np.nan_to_num(r['sep'], nan=-1e3))[:, None]
    G = np.where(np.isfinite(r['sep'])[:, None], G, 1e3)
    return F, G, r
