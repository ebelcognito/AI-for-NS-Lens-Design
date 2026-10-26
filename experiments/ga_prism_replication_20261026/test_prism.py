"""Physics checks for the batched tracer (the paper validates against CODE V; we validate against closed forms)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prism as M


def one(vertex, tilt, R, K=0., poly=None):
    return M.Surface(np.array([vertex], float), np.array([[tilt]], float), np.array([[R]], float),
                     np.array([[K]], float), np.zeros((1, 5)) if poly is None else np.array([poly], float))


def rays(o, d):
    d = np.asarray(d, float)
    return np.asarray(o, float)[None], (d/np.linalg.norm(d, axis=-1, keepdims=True))[None]


def test_mirror_paraxial_focus():
    # Concave mirror R = -100 facing -X: paraxial rays focus at R/2 in front of the vertex.
    m = one((50., 0, 0), 0., -100.)
    h = np.array([.1, .2, -.3])
    o, d = rays(np.stack([0*h, h, 0*h], -1), np.tile([1., 0, 0], (3, 1)))
    p, n, _, ok = m.hit(o, d)
    assert ok.all()
    d2 = M.reflect(d, n)
    t = -p[0, :, 1]/d2[0, :, 1]
    assert np.allclose(p[0, :, 0] + t*d2[0, :, 0], 0., atol=h.max()**2/100 + 1e-9)  # spherical aberration ~ h^2/(4|R|)


def test_newton_matches_bisection_on_tilted_freeform():
    rng = np.random.default_rng(0)
    s = one((20., 0, 0), 17., 80., -.4, [2e-3, -3e-3, 4e-3, -2e-3, 3e-3])
    o = np.c_[np.zeros(20), rng.uniform(-5, 5, 20), rng.uniform(-5, 5, 20)]
    d = np.c_[np.ones(20), rng.uniform(-.2, .2, 20), rng.uniform(-.2, .2, 20)]
    o, d = rays(o, d)
    p, _, _, ok = s.hit(o, d)
    assert ok.all()
    for k in range(20):
        f = lambda t: s.beyond((o[0, k] + t*d[0, k])[None, None])[0, 0]
        lo, hi = 0., 60.
        assert f(lo) < 0 < f(hi)
        for _ in range(200):
            mid = (lo + hi)/2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        assert np.allclose(o[0, k] + lo*d[0, k], p[0, k], atol=1e-6)


def test_snell_invariants():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(1, 50, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = rng.normal(size=(1, 50, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    out, ok = M.refract(d, n, 1., 1.53)
    assert ok.all() and np.allclose(np.linalg.norm(out, axis=-1), 1.)
    sin_i = np.linalg.norm(np.cross(d, n), axis=-1)
    sin_t = np.linalg.norm(np.cross(out, n), axis=-1)
    assert np.allclose(sin_i, 1.53*sin_t)
    assert np.allclose(np.einsum('pni,pni->pn', np.cross(d, n), out), 0., atol=1e-12)   # coplanar


def test_tir_threshold():
    crit = np.arcsin(1/1.53)
    n = np.array([[[1., 0, 0]]])
    for a, escapes in ((crit - 1e-6, True), (crit + 1e-6, False)):
        d = np.array([[[np.cos(a), 0, np.sin(a)]]])
        assert M.refract(d, n, 1.53, 1.)[1][0, 0] == escapes


def test_plate_lateral_shift():
    # Plane-parallel plate between two flat A-like faces: exit direction equals entry direction.
    a, b = one((10., 0, 0), 0., 1e12), one((15., 0, 0), 0., 1e12)
    o, d = rays([[0, 0, 0]], [[1., 0, .3]])
    p1, n1, _, _ = a.hit(o, d); d1, _ = M.refract(d, n1, 1., 1.53)
    p2, n2, _, _ = b.hit(p1, d1); d2, _ = M.refract(d1, n2, 1.53, 1.)
    assert np.allclose(d2, d)
    th = np.arctan(.3); tt = np.arcsin(np.sin(th)/1.53)
    assert np.isclose(p2[0, 0, 2], 10*np.tan(th) + 5*np.tan(tt))


def test_ray_grid_matches_paper():
    assert M.RAY_O.shape == (81, 3)
    diag = np.degrees(np.arccos(M.RAY_D[0] @ M.RAY_D[80]))
    assert np.isclose(diag, 35.0, atol=.5)  # corner-to-corner angle of the 25 x 25 deg field


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
