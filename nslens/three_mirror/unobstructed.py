"""Unobstructed-by-construction three-mirror generation and refinement (3D).

Planar folded layouts (chief ray in the world XY plane), traced in full 3D with
the shared polynomial-surface engine. Obstruction is part of the merit from the
first evaluation, not a post-hoc screen:

* every beam segment (entrance->M1, M1->M2, M2->M3, M3->detector) is checked
  against every element it must not touch;
* mirrors are zero-thickness disks enlarged by MIRROR_CLEAR keep-out;
* the detector is a rectangle spanning the image footprint plus DETECTOR_CLEAR,
  opaque from both sides;
* clear apertures are sized from the ray footprint plus FOOTPRINT_MARGIN.

A smooth hinge (penetration depth in mm) guides the optimizer; an exact
surface-intersection count is the hard check. Final acceptance uses the
independent nearest-hit tracer `model3d.trace_rays` with a finite opaque
detector. No substrates, mounts or housings are modelled.

Stage 1: CMA-ES over 13 geometry genes per fold-sign topology.
Stage 2: per-ray least squares (TRA residuals) over degree 2-4 freeform terms of
all three mirrors plus detector focus/tilt, with obstruction residuals.
"""
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nslens.spectrometer import model as optics
from nslens.spectrometer.intersections import intersect
from . import model3d as m

FOCAL = 50.
APERTURE = 27.78
HALF_FIELD = (4., 3.)
MIRROR_CLEAR = 2.0       # mm keep-out around every mirror clear aperture
DETECTOR_CLEAR = 1.0     # mm around the image footprint
FOOTPRINT_MARGIN = 1.0   # mm clear aperture beyond the illuminated footprint
MAX_RADIUS = 45.
BIG = 1e4
N_GENES = 13
TOPOLOGIES = {'ppp': (1, 1, 1), 'ppm': (1, 1, -1), 'pmp': (1, -1, 1), 'pmm': (1, -1, -1)}


def _frame(normal):
    normal = normal/np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0], 0.])
    return np.column_stack((tangent, [0., 0., 1.], normal))


def decode(g, signs):
    """13 genes: 3 turn magnitudes, 3 legs, 3x(u2, v2) curvature, detector tilt."""
    g = np.clip(np.asarray(g, float), -1, 1)
    turns = np.asarray(signs)*(110.+50.*g[0:3])          # 60..160 degrees
    legs = 60.+40.*g[3:6]                                 # 20..100 mm
    curves = .02*g[6:12].reshape(3, 2)                    # sag = c*u^2
    p, d, surfaces = np.zeros(3), np.array([1., 0., 0.]), []
    for k in range(3):
        a = np.deg2rad(turns[k])
        out = np.array([np.cos(a)*d[0]-np.sin(a)*d[1], np.sin(a)*d[0]+np.cos(a)*d[1], 0.])
        coeff = np.zeros(len(m.POWERS))
        coeff[m.POWERS.index((2, 0))] = curves[k, 0]*m.SCALE**2
        coeff[m.POWERS.index((0, 2))] = curves[k, 1]*m.SCALE**2
        surfaces.append(dict(vertex=p.copy(), basis=_frame(d-out), coefficients=coeff, radius=MAX_RADIUS))
        p, d = p+legs[k]*out, out
    t = np.deg2rad(30.*g[12])
    dn = np.array([np.cos(t)*d[0]-np.sin(t)*d[1], np.sin(t)*d[0]+np.cos(t)*d[1], 0.])
    return dict(surfaces=surfaces, detector=dict(vertex=p, basis=_frame(dn)),
                task=dict(focal=FOCAL, aperture=APERTURE, half_field_deg=list(HALF_FIELD)),
                genes=g.tolist(), signs=list(signs))


def launch(layout, fields, pupils):
    """Same convention as model3d.launch_rays (pupil intercept at X=0, chief +X),
    but launched upstream of the detector as well as all mirrors."""
    xs = [s['vertex'][0]-s['radius'] for s in layout['surfaces']]+[layout['detector']['vertex'][0]-30.]
    x0 = min(xs)-60.
    f, p = np.asarray(fields, float), np.asarray(pupils, float)
    ang = np.repeat(f, len(p), axis=0); pup = np.tile(p, (len(f), 1)); tan = np.tan(np.deg2rad(ang))
    o = np.column_stack((np.full(len(ang), x0), pup+x0*tan))
    return o, m.unit(np.column_stack((np.ones(len(ang)), tan)))


def _surf(s, aperture):
    out = m.optical_surface(s); out['aperture'] = aperture
    return out


def sequential(layout, o, d):
    """Fixed-order trace M1, M2, M3, detector plane, with unbounded apertures."""
    n = len(o); ok = np.ones(n, bool); pts = [o.copy()]; uv = []; side = []
    for s in layout['surfaces']:
        hit, nrm, t, valid = intersect(o, d, _surf(s, BIG))
        ok &= valid
        side.append(np.sum(d*nrm, axis=1))
        d = optics.reflect(d, nrm); o = hit
        pts.append(hit.copy()); uv.append(((hit-s['vertex']) @ s['basis'])[:, :2])
    det = layout['detector']; nn = det['basis'][:, 2]; den = d @ nn
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((det['vertex']-o) @ nn)/den
    ok &= np.isfinite(t) & (t > 1e-6) & (den > 0)
    hit = o+np.where(ok, t, 0.)[:, None]*d
    pts.append(hit); image = ((hit-det['vertex']) @ det['basis'])[:, :2]
    return dict(ok=ok, points=pts, uv=uv, side=np.array(side), image=image)


def obstruction(layout, tr, radii, exact=True):
    """Hinge penetration depth (mm) per ray and exact blocked mask."""
    pts, ok = tr['points'], tr['ok']; n = len(ok)
    img = tr['image'][ok]
    lo, hi = (img.min(axis=0)-DETECTOR_CLEAR, img.max(axis=0)+DETECTOR_CLEAR) if len(img) else (np.zeros(2), np.zeros(2))
    det = layout['detector']
    elements = [('m', j) for j in range(3)]+[('d', 3)]
    viol = np.zeros(n); blocked = np.zeros(n, bool)
    for k in range(4):                       # segment k: element k-1 -> element k
        P, Q = pts[k], pts[k+1]; seg = Q-P; L = np.linalg.norm(seg, axis=1)
        for kind, j in elements:
            if j in (k-1, k): continue
            V = det['vertex'] if kind == 'd' else layout['surfaces'][j]['vertex']
            B = det['basis'] if kind == 'd' else layout['surfaces'][j]['basis']
            N = B[:, 2]; den = seg @ N
            with np.errstate(divide='ignore', invalid='ignore'):
                s = ((V-P) @ N)/den
            inside = ok & np.isfinite(s) & (s > 0) & (s < 1)
            q = P+np.where(inside, s, 0.)[:, None]*seg
            loc = ((q-V) @ B)[:, :2]
            if kind == 'm':
                depth = radii[j]+MIRROR_CLEAR-np.linalg.norm(loc, axis=1)
                if exact:
                    surf = layout['surfaces'][j]
                    _, _, t, valid = intersect(P, seg/np.maximum(L, 1e-12)[:, None], _surf(surf, radii[j]+MIRROR_CLEAR))
                    blocked |= ok & valid & (t < L-1e-6)
            else:
                depth = np.min(np.column_stack((loc-lo, hi-loc)), axis=1)
                blocked |= inside & (depth > 0)
            viol = np.hypot(viol, np.where(inside, np.maximum(depth, 0), 0.))
    blocked |= False
    return viol, blocked, (lo, hi)


def _parity(cent, target):
    return np.array([(-1, 1)[int(np.argmin([np.sum((cent[:, a]-s*target[:, a])**2) for s in (-1, 1)]))] for a in range(2)])


def measure(layout, fields, pupils, radii=None, exact=True):
    o, d = launch(layout, fields, pupils)
    tr = sequential(layout, o, d)
    ok = tr['ok']; F, P = len(fields), len(pupils)
    uvr = np.array([np.linalg.norm(u, axis=1) for u in tr['uv']])
    if radii is None:
        radii = np.array([np.nanmax(np.where(ok, r, 0.)) for r in uvr])+FOOTPRINT_MARGIN
    viol, blocked, bounds = obstruction(layout, tr, radii, exact)
    xy = np.where(ok[:, None], tr['image'], np.nan).reshape(F, P, 2)
    with np.errstate(invalid='ignore'):
        cent = np.nanmean(xy, axis=1)
        spot = np.sqrt(np.nanmean(np.sum((xy-cent[:, None])**2, axis=-1), axis=1))
    target = FOCAL*np.tan(np.deg2rad(fields))
    par = _parity(np.nan_to_num(cent), target)
    mapping = np.linalg.norm(cent-par*target, axis=1)
    return dict(tr=tr, ok=ok, radii=radii, viol=viol, blocked=blocked, bounds=bounds, xy=xy, cent=cent,
                spot=spot, mapping=mapping, parity=par, target=target, uvr=uvr)


def stage1_merit(g, signs, fields, pupils):
    pen = 10*np.sum(np.maximum(abs(np.asarray(g))-1, 0))
    lay = decode(g, signs)
    try:
        r = measure(lay, fields, pupils)
    except Exception:
        return 1e3
    lost = 1-r['ok'].mean()
    if lost > .5 or not np.isfinite(r['spot']).all(): return 1e3+lost
    side = float(np.mean(r['tr']['side'][:, r['ok']] <= 0)) if r['ok'].any() else 1.
    return float(np.max(r['spot'])+.5*np.sqrt(np.mean(r['spot']**2))
                 + .5*max(np.max(r['mapping'])-.2, 0)
                 + 10*np.mean(r['viol'])+100*r['blocked'].mean()+100*lost+100*side
                 + np.sum(np.maximum(r['radii']-MAX_RADIUS, 0))+pen)


def cma(f, x0, sigma, budget, rng):
    """Minimal (mu/mu_w, lambda) CMA-ES (Hansen tutorial constants)."""
    n = len(x0); lam = 4+int(3*np.log(n)); mu = lam//2
    w = np.log(mu+.5)-np.log(np.arange(1, mu+1)); w /= w.sum(); mueff = 1/np.sum(w**2)
    cc = (4+mueff/n)/(n+4+2*mueff/n); cs = (mueff+2)/(n+mueff+5)
    c1 = 2/((n+1.3)**2+mueff); cmu = min(1-c1, 2*(mueff-2+1/mueff)/((n+2)**2+mueff))
    damps = 1+2*max(0, np.sqrt((mueff-1)/(n+1))-1)+cs; chin = np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    mean = np.array(x0, float); C = np.eye(n); pc = np.zeros(n); ps = np.zeros(n)
    best = (np.inf, mean.copy()); evals = 0; gen = 0
    while evals+lam <= budget:
        vals, Bm = np.linalg.eigh(C); D = np.sqrt(np.maximum(vals, 1e-20))
        z = rng.standard_normal((lam, n)); y = z*D @ Bm.T; x = mean+sigma*y
        fx = np.array([f(xi) for xi in x]); evals += lam; gen += 1
        idx = np.argsort(fx)
        if fx[idx[0]] < best[0]: best = (float(fx[idx[0]]), x[idx[0]].copy())
        yw = w @ y[idx[:mu]]; mean = mean+sigma*yw
        ps = (1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(Bm @ ((Bm.T @ yw)/D))
        hs = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*gen))/chin < 1.4+2/(n+1)
        pc = (1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*yw
        yk = y[idx[:mu]]
        C = (1-c1-cmu)*C+c1*(np.outer(pc, pc)+(1-hs)*cc*(2-cc)*C)+cmu*(yk.T*w) @ yk
        sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chin-1))
        if sigma < 1e-6: break
    return best, evals


# ---------------- stage 2: per-ray least squares with obstruction residuals

FREEFORM = [(j, k) for j in range(3) for k, p in enumerate(m.POWERS) if sum(p) <= 4]


def apply_params(base, x):
    lay = deepcopy(base)
    for (j, k), v in zip(FREEFORM, x[:len(FREEFORM)]):
        lay['surfaces'][j]['coefficients'][k] = v
    focus, tz, tt = x[len(FREEFORM):]
    det = lay['detector']; B = det['basis']
    det['vertex'] = det['vertex']+focus*B[:, 2]
    R = Rotation.from_rotvec(np.deg2rad(tz)*np.array([0, 0, 1.])+np.deg2rad(tt)*B[:, 0]).as_matrix()
    det['basis'] = R @ B
    return lay


def refine(base, radii, parity, fields, pupils, max_nfev=60, map_weight=.3):
    x0 = np.r_[[base['surfaces'][j]['coefficients'][k] for j, k in FREEFORM], 0., 0., 0.]
    span = np.r_[np.full(len(FREEFORM), 3.), 10., 15., 15.]
    target = parity*FOCAL*np.tan(np.deg2rad(fields)); F, P = len(fields), len(pupils); N = F*P
    calls = [0]

    def fun(x):
        calls[0] += 1
        lay = apply_params(base, x)
        o, d = launch(lay, fields, pupils)
        tr = sequential(lay, o, d); ok = tr['ok']
        xy = tr['image'].reshape(F, P, 2); okr = ok.reshape(F, P)
        cent = np.array([xy[f][okr[f]].mean(axis=0) if okr[f].any() else np.zeros(2) for f in range(F)])
        err = np.where(okr[..., None], xy-cent[:, None], 0.)
        viol, blocked, _ = obstruction(lay, tr, radii, exact=False)
        over = np.concatenate([np.maximum(np.linalg.norm(u, axis=1)-(radii[j]-.5), 0) for j, u in enumerate(tr['uv'])])
        side = np.concatenate([np.maximum(-s, 0) for s in tr['side']])
        r = np.r_[err.ravel()/np.sqrt(N), map_weight*(cent-target).ravel()/np.sqrt(F),
                  10*viol/np.sqrt(N), 10*np.nan_to_num(over)/np.sqrt(N), 10*side/np.sqrt(N), 10*(~ok)/np.sqrt(N)]
        return np.nan_to_num(r, nan=10., posinf=10., neginf=-10.)

    r0 = fun(x0)
    opt = least_squares(fun, x0, bounds=(x0-span, x0+span), method='trf', jac='2-point',
                        diff_step=1e-6, x_scale='jac', max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return apply_params(base, opt.x), dict(initial_cost=float(r0 @ r0), final_cost=float(2*opt.cost), nfev=int(opt.nfev),
                                         njev=int(opt.njev) if opt.njev is not None else None, residual_calls=calls[0],
                                         status=int(opt.status), message=str(opt.message), parameters=len(x0),
                                         x=opt.x.tolist())


# ---------------- independent final audit

def audit(layout, radii, fields, pupils):
    """Independent nearest-hit trace with finite opaque detector + exact clearance check."""
    lay = deepcopy(layout)
    for s, R in zip(lay['surfaces'], radii): s['radius'] = float(R)
    o, d = launch(lay, fields, pupils)
    free = m.trace_rays(lay, o, d)                      # infinite detector: footprint
    xy = free['image_mm'][free['good']]
    if not len(xy): return dict(passed=False, reason='no rays')
    lay['detector']['bounds_mm'] = np.array([xy.min(axis=0)-DETECTOR_CLEAR, xy.max(axis=0)+DETECTOR_CLEAR])
    tr = m.trace_rays(lay, o, d)
    order_ok = np.all(tr['route'][:, :3] == np.arange(3), axis=1)
    seq = sequential(lay, o, d)
    viol, blocked, _ = obstruction(lay, seq, radii, exact=True)
    sides = [bool(np.all(s > 0)) for s in seq['side']]
    F, P = len(fields), len(pupils)
    img = tr['image_mm'].reshape(F, P, 2); good = tr['good'].reshape(F, P)
    cent = np.array([img[f][good[f]].mean(axis=0) if good[f].any() else [np.nan]*2 for f in range(F)])
    spot = np.array([np.linalg.norm(np.std(img[f][good[f]], axis=0)) if good[f].sum() > 1 else np.nan for f in range(F)])
    target = FOCAL*np.tan(np.deg2rad(fields)); par = _parity(np.nan_to_num(cent), target)
    mapping = np.linalg.norm(cent-par*target, axis=1)
    passed = bool(tr['good'].all() and order_ok.all() and not blocked.any() and all(sides))
    return dict(passed=passed, rays=int(len(o)), received=int(tr['good'].sum()), sequential_order=bool(order_ok.all()),
                clearance_blocked=int(blocked.sum()), max_keepout_penetration_mm=float(viol.max()),
                front_side_all=sides, worst_rms_spot_um=float(np.nanmax(spot)*1e3),
                rms_spot_um=float(np.sqrt(np.nanmean(spot**2))*1e3), worst_mapping_mm=float(np.nanmax(mapping)),
                detector_bounds_mm=lay['detector']['bounds_mm'].tolist(), radii_mm=[float(r) for r in radii])
