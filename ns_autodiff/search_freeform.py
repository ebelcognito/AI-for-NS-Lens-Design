"""QD SEARCH with freeform (conic+polynomial) surfaces.

The refinement chain freezes a topology and polishes geometry. SEARCH is the opposite: it must
DISCOVER topology (different candidates fold a different number of times), so it needs a
*topology-discovering* freeform tracer -- one that, per ray, repeatedly takes the nearest forward
surface hit and decides refract-vs-TIR by physics until the ray leaves the element, with NO frozen
step count. This module provides that tracer (vectorized numpy, no autograd -- search is
gradient-free) and runs CMA-ME over an extended genotype:

    genotype = [ 9 base DOF (fold_demo) | u_a2_A u_a4_A u_a2_B u_a4_B ]
               where a_power = u * H_CHAR**(-power)  (u ~ edge sag bump in mm; well-scaled for CMA)

Fitness + behaviour axes are IDENTICAL to ns_probe/run_fold_qd.py (focused throughput; BC =
(dominant TIR-bounce count, thinness T)) so the freeform archive is directly comparable to the
circular-arc archive -- the only change is that faces A/B are now freeform.

  # validate the tracer reproduces the JS oracle at coeffs=0, then search:
  PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
  $PY -m ns_autodiff.search_freeform --validate
  $PY -m ns_autodiff.search_freeform --evals 4000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from cma_me import CMAES, GridArchive                                                 # noqa: E402

ELEM_H = 30.0
X_SENSOR = 120.0
FOCUS_W = 8.0
N_RAYS = 21
BEAM_W = 8.0
H_CHAR = 30.0
EPS = 1e-9

# base 9 DOF bounds (fold_demo.LO/HI) + 4 scaled aspheric coeffs
BASE_LO = np.array([-20.0, -55.0, 35.0, 6.0, -18.0, -18.0, -25.0, -25.0, 1.50])
BASE_HI = np.array([20.0, 55.0, 85.0, 45.0, 18.0, 18.0, 25.0, 25.0, 2.00])
U_LO = np.array([-2.0, -2.0, -2.0, -2.0])     # u_a2_A, u_a4_A, u_a2_B, u_a4_B
U_HI = np.array([2.0, 2.0, 2.0, 2.0])
LO = np.concatenate([BASE_LO, U_LO])
HI = np.concatenate([BASE_HI, U_HI])
POWERS = (2, 4)
U_SCALE = np.array([H_CHAR ** (-pw) for pw in POWERS])     # a = u * scale
PARAM_NAMES = ["ray_y", "ang", "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b", "n",
               "u_a2_A", "u_a4_A", "u_a2_B", "u_a4_B"]

BC_BINS = (9, 16)
BC_BOUNDS = ((0.0, 9.0), (10.0, 120.0))


# --------------------------------------------------------------------------- #
#  numpy freeform geometry (vectorized over rays)
# --------------------------------------------------------------------------- #
def _circumcenter(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
    ux = (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
    uy = (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
    c = np.array([ux, uy]); return c, float(np.hypot(*(c - p1)))


def _sag(h, c, k, coeffs):
    A = (1.0 + k) * c * c
    u = np.clip(1.0 - A * h * h, 0.0, None); w = np.sqrt(u + 1e-12)
    denom = 1.0 + w; z = c * h * h / denom
    wp = (-A * h) / np.maximum(w, 1e-12)
    dz = (2.0 * c * h * denom - c * h * h * wp) / (denom * denom)
    for pw, a in zip(POWERS, coeffs):
        z = z + a * h ** pw; dz = dz + a * pw * h ** (pw - 1)
    return z, dz


def _freeform_surface(apex, center, r, end1, end2, coeffs):
    axis = (apex - center); axis = axis / max(np.hypot(*axis), EPS)
    tang = np.array([-axis[1], axis[0]])
    c = -1.0 / max(r, EPS)
    h1 = (end1 - apex) @ tang; h2 = (end2 - apex) @ tang
    return dict(kind="ff", v=apex, axis=axis, tang=tang, c=c, k=0.0,
                coeffs=np.asarray(coeffs), h_min=min(h1, h2), h_max=max(h1, h2))


def _roots_ff(O, D, s, *, n_scan=64, n_bisect=40, t_min=1e-6):
    """Nearest forward valid intersection of rays O[m,2],D[m,2] with a freeform surface.
    Returns t[m], hit[m,2], normal[m,2], valid[m] (vectorized; mirrors geometry_freeform)."""
    m = O.shape[0]
    v, axis, tang = s["v"], s["axis"], s["tang"]
    h0 = (O - v) @ tang; hd = D @ tang
    a0 = (O - v) @ axis; ad = D @ axis
    hd_safe = np.where(np.abs(hd) < EPS, EPS, hd)
    hmin, hmax = s["h_min"], s["h_max"]
    h_b = h0 + t_min * hd
    fwd_high = hd > 0
    lo_f = np.where(fwd_high, np.maximum(hmin, h_b), hmin)
    hi_f = np.where(fwd_high, hmax, np.minimum(hmax, h_b))
    nonempty = hi_f > lo_f + 1e-12
    g = np.linspace(0.0, 1.0, n_scan)[:, None]
    Hg = lo_f[None, :] + g * (hi_f - lo_f)[None, :]        # [n_scan, m]
    z, _ = _sag(Hg, s["c"], s["k"], s["coeffs"])
    Tg = (Hg - h0[None, :]) / hd_safe[None, :]
    Fg = a0[None, :] + Tg * ad[None, :] - z
    sgn = np.sign(Fg)
    cross = (sgn[:-1] * sgn[1:] < 0)                       # [n_scan-1, m]
    has = cross.any(0) & nonempty
    # nearest forward bracket = smallest t among crossings
    t_lo, t_hi = Tg[:-1], Tg[1:]
    cand = np.where(cross, np.minimum(t_lo, t_hi), np.inf)
    sel = np.argmin(cand, axis=0)
    ii = np.arange(m)
    a_t = np.minimum(t_lo, t_hi)[sel, ii]
    b_t = np.maximum(t_lo, t_hi)[sel, ii]

    def F_at(t):
        h = h0 + t * hd
        z2, _ = _sag(h, s["c"], s["k"], s["coeffs"])
        return a0 + t * ad - z2
    Fa = F_at(a_t)
    swap = Fa > 0
    lo = np.where(swap, b_t, a_t); hi = np.where(swap, a_t, b_t)
    t = 0.5 * (lo + hi)
    for _ in range(n_bisect):
        Ft = F_at(t)
        neg = Ft < 0
        lo = np.where(neg, t, lo); hi = np.where(neg, hi, t)
        t = 0.5 * (lo + hi)
    h = h0 + t * hd
    z2, dz = _sag(h, s["c"], s["k"], s["coeffs"])
    hit = O + t[:, None] * D
    n_geom = axis[None, :] - dz[:, None] * tang[None, :]
    n_geom = n_geom / np.maximum(np.linalg.norm(n_geom, axis=1, keepdims=True), EPS)
    sign = -np.sign((n_geom * D).sum(1)); sign = np.where(sign == 0, 1.0, sign)
    normal = n_geom * sign[:, None]
    in_aper = (h >= min(hmin, hmax) - 1e-6) & (h <= max(hmin, hmax) + 1e-6)
    valid = has & (t > t_min) & in_aper & (np.abs(hd) >= EPS)
    return t, hit, normal, valid


def _roots_line(O, D, a, b, *, t_min=1e-6):
    e = b - a
    denom = D[:, 0] * (-e[1]) - D[:, 1] * (-e[0])
    denom_s = np.where(np.abs(denom) < EPS, EPS, denom)
    diff = a[None, :] - O
    t = (diff[:, 0] * (-e[1]) - diff[:, 1] * (-e[0])) / denom_s
    hit = O + t[:, None] * D
    sp = ((hit - a[None, :]) * e[None, :]).sum(1) / max(e @ e, EPS)
    n_geom = np.array([-e[1], e[0]]); n_geom = n_geom / max(np.hypot(*n_geom), EPS)
    n_geom = np.broadcast_to(n_geom, O.shape).copy()
    sign = -np.sign((n_geom * D).sum(1)); sign = np.where(sign == 0, 1.0, sign)
    normal = n_geom * sign[:, None]
    valid = (t > t_min) & (np.abs(denom) >= EPS) & (sp >= -1e-6) & (sp <= 1.0 + 1e-6)
    return t, hit, normal, valid


# --------------------------------------------------------------------------- #
#  element + discovering trace
# --------------------------------------------------------------------------- #
def build_element(x):
    """4 surfaces from the extended genotype: freeform A,B + line T,D (mirrors fold_demo.element
    with the curved faces upgraded). Returns dict + cx (for the positional face classifier)."""
    p = x[:9]
    ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n = p
    u = x[9:]
    cA = u[0:2] * U_SCALE; cB = u[2:4] * U_SCALE
    H = ELEM_H
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * H; dB = math.tan(math.radians(tilt_b)) * H
    v0 = np.array([lx + dA, -H]); v1 = np.array([rx + dB, -H]); v2 = np.array([rx + bsag, 0.0])
    v3 = np.array([rx - dB, H]); v4 = np.array([lx - dA, H]); v5 = np.array([lx - fsag, 0.0])
    cBc, rB = _circumcenter(v1, v2, v3)
    cAc, rA = _circumcenter(v4, v5, v0)
    surfaces = dict(
        A=_freeform_surface(v5, cAc, rA, v4, v0, cA),
        B=_freeform_surface(v2, cBc, rB, v1, v3, cB),
        T=dict(kind="line", a=v0, b=v1), D=dict(kind="line", a=v3, b=v4))
    return surfaces, float(cx), float(n)


def geometry_valid(x):
    _, _, cx, thick, _, _, tilt_a, tilt_b, _ = x[:9]
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * ELEM_H; dB = math.tan(math.radians(tilt_b)) * ELEM_H
    return ((rx + dB) - (lx + dA) > 1.0) and ((rx - dB) - (lx - dA) > 1.0)


def face_label(x, y, cx, eps=1e-3):
    if y <= -ELEM_H + eps:
        return "T"
    if y >= ELEM_H - eps:
        return "D"
    return "A" if x < cx else "B"


def trace_discover(x, *, n_rays=N_RAYS, beam_w=BEAM_W, max_bounces=14):
    """Topology-DISCOVERING freeform NS trace. Per ray: repeatedly take the nearest forward
    surface hit, refract/TIR by physics, until it leaves the element; then propagate to the
    sensor. Returns per-ray dicts {seq:[(type,face,x,y,b)], y_sensor, b_sensor, forward}."""
    surfaces, cx, n = build_element(x)
    faces = ["A", "B", "T", "D"]
    ray_y, ang = x[0], x[1]
    a = math.radians(ang)
    dys = np.linspace(-beam_w / 2, beam_w / 2, n_rays)
    m = n_rays
    O = np.stack([np.zeros(m), ray_y + dys], 1)
    D = np.broadcast_to(np.array([math.cos(a), math.sin(a)]), (m, 2)).copy()
    b_s = np.full(m, 0.5); b_p = np.full(m, 0.5)
    inside = np.zeros(m, dtype=bool)
    active = np.ones(m, dtype=bool)
    seqs = [[] for _ in range(m)]

    for _b in range(max_bounces):
        if not active.any():
            break
        ts = np.full((4, m), np.inf); hits = np.zeros((4, m, 2))
        nrms = np.zeros((4, m, 2)); vals = np.zeros((4, m), dtype=bool)
        for fi, f in enumerate(faces):
            s = surfaces[f]
            if s["kind"] == "ff":
                t, hit, nrm, val = _roots_ff(O, D, s)
            else:
                t, hit, nrm, val = _roots_line(O, D, s["a"], s["b"])
            ts[fi] = np.where(val, t, np.inf); hits[fi] = hit; nrms[fi] = nrm; vals[fi] = val
        ts[:, ~active] = np.inf
        sel = np.argmin(ts, axis=0)
        ii = np.arange(m)
        any_hit = np.isfinite(ts[sel, ii]) & active
        hit = hits[sel, ii]; normal = nrms[sel, ii]
        # refract vs TIR per ray
        n1 = np.where(inside, n, 1.0 / n)
        cos1 = -(D * normal).sum(1)
        sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        tir = (sq1 < 0)
        do = any_hit
        # record event (face from positional classifier at the hit)
        for i in np.where(do)[0]:
            fl = face_label(hit[i, 0], hit[i, 1], cx)
            seqs[i].append(("tir" if tir[i] else "refract", fl, float(hit[i, 0]),
                            float(hit[i, 1]), float(b_s[i] + b_p[i])))
        # TIR rays: reflect; refract rays: snell + fresnel + toggle inside
        cos2 = np.sqrt(np.clip(sq1, 0.0, None) + 1e-12)
        refl = D + 2.0 * cos1[:, None] * normal
        refr = n1[:, None] * D + (n1 * cos1 - cos2)[:, None] * normal
        is_tir = do & tir
        is_ref = do & (~tir)
        # fresnel for refract rays
        Rs = ((n1 * cos1 - cos2) / (n1 * cos1 + cos2)) ** 2
        Rp = ((n1 * cos2 - cos1) / (n1 * cos2 + cos1)) ** 2
        b_s = np.where(is_ref, b_s * (1.0 - Rs), b_s)
        b_p = np.where(is_ref, b_p * (1.0 - Rp), b_p)
        newD = np.where(is_tir[:, None], refl, np.where(is_ref[:, None], refr, D))
        newD = newD / np.maximum(np.linalg.norm(newD, axis=1, keepdims=True), EPS)
        D = np.where(do[:, None], newD, D)
        O = np.where(do[:, None], hit, O)
        inside = np.where(is_ref, ~inside, inside)
        active = active & do          # rays with no hit this round have left the element

    # propagate every ray to the sensor plane
    out = []
    for i in range(m):
        dx = D[i, 0]
        fwd = dx > 1e-9
        if fwd:
            t_s = (X_SENSOR - O[i, 0]) / dx
            ys = O[i, 1] + t_s * D[i, 1] if t_s > 0 else None
        else:
            ys = None
        out.append(dict(seq=seqs[i], y=ys, b=float(b_s[i] + b_p[i]),
                        forward=bool(fwd and ys is not None)))
    return out


# --------------------------------------------------------------------------- #
#  fitness + behaviour (identical recipe to run_fold_qd.evaluate)
# --------------------------------------------------------------------------- #
def evaluate(x):
    if not geometry_valid(x):
        return None
    try:
        rays = trace_discover(x)
    except Exception:
        return None
    fwd = [r for r in rays if r["forward"] and r["b"] > 0]
    if not fwd:
        return None
    emitted = float(len(rays))                      # each input ray carries 1.0
    by_sig = {}
    for r in fwd:
        sig = tuple((t, f) for (t, f, *_x) in r["seq"])
        by_sig[sig] = by_sig.get(sig, 0.0) + r["b"]
    if not by_sig:
        return None
    dom_sig = max(by_sig, key=by_sig.get); dom_energy = by_sig[dom_sig]
    tir_count = sum(1 for (t, _f) in dom_sig if t == "tir")
    ys = np.array([r["y"] for r in fwd]); bs = np.array([r["b"] for r in fwd])
    centroid = float(np.average(ys, weights=bs))
    window = float(bs[np.abs(ys - centroid) <= FOCUS_W].sum())
    fitness = window / emitted
    spot = float(np.sqrt(np.average((ys - centroid) ** 2, weights=bs)))
    optic_xs = [vx for r in fwd for (_t, _f, vx, _vy, _b) in r["seq"] if 0.5 < vx < X_SENSOR - 0.5]
    T = X_SENSOR - (min(optic_xs) if optic_xs else 0.0)
    return dict(fitness=float(fitness), bc=(float(tir_count), float(T)),
                info=dict(spot=spot, tir_count=int(tir_count), thinness_T=float(T),
                          dom_energy_share=float(dom_energy / emitted),
                          dom_signature=" -> ".join(f"{'TIR' if t=='tir' else 'T'}@{f}"
                                                    for t, f in dom_sig),
                          efficiency=float(bs.sum() / emitted), centroid=centroid,
                          n_forward=len(fwd),
                          aspheric=[float(v) for v in x[9:]]))


# --------------------------------------------------------------------------- #
#  validation: tracer vs JS oracle at coeffs=0
# --------------------------------------------------------------------------- #
def validate(cells=("3,5", "4,11", "0,1")):
    from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
    import fold_demo as FD
    from sequence_tools import (aggregate_by_signature, make_single_element_face_classifier,
                                signature_of)
    archive = json.loads((REPO / "ns_probe" / "results_fold_qd" / "archive.json").read_text())
    print("=" * 74)
    print("TRACER VALIDATION  (discovering freeform tracer, coeffs=0, vs JS oracle)")
    print("=" * 74)
    allok = True
    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        for cs in cells:
            i0, i1 = (int(v) for v in cs.split(","))
            cell = next((c for c in archive["cells"] if c["i0"] == i0 and c["i1"] == i1), None)
            if cell is None:
                print(f"  {cs}: no cell"); continue
            base = np.array(cell["genotype"], float)
            x = np.concatenate([base, np.zeros(4)])     # coeffs = 0 -> arcs
            ev = evaluate(x)
            res, _ = runner.simulate(FD.scene(FD.vec_to_params(base), n_rays=N_RAYS,
                                              beam_w=BEAM_W, record=True))
            face_of = make_single_element_face_classifier(float(base[2]), ELEM_H)
            by = aggregate_by_signature(res.get("raySequences", []), face_of)
            recs = [r for r in res.get("raySequences", [])
                    if r.get("dirx", 0) > 0 and r.get("b", 0) > 0
                    and signature_of(r, face_of) is not None]
            ys = np.array([r["y"] for r in recs]); bs = np.array([r["b"] for r in recs])
            cen = np.average(ys, weights=bs)
            o_spot = float(np.sqrt(np.average((ys - cen) ** 2, weights=bs)))
            o_dom = max(by, key=by.get)
            o_tir = sum(1 for (t, _f) in o_dom if t == "tir")
            ok = (ev is not None and ev["info"]["tir_count"] == o_tir
                  and abs(ev["info"]["spot"] - o_spot) < 0.05)
            allok &= ok
            print(f"  cell {cs:5s}: tracer tir={ev['info']['tir_count']} spot={ev['info']['spot']:.3f} "
                  f"| oracle tir={o_tir} spot={o_spot:.3f}  {'OK' if ok else 'MISMATCH'}")
            print(f"            tracer dom: {ev['info']['dom_signature']}")
    print("=" * 74); print(f"OVERALL: {'PASS' if allok else 'FAIL'}")
    return allok


# --------------------------------------------------------------------------- #
#  search
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--evals", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_search_freeform"))
    ap.add_argument("--lock-aspheric", action="store_true",
                    help="ablation: freeze the 4 aspheric DOF at 0 (pure circular arcs) -- the "
                         "apples-to-apples arc baseline at the same budget/tracer")
    args = ap.parse_args()

    if args.validate:
        validate(); return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = LO.copy(), HI.copy()
    if args.lock_aspheric:
        lo[9:] = 0.0; hi[9:] = 0.0
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    scale = hi - lo
    t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(lo, hi)
        return CMAES(np.asarray(x0, float), args.sigma0 * float(np.mean(scale[scale > 0]) if (scale > 0).any() else 1.0),
                     popsize=args.popsize, seed=s, bounds=(lo, hi))

    emitters = [new_emitter(args.seed + i) for i in range(args.emitters)]
    n = 0; n_invalid = 0; history = []
    while n < args.evals:
        for ei in range(args.emitters):
            if n >= args.evals:
                break
            em = emitters[ei]; X = em.ask(); keys = np.empty(len(X)); added = False
            for k, xx in enumerate(X):
                ev = evaluate(xx); n += 1
                if ev is None:
                    keys[k] = -1e9; n_invalid += 1; continue
                st, old = archive.add(xx, ev["fitness"], ev["bc"], info=ev["info"])
                if st == "new":
                    keys[k] = 2e12 + ev["fitness"]; added = True
                elif st == "improved":
                    keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                else:
                    keys[k] = ev["fitness"]
            em.tell(np.argsort(-keys))
            if (not added) or em.restart_needed:
                emitters[ei] = new_emitter(int(rng.integers(0, 1 << 30)))
        if n // 1000 > len(history):
            rec = dict(evals=n, filled=len(archive.cells), coverage=archive.coverage,
                       best=archive.best_fitness, invalid=n_invalid)
            history.append(rec)
            print(f"[ff-search] evals={n:6d} filled={rec['filled']:3d} "
                  f"cov={rec['coverage']*100:5.1f}% best_focthru={rec['best']:.3f} "
                  f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), tir_count=float(c.bc[0]), T=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info); cells.append(rec)
    payload = dict(meta=dict(experiment=("fold_qd_ARC_baseline" if args.lock_aspheric
                                         else "fold_qd_FREEFORM"),
                             lock_aspheric=bool(args.lock_aspheric),
                             evals=int(n), seed=int(args.seed),
                             invalid=int(n_invalid), bc_bins=BC_BINS, bc_bounds=BC_BOUNDS,
                             param_names=PARAM_NAMES, powers=list(POWERS),
                             fitness="focused_throughput", wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    base = out / ("archive_arc.json" if args.lock_aspheric else "archive_freeform.json")
    rep, k = base, 0
    while rep.exists():
        k += 1; rep = base.with_name(f"archive_freeform_{k}.json")
    rep.write_text(json.dumps(payload, indent=2))
    print(f"[ff-search] wrote {rep} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
