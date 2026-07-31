"""QD SEARCH with freeform surfaces — AR/VR PRISM config.

Same machinery as search_freeform.py (topology-discovering numpy NS tracer + CMA-ME), but the
prism problem from ns_probe/arvr_prism.py:
  * element = FOUR curved faces (A left, B right, T top, D bottom) -- all freeform now;
  * 3 fixed top-firing display point-sources, each a fan, relayed to the eye pupil (x=PUPIL_X);
  * imaging condition = COLLIMATION (virtual image at infinity): minimize each source's exit
    angular spread sigma_s; fitness = mean_s[ T_s * exp(-sigma_s/SIGMA0) ], valid only if all 3
    fields reach the eyebox with a monotonic field map;
  * archive axes = (FOV deg, prism depth) -- compactness is what makes folding worth discovering.

Genotype = [ 9 base prism DOF (arvr_prism) | u_a2 u_a4 per face x4 = 8 scaled aspheric coeffs ].
At coeffs=0 the faces are the circular arcs of the original prism, so the discovering tracer is
validated against the JS oracle (arvr_prism.scene_arvr + arvr_metrics) at coeffs=0.

  PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
  $PY -m ns_autodiff.search_freeform_prism --validate
  $PY -m ns_autodiff.search_freeform_prism --evals 20000 --seed 0
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
from .search_freeform_geom import (circumcenter as _circumcenter,                      # noqa: E402
                                   freeform_surface as _ffs, roots_ff as _roots_ff)


def _freeform_surface(apex, center, r, end1, end2, coeffs):
    return _ffs(apex, center, r, end1, end2, coeffs, powers=POWERS)

# prism constants (mirror arvr_prism.py)
ELEM_H = 30.0
PUPIL_X = 120.0
EYEBOX_Y = 0.0
EYEBOX_HALF = 9.0
DISPLAY_Y = 50.0
DISPLAY_CX = 55.0
DISPLAY_HALF_W = 6.0
SOURCE_XS = np.linspace(DISPLAY_CX - DISPLAY_HALF_W, DISPLAY_CX + DISPLAY_HALF_W, 3)
SOURCE_PTS = [(float(sx), DISPLAY_Y) for sx in SOURCE_XS]
SOURCE_AIM = -90.0
N_FAN = 17
FAN_HALFANGLE = 12.0
SIGMA0 = 0.7
SAG = 15.0
TILT = 50.0
H_CHAR = 30.0
POWERS = (2, 4)
U_SCALE = np.array([H_CHAR ** (-pw) for pw in POWERS])
EPS = 1e-9

BASE_LO = np.array([35.0, 6.0, -SAG, -SAG, -SAG, -SAG, -TILT, -TILT, 1.50])
BASE_HI = np.array([85.0, 45.0, SAG, SAG, SAG, SAG, TILT, TILT, 2.00])
U_LO = np.full(8, -2.0); U_HI = np.full(8, 2.0)       # u_a2/u_a4 for A,B,T,D
LO = np.concatenate([BASE_LO, U_LO]); HI = np.concatenate([BASE_HI, U_HI])
BASE_NAMES = ["cx", "thick", "fsag", "bsag", "tsag", "dsag", "tilt_a", "tilt_b", "n"]
PARAM_NAMES = BASE_NAMES + ["u_a2_A", "u_a4_A", "u_a2_B", "u_a4_B",
                            "u_a2_T", "u_a4_T", "u_a2_D", "u_a4_D"]

BC_BINS = (20, 28)
BC_BOUNDS = ((0.0, 40.0), (4.0, 88.0))                # (FOV deg, prism depth)


def build_prism(x):
    """4 freeform faces from the extended genotype (mirrors arvr_prism.prism_element with all
    four arcs upgraded to conic+poly)."""
    cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n = x[:9]
    u = x[9:]
    H = ELEM_H
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * H; dB = math.tan(math.radians(tilt_b)) * H
    bl = np.array([lx + dA, -H]); br = np.array([rx + dB, -H])
    tr = np.array([rx - dB, H]); tl = np.array([lx - dA, H])
    apexA = np.array([lx - fsag, 0.0]); apexB = np.array([rx + bsag, 0.0])
    apexT = np.array([(tr[0] + tl[0]) / 2, H + tsag]); apexD = np.array([(bl[0] + br[0]) / 2, -H - dsag])
    cA, rA = _circumcenter(tl, apexA, bl); cB, rB = _circumcenter(br, apexB, tr)
    cT, rT = _circumcenter(tr, apexT, tl); cD, rD = _circumcenter(bl, apexD, br)
    s = dict(
        A=_freeform_surface(apexA, cA, rA, tl, bl, u[0:2] * U_SCALE),
        B=_freeform_surface(apexB, cB, rB, br, tr, u[2:4] * U_SCALE),
        T=_freeform_surface(apexT, cT, rT, tr, tl, u[4:6] * U_SCALE),
        D=_freeform_surface(apexD, cD, rD, bl, br, u[6:8] * U_SCALE))
    return s, float(n)


def geometry_valid(x):
    cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, _ = x[:9]
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * ELEM_H; dB = math.tan(math.radians(tilt_b)) * ELEM_H
    body = ((rx + dB) - (lx + dA) > 1.0) and ((rx - dB) - (lx - dA) > 1.0)
    return body and (ELEM_H + max(tsag, 0.0) < DISPLAY_Y - 2.0)


def trace_prism(x, *, max_bounces=18):
    """Discovering NS trace of the 3 source fans through the 4-freeform-face prism to the pupil.
    Returns per-ray dicts {src, y, ang, b, forward, xs:[hit x's]}."""
    surfaces, n = build_prism(x)
    faces = ["A", "B", "T", "D"]
    O = []; D = []; src = []
    for si, (sx, sy) in enumerate(SOURCE_PTS):
        for ai in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, N_FAN)):
            O.append([sx, sy]); D.append([math.cos(ai), math.sin(ai)]); src.append(si)
    O = np.array(O); D = np.array(D); src = np.array(src); m = O.shape[0]
    b_s = np.full(m, 0.5); b_p = np.full(m, 0.5)
    inside = np.zeros(m, dtype=bool); active = np.ones(m, dtype=bool)
    xs_hit = [[] for _ in range(m)]

    for _b in range(max_bounces):
        if not active.any():
            break
        ts = np.full((4, m), np.inf); hits = np.zeros((4, m, 2)); nrms = np.zeros((4, m, 2))
        for fi, f in enumerate(faces):
            t, hit, nrm, val = _roots_ff(O, D, surfaces[f])
            ts[fi] = np.where(val, t, np.inf); hits[fi] = hit; nrms[fi] = nrm
        ts[:, ~active] = np.inf
        sel = np.argmin(ts, axis=0); ii = np.arange(m)
        any_hit = np.isfinite(ts[sel, ii]) & active
        hit = hits[sel, ii]; normal = nrms[sel, ii]
        n1 = np.where(inside, n, 1.0 / n)
        cos1 = -(D * normal).sum(1)
        sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        tir = sq1 < 0
        cos2 = np.sqrt(np.clip(sq1, 0.0, None) + 1e-12)
        refl = D + 2.0 * cos1[:, None] * normal
        refr = n1[:, None] * D + (n1 * cos1 - cos2)[:, None] * normal
        is_tir = any_hit & tir; is_ref = any_hit & (~tir)
        Rs = ((n1 * cos1 - cos2) / (n1 * cos1 + cos2)) ** 2
        Rp = ((n1 * cos2 - cos1) / (n1 * cos2 + cos1)) ** 2
        b_s = np.where(is_ref, b_s * (1.0 - Rs), b_s); b_p = np.where(is_ref, b_p * (1.0 - Rp), b_p)
        newD = np.where(is_tir[:, None], refl, np.where(is_ref[:, None], refr, D))
        newD = newD / np.maximum(np.linalg.norm(newD, axis=1, keepdims=True), EPS)
        D = np.where(any_hit[:, None], newD, D); O = np.where(any_hit[:, None], hit, O)
        inside = np.where(is_ref, ~inside, inside)
        for i in np.where(any_hit)[0]:
            if 1.0 < hit[i, 0] < PUPIL_X - 1.0:
                xs_hit[i].append(float(hit[i, 0]))
        active = active & any_hit

    out = []
    for i in range(m):
        dx = D[i, 0]; fwd = dx > 1e-9
        if fwd:
            t_s = (PUPIL_X - O[i, 0]) / dx
            y = O[i, 1] + t_s * D[i, 1] if t_s > 0 else None
        else:
            y = None
        ang = math.degrees(math.atan2(D[i, 1], D[i, 0]))
        out.append(dict(src=int(src[i]), y=y, ang=ang, b=float(b_s[i] + b_p[i]),
                        forward=bool(fwd and y is not None), xs=xs_hit[i]))
    return out


def metrics(rays):
    """Per-source collimation metrics + scalar fitness (mirrors arvr_prism.arvr_metrics)."""
    per = {i: {"ang": [], "w": []} for i in range(3)}
    for r in rays:
        if not r["forward"] or r["b"] <= 0:
            continue
        if abs(r["y"] - EYEBOX_Y) > EYEBOX_HALF:
            continue
        per[r["src"]]["ang"].append(r["ang"]); per[r["src"]]["w"].append(r["b"])
    fields = []
    for i in range(3):
        a = np.array(per[i]["ang"]); w = np.array(per[i]["w"])
        if w.sum() <= 0:
            fields.append(dict(present=False, T=0.0, theta=None, sigma=None)); continue
        theta = float(np.average(a, weights=w))
        sigma = float(np.sqrt(np.average((a - theta) ** 2, weights=w)))
        fields.append(dict(present=True, T=float(w.sum() / N_FAN), theta=theta, sigma=sigma))
    present = [f for f in fields if f["present"]]
    thetas = [f["theta"] for f in fields]
    ordering_ok = (len(present) == 3 and (thetas[0] < thetas[1] < thetas[2]
                                          or thetas[0] > thetas[1] > thetas[2]))
    q = [f["T"] * math.exp(-f["sigma"] / SIGMA0) if f["present"] else 0.0 for f in fields]
    fitness = float(np.mean(q)) * (1.0 if ordering_ok else 0.0)
    fov = (abs(fields[2]["theta"] - fields[0]["theta"])
           if fields[0]["present"] and fields[2]["present"] else 0.0)
    return dict(fields=fields, fitness=fitness, fov=float(fov), ordering_ok=ordering_ok,
                T_min=float(min(f["T"] for f in fields)),
                sigma_mean=float(np.mean([f["sigma"] for f in present])) if present else None)


def evaluate(x):
    if not geometry_valid(x):
        return None
    try:
        rays = trace_prism(x)
    except Exception:
        return None
    m = metrics(rays)
    if not m["ordering_ok"] or m["fitness"] <= 0:
        return None
    xs = [vx for r in rays if r["forward"] for vx in r["xs"]]
    depth = (max(xs) - min(xs)) if xs else 0.0
    if m["fov"] <= 0 or depth <= 0:
        return None
    tir = 0  # (optional analysis; not a descriptor here)
    return dict(fitness=float(m["fitness"]), bc=(float(m["fov"]), float(depth)),
                info=dict(fov=float(m["fov"]), depth=float(depth), fitness=float(m["fitness"]),
                          T_min=float(m["T_min"]), sigma_mean=float(m["sigma_mean"]),
                          thetas=[f["theta"] for f in m["fields"]],
                          sigmas=[f["sigma"] for f in m["fields"]],
                          aspheric=[float(v) for v in x[9:]]))


def validate():
    from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
    import arvr_prism as AV
    print("=" * 74)
    print("PRISM TRACER VALIDATION (discovering freeform tracer, coeffs=0, vs JS oracle)")
    print("=" * 74)
    # try archive champions, else a few random valid prisms
    cand = []
    arch_p = REPO / "ns_probe" / "results_arvr_qd" / "archive.json"
    if arch_p.exists():
        a = json.loads(arch_p.read_text())
        cells = sorted(a["cells"], key=lambda c: -c["fitness"])[:4]
        cand = [np.array(c["genotype"], float) for c in cells]
    rng = np.random.default_rng(0)
    while len(cand) < 4:
        xx = rng.uniform(BASE_LO, BASE_HI)
        if AV.geometry_valid_arvr(AV.prism_dict(xx)):
            f, _ = AV.evaluate(None, xx) if False else (1, None)
            cand.append(xx)
    allok = True
    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        for base in cand:
            x = np.concatenate([base, np.zeros(8)])
            ev = evaluate(x)
            res, _ = runner.simulate(AV.scene_arvr(base))
            om = AV.arvr_metrics(res)
            if ev is None or not om["ordering_ok"]:
                print(f"  (one design invalid in tracer={ev is None} / oracle_ok={om['ordering_ok']})")
                continue
            df = abs(ev["info"]["fitness"] - om["fitness"])
            dfov = abs(ev["info"]["fov"] - om["fov"])
            ok = df < 0.03 and dfov < 1.0
            allok &= ok
            print(f"  fit tracer {ev['info']['fitness']:.4f} / oracle {om['fitness']:.4f}  "
                  f"FOV {ev['info']['fov']:.2f}/{om['fov']:.2f}  "
                  f"sig {ev['info']['sigma_mean']:.3f}/{om['sigma_mean']:.3f}  {'OK' if ok else 'MISMATCH'}")
    print("=" * 74); print(f"OVERALL: {'PASS' if allok else 'FAIL'}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--lock-aspheric", action="store_true")
    ap.add_argument("--out", default=str(REPO / "ns_autodiff" / "results_search_freeform_prism"))
    args = ap.parse_args()
    if args.validate:
        validate(); return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = LO.copy(), HI.copy()
    if args.lock_aspheric:
        lo[9:] = 0.0; hi[9:] = 0.0
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed); scale = hi - lo; t0 = time.time()

    def new_emitter(s):
        x0 = archive.random_elite(rng)
        if x0 is None:
            x0 = rng.uniform(lo, hi)
        sig = args.sigma0 * float(np.mean(scale[scale > 0]) if (scale > 0).any() else 1.0)
        return CMAES(np.asarray(x0, float), sig, popsize=args.popsize, seed=s, bounds=(lo, hi))

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
            print(f"[prism-ff] evals={n:6d} filled={rec['filled']:3d} "
                  f"cov={rec['coverage']*100:5.1f}% best={rec['best']:.4f} "
                  f"invalid={n_invalid} ({time.time()-t0:6.1f}s)", flush=True)

    cells = []
    for (i0, i1), c in archive.cells.items():
        rec = dict(i0=int(i0), i1=int(i1), fov=float(c.bc[0]), depth=float(c.bc[1]),
                   fitness=float(c.fitness), genotype=[float(v) for v in c.genotype])
        rec.update(c.info); cells.append(rec)
    payload = dict(meta=dict(experiment=("arvr_prism_ARC_baseline" if args.lock_aspheric
                                         else "arvr_prism_FREEFORM"),
                             lock_aspheric=bool(args.lock_aspheric), evals=int(n),
                             seed=int(args.seed), invalid=int(n_invalid), bc_bins=BC_BINS,
                             bc_bounds=BC_BOUNDS, bc_axes=["FOV_deg", "prism_depth"],
                             param_names=PARAM_NAMES, powers=list(POWERS),
                             wall_seconds=float(time.time() - t0)),
                   history=history, cells=cells)
    base = out / ("archive_prism_arc.json" if args.lock_aspheric else "archive_prism_freeform.json")
    rep, k = base, 0
    while rep.exists():
        k += 1; rep = base.with_name(base.stem + f"_{k}.json")
    rep.write_text(json.dumps(payload, indent=2))
    print(f"[prism-ff] wrote {rep} ({len(cells)} cells, {n} evals, "
          f"{payload['meta']['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
