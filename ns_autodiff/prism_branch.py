"""Full branching (Fresnel ray-tree) non-sequential tracer for the AR/VR prism — Step 1.

Fixes the two faithfulness failures of the single-path discovering tracer:
  (1) BRANCHING: at every refraction it spawns BOTH the transmitted ray AND the Fresnel
      partial-reflection ray (the "ghost"), following each until it exits / drops below an
      energy floor / hits max depth -- exactly what the JS non-sequential engine does. The
      prism's collimation metric is sensitive to ghost energy reaching the eyebox, so this is
      required for the metric to match the oracle on the messy designs a search explores.
  (2) >90-degree ARCS: faces are intersected as FULL CIRCLES with a chord-side on-arc test
      (valid for any arc up to 180deg), instead of a single-valued sag cap (valid only <90deg).
      The measured prism folds need 112-137deg arcs.

This step uses CIRCULAR faces (no aspheric yet) so it can be validated against the oracle in
isolation; the parametric aspheric is added in step 2 once branching is proven faithful.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

# prism constants (mirror arvr_prism.py / search_freeform_prism.py)
ELEM_H = 30.0
PUPIL_X = 120.0
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
MAX_DEPTH = 16            # matches scene maxRayDepth
E_MIN = 1e-3             # energy floor for pruning ghost branches
EPS = 1e-9
BASE_NAMES = ["cx", "thick", "fsag", "bsag", "tsag", "dsag", "tilt_a", "tilt_b", "n"]


def _circumcenter(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
    ux = (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
    uy = (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
    return np.array([ux, uy]), float(np.hypot(ux - ax, uy - ay))


def build_arcs(x):
    """4 circular faces (center, r, two corners, apex) from the 9-DOF prism genotype."""
    cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n = x[:9]
    H = ELEM_H
    lx, rx = cx - thick / 2, cx + thick / 2
    dA = math.tan(math.radians(tilt_a)) * H; dB = math.tan(math.radians(tilt_b)) * H
    bl = np.array([lx + dA, -H]); br = np.array([rx + dB, -H])
    tr = np.array([rx - dB, H]); tl = np.array([lx - dA, H])
    apexA = np.array([lx - fsag, 0.0]); apexB = np.array([rx + bsag, 0.0])
    apexT = np.array([(tr[0] + tl[0]) / 2, H + tsag]); apexD = np.array([(bl[0] + br[0]) / 2, -H - dsag])
    arcs = {}
    for name, (c1, ap, c2) in dict(A=(tl, apexA, bl), B=(br, apexB, tr),
                                   T=(tr, apexT, tl), D=(bl, apexD, br)).items():
        center, r = _circumcenter(c1, ap, c2)
        arcs[name] = dict(center=center, r=r, c1=np.asarray(c1, float),
                          c2=np.asarray(c2, float), apex=np.asarray(ap, float))
    return arcs, float(n)


def _chord_side(P, c1, c2, ref):
    e = c2 - c1
    return np.sign(e[0] * (P[1] - c1[1]) - e[1] * (P[0] - c1[0])) == \
           np.sign(e[0] * (ref[1] - c1[1]) - e[1] * (ref[0] - c1[0]))


def nearest_hit(O, D, arcs, t_min=1e-6):
    """Nearest forward intersection with the 4 circular faces, kept only if on the real arc
    (chord-side test, valid for any arc up to 180deg). Returns (t, P, normal, name) or None."""
    best = None
    for name, a in arcs.items():
        oc = O - a["center"]; b = D @ oc; c = oc @ oc - a["r"] * a["r"]
        disc = b * b - c
        if disc < 0:
            continue
        sq = math.sqrt(disc)
        for t in (-b - sq, -b + sq):
            if t <= t_min:
                continue
            P = O + t * D
            if not _chord_side(P, a["c1"], a["c2"], a["apex"]):
                continue
            if best is None or t < best[0]:
                ng = (P - a["center"]) / max(a["r"], EPS)
                nrm = ng * (-np.sign(ng @ D) if (ng @ D) != 0 else 1.0)
                best = (t, P, nrm, name)
    return best


def trace_branching(x, *, max_depth=MAX_DEPTH, e_min=E_MIN):
    """Fresnel ray-tree. Returns exit rays {src, y, ang, b, xs} that reach the pupil forward."""
    arcs, n = build_arcs(x)
    exits = []
    stack = []
    for si, (sx, sy) in enumerate(SOURCE_PTS):
        for ang in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, N_FAN)):
            stack.append((np.array([sx, sy]), np.array([math.cos(ang), math.sin(ang)]),
                          0.5, 0.5, False, 0, si, ()))
    while stack:
        O, D, bs, bp, inside, depth, src, xs = stack.pop()
        if bs + bp < e_min:
            continue
        hit = nearest_hit(O, D, arcs) if depth < max_depth else None
        if hit is None:
            if D[0] > 1e-9:                              # exits body toward the eye
                t_s = (PUPIL_X - O[0]) / D[0]
                if t_s > 0:
                    y = O[1] + t_s * D[1]
                    exits.append(dict(src=src, y=float(y),
                                      ang=math.degrees(math.atan2(D[1], D[0])),
                                      b=float(bs + bp), xs=xs))
            continue
        t, P, normal, name = hit
        new_xs = xs + (float(P[0]),) if 1.0 < P[0] < PUPIL_X - 1.0 else xs
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal)
        sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:                                      # TIR (single child)
            Dr = D + 2.0 * cos1 * normal
            stack.append((P, Dr / np.linalg.norm(Dr), bs, bp, inside, depth + 1, src, new_xs))
        else:
            cos2 = math.sqrt(sq1)
            Rs = ((n1 * cos1 - cos2) / (n1 * cos1 + cos2)) ** 2
            Rp = ((n1 * cos2 - cos1) / (n1 * cos2 + cos1)) ** 2
            Dt = n1 * D + (n1 * cos1 - cos2) * normal     # transmitted
            stack.append((P, Dt / np.linalg.norm(Dt), bs * (1 - Rs), bp * (1 - Rp),
                          not inside, depth + 1, src, new_xs))
            Dr = D + 2.0 * cos1 * normal                  # Fresnel ghost
            stack.append((P, Dr / np.linalg.norm(Dr), bs * Rs, bp * Rp, inside,
                          depth + 1, src, new_xs))
    return exits


def metrics(exits):
    per = {i: {"ang": [], "w": []} for i in range(3)}
    for r in exits:
        if r["b"] <= 0 or abs(r["y"]) > EYEBOX_HALF:
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


# --------------------------------------------------------------------------- #
#  Step 2: parametric aspheric (radial bump on normalized arc position s in [-1,1])
#  Works at ANY arc span (no sag-cap >90deg limit). coeffs=0 => identical to the circle tracer.
# --------------------------------------------------------------------------- #
POWERS = (2, 4)


def _unwrap(th, ref):
    return ref + ((th - ref + math.pi) % (2 * math.pi) - math.pi)


def build_arcs_ff(x):
    """4 faces with a parametric aspheric: radial bump Delta(s)=sum a_i s^i, s=normalized arc
    position (apex s=0, corners s=+-1). x = 9 base + 8 coeffs (a2,a4 per face A,B,T,D)."""
    arcs, n = build_arcs(x)
    u = x[9:] if len(x) > 9 else np.zeros(8)
    for fi, name in enumerate(("A", "B", "T", "D")):
        a = arcs[name]; ce = a["center"]
        th_ap = math.atan2(a["apex"][1] - ce[1], a["apex"][0] - ce[0])
        th1 = _unwrap(math.atan2(a["c1"][1] - ce[1], a["c1"][0] - ce[0]), th_ap)
        th2 = _unwrap(math.atan2(a["c2"][1] - ce[1], a["c2"][0] - ce[0]), th_ap)
        a["th_apex"] = th_ap
        a["half_span"] = max(abs(th1 - th_ap), abs(th2 - th_ap), 1e-6)
        a["coeffs"] = u[2 * fi:2 * fi + 2]
    return arcs, n


def _delta(s, coeffs):
    d = dd = 0.0
    for pw, c in zip(POWERS, coeffs):
        d += c * s ** pw; dd += c * pw * s ** (pw - 1)
    return d, dd


def hit_ff(O, D, a, t_min=1e-6):
    """Nearest forward valid hit on one aspheric face, or None. Newton-corrects the circle hit
    for the radial bump; validity = normalized arc position |s|<=1."""
    ce, r = a["center"], a["r"]
    oc = O - ce; b = D @ oc; c = oc @ oc - r * r
    disc = b * b - c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    for tc in sorted((-b - sq, -b + sq)):
        if tc <= t_min:
            continue
        t = tc
        for _ in range(6):                              # Newton on G(t)=(|P-ce|-r)-Delta(s)
            P = O + t * D; rel = P - ce; rad = math.hypot(rel[0], rel[1])
            s = (_unwrap(math.atan2(rel[1], rel[0]), a["th_apex"]) - a["th_apex"]) / a["half_span"]
            G = (rad - r) - _delta(s, a["coeffs"])[0]
            P2 = O + (t + 1e-6) * D; rel2 = P2 - ce; rad2 = math.hypot(rel2[0], rel2[1])
            s2 = (_unwrap(math.atan2(rel2[1], rel2[0]), a["th_apex"]) - a["th_apex"]) / a["half_span"]
            G2 = (rad2 - r) - _delta(s2, a["coeffs"])[0]
            dG = (G2 - G) / 1e-6
            if abs(dG) < 1e-12:
                break
            t = t - G / dG
        P = O + t * D; rel = P - ce; rad = math.hypot(rel[0], rel[1])
        if rad < 1e-9 or t <= t_min:
            continue
        s = (_unwrap(math.atan2(rel[1], rel[0]), a["th_apex"]) - a["th_apex"]) / a["half_span"]
        if abs(s) > 1.0 + 1e-6:
            continue
        radial = rel / rad; tang = np.array([-radial[1], radial[0]])
        dd = _delta(s, a["coeffs"])[1]
        ng = radial - (dd / (a["half_span"] * rad)) * tang
        ng = ng / np.linalg.norm(ng)
        nrm = ng * (-np.sign(ng @ D) if (ng @ D) != 0 else 1.0)
        return t, P, nrm
    return None


def nearest_hit_ff(O, D, arcs):
    best = None
    for name, a in arcs.items():
        h = hit_ff(O, D, a)
        if h is not None and (best is None or h[0] < best[0]):
            best = (h[0], h[1], h[2], name)
    return best


def _aspheric_fields(a, coeffs):
    """Attach (th_apex, half_span, coeffs) to a raw arc dict so nearest_hit_ff can use it."""
    ce = a["center"]
    th_ap = math.atan2(a["apex"][1] - ce[1], a["apex"][0] - ce[0])
    th1 = _unwrap(math.atan2(a["c1"][1] - ce[1], a["c1"][0] - ce[0]), th_ap)
    th2 = _unwrap(math.atan2(a["c2"][1] - ce[1], a["c2"][0] - ce[0]), th_ap)
    a["th_apex"] = th_ap
    a["half_span"] = max(abs(th1 - th_ap), abs(th2 - th_ap), 1e-6)
    a["coeffs"] = np.asarray(coeffs, float)
    return a


def build_triangle(x):
    """3-edge triangular prism (circular faces): top entry edge + two slanted edges that fold the
    display light toward the eye. Genotype x[:8] = [cx, w, apex_x, apex_y, s_top, s_left, s_right, n]
    where (cx,w) is the top edge at y=ELEM_H, (apex_x,apex_y) the third corner, s_* the edge sags."""
    cx, w, apex_x, apex_y, s_top, s_left, s_right, n = x[:8]
    H = ELEM_H
    c1 = np.array([cx - w / 2, H]); c2 = np.array([cx + w / 2, H]); c3 = np.array([apex_x, apex_y])
    cen = (c1 + c2 + c3) / 3.0
    faces = {}
    for name, (a, b, sg) in [("TOP", (c1, c2, s_top)), ("RIGHT", (c2, c3, s_right)),
                             ("LEFT", (c3, c1, s_left))]:
        if abs(sg) < 0.3:
            sg = math.copysign(0.3, sg if sg != 0 else 1.0)     # never exactly flat (collinear -> degenerate circle)
        mid = 0.5 * (a + b); out = mid - cen
        out = out / max(np.hypot(out[0], out[1]), EPS)          # outward (away from centroid)
        apex = mid + sg * out
        center, r = _circumcenter(a, apex, b)
        faces[name] = _aspheric_fields(dict(center=center, r=r, c1=a, c2=b, apex=apex), np.zeros(2))
    return faces, float(n)


def geometry_valid_tri(x):
    cx, w, apex_x, apex_y, _s1, _s2, _s3, _n = x[:8]
    c1 = np.array([cx - w / 2, ELEM_H]); c2 = np.array([cx + w / 2, ELEM_H])
    c3 = np.array([apex_x, apex_y])
    area = 0.5 * abs((c2[0] - c1[0]) * (c3[1] - c1[1]) - (c3[0] - c1[0]) * (c2[1] - c1[1]))
    return area > 60.0 and apex_y < ELEM_H - 5.0          # non-degenerate, apex below the entry edge


def chief_ray_folds(faces, n, max_depth=MAX_DEPTH):
    """TIR-bounce count of the central source's chief ray (the fold-count descriptor)."""
    sx, sy = SOURCE_PTS[1]
    O = np.array([sx, sy]); D = np.array([math.cos(math.radians(SOURCE_AIM)),
                                          math.sin(math.radians(SOURCE_AIM))])
    inside = False; ntir = 0
    for _ in range(max_depth):
        hit = nearest_hit_ff(O, D, faces)
        if hit is None:
            break
        _t, P, normal, _name = hit
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:
            D = D + 2.0 * cos1 * normal; D = D / np.linalg.norm(D); ntir += 1
        else:
            cos2 = math.sqrt(sq1); D = n1 * D + (n1 * cos1 - cos2) * normal
            D = D / np.linalg.norm(D); inside = not inside
        O = P
    return ntir


def trace_branching_ff(x, *, max_depth=MAX_DEPTH, e_min=E_MIN):
    """Branching Fresnel ray-tree, 4 aspheric faces (the freeform prism tracer)."""
    return trace_faces(*build_arcs_ff(x), max_depth=max_depth, e_min=e_min)


def trace_faces(faces, n, *, max_depth=MAX_DEPTH, e_min=E_MIN, max_seg=6000):
    """Branching Fresnel ray-tree over an arbitrary face set (4-edge prism, 3-edge triangle, ...).
    `max_seg` hard-caps the total ray-segments processed: a trapping cavity (rays refracting in/out
    repeatedly with the bright branch never pruning) can otherwise explode the tree ~2^depth. Hitting
    the cap returns (None) -> the design is rejected as pathological rather than hanging the search."""
    exits = []; stack = []; nseg = 0
    for si, (sx, sy) in enumerate(SOURCE_PTS):
        for ang in np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, N_FAN)):
            stack.append((np.array([sx, sy]), np.array([math.cos(ang), math.sin(ang)]),
                          0.5, 0.5, False, 0, si, ()))
    while stack:
        nseg += 1
        if nseg > max_seg:
            return None                                  # pathological / trapping -> reject
        O, D, bs, bp, inside, depth, src, xs = stack.pop()
        if bs + bp < e_min:
            continue
        hit = nearest_hit_ff(O, D, faces) if depth < max_depth else None
        if hit is None:
            if D[0] > 1e-9:
                t_s = (PUPIL_X - O[0]) / D[0]
                if t_s > 0:
                    exits.append(dict(src=src, y=float(O[1] + t_s * D[1]),
                                      ang=math.degrees(math.atan2(D[1], D[0])),
                                      b=float(bs + bp), xs=xs))
            continue
        t, P, normal, name = hit
        new_xs = xs + (float(P[0]),) if 1.0 < P[0] < PUPIL_X - 1.0 else xs
        n1 = (1.0 / n) if not inside else n
        cos1 = -(D @ normal); sq1 = 1.0 - n1 * n1 * (1.0 - cos1 * cos1)
        if sq1 < 0:
            Dr = D + 2.0 * cos1 * normal
            stack.append((P, Dr / np.linalg.norm(Dr), bs, bp, inside, depth + 1, src, new_xs))
        else:
            cos2 = math.sqrt(sq1)
            Rs = ((n1 * cos1 - cos2) / (n1 * cos1 + cos2)) ** 2
            Rp = ((n1 * cos2 - cos1) / (n1 * cos2 + cos1)) ** 2
            Dt = n1 * D + (n1 * cos1 - cos2) * normal
            stack.append((P, Dt / np.linalg.norm(Dt), bs * (1 - Rs), bp * (1 - Rp),
                          not inside, depth + 1, src, new_xs))
            Dr = D + 2.0 * cos1 * normal
            stack.append((P, Dr / np.linalg.norm(Dr), bs * Rs, bp * Rp, inside, depth + 1, src, new_xs))
    return exits


def validate(n_random=10):
    """Validate the BRANCHING circle tracer vs the JS oracle on champions AND random designs."""
    from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
    import arvr_prism as AV
    rng = np.random.default_rng(3)
    cand = []
    ap = REPO / "ns_probe" / "results_arvr_qd" / "archive.json"
    if ap.exists():
        a = json.loads(ap.read_text())
        cand = [("champ", np.array(c["genotype"], float))
                for c in sorted(a["cells"], key=lambda c: -c["fitness"])[:4]]
    print("=" * 78)
    print("BRANCHING PRISM TRACER vs JS oracle  (champions + random designs)")
    print("=" * 78)
    passes = tot = 0
    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        while len([c for c in cand if c[0] == "rand"]) < n_random:
            xx = rng.uniform(AV.LO, AV.HI)
            if AV.geometry_valid_arvr(AV.prism_dict(xx)):
                cand.append(("rand", xx))
        for tag, base in cand:
            res, _ = runner.simulate(AV.scene_arvr(base)); om = AV.arvr_metrics(res)
            ex = trace_branching(base); m = metrics(ex)
            tot += 1
            if not om["ordering_ok"] and not m["ordering_ok"]:
                passes += 1; print(f"  [{tag}] both reject (ok)"); continue
            dfit = abs(m["fitness"] - om["fitness"])
            dfov = abs(m["fov"] - om["fov"])
            ok = (m["ordering_ok"] == om["ordering_ok"]) and dfit < 0.05 and dfov < 1.5
            passes += ok
            print(f"  [{tag}] fit {m['fitness']:.3f}/{om['fitness']:.3f}  FOV {m['fov']:.2f}/{om['fov']:.2f}  "
                  f"sig {m['sigma_mean'] or 0:.3f}/{om['sigma_mean'] or 0:.3f}  "
                  f"Tmin {m['T_min']:.2f}/{om['T_min']:.2f}  {'OK' if ok else 'MISMATCH'}")
    print("=" * 78); print(f"PASS {passes}/{tot}")
    return passes == tot


if __name__ == "__main__":
    validate()
