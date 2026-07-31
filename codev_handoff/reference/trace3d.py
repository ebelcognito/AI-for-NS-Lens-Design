"""Independent 3D vector trace of the Tier-A prism, from the face point+normal table in
the benchmark prompt (NOT from the closed-form angles in benchmark_answer_key.py).
Checks: hit points, incidence angles, TIR margin over the FULL field+pupil, apertures."""
import numpy as np
import itertools

ND, VD = 1.5300, 56.0
DNFC = (ND - 1.0) / VD                 # nF - nC for the model glass
NC = ND - 0.31 * DNFC                  # ~C line (red)  -> worst case for TIR
NF = NC + DNFC                         # ~F line (blue)
IDX = {"C 656.27": NC, "d 587.56": ND, "F 486.13": NF}

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

# --- faces: point on plane, outward normal (global XYZ) ----------------------
FACES = {
    "S1": (np.array([0.0, 0.0, 18.0]),     unit([0, 0.241922, 0.970296]), (24.0, 20.0)),
    "S2": (np.array([0.0, 0.6409, 25.4726]), unit([0, -0.275637, 0.961262]), (24.0, 16.0)),
    "S3": (np.array([0.0, 11.3144, 17.7788]), unit([0, 0.984808, 0.173648]), (24.0, 14.0)),
}

def plane_hit(P, d, face):
    Q, n, _ = FACES[face]
    denom = np.dot(d, n)
    if abs(denom) < 1e-12:
        return None, None
    t = np.dot(Q - P, n) / denom
    return t, P + t * d

def local_xy(pt, face):
    """in-plane coords of a hit point, origin at the face's reference point."""
    Q, n, _ = FACES[face]
    ex = np.array([1.0, 0.0, 0.0])                 # X is in-plane for every face here
    ey = np.cross(n, ex)
    v = pt - Q
    return np.dot(v, ex), np.dot(v, ey)

def refract(d, n, n1, n2):
    """d incoming unit dir, n face normal; returns refracted dir or None for TIR."""
    c = -np.dot(unit(n), d)
    nn = unit(n) * np.sign(c)                      # normal facing into the incoming ray
    c = abs(c)
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - c * c)
    if k < 0:
        return None
    return unit(eta * d + (eta * c - np.sqrt(k)) * nn)

def reflect(d, n):
    n = unit(n)
    return unit(d - 2.0 * np.dot(d, n) * n)

def incidence(d, n):
    return np.degrees(np.arccos(min(1.0, abs(np.dot(unit(d), unit(n))))))

def trace(theta_deg, py, px, nglass):
    """chief/marginal ray: field angle theta (deg, in YZ), pupil point (px,py) at z=0."""
    d = unit([0.0, np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))])
    P = np.array([px, py, 0.0])
    log = []
    # 1. S1 refract in
    t, P1 = plane_hit(P, d, "S1")
    i1 = incidence(d, FACES["S1"][1])
    d = refract(d, FACES["S1"][1], 1.0, nglass)
    log.append(("S1", "refract", t, P1, i1, True))
    P = P1
    # 2. S2 reflect (mirror)
    t, P2 = plane_hit(P, d, "S2")
    i2 = incidence(d, FACES["S2"][1])
    d = reflect(d, FACES["S2"][1])
    log.append(("S2", "mirror", t, P2, i2, True))
    P = P2
    # 3. S1 again, must TIR
    t, P3 = plane_hit(P, d, "S1")
    i3 = incidence(d, FACES["S1"][1])
    crit = np.degrees(np.arcsin(1.0 / nglass))
    tir_ok = i3 > crit
    d = reflect(d, FACES["S1"][1])
    log.append(("S1", "TIR", t, P3, i3, tir_ok))
    P = P3
    # 4. S3 refract out
    t, P4 = plane_hit(P, d, "S3")
    i4 = incidence(d, FACES["S3"][1])
    dout = refract(d, FACES["S3"][1], nglass, 1.0)
    log.append(("S3", "refract", t, P4, i4, dout is not None))
    return log, P4, dout

# ---------------------------------------------------------------- central chief
print(f"{'CENTRAL CHIEF RAY (d line)':-^78}")
crit_d = np.degrees(np.arcsin(1.0 / ND))
print(f"n_d={ND}  critical={crit_d:.3f} deg   "
      f"(n_C={NC:.5f} -> crit {np.degrees(np.arcsin(1/NC)):.3f} deg)\n")
log, P4, dout = trace(0.0, 0.0, 0.0, ND)
print(f"{'face':5}{'event':10}{'seg mm':>10}{'y':>10}{'z':>10}{'incid':>9}   local(x,y)")
for face, ev, t, pt, inc, ok in log:
    lx, ly = local_xy(pt, face)
    print(f"{face:5}{ev:10}{t:10.4f}{pt[1]:10.4f}{pt[2]:10.4f}{inc:9.3f}   "
          f"({lx:6.3f},{ly:7.3f}) {'' if ok else '  <-- FAILS'}")
print(f"\nexit dir (x,y,z) = ({dout[0]:.6f}, {dout[1]:.6f}, {dout[2]:.6f})")
print(f"  benchmark says   (0, 0.834030, 0.551718)   "
      f"delta = {np.linalg.norm(dout - np.array([0,0.834030,0.551718])):.2e}")
print(f"exit angle from +Z = {np.degrees(np.arctan2(dout[1], dout[2])):.4f} deg")

# ------------------------------------------------- full field x pupil x colour
print(f"\n{'FULL FIELD x PUPIL x WAVELENGTH SWEEP':-^78}")
fields = [0.0, 11.0, -11.0, 22.0, -22.0]
# pupil sampling: rim + centre of the 8 mm EPD
pup = [(0, 0)] + [(4 * np.cos(a), 4 * np.sin(a))
                  for a in np.radians(np.arange(0, 360, 30))]
print(f"{'field':>7} {'wl':>10} {'S1 in':>8} {'S2':>8} {'S1 TIR':>9} {'crit':>7} "
      f"{'margin':>8} {'S3':>8}  status")
worst = ("", 1e9)
for th in fields:
    for wl, ng in IDX.items():
        crit = np.degrees(np.arcsin(1.0 / ng))
        i1s, i2s, i3s, i4s, bad = [], [], [], [], []
        for px, py in pup:
            log, _, dout = trace(th, py, px, ng)
            i1s.append(log[0][4]); i2s.append(log[1][4])
            i3s.append(log[2][4]); i4s.append(log[3][4])
            if not log[2][5]:
                bad.append("TIR-FAIL")
            if log[1][4] > crit:
                bad.append("S2>crit")
            if dout is None:
                bad.append("S3-TIRtrap")
        m = min(i3s) - crit
        if m < worst[1]:
            worst = (f"field {th:+.0f} deg, {wl}", m)
        status = "OK" if not bad else " ".join(sorted(set(bad)))
        print(f"{th:+7.0f} {wl:>10} {min(i1s):8.2f} {max(i2s):8.2f} {min(i3s):9.2f} "
              f"{crit:7.2f} {m:+8.2f} {min(i4s):8.2f}  {status}")
print(f"\nworst TIR margin anywhere: {worst[1]:+.2f} deg  at {worst[0]}")

# ------------------------------------------------------------ aperture check
print(f"\n{'FOOTPRINT vs STATED CLEAR APERTURE':-^78}")
foot = {k: [1e9, -1e9, 1e9, -1e9] for k in ("S1", "S2", "S3")}
for th in fields:
    for wl, ng in IDX.items():
        for px, py in pup:
            log, _, _ = trace(th, py, px, ng)
            for face, ev, t, pt, inc, ok in log:
                lx, ly = local_xy(pt, face)
                f = foot[face]
                f[0], f[1] = min(f[0], lx), max(f[1], lx)
                f[2], f[3] = min(f[2], ly), max(f[3], ly)
for face, (xmin, xmax, ymin, ymax) in foot.items():
    ax, ay = FACES[face][2]
    okx = (xmax - xmin) <= ax and abs(xmin) <= ax / 2 and abs(xmax) <= ax / 2
    oky = abs(ymin) <= ay / 2 and abs(ymax) <= ay / 2
    print(f"{face}: local x [{xmin:7.3f},{xmax:7.3f}] vs +-{ax/2:.1f}  {'ok' if okx else 'OVER'}"
          f"   |   local y [{ymin:7.3f},{ymax:7.3f}] vs +-{ay/2:.1f}  {'ok' if oky else 'OVER'}")
