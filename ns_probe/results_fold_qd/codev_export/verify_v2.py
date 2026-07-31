"""Self-contained verifier for BENCHMARK_PROMPT_V2.md.

Regenerates every number in §2 from the 8 geometry parameters, and re-checks that the
S1 -> S2 -> S1 -> S3 path closes for all 5 fields over the whole 8 mm pupil at all 3
wavelengths. Run it before editing the prompt; the V1 answer key was central-ray-only
and that is exactly how the unsatisfiable spec got through.

    python3 verify_v2.py
"""
import numpy as np

# ---- model glass and waveband ------------------------------------------------
ND, VD = 1.5300, 56.0
DNFC = (ND - 1.0) / VD
NC = ND - 0.31 * DNFC                  # ~C line (red): worst case for TIR
NF = NC + DNFC                         # ~F line (blue)
WL = {"C 656.27": NC, "d 587.56": ND, "F 486.13": NF}

# ---- the design --------------------------------------------------------------
Z1 = 18.0                              # eye relief, pupil -> S1 vertex
B1, B2, B3 = 12.856231, -16.879800, 52.965266          # face tilts, deg from +Z
L2, L4 = 9.709077, 16.813226           # S1->S2 and S1'->S3 along the central chief ray
C1, C2, C3 = -0.011554, -0.021733, -0.002708           # curvatures, 1/mm
FIELDS = [0.0, 11.0, -11.0, 22.0, -22.0]
EPD = 8.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def vec(a):
    a = np.radians(a)
    return np.array([0.0, np.sin(a), np.cos(a)])


def pupil(nring=12):
    r = EPD / 2
    return [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a))
                           for a in np.radians(np.arange(0, 360, 360 / nring))]


def rays(fields, nring=12):
    P, D, N = [], [], []
    for th in fields:
        d = unit([0.0, np.sin(np.radians(th)), np.cos(np.radians(th))])
        for px, py in pupil(nring):
            for n in WL.values():
                P.append([px, py, 0.0]); D.append(d); N.append(n)
    return np.array(P), np.array(D), np.array(N)


# ---- surface maths (tilted sphere: vertex + axis + curvature) ----------------
def snorm(pts, face):
    V, ax, c = face
    if abs(c) < 1e-12:
        return np.repeat(ax[None, :], len(pts), axis=0)
    n = (pts - (V + ax / c)) * np.sign(1.0 / c)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def shit(P, D, face):
    """intersection, picking the root nearest the tangent-plane crossing."""
    V, ax, c = face
    den = D @ ax
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    tp = ((V - P) @ ax) / den
    if abs(c) < 1e-12:
        return tp, P + tp[:, None] * D
    R = 1.0 / c
    L = P - (V + R * ax)
    b = 2 * (D * L).sum(1)
    disc = b * b - 4 * ((L * L).sum(1) - R * R)
    if np.any(disc < 0):
        return None, None
    sq = np.sqrt(disc)
    ta, tb = (-b - sq) / 2, (-b + sq) / 2
    t = np.where(np.abs(ta - tp) < np.abs(tb - tp), ta, tb)
    return t, P + t[:, None] * D


def srefr(D, N, n1, n2):
    c = -(D * N).sum(1)
    NN = N * np.sign(c)[:, None]
    c = np.abs(c)
    eta = n1 / n2
    k = 1 - eta ** 2 * (1 - c ** 2)
    out = eta[:, None] * D + (eta * c - np.sqrt(np.clip(k, 0, None)))[:, None] * NN
    return out / np.linalg.norm(out, axis=1, keepdims=True), k > 0


def srefl(D, N):
    return D - 2 * (D * N).sum(1)[:, None] * N


def sinc(D, N):
    return np.degrees(np.arccos(np.clip(np.abs((D * N).sum(1)), 0, 1)))


def localy(pt, face):
    V, ax, _ = face
    return (pt - V) @ np.cross(ax, np.array([1.0, 0.0, 0.0]))


# ---- build the three faces along the central chief ray -----------------------
def build():
    F = {"S1": (np.array([0.0, 0.0, Z1]), vec(B1), C1)}
    P = np.zeros((1, 3)); D = np.array([[0.0, 0.0, 1.0]])
    one, ng = np.array([1.0]), np.array([ND])
    t, P = shit(P, D, F["S1"])
    D = srefr(D, snorm(P, F["S1"]), one, ng)[0]
    F["S2"] = (P[0] + L2 * D[0], vec(B2), C2)
    t2, P2 = shit(P, D, F["S2"])
    D = srefl(D, snorm(P2, F["S2"]))
    t3, P3 = shit(P2, D, F["S1"])
    D = srefl(D, snorm(P3, F["S1"]))
    F["S3"] = (P3[0] + L4 * D[0], vec(B3), C3)
    return F


def march(F, P, D, NG):
    """S1 refract -> S2 mirror -> S1 TIR -> S3 refract."""
    one = np.ones_like(NG)
    out = []
    t, H = shit(P, D, F["S1"]); N = snorm(H, F["S1"])
    out.append(("S1", "refract", t, H, sinc(D, N)))
    D = srefr(D, N, one, NG)[0]
    t, H = shit(H, D, F["S2"]); N = snorm(H, F["S2"])
    out.append(("S2", "mirror", t, H, sinc(D, N)))
    D = srefl(D, N)
    t, H = shit(H, D, F["S1"]); N = snorm(H, F["S1"])
    out.append(("S1", "TIR", t, H, sinc(D, N)))
    D = srefl(D, N)
    t, H = shit(H, D, F["S3"]); N = snorm(H, F["S3"])
    out.append(("S3", "refract", t, H, sinc(D, N)))
    D, ok = srefr(D, N, NG, one)
    return out, D, ok


def main():
    F = build()
    crit_d = np.degrees(np.arcsin(1.0 / ND))
    print(f"{' FACES ':=^78}")
    print(f"n_d {ND}  V_d {VD}  critical {crit_d:.4f} deg (d), "
          f"{np.degrees(np.arcsin(1/NC)):.4f} (C), {np.degrees(np.arcsin(1/NF)):.4f} (F)")
    for k in ("S1", "S2", "S3"):
        V, ax, c = F[k]
        R = 1.0 / c
        C = V + R * ax
        print(f"{k}: vertex (0,{V[1]:11.6f},{V[2]:11.6f})  "
              f"normal (0,{ax[1]:+.6f},{ax[2]:+.6f})  "
              f"tilt {np.degrees(np.arctan2(ax[1], ax[2])):+9.4f}  "
              f"RDY {R:10.4f}  centre (0,{C[1]:.4f},{C[2]:.4f})")

    print(f"\n{' CENTRAL CHIEF RAY ':=^78}")
    log, D, _ = march(F, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
                      np.array([ND]))
    print(f"{'#':>2} {'face':5}{'event':9}{'seg mm':>10}{'y':>11}{'z':>11}"
          f"{'incid':>10}  note")
    for i, (face, ev, t, H, inc) in enumerate(log, start=1):
        note = (f"below critical -> MIRROR" if ev == "mirror" else
                f"+{inc[0]-crit_d:.3f} over critical" if ev == "TIR" else "")
        print(f"{i:2} {face:5}{ev:9}{t[0]:10.4f}{H[0][1]:11.4f}{H[0][2]:11.4f}"
              f"{inc[0]:10.4f}  {note}")
    d = D[0]
    print(f"\nexit direction (0,{d[1]:.6f},{d[2]:.6f}) = "
          f"{np.degrees(np.arctan2(d[1], d[2])):.4f} deg from +Z  "
          f"-> relay-axis ADE {-np.degrees(np.arctan2(d[1], d[2])):.4f}")
    V3 = F["S3"][0]
    print(f"doublet front vertex at 4.0 mm: (0,{(V3+4*d)[1]:.4f},{(V3+4*d)[2]:.4f})")

    print(f"\n{' FULL FIELD x PUPIL x WAVELENGTH ':=^78}")
    P0, D0, NG = rays(FIELDS)
    log, Dout, okout = march(F, P0, D0, NG)
    crit = np.degrees(np.arcsin(1.0 / NG))
    npf = len(NG) // len(FIELDS)
    print(f"{'field':>7} {'S1 TIR min':>11} {'margin':>9} {'S2 max':>9} "
          f"{'S3 max':>9} {'min seg':>9}  status")
    allok = True
    for k, th in enumerate(FIELDS):
        s = slice(k * npf, (k + 1) * npf)
        marg = (log[2][4][s] - crit[s]).min()
        mseg = min(l[2][s].min() for l in log)
        bad = []
        if marg <= 0:
            bad.append("TIR-FAIL")
        if (log[1][4][s] - crit[s]).max() >= 0:
            bad.append("S2-would-TIR")
        if not okout[s].all():
            bad.append("S3-TRAPPED")
        if mseg <= 0:
            bad.append("BACKWARD")
        allok &= not bad
        print(f"{th:+7.1f} {log[2][4][s].min():11.3f} {marg:+9.3f} "
              f"{log[1][4][s].max():9.3f} {log[3][4][s].max():9.3f} {mseg:9.4f}  "
              f"{'closes' if not bad else ' '.join(bad)}")

    print(f"\n{' REQUIRED CLEAR APERTURES ':=^78}")
    for face, idx in (("S1", [0, 2]), ("S2", [1]), ("S3", [3])):
        v = np.concatenate([localy(log[i][3], F[face]) for i in idx])
        lo, hi = v.min(), v.max()
        print(f"{face}: local y [{lo:8.3f},{hi:8.3f}]  centre {0.5*(lo+hi):+7.3f}  "
              f"half {0.5*(hi-lo):7.3f}  -> {2*np.ceil(0.5*(hi-lo)+0.5):.0f} mm tall")

    print(f"\n{'ALL FIELDS CLOSE' if allok else 'SPEC IS NOT SATISFIABLE'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
