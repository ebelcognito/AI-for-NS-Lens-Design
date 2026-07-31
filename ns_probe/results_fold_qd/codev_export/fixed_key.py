"""Regenerate the full Tier-A answer key for a candidate mirror tilt B2, and report
the decentred clear apertures the real footprints need."""
import numpy as np
import trace3d as T

ND, NC, NF = T.ND, T.NC, T.NF
B1, B3, L2, L4, Z1 = 14.0, 80.0, 7.5, 4.0, 18.0
FIELDS = [0.0, 11.0, -11.0, 22.0, -22.0]


def vec(a):
    a = np.radians(a)
    return np.array([0.0, np.sin(a), np.cos(a)])


def geom(B2):
    ta = B1 - np.degrees(np.arcsin(np.sin(np.radians(B1)) / ND))
    P1 = np.array([0.0, 0.0, Z1])
    P2 = P1 + L2 * vec(ta)
    nS1, d3 = vec(B1), vec(180 - 2 * abs(B2) - ta)
    L3 = np.dot(P1 - P2, nS1) / np.dot(d3, nS1)
    P3 = P2 + L3 * d3
    tc = 2 * B1 + 2 * abs(B2) + ta
    P4 = P3 + L4 * vec(tc)
    return dict(ta=ta, P1=P1, P2=P2, P3=P3, P4=P4, L3=L3, tc=tc,
                faces={"S1": (P1, vec(B1), (24., 20.)),
                       "S2": (P2, vec(B2), (24., 16.)),
                       "S3": (P4, vec(B3), (24., 14.))})


def efl_thick(R, Tt, nA, nB):
    y, u = 1.0, 0.0
    for i, (r, n1, n2) in enumerate([(R[0], 1.0, nA), (R[1], nA, nB), (R[2], nB, 1.0)]):
        u = (n1 * u - y * (n2 - n1) / r) / n2
        if i < len(Tt):
            y += u * Tt[i]
    return -1.0 / u, -y / u


def report(B2):
    g = geom(B2)
    T.FACES.clear(); T.FACES.update(g["faces"])
    crit = np.degrees(np.arcsin(1.0 / ND))
    print(f"\n{f'  B2 = {B2:+.1f} deg  ':=^78}")
    log, P4, dout = T.trace(0.0, 0.0, 0.0, ND)
    print(f"{'#':2} {'face':5}{'event':9}{'seg mm':>10}{'y':>10}{'z':>10}{'incid':>9}  note")
    segs = [Z1] + [l[2] for l in log[1:]]
    for i, ((face, ev, t, pt, inc, ok), sg) in enumerate(zip(log, segs), start=2):
        note = ""
        if face == "S2":
            note = f"MIRROR ({inc:.2f} < crit {crit:.2f})"
        elif ev == "TIR":
            note = f"TIR ok (+{inc-crit:.2f} margin)"
        print(f"{i:2} {face:5}{ev:9}{sg:10.4f}{pt[1]:10.4f}{pt[2]:10.4f}{inc:9.3f}  {note}")
    exit_ang = np.degrees(np.arctan2(dout[1], dout[2]))
    ys = [g["P1"][1], g["P2"][1], g["P3"][1], g["P4"][1]]
    zs = [g["P1"][2], g["P2"][2], g["P3"][2], g["P4"][2]]
    print(f"\nexit dir (0,{dout[1]:.6f},{dout[2]:.6f})  = {exit_ang:.4f} deg from +Z")
    print(f"prism envelope: z {min(zs):.3f}..{max(zs):.3f} = {max(zs)-min(zs):.3f} mm,"
          f"  y {min(ys):.3f}..{max(ys):.3f} = {max(ys)-min(ys):.3f} mm")

    # per-field TIR margin over the full pupil and all three colours
    print(f"\n{'field':>7} {'min TIR margin':>16} {'max S2 inc':>12}  status")
    worst = 1e9
    for th in FIELDS:
        mm, s2m = 1e9, -1e9
        for ng in (NC, ND, NF):
            c = np.degrees(np.arcsin(1.0 / ng))
            for a in np.radians(np.arange(0, 360, 30)):
                for px, py in [(0, 0), (4*np.cos(a), 4*np.sin(a))]:
                    lg, _, _ = T.trace(th, py, px, ng)
                    mm = min(mm, lg[2][4] - c); s2m = max(s2m, lg[1][4])
        worst = min(worst, mm)
        print(f"{th:+7.0f} {mm:+16.2f} {s2m:12.2f}  "
              f"{'FAILS' if mm < 0 else ('tight' if mm < 5 else 'ok')}"
              f"{'  S2 WOULD TIR' if s2m > crit else ''}")
    print(f"worst-case TIR margin over all fields/pupil/colours: {worst:+.2f} deg")

    # footprints -> what the clear apertures must actually be
    print(f"\n{'face':5} {'local x span':>22} {'local y span':>24}  suggested CA")
    for face in ("S1", "S2", "S3"):
        xs, yss = [], []
        for th in FIELDS:
            for ng in (NC, ND, NF):
                for a in np.radians(np.arange(0, 360, 30)):
                    lg, _, _ = T.trace(th, 4*np.sin(a), 4*np.cos(a), ng)
                    for f2, ev, t, pt, inc, ok in lg:
                        if f2 == face:
                            lx, ly = T.local_xy(pt, face)
                            xs.append(lx); yss.append(ly)
        xr = (min(xs), max(xs)); yr = (min(yss), max(yss))
        yc = 0.5 * (yr[0] + yr[1]); yh = 0.5 * (yr[1] - yr[0])
        print(f"{face:5} [{xr[0]:8.3f},{xr[1]:8.3f}] [{yr[0]:9.3f},{yr[1]:9.3f}]  "
              f"REX {max(abs(xr[0]),abs(xr[1]))+1:.1f}  REY {yh+1:.1f} "
              f"decentred YDE {yc:+.3f}")

    # relay
    f, bfd = efl_thick([6.40, -4.80, -20.00], [2.50, 1.00], 1.51680, 1.67270)
    d = np.array([0.0, dout[1], dout[2]])
    V = g["P4"] + 4.0 * d
    Pi = g["P4"] + (4.0 + 2.5 + 1.0 + bfd) * d
    print(f"\nrelay EFL'={f:.4f} BFD={bfd:.4f}  half-img@22deg={f*np.tan(np.radians(22)):.4f}")
    print(f"relay L1 front vertex (y,z) = ({V[1]:.4f}, {V[2]:.4f})")
    print(f"display centre        (y,z) = ({Pi[1]:.4f}, {Pi[2]:.4f})")
    print(f"pupil->display path = {Z1+L2+g['L3']+L4+4.0+2.5+1.0+bfd:.4f} mm")
    return worst


if __name__ == "__main__":
    for b in (-16.0, -19.0, -20.0, -21.0):
        report(b)
