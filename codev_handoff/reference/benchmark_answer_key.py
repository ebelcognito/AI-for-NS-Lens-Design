"""Final Tier-A geometry + answer key for the freeform-prism CODE V benchmark."""
import numpy as np

N = 1.5300                                   # prism model glass nd
CRIT = np.degrees(np.arcsin(1.0 / N))

B1, B2, B3 = 14.0, -16.0, 80.0               # face tilts (deg, normal from +Z toward +Y)
L2, L4 = 7.5, 4.0                            # S1->S2 and S1(TIR)->S3 path lengths
Z1 = 18.0                                    # eye relief


def vec(a):
    a = np.radians(a)
    return np.array([np.sin(a), np.cos(a)])  # (y, z)


ta = B1 - np.degrees(np.arcsin(np.sin(np.radians(B1)) / N))   # dir inside after S1
tirS1 = B1 + 2 * abs(B2) + ta                                  # incidence at S1 (2nd hit)
tc = 2 * B1 + 2 * abs(B2) + ta                                 # dir after TIR at S1

P1 = np.array([0.0, Z1])
P2 = P1 + L2 * vec(ta)
nS1, d3 = vec(B1), vec(180 - 2 * abs(B2) - ta)
L3 = np.dot(P1 - P2, nS1) / np.dot(d3, nS1)
P3 = P2 + L3 * d3
P4 = P3 + L4 * vec(tc)

s2i = ta + abs(B2)
s3i = abs(tc - B3)
exit_ang = B3 - np.degrees(np.arcsin(N * np.sin(np.radians(s3i))))   # tc < B3

print(f"prism nd={N}  critical={CRIT:.3f} deg\n")
print(f"{'#':2} {'face':4} {'event':9} {'seg mm':>9} {'  y':>9} {'  z':>9} {'incid':>8}  note")
rows = [(1, "PUPIL", "stop", 0.0, 0.0, 0.0, None, "EPD 8.0, object at inf"),
        (2, "S1", "refract", Z1, P1[0], P1[1], B1, f"->{ta:.3f} deg inside"),
        (3, "S2", "reflect", L2, P2[0], P2[1], s2i, f"MIRROR ({s2i:.2f} < crit)"),
        (4, "S1", "TIR", L3, P3[0], P3[1], tirS1, f"TIR ok (+{tirS1-CRIT:.2f} margin)"),
        (5, "S3", "refract", L4, P4[0], P4[1], s3i, f"exit {exit_ang:.3f} deg from +Z")]
for i, f, e, t, y, z, a, n in rows:
    ai = f"{a:8.3f}" if a is not None else "     -  "
    print(f"{i:2} {f:5} {e:9} {t:9.4f} {y:9.4f} {z:9.4f} {ai}  {n}")

print(f"\nprism envelope: depth(z) {min(P1[1],P3[1]):.3f}..{max(P2[1],P1[1]):.3f} "
      f"= {max(P2[1],P1[1])-min(P1[1],P3[1]):.3f} mm, height(y) "
      f"{min(P1[0],P2[0]):.3f}..{P4[0]:.3f} = {P4[0]-min(P1[0],P2[0]):.3f} mm")
print(f"relay axis: {exit_ang:.3f} deg from +Z  ({90-exit_ang:.3f} deg from +Y/vertical)")

# ---- relay: cemented achromat, thick-lens EFL -------------------------------
nA, nB = 1.51680, 1.67270
R = [6.40, -4.80, -20.00]
T = [2.50, 1.00]


def efl_thick(R, T, nA, nB):
    # ray trace y=1, u=0 through the cemented doublet (paraxial, y-u)
    y, u = 1.0, 0.0
    surf = [(R[0], 1.0, nA), (R[1], nA, nB), (R[2], nB, 1.0)]
    for i, (r, n1, n2) in enumerate(surf):
        u = (n1 * u - y * (n2 - n1) / r) / n2
        if i < len(T):
            y = y + u * T[i]
    return -1.0 / u, y, u


f, ylast, ulast = efl_thick(R, T, nA, nB)
bfd = -ylast / ulast
print(f"\nrelay achromat  R={R} T={T}  nA={nA} nB={nB}")
print(f"  EFL' = {f:.4f} mm   BFD (last vtx -> paraxial focus) = {bfd:.4f} mm")
halfimg = f * np.tan(np.radians(22.0))
print(f"  half-image height at 22 deg field = {halfimg:.4f} mm  -> "
      f"display {2*halfimg:.2f} mm wide in the drawn meridian")

# relay placed on the exit chief ray, 4 mm after P4
d_exit = vec(exit_ang)
V_relay = P4 + 4.0 * d_exit
print(f"\nrelay L1 front vertex (global y,z) = ({V_relay[0]:.4f}, {V_relay[1]:.4f})")
tot = 4.0 + T[0] + T[1] + bfd
P_img = P4 + tot * d_exit
print(f"microdisplay centre (global y,z)   = ({P_img[0]:.4f}, {P_img[1]:.4f})")
print(f"overall pupil->display path length = {Z1+L2+L3+L4+tot:.4f} mm")
