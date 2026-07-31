"""Unclipped chief-ray hit points, all 5 fields, d line, as-prescribed geometry."""
import numpy as np
import trace3d as T
import fixed_key as FK

g = FK.geom(-16.0)
T.FACES.clear(); T.FACES.update(g["faces"])
crit = np.degrees(np.arcsin(1.0 / T.ND))
print(f"prism n_d=1.53, critical = {crit:.3f} deg;  stated CA half-heights: "
      f"S1 +-10, S2 +-8, S3 +-7 mm\n")

hdr = f"{'field':>6} {'face':>5} {'event':>8} {'seg mm':>9} {'global y':>10} {'global z':>10} " \
      f"{'local y':>9} {'incid':>8}  flag"
for th in [0.0, 11.0, -11.0, 22.0, -22.0]:
    print(hdr if th == 0.0 else "")
    log, _, dout = T.trace(th, 0.0, 0.0, T.ND)
    ca = {"S1": 10.0, "S2": 8.0, "S3": 7.0}
    for i, (face, ev, t, pt, inc, ok) in enumerate(log):
        lx, ly = T.local_xy(pt, face)
        flags = []
        if abs(ly) > ca[face]:
            flags.append(f"OFF FACE (|{ly:.1f}| > {ca[face]:.0f})")
        if ev == "TIR" and inc <= crit:
            flags.append(f"NO TIR ({inc:.2f} < {crit:.2f})")
        print(f"{th:+6.0f} {face:>5} {ev:>8} {t:9.3f} {pt[1]:10.3f} {pt[2]:10.3f} "
              f"{ly:9.3f} {inc:8.3f}  {'; '.join(flags)}")

print("\nprism envelope needed by each field's chief ray (global y and z extent):")
for th in [0.0, 11.0, -11.0, 22.0, -22.0]:
    log, _, _ = T.trace(th, 0.0, 0.0, T.ND)
    ys = [0.0] + [l[3][1] for l in log]
    zs = [18.0] + [l[3][2] for l in log]
    print(f"  field {th:+6.1f}: y {min(ys):8.2f}..{max(ys):8.2f} ({max(ys)-min(ys):6.2f} mm), "
          f"z {min(zs):7.2f}..{max(zs):7.2f} ({max(zs)-min(zs):6.2f} mm)   "
          f"[spec says <=20 y, <=13 z]")
