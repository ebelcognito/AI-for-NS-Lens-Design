"""For the as-prescribed plane-faced prism: which field angles actually CLOSE the
S1->S2->S1->S3 path? Separate the three failure modes so the repair is targeted."""
import numpy as np
import trace3d as T
import fixed_key as FK

CA = {"S1": 10.0, "S2": 8.0, "S3": 7.0}      # stated half-heights


def check(th, B2=-16.0, pupil=0.0, aperture=True):
    g = FK.geom(B2)
    T.FACES.clear(); T.FACES.update(g["faces"])
    fails = []
    for ng in (T.NC, T.ND, T.NF):
        crit = np.degrees(np.arcsin(1.0 / ng))
        pts = [(0.0, 0.0)] if pupil == 0 else \
              [(pupil * np.cos(a), pupil * np.sin(a)) for a in np.radians(range(0, 360, 30))]
        for px, py in pts:
            log, _, dout = T.trace(th, py, px, ng)
            for i, (face, ev, t, pt, inc, ok) in enumerate(log):
                lx, ly = T.local_xy(pt, face)
                if t < 0:
                    fails.append("backward")
                if aperture and abs(ly) > CA[face]:
                    fails.append("off-face")
                if ev == "TIR" and inc <= crit:
                    fails.append("no-TIR")
            if dout is None:
                fails.append("trapped")
    return sorted(set(fails))


print(f"{'CLOSURE SCAN, as-prescribed geometry (B2=-16), chief rays':-^74}")
print(f"{'field':>7}  {'apertures enforced':<26} {'apertures ignored':<20}")
ok_ap, ok_noap = [], []
for th in np.arange(-24, 24.1, 1.0):
    fa = check(th, aperture=True)
    fn = check(th, aperture=False)
    if not fa:
        ok_ap.append(th)
    if not fn:
        ok_noap.append(th)
    if abs(th) % 2 == 0:
        print(f"{th:+7.0f}  {(', '.join(fa) or 'CLOSES'):<26} {(', '.join(fn) or 'CLOSES'):<20}")

print(f"\nchief ray closes WITH stated apertures : "
      f"{min(ok_ap):+.0f} .. {max(ok_ap):+.0f} deg" if ok_ap else "never")
print(f"chief ray closes IGNORING apertures    : "
      f"{min(ok_noap):+.0f} .. {max(ok_noap):+.0f} deg" if ok_noap else "never")

# now with the real 8 mm pupil
print(f"\n{'same, but requiring the whole 8 mm pupil to close':-^74}")
ok_full = [th for th in np.arange(-24, 24.1, 0.5)
           if not check(th, pupil=4.0, aperture=True)]
if ok_full:
    print(f"full-pupil closure: {min(ok_full):+.1f} .. {max(ok_full):+.1f} deg "
          f"-> largest symmetric field about 0 = +-{min(abs(min(ok_full)), abs(max(ok_full))):.1f} deg")
    c = 0.5 * (min(ok_full) + max(ok_full))
    print(f"largest field centred at {c:+.2f} deg = +-{0.5*(max(ok_full)-min(ok_full)):.2f} deg")
else:
    print("full-pupil closure: NONE")

# does raising |B2| help the aperture/backward failures at all?
print(f"\n{'does changing the mirror tilt B2 rescue the +field?':-^74}")
for B2 in (-16.0, -18.0, -21.0, -24.0):
    ok = [th for th in np.arange(-24, 24.1, 1.0) if not check(th, B2=B2, aperture=True)]
    rng = f"{min(ok):+.0f}..{max(ok):+.0f}" if ok else "none"
    print(f"  B2={B2:+.0f}: chief-ray closure {rng:>12}   "
          f"(+11 deg: {', '.join(check(11.0, B2=B2)) or 'closes'})")
