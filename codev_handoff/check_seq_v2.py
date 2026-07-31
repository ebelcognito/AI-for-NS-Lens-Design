"""Verify benchmark_tierA_v2.seq against the geometry verify_v2.py generates.

Parses the .seq, rebuilds each surface's global frame from its GLB/XDE/YDE/ZDE/ADE
records, and checks vertex, tilt, radius sign and centre of curvature against the
independently computed design. Also re-runs the full field x pupil x wavelength
closure test so a numeric edit to the .seq cannot silently break satisfiability.

This covers every rubric item that does not need CODE V itself (checks 1-5, 7, 8).
Only "does a real ray trace in CODE V" needs the keyboard.

    python3 check_seq_v2.py
"""
import re
from pathlib import Path

import numpy as np

import verify_v2 as V

import sys

DEFAULT_SEQ = "benchmark_tierA_v2c.seq"
SEQ = Path(__file__).with_name(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEQ)
TOL_MM, TOL_DEG = 0.05, 0.05          # rubric check 5 tolerances

REFLECT_TOKENS = {"REFL", "TIRO", "REFLTIRO"}


def parse(path):
    """Return {surface_label: {field: value}} plus the ordered S-line list."""
    surf, order, idx = {}, [], 0
    for raw in path.read_text().splitlines():
        line = raw.split("!")[0].strip()
        if not line:
            continue
        tok = line.split()
        key = tok[0].upper()

        if key in ("S", "SI") and not re.fullmatch(r"S\d+", tok[0]):
            idx += 1
            label = "SI" if key == "SI" else f"S{idx}"
            tail = tok[3:]
            rec = {"rdy": float(tok[1]), "thi": float(tok[2]),
                   "glass": [t.strip("'") for t in tail if t.startswith("'")],
                   "flags": [t.upper() for t in tail
                             if t.upper() in REFLECT_TOKENS],
                   "junk": [t for t in tail
                            if not t.startswith("'")
                            and t.upper() not in REFLECT_TOKENS]}
            surf[label] = rec
            order.append(label)
        elif len(tok) >= 3 and re.fullmatch(r"S(\d+|I)", tok[1].upper()):
            lbl = tok[1].upper()
            surf.setdefault(lbl, {})[key] = tok[2]
        elif len(tok) == 2 and re.fullmatch(r"S(\d+|I)", tok[1].upper()):
            # bare per-surface flags: STO Sn, DAR Sn, ...
            surf.setdefault(tok[1].upper(), {})[key] = True
        elif key == "PIK":
            surf.setdefault(tok[2].upper(), {}).setdefault("PIK", []).append(tok[1].upper())
    return surf, order


def _num(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def frame(rec):
    """Global vertex and local +Z axis implied by this surface's YDE/ZDE/ADE."""
    y = float(rec.get("YDE", 0.0))
    z = float(rec.get("ZDE", 0.0))
    a = np.radians(float(rec.get("ADE", 0.0)))
    # right-handed rotation about +X by ADE carries zhat=(0,0,1) -> (0,-sinA,cosA)
    return np.array([0.0, y, z]), np.array([0.0, -np.sin(a), np.cos(a)])


def main():
    F = V.build()
    surf, order = parse(SEQ)
    fails = []

    def chk(ok, label, got, want, unit=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {label:38} got {got:>14}  want {want:>14} {unit}")
        if not ok:
            fails.append(label)

    print(f"{' PARSED SURFACE LIST ':=^86}")
    print("  " + "  ".join(order))
    expect = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "SI"]
    chk(order == expect, "surface order", str(len(order)) + " surfs", "11 surfs")

    print(f"\n{' PRISM FACE GEOMETRY (vs verify_v2 build) ':=^86}")
    # CODE V surface -> physical face in the prescription
    for cv, face, ev in (("S2", "S1", "refract"), ("S3", "S2", "mirror"),
                         ("S4", "S1", "TIR"), ("S5", "S3", "refract")):
        Vx, ax, c = F[face]
        R = 1.0 / c
        vtx, zhat = frame(surf[cv])
        print(f"\n  {cv}  = physical face {face}  ({ev})")
        chk(np.allclose(vtx, Vx, atol=TOL_MM), f"{cv} vertex (y,z)",
            f"{vtx[1]:.4f},{vtx[2]:.4f}", f"{Vx[1]:.4f},{Vx[2]:.4f}", "mm")
        got_t = np.degrees(np.arctan2(zhat[1], zhat[2]))
        want_t = np.degrees(np.arctan2(ax[1], ax[2]))
        chk(abs(got_t - want_t) < TOL_DEG, f"{cv} local +Z along outward normal",
            f"{got_t:+.4f}", f"{want_t:+.4f}", "deg")
        chk(abs(surf[cv]["rdy"] - R) < 0.05, f"{cv} RDY", f"{surf[cv]['rdy']:.4f}",
            f"{R:.4f}", "mm")
        # centre of curvature must land where the answer key says
        got_c = vtx + surf[cv]["rdy"] * zhat
        want_c = Vx + R * ax
        chk(np.allclose(got_c, want_c, atol=0.5), f"{cv} centre of curvature (y,z)",
            f"{got_c[1]:.2f},{got_c[2]:.2f}", f"{want_c[1]:.2f},{want_c[2]:.2f}", "mm")

    print(f"\n{' S-LINE TAIL IS PARSEABLE (the silent-drop trap) ':=^86}")
    # CODE V takes ONE glass token on an S line. An n_d/V_d pair made it discard
    # the rest of the line -- INCLUDING REFL/REFLTIRO -- with only a warning.
    for s in order:
        junk = surf[s].get("junk", [])
        if junk or len(surf[s].get("glass", [])) > 1:
            chk(False, f"{s} S-line tail", str(junk or surf[s]['glass']), "1 glass token")
    chk(all(not surf[s].get("junk") and len(surf[s].get("glass", [])) <= 1
            for s in order),
        "no S line has droppable extra data", "clean", "clean")

    print(f"\n{' REFLECTION TYPES (physics, not syntax) ':=^86}")
    chk("REFL" in surf["S3"]["flags"] and not (REFLECT_TOKENS - {"REFL"}) & set(surf["S3"]["flags"]),
        "S3 is a coated mirror (REFL)", str(surf["S3"]["flags"]), "['REFL']")
    chk(bool({"TIRO", "REFLTIRO"} & set(surf["S4"]["flags"])), "S4 is TIR (TIRO)",
        str(surf["S4"]["flags"]), "['TIRO']")
    chk(surf["S2"]["flags"] == [], "S2 is a plain refraction", str(surf["S2"]["flags"]), "[]")
    chk(surf["S5"]["flags"] == [], "S5 is a plain refraction", str(surf["S5"]["flags"]), "[]")

    print(f"\n{' GLASS ASSIGNMENT (medium after each surface) ':=^86}")
    # S3 is REFL: a reflection does not change the medium, so it carries NO glass
    # token (CODE V warns "Extra data ignored" if you give one) and inherits PRISM.
    for s, want in (("S2", "PRISM"), ("S3", None), ("S4", "PRISM"),
                    ("S5", None), ("S6", "CROWN"), ("S7", "FLINT"), ("S8", None)):
        got = (surf[s].get("glass") or [None])[0]
        note = " (inherits PRISM)" if s == "S3" else ""
        chk(got == want, f"{s} medium after{note}", str(got), str(want or "AIR"))

    print(f"\n{' COUPLED FACE (S2 == S4, one physical surface) ':=^86}")
    piks = set(surf["S4"].get("PIK", []))
    chk({"RDY", "YDE", "ZDE", "ADE"} <= piks, "S4 picks up S2",
        str(sorted(piks)), "RDY YDE ZDE ADE")
    chk(surf["S2"]["rdy"] == surf["S4"]["rdy"], "S2/S4 explicit RDY agree",
        f"{surf['S4']['rdy']:.4f}", f"{surf['S2']['rdy']:.4f}")
    chk(surf["S2"]["ADE"] == surf["S4"]["ADE"], "S2/S4 explicit ADE agree",
        surf["S4"]["ADE"], surf["S2"]["ADE"])

    print(f"\n{' STOP, GLOBAL REFERENCE, APERTURES ':=^86}")
    chk(surf["S1"].get("STO") is True, "S1 is the stop", str(surf["S1"].get("STO")), "True")
    # Placement must make YDE/ZDE/ADE mean GLOBAL coordinates. Two ways:
    #   GLB (syntax not yet established on this build), or
    #   DAR + THI 0 so the base frame never accumulates.
    glb = {s: surf[s].get("GLB") for s in ("S2", "S3", "S4", "S5", "S6")}
    dar = {s: surf[s].get("DAR") for s in ("S2", "S3", "S4", "S5")}
    if any(glb.values()):
        chk(all(v in ("1", "S1") for v in glb.values()),
            "S2..S6 globally referenced (GLB)", str(sorted(set(glb.values()))), "GLB")
    else:
        chk(all(dar.values()), "S2..S5 placed absolutely (DAR)",
            str(sum(bool(v) for v in dar.values())) + "/4", "4/4")
        chk(all(abs(surf[s]["thi"]) < 1e-12 for s in ("S1", "S2", "S3", "S4", "S5")),
            "THI 0 through the prism (frame cannot drift)", "all 0", "all 0")
        chk(not surf["S6"].get("DAR"), "S6 has NO DAR (tilt must propagate to relay)",
            str(bool(surf["S6"].get("DAR"))), "False")
    for s, rey in (("S2", 19.0), ("S3", 16.0), ("S4", 19.0), ("S5", 14.0)):
        chk(abs(float(surf[s]["REY"]) - rey) < 1e-6, f"{s} REY", surf[s]["REY"], f"{rey}")

    # apertures must actually contain every traced ray
    P0, D0, NG = V.rays(V.FIELDS)
    log, _, _ = V.march(F, P0, D0, NG)
    print(f"\n{' APERTURES CONTAIN ALL TRACED RAYS ':=^86}")
    for cv, face, idxs in (("S2", "S1", [0]), ("S3", "S2", [1]),
                           ("S4", "S1", [2]), ("S5", "S3", [3])):
        v = np.concatenate([V.localy(log[i][3], F[face]) for i in idxs])
        half, cen = float(surf[cv]["REY"]), float(surf[cv]["ADY"])
        need = max(abs(v.max() - cen), abs(cen - v.min()))
        chk(need <= half, f"{cv} rays fit in REY about ADY",
            f"{need:.3f}", f"<= {half:.1f}", "mm")

    print(f"\n{' RELAY AXIS ':=^86}")
    _, _, _ = F["S3"]
    log1, D, _ = V.march(F, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
                         np.array([V.ND]))
    d = D[0]
    want_ade = -np.degrees(np.arctan2(d[1], d[2]))
    chk(abs(float(surf["S6"]["ADE"]) - want_ade) < TOL_DEG, "S6 relay-axis ADE",
        surf["S6"]["ADE"], f"{want_ade:.4f}", "deg")
    want_v = F["S3"][0] + 4.0 * d
    got_v, _ = frame(surf["S6"])
    chk(np.allclose(got_v, want_v, atol=TOL_MM), "S6 vertex 4 mm along exit axis",
        f"{got_v[1]:.4f},{got_v[2]:.4f}", f"{want_v[1]:.4f},{want_v[2]:.4f}", "mm")

    print(f"\n{' CLOSURE: 5 FIELDS x 8 mm PUPIL x 3 WAVELENGTHS ':=^86}")
    crit = np.degrees(np.arcsin(1.0 / NG))
    npf = len(NG) // len(V.FIELDS)
    allok = True
    for k, th in enumerate(V.FIELDS):
        s = slice(k * npf, (k + 1) * npf)
        marg = (log[2][4][s] - crit[s]).min()
        s2max = (log[1][4][s] - crit[s]).max()
        mseg = min(l[2][s].min() for l in log)
        ok = marg > 0 and s2max < 0 and mseg > 0
        allok &= ok
        print(f"  [{'ok' if ok else 'FAIL'}] field {th:+6.1f} deg   "
              f"TIR margin {marg:+7.3f}   S3 below crit by {-s2max:6.3f}   "
              f"min seg {mseg:7.4f} mm")
    if not allok:
        fails.append("closure")

    print("\n" + "=" * 86)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: " + ", ".join(fails))
        return 1
    print("ALL CHECKS PASS -- everything gradable without CODE V is correct.")
    print("Remaining at the keyboard: does a real ray trace (rubric check 6),")
    print("and do the centres of curvature listed in the .seq header come out right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
