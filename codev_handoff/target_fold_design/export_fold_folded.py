"""Export fold designs to CODE V as a FOLDED-AXIS model (real thicknesses threading
the chief ray + fold mirrors), instead of the zero-thickness global-DAR model.

WHY: the global-DAR export (export_fold_for_codev.py) imports but (a) has no glass
VOLUME (THI 0 everywhere) so CODE V draws faces as lines, and (b) gives the reference
ray no coherent axis -> "Unable to find chief ray". A folded-axis model fixes both:
real thicknesses = a real glass block that renders solid AND a chief ray that threads
the fold. This is how CODE V's own prism examples (Pechan/Dove/penta) are built.

CONSTRUCTION (per surface, in hit order along the chief ray):
  * THI  = physical segment length to the NEXT hit (the axis follows the beam).
  * RDY  = the face radius at that hit (curved faces) or INFINITY (flat edges).
  * tilt = the surface's tilt in the running local frame:
             - refractive face: signed incidence angle of the axis on the face,
             - fold mirror:     signed HALF of the axis deflection (a mirror tilted
                                by a deflects the beam by 2a).
  * REFL/TIRO for TIR folds (stay in glass n); glass toggles at the two refractions.
  * REFLECTED SPACE: after each reflection CODE V works in reflected coordinates, so
    the thickness sign flips at every fold (thi_sign = (-1)**n_reflections_so_far).

Geometry is from _folded_trace.trace() (analytic chief ray, validated vs CONTEXT §6
to <0.001 mm). Coordinate map unchanged: CODE V Z=engine x, Y=engine y, X=out-of-plane
(curved faces are cylinders along X; a sphere traces identically in the meridional plane
so the .seq emits RDY only -- convert to Y-cylinder in the editor if you want).

HONEST NOTE: no CODE V here to trace against. The GEOMETRY (thicknesses, radii,
incidence angles) is exact. The one thing that needs a keyboard check is the CODE V
fold SIGN convention (which way each mirror leans / ADE handedness). Validate it on a
1-bounce design first (export_fold_folded.py --tir 1) -- if the single fold goes the
right way, the multi-bounce signs follow. Then scale to the 4-bounce champion.

    python3 ns_probe/export_fold_folded.py --tir 3      # the champion you pointed at
    python3 ns_probe/export_fold_folded.py --tir 1      # simplest fold: validate signs
    python3 ns_probe/export_fold_folded.py --cell 4 11
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from _folded_trace import trace
from export_fold_for_codev import faces_of, champions_by_depth, WL_NM, BEAM_W, X_SENSOR

REPO = Path(__file__).resolve().parents[1]
SENSOR_HALF = 20.0   # default detector half-size (mm); user controls this


def _seg_len(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _ang(d):  # global direction angle of a unit vector, deg
    return math.degrees(math.atan2(d[1], d[0]))


def build_folded(p, sig):
    """Return an ordered list of folded-axis surface records + the sensor record."""
    hits, sensor = trace(p, sig)

    # segment endpoints: source, each hit, sensor
    src = (0.0, p["ray_y"])
    pts = [src] + [h["point"] for h in hits] + [sensor]

    recs = []
    medium = "AIR"
    n_refl = 0
    for i, h in enumerate(hits):
        f = faces_of(p)[h["face"]]
        seg_in = _seg_len(pts[i], pts[i + 1])       # source/prev-hit -> this hit
        seg_out = _seg_len(pts[i + 1], pts[i + 2])  # this hit -> next hit/sensor
        phi_in = _ang(h["n_in"])
        phi_out = _ang(h["n_out"])
        deflect = ((phi_out - phi_in + 180.0) % 360.0) - 180.0   # signed, (-180,180]

        reflective = (h["event"] == "tir")
        if reflective:
            local_tilt = deflect / 2.0                 # mirror tilt = half the deflection
        else:
            # signed incidence: sign from cross(n_in, normal) about +X (out of plane)
            s = 1.0 if (h["n_in"][0] * h["normal"][1] - h["n_in"][1] * h["normal"][0]) > 0 else -1.0
            local_tilt = s * h["incidence"]

        if reflective:
            med_after = medium
        else:
            med_after = "GLASS" if medium == "AIR" else "AIR"

        recs.append(dict(
            i=i + 1, face=h["face"], event=h["event"], reflective=reflective,
            kind=f["kind"], radius=f["radius"],
            thi_in=seg_in, thi_out=seg_out,
            incidence=h["incidence"], deflect=deflect, local_tilt=local_tilt,
            n_refl_before=n_refl, medium_before=medium, medium_after=med_after,
            global_point=h["point"], global_normal_ang=_ang(h["normal"])))
        medium = med_after
        if reflective:
            n_refl += 1

    return recs, sensor, n_refl


def emit_spec_txt(cell, p, sig, recs, sensor, n_refl):
    L = []
    cid = f"fold_cell_{int(cell['i0'])}_{int(cell['i1'])}"
    L.append(f"# FOLDED-AXIS CODE V spec : {cid}")
    L.append(f"# dom_signature          : {sig}")
    L.append(f"# n={p['n']:.4f}  wl={WL_NM} nm  critical={math.degrees(math.asin(1/p['n'])):.3f} deg")
    L.append("# Real thicknesses thread the chief ray; folds are REFL/TIRO (stay in glass).")
    L.append("# thi_sign flips at each reflection (CODE V reflected space).")
    L.append("#")
    L.append(f"OBJECT: infinity (collimated)   field YAN = {p['ang']:.4f} deg")
    L.append(f"STOP  : at source, YDE={p['ray_y']:.4f}, semi-aperture={BEAM_W/2:.3f} mm")
    L.append(f"        thickness stop->face1 = {_seg_len((0.0,p['ray_y']), recs[0]['global_point']):.4f} mm")
    L.append("")
    hdr = (f"{'i':>2} {'face':>4} {'event':>8} {'kind':>5} {'RDY(mm)':>13} "
           f"{'THI_out':>9} {'incid':>7} {'deflect':>8} {'tilt(loc)':>10} {'medium_after':>13}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in recs:
        rdy = "INF" if r["radius"] is None else f"{r['radius']:.4f}"
        thi = r["thi_out"] * ((-1) ** (r["n_refl_before"] + (1 if r["reflective"] else 0)))
        med = ("REFL/glass" if r["reflective"] else r["medium_after"])
        L.append(f"{r['i']:>2} {r['face']:>4} {r['event']:>8} "
                 f"{('cyl' if r['kind']=='cyl' else 'flat'):>5} {rdy:>13} "
                 f"{thi:>9.4f} {r['incidence']:>7.3f} {r['deflect']:>8.3f} "
                 f"{r['local_tilt']:>10.4f} {med:>13}")
    L.append("")
    L.append(f"IMAGE : detector plane, {n_refl} folds total.  "
             f"sensor hit (engine x,y) = ({sensor[0]:.3f}, {sensor[1]:.3f})")
    L.append(f"        semi-aperture (detector half-size) = {SENSOR_HALF:.1f} mm  <-- edit to size the sensor")
    return "\n".join(L) + "\n"


def emit_seq(cell, p, sig, recs, sensor, n_refl):
    cid = f"fold_cell_{int(cell['i0'])}_{int(cell['i1'])}"
    n = p["n"]
    stop_thi = _seg_len((0.0, p["ray_y"]), recs[0]["global_point"])
    S = []
    S.append(f"! FOLDED-AXIS CODE V macro : {cid}")
    S.append(f"! dom_signature : {sig}")
    S.append("! Real thicknesses thread the chief ray; folds = TIRO (stay in glass).")
    S.append("! ADE handedness/fold sign: VALIDATE on a 1-bounce design first (see header of exporter).")
    S.append("LEN NEW")
    S.append("DIM M")
    S.append(f"WL {WL_NM}")
    S.append("REF 1")
    S.append(f"EPD {BEAM_W:.6f}")
    S.append(f"YAN {p['ang']:.6f}")
    S.append("! constant-index model glass")
    S.append("PRV"); S.append(f"PWL {WL_NM}"); S.append(f"'FOLDN' {n:.6f}"); S.append("END")
    S.append("SO 0 1e18")
    # S1 = decentered aperture STOP at the source; thickness to the first face.
    S.append(f"YDE S1 {p['ray_y']:.6f}")
    S.append("STO S1")
    S.append(f"THI S1 {stop_thi:.6f}")
    si = 1
    for k, r in enumerate(recs):
        si += 1
        rdy = "0" if r["radius"] is None else f"{-r['radius']:.6f}"  # sign per running frame; verify
        # thickness AFTER this surface, in reflected-space sign:
        refl_state = r["n_refl_before"] + (1 if r["reflective"] else 0)
        thi = r["thi_out"] * ((-1) ** refl_state)
        if r["reflective"]:
            glass = "TIRO 'FOLDN'"
        elif r["medium_after"] == "GLASS":
            glass = "'FOLDN'"
        else:
            glass = ""
        ade = -r["local_tilt"]   # CODE V ADE negated vs engine tilt (confirmed convention)
        S.append(f"! S{si}: {r['event']} @{r['face']}  incid={r['incidence']:.2f}  deflect={r['deflect']:.2f}")
        S.append(f"S {rdy} {thi:.6f} {glass}".rstrip())
        S.append(f"ADE S{si} {ade:.6f}")
        # TIRO already reflects + the negative THI carries reflected space; no REV needed.
    # image / detector
    S.append("! detector plane; CIR sets the sensor half-size")
    S.append("SI 0 0")
    S.append(f"CIR SI {SENSOR_HALF:.4f}")
    S.append("GO")
    return "\n".join(S) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_fold_qd" / "archive.json"))
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_fold_qd" / "codev_export"))
    ap.add_argument("--tir", type=int)
    ap.add_argument("--cell", type=int, nargs=2, metavar=("I0", "I1"))
    args = ap.parse_args()

    arc = json.loads(Path(args.archive).read_text())
    cells = arc["cells"]
    names = arc["meta"]["param_names"]

    if args.cell:
        sel = [c for c in cells if int(c["i0"]) == args.cell[0] and int(c["i1"]) == args.cell[1]]
    elif args.tir is not None:
        champs = champions_by_depth(cells)
        sel = [champs[args.tir]] if args.tir in champs else []
    else:
        champs = champions_by_depth(cells)
        sel = [champs[3]] if 3 in champs else []
    if not sel:
        raise SystemExit("no matching design")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for c in sel:
        sig = c.get("dom_signature") or ""
        if not sig:
            print(f"[skip] cell ({c['i0']},{c['i1']}) has no fold"); continue
        p = dict(zip(names, [float(v) for v in c["genotype"]]))
        recs, sensor, n_refl = build_folded(p, sig)
        cid = f"fold_cell_{int(c['i0'])}_{int(c['i1'])}"
        (outdir / f"{cid}.folded.spec.txt").write_text(emit_spec_txt(c, p, sig, recs, sensor, n_refl))
        (outdir / f"{cid}.folded.seq").write_text(emit_seq(c, p, sig, recs, sensor, n_refl))
        print(f"[folded] {cid}  TIR={n_refl}  surfaces={len(recs)}  '{sig}'")
    print(f"[done] -> {outdir}/  (*.folded.seq + *.folded.spec.txt)")


if __name__ == "__main__":
    main()
