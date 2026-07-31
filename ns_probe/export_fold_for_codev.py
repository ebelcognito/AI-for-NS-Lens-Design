"""
Export collinear-FOLD designs (fold_demo / run_fold_qd family) to CODE V.

WHAT THE DESIGN IS (source of truth: fold_demo.py + sequence_tools.py)
---------------------------------------------------------------------
A 2-D meridional, NON-SEQUENTIAL model in the ray-optics JS engine:

  * ONE glass element, 4 faces:  A = left optical face  (curved arc, tilt_a),
                                 B = right optical face (curved arc, tilt_b),
                                 T = bottom edge  (flat, y = -ELEM_H),
                                 D = top edge     (flat, y = +ELEM_H).
    (T/D naming follows sequence_tools.make_single_element_face_classifier:
     y<=-H -> 'T', y>=+H -> 'D'.)
  * refractive index n, CONSTANT (monochromatic; the search never used dispersion).
  * input = a COLLIMATED beam (21 rays, width beam_w) entering at x=0, centred on
    y=ray_y, travelling at angle `ang` deg above +x.
  * the beam TIR-folds inside the element and exits to a detector plane at x=X_SENSOR.
  * a design's frozen ray path is its `dom_signature`, e.g.
       "T@A -> TIR@D -> TIR@B -> TIR@B -> T@B"
    (T = transmit/refract, TIR = total internal reflection, @FACE).

HOW WE MAP IT TO CODE V
-----------------------
Coordinate mapping (2-D meridional -> CODE V global Y-Z plane):
      CODE V  Z  =  engine  x        (nominal optical axis / propagation)
      CODE V  Y  =  engine  y        (meridional transverse)
      CODE V  X  =  out of plane     (the design is 2-D; faces are CYLINDERS along X)

Each curved face becomes a CYLINDRICAL surface (RDY = arc radius, flat in X): the
design was optimised purely in the meridional plane, so a sphere would be WRONG --
a cylinder reproduces the 2-D physics exactly.

The non-sequential TIR path is unrolled into a SEQUENTIAL surface list in GLOBAL
coordinates: every surface is placed by its absolute (YDE, ZDE) vertex + ADE tilt
(rotation about X == engine z), using decenter-and-return (DAR) with zero axial
thickness, so all surfaces live in one frame and the ray finds each at its true
location. TIR hits become REFL (reflective) surfaces that keep the ray in glass n;
refract hits toggle air<->glass. The same physical face reused N times along the
path is emitted N times with IDENTICAL vertex/tilt/radius.

Object = infinity (afocal collimated input), single field YAN = ang. The entrance
beam offset/width is reproduced by a decentred aperture STOP at engine x=0
(YDE = ray_y, semi-aperture = beam_w/2). Image = flat plane at engine x=X_SENSOR.

OUTPUTS (per design):  <id>.seq   CODE V command-language macro (best-effort)
                       <id>.spec.json   exact, unambiguous geometry table
                       <id>.spec.txt    human-readable surface table
Plus a combined summary.json.

NOTE (be honest): this generates a faithful GEOMETRY spec and a best-effort .seq.
It cannot be trace-verified without CODE V here. The most likely fixups after the
first CODE V trace are sign conventions (radius sign, ADE/YAN handedness); the
.spec.* files give the exact numbers so any such fix is a one-liner.

    python3 ns_probe/export_fold_for_codev.py --tir 3
    python3 ns_probe/export_fold_for_codev.py --all-champs
    python3 ns_probe/export_fold_for_codev.py --top 5
    python3 ns_probe/export_fold_for_codev.py --cell 1 10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ELEM_H = 30.0            # fold_demo.ELEM_H
X_SENSOR = 120.0         # fold_demo.X_SENSOR
BEAM_W = 8.0             # fold_demo default beam width
WL_NM = 587.56          # d-line; monochromatic (index is constant in the model)

PRETTY2TYPE = {"T": "refract", "TIR": "tir", "PR": "partial_reflect", "M": "mirror"}


# ----------------------------------------------------------------------------- geometry
def circumcircle(p1, p2, p3):
    """Center + radius of the circle through 3 points; (None, None) if ~collinear."""
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return [ux, uy], math.hypot(ax - ux, ay - uy)


def faces_of(p):
    """The four faces of the fold element as CODE V surface descriptors, keyed by face
    label. Mirrors fold_demo.element() exactly. Vertices/centres in engine (x,y)."""
    lx, rx = p["cx"] - p["thick"] / 2.0, p["cx"] + p["thick"] / 2.0
    dA = math.tan(math.radians(p["tilt_a"])) * ELEM_H
    dB = math.tan(math.radians(p["tilt_b"])) * ELEM_H
    # engine corners
    A_bot, A_top = (lx + dA, -ELEM_H), (lx - dA, ELEM_H)       # left face, bottom->top
    B_bot, B_top = (rx + dB, -ELEM_H), (rx - dB, ELEM_H)       # right face, bottom->top
    A_apex = (lx - p["fsag"], 0.0)                              # left arc apex (y=0)
    B_apex = (rx + p["bsag"], 0.0)                              # right arc apex (y=0)

    def curved(apex, c_bot, c_top, ade_deg, into_x_sign):
        """Cylindrical face. ade_deg = tilt of local +Z about engine-z (== the face tilt).
        into_x_sign = +1 if local +Z (propagation) points toward +engine-x else -1."""
        center, R = circumcircle(c_bot, apex, c_top)
        a = math.radians(ade_deg)
        zhat = (math.cos(a) * into_x_sign, math.sin(a) * into_x_sign)   # local +Z in engine xy
        if center is None:                                             # ~flat -> treat as plane
            return dict(kind="flat", vertex=list(apex), ade_deg=ade_deg,
                        zhat=list(zhat), radius=None, center=None)
        # CODE V radius sign: + if centre of curvature is on the +local-Z side of the vertex
        s = 1.0 if ((center[0] - apex[0]) * zhat[0] + (center[1] - apex[1]) * zhat[1]) > 0 else -1.0
        return dict(kind="cyl", vertex=list(apex), ade_deg=ade_deg, zhat=list(zhat),
                    radius=s * R, center=center)

    # A: local +Z into glass points +x  -> ade = tilt_a ; B exit propagation +x -> ade = tilt_b
    A = curved(A_apex, A_bot, A_top, p["tilt_a"], +1.0)
    B = curved(B_apex, B_bot, B_top, p["tilt_b"], +1.0)
    # flat edges: horizontal planes y=const; local +Z along +y  -> ade = 90 deg
    T = dict(kind="flat", vertex=[(A_bot[0] + B_bot[0]) / 2.0, -ELEM_H], ade_deg=90.0,
             zhat=[0.0, 1.0], radius=None, center=None)            # bottom edge  y=-H
    D = dict(kind="flat", vertex=[(A_top[0] + B_top[0]) / 2.0, +ELEM_H], ade_deg=90.0,
             zhat=[0.0, 1.0], radius=None, center=None)            # top edge     y=+H
    return {"A": A, "B": B, "T": T, "D": D}


def parse_signature(sig):
    """'T@A -> TIR@D -> TIR@B -> T@B' -> [('refract','A'), ('tir','D'), ...]."""
    out = []
    for tok in sig.replace("->", " ").split():
        t, face = tok.split("@")
        out.append((PRETTY2TYPE.get(t, t.lower()), face))
    return out


def build_surface_list(p, sig):
    """Ordered sequential surfaces (object..image) in engine/global coords + media."""
    faces = faces_of(p)
    events = parse_signature(sig)
    surfaces = []
    medium = "AIR"                                # medium the ray is currently in
    for i, (etype, face) in enumerate(events):
        f = faces[face]
        reflective = (etype == "tir")
        if etype in ("tir", "partial_reflect", "mirror"):
            med_after = medium                    # reflection: stay in same medium
        else:                                     # refract: toggle
            med_after = "GLASS" if medium == "AIR" else "AIR"
        surfaces.append(dict(
            role=("entry" if i == 0 else "exit" if i == len(events) - 1 else "fold"),
            face=face, event=etype, reflective=reflective,
            kind=f["kind"], vertex=f["vertex"], ade_deg=f["ade_deg"],
            radius=f["radius"], center=f["center"],
            medium_before=medium, medium_after=med_after))
        medium = med_after
    return surfaces


# ----------------------------------------------------------------------------- emit
def to_codev_coords(vertex):
    """engine (x,y) -> CODE V global (YDE, ZDE)."""
    return dict(YDE=round(vertex[1], 6), ZDE=round(vertex[0], 6))


def make_spec(cell, p, sig):
    surfaces = build_surface_list(p, sig)
    spec_surfaces = []
    for k, s in enumerate(surfaces, start=1):
        yz = to_codev_coords(s["vertex"])
        spec_surfaces.append(dict(
            i=k, role=s["role"], face=s["face"], event=s["event"],
            type=("CYLINDER" if s["kind"] == "cyl" else "PLANE"),
            reflective=s["reflective"],
            YDE_mm=yz["YDE"], ZDE_mm=yz["ZDE"], ADE_deg=round(s["ade_deg"], 6),
            RDY_mm=(round(s["radius"], 6) if s["radius"] is not None else "INFINITY (flat)"),
            medium_after=("REFL (stay in glass n)" if s["reflective"]
                          else "GLASS n" if s["medium_after"] == "GLASS" else "AIR")))
    return dict(
        id=f"fold_cell_{int(cell['i0'])}_{int(cell['i1'])}",
        descriptor=dict(tir_count=int(round(cell["tir_count"])),
                        thinness_T=cell.get("thinness_T"), spot=cell.get("spot"),
                        fitness=cell.get("fitness"), efficiency=cell.get("efficiency")),
        genotype={k: float(v) for k, v in p.items()},
        dom_signature=sig,
        coordinate_map="CODEV Z=engine_x (axis), Y=engine_y, X=out-of-plane; cylinders along X",
        units="mm (engine units ~= mm)", wavelength_nm=WL_NM,
        refractive_index=float(p["n"]),
        object="infinity (afocal collimated input)",
        field_YAN_deg=float(p["ang"]),
        stop=dict(at_engine_x=0.0, YDE_mm=round(p["ray_y"], 6),
                  semi_aperture_mm=BEAM_W / 2.0, note="reproduces the input beam offset+width"),
        image=dict(type="PLANE", at_engine_x=X_SENSOR, ZDE_mm=X_SENSOR,
                   note="detector plane; refocus in CODE V as desired"),
        surfaces=spec_surfaces)


def spec_to_text(spec):
    L = []
    L.append(f"# CODE V export spec : {spec['id']}")
    L.append(f"# dom_signature      : {spec['dom_signature']}")
    d = spec["descriptor"]
    L.append(f"# descriptor         : TIR={d['tir_count']}  spot={d['spot']:.3f}  "
             f"fitness={d['fitness']:.3f}  effic={d.get('efficiency', float('nan')):.3f}")
    L.append(f"# units={spec['units']}  wl={spec['wavelength_nm']} nm  n={spec['refractive_index']:.6f}")
    L.append(f"# {spec['coordinate_map']}")
    L.append("#")
    L.append(f"OBJECT : {spec['object']} ;  field YAN = {spec['field_YAN_deg']:.6f} deg")
    st = spec["stop"]
    L.append(f"STOP   : flat aperture at engine x={st['at_engine_x']}  YDE={st['YDE_mm']:.4f} mm  "
             f"semi-aperture={st['semi_aperture_mm']:.3f} mm")
    L.append("")
    hdr = f"{'i':>2} {'role':>5} {'fc':>2} {'event':>7} {'type':>9} {'refl':>4} " \
          f"{'YDE(mm)':>10} {'ZDE(mm)':>10} {'ADE(deg)':>10} {'RDY(mm)':>16} {'medium_after':>22}"
    L.append(hdr)
    L.append("-" * len(hdr))
    for s in spec["surfaces"]:
        rdy = s["RDY_mm"] if isinstance(s["RDY_mm"], str) else f"{s['RDY_mm']:.4f}"
        L.append(f"{s['i']:>2} {s['role']:>5} {s['face']:>2} {s['event']:>7} {s['type']:>9} "
                 f"{('Y' if s['reflective'] else '-'):>4} {s['YDE_mm']:>10.4f} {s['ZDE_mm']:>10.4f} "
                 f"{s['ADE_deg']:>10.4f} {rdy:>16} {s['medium_after']:>22}")
    im = spec["image"]
    L.append("")
    L.append(f"IMAGE  : PLANE at engine x={im['at_engine_x']}  ZDE={im['ZDE_mm']:.4f} mm  ({im['note']})")
    return "\n".join(L) + "\n"


def spec_to_seq(spec):
    """Best-effort CODE V command-language macro. Global coords via decenter-and-return
    (DAR); TIR faces via 'TIRO' (TIR-only, stays in glass); constant-index model glass
    via the private catalog. Confirmed valid tokens (from a live CODE V trace):
      SO <rdy> <thi>            object (infinity = 0 huge_thi)
      S  <rdy> <thi> [glass | REFL | TIRO [glass]]   append a surface (NO index!)
      EPD <d> ; YAN <ang> ; WL <nm> ; DIM M ; REF 1
      YDE Sn / ZDE Sn / ADE Sn / DAR Sn ; STO Sn ; SI <rdy> <thi>
    Curved faces are emitted as SPHERES (just RDY): in the meridional plane a sphere and
    a cylinder of the same radius trace IDENTICALLY, so this reproduces the 2-D design
    exactly. Convert the curved surfaces to Y-cylinders/toroids in the lens editor if you
    need the (unconstrained) sagittal to stay flat. Verify with a trace; the only likely
    fixups are sign conventions (RDY sign, ADE/YAN handedness) -- one-liners vs .spec.txt."""
    n = spec["refractive_index"]
    st = spec["stop"]
    im = spec["image"]
    S = []
    S.append(f"! CODE V export : {spec['id']}")
    S.append(f"! dom_signature : {spec['dom_signature']}")
    S.append(f"! {spec['coordinate_map']}")
    S.append("! Global coords: every surface DECENTER-AND-RETURN (DAR), THI 0 -> one frame.")
    S.append("! Curved faces = spheres (meridionally exact); TIR faces = TIRO (stay in glass n).")
    S.append("LEN NEW")
    S.append("DIM M")                                   # mm
    S.append(f"WL {WL_NM}")
    S.append("REF 1")
    S.append(f"EPD {BEAM_W:.6f}")                        # entrance pupil dia = input beam width
    S.append(f"YAN {spec['field_YAN_deg']:.6f}")        # single collimated field
    S.append("! ---- constant-index model glass (private catalog; worked in live trace)")
    S.append("PRV")
    S.append(f"PWL {WL_NM}")
    S.append(f"'FOLDN' {n:.6f}")
    S.append("END")
    S.append("! ---- object at infinity (radius 0, huge thickness)")
    S.append("SO 0 1e18")
    # S1 already exists (the LEN NEW dummy: flat, thi 0, air). Reuse it as the decentered
    # aperture STOP -- reproduces the input beam's offset (ray_y) and width (EPD).
    S.append("! ---- S1: decentered aperture STOP at engine x=0 (input beam offset+width)")
    S.append(f"YDE S1 {st['YDE_mm']:.6f}")
    S.append("DAR S1")
    S.append("STO S1")
    # optical surfaces append as S2, S3, ... (S command takes NO index)
    si = 1
    for s in spec["surfaces"]:
        si += 1
        rdy = "0" if isinstance(s["RDY_mm"], str) else f"{s['RDY_mm']:.6f}"
        if s["reflective"]:
            glass = "TIRO 'FOLDN'"                        # TIR-only, continue in glass n
        elif s["medium_after"] == "GLASS n":
            glass = "'FOLDN'"                            # refract into glass n
        else:
            glass = ""                                  # refract out to air
        S.append(f"! S{si}: {s['event']} @face {s['face']} ({s['role']})"
                 + ("  [TIR]" if s["reflective"] else ""))
        S.append(f"S {rdy} 0 {glass}".rstrip())          # append: radius, thickness 0, glass
        S.append(f"YDE S{si} {s['YDE_mm']:.6f}")
        S.append(f"ZDE S{si} {s['ZDE_mm']:.6f}")
        S.append(f"ADE S{si} {s['ADE_deg']:.6f}")
        S.append(f"DAR S{si}")
    # image plane at engine x = X_SENSOR (edit the pre-existing image surface SI)
    S.append(f"! image: plane at engine x={im['at_engine_x']}")
    S.append("SI 0 0")
    S.append(f"ZDE SI {im['ZDE_mm']:.6f}")
    S.append("DAR SI")
    S.append("GO")
    return "\n".join(S) + "\n"


# ----------------------------------------------------------------------------- selection
def champions_by_depth(cells):
    by = {}
    for c in cells:
        d = int(round(c["tir_count"]))
        if d not in by or c["fitness"] > by[d]["fitness"]:
            by[d] = c
    return dict(sorted(by.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_fold_qd" / "archive.json"))
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_fold_qd" / "codev_export"))
    ap.add_argument("--tir", type=int, help="export the champion at this TIR-bounce depth")
    ap.add_argument("--all-champs", action="store_true", help="export the champion at every depth")
    ap.add_argument("--top", type=int, help="export the top-N cells by fitness")
    ap.add_argument("--cell", type=int, nargs=2, metavar=("I0", "I1"), help="export one cell")
    args = ap.parse_args()

    arc = json.loads(Path(args.archive).read_text())
    cells = arc["cells"]
    names = arc["meta"]["param_names"]

    if args.cell:
        sel = [c for c in cells if int(c["i0"]) == args.cell[0] and int(c["i1"]) == args.cell[1]]
    elif args.all_champs:
        sel = list(champions_by_depth(cells).values())
    elif args.tir is not None:
        champs = champions_by_depth(cells)
        sel = [champs[args.tir]] if args.tir in champs else []
    elif args.top:
        sel = sorted(cells, key=lambda c: -c["fitness"])[:args.top]
    else:
        # default: the design the user pointed at -> champion at TIR depth 3
        champs = champions_by_depth(cells)
        sel = [champs[3]] if 3 in champs else sorted(cells, key=lambda c: -c["fitness"])[:1]

    if not sel:
        raise SystemExit("no matching designs in archive")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for c in sel:
        sig = c.get("dom_signature") or ""
        if not sig:
            print(f"[skip] cell ({c['i0']},{c['i1']}) has no dom_signature "
                  f"(tir_count={c['tir_count']}) -- straight-through, nothing to fold")
            continue
        p = dict(zip(names, [float(v) for v in c["genotype"]]))
        spec = make_spec(c, p, sig)
        (outdir / f"{spec['id']}.spec.json").write_text(json.dumps(spec, indent=2))
        (outdir / f"{spec['id']}.spec.txt").write_text(spec_to_text(spec))
        (outdir / f"{spec['id']}.seq").write_text(spec_to_seq(spec))
        summary.append(dict(id=spec["id"], cell=[int(c["i0"]), int(c["i1"])],
                            tir=int(round(c["tir_count"])), fitness=c["fitness"],
                            spot=c["spot"], signature=sig))
        print(f"[export] {spec['id']}  TIR={int(round(c['tir_count']))}  "
              f"n_surfaces={len(spec['surfaces'])}  '{sig}'")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {len(summary)} design(s) -> {outdir}/")
    print(f"       each: .seq (CODE V macro) + .spec.txt (exact table) + .spec.json")


if __name__ == "__main__":
    main()
