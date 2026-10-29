"""Emit a discovered fold design as a CODE V prescription with XY-POLYNOMIAL faces.

WHY THIS IS NOT THE FOLDED-AXIS EXPORTER WITH A DIFFERENT SURFACE TYPE
----------------------------------------------------------------------
The folded-axis model (export_fold_folded.py) puts each surface's vertex at the point the
chief ray hits it, and threads real thicknesses between those points. For a SPHERE that is
harmless: a sphere of radius R tangent at the hit point with the correct normal *is* the
same sphere, so the model and the physical face agree everywhere.

For a freeform it is fatal. An XY polynomial is defined about its vertex. Face A is hit
twice, ~37 mm apart in y, so a folded-axis model would expand one physical face about two
different origins -- and the `PIK` coupling that is the only thing keeping the two entries
the same piece of glass becomes meaningless. An optimiser would then be free to grow two
incompatible surfaces and call it a prism.

So going freeform FORCES the absolute-placement build: one global frame, every surface
placed in it, repeated faces sharing one vertex, one radius and one coefficient set.

THE FRAME
---------
Not the engine's frame. The origin is the source point (0, ray_y) and +Z is the input chief
ray, i.e. everything is rotated by -`ang`. Two things fall out for free: the stop sits ON
the axis instead of being decentered by ray_y, and YAN becomes the true field angle, so the
+-5 deg field set maps straight across. The engine's own frame would have put a 47 deg
tilt between the axis and the beam, which makes EPD mean the wrong width.

    engine (x, y)  --translate by -(0, ray_y), rotate by -ang-->  (Z, Y);  X = out of plane

SIGN CONVENTION
---------------
ADE = -beta, for a face whose outward local +Z lies at +beta from +Z toward +Y. This is
what codev_handoff/verify_v2.py and check_seq_v2.py encode (a right-handed rotation about
+X carries zhat=(0,0,1) -> (0,-sinA,cosA)), and it is the opposite of what the older
export_fold_for_codev.py emits -- see the note printed at the end of a run.

    PYTHONPATH=.:ns_probe:codev_handoff/target_fold_design \
        python3 ns_probe/export_freeform_codev.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from nslens.engine import Runner
from nslens.qd import experiments

from _folded_trace import trace
from export_fold_for_codev import faces_of

REPO = Path(__file__).resolve().parents[1]

WL_NM = 587.56
GHOST_B = 0.1
APERTURE_PAD = 1.0      # mm added to each measured footprint edge


# --------------------------------------------------------------------- XY polynomial
def xy_terms(max_order: int = 6):
    """CODE V's XY-polynomial index -> (m, n) for the x^m y^n term.

    Ordered by total degree, and within a degree by increasing power of y, with no piston
    term: C1=x, C2=y, C3=x^2, C4=xy, C5=y^2, C6=x^3, C7=x^2y, ...

    Cross-checked against the coefficient lists published in the freeform three-mirror
    patents (US20150363973 and relatives), which enumerate exactly the plane-symmetric
    subset A2, A3, A5, A7, A9, A10, A12, A14, A16, A18, A20, A21, A23, A25, A27 -- i.e.
    every index whose x power is even under this ordering, and no other.
    """
    out, i = {}, 0
    for deg in range(1, max_order + 1):
        for k in range(deg + 1):
            i += 1
            out[i] = (deg - k, k)
    return out


def term_label(m: int, n: int) -> str:
    parts = []
    if m:
        parts.append("x" if m == 1 else f"x^{m}")
    if n:
        parts.append("y" if n == 1 else f"y^{n}")
    return "*".join(parts) or "1"


def variable_sets(max_order: int = 6):
    """The coefficients to free, split into the two stages this design supports.

    `meridional` (m == 0, pure y powers) is everything our 2-D engine can independently
    reproduce, so a CODE V result using only these can be checked against our own tracer.
    `sagittal` (m even and > 0) is real out-of-plane freeform freedom that no 2-D model
    here can see or verify -- unlock it only once the meridional stage has converged.

    Excluded deliberately:
      * odd powers of x -- the system is symmetric about the yz plane, so they are
        unphysical here and would tilt the image out of plane;
      * C2 (y) -- a linear sag ramp is degenerate with the surface's own ADE;
      * C5 (y^2) -- degenerate with the base RDY. Vary the radius or the quadratic
        coefficient, never both, or they trade against each other and the Jacobian goes
        singular.

    C3 (x^2) is deliberately NOT excluded, and the distinction matters when the faces
    being replaced were TOROIDS. An XY polynomial's base is a rotationally symmetric
    conic, so it has no RDX: swapping a toroid for RDY + polynomial silently discards the
    independent out-of-plane radius. Only the SUM of the two quadratic terms is degenerate
    with the base curvature; their DIFFERENCE is the anamorphic term, and C3 is what
    carries it. Holding both C3 and C5 would hand the optimiser a strictly less capable
    surface than the toroid it replaced.
    """
    terms = xy_terms(max_order)
    degenerate = {2, 5}
    meridional = [i for i, (m, _) in terms.items() if m == 0 and i not in degenerate]
    sagittal = [i for i, (m, _) in terms.items()
                if m > 0 and m % 2 == 0 and i not in degenerate]
    return terms, meridional, sagittal


# --------------------------------------------------------------------- frames
class Frame:
    """engine (x, y) <-> CODE V (Z, Y), origin at the source, +Z along the input ray."""

    def __init__(self, source_xy, ang_deg):
        self.s = np.asarray(source_xy, float)
        self.a = math.radians(ang_deg)
        c, s = math.cos(-self.a), math.sin(-self.a)
        self.R = np.array([[c, -s], [s, c]])

    def point(self, p):
        q = self.R @ (np.asarray(p, float) - self.s)
        return float(q[0]), float(q[1])          # (Z, Y)

    def angle(self, deg):
        """An engine-frame direction angle -> the same direction in the CODE V frame."""
        return ((float(deg) - math.degrees(self.a)) + 180.0) % 360.0 - 180.0


def face_records(p, frame):
    """One record per PHYSICAL face: its vertex, its outward axis, its radius.

    The polynomial lives here, not on the surface instances -- that is the whole point of
    the absolute build.
    """
    out = {}
    for label, f in faces_of(p).items():
        beta = frame.angle(f["ade_deg"])
        Z, Y = frame.point(f["vertex"])
        rec = dict(face=label, kind=f["kind"], Z=Z, Y=Y,
                   beta_deg=beta, ADE=-beta,
                   RDY=(float(f["radius"]) if f["radius"] is not None else 0.0))
        if f["center"] is not None:
            cz, cy = frame.point(f["center"])
            rec["centre_of_curvature"] = (cz, cy)
        out[label] = rec
    return out


# --------------------------------------------------------------------- engine sampling
def _layout(problem, p, field_deg, *, rays):
    base = problem.layout(p, field_deg=field_deg)
    return type(base)(
        source=type(base.source)(**{**base.source.__dict__, "n_rays": rays}),
        element=base.element, detector=base.detector, max_depth=24, record=True)


def footprints(runner, problem, p, sig, fields, frame, faces, *, rays=31):
    """Where the real bundle lands on each surface instance, and how much of it follows
    the sequential path at all.

    CODE V will trace the surface list in order for EVERY ray. Our engine does not: a ray
    that misses a face, or TIRs where the chief ray refracts, simply does something else.
    Those rays are not modelled by the .seq -- they are the difference between the design's
    efficiency and its focused throughput. Reporting the fraction that DOES follow the
    signature is the honest statement of how much of this design the export represents.

    Measured against the EMITTED energy, not against the rays that happen to survive.
    Every ray leaves the source at b = 1, so summing b over the signature-followers and
    dividing by the ray count gives the fraction of the light the .seq represents -- which
    is the number comparable to the archive's dom_energy_share. Dividing by the survivors
    instead would score a design that loses a quarter of its bundle as a perfect 100%.

    Footprint is measured in each face's own local y, which is the coordinate the XY
    polynomial and the REY/ADY apertures are expressed in.
    """
    n_events = len(sig.split("->"))
    lo = [math.inf] * n_events
    hi = [-math.inf] * n_events
    kept = total = 0.0
    per_field = {}

    for fd in fields:
        res, _ = runner.simulate(_layout(problem, p, fd, rays=rays).to_scene_dict())
        f_kept = 0.0
        for s in res.get("raySequences") or []:
            b = float(s.get("b", 0.0))
            if b <= GHOST_B:
                continue
            v = s["seq"]
            # seq[0] is the launch point; one vertex per interaction after that, then the
            # detector hit. A ray that followed the signature has exactly n_events + 2.
            if len(v) != n_events + 2:
                continue
            f_kept += b
            for k, ev in enumerate(sig.replace("->", " ").split()):
                label = ev.split("@")[1]
                fr = faces[label]
                Z, Y = frame.point((v[k + 1]["x"], v[k + 1]["y"]))
                # local y = displacement from the vertex, perpendicular to the face axis
                bb = math.radians(fr["beta_deg"])
                u = -math.sin(bb) * (Z - fr["Z"]) + math.cos(bb) * (Y - fr["Y"])
                lo[k] = min(lo[k], u)
                hi[k] = max(hi[k], u)
        kept += f_kept
        total += rays
        per_field[f"{fd:+.2f}"] = round(f_kept / rays, 4)

    return dict(local_y_lo=lo, local_y_hi=hi,
                closure=round(kept / total, 4) if total else 0.0,
                closure_per_field=per_field, rays_per_field=rays)


# --------------------------------------------------------------------- build
def build(p, sig, detector, frame, fp):
    faces = face_records(p, frame)
    hits, _ = trace(p, sig)
    crit = math.degrees(math.asin(1.0 / p["n"]))

    surfaces, medium = [], "AIR"
    for k, h in enumerate(hits):
        reflective = h["event"] == "tir"
        med_after = medium if reflective else ("GLASS" if medium == "AIR" else "AIR")
        lo, hi = fp["local_y_lo"][k], fp["local_y_hi"][k]
        surfaces.append(dict(
            i=k + 2,                                  # S1 is the stop
            face=h["face"], event=h["event"], reflective=reflective,
            incidence=round(h["incidence"], 4),
            tir_margin=(round(h["incidence"] - crit, 4) if reflective else None),
            medium_before=medium, medium_after=med_after,
            local_y_lo=round(lo, 4), local_y_hi=round(hi, 4),
            REY=round((hi - lo) / 2.0 + APERTURE_PAD, 4),
            ADY=round((hi + lo) / 2.0, 4)))
        medium = med_after

    # the first surface carrying each physical face owns its values; later ones pick up
    owner = {}
    for s in surfaces:
        if s["face"] in owner:
            s["pickup_from"] = owner[s["face"]]
        else:
            owner[s["face"]] = s["i"]
            s["pickup_from"] = None

    cz, cy = frame.point(detector["centre"])
    img = dict(Z=round(cz, 6), Y=round(cy, 6),
               ADE=round(-frame.angle(detector["normal_deg"]), 6),
               half_len=detector["length_mm"] / 2.0)

    return faces, surfaces, img, crit


# --------------------------------------------------------------------- emit
def emit_seq(meta, faces, surfaces, img, terms, mer, sag) -> str:
    p, L = meta["params"], []
    A = L.append
    A(f"! CODE V freeform-prism export : {meta['design']}")
    A(f"! dom_signature : {meta['signature']}")
    A("! Frame: origin at the source, +Z along the input chief ray (engine frame rotated")
    A(f"!        by -{p['ang']:.6f} deg). CODE V Z=along-axis, Y=meridional, X=out of plane.")
    A("! Placement: absolute. Every surface DAR + THI 0, so one global frame and a")
    A("!            repeated physical face can share ONE vertex and ONE coefficient set.")
    A(f"! Sequential closure: {meta['closure']:.1%} of the traced bundle follows this")
    A("!            surface list; the rest is not represented by this file.")
    A("!")
    A("! UNVERIFIED IN CODE V: the SPS XYP block and the C<j> coefficient spelling below.")
    A("!            Everything else uses constructs confirmed by a live run. The term")
    A("!            ORDER is confirmed independently (see export_freeform_codev.xy_terms).")
    A("! EXPECT: THI is 0 on every surface, because absolute placement is what lets a")
    A("!            repeated face share one coefficient set. CODE V may therefore draw the")
    A("!            faces as separate sheets rather than one solid block -- that is a")
    A("!            rendering consequence of THI 0, not a placement error. Judge this file")
    A("!            on the global coordinate listing and whether rays reach SI, not on")
    A("!            whether View Lens shows a prism.")
    A("LEN NEW")
    A("DIM M")
    A(f"WL {WL_NM}")
    A("REF 1")
    A("! EPD is the bundle's cross-section PERPENDICULAR to its own direction. The engine")
    A(f"!   emits along a vertical segment of beam_w = {meta['beam_w']:.3f} mm, so that "
      "cross-section is")
    A(f"!   beam_w*cos(aim) and it varies with field: "
      + "  ".join(f"{k}:{v}" for k, v in meta["epd_by_field"].items()))
    A("!   CODE V has one EPD for all fields, so this matches the engine at the centre")
    A("!   field only. Worth fixing at the source if it ever matters.")
    A(f"EPD {meta['epd']:.6f}")
    A(f"YAN {' '.join(f'{v:.6f}' for v in meta['fields'])}")
    A("! ---- constant-index model glass. The engine carries ONE index and never splits a")
    A("!      ray by wavelength, so this is monochromatic by construction, not by choice.")
    A("PRV")
    A(f"PWL {WL_NM}")
    A(f"'FOLDN' {p['n']:.6f}")
    A("END")
    A("SO 0 1e18")
    A("! ---- S1: real stop, in air, ahead of all optics and ON axis (the rotated frame")
    A("!      is what buys that: in the engine frame this stop is decentered by ray_y).")
    A(f"CIR S1 {meta['epd'] / 2.0:.6f}")
    A("STO S1")
    A("THI S1 0")

    for s in surfaces:
        f = faces[s["face"]]
        i = s["i"]
        A("")
        A(f"! ---- S{i}: {s['event']} @ face {s['face']}"
          + (f"   incidence {s['incidence']:.3f} deg"
             + (f", {s['tir_margin']:+.3f} vs critical" if s["reflective"] else "")))
        if s["reflective"]:
            glass = "TIRO 'FOLDN'"
        elif s["medium_after"] == "GLASS":
            glass = "'FOLDN'"
        else:
            glass = ""
        rdy = "0" if f["kind"] != "cyl" else f"{f['RDY']:.6f}"
        A(f"S {rdy} 0 {glass}".rstrip())
        A(f"ZDE S{i} {f['Z']:.6f}")
        A(f"YDE S{i} {f['Y']:.6f}")
        A(f"ADE S{i} {f['ADE']:.6f}")
        A(f"DAR S{i}")
        A(f"REY S{i} {s['REY']:.4f}")
        A(f"REX S{i} {meta['epd'] / 2.0 + APERTURE_PAD:.4f}")
        A(f"ADY S{i} {s['ADY']:.4f}")
        if f["kind"] == "cyl":
            A(f"! face {s['face']} freeform. Zero coefficients == the arc this design was")
            A("! discovered with, so the starting trace is unchanged and every subsequent")
            A("! change is attributable to the optimiser.")
            A(f"SPS XYP S{i}")
            for j in sorted(mer + sag):
                m, n = terms[j]
                A(f"C{j} S{i} 0.0   ! {term_label(m, n)}")
        if s["pickup_from"]:
            src = s["pickup_from"]
            A(f"! same physical face as S{src} -- couple it, or the optimiser grows two.")
            for item in ("RDY", "YDE", "ZDE", "ADE"):
                A(f"PIK {item} S{i} S{src}")
            if f["kind"] == "cyl":
                for j in sorted(mer + sag):
                    A(f"PIK C{j} S{i} S{src}")

    A("")
    A("! ---- image: the plane solved by render_tilted_detector.py, NOT the search's")
    A("!      vertical detector. Tilt and position are part of the design.")
    A("SI 0 0")
    A(f"ZDE SI {img['Z']:.6f}")
    A(f"YDE SI {img['Y']:.6f}")
    A(f"ADE SI {img['ADE']:.6f}")
    A("DAR SI")
    A(f"CIR SI {img['half_len']:.4f}")
    A("GO")
    return "\n".join(L) + "\n"


def emit_opt(meta, faces, surfaces, terms, mer, sag) -> str:
    """The optimisation block, kept in its own file so the prescription can be loaded and
    inspected without running anything."""
    L, A = [], lambda s: L.append(s)
    A(f"! Optimisation setup for {meta['design']}.")
    A("! Load the prescription and confirm it TRACES first -- this file only declares")
    A("! variables. The error function and the AUT invocation are deliberately not")
    A("! written here: that syntax has not been confirmed on this build, and this project")
    A("! does not guess CODE V syntax twice.")
    A("!")
    A("! STAGE 1 -- meridional terms only. These are the ones our own 2-D tracer can")
    A("! independently reproduce, so a stage-1 result is checkable outside CODE V.")
    A("")
    owners = {s["face"]: s["i"] for s in surfaces if s["pickup_from"] is None}
    for label, i in owners.items():
        if faces[label]["kind"] != "cyl":
            continue
        A(f"! face {label} (owner S{i}; its repeats follow by pickup)")
        for j in mer:
            m, n = terms[j]
            A(f"VAR C{j} S{i}          ! {term_label(m, n)}")
    A("")
    A("! Hold the base radii. They and the quadratic coefficients describe the same")
    A("! degree of freedom; freeing both makes the Jacobian singular.")
    A("")
    A("! ---- constraints that are NOT optional --------------------------------------")
    A(f"! 1. TIR margin. n = {meta['params']['n']:.6f} -> critical {meta['critical']:.4f} deg.")
    for s in surfaces:
        if s["reflective"]:
            A(f"!    S{s['i']} (face {s['face']}) chief-ray incidence {s['incidence']:.3f} deg"
              f"  = {s['tir_margin']:+.3f} deg of margin")
    A("!    Those are CHIEF-RAY margins. The worst case over the whole bundle and all")
    A("!    fields is tighter still -- the archive records +2.353 deg -- so the binding")
    A("!    constraint is roughly 2.3 deg, not the 3.7 deg the chief ray suggests.")
    A("!    Constrain incidence > critical at every field and pupil point on each TIRO")
    A("!    surface. An optimiser handed free tilts will walk a face below critical to buy")
    A("!    spot size, and the result traces perfectly while being physically impossible.")
    A("! 2. No obscuration and no surface interference -- the faces must still bound one")
    A("!    solid, and the beam must not clip a face it is not interacting with.")
    A("! 3. Hold the overall geometry. The packaging is what the search was FOR.")
    A("")
    A("! ---- stage 2, only after stage 1 converges -----------------------------------")
    A("! Unlock the even-x terms below. They are genuine out-of-plane freeform freedom")
    A("! that NO model in this repo can see: the engine is 2-D meridional, so from here")
    A("! on CODE V is the only source of truth for the sagittal.")
    for label, i in owners.items():
        if faces[label]["kind"] != "cyl":
            continue
        for j in sag:
            m, n = terms[j]
            A(f"! VAR C{j} S{i}        ! {term_label(m, n)}")
    return "\n".join(L) + "\n"


def emit_spec(meta, faces, surfaces, img, terms, mer, sag) -> str:
    L, A = [], lambda s: L.append(s)
    A(f"# CODE V freeform export : {meta['design']}")
    A(f"# signature   : {meta['signature']}")
    A(f"# genotype    : " + "  ".join(f"{k}={v:.4f}" for k, v in meta["params"].items()))
    A(f"# frame       : origin at source (0, {meta['params']['ray_y']:.4f}), "
      f"+Z along the input ray (engine rotated by -{meta['params']['ang']:.4f} deg)")
    A(f"# index       : n={meta['params']['n']:.6f}  critical={meta['critical']:.4f} deg")
    A(f"# closure     : {meta['closure']:.1%} of the bundle follows this surface list")
    A("#               per field: " + "  ".join(
        f"{k}:{v:.0%}" for k, v in meta["closure_per_field"].items()))
    A("")
    A("PHYSICAL FACES  (one vertex, one radius, one coefficient set each)")
    hdr = (f"{'face':>4} {'kind':>5} {'Z(mm)':>11} {'Y(mm)':>11} {'ADE(deg)':>10} "
           f"{'RDY(mm)':>13} {'centre of curvature':>26}")
    A(hdr)
    A("-" * len(hdr))
    for label in ("A", "B", "T", "D"):
        f = faces[label]
        cc = f.get("centre_of_curvature")
        rdy = f"{f['RDY']:.4f}" if f["kind"] == "cyl" else "INF"
        cot = f"({cc[0]:.4f}, {cc[1]:.4f})" if cc else "-"
        A(f"{label:>4} {f['kind']:>5} {f['Z']:>11.4f} {f['Y']:>11.4f} {f['ADE']:>10.4f} "
          f"{rdy:>13} {cot:>26}")
    A("")
    A("SURFACE LIST  (the dominant-ray event order; a repeat picks up its owner)")
    hdr = (f"{'S':>3} {'face':>4} {'event':>8} {'incid':>8} {'margin':>8} {'medium_after':>13} "
           f"{'local y lo':>11} {'hi':>9} {'REY':>8} {'ADY':>9} {'pickup':>7}")
    A(hdr)
    A("-" * len(hdr))
    for s in surfaces:
        margin = f"{s['tir_margin']:+.3f}" if s["reflective"] else "-"
        pick = f"S{s['pickup_from']}" if s["pickup_from"] else "-"
        A(f"{s['i']:>3} {s['face']:>4} {s['event']:>8} {s['incidence']:>8.3f} "
          f"{margin:>8} {s['medium_after']:>13} {s['local_y_lo']:>11.3f} "
          f"{s['local_y_hi']:>9.3f} {s['REY']:>8.3f} {s['ADY']:>9.3f} {pick:>7}")
    A("")
    A(f"IMAGE  tilted plane   Z={img['Z']:.4f}  Y={img['Y']:.4f}  "
      f"ADE={img['ADE']:.4f}  half-length {img['half_len']:.2f} mm")
    A("")
    A("USED ZONES PER PHYSICAL FACE  (local y; a gap is unconstrained polynomial)")
    zones = {}
    for s in surfaces:
        zones.setdefault(s["face"], []).append((s["local_y_lo"], s["local_y_hi"], s["i"]))
    for label, zs in zones.items():
        zs.sort()
        span = f"{zs[0][0]:.2f} .. {zs[-1][1]:.2f}"
        bands = "  ".join(f"S{i}[{lo:.2f},{hi:.2f}]" for lo, hi, i in zs)
        A(f"  face {label}: {bands}    full span {span} mm")
        for (lo1, hi1, i1), (lo2, _, i2) in zip(zs, zs[1:]):
            if lo2 > hi1:
                A(f"    !! {lo2 - hi1:.2f} mm gap between S{i1} and S{i2}. The polynomial "
                  "is unconstrained")
                A("       there and a 6th-order fit will ring across it. Either constrain "
                  "sag/slope")
                A("       in the gap or keep the order down until the zones are joined.")
    A("")
    A("FREEFORM VARIABLES")
    A(f"  stage 1, meridional (m=0): " + ", ".join(
        f"C{j}({term_label(*terms[j])})" for j in mer))
    A(f"  stage 2, sagittal (m even>0): " + ", ".join(
        f"C{j}({term_label(*terms[j])})" for j in sag))
    A("  held: C2 (degenerate with ADE), C5 (degenerate with RDY), all odd-x terms")
    A("        (the system is symmetric about the yz plane).")
    A("  note: C3 (x^2) is a STAGE-2 variable, not a held one. An XY polynomial's base is")
    A("        rotationally symmetric, so replacing a toroid with RDY+polynomial drops the")
    A("        independent RDX. C3 is what carries that anamorphic freedom back.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(REPO / "runs" / "nd151_gallery"))
    ap.add_argument("--design", default="fold24_nd151_fov50_02_3fold_fit0.633_CLEAN")
    ap.add_argument("--tilted", default=None,
                    help="the *_tilted/design.json written by render_tilted_detector.py; "
                         "default <gallery>/fov50_02_tilted/design.json")
    ap.add_argument("--experiment", default="fold24_wide")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--rays", type=int, default=31)
    ap.add_argument("--out", default=str(REPO / "codev_handoff" / "freeform"))
    args = ap.parse_args()

    gallery = Path(args.gallery)
    entry = next(d for d in json.loads((gallery / "manifest.json").read_text())
                 if d["name"] == args.design)
    tilted = Path(args.tilted) if args.tilted else gallery / "fov50_02_tilted" / "design.json"
    det = json.loads(tilted.read_text())["detector"]

    problem = experiments.get(args.experiment).problem
    p = problem.vec_to_params(
        np.array([float(entry["params"][k]) for k in problem.param_names]))
    sig = entry["signature"]
    hf = float(entry["half_fov"])
    fields = [round(float(v), 4) for v in np.linspace(-hf, hf, 9)]

    frame = Frame((0.0, p["ray_y"]), p["ang"])
    faces = face_records(p, frame)

    with Runner(timeout_s=60) as runner:
        # the export is only meaningful if our analytic geometry still IS the engine's
        hits, _ = trace(p, sig)
        res, _ = runner.simulate(_layout(problem, p, 0.0, rays=17).to_scene_dict())
        cand = sorted((s for s in res["raySequences"]
                       if abs(s["seq"][0]["y"] - p["ray_y"]) < 1e-6
                       and float(s.get("b", 0)) > GHOST_B),
                      key=lambda s: -float(s["b"]))
        err = max(math.hypot(h["point"][0] - v["x"], h["point"][1] - v["y"])
                  for h, v in zip(hits, cand[0]["seq"][1:]))
        if err > 1e-3:
            print(f"ABORT: analytic trace disagrees with the engine by {err:.5f} mm")
            return 1
        fp = footprints(runner, problem, p, sig, fields, frame, faces, rays=args.rays)

    faces, surfaces, img, crit = build(p, sig, det, frame, fp)
    terms, mer, sag = variable_sets(args.order)

    # ParallelBundle spreads its rays along a VERTICAL segment of length beam_w and aims
    # them all at `ang`, so the bundle's cross-section perpendicular to its own direction
    # is beam_w*cos(aim) -- and `aim` is ang+field, so the effective aperture SHRINKS as
    # the field goes positive. CODE V's EPD is one number for all fields, so the export
    # can only match the engine at one field. It is set at the central field; the spread
    # over +-half_fov is reported so the mismatch is visible rather than silent.
    epd = problem.beam_w * math.cos(math.radians(p["ang"]))
    epd_by_field = {f"{fd:+.2f}": round(problem.beam_w
                                        * math.cos(math.radians(p["ang"] + fd)), 4)
                    for fd in fields}

    meta = dict(design=args.design, signature=sig, params=p, critical=crit,
                epd=epd, epd_by_field=epd_by_field, beam_w=problem.beam_w, fields=fields,
                closure=fp["closure"], closure_per_field=fp["closure_per_field"],
                bundle_margin="+2.353 deg over the bundle (archive crit_margin); the "
                              "chief-ray margins below are all looser than that")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "fov50_02_freeform"
    (out / f"{stem}.seq").write_text(emit_seq(meta, faces, surfaces, img, terms, mer, sag))
    (out / f"{stem}.opt.seq").write_text(emit_opt(meta, faces, surfaces, terms, mer, sag))
    (out / f"{stem}.spec.txt").write_text(
        emit_spec(meta, faces, surfaces, img, terms, mer, sag))
    (out / f"{stem}.spec.json").write_text(json.dumps(
        dict(meta=dict(meta, params={k: float(v) for k, v in p.items()}),
             faces=faces, surfaces=surfaces, image=img,
             variables=dict(meridional=mer, sagittal=sag,
                            terms={str(k): term_label(*v) for k, v in terms.items()})),
        indent=2, default=float))

    print(f"analytic vs engine chief ray: {err:.6f} mm  OK")
    print(f"sequential closure: {fp['closure']:.1%} of the bundle "
          f"({args.rays} rays x {len(fields)} fields)")
    for k, v in fp["closure_per_field"].items():
        print(f"   {k} deg : {v:.0%}")
    print(f"\nwrote {out}/{stem}.seq, .opt.seq, .spec.txt, .spec.json")
    print("\nNOTE export_fold_for_codev.spec_to_seq emits ADE with the opposite sign to "
          "the\n     convention verify_v2/check_seq_v2 encode (ADE = -beta). This file "
          "uses -beta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
