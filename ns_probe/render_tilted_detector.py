"""Re-render ONE design with its detector on the plane the beam actually images onto.

The fold24_clean env scores every design against a vertical detector at x = 120, because
that plane has to be the same for every genome a search visits. A finished design has no
such constraint: 02_seed0_3fold_w0p826 leaves the glass at a chief angle of -30.0 deg
that moves by less than 1 deg over the whole +-8 deg field, so one plane perpendicular to
that ray catches every field, 21 mm out, with a worst-field RMS of 0.20 mm instead of the
2.73 mm the x = 120 plane sees. This script pins the detector there and re-renders.

Nothing here feeds the search. It is a post-hoc repositioning of the sensor for a design
that is already chosen, which is why the plane is solved from the design's own exit
bundle rather than being a parameter of the problem.

    PYTHONPATH=.:ns_probe python ns_probe/render_tilted_detector.py
    PYTHONPATH=.:ns_probe python ns_probe/render_tilted_detector.py --design 01_seed1_3fold_w0p838
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from nslens.engine import Runner
from nslens.qd import experiments
from nslens.scene import CropBox, Detector

from render_best_designs import contact_sheet

REPO = Path(__file__).resolve().parents[1]

HQ_RAYS = 61
HQ_W = 1100
HQ_DEPTH = 24
PAD = 8.0
GHOST_B = 0.1           # below this brightness a "ray" is a Fresnel ghost, not the bundle
DET_LEN = 24.0          # detector segment length, mm

#: The engine prints the detector's P / F-perp / F-par readout anchored at p2, at a fixed
#: font size in SCENE units. At x = 120 that anchor sat outside every crop so it was never
#: seen; a detector 21 mm off the glass puts it in the middle of the picture, three lines
#: tall. Zero alpha is the only way off -- the draw is unconditional. The segment itself is
#: recoloured green, both to be legible against the beam and because that is the colour
#: this plane was first sketched in.
THEME = {"detectorText": {"color": {"r": 0, "g": 0, "b": 0, "a": 0}},
         "detector": {"color": {"r": 0.2, "g": 0.9, "b": 0.3, "a": 1},
                      "dash": [1.2, 1.2], "width": 2}}


def _layout(problem, p, field_deg, *, rays, detector, crop=None, record=True):
    base = problem.layout(p, field_deg=field_deg)
    return type(base)(
        source=type(base.source)(**{**base.source.__dict__, "n_rays": rays}),
        element=base.element, detector=detector,
        max_depth=HQ_DEPTH, record=record, crop=crop)


def _scene(lay) -> dict:
    return {**lay.to_scene_dict(), "theme": THEME}


def exit_bundle(runner, problem, p, field_deg, detector, *, rays=17):
    """Last leg of every real ray: its launch point off the glass and its direction."""
    lay = _layout(problem, p, field_deg, rays=rays, detector=detector)
    res, _ = runner.simulate(_scene(lay))
    P, D, B = [], [], []
    for s in res.get("raySequences") or []:
        v = s["seq"]
        if len(v) < 2 or float(s.get("b", 0.0)) <= GHOST_B:
            continue
        a = np.array([v[-2]["x"], v[-2]["y"]], float)
        b = np.array([v[-1]["x"], v[-1]["y"]], float)
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        P.append(a); D.append(d / n); B.append(float(s["b"]))
    return np.array(P), np.array(D), np.array(B)


def solve_plane(runner, problem, p, fields, detector0):
    """The detector pose: perpendicular to the central field's exit chief ray, at the
    distance along it that minimises the WORST field's RMS spot.

    Worst-field rather than mean, to match how fold24_clean scores fitness -- a plane
    that is sharp on axis and soft at +-8 deg is not what this design is for.
    """
    data = {fd: exit_bundle(runner, problem, p, fd, detector0) for fd in fields}
    P0, D0, _ = data[0.0] if 0.0 in data else data[fields[len(fields) // 2]]
    origin = P0.mean(axis=0)
    axis = D0.mean(axis=0)
    axis /= np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0]])

    def spots(s):
        """Per-field RMS about each field's own centroid, in the plane's coordinate."""
        c = origin + s * axis
        out = []
        for fd in fields:
            Pf, Df, Bf = data[fd]
            t = ((c - Pf) @ axis) / (Df @ axis)
            u = ((Pf + t[:, None] * Df) - c) @ perp
            m = np.average(u, weights=Bf)
            out.append((float(np.sqrt(np.average((u - m) ** 2, weights=Bf))), float(m)))
        return c, out

    ss = np.arange(0.0, 60.0, 0.05)
    best_s = float(ss[int(np.argmin([max(r for r, _ in spots(s)[1]) for s in ss]))])
    centre, per_field = spots(best_s)
    return dict(origin=origin, axis=axis, perp=perp, s=best_s, centre=centre,
                rms=[r for r, _ in per_field], centroid=[m for _, m in per_field])


def flat_plane_rms(runner, problem, p, fields, flat):
    """Same bundles measured on the ORIGINAL vertical detector, so the tilted plane has
    something to be compared against rather than just asserted to be better."""
    out = []
    for fd in fields:
        Pf, Df, Bf = exit_bundle(runner, problem, p, fd, flat)
        t = (problem.x_sensor - Pf[:, 0]) / Df[:, 0]
        y = Pf[:, 1] + t * Df[:, 1]
        out.append(float(np.sqrt(np.average((y - np.average(y, weights=Bf)) ** 2,
                                            weights=Bf))))
    return out


def frame_for(runner, problem, p, fields, detector, end_x):
    """Crop containing every path AND the whole detector segment.

    Rays are not stopped by a detector -- they carry on to the engine's truncation
    distance, far off to the lower right -- so vertices past `end_x` are dropped. The
    detector's own endpoints then have to be added back explicitly, or the plane the
    picture exists to show gets cropped out of it.
    """
    xs = [0.0, end_x, detector.p1[0], detector.p2[0]]
    ys = [detector.p1[1], detector.p2[1]]
    for fd in fields:
        lay = _layout(problem, p, fd, rays=9, detector=detector)
        try:
            res, _ = runner.simulate(_scene(lay))
        except Exception:
            continue
        for r in res.get("raySequences") or []:
            if float(r.get("b", 0.0)) <= GHOST_B:
                continue
            for v in r["seq"]:
                if v["x"] <= end_x:
                    xs.append(float(v["x"])); ys.append(float(v["y"]))
    return ((min(xs) - PAD, min(ys) - PAD), (max(xs) + PAD, max(ys) + PAD))


def render(runner, problem, p, field_deg, detector, frame):
    (x0, y0), (x1, y1) = frame
    lay = _layout(problem, p, field_deg, rays=HQ_RAYS, detector=detector, record=False,
                  crop=CropBox(p1=(x0, y0), p4=(x1, y1), width=HQ_W))
    try:
        res, _ = runner.simulate(_scene(lay))
    except Exception:
        return None
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        return None
    return base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])


def genotype_of(entry, problem) -> np.ndarray:
    """The genome vector, however this manifest chose to record it.

    fold24_clean's gallery stores `genotype`; the pinned-index gallery stores only the
    decoded `params` dict, rounded to 4 dp. Decoding is the inverse of vec_to_params, so
    a `relative_sag` problem would need the sag genes divided back out by their chord --
    refuse rather than silently rebuild the wrong element.
    """
    if entry.get("genotype") is not None:
        return np.asarray(entry["genotype"], float)
    if getattr(problem, "relative_sag", False):
        raise SystemExit("manifest has no genotype and the problem uses relative_sag; "
                         "re-render the gallery with genotypes")
    return np.array([float(entry["params"][k]) for k in problem.param_names], float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(REPO / "runs" / "fold24_clean" / "gallery"))
    ap.add_argument("--design", default="02_seed0_3fold_w0p826")
    ap.add_argument("--experiment", default="fold24_clean")
    ap.add_argument("--half-fov", type=float, default=None,
                    help="override the experiment's fields with 9 spread over +-this. "
                         "The pinned-index arms were searched at their own half-FOV, so "
                         "solving the plane at fold24_wide's +-8 would optimise a plane "
                         "for fields the design was never scored on.")
    ap.add_argument("--out", default=None, help="default: <gallery>/<design>_tilted")
    ap.add_argument("--det-len", type=float, default=DET_LEN)
    args = ap.parse_args()

    gallery = Path(args.gallery)
    entry = next((d for d in json.loads((gallery / "manifest.json").read_text())
                  if d["name"] == args.design), None)
    if entry is None:
        print(f"no design named {args.design} in {gallery}/manifest.json")
        return 1

    cfg = experiments.get(args.experiment)
    problem = cfg.problem
    hf = args.half_fov if args.half_fov is not None else entry.get("half_fov")
    fields = ([round(float(v), 4) for v in np.linspace(-hf, hf, 9)] if hf
              else [float(f) for f in (cfg.fields or (0.0,))])
    p = problem.vec_to_params(genotype_of(entry, problem))
    out = Path(args.out) if args.out else gallery / f"{args.design}_tilted"
    out.mkdir(parents=True, exist_ok=True)

    with Runner(timeout_s=30) as runner:
        flat = Detector.vertical(problem.x_sensor)
        pl = solve_plane(runner, problem, p, fields, flat)
        c, perp, half = pl["centre"], pl["perp"], args.det_len / 2.0
        det = Detector(p1=tuple(c - half * perp), p2=tuple(c + half * perp))
        tilt = float(np.degrees(np.arctan2(perp[1], perp[0])))
        print(f"detector: ({det.p1[0]:.2f},{det.p1[1]:.2f}) -> ({det.p2[0]:.2f},{det.p2[1]:.2f})"
              f"  tilt {tilt:.2f} deg  centre ({c[0]:.2f},{c[1]:.2f})  s={pl['s']:.2f} mm")
        flat_rms = flat_plane_rms(runner, problem, p, fields, flat)
        print(f"worst-field RMS {max(pl['rms']):.3f} mm"
              f"   (vertical x={problem.x_sensor:.0f}: {max(flat_rms):.3f} mm, "
              f"{max(flat_rms) / max(pl['rms']):.1f}x worse)")

        end_x = max(det.p1[0], det.p2[0])
        frame = frame_for(runner, problem, p, fields, det, end_x)
        panels, files = [], {}
        for fd, rms, cen in zip(fields, pl["rms"], pl["centroid"]):
            png = render(runner, problem, p, fd, det, frame)
            if png is None:
                print(f"  ! {fd:+.0f} failed")
                continue
            fn = out / f"field_{fd:+05.1f}.png".replace("+", "p").replace("-", "m")
            fn.write_bytes(png)
            files[f"{fd:+.1f}"] = fn.name
            panels.append((f"{fd:+.0f} deg   rms {rms:.3f}  y {cen:+.2f}", png))
            print(f"  {fd:+5.1f} deg   rms {rms:.3f} mm   centroid {cen:+7.3f} mm")

        img_h = max(pl["centroid"]) - min(pl["centroid"])
        sub = (f"detector tilted {tilt:.1f} deg at ({c[0]:.2f}, {c[1]:.2f}), "
               f"{args.det_len:.0f} mm  ·  worst-field rms {max(pl['rms']):.3f} mm  ·  "
               f"image {img_h:.2f} mm over {fields[0]:+.0f}..{fields[-1]:+.0f} deg")
        sheet = contact_sheet(panels, f"{args.design}  [tilted detector]", sub)
        if sheet:
            (out / f"{args.design}_tilted__sheet.png").write_bytes(sheet)

        # a 0.2 mm spot on a 94 mm-wide picture is two pixels: the whole-system view
        # cannot show whether the plane is actually at the waist, so zoom on it
        zoom_h = 1.6 * max(abs(v) for v in pl["centroid"]) + 8.0
        zf = ((c[0] - zoom_h, c[1] - zoom_h), (c[0] + zoom_h, c[1] + zoom_h))
        zpanels = []
        for fd, rms in zip(fields, pl["rms"]):
            png = render(runner, problem, p, fd, det, zf)
            if png is None:
                continue
            fn = out / f"zoom_{fd:+05.1f}.png".replace("+", "p").replace("-", "m")
            fn.write_bytes(png)
            zpanels.append((f"{fd:+.0f} deg   rms {rms:.3f}", png))
        zsheet = contact_sheet(zpanels, f"{args.design}  [detector zoom, "
                                        f"{2 * zoom_h:.0f} mm across]", sub)
        if zsheet:
            (out / f"{args.design}_tilted__zoom_sheet.png").write_bytes(zsheet)

    doc = dict(entry)
    doc["detector"] = dict(
        kind="tilted", p1=[round(float(v), 4) for v in det.p1],
        p2=[round(float(v), 4) for v in det.p2],
        centre=[round(float(v), 4) for v in c], tilt_deg=round(tilt, 4),
        length_mm=args.det_len,
        normal_deg=round(float(np.degrees(np.arctan2(pl["axis"][1], pl["axis"][0]))), 4),
        s_from_exit_mm=round(pl["s"], 4),
        exit_point=[round(float(v), 4) for v in pl["origin"]],
        replaces=f"vertical x={problem.x_sensor}")
    doc["tilted_spot"] = dict(
        fields=fields, rms_mm=[round(v, 4) for v in pl["rms"]],
        centroid_mm=[round(v, 4) for v in pl["centroid"]],
        worst_rms_mm=round(max(pl["rms"]), 4), image_height_mm=round(img_h, 4),
        mm_per_deg=round(img_h / (fields[-1] - fields[0]), 4),
        worst_rms_mm_on_flat_plane=round(max(flat_rms), 4),
        flat_plane_rms_mm=[round(v, 4) for v in flat_rms])
    doc["fields"] = files
    (out / "design.json").write_text(json.dumps(doc, indent=2))

    scene = _layout(problem, p, 0.0, rays=problem.n_rays, detector=det).to_scene_dict()
    (out / "scene_field0.json").write_text(json.dumps(scene, indent=2))
    print(f"\nwrote {out}/design.json, scene_field0.json, {len(files)} fields + sheet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
