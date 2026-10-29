"""Render every wide-field-acceptance design from a fold24_wide archive.

One row per design, three panels across the field (-8 / 0 / +8 deg), so the acceptance is
something you SEE -- if the bundle still folds and still lands in the same focus at the
edge fields, the design is genuinely wide-field rather than merely scoring well on axis.

Sorted by fold count, then by worst-field throughput.

    PYTHONPATH=.:ns_probe python ns_probe/render_wide_folds.py --min-accept 6
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path

import numpy as np

from nslens.engine import Runner
from nslens.qd import experiments
from nslens.scene import CropBox

REPO = Path(__file__).resolve().parents[1]

HQ_RAYS = 61            # denser than the 17 used for scoring: this is for the eye
HQ_W = 1500
HQ_DEPTH = 24
PAD = 8.0


def _layout(problem, p, field_deg, *, rays, crop=None, record=False):
    base = problem.layout(p, field_deg=field_deg)
    return type(base)(
        source=type(base.source)(**{**base.source.__dict__, "n_rays": rays}),
        element=base.element, detector=base.detector,
        max_depth=HQ_DEPTH, record=record, crop=crop)


def frame_for(runner, problem, p, fields) -> tuple:
    """A crop that contains every ray path at every field, so the three panels of one
    design share a frame and can be compared by eye."""
    ys, xs = [], [0.0, problem.x_sensor]
    for fd in fields:
        res, _ = runner.simulate(_layout(problem, p, fd, rays=9, record=True)
                                 .to_scene_dict())
        for r in res.get("raySequences", []) or []:
            for v in r["seq"]:
                xs.append(float(v["x"])); ys.append(float(v["y"]))
            ys.append(float(r["y"]))
    if not ys:
        ys = [-problem.half_h, problem.half_h]
    return ((min(xs) - PAD, min(ys) - PAD), (max(xs) + PAD, max(ys) + PAD))


def render(runner, problem, p, field_deg: float, frame) -> bytes | None:
    """One PNG of this design traced at one field angle."""
    (x0, y0), (x1, y1) = frame
    layout = _layout(problem, p, field_deg, rays=HQ_RAYS,
                     crop=CropBox(p1=(x0, y0), p4=(x1, y1), width=HQ_W))
    res, _ = runner.simulate(layout.to_scene_dict())
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        return None
    return base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs" / "fold24_wide"))
    ap.add_argument("--experiment", default="fold24_wide")
    ap.add_argument("--min-accept", type=float, default=6.0,
                    help="keep designs whose acceptance half-angle is at least this")
    ap.add_argument("--out", default=str(REPO / "runs" / "fold24_wide" / "gallery"))
    args = ap.parse_args()

    cfg = experiments.get(args.experiment)
    problem = cfg.problem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = []
    for f in sorted(glob.glob(str(Path(args.runs) / "seed*" / "archive.json"))):
        seed = Path(f).parent.name
        for c in json.loads(Path(f).read_text())["cells"]:
            c["_seed"] = seed
            cells.append(c)

    keep = [c for c in cells if c["accept_deg"] >= args.min_accept]
    keep.sort(key=lambda c: (c["tir_count"], -c["fitness"]))
    print(f"{len(keep)} of {len(cells)} designs have acceptance >= +-{args.min_accept:g} deg")

    panels = [-8.0, 0.0, 8.0]
    manifest = []
    with Runner(timeout_s=30) as runner:
        for i, c in enumerate(keep, 1):
            p = problem.vec_to_params(c["genotype"])
            name = f"{int(c['tir_count'])}fold_{i:02d}_{c['_seed']}"
            files = {}
            frame = frame_for(runner, problem, p, panels)
            for fd in panels:
                png = render(runner, problem, p, fd, frame)
                if png is None:
                    print(f"  ! render failed {name} @ {fd:+.0f}")
                    continue
                fn = out / f"{name}_{fd:+.0f}deg.png".replace("+", "p").replace("-", "m")
                fn.write_bytes(png)
                files[f"{fd:+.0f}"] = fn.name
            rec = dict(name=name, folds=int(c["tir_count"]),
                       accept_deg=c["accept_deg"], worst=c["fitness"],
                       mean=c["mean_focthru"], per_field=c["per_field"],
                       signature=c["dom_signature"], crit_margin=c["crit_margin"],
                       thinness_T=c["thinness_T"], ns_frac=c["ns_frac"],
                       seed=c["_seed"], params={k: round(v, 4) for k, v in p.items()},
                       panels=files)
            manifest.append(rec)
            print(f"  {name}: worst={c['fitness']:.3f} accept=+-{c['accept_deg']:.0f}deg "
                  f"{c['dom_signature']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} designs x {len(panels)} panels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
