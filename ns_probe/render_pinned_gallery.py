"""Render the good folds from the pinned-index runs, three fields per design.

Selection is "at least 2 folds AND worth looking at" -- a 0-fold design is not a fold and
a 0.1 design is not a design. Each row is one design at -half_fov / 0 / +half_fov, sharing
one frame, so field acceptance is visible rather than tabulated.

Every design is re-scored under clean_pack at all fields before rendering and the verdict
goes in the FILENAME, because the width-only gate these runs used admits caustics: the
highest-throughput design in the ladder puts a line focus inside the glass. A gallery that
did not say so would be advertising the worst design as the best one.

    PYTHONPATH=. python ns_probe/render_pinned_gallery.py
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
from nslens.qd.driver import _evaluate
from nslens.scene import CropBox

REPO = Path(__file__).resolve().parents[1]

HQ_RAYS = 61
HQ_W = 1500
HQ_DEPTH = 24
PAD = 8.0


def _layout(problem, p, field_deg, *, rays, crop=None, record=False):
    base = problem.layout(p, field_deg=field_deg)
    return type(base)(
        source=type(base.source)(**{**base.source.__dict__, "n_rays": rays}),
        element=base.element, detector=base.detector,
        max_depth=HQ_DEPTH, record=record, crop=crop)


def frame_for(runner, problem, p, fields):
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


def render(runner, problem, p, field_deg, frame):
    (x0, y0), (x1, y1) = frame
    res, _ = runner.simulate(
        _layout(problem, p, field_deg, rays=HQ_RAYS,
                crop=CropBox(p1=(x0, y0), p4=(x1, y1), width=HQ_W)).to_scene_dict())
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        return None
    return base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[
        "runs/fold24_nd151", "runs/fold24_nd151_fov75",
        "runs/fold24_nd151_fov50", "runs/fold24_n150"])
    ap.add_argument("--min-folds", type=int, default=2)
    ap.add_argument("--min-fitness", type=float, default=0.35)
    ap.add_argument("--out", default=str(REPO / "runs" / "nd151_gallery"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    with Runner(timeout_s=30) as runner:
        for arm in a.arms:
            cells, hf = [], None
            for f in sorted(glob.glob(str(REPO / arm / "seed*" / "archive.json"))):
                d = json.loads(Path(f).read_text())
                hf = d["meta"].get("half_fov_deg") or max(d["meta"]["fields"])
                for c in d["cells"]:
                    if c["tir_count"] >= a.min_folds and c["fitness"] >= a.min_fitness:
                        c["_seed"] = Path(f).parent.name
                        cells.append(c)
            if not cells:
                continue
            cells.sort(key=lambda c: -c["fitness"])

            # the clean_pack verdict, at the same fields the arm was scored at
            cfg = experiments.get("fold24_wide")
            cfg.fields = tuple(round(float(v), 4) for v in np.linspace(-hf, hf, 9))
            cfg.require_monotone_convergence = True
            cfg.convergence_gate = "clean_pack"
            cfg.gate_all_fields = True
            problem = cfg.problem

            arm_tag = Path(arm).name
            panels = [-hf, 0.0, hf]
            for i, c in enumerate(cells, 1):
                p = problem.vec_to_params(c["genotype"])
                ev = _evaluate(runner, problem, cfg,
                               np.asarray(c["genotype"], float))
                verdict = "CLEAN" if ev is not None else "caustic"
                name = (f"{arm_tag}_{i:02d}_{c['tir_count']}fold_"
                        f"fit{c['fitness']:.3f}_{verdict}")
                frame = frame_for(runner, problem, p, panels)
                files = {}
                for fd in panels:
                    png = render(runner, problem, p, fd, frame)
                    if png is None:
                        print(f"  ! render failed {name} @ {fd:+.1f}")
                        continue
                    fn = out / (f"{name}_{fd:+.1f}deg.png"
                                .replace("+", "p").replace("-", "m"))
                    fn.write_bytes(png)
                    files[f"{fd:+.1f}"] = fn.name
                manifest.append(dict(
                    name=name, arm=arm_tag, half_fov=hf, folds=int(c["tir_count"]),
                    worst_field=c["fitness"], clean_pack=(ev is not None),
                    accept_deg=c.get("accept_deg"), thinness_T=c.get("thinness_T"),
                    spot=c.get("spot"), efficiency=c.get("efficiency"),
                    dom_energy_share=c.get("dom_energy_share"),
                    signature=c.get("dom_signature"), per_field=c.get("per_field"),
                    seed=c["_seed"], params={k: round(v, 4) for k, v in p.items()},
                    panels=files))
                print(f"  {name}  T={c.get('thinness_T', 0):.1f} "
                      f"spot={c.get('spot', 0):.2f}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} designs x 3 panels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
