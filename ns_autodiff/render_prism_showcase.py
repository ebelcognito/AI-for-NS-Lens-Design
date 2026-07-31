"""Slide figures for the refined 1-fold AR/VR prisms:
  (A) 2x3 grid of the best-6 REFINED designs (after only), each annotated with the full design
      metrics: FOV, etendue, collimation sigma, throughput, fold count, prism depth, index.
  (B) the single best before->after refinement improvement, side by side.

Dark old-theme renders (dead faces auto-flattened). Recomputes metrics from each refined genotype
via the faithful branching tracer so the numbers are honest.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_prism_showcase
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from . import prism_branch as PB
from .render_prism_refine_compare import draw, BG
from .refine_prism_ff import fold_count

EYEBOX_MM = 2 * PB.EYEBOX_HALF       # 18 mm aperture


def design_info(x):
    """Honest metrics from the branching tracer + geometry. Returns a dict."""
    x = np.asarray(x, float)
    ex = PB.trace_branching_ff(x); m = PB.metrics(ex)
    xs = [v for r in ex if r["b"] > 0 for v in r["xs"]]
    depth = (max(xs) - min(xs)) if xs else 0.0
    Ts = [f["T"] for f in m["fields"] if f["present"]]
    fov = m["fov"]
    etendue = EYEBOX_MM * math.radians(fov)        # mm*rad  (eyebox aperture x angular FOV)
    return dict(fov=fov, sigma=m["sigma_mean"] or 0.0, T_min=m["T_min"],
                T_avg=float(np.mean(Ts)) if Ts else 0.0, fitness=m["fitness"],
                folds=fold_count(x), depth=depth, n=float(x[8]), etendue=etendue)


def info_title(cell, info, prefix=""):
    return (f"{prefix}cell {cell['i0']},{cell['i1']}   ({info['folds']} fold)\n"
            f"FOV {info['fov']:.1f}°   etendue {info['etendue']:.1f} mm·rad   (eyebox {EYEBOX_MM:.0f} mm)\n"
            f"$\\sigma$ {info['sigma']:.3f}°   throughput {100*info['T_avg']:.0f}% (min {100*info['T_min']:.0f}%)\n"
            f"depth {info['depth']:.0f} mm   n {info['n']:.2f}   fit {info['fitness']:.3f}")


def load_1fold(indir):
    reps = []
    for p in sorted(glob.glob(str(Path(indir) / "refine_cell*.json"))):
        r = json.loads(Path(p).read_text())
        if "genotype_refined" in r and r.get("folds", 9) == 1:
            reps.append(r)
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="ns_autodiff/results_refine_prism_ff")
    ap.add_argument("--suffix", default="_LATEST")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = load_1fold(args.indir)
    for r in reps:
        r["_info"] = design_info(r["genotype_refined"])
    reps.sort(key=lambda r: -r["_info"]["fitness"])
    best6 = reps[:6]

    # ---- Figure A: 2x3 grid of refined designs (after only) ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2)); fig.patch.set_facecolor(BG)
    for ax, r in zip(axes.ravel(), best6):
        draw(ax, r["genotype_refined"], info_title(r, r["_info"]))
    fig.suptitle("AR/VR 90° prism — best 6 refined single-fold relays (collimated near-eye display)\n"
                 "gold = display→fold→eyebox rays   red = display sources   dashed = eye pupil",
                 color="#f5f5f5", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outA = Path(args.indir) / f"prism_showcase_best6{args.suffix}.png"
    fig.savefig(outA, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    print(f"[showcase] {outA}")

    # ---- Figure B: best before->after improvement (by sigma %) ----
    best = max(reps, key=lambda r: r.get("sigma_improve_pct", 0))
    si = design_info(best["genotype_start"]); ri = best["_info"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6)); fig.patch.set_facecolor(BG)
    draw(axes[0], best["genotype_start"], info_title(best, si, prefix="BEFORE   "))
    draw(axes[1], best["genotype_refined"], info_title(best, ri, prefix="AFTER   "))
    fig.suptitle(f"AR/VR 90° prism — best refinement: cell {best['i0']},{best['i1']}   "
                 f"$\\sigma$ {si['sigma']:.3f}° → {ri['sigma']:.3f}°  "
                 f"({best['sigma_improve_pct']:+.0f}% sharper)   "
                 f"fit {si['fitness']:.3f} → {ri['fitness']:.3f}",
                 color="#f5f5f5", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outB = Path(args.indir) / f"prism_showcase_bestrefine{args.suffix}.png"
    fig.savefig(outB, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    print(f"[showcase] {outB}")

    print("\nbest 6 (by refined fitness):")
    for r in best6:
        i = r["_info"]
        print(f"  cell {r['i0']},{r['i1']}: FOV {i['fov']:.1f}  etendue {i['etendue']:.1f}  "
              f"sigma {i['sigma']:.3f}  T_avg {100*i['T_avg']:.0f}%  depth {i['depth']:.0f}  fit {i['fitness']:.3f}")


if __name__ == "__main__":
    main()
