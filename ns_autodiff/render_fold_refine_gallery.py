"""Figure: general freeform FOLD refinement — 6 geometrically-diverse designs, before/after.

Picks the tightest well-imaging fold at each of several TIR depths (distinct shapes), freezes its
discovered topology and refines (refine_discovered.refine_one), and renders each DISCOVERED|REFINED
pair in the OLD ray-optics theme (black ground, glowing gold rays, translucent grey glass, dashed
sensor). The point: the search+refine pipeline yields a DIVERSITY of sharp freeform folds.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_fold_refine_gallery            # -> fold_refine_gallery_LATEST.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .refine_discovered import refine_one
from .search_freeform import BEAM_W, X_SENSOR, _sag, build_element, trace_discover

REPODIR = Path(__file__).resolve().parents[1]
ARCHIVE = REPODIR / "ns_autodiff" / "results_search_freeform" / "archive_freeform.json"
OUTDIR = REPODIR / "ns_autodiff" / "results_search_freeform"

BG = "black"; GLASS = "#9a9a9a"; RAY = "#ece24a"; SENSOR = "#ff5d5d"; EDGE = "#cccccc"


def surface_curve(s, n=80):
    h = np.linspace(s["h_min"], s["h_max"], n)
    z, _ = _sag(h, s["c"], s["k"], s["coeffs"])
    return s["v"][None, :] + z[:, None] * s["axis"][None, :] + h[:, None] * s["tang"][None, :]


def glass_polygon(surfaces):
    parts = [surface_curve(surfaces["A"]), surface_curve(surfaces["B"])]
    for f in ("T", "D"):
        s = surfaces[f]; parts.append(np.array([s["a"], s["b"]]))
    pts = np.vstack(parts); cen = pts.mean(0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))]


def draw(ax, x, title, n_rays=15):
    ax.set_facecolor(BG)
    surfaces, cx, _ = build_element(x)
    poly = glass_polygon(surfaces)
    ax.fill(poly[:, 0], poly[:, 1], color=GLASS, alpha=0.26, zorder=1, lw=0)
    for f in ("A", "B"):
        c = surface_curve(surfaces[f]); ax.plot(c[:, 0], c[:, 1], color=EDGE, lw=1.2, alpha=0.6, zorder=2)
    for f in ("T", "D"):
        s = surfaces[f]; ax.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                                 color=EDGE, lw=1.2, alpha=0.6, zorder=2)
    rays = trace_discover(x, n_rays=n_rays, beam_w=BEAM_W)
    dys = np.linspace(-BEAM_W / 2, BEAM_W / 2, n_rays); ray_y = x[0]; allpts = []
    for i, r in enumerate(rays):
        if not r["forward"]:
            continue
        pts = np.array([[0.0, ray_y + dys[i]]] + [[vx, vy] for (_t, _f, vx, vy, _b) in r["seq"]]
                       + [[X_SENSOR, r["y"]]])
        ax.plot(pts[:, 0], pts[:, 1], color=RAY, lw=2.0, alpha=0.05, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], color=RAY, lw=0.55, alpha=0.55, zorder=4)
        allpts.append(pts)
    ax.plot([X_SENSOR, X_SENSOR], [-60, 60], color=SENSOR, ls=(0, (6, 5)), lw=1.1, zorder=5)
    if allpts:
        Y = np.concatenate([p[:, 1] for p in allpts])
        ylo = max(min(np.percentile(Y, 12) - 8, -34), -62); yhi = min(max(np.percentile(Y, 88) + 8, 34), 62)
        ax.set_ylim(ylo, yhi)
    ax.set_xlim(-3, X_SENSOR + 5); ax.set_aspect("auto"); ax.axis("off")
    ax.set_title(title, color="#f0f0f0", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 4, 7])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--suffix", default="_LATEST")
    ap.add_argument("--archive", default=str(ARCHIVE))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    good = [c for c in arch["cells"]
            if c.get("efficiency", 0) >= 0.6 and c.get("n_forward", 0) >= 12 and c["spot"] < 10.0]
    picks = []
    for d in args.depths:
        cand = sorted([c for c in good if int(c["tir_count"]) == d], key=lambda c: c["spot"])
        for c in cand:                                   # take the best that actually freezes/refines
            x = np.array(c["genotype"], float)
            rr = refine_one(x, iters=args.iters)
            if rr is not None:
                rr["tir"] = d; rr["x_disc"] = x; picks.append(rr)
                print(f"[fold] {d}-TIR  spot {rr['start_spot']:.3f} -> {rr['refined_spot']:.3f}", flush=True)
                break
    if not picks:
        print("nothing refined"); return

    nrow = len(picks)
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 2.9 * nrow)); fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    for i, rr in enumerate(picks):
        g = 100 * (1 - rr["refined_spot"] / rr["start_spot"]) if rr["start_spot"] > 0 else 0
        draw(axes[i, 0], rr["x_disc"], f"{rr['tir']}-TIR  DISCOVERED   spot {rr['start_spot']:.3f} mm")
        draw(axes[i, 1], rr["x13"], f"REFINED   spot {rr['refined_spot']:.3f} mm   ({g:+.0f}% tighter)")
    fig.suptitle("General freeform FOLD — search-discovered, then freeze→refine  (geometric diversity, before→after)\n"
                 "gold = imaging rays   dashed = sensor plane", color="#f5f5f5", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = OUTDIR / f"fold_refine_gallery{args.suffix}.png"
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(
        [dict(tir=r["tir"], start_spot=r["start_spot"], refined_spot=r["refined_spot"],
              genotype=[float(v) for v in r["x13"]]) for r in picks], indent=1))
    print(f"[fold-refine-gallery] {out}  ({nrow} designs)")


if __name__ == "__main__":
    main()
