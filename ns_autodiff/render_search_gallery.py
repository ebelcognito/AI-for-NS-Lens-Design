"""Gallery of the highest-quality freeform folds discovered by the search.

Picks, from the freeform QD archive, the tightest-spot well-imaging design at each of several
fold depths (TIR-bounce counts), and renders each as a ray diagram (freeform element + rays +
sensor focus). Shows that the search discovers a DIVERSITY of high-quality freeform folds.

Never overwrites: unique PNG + JSON sidecar.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_search_gallery
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .search_freeform import (BEAM_W, ELEM_H, X_SENSOR, _sag, build_element, evaluate,
                              trace_discover)


def surface_curve(s, n=80):
    h = np.linspace(s["h_min"], s["h_max"], n)
    z, _ = _sag(h, s["c"], s["k"], s["coeffs"])
    return s["v"][None, :] + z[:, None] * s["axis"][None, :] + h[:, None] * s["tang"][None, :]


def draw(ax, x, title):
    surfaces, cx, _ = build_element(x)
    for f in ("A", "B"):
        c = surface_curve(surfaces[f]); ax.plot(c[:, 0], c[:, 1], "k", lw=1.8, zorder=5)
    for f in ("T", "D"):
        s = surfaces[f]; ax.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]], "k", lw=1.8, zorder=5)
    n_rays = 15
    rays = trace_discover(x, n_rays=n_rays, beam_w=BEAM_W)
    dys = np.linspace(-BEAM_W / 2, BEAM_W / 2, n_rays)
    ray_y = x[0]
    ys_sensor = []; allpts = []
    for i, r in enumerate(rays):
        if not r["forward"]:
            continue
        pts = [[0.0, ray_y + dys[i]]] + [[vx, vy] for (_t, _f, vx, vy, _b) in r["seq"]] + [[X_SENSOR, r["y"]]]
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", lw=0.6, alpha=0.7, zorder=3)
        ys_sensor.append(r["y"]); allpts.append(pts)
    ax.axvline(X_SENSOR, color="tab:red", ls="--", lw=1.0, zorder=4)
    # clip the view to the main bundle so a stray ray doesn't flatten the diagram
    if allpts:
        Y = np.concatenate([p[:, 1] for p in allpts])
        ylo, yhi = np.percentile(Y, 15), np.percentile(Y, 85)
        ylo = max(min(ylo - 8, -34), -60); yhi = min(max(yhi + 8, 34), 60)   # always show element; clip strays
        ax.set_ylim(ylo, yhi)
    ax.set_xlim(-3, X_SENSOR + 5)
    ax.set_aspect("auto"); ax.grid(alpha=0.2); ax.set_title(title, fontsize=9)
    return ys_sensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="ns_autodiff/results_search_freeform/archive_freeform.json")
    ap.add_argument("--out", default="ns_autodiff/results_search_freeform")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 2, 3, 4, 5, 6])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    # quality filter: most energy delivered + most rays reaching + concentrated
    good = [c for c in arch["cells"]
            if c.get("efficiency", 0) >= 0.75 and c.get("n_forward", 0) >= 12
            and c["fitness"] >= 0.8 and c["spot"] < 5.0]
    picks = []
    for d in args.depths:
        cand = [c for c in good if int(c["tir_count"]) == d]
        if cand:
            picks.append(min(cand, key=lambda c: c["spot"]))
    if not picks:
        raise SystemExit("no high-quality designs matched the filter")

    ncol = 3; nrow = int(math.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, c in zip(axes, picks):
        x = np.array(c["genotype"], float)
        title = (f"{int(c['tir_count'])}-TIR fold   spot {c['spot']:.3f} mm   "
                 f"thru {c['fitness']:.2f}\n{c['dom_signature']}")
        draw(ax, x, title)
    fig.suptitle("High-quality freeform folds discovered by the QD search "
                 "(tightest spot at each fold depth)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.out)
    base = out / "gallery_freeform.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"gallery_freeform_{k}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    png.with_suffix(".json").write_text(json.dumps(
        dict(source=str(Path(args.archive).name), depths=args.depths,
             picks=[dict(tir=int(c["tir_count"]), spot=c["spot"], thru=c["fitness"],
                         signature=c["dom_signature"], genotype=c["genotype"]) for c in picks]),
        indent=2))
    print(f"[gallery] {png}")
    for c in picks:
        print(f"  {int(c['tir_count'])}-TIR  spot {c['spot']:.3f}  thru {c['fitness']:.2f}  {c['dom_signature']}")


if __name__ == "__main__":
    main()
