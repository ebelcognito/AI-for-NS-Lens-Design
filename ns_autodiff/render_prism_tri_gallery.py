"""Diversity gallery for the 3-edge triangular prism: the best relay at EACH fold count, so the
panels show structurally-different relays (0-fold straight-through ... N-fold zigzag), annotated
with collimation fitness so the diversity-vs-quality tradeoff is visible."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .prism_branch import (build_triangle, SOURCE_PTS, SOURCE_AIM, FAN_HALFANGLE, PUPIL_X,
                           EYEBOX_HALF)
from .render_prism_gallery import face_curve, trace_draw


def draw_tri(ax, x, title):
    faces, n = build_triangle(x)
    for a in faces.values():
        cx, cy = face_curve(a); ax.plot(cx, cy, "k", lw=1.6, zorder=5)
    folds = 0
    for (sx, sy) in SOURCE_PTS:
        ax.plot(sx, sy, "o", color="tab:orange", ms=4, zorder=6)
        for k, ai in enumerate(np.radians(SOURCE_AIM + np.linspace(-FAN_HALFANGLE, FAN_HALFANGLE, 5))):
            pts, ntir = trace_draw(np.array([sx, sy]),
                                   np.array([math.cos(ai), math.sin(ai)]), faces, n)
            if k == 2:
                folds = ntir
            ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", lw=0.6, alpha=0.7, zorder=3)
    ax.axvline(PUPIL_X, color="tab:red", ls="--", lw=1.0, zorder=4)
    ax.plot([PUPIL_X, PUPIL_X], [-EYEBOX_HALF, EYEBOX_HALF], color="tab:red", lw=3, zorder=7)
    ax.set_xlim(-5, 128); ax.set_ylim(-55, 60); ax.set_aspect("auto"); ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="ns_autodiff/results_search_prism_tri/archive_tri.json")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--out", default="ns_autodiff/results_search_prism_tri")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.loads(Path(args.archive).read_text())["cells"]
    picks = []
    for f in args.folds:
        sub = [c for c in cells if int(c["folds"]) == f]
        if sub:
            picks.append(max(sub, key=lambda c: c["fitness"]))

    ncol = 3; nrow = int(math.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.7 * nrow)); axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, c in zip(axes, picks):
        draw_tri(ax, np.array(c["genotype"], float),
                 f"{int(c['folds'])}-FOLD relay   FOV {c['fov']:.1f}deg   collimation fit {c['fitness']:.3f}")
    fig.suptitle("3-edge triangular prism — best relay at each FOLD COUNT (structural diversity)\n"
                 "orange=display sources, red bar=eyebox, blue=relay rays",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out); base = out / "gallery_tri.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"gallery_tri_{k}.png")
    fig.savefig(png, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"[tri-gallery] {png}")
    for c in picks:
        print(f"  {int(c['folds'])}-fold  FOV {c['fov']:.1f}  fit {c['fitness']:.3f}")


if __name__ == "__main__":
    main()
