"""Dark-theme diversity gallery of a prism archive: the best design at each fold count.

Shows the structural spread the fold-count archive axis buys -- one panel per fold count (the best
collimating relay in that niche), in the old ray-optics theme. Works on any prism archive JSON
(multifold or freeform); fold count is recomputed from the genotype (chief-ray TIR bounces).

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_prism_archive_gallery --archive ns_autodiff/results_search_prism_multifold/archive_prism_multifold.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .render_prism_refine_compare import draw, BG
from .search_prism_multifold import fold_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--suffix", default="_LATEST")
    ap.add_argument("--per-fold", type=int, default=1, help="best-N designs per fold count")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    cells = arch["cells"]
    byfold = {}
    for c in cells:
        f = fold_count(np.asarray(c["genotype"], float))
        byfold.setdefault(f, []).append(c)
    picks = []
    for f in sorted(byfold):
        best = sorted(byfold[f], key=lambda c: -c["fitness"])[:args.per_fold]
        for c in best:
            picks.append((f, c))
    if not picks:
        print("no cells"); return

    ncol = args.per_fold
    nrow = len(picks) // ncol + (len(picks) % ncol > 0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.4 * nrow), squeeze=False)
    fig.patch.set_facecolor(BG)
    flat = axes.ravel()
    for ax in flat[len(picks):]:
        ax.axis("off"); ax.set_facecolor(BG)
    for ax, (f, c) in zip(flat, picks):
        sig = c.get("sigma_mean", 0) or 0
        draw(ax, np.asarray(c["genotype"], float),
             f"{f} fold   FOV {c['fov']:.1f}°   fit {c['fitness']:.3f}   $\\sigma$ {sig:.2f}°")
    exp = arch.get("meta", {}).get("experiment", "")
    fig.suptitle(f"AR/VR prism — fold-count diversity (best collimator per fold count)  [{exp}]\n"
                 "gold = display→fold→eyebox rays   red = sources   dashed = eye pupil",
                 color="#f5f5f5", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.archive).parent / f"prism_archive_diversity{args.suffix}.png"
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    print(f"[archive-gallery] {out}  ({len(picks)} panels, folds {sorted(byfold)})")


if __name__ == "__main__":
    main()
