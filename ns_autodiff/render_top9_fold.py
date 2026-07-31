"""Top-9 gallery from the FOLD freeform search archive (3x3 ray diagrams)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .render_search_gallery import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="ns_autodiff/results_search_freeform/long_default/archive_freeform.json")
    ap.add_argument("--out", default="ns_autodiff/results_search_freeform")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arch = json.loads(Path(args.archive).read_text())
    good = [c for c in arch["cells"]
            if c.get("efficiency", 0) >= 0.75 and c.get("n_forward", 0) >= 14 and c["spot"] < 6.0]
    # rank by the search objective (focused throughput) then sharpness; cap 2 per fold depth
    good.sort(key=lambda c: (-c["fitness"], c["spot"]))
    picks, per_depth = [], {}
    for c in good:
        d = int(c["tir_count"])
        if per_depth.get(d, 0) >= 2:
            continue
        per_depth[d] = per_depth.get(d, 0) + 1
        picks.append(c)
        if len(picks) == 9:
            break
    picks.sort(key=lambda c: (int(c["tir_count"]), c["spot"]))

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    axes = axes.ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, c in zip(axes, picks):
        x = np.array(c["genotype"], float)
        draw(ax, x, f"{int(c['tir_count'])}-TIR fold  spot {c['spot']:.2f} mm  thru {c['fitness']:.2f}\n{c['dom_signature']}")
    fig.suptitle("FOLD freeform search — top 9 discovered designs (30k evals; ray diagrams)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out); base = out / "top9_fold.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"top9_fold_{k}.png")
    fig.savefig(png, dpi=115, bbox_inches="tight"); plt.close(fig)
    png.with_suffix(".json").write_text(json.dumps(
        [dict(tir=int(c["tir_count"]), spot=c["spot"], thru=c["fitness"],
              signature=c["dom_signature"], genotype=c["genotype"]) for c in picks], indent=2))
    print(f"[top9-fold] {png}")
    for c in picks:
        print(f"  {int(c['tir_count'])}-TIR  spot {c['spot']:.3f}  thru {c['fitness']:.2f}  {c['dom_signature']}")


if __name__ == "__main__":
    main()
