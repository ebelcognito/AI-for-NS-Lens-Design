"""Freeform-vs-arc fold archive comparison: elites maps (TIR-depth x thinness) colored by the
achievable spot, side by side, + a per-niche delta. Same 30k-eval budget, same tracer, the only
difference being whether the 4 aspheric DOF are unlocked."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BC_BINS = (9, 16)
BC_BOUNDS = ((0.0, 9.0), (10.0, 120.0))


def grid(cells, key):
    g = np.full(BC_BINS, np.nan)
    for c in cells:
        g[int(c["i0"]), int(c["i1"])] = c[key]
    return g


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    base = Path("ns_autodiff/results_search_freeform")
    ff = json.loads((base / "long_default" / "archive_freeform.json").read_text())["cells"]
    arc = json.loads((base / "long_arc" / "archive_arc.json").read_text())["cells"]
    g_ff, g_arc = grid(ff, "spot"), grid(arc, "spot")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    extent = [BC_BOUNDS[1][0], BC_BOUNDS[1][1], BC_BOUNDS[0][0], BC_BOUNDS[0][1]]
    norm = LogNorm(vmin=0.05, vmax=10)
    for a, g, title, ncell in [(ax[0], g_ff, "FREEFORM", len(ff)), (ax[1], g_arc, "ARC", len(arc))]:
        im = a.imshow(g, origin="lower", aspect="auto", extent=extent, cmap="viridis_r", norm=norm)
        a.set(title=f"{title} archive  ({ncell} niches)", xlabel="thinness T (compact ->)",
              ylabel="fold depth (TIR bounces)")
        fig.colorbar(im, ax=a, label="spot (mm, lower=sharper)")
    # delta: log2(arc/ff) -> positive (blue) = freeform sharper
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.log2(g_arc / g_ff)
    im = ax[2].imshow(d, origin="lower", aspect="auto", extent=extent, cmap="RdBu", vmin=-3, vmax=3)
    ax[2].set(title="log2(arc spot / freeform spot)\nblue = freeform sharper, per shared niche",
              xlabel="thinness T", ylabel="fold depth (TIR)")
    fig.colorbar(im, ax=ax[2], label="<- arc better | freeform better ->")
    fig.suptitle("Fold QD: freeform vs circular-arc faces (same 30k-eval budget, same tracer)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = base / "fold_compare.png"; png, k = out, 0
    while png.exists():
        k += 1; png = out.with_name(f"fold_compare_{k}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[fold-compare] {png}")


if __name__ == "__main__":
    main()
