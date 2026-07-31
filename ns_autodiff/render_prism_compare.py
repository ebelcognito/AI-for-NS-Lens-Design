"""Prism freeform-vs-arc comparison: (FOV x form-factor) elites maps colored by collimation
fitness, side by side, + per-niche delta. Same 24k-eval budget, same branching tracer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BC_BINS = (20, 28)
BC_BOUNDS = ((0.0, 40.0), (4.0, 88.0))


def grid(cells, key):
    g = np.full(BC_BINS, np.nan)
    for c in cells:
        g[int(c["i0"]), int(c["i1"])] = c[key]
    return g


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path("ns_autodiff/results_search_prism_freeform")
    ff = json.loads((base / "archive_prism_freeform.json").read_text())["cells"]
    arc = json.loads((base / "archive_prism_arc.json").read_text())["cells"]
    g_ff, g_arc = grid(ff, "fitness"), grid(arc, "fitness")
    extent = [BC_BOUNDS[1][0], BC_BOUNDS[1][1], BC_BOUNDS[0][0], BC_BOUNDS[0][1]]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, g, title, nc in [(ax[0], g_ff, "FREEFORM", len(ff)), (ax[1], g_arc, "ARC", len(arc))]:
        im = a.imshow(g, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0, vmax=0.55)
        a.set(title=f"{title} archive  ({nc} niches)", xlabel="prism depth (compact ->)",
              ylabel="FOV (deg)")
        fig.colorbar(im, ax=a, label="collimation fitness (higher=better)")
    d = g_ff - g_arc
    im = ax[2].imshow(d, origin="lower", aspect="auto", extent=extent, cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax[2].set(title="freeform fitness - arc fitness (per shared niche)\nblue = freeform better",
              xlabel="prism depth", ylabel="FOV (deg)")
    fig.colorbar(im, ax=ax[2], label="<- arc better | freeform better ->")
    fig.suptitle("AR/VR prism QD: freeform vs circular-arc faces (same 24k-eval budget, branching tracer)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = base / "prism_compare.png"; png, k = out, 0
    while png.exists():
        k += 1; png = out.with_name(f"prism_compare_{k}.png")
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)

    # stats
    fc = {(c["i0"], c["i1"]): c for c in ff}; ac = {(c["i0"], c["i1"]): c for c in arc}
    both = set(fc) & set(ac)
    win = sum(1 for k in both if fc[k]["fitness"] > ac[k]["fitness"])
    print(f"[prism-compare] {png}")
    print(f"  freeform {len(ff)} niches (best {max(c['fitness'] for c in ff):.3f}) | "
          f"arc {len(arc)} niches (best {max(c['fitness'] for c in arc):.3f})")
    print(f"  shared {len(both)}: freeform fitness higher in {win}/{len(both)} "
          f"({100*win/len(both):.0f}%); median delta {np.median([fc[k]['fitness']-ac[k]['fitness'] for k in both]):+.3f}")


if __name__ == "__main__":
    main()
