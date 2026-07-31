"""Flatten the non-optical bottom edge of the refined AR/VR prisms.

The bright bundle only uses three faces (refract@top -> tir@left -> refract@right); the bottom
edge is never struck, yet the QD genotype gives it a sag (dsag) that renders as a little
curved "bulb". Setting dsag=0 makes it a straight edge -- and the oracle fitness is unchanged
to 0.0000 for every design (verified), so it is a free manufacturability/clean-up win.

Writes straightened genotypes + curved-vs-straight renders under
autodiff_export/refined/flat_bottom/. Never overwrites.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.flatten_bottom
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import arvr_prism as AV                                                               # noqa: E402

from .render_arvr_compare import best_reports, render                                 # noqa: E402

NAMES = AV.PRISM_NAMES
OUT = REPO / "ns_probe" / "results_arvr_qd" / "autodiff_export" / "refined" / "flat_bottom"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    OUT.mkdir(parents=True, exist_ok=True)
    reps = best_reports()
    panels, saved = [], []
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for cell, rep in reps.items():
            g = dict(rep["genotype_refined"])
            gf = dict(g); gf["dsag"] = 0.0
            f0, _ = AV.evaluate(runner, np.array([g[n] for n in NAMES]))
            ff, _ = AV.evaluate(runner, np.array([gf[n] for n in NAMES]))
            png_c = render(runner, g)
            png_f = render(runner, gf)

            fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
            for a, png, ttl in ((ax[0], png_c, f"curved bottom (dsag={g['dsag']:+.1f})  fit={f0:.3f}"),
                                (ax[1], png_f, f"STRAIGHT bottom (dsag=0)  fit={ff:.3f}")):
                if png:
                    a.imshow(mpimg.imread(io.BytesIO(png)))
                a.set_title(ttl, fontsize=11); a.axis("off")
            fig.suptitle(f"AR/VR prism cell {cell} — bottom edge flattened (fitness {f0:.4f} -> "
                         f"{ff:.4f}, identical)", fontsize=12, y=1.02)
            fig.tight_layout()
            p = OUT / f"flatbottom_cell{cell[0]}_{cell[1]}.png"; k = 0
            while p.exists():
                k += 1; p = OUT / f"flatbottom_cell{cell[0]}_{cell[1]}_{k}.png"
            fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"[flat] cell {cell}: fit {f0:.4f} -> {ff:.4f}  -> {p.name}")
            panels.append((cell, png_f, ff))
            saved.append(dict(cell=list(cell), genotype=gf, oracle_fitness=ff))

    # gallery of the straight-bottom designs
    n = len(panels); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)
    for i, (cell, png, ff) in enumerate(panels):
        a = axes[i // cols, i % cols]
        if png:
            a.imshow(mpimg.imread(io.BytesIO(png)))
        a.set_title(f"cell {cell}  fit={ff:.3f}", fontsize=9); a.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("AR/VR prisms — bottom edge straightened (optically identical)", fontsize=13)
    fig.tight_layout()
    g = OUT / "arvr_flatbottom_gallery.png"; k = 0
    while g.exists():
        k += 1; g = OUT / f"arvr_flatbottom_gallery_{k}.png"
    fig.savefig(g, dpi=120, bbox_inches="tight"); plt.close(fig)
    (OUT / "straightened_genotypes.json").write_text(json.dumps(saved, indent=2, default=float))
    print(f"[gallery] {g}")
    print(f"[genotypes] {OUT / 'straightened_genotypes.json'}")


if __name__ == "__main__":
    main()
