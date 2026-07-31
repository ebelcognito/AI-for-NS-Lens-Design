"""Before/after ray-diagram renders for the refined AR/VR prisms.

For each refined design, render the champion (genotype_start) and the autodiff-refined
(genotype_refined) prism through the JS engine (crop render, same as arvr_prism), stitch them
side-by-side with the oracle fitness / collimation / FOV deltas, and also build a contact-sheet
gallery of all designs. Reads the per-cell reports under autodiff_export/refined/ (best-by-
oracle-fitness report per cell). Never overwrites.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.render_arvr_compare
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402
import arvr_prism as AV                                                               # noqa: E402

REFINED = REPO / "ns_probe" / "results_arvr_qd" / "autodiff_export" / "refined"
PRISM_NAMES = AV.PRISM_NAMES


def best_reports():
    """Newest/best report per cell (max refined fitness), keyed by (i0,i1)."""
    by_cell = {}
    for f in REFINED.glob("refined_cell*.json"):
        if f.name == "refine_arvr_summary.json":
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if "cell" not in r or "genotype_refined" not in r:
            continue
        c = tuple(r["cell"])
        fitF = r.get("oracle_fitness_refined", -1)
        if c not in by_cell or fitF > by_cell[c].get("oracle_fitness_refined", -1):
            by_cell[c] = r
    return dict(sorted(by_cell.items(), key=lambda kv: -kv[1].get("oracle_fitness_refined", 0)))


def render(runner, geno_dict):
    x = np.array([geno_dict[n] for n in PRISM_NAMES])
    res, _ = runner.simulate(AV.scene_arvr(x, n_fan=9, brightness=0.6, record=False, crop=True))
    imgs = res.get("images") or []
    if imgs and imgs[0].get("dataUrl"):
        return base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    reports = best_reports()
    if not reports:
        raise SystemExit(f"no refined reports under {REFINED}")
    outdir = REFINED / "compare"
    outdir.mkdir(parents=True, exist_ok=True)

    panels = []  # (cell, after_png_bytes, title) for the gallery
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for cell, r in reports.items():
            png0 = render(runner, r["genotype_start"])
            pngF = render(runner, r["genotype_refined"])
            f0 = r.get("oracle_fitness_start", float("nan"))
            fF = r.get("oracle_fitness_refined", float("nan"))
            m0, mF = r.get("oracle_metrics_start"), r.get("oracle_metrics_refined")
            def sig(m):
                return (np.mean([f["sigma"] for f in m["fields"] if f["present"]])
                        if m else float("nan"))
            def fov(m):
                return m["fov"] if m else float("nan")
            s0, sF = sig(m0), sig(mF)
            v0, vF = fov(m0), fov(mF)

            # per-design before|after
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
            for a, png, ttl in ((ax[0], png0, f"BEFORE  fit={f0:.3f}  sigma={s0:.3f}d  FOV={v0:.1f}d"),
                                (ax[1], pngF, f"AFTER  fit={fF:.3f}  sigma={sF:.3f}d  FOV={vF:.1f}d")):
                if png:
                    a.imshow(mpimg.imread(io.BytesIO(png)))
                a.set_title(ttl, fontsize=11); a.axis("off")
            dpct = 100 * (fF - f0) / f0 if f0 else 0
            fig.suptitle(f"AR/VR prism cell {cell}   fitness {f0:.3f} -> {fF:.3f} "
                         f"({dpct:+.1f}%)   collimation {s0:.3f} -> {sF:.3f} deg",
                         fontsize=12, y=1.02)
            fig.tight_layout()
            base = outdir / f"compare_cell{cell[0]}_{cell[1]}.png"; p, k = base, 0
            while p.exists():
                k += 1; p = base.with_name(f"compare_cell{cell[0]}_{cell[1]}_{k}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"[compare] {p.name}  fit {f0:.3f}->{fF:.3f} ({dpct:+.1f}%)")
            panels.append((cell, pngF, f0, fF, s0, sF, dpct))

    # contact-sheet gallery of refined designs
    n = len(panels); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows))
    axes = np.atleast_2d(axes)
    for i, (cell, png, f0, fF, s0, sF, dpct) in enumerate(panels):
        a = axes[i // cols, i % cols]
        if png:
            a.imshow(mpimg.imread(io.BytesIO(png)))
        a.set_title(f"cell {cell}: fit {f0:.3f}->{fF:.3f} ({dpct:+.0f}%)\n"
                    f"collim {s0:.3f}->{sF:.3f}d", fontsize=9); a.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("AR/VR display-prism designs — autodiff-refined (after)", fontsize=13)
    fig.tight_layout()
    g = outdir / "arvr_refine_gallery.png"; k = 0
    while g.exists():
        k += 1; g = outdir / f"arvr_refine_gallery_{k}.png"
    fig.savefig(g, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[gallery] {g}")


if __name__ == "__main__":
    main()
