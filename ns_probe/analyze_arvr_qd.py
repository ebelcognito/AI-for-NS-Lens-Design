"""
Analyze the AR/VR prism QD archive: the FOV x form-factor heatmap, the fold-emergence
test (do TIR bounces appear in the COMPACT cells once thinness is rewarded?), and renders
of a compact-folded champion vs a wide-FOV champion.

    python3 ns_probe/analyze_arvr_qd.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import arvr_prism as AV


def render_champion(runner, cell, outdir, tag):
    x = np.array([cell["genotype"][AV.PRISM_NAMES.index(n)] if False else cell["genotype"][i]
                  for i, n in enumerate(AV.PRISM_NAMES)])
    rres, _ = runner.simulate(AV.scene_arvr(x, n_fan=7, brightness=0.6,
                                            record=False, crop=True))
    imgs = rres.get("images") or []
    if not (imgs and imgs[0].get("dataUrl")):
        return None
    out = outdir / f"arvr_{tag}.png"
    k = 0
    while out.exists():
        k += 1; out = outdir / f"arvr_{tag}_{k}.png"
    out.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "ns_probe" / "results_arvr_qd"))
    args = ap.parse_args()
    outdir = Path(args.dir)
    arc = json.loads((outdir / "archive.json").read_text())
    cells = arc["cells"]
    print(f"archive: {len(cells)} cells, {arc['meta']['evals']} evals, "
          f"best fitness {max(c['fitness'] for c in cells):.4f}")
    print(f"FOV range {min(c['fov'] for c in cells):.1f}-{max(c['fov'] for c in cells):.1f} deg, "
          f"depth range {min(c['depth'] for c in cells):.1f}-{max(c['depth'] for c in cells):.1f}")

    # --- fold-emergence test: TIR bounces vs prism depth (compactness) ---
    print("\nFOLD-EMERGENCE TEST  (does folding appear as the prism is forced compact?)")
    print(f"  {'depth band':>14} {'#cells':>6} {'med TIR':>8} {'max TIR':>8} {'med fit':>8} {'med FOV':>8}")
    bands = [(4, 20), (20, 35), (35, 50), (50, 88)]
    for lo, hi in bands:
        g = [c for c in cells if lo <= c["depth"] < hi]
        if not g:
            continue
        tir = [c.get("tir", 0) for c in g]
        print(f"  {f'[{lo},{hi})':>14} {len(g):>6} {np.median(tir):>8.1f} {max(tir):>8d} "
              f"{np.median([c['fitness'] for c in g]):>8.3f} {np.median([c['fov'] for c in g]):>8.1f}")

    folded = [c for c in cells if c.get("tir", 0) >= 1]
    print(f"\n  folded cells (>=1 TIR): {len(folded)}/{len(cells)}; "
          f"best folded fitness {max((c['fitness'] for c in folded), default=0):.4f}")

    # --- champions ---
    best = max(cells, key=lambda c: c["fitness"])
    widest = max(cells, key=lambda c: c["fov"])
    compact = min((c for c in cells if c["fitness"] > 0.3), key=lambda c: c["depth"], default=best)
    best_fold = max(folded, key=lambda c: c["fitness"]) if folded else None
    print(f"\nbest overall : fit={best['fitness']:.3f} FOV={best['fov']:.1f} depth={best['depth']:.1f} TIR={best.get('tir',0)}")
    print(f"widest FOV   : fit={widest['fitness']:.3f} FOV={widest['fov']:.1f} depth={widest['depth']:.1f} TIR={widest.get('tir',0)}")
    print(f"most compact : fit={compact['fitness']:.3f} FOV={compact['fov']:.1f} depth={compact['depth']:.1f} TIR={compact.get('tir',0)}")
    if best_fold:
        print(f"best FOLDED  : fit={best_fold['fitness']:.3f} FOV={best_fold['fov']:.1f} depth={best_fold['depth']:.1f} TIR={best_fold.get('tir',0)}")

    # --- heatmap + renders ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = arc["meta"]["bc_bins"]; (f0, f1), (d0, d1) = arc["meta"]["bc_bounds"]
        grid = np.full(bins, np.nan); tirg = np.full(bins, np.nan)
        for c in cells:
            grid[c["i0"], c["i1"]] = c["fitness"]; tirg[c["i0"], c["i1"]] = c.get("tir", 0)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        for a, (G, ttl, lab) in zip(ax, [(grid, "fitness (collimated throughput)", "fitness"),
                                         (tirg, "TIR bounces per cell", "# TIR")]):
            im = a.imshow(G.T, origin="lower", aspect="auto", cmap="viridis",
                          extent=[f0, f1, d0, d1])
            a.set(title=ttl, xlabel="FOV (deg)", ylabel="prism depth (compact=down)")
            fig.colorbar(im, ax=a, label=lab)
        fig.suptitle("AR/VR prism QD archive — FOV x form-factor", y=1.0)
        fig.tight_layout()
        fig.savefig(outdir / "arvr_qd_archive.png", dpi=120); plt.close(fig)
        print(f"\n[figure] {outdir/'arvr_qd_archive.png'}")
    except Exception as e:
        print(f"[figure] skipped ({e})")

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for cell, tag in [(best, "best"), (widest, "widest_fov"), (compact, "compact"),
                          (best_fold, "best_folded")]:
            if cell is None:
                continue
            p = render_champion(runner, cell, outdir, tag)
            if p:
                print(f"[render] {tag}: {p.name}  (FOV={cell['fov']:.1f} depth={cell['depth']:.1f} "
                      f"TIR={cell.get('tir',0)} fit={cell['fitness']:.3f})")


if __name__ == "__main__":
    main()
