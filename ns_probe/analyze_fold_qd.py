"""
Analyze + visualize the fold-QD archive: does the QD search discover a DIVERSITY of
efficient, focused non-sequential folds (now that geometry allows prismatic tilt and the
engine measures typed interactions + energy honestly)?

Produces:
  * ns_probe/results_fold_qd/fold_qd_figure.png  (archive heatmap; champions vs fold depth)
  * ns_probe/results_fold_qd/renders/contact_folds.png  (best design at each TIR-bounce depth)
  * a printed table of champions per fold depth with their exact dominant interaction sequence
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import fold_demo as FD
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

ARCH = REPO / "ns_probe" / "results_fold_qd" / "archive.json"


def uniq(path: Path) -> Path:
    """Never overwrite: if `path` exists, return path_1, path_2, ... (standing user rule)."""
    if not path.exists():
        return path
    k = 1
    while path.with_name(f"{path.stem}_{k}{path.suffix}").exists():
        k += 1
    return path.with_name(f"{path.stem}_{k}{path.suffix}")


def champions_by_depth(cells):
    by = {}
    for c in cells:
        d = int(round(c["tir_count"]))
        if d not in by or c["fitness"] > by[d]["fitness"]:
            by[d] = c
    return dict(sorted(by.items()))


def figure(d, out_png):
    cells = d["cells"]
    bins = d["meta"]["bc_bins"]
    (lo0, hi0), (lo1, hi1) = d["meta"]["bc_bounds"]

    fig = plt.figure(figsize=(15, 4.6))

    # Panel A: heatmap fitness over (TIR count x thinness T)
    axA = fig.add_subplot(1, 3, 1)
    grid = np.full(bins, np.nan)
    for c in cells:
        grid[c["i0"], c["i1"]] = c["fitness"]
    im = axA.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                    extent=[lo1, hi1, lo0, hi0], vmin=0, vmax=max(0.3, np.nanmax(grid)))
    axA.set_xlabel("thinness  T = X_sensor - front optic   (thin = small)")
    axA.set_ylabel("dominant-path TIR bounces  (fold depth)")
    axA.set_title("Archive: focused throughput per cell")
    fig.colorbar(im, ax=axA, label="focused throughput (window E / input)")

    # Panel B: best focused-throughput & efficiency vs fold depth
    axB = fig.add_subplot(1, 3, 2)
    champs = champions_by_depth(cells)
    depths = list(champs.keys())
    fit = [champs[k]["fitness"] for k in depths]
    eff = [champs[k].get("efficiency", np.nan) for k in depths]
    axB.plot(depths, fit, "o-", color="#d62728", label="focused throughput")
    axB.plot(depths, eff, "s--", color="#1f77b4", label="efficiency (delivered/input)")
    axB.set_xlabel("dominant-path TIR bounces (fold depth)")
    axB.set_ylabel("fraction of input energy")
    axB.set_ylim(0, 1)
    axB.set_title("Champion per fold depth")
    axB.legend(fontsize=8)
    for k in depths:
        axB.annotate(f"spot {champs[k].get('spot',0):.1f}", (k, champs[k]["fitness"]),
                     fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")

    # Panel C: count of filled cells per fold depth (diversity reached)
    axC = fig.add_subplot(1, 3, 3)
    perdepth = {}
    for c in cells:
        perdepth[int(round(c["tir_count"]))] = perdepth.get(int(round(c["tir_count"])), 0) + 1
    ks = sorted(perdepth)
    axC.bar(ks, [perdepth[k] for k in ks], color="#2ca02c")
    axC.set_xlabel("dominant-path TIR bounces (fold depth)")
    axC.set_ylabel("# archive cells filled")
    axC.set_title("Diversity reached per fold depth")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[analyze] wrote {out_png}")


def render_champions(d, champs, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    items = []
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for depth, c in champs.items():
            p = dict(zip(FD.PARAM_NAMES, c["genotype"]))
            rres, _ = runner.simulate(FD.scene(p, n_rays=11, beam_w=FD.__dict__.get("BEAM_W", 8.0)
                                               if False else 8.0, ray_brightness=0.8,
                                               record=False, crop=True))
            imgs = rres.get("images") or []
            if not imgs or not imgs[0].get("dataUrl"):
                continue
            png = uniq(outdir / f"champ_tir{depth}.png")
            png.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
            lab = (f"{depth} TIR bounces   focused-throughput={c['fitness']:.2f}   "
                   f"efficiency={c.get('efficiency',0):.2f}   spot={c.get('spot',0):.1f}   "
                   f"T={c['T']:.0f}\n   {c.get('dom_signature','')}")
            items.append((png, lab))
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]
    for ax, (png, lab) in zip(axes, items):
        ax.imshow(mpimg.imread(png)); ax.set_title(lab, fontsize=9, loc="left"); ax.axis("off")
    fig.suptitle("Fold-QD champions: best efficient design at each TIR-bounce depth  [11 rays]",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    cs = uniq(outdir / "contact_folds.png")
    fig.savefig(cs, dpi=120); plt.close(fig)
    print(f"[analyze] wrote {cs}")


def main():
    d = json.loads(ARCH.read_text())
    m = d["meta"]
    cells = d["cells"]
    champs = champions_by_depth(cells)

    print("=" * 80)
    print("FOLD-QD — diversity of efficient non-sequential folds (tilt-enabled, uncoated)")
    print("=" * 80)
    print(f"evals={m['evals']} cells={len(cells)} invalid={m.get('invalid')} "
          f"wall={m['wall_seconds']:.0f}s  fitness={m['fitness']}")
    print(f"{'foldDepth':>9} {'cells':>5} {'bestFocThru':>11} {'effic':>6} {'spot':>6} "
          f"{'T':>5}   dominant interaction sequence")
    perdepth = {}
    for c in cells:
        perdepth[int(round(c["tir_count"]))] = perdepth.get(int(round(c["tir_count"])), 0) + 1
    for depth, c in champs.items():
        print(f"{depth:>9} {perdepth.get(depth,0):>5} {c['fitness']:>11.3f} "
              f"{c.get('efficiency',0):>6.2f} {c.get('spot',0):>6.2f} {c['T']:>5.0f}   "
              f"{c.get('dom_signature','')}")

    folded = [c for c in cells if c["tir_count"] >= 2]
    good_folded = [c for c in folded if c["fitness"] >= 0.5]
    print("-" * 80)
    print(f"folded cells (>=2 TIR): {len(folded)};  with focused-throughput >= 0.5: {len(good_folded)}")
    if good_folded:
        b = max(good_folded, key=lambda c: c["fitness"])
        print(f"best efficient fold: {int(b['tir_count'])} TIR bounces, "
              f"focused-throughput={b['fitness']:.3f}, efficiency={b.get('efficiency',0):.3f}, "
              f"spot={b.get('spot',0):.2f}, T={b['T']:.0f}")
        print(f"  sequence: {b.get('dom_signature','')}")

    figure(d, uniq(ARCH.parent / "fold_qd_figure.png"))
    render_champions(d, champs, ARCH.parent / "renders")


if __name__ == "__main__":
    main()
