"""
Render the three stages of ONE CMA-ME step captured by capture_cma_step.py:
  (1) the elite the emitter was seeded from       -> panel1_seed.png
  (2) the population em.ask() drew around it       -> panel2_population.png (labelled grid)
  (3) the offspring the archive accepted            -> panel3_winner.png
plus a stitched storyboard.png. Each genotype is turned into a ray-trace layout with the
same engine call fold_demo.main uses (scene(..., record=False, crop=True) -> images[0].dataUrl).

    PYTHONPATH=.:ns_probe python ns_probe/render_cma_step.py --evals 11430
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import fold_demo as FD
from run_fold_qd import evaluate
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

REPO = Path(__file__).resolve().parents[1]
CAP = REPO / "ns_probe" / "results_disc" / "cma_step_capture"

WIN = "#31d17a"      # winner accent (green)
FOLD = "#f5c518"     # folded member (gold)
TRANS = "#5aa9e6"    # transmission member (blue)
BAD = "#555555"      # invalid geometry (gray)


def render_geno(runner, geno, *, n_rays=9):
    """Genotype -> PIL RGB image of its ray-trace layout, or None if geometry invalid."""
    p = FD.vec_to_params(np.asarray(geno, float))
    if not FD.geometry_valid(p):
        return None
    res, _ = runner.simulate(FD.scene(p, n_rays=n_rays, beam_w=8.0, ray_brightness=0.85,
                                      record=False, crop=True))
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        return None
    raw = base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def imshow(ax, img, title=None, edge=None, lw=3):
    ax.set_xticks([]); ax.set_yticks([])
    if img is None:
        ax.set_facecolor("#111111")
        ax.text(0.5, 0.5, "invalid\ngeometry", ha="center", va="center",
                color="#999", fontsize=9, transform=ax.transAxes)
    else:
        ax.imshow(img)
    if edge:
        for s in ax.spines.values():
            s.set_color(edge); s.set_linewidth(lw)
    else:
        for s in ax.spines.values():
            s.set_color("#333"); s.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=8.5, color="#ddd", pad=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=11430, help="n_evals key of the highlight step")
    ap.add_argument("--out", default=str(CAP / "renders"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(CAP / "steps_summary.json"))
    step = next((s for s in d["highlights"] if s["n_evals"] == args.evals), None)
    if step is None:
        raise SystemExit(f"no highlight with n_evals={args.evals}; "
                         f"available: {[s['n_evals'] for s in d['highlights']]}")

    seed_geno = step["origin"]["genotype"]
    members = step["members"]
    wi = step["winner_idx"]
    tag = f"e{args.evals}"

    plt.rcParams.update({"figure.facecolor": "#0d0d0d", "savefig.facecolor": "#0d0d0d",
                         "axes.facecolor": "#0d0d0d", "text.color": "#eee"})

    with PersistentRayOpticsRunner(timeout_s=25) as r:
        seed_ev = evaluate(r, np.asarray(seed_geno, float))
        seed_img = render_geno(r, seed_geno)
        mem_imgs = [render_geno(r, m["genotype"]) for m in members]
        win_img = mem_imgs[wi]
        win = members[wi]

        # ---------- Panel 1: the seed elite ----------
        fig, ax = plt.subplots(figsize=(7, 3.6))
        sfoc = seed_ev["fitness"] if seed_ev else float("nan")
        stir = seed_ev["info"]["tir_count"] if seed_ev else 0
        sspot = seed_ev["info"]["spot"] if seed_ev else float("nan")
        imshow(ax, seed_img)
        ax.set_title(f"(1) SEED — elite drawn from the archive to start this CMA step\n"
                     f"foc={sfoc:.3f}   TIR bounces={stir}   spot={sspot:.2f}",
                     fontsize=10, color=FOLD, pad=6)
        fig.tight_layout(); fig.savefig(out / f"panel1_seed_{tag}.png", dpi=140); plt.close(fig)

        # ---------- Panel 2: the population grid ----------
        n = len(members); cols = 5; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.1 * rows))
        axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i >= n:
                ax.axis("off"); continue
            m = members[i]
            if not m["valid"]:
                edge, lab = BAD, f"m{i}  invalid"
            else:
                folded = m["tir"] >= 2
                edge = WIN if i == wi else (FOLD if folded else TRANS)
                kind = f"TIR={m['tir']}" if folded else "transmit"
                win_tag = "  ★WINNER" if i == wi else ""
                lab = f"m{i}  foc={m['fitness']:.2f}  {kind}{win_tag}"
            imshow(ax, mem_imgs[i], title=lab, edge=edge, lw=(4 if i == wi else 1.5))
        fig.suptitle(f"(2) POPULATION — the {n} offspring em.ask() sampled around the seed "
                     f"(σ={step['sigma']:.2f})\n"
                     f"gold = folded (≥2 TIR)   blue = straight transmission   "
                     f"green = accepted winner   gray = invalid",
                     fontsize=11, color="#eee", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"panel2_population_{tag}.png", dpi=140); plt.close(fig)

        # ---------- Panel 3: the winner ----------
        fig, ax = plt.subplots(figsize=(7, 3.6))
        imshow(ax, win_img, edge=WIN, lw=4)
        ax.set_title(f"(3) WINNER — offspring the archive accepted ({win['status']})\n"
                     f"foc={win['fitness']:.3f}   TIR bounces={win['tir']}   spot={win['spot']:.2f}"
                     f"   [{win['dom']}]",
                     fontsize=10, color=WIN, pad=6)
        fig.tight_layout(); fig.savefig(out / f"panel3_winner_{tag}.png", dpi=140); plt.close(fig)

        # ---------- Storyboard: seed -> population -> winner ----------
        fig = plt.figure(figsize=(16, 7.6))
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1.25] + [1] * rows,
                              left=0.055, right=0.99, top=0.86, bottom=0.03,
                              hspace=0.42, wspace=0.12)
        # top band: seed (left) and winner (right), with an arrow between
        axs = fig.add_subplot(gs[0, :2])
        imshow(axs, seed_img, edge=FOLD, lw=3)
        axs.set_title(f"(1) SEED elite   foc={sfoc:.3f}  TIR={stir}  spot={sspot:.2f}",
                      fontsize=10, color=FOLD)
        axw = fig.add_subplot(gs[0, 3:])
        imshow(axw, win_img, edge=WIN, lw=3)
        axw.set_title(f"(3) WINNER   foc={win['fitness']:.3f}  TIR={win['tir']}  spot={win['spot']:.2f}",
                      fontsize=10, color=WIN)
        fig.suptitle("one CMA-ME step  —  seed 2, "
                     f"eval {args.evals} (emitter {step['emitter']}, age {step['emitter_age']})",
                     fontsize=14, color="#eee", weight="bold", y=0.965)
        arr = FancyArrowPatch((0.415, 0.74), (0.60, 0.74), transform=fig.transFigure,
                              arrowstyle="-|>", mutation_scale=22, color="#eee", lw=2)
        fig.add_artist(arr)
        fig.text(0.5, 0.775, "CMA samples,\narchive keeps the best", ha="center",
                 fontsize=9.5, color="#bbb")
        # left-margin band label (no overlap with tiles)
        fig.text(0.02, 0.36, "(2) POPULATION\nem.ask() → 10 offspring", ha="center",
                 va="center", rotation=90, fontsize=11, color="#eee")
        # middle/bottom bands: population grid
        for i in range(rows * cols):
            ax = fig.add_subplot(gs[1 + i // cols, i % cols])
            if i >= n:
                ax.axis("off"); continue
            m = members[i]
            if not m["valid"]:
                edge, lab = BAD, f"m{i} invalid"
            else:
                folded = m["tir"] >= 2
                edge = WIN if i == wi else (FOLD if folded else TRANS)
                kind = f"TIR{m['tir']}" if folded else "trans"
                lab = f"m{i} {m['fitness']:.2f} {kind}" + ("  ★" if i == wi else "")
            imshow(ax, mem_imgs[i], title=lab, edge=edge, lw=(3.5 if i == wi else 1.2))
        fig.savefig(out / f"storyboard_{tag}.png", dpi=130); plt.close(fig)

    print(f"[render] wrote panels + storyboard for eval {args.evals} to {out}")
    for f in sorted(out.glob(f"*_{tag}.png")):
        print("   ", f.relative_to(REPO))


if __name__ == "__main__":
    main()
