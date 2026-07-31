"""
Render the probe's discovered designs (setup + individual ray paths) with the ray-optics
native renderer (node-canvas + CropBox). Works for both the single-element (7-D genotype) and
2-element (12-D genotype) probes — the builder is chosen by genotype length.

Renders with a SMALL number of distinct rays (default ~10) so individual ray lines and their
folds are visible, rather than dense filled bundles.

    python3 ns_probe/render_designs.py --archive ns_probe/results_single/archive.json \
        --outdir ns_probe/results_single/renders --rays 10
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ns_probe import probe, probe_single
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

CROP_X = (-6.0, 128.0)
CROP_Y = (-48.0, 48.0)
CROP_W = 1280


def builder_for(genotype):
    return probe_single.build_scene if len(genotype) == probe_single.DIM else probe.build_scene


def render_scene(genotype, brightness, n_rays):
    """Optics from the matching probe builder, but few rays + a CropBox for rendering."""
    scene = builder_for(genotype)(genotype, record=False)
    scene.pop("recordInteractions", None)
    # few distinct rays: Beam emits ~ (height * rayModeDensity) rays
    scene["rayModeDensity"] = float(n_rays) / (2.0 * probe.BEAM_HALF_H)
    scene["objs"][0]["brightness"] = float(brightness)   # the Beam is objs[0]
    scene["objs"].append({"type": "CropBox",
                          "p1": {"x": CROP_X[0], "y": CROP_Y[0]},
                          "p4": {"x": CROP_X[1], "y": CROP_Y[1]},
                          "width": CROP_W})
    return scene


def render_png(runner, genotype, brightness, n_rays, out_path) -> bool:
    res, _ = runner.simulate(render_scene(genotype, brightness, n_rays))
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        print(f"  ! no image for {out_path.name}: err={res.get('error')}")
        return False
    out_path.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
    return True


def label(c, kind, rank):
    g = np.array(c["genotype"], float)
    if len(g) == probe_single.DIM:
        refl = f"front={g[5]:.2f} back={g[6]:.2f}"
    else:
        gg = g.reshape(probe.N_ELEM, -1)
        refl = "refl=[" + ", ".join(f"{row[5]:.2f}" for row in gg) + "]"
    return (f"{kind} #{rank}   {refl}   reversals={c['mean_reversals']:.1f}   "
            f"flux={c.get('flux',0):.2f}   spot={c.get('spot_rms',0):.2f}   "
            f"foldfrac={c.get('folded_frac',0):.2f}   T={c['T']:.0f}   fit={c['fitness']:.2f}")


def contact_sheet(items, out_path, title):
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.7 * n))
    if n == 1:
        axes = [axes]
    for ax, (png, lab) in zip(axes, items):
        ax.imshow(mpimg.imread(png)); ax.set_title(lab, fontsize=9, loc="left"); ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[render] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_single" / "archive.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-folded", type=int, default=5)
    ap.add_argument("--n-seq", type=int, default=3)
    ap.add_argument("--rays", type=int, default=10)
    ap.add_argument("--brightness", type=float, default=2.5)
    args = ap.parse_args()

    d = json.loads(Path(args.archive).read_text())
    cells = d["cells"]
    variant = d["meta"].get("variant", "two_element")
    outdir = Path(args.outdir) if args.outdir else Path(args.archive).parent / "renders"
    outdir.mkdir(parents=True, exist_ok=True)

    folded = sorted([c for c in cells if c["mean_reversals"] >= 2.0 and c["fitness"] > -900],
                    key=lambda c: c["fitness"], reverse=True)
    seq = sorted([c for c in cells if c["mean_reversals"] < 1.0 and c["fitness"] > -900],
                 key=lambda c: c["fitness"], reverse=True)
    tag = "single-element" if variant == "single_element" else "2-element"

    with PersistentRayOpticsRunner(timeout_s=40) as runner:
        ns_items = []
        for i, c in enumerate(folded[: args.n_folded], 1):
            png = outdir / f"ns_top{i}.png"
            if render_png(runner, np.array(c["genotype"], float), args.brightness, args.rays, png):
                ns_items.append((png, label(c, "NON-SEQUENTIAL (folded)", i))); print("  " + ns_items[-1][1])
        seq_items = []
        for i, c in enumerate(seq[: args.n_seq], 1):
            png = outdir / f"seq_top{i}.png"
            if render_png(runner, np.array(c["genotype"], float), args.brightness, args.rays, png):
                seq_items.append((png, label(c, "SEQUENTIAL (transmission)", i))); print("  " + seq_items[-1][1])

    if ns_items:
        contact_sheet(ns_items, outdir / "contact_non_sequential.png",
                      f"Probe ({tag}) — top non-sequential (folded) designs  [{args.rays} rays]")
    if seq_items:
        contact_sheet(seq_items, outdir / "contact_sequential.png",
                      f"Probe ({tag}) — top sequential (transmission) designs  [{args.rays} rays]")


if __name__ == "__main__":
    main()
