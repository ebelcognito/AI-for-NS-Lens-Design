"""
Render a QD champion BEFORE vs AFTER LM refinement as a side-by-side ray diagram.

Reads a refine_lm.py report JSON (which carries both genotype_start and
genotype_refined plus the 21-ray spot numbers), traces each setup through the JS
engine with cropping ON (so the engine returns a rendered PNG of the rays), and
stitches the two into one labelled comparison image. Never overwrites.

    python3 ns_probe/render_refine_comparison.py                         # headline 3-TIR
    python3 ns_probe/render_refine_comparison.py --report ns_probe/results_refine_lm/refine_cell4_11.json
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import fold_demo as FD


def render_png(runner, p, *, n_rays, beam_w, brightness):
    """Return an RGBA image array of the cropped ray render for geometry p."""
    res, _ = runner.simulate(FD.scene(p, n_rays=n_rays, beam_w=beam_w,
                                      ray_brightness=brightness, record=False, crop=True))
    imgs = res.get("images") or []
    if not (imgs and imgs[0].get("dataUrl")):
        raise RuntimeError("engine returned no rendered image (crop render failed)")
    png = base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])
    return mpimg.imread(io.BytesIO(png))


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPO / "ns_probe" / "results_refine_lm"
                                            / "refine_cell3_5_2.json"))
    ap.add_argument("--rays", type=int, default=25)
    ap.add_argument("--beam-w", type=float, default=8.0)
    ap.add_argument("--brightness", type=float, default=0.55)
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())
    ch = rep["champion"]
    p0 = {k: float(v) for k, v in ch["genotype_start"].items()}
    pF = {k: float(v) for k, v in ch["genotype_refined"].items()}
    s0 = rep["result"]["spot_21ray_start"]
    sF = rep["result"]["spot_21ray_refined"]
    cell = ch["cell"]
    domsig = ch["dom_signature"]

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        img0 = render_png(runner, p0, n_rays=args.rays, beam_w=args.beam_w,
                          brightness=args.brightness)
        imgF = render_png(runner, pF, n_rays=args.rays, beam_w=args.beam_w,
                          brightness=args.brightness)

    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    ax[0].imshow(img0); ax[0].axis("off")
    ax[0].set_title(f"BEFORE  —  QD champion (cell {cell[0]},{cell[1]})\n"
                    f"21-ray RMS spot = {s0:.2f}", fontsize=13, color="tab:red")
    ax[1].imshow(imgF); ax[1].axis("off")
    ax[1].set_title(f"AFTER  —  FD-Jacobian + Levenberg–Marquardt refinement\n"
                    f"21-ray RMS spot = {sF:.2f}   ({100*(1-sF/s0):+.0f}%)",
                    fontsize=13, color="tab:green")
    fig.suptitle(f"Non-sequential fold  {domsig}   |   throughput & topology preserved",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = Path(args.report).with_name(f"compare_cell{cell[0]}_{cell[1]}.png")
    k = 0
    while out.exists():
        k += 1
        out = out.with_name(f"compare_cell{cell[0]}_{cell[1]}_{k}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[render] before/after comparison -> {out}")


if __name__ == "__main__":
    main()
