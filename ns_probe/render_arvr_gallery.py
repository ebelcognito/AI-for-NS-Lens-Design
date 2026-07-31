"""
Render the best-N AR/VR prism designs from the QD archive as a single grid image.

    python3 ns_probe/render_arvr_gallery.py            # best 12, 3x4 grid
    python3 ns_probe/render_arvr_gallery.py --rows 3 --cols 4 --sort fitness
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import arvr_prism as AV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO / "ns_probe" / "results_arvr_qd" / "archive.json"))
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--sort", default="fitness", choices=["fitness", "fov"])
    ap.add_argument("--n-fan", type=int, default=9)
    args = ap.parse_args()

    n = args.rows * args.cols
    arc = json.loads(Path(args.archive).read_text())
    cells = sorted(arc["cells"], key=lambda c: -c[args.sort])[:n]
    print(f"top {len(cells)} by {args.sort}: "
          f"fitness {cells[0]['fitness']:.3f}..{cells[-1]['fitness']:.3f}")

    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        rendered = []
        for c in cells:
            x = np.array(c["genotype"], dtype=float)
            res, _ = runner.simulate(AV.scene_arvr(x, n_fan=args.n_fan, brightness=0.6,
                                                   record=False, crop=True))
            imgs = res.get("images") or []
            if not (imgs and imgs[0].get("dataUrl")):
                rendered.append((None, c)); continue
            img = mpimg.imread(io.BytesIO(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])))
            rendered.append((img, c))

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 5.2, args.rows * 4.4))
    for ax, (img, c) in zip(np.atleast_1d(axes).flat, rendered):
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"fit {c['fitness']:.3f}  |  FOV {c['fov']:.1f}°  |  "
                     f"depth {c['depth']:.0f}  |  TIR {int(c.get('tir', 0))}",
                     fontsize=11)
    for ax in np.atleast_1d(axes).flat[len(rendered):]:
        ax.axis("off")
    fig.suptitle(f"AR/VR off-axis prism QD — best {n} designs by {args.sort} "
                 f"(display → TIR fold → collimated to eye pupil)", fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = REPO / "ns_probe" / "results_arvr_qd" / f"arvr_gallery_top{n}.png"
    k = 0
    while out.exists():
        k += 1; out = out.with_name(f"arvr_gallery_top{n}_{k}.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"[gallery] {out}")


if __name__ == "__main__":
    main()
