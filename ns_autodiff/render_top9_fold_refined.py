"""Top-9 FOLD gallery, REFINED — each discovered design run through freeze->refine, then drawn.
The high-quality view of the fold experiment's output (3x3 refined ray diagrams)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .refine_discovered import refine_one
from .render_search_gallery import draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="ns_autodiff/results_search_freeform/long_default/archive_freeform.json")
    ap.add_argument("--out", default="ns_autodiff/results_search_freeform")
    ap.add_argument("--tir-set", type=int, nargs="+", default=[2, 3],
                    help="fold depths (TIR-bounce counts) to draw from")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # top N among the chosen fold depths (2-3 TIR), tightest discovered spot (most refinable)
    arch = json.loads(Path(args.archive).read_text())
    good = [c for c in arch["cells"]
            if int(c["tir_count"]) in args.tir_set
            and c.get("efficiency", 0) >= 0.7 and c.get("n_forward", 0) >= 14 and c["spot"] < 8.0]
    good.sort(key=lambda c: c["spot"])
    picks = [dict(genotype=c["genotype"], tir=int(c["tir_count"]), spot=c["spot"],
                  signature=c["dom_signature"]) for c in good[:args.n]]
    results = []
    for c in picks:
        x = np.array(c["genotype"], float)
        print(f"[refine] {c['tir']}-TIR  discovered spot {c['spot']:.3f} ...", flush=True)
        rr = refine_one(x, iters=args.iters)
        if rr is None:
            print("   (not freezable, showing discovered)")
            results.append(dict(x13=x, tir=c["tir"], start=c["spot"], refined=c["spot"], sig=c["signature"]))
            continue
        rr.update(tir=c["tir"], sig=c["signature"])
        results.append(dict(x13=rr["x13"], tir=c["tir"], start=rr["start_spot"],
                            refined=rr["refined_spot"], sig=c["signature"]))
        print(f"   spot {rr['start_spot']:.3f} -> {rr['refined_spot']:.3f}", flush=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10)); axes = axes.ravel()
    for ax in axes[len(results):]:
        ax.axis("off")
    for ax, r in zip(axes, results):
        draw(ax, r["x13"], f"{r['tir']}-TIR  refined spot {r['refined']:.3f} mm  (from {r['start']:.2f})\n{r['sig']}")
    fig.suptitle("FOLD freeform search — top 9 designs, REFINED (discovered → freeze → polish)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out); base = out / "top9_fold_refined.png"; png, k = base, 0
    while png.exists():
        k += 1; png = base.with_name(f"top9_fold_refined_{k}.png")
    fig.savefig(png, dpi=115, bbox_inches="tight"); plt.close(fig)
    png.with_suffix(".json").write_text(json.dumps(
        [dict(tir=r["tir"], start_spot=r["start"], refined_spot=r["refined"], signature=r["sig"],
              genotype=[float(v) for v in r["x13"]]) for r in results], indent=2))
    print(f"[top9-fold-refined] {png}")


if __name__ == "__main__":
    main()
