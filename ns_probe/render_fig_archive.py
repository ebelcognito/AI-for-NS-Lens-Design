"""
Render the figure-geometry QD archive: fold_count x FOV heatmap (coloured by collimation
fitness) + per-fold best-fitness / best-collimation bars. Answers directly: which fold
counts can collimate, and how well, under the Cheng-Hua figure layout.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY ns_probe/render_fig_archive.py [archive.json] [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
arc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "ns_probe" / "results_arvr_fig" / "archive.json"
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else arc_path.parent
outdir.mkdir(parents=True, exist_ok=True)

A = json.loads(arc_path.read_text())
cells = A["cells"]
meta = A["meta"]
bins = meta["bc_bins"]; bounds = meta["bc_bounds"]
max_fold = bins[0] - 1
fov_bins = bins[1]; fov_lo, fov_hi = bounds[1]

# heatmap grid: rows = fold (0..max_fold), cols = FOV bin; value = best fitness in cell
grid = np.full((max_fold + 1, fov_bins), np.nan)
for c in cells:
    fi = int(round(c["folds"]))
    if not (0 <= fi <= max_fold):
        continue
    cj = int(np.clip((c["fov"] - fov_lo) / (fov_hi - fov_lo) * fov_bins, 0, fov_bins - 1))
    v = c["fitness"]
    if np.isnan(grid[fi, cj]) or v > grid[fi, cj]:
        grid[fi, cj] = v

# per-fold summaries
folds = sorted({int(round(c["folds"])) for c in cells})
best_fit = [max(c["fitness"] for c in cells if int(round(c["folds"])) == f) for f in folds]
best_sig = [min(c["sigma_mean"] for c in cells if int(round(c["folds"])) == f) for f in folds]
n_cells = [sum(1 for c in cells if int(round(c["folds"])) == f) for f in folds]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})

im = ax[0].imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                  extent=[fov_lo, fov_hi, -0.5, max_fold + 0.5])
ax[0].set_xlabel("FOV (deg)"); ax[0].set_ylabel("fold count (TIR bounces)")
ax[0].set_yticks(range(max_fold + 1))
ax[0].set_title("Archive: best collimation-fitness per (fold, FOV) cell")
fig.colorbar(im, ax=ax[0], label="fitness = T*exp(-sigma/sigma0)")

ax[1].barh(folds, best_fit, color="steelblue")
ax[1].set_yticks(folds); ax[1].set_ylabel("fold count")
ax[1].set_xlabel("best fitness"); ax[1].set_title("Best imaging quality vs folds")
for f, v, n in zip(folds, best_fit, n_cells):
    ax[1].text(v, f, f" {v:.3f} (n={n})", va="center", fontsize=8)

ax[2].barh(folds, best_sig, color="indianred")
ax[2].set_yticks(folds); ax[2].set_xlabel("best collimation sigma (deg, lower=better)")
ax[2].set_title("Tightest collimation vs folds")
for f, v in zip(folds, best_sig):
    ax[2].text(v, f, f" {v:.2f}", va="center", fontsize=8)

ttl = (f"AR/VR freeform prism (figure layout) | {meta['evals']} evals | "
       f"source_aim={meta.get('source_aim')}deg | folding NOT rewarded")
fig.suptitle(ttl, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = outdir / "fig_archive_heatmap.png"
k = 0
while out.exists():
    k += 1; out = outdir / f"fig_archive_heatmap_{k}.png"
fig.savefig(out, dpi=130)
print(f"[render] {out}")
print("\nper-fold summary:")
print(f"  {'folds':>5} {'cells':>6} {'best_fit':>9} {'min_sigma':>10}")
for f, n, bf, bs in zip(folds, n_cells, best_fit, best_sig):
    print(f"  {f:>5} {n:>6} {bf:>9.4f} {bs:>10.3f}")
