"""Plot what RANDOM sampling (arm B0) actually produces: (1) a rendered gallery of its best
folds at matched 14k x 5-seed budget, parallel to top10_folds.png; (2) a distribution figure
contrasting random vs CMA-ME fold quality + the E2a random-hit funnel."""
from __future__ import annotations
import base64, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))
import fold_demo as FD
from fold_gallery import hq_scene
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

DISC = REPO / "ns_probe" / "results_disc"
OUT = DISC / "random_top"; OUT.mkdir(parents=True, exist_ok=True)


def fold_cells(glob):
    cells = []
    for ap in sorted(REPO.glob(glob)):
        for c in json.loads(ap.read_text())["cells"]:
            if c.get("tir_count", 0) >= 2:
                c["_seed"] = ap.parent.name; cells.append(c)
    return cells


# ---- (1) rendered gallery of random's best folds (matched 14k x 5 seeds) ----
rand = sorted(fold_cells("ns_probe/results_disc/random/seed*/archive.json"),
              key=lambda c: -c["fitness"])[:10]
items = []
with PersistentRayOpticsRunner(timeout_s=30) as runner:
    for i, c in enumerate(rand, 1):
        p = dict(zip(FD.PARAM_NAMES, c["genotype"]))
        rres, _ = runner.simulate(hq_scene(p))
        imgs = rres.get("images") or []
        if not imgs or not imgs[0].get("dataUrl"):
            print(f"  ! render #{i} failed"); continue
        png = OUT / f"rand_{i:02d}_{c['_seed']}.png"
        png.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
        flag = "" if c["fitness"] >= 0.9 else "  (below 0.9)"
        items.append((png, f"#{i}  {c['_seed']} | focthru={c['fitness']:.2f}{flag}  spot={c['spot']:.2f}  "
                           f"eff={c['efficiency']:.0%}  {int(c['tir_count'])} TIR\n{c['dom_signature']}"))
rows = (len(items) + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(15, 2.7 * rows))
for ax in np.atleast_2d(axes).ravel(): ax.axis("off")
for ax, (png, lab) in zip(np.atleast_2d(axes).ravel(), items):
    ax.imshow(mpimg.imread(png)); ax.set_title(lab, fontsize=8, loc="left"); ax.axis("off")
fig.suptitle("Best folds RANDOM sampling found (arm B0, matched 14k x 5 seeds; top-10 by focthru). "
             "Only 1 reaches the 0.9 high-eff bar.", fontsize=11, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(DISC / "random_top10_folds.png", dpi=120)
print(f"wrote {DISC/'random_top10_folds.png'}")

# ---- (2) distribution + funnel figure (no engine) ----
cm = [c["fitness"] for c in fold_cells("ns_probe/results_disc/cmame/seed*/archive.json")]
rd = [c["fitness"] for c in fold_cells("ns_probe/results_disc/random/seed*/archive.json")]
cm_t = fold_cells("ns_probe/results_disc/cmame/seed*/archive.json")
rd_t = fold_cells("ns_probe/results_disc/random/seed*/archive.json")
hr = json.loads((DISC / "random50k" / "archive.json").read_text())["hit_rate"]

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
bins = np.linspace(0, 1, 26)
ax[0].hist(rd, bins=bins, alpha=0.6, color="#1f77b4", label=f"random (n={len(rd)})")
ax[0].hist(cm, bins=bins, alpha=0.6, color="#d62728", label=f"CMA-ME (n={len(cm)})")
ax[0].axvline(0.9, color="k", ls="--", lw=1); ax[0].text(0.905, ax[0].get_ylim()[1]*0.9, "high-eff 0.9", fontsize=8)
ax[0].set(title="(a) fold-cell focthru distribution (matched 14k)", xlabel="focused throughput", ylabel="# archive fold-cells")
ax[0].legend()

ax[1].scatter([c["tir_count"]+np.random.uniform(-0.15,0.15) for c in rd_t], [c["fitness"] for c in rd_t],
              s=12, alpha=0.4, color="#1f77b4", label="random")
ax[1].scatter([c["tir_count"]+np.random.uniform(-0.15,0.15) for c in cm_t], [c["fitness"] for c in cm_t],
              s=12, alpha=0.5, color="#d62728", label="CMA-ME")
ax[1].axhline(0.9, color="k", ls="--", lw=1)
ax[1].set(title="(b) focthru vs fold depth (matched 14k)", xlabel="TIR bounces (dominant path)", ylabel="focused throughput")
ax[1].legend()

stages = ["all draws\n50,000", "valid\n37,337", "folded >=2TIR\n2,141", "good foc>=0.5\n15", "high foc>=0.9\n0"]
vals = [hr["n_evals"], hr["n_valid"], hr["n_fold"], hr["n_good_fold"], max(hr["n_high_fold"], 0.4)]
y = np.arange(len(stages))[::-1]
ax[2].barh(y, vals, color=["#999","#4c9","#49c","#f93","#d33"])
ax[2].set_yticks(y); ax[2].set_yticklabels(stages, fontsize=9)
ax[2].set_xscale("log"); ax[2].set(title="(c) E2a random-hit funnel (50k draws)", xlabel="count (log scale)")
for yi, v in zip(y, vals):
    ax[2].text(v*1.2, yi, f"{int(v) if v>=1 else 0}", va="center", fontsize=9)
fig.suptitle("What random sampling produces vs CMA-ME", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(DISC / "random_vs_cmame_distribution.png", dpi=130)
print(f"wrote {DISC/'random_vs_cmame_distribution.png'}")
