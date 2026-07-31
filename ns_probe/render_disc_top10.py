"""Render the top-10 high-eff fold designs CMA-ME found (E1 archives) as a contact sheet."""
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

OUT = REPO / "ns_probe" / "results_disc" / "top10"
OUT.mkdir(parents=True, exist_ok=True)

cells = []
for ap in sorted((REPO / "ns_probe" / "results_disc" / "cmame").glob("seed*/archive.json")):
    for c in json.loads(ap.read_text())["cells"]:
        if c.get("tir_count", 0) >= 2 and c["fitness"] >= 0.9:
            c["_seed"] = ap.parent.name; cells.append(c)
# top 10 distinct FAMILIES (best fitness per dom_signature), then sort by fitness
best_per_fam = {}
for c in sorted(cells, key=lambda c: -c["fitness"]):
    best_per_fam.setdefault(c["dom_signature"], c)
picks = sorted(best_per_fam.values(), key=lambda c: -c["fitness"])[:10]

items = []
with PersistentRayOpticsRunner(timeout_s=30) as runner:
    for i, c in enumerate(picks, 1):
        p = dict(zip(FD.PARAM_NAMES, c["genotype"]))
        rres, _ = runner.simulate(hq_scene(p))
        imgs = rres.get("images") or []
        if not imgs or not imgs[0].get("dataUrl"):
            print(f"  ! render failed #{i}: {rres.get('error')}"); continue
        png = OUT / f"top_{i:02d}_{c['_seed']}.png"
        png.write_bytes(base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1]))
        lab = (f"#{i}  {c['_seed']} | focthru={c['fitness']:.2f}  spot={c['spot']:.2f}  "
               f"eff={c['efficiency']:.0%}  {int(c['tir_count'])} TIR  T={c['thinness_T']:.0f}\n"
               f"{c['dom_signature']}")
        items.append((png, lab))

rows = (len(items) + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(15, 2.7 * rows))
for ax in np.atleast_2d(axes).ravel(): ax.axis("off")
for ax, (png, lab) in zip(np.atleast_2d(axes).ravel(), items):
    ax.imshow(mpimg.imread(png)); ax.set_title(lab, fontsize=8, loc="left"); ax.axis("off")
fig.suptitle("Top-10 high-eff non-sequential folds CMA-ME found (E1, focthru>=0.9, >=2 TIR; "
             "one per distinct fold family)", fontsize=11, y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.99])
out = REPO / "ns_probe" / "results_disc" / "top10_folds.png"
fig.savefig(out, dpi=120); print(f"wrote {out}")
