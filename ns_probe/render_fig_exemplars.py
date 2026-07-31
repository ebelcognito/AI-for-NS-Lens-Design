"""Render specific (fold-count) exemplars from the figure-geometry archive as ray diagrams.
Picks, for each requested fold count, the best-fitness cell and the tightest-collimation
cell (T_min>0.25), and renders both (never overwriting) with sidecars.

    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY ns_probe/render_fig_exemplars.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ns_probe"))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
import arvr_prism_fig as FG

arc = json.loads((REPO / "ns_probe" / "results_arvr_fig" / "archive.json").read_text())
cells = arc["cells"]
outdir = REPO / "ns_probe" / "results_arvr_fig"


def pick(fold, key, reverse, need_T=0.0):
    pool = [c for c in cells if int(round(c["folds"])) == fold and c["T_min"] >= need_T]
    if not pool:
        return None
    return sorted(pool, key=lambda c: key(c), reverse=reverse)[0]


with PersistentRayOpticsRunner(timeout_s=30) as runner:
    targets = []
    for fold in (2, 3, 4):
        bf = pick(fold, lambda c: c["fitness"], True)
        if bf:
            targets.append((f"fig_fold{fold}_bestfit", bf))
        bs = pick(fold, lambda c: c["sigma_mean"], False, need_T=0.25)
        if bs and bs is not bf:
            targets.append((f"fig_fold{fold}_tightest", bs))
    for stem, c in targets:
        x = np.array(c["genotype"])
        res, _ = runner.simulate(FG.scene_fig(x))
        m = FG.arvr_metrics_fig(res)
        FG._render(runner, x, outdir, stem, m, c["folds"])
        print(f"   -> {stem}: folds={c['folds']} fit={c['fitness']:.3f} "
              f"FOV={c['fov']:.1f} Tmin={c['T_min']:.2f} sig={c['sigma_mean']:.3f}")
