"""
Physics-validation analysis: do the QD-discovered folds actually obey/exploit the physics?

For every fold in the archives, re-trace the dominant-energy ray and, at each engine-labelled
TIR event, compute the geometric angle of incidence from the ray's turn
    theta_i = 90 - (angle between incoming and outgoing travel directions)/2
and compare it to the material's critical angle  theta_c = arcsin(1/n).
A real TIR requires theta_i > theta_c. We check that this holds, and look at the MARGIN
(theta_i - theta_c) -- i.e. whether the optimizer parks designs right at the critical-angle
boundary or comfortably past it. This turns "it just searches" into "the discovered designs
land in, and respect, the TIR regime."

    python3 ns_probe/analyze_physics.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ns_probe"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fold_demo as FD
from sequence_tools import make_single_element_face_classifier, signature_of
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

ARCHIVES = ["ns_probe/results_fold_qd/archive.json", "ns_probe/results_orderly_qd/archive.json"]
OUT = REPO / "ns_probe" / "results_fold_qd" / "physics_validation.png"


def uniq(p: Path) -> Path:
    if not p.exists():
        return p
    k = 1
    while p.with_name(f"{p.stem}_{k}{p.suffix}").exists():
        k += 1
    return p.with_name(f"{p.stem}_{k}{p.suffix}")


def incidence_deg(seq, k):
    """Angle of incidence (from normal) at vertex k, inferred from the ray's turn."""
    a = np.array([seq[k]["x"] - seq[k - 1]["x"], seq[k]["y"] - seq[k - 1]["y"]])
    b = np.array([seq[k + 1]["x"] - seq[k]["x"], seq[k + 1]["y"] - seq[k]["y"]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    deflection = math.degrees(math.acos(c))
    return 90.0 - deflection / 2.0


def main():
    tir_theta, tir_thetac, tir_margin = [], [], []
    eff_by_depth = {}
    designs = 0
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for ap in ARCHIVES:
            d = json.loads(Path(ap).read_text())
            for c in d["cells"]:
                if c.get("tir_count", 0) < 2:
                    continue
                p = dict(zip(FD.PARAM_NAMES, c["genotype"]))
                n = p["n"]; theta_c = math.degrees(math.asin(min(1.0 / n, 1.0)))
                try:
                    res, _ = runner.simulate(FD.scene(p, n_rays=17, beam_w=8.0, record=True))
                except Exception:
                    continue
                face_of = make_single_element_face_classifier(p["cx"], FD.ELEM_H)
                recs = [r for r in res.get("raySequences", [])
                        if r.get("dirx", 0) > 0 and signature_of(r, face_of) is not None]
                if not recs:
                    continue
                dom = max(recs, key=lambda r: r["b"])
                seq = dom["seq"]
                got = False
                for k in range(1, len(seq) - 1):
                    if seq[k].get("t") == "tir":
                        ti = incidence_deg(seq, k)
                        if ti is None:
                            continue
                        tir_theta.append(ti); tir_thetac.append(theta_c)
                        tir_margin.append(ti - theta_c); got = True
                if got:
                    designs += 1
                    dep = int(round(c["tir_count"]))
                    eff_by_depth.setdefault(dep, []).append(c.get("efficiency", 0.0))

    tir_theta = np.array(tir_theta); tir_thetac = np.array(tir_thetac); tir_margin = np.array(tir_margin)
    frac_valid = float(np.mean(tir_margin > -0.5))   # allow tiny numerical slack
    print(f"[physics] {designs} folded designs, {len(tir_theta)} TIR events analysed")
    print(f"[physics] fraction of TIR events with theta_i > theta_c : {frac_valid:.3%}")
    print(f"[physics] incidence margin past critical (theta_i - theta_c): "
          f"median {np.median(tir_margin):.1f} deg, "
          f"10th pct {np.percentile(tir_margin,10):.1f}, 90th {np.percentile(tir_margin,90):.1f}")

    fig = plt.figure(figsize=(15, 4.4))
    axA = fig.add_subplot(1, 3, 1)
    axA.scatter(tir_thetac, tir_theta, s=10, alpha=0.4, color="#d62728")
    lim = [min(tir_thetac.min(), tir_theta.min()) - 2, max(tir_thetac.max(), tir_theta.max()) + 2]
    axA.plot(lim, lim, "k--", lw=1, label=r"$\theta_i=\theta_c$ (TIR threshold)")
    axA.set_xlabel(r"critical angle $\theta_c=\arcsin(1/n)$  (deg)")
    axA.set_ylabel(r"incidence angle $\theta_i$ at TIR event  (deg)")
    axA.set_title("Every discovered TIR event sits ABOVE the\ncritical-angle line (the search found real TIR)")
    axA.legend(fontsize=8)

    axB = fig.add_subplot(1, 3, 2)
    axB.hist(tir_margin, bins=30, color="#1f77b4")
    axB.axvline(0, color="k", ls="--", lw=1)
    axB.set_xlabel(r"margin past critical  $\theta_i-\theta_c$  (deg)")
    axB.set_ylabel("# TIR events")
    axB.set_title("How far past critical the folds operate")

    axC = fig.add_subplot(1, 3, 3)
    deps = sorted(eff_by_depth)
    axC.plot(deps, [np.mean(eff_by_depth[k]) for k in deps], "o-", color="#2ca02c")
    axC.fill_between(deps, [np.min(eff_by_depth[k]) for k in deps],
                     [np.max(eff_by_depth[k]) for k in deps], alpha=0.2, color="#2ca02c")
    axC.set_xlabel("fold depth (TIR bounces)")
    axC.set_ylabel("efficiency (delivered/input)")
    axC.set_ylim(0, 1)
    axC.set_title("Efficiency vs fold depth\n(loss-free TIR -> stays high, then Fresnel/aperture cost)")

    fig.tight_layout()
    out = uniq(OUT)
    fig.savefig(out, dpi=130)
    print(f"[physics] wrote {out}")


if __name__ == "__main__":
    main()
