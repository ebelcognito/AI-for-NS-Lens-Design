"""
Two-panel summary of the descriptor ablation.
(A) seed-0 non-sequential CELL COUNT for every BC1 axis -> shows the collapse: axes that are
    NOT fold-defining let transmission/shallow designs evict folds from their niches.
(B) 3-seed mean of HIGH-EFFICIENCY fold families (focthru>=0.9) for the fold-defining axes ->
    locus reliably surfaces ~2x baseline; depth x locus matches it with a smaller archive.

    python3 ns_probe/descriptor_summary_figure.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def cells(p):
    return json.load(open(p))["cells"] if Path(p).exists() else None


def ns_count(p):
    cs = cells(p); return sum(1 for c in cs if c.get("tir_count", 0) >= 2) if cs else 0


def high_fam(p):
    cs = cells(p)
    return len({c["dom_signature"] for c in cs
                if c.get("tir_count", 0) >= 2 and c["fitness"] >= 0.9}) if cs else 0


def base(bc): return f"ns_probe/results_desc_{bc}/archive.json"

# panel A: all 7 single-axis runs at seed 0 (BC1 x thinness)
A_AXES = ["baseline", "nsfrac", "tilt", "critmargin", "locus", "robust"]
A_LBL = {"baseline": "depth\n(TIR count)", "nsfrac": "NS-frac\n(continuous)", "tilt": "tilt\nmagnitude",
         "critmargin": "crit-angle\nmargin", "locus": "fold locus\n(edge-TIR)", "robust": "tolerance"}
A_vals = [ns_count(base(a)) for a in A_AXES]
fold_defining = {"baseline", "locus"}

# panel B: 3-seed means for the fold-defining pairings
B = {
    "depth x thinness\n(baseline)": [base("baseline"),
                                     "ns_probe/results_desc_baseline_s1/archive.json",
                                     "ns_probe/results_desc_baseline_s2/archive.json"],
    "locus x thinness": [base("locus"),
                         "ns_probe/results_desc_locus_s1/archive.json",
                         "ns_probe/results_desc_locus_s2/archive.json"],
    "depth x locus\n(combined)": ["ns_probe/results_desc_combo_s0/archive.json",
                                  "ns_probe/results_desc_combo_s1/archive.json",
                                  "ns_probe/results_desc_combo_s2/archive.json"],
}
B_lbl, B_mean, B_err = [], [], []
for k, ps in B.items():
    vals = [high_fam(p) for p in ps if Path(p).exists()]
    B_lbl.append(k); B_mean.append(st.mean(vals)); B_err.append(st.pstdev(vals))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
colA = ["#2ca02c" if a in fold_defining else "#d62728" for a in A_AXES]
axA.bar(range(len(A_AXES)), A_vals, color=colA)
axA.set_xticks(range(len(A_AXES))); axA.set_xticklabels([A_LBL[a] for a in A_AXES], fontsize=8)
axA.set_ylabel("non-sequential cells retained (≥2 TIR)")
axA.set_title("(A) Only FOLD-DEFINING axes preserve folds (seed 0)\n"
              "green = fold-defining; red = non-fold axis → transmission evicts folds")
for i, v in enumerate(A_vals):
    axA.text(i, v + 1, str(v), ha="center", fontsize=9)

axB.bar(range(len(B_lbl)), B_mean, yerr=B_err, capsize=5,
        color=["#7f7f7f", "#1f77b4", "#9467bd"])
axB.set_xticks(range(len(B_lbl))); axB.set_xticklabels(B_lbl, fontsize=8)
axB.set_ylabel("high-efficiency fold families (focthru ≥ 0.9)")
axB.set_title("(B) Quality of surfaced folds, mean ± sd over 3 seeds\n"
              "locus reliably ~2× baseline; baseline hit ZERO at one seed")
for i, v in enumerate(B_mean):
    axB.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=9)

fig.tight_layout()
out = REPO / "ns_probe" / "descriptor_summary.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
