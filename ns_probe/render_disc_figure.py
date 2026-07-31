"""
§6 supporting figure — objective evolution WITH optimization (the researcher's explicit ask).

Two-row slide-grade figure:
  Row 1 (fold): how the objective evolves as search proceeds, per arm.
    (a) running-max fold focthru vs eval index (from good_pool first_seen) -- the champion's
        optimization curve; markers where the first tir>=2 and first tir>=4 fold appears.
    (b) cumulative # distinct high-eff fold FAMILIES vs eval (the discovery curve; E1 metric 4).
  Row 2 (AR prism, E2b): best collimated-throughput vs eval across the tilt-range sweep --
    visual of the fold-starvation threshold (narrow tilt starves, wide tilt unlocks).

    PYTHONPATH=.:ns_probe python ns_probe/render_disc_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DISC = REPO / "ns_probe" / "results_disc"
NS_TIR, HIGH = 2, 0.9
COL = {"cmame": "#d62728", "random": "#1f77b4", "restart_ls": "#2ca02c"}
LAB = {"cmame": "A  CMA-ME", "random": "B0 random", "restart_ls": "B1 restart-LS"}


def load(p):
    return json.loads(Path(p).read_text())


def running_max_curve(good_pool, evals, key="fitness", tir_min=NS_TIR):
    """Step curve of running max `key` among folds (tir>=tir_min) vs eval index."""
    pts = sorted([(r["first_seen"], r[key]) for r in good_pool if r["tir_count"] >= tir_min])
    xs = np.arange(1, evals + 1)
    ys = np.zeros(evals)
    cur, j = 0.0, 0
    for i, e in enumerate(xs):
        while j < len(pts) and pts[j][0] <= e:
            cur = max(cur, pts[j][1]); j += 1
        ys[i] = cur
    return xs, ys


def families_curve(good_pool, evals):
    """Cumulative distinct high-eff (foc>=0.9,tir>=2) fold families vs eval index."""
    ev = sorted([(r["first_seen"], r["dom_signature"]) for r in good_pool
                 if r["tir_count"] >= NS_TIR and r["fitness"] >= HIGH])
    xs = np.arange(1, evals + 1)
    ys = np.zeros(evals, int)
    seen, j, cnt = set(), 0, 0
    for i, e in enumerate(xs):
        while j < len(ev) and ev[j][0] <= e:
            if ev[j][1] not in seen:
                seen.add(ev[j][1]); cnt += 1
            j += 1
        ys[i] = cnt
    return xs, ys


def mean_band(curves):
    Y = np.vstack(curves)
    m = Y.mean(0)
    h = 1.96 * Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros_like(m)
    return m, h


def first_event_eval(good_pool, tir_min):
    e = [r["first_seen"] for r in good_pool if r["tir_count"] >= tir_min and r["fitness"] >= HIGH]
    return min(e) if e else None


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    evals = 14000

    # ---- Row 1: fold objective evolution ----
    for mode in ["cmame", "random", "restart_ls"]:
        seeds = sorted((DISC / mode).glob("seed*/archive.json"))
        if not seeds:
            continue
        gps = [load(p) for p in seeds]
        # (a) running-max fold focthru
        curves = [running_max_curve(d["good_pool"], evals)[1] for d in gps]
        m, h = mean_band(curves)
        xs = np.arange(1, evals + 1)
        ax[0, 0].plot(xs, m, color=COL[mode], label=LAB[mode], lw=2)
        ax[0, 0].fill_between(xs, m - h, m + h, color=COL[mode], alpha=0.15)
        # (b) families discovery curve
        fcurves = [families_curve(d["good_pool"], evals)[1].astype(float) for d in gps]
        fm, fh = mean_band(fcurves)
        ax[0, 1].plot(xs, fm, color=COL[mode], label=LAB[mode], lw=2)
        ax[0, 1].fill_between(xs, fm - fh, fm + fh, color=COL[mode], alpha=0.15)
        # deep-fold first-appearance marker (median over seeds)
        deep = [first_event_eval(d["good_pool"], 4) for d in gps]
        deep = [e for e in deep if e]
        if deep:
            ax[0, 0].axvline(np.median(deep), color=COL[mode], ls=":", alpha=0.6)

    ax[0, 0].set(title="(a) fold champion: running-max focused throughput",
                 xlabel="evaluation index", ylabel="best fold focthru (tir>=2)")
    ax[0, 0].axhline(HIGH, color="k", ls="--", lw=0.8, alpha=0.5)
    ax[0, 0].text(200, HIGH + 0.005, "high-eff threshold 0.9", fontsize=8, alpha=0.7)
    ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=0.25)
    ax[0, 1].set(title="(b) discovery curve: distinct high-eff fold families",
                 xlabel="evaluation index", ylabel="# eps-distinct families (foc>=0.9)")
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=0.25)
    ax[0, 0].text(0.98, 0.02, "dotted line = median eval of first tir>=4 fold",
                  transform=ax[0, 0].transAxes, ha="right", va="bottom", fontsize=8, alpha=0.6)

    # ---- Row 2: AR tilt-range threshold ----
    ax2 = ax[1, 0]
    tcol = plt.cm.viridis(np.linspace(0.15, 0.9, 4))
    for c, tr in zip(tcol, [25, 35, 45, 50]):
        p = DISC / f"e2b_tilt{tr}" / "archive.json"
        if not p.exists():
            continue
        d = load(p)
        h = d["history"]
        e = [r["evals"] for r in h]; b = [r["best"] for r in h]
        ax2.plot(e, b, color=c, lw=2, label=f"tilt +-{tr} deg")
    ax2.set(title="(c) AR prism: best collimated throughput vs eval (tilt-range sweep)",
            xlabel="evaluation index", ylabel="best fitness")
    ax2.legend(fontsize=9, title="face-tilt bound"); ax2.grid(alpha=0.25)

    # (d) AR endpoint bar: fold-starvation threshold
    ax3 = ax[1, 1]
    trs, bests, folded = [], [], []
    for tr in [25, 35, 45, 50]:
        p = DISC / f"e2b_tilt{tr}" / "archive.json"
        if not p.exists():
            continue
        d = load(p)
        trs.append(f"+-{tr}")
        bests.append(d["history"][-1]["best"] if d["history"] else 0.0)
        folded.append(sum(1 for c in d["cells"] if c.get("tir", 0) >= 2))
    xb = np.arange(len(trs))
    ax3b = ax3.twinx()
    ax3.bar(xb - 0.2, bests, 0.4, color="#8856a7", label="best fitness")
    ax3b.bar(xb + 0.2, folded, 0.4, color="#43a2ca", label="# folded cells (tir>=2)")
    ax3.set_xticks(xb); ax3.set_xticklabels(trs)
    ax3.set(title="(d) AR endpoint: fold unlock vs face-tilt range",
            xlabel="face-tilt bound (deg)", ylabel="best fitness")
    ax3b.set_ylabel("# folded cells (tir>=2)")
    ax3.legend(loc="upper left", fontsize=8); ax3b.legend(loc="upper right", fontsize=8)

    fig.suptitle("Discovery control — objective evolution with optimization (§6)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = DISC / "disc_objective_evolution.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
