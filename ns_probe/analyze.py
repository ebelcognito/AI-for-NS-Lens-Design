"""
Analyze the NS folding-emergence probe archive and answer the phase-0 question:

    Does any high-thinness, good-imaging archive cell get populated by a *folded*
    (multi-bounce) design, or does the good region stay all-transmission?

Produces:
  * ns_probe/results/probe_figure.png  (archive heatmap; transmission-vs-folded frontier; best-fold ray path)
  * prints a quantitative verdict
  * writes agent_context/ns_rl_research/05_probe_phase0_findings.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ns_probe import probe

FOLD_REV = 2.0          # mean_reversals >= this  => folded (multi-bounce)
TRANS_REV = 1.0         # mean_reversals <  this  => transmission-dominated
THIN_T = 45.0           # device depth T <= this  => "thin" (optics within 45u of sensor)
SPOT_GOOD = 3.0         # spot_rms <= this (world units) => "good imaging" (at flux>=floor)


def load(path):
    return json.loads(Path(path).read_text())


def is_good(c, flux_floor):
    return (c["fitness"] > -900.0 and c.get("flux", 0) >= flux_floor
            and c.get("spot_rms", 1e9) <= SPOT_GOOD)


def verdict(d):
    cells = d["cells"]
    ff = d["meta"]["flux_floor"]
    trans = [c for c in cells if c["mean_reversals"] < TRANS_REV]
    folded = [c for c in cells if c["mean_reversals"] >= FOLD_REV]
    good_trans = [c for c in trans if is_good(c, ff)]
    good_folded = [c for c in folded if is_good(c, ff)]
    thin_good_trans = [c for c in good_trans if c["T"] <= THIN_T]
    thin_good_folded = [c for c in good_folded if c["T"] <= THIN_T]

    def best(cs):
        return max(cs, key=lambda c: c["fitness"]) if cs else None

    return dict(
        n_cells=len(cells), ff=ff,
        n_trans=len(trans), n_folded=len(folded),
        n_good_trans=len(good_trans), n_good_folded=len(good_folded),
        n_thin_good_trans=len(thin_good_trans), n_thin_good_folded=len(thin_good_folded),
        best_trans=best(good_trans), best_folded=best(good_folded),
        best_thin_trans=best(thin_good_trans), best_thin_folded=best(thin_good_folded),
        best_folded_any=best([c for c in folded if c["fitness"] > -900]),
    )


def figure(d, v, out_png):
    cells = d["cells"]
    bins = d["meta"]["bc_bins"]
    (lo0, hi0), (lo1, hi1) = d["meta"]["bc_bounds"]
    ff = d["meta"]["flux_floor"]

    fig = plt.figure(figsize=(15, 4.6))

    # --- Panel A: archive heatmap (T x reversals), color = fitness, flux<floor hatched ---
    axA = fig.add_subplot(1, 3, 1)
    grid = np.full(bins, np.nan)
    for c in cells:
        if c["fitness"] > -900 and c.get("flux", 0) >= ff:
            grid[c["i0"], c["i1"]] = c["fitness"]
    # rows i0 = T bins (y), cols i1 = reversal bins (x)
    im = axA.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                    extent=[lo1, hi1, lo0, hi0])
    axA.axhline(THIN_T, color="w", ls="--", lw=1)
    axA.axvline(FOLD_REV, color="r", ls="--", lw=1)
    axA.set_xlabel("mean axial reversals  (transmission ← 0  |  folded ≥ 2 →)")
    axA.set_ylabel("device depth T  (thin ↓ small)")
    axA.set_title("Archive: best fitness per cell\n(flux ≥ floor only)")
    axA.text(0.3, THIN_T + 2, "thin", color="w", fontsize=8)
    fig.colorbar(im, ax=axA, label="fitness (= −spot_rms + flux)")

    # --- Panel B: best fitness vs T, transmission vs folded ---
    axB = fig.add_subplot(1, 3, 2)
    Tedges = np.linspace(lo0, hi0, bins[0] + 1)
    Tcent = 0.5 * (Tedges[:-1] + Tedges[1:])
    bt = np.full(bins[0], np.nan); bf = np.full(bins[0], np.nan)
    for k in range(bins[0]):
        col = [c for c in cells if c["i0"] == k and c["fitness"] > -900 and c.get("flux", 0) >= ff]
        ct = [c["fitness"] for c in col if c["mean_reversals"] < TRANS_REV]
        cf = [c["fitness"] for c in col if c["mean_reversals"] >= FOLD_REV]
        if ct: bt[k] = max(ct)
        if cf: bf[k] = max(cf)
    axB.plot(Tcent, bt, "o-", color="#1f77b4", label="transmission (rev<1)", ms=4)
    axB.plot(Tcent, bf, "s-", color="#d62728", label="folded (rev≥2)", ms=4)
    axB.axvline(THIN_T, color="k", ls="--", lw=1, alpha=0.5)
    axB.set_xlabel("device depth T  (thin = small)")
    axB.set_ylabel("best fitness in column")
    axB.set_title("Best imaging vs thinness:\ntransmission vs folded")
    axB.legend(fontsize=8)

    # --- Panel C: ray path of the best folded design (re-traced) ---
    axC = fig.add_subplot(1, 3, 3)
    target = v["best_thin_folded"] or v["best_folded"] or v["best_folded_any"]
    if target is not None:
        try:
            from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
            g = np.array(target["genotype"], float)
            with PersistentRayOpticsRunner(timeout_s=12) as runner:
                res, _ = runner.simulate(probe.build_scene(g, record=True))
            recs = [s for s in res.get("raySequences", []) if s.get("dirx", 0) > 0]
            recs.sort(key=lambda s: s["b"])
            for s in recs[-120:]:
                xs = [vv["x"] for vv in s["seq"]]; ys = [vv["y"] for vv in s["seq"]]
                rev = sum(1 for k in range(1, len(xs) - 1) if (xs[k + 1] - xs[k]) * (xs[k] - xs[k - 1]) < 0)
                col = "#d62728" if rev >= 2 else "#999999"
                axC.plot(xs, ys, color=col, lw=0.6, alpha=0.5)
            g2 = g.reshape(probe.N_ELEM, -1)
            for row in g2:
                cx, th = row[0], row[1]
                axC.axvspan(cx - th / 2, cx + th / 2, color="#4444ff", alpha=0.10)
                axC.axvline(cx, color="#4444ff", lw=0.8, alpha=0.4)
            axC.axvline(probe.X_SENSOR, color="k", lw=1.5)
            axC.set_title(f"Best folded design (re-traced)\nT={target['T']:.0f} rev={target['mean_reversals']:.1f} "
                          f"flux={target.get('flux',0):.2f} spot={target.get('spot_rms',0):.2f}")
        except Exception as exc:
            axC.text(0.1, 0.5, f"ray-path render failed:\n{exc}", fontsize=8)
        axC.set_xlabel("x (optical axis)"); axC.set_ylabel("y")
    else:
        axC.text(0.1, 0.5, "no folded cell to render", fontsize=10)
        axC.set_title("Best folded design")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[analyze] wrote {out_png}")


def fmt_design(c):
    if c is None:
        return "  (none)"
    g = np.array(c["genotype"], float).reshape(probe.N_ELEM, -1)
    lines = [f"  fitness={c['fitness']:.3f}  T={c['T']:.1f}  mean_reversals={c['mean_reversals']:.2f}  "
             f"flux={c.get('flux',0):.3f}  spot_rms={c.get('spot_rms',0):.3f}  "
             f"folded_frac={c.get('folded_frac',0):.2f}  opl_ratio={c.get('opl_ratio',0):.2f}"]
    for i, row in enumerate(g):
        lines.append(f"    elem{i}: cx={row[0]:.1f} thick={row[1]:.1f} fsag={row[2]:+.1f} "
                     f"bsag={row[3]:+.1f} n={row[4]:.2f} refl={row[5]:.2f}")
    return "\n".join(lines)


def main():
    arch = REPO / "ns_probe" / "results" / "archive.json"
    d = load(arch)
    v = verdict(d)
    m = d["meta"]

    print("=" * 78)
    print("NS FOLDING-EMERGENCE PROBE — phase-0 verdict")
    print("=" * 78)
    print(f"evals={m['evals']}  cells_filled={v['n_cells']}  sim_errors={m['sim_errors']}  "
          f"wall={m['wall_seconds']:.0f}s  flux_floor={v['ff']}")
    print(f"transmission cells (rev<1): {v['n_trans']}   folded cells (rev>=2): {v['n_folded']}")
    print(f"GOOD-imaging (flux>=floor, spot<= {SPOT_GOOD}):  "
          f"transmission={v['n_good_trans']}   folded={v['n_good_folded']}")
    print(f"GOOD-imaging AND thin (T<= {THIN_T}):  "
          f"transmission={v['n_thin_good_trans']}   folded={v['n_thin_good_folded']}")
    print("-" * 78)
    print("best GOOD transmission design:\n" + fmt_design(v["best_trans"]))
    print("best GOOD folded design:\n" + fmt_design(v["best_folded"]))
    print("best GOOD folded AND thin design:\n" + fmt_design(v["best_thin_folded"]))
    print("best folded design at any fitness:\n" + fmt_design(v["best_folded_any"]))

    out_png = REPO / "ns_probe" / "results" / "probe_figure.png"
    figure(d, v, out_png)

    # headline
    if v["n_thin_good_folded"] > 0:
        head = ("ALIVE: folded designs DID populate thin, good-imaging cells. "
                "Folding emerged as a competitive imaging strategy under thinness with no fold prescription.")
    elif v["n_good_folded"] > 0:
        head = ("PARTIAL: folded good-imaging designs exist, but not in the THIN region "
                "(T<=%.0f). Folding is reachable but did not win the compactness regime here." % THIN_T)
    else:
        head = ("NEGATIVE (this configuration): no folded design reached the good-imaging bar. "
                "The archive's good region stayed transmission-dominated.")
    print("-" * 78)
    print("HEADLINE:", head)
    return d, v, head


if __name__ == "__main__":
    main()
