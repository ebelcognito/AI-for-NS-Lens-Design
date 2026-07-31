"""
Score every descriptor-variant archive on a COMMON footing and ask the actual question:
does swapping the diversity axis surface MORE / BETTER non-sequential designs, with the
fold-blind focused-throughput fitness held fixed?

Each archive (results_desc_<bc>/archive.json) stores, for every retained elite, the full
descriptor set + focused-throughput (fitness) + spot + efficiency + dominant signature. So
we can compare runs that used different BC axes by re-scoring their elites identically.

Two binning-INDEPENDENT headline metrics (a run with more bins trivially gets more cells, so
these do NOT just reward fine binning):
  * distinct good fold FAMILIES  = # of distinct dominant interaction-signatures among
        retained elites that are non-sequential (>=2 TIR) AND good (focused-thru >= thr)
  * best NS design               = min spot among NS elites with focused-thru >= 0.9
Plus secondary (binning-dependent) cell tallies and a "junk fraction" of the NS region.

    python3 ns_probe/compare_descriptors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NS_TIR = 2                      # "non-sequential" := >=2 TIR bounces on the dominant path
GOOD, HIGH = 0.5, 0.9          # focused-throughput thresholds
ORDER = ["baseline", "nsfrac", "tilt", "critmargin", "locus", "robust"]


def load(bc):
    p = REPO / "ns_probe" / f"results_desc_{bc}" / "archive.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def score(d):
    cells = d["cells"]
    foc = np.array([c["fitness"] for c in cells])
    tir = np.array([c.get("tir_count", 0) for c in cells])
    ns = tir >= NS_TIR
    ns_cells = int(ns.sum())
    good_ns = ns & (foc >= GOOD)
    high_ns = ns & (foc >= HIGH)
    # binning-independent family counts (distinct dominant signatures)
    fam_good = {c["dom_signature"] for c, g in zip(cells, good_ns) if g}
    fam_high = {c["dom_signature"] for c, g in zip(cells, high_ns) if g}
    # best NS design (sharpest high-efficiency fold)
    hi = [c for c, g in zip(cells, high_ns) if g]
    champ = min(hi, key=lambda c: c["spot"]) if hi else None
    n_bins = int(np.prod(d["meta"]["bc_bins"]))
    return dict(
        bc=d["meta"].get("bc1_axis", "?"),
        cells=len(cells), n_bins=n_bins,
        cov=len(cells) / n_bins,
        best_focthru=float(foc.max()),
        ns_cells=ns_cells,
        good_ns_cells=int(good_ns.sum()),
        high_ns_cells=int(high_ns.sum()),
        fam_good=len(fam_good),
        fam_high=len(fam_high),
        junk_frac=(1.0 - good_ns.sum() / ns_cells) if ns_cells else float("nan"),
        champ=champ,
    )


def main():
    rows = []
    for bc in ORDER:
        d = load(bc)
        if d is None:
            print(f"  (skip {bc}: no archive yet)")
            continue
        rows.append(score(d))
    if not rows:
        sys.exit("no archives found")

    print("\n=== descriptor comparison (fitness = focused-throughput, held FIXED across all) ===")
    print("NS := >=2 TIR on dominant path.  'fam' = distinct fold topologies (binning-independent).\n")
    hdr = (f"{'BC1 axis':<11} {'cells':>5} {'cov%':>5} {'NS':>4} "
           f"{'good_fam':>8} {'high_fam':>8} {'good_cells':>10} {'junk%':>6} "
           f"{'best_fold_spot':>14} {'(focthru,depth)':>16}")
    print(hdr); print("-" * len(hdr))
    base = next((r for r in rows if r["bc"] == "baseline"), None)
    for r in rows:
        c = r["champ"]
        spot = f"{c['spot']:.2f}" if c else "  -- "
        fd = f"({c['fitness']:.2f},{int(c['tir_count'])})" if c else "   --   "
        print(f"{r['bc']:<11} {r['cells']:>5} {r['cov']*100:>5.0f} {r['ns_cells']:>4} "
              f"{r['fam_good']:>8} {r['fam_high']:>8} {r['good_ns_cells']:>10} "
              f"{r['junk_frac']*100:>5.0f}% {spot:>14} {fd:>16}")

    if base:
        print("\n--- vs baseline (distinct GOOD fold families, the headline number) ---")
        for r in rows:
            if r["bc"] == "baseline":
                continue
            dlt = r["fam_good"] - base["fam_good"]
            dltH = r["fam_high"] - base["fam_high"]
            print(f"  {r['bc']:<11} good families {r['fam_good']:>3} "
                  f"({dlt:+d} vs base)   high-eff families {r['fam_high']:>3} ({dltH:+d})")

    # champions per run
    print("\n--- best non-sequential design per run (sharpest fold at focthru>=0.9) ---")
    for r in rows:
        c = r["champ"]
        if not c:
            print(f"  {r['bc']:<11} (none above {HIGH})")
            continue
        print(f"  {r['bc']:<11} spot {c['spot']:.2f}  focthru {c['fitness']:.3f}  "
              f"eff {c['efficiency']:.2f}  depth {int(c['tir_count'])}TIR  "
              f"T={c['thinness_T']:.0f}  [{c['dom_signature']}]")

    # write markdown summary
    out = REPO / "ns_probe" / "descriptor_comparison.md"
    lines = ["# Descriptor comparison — which BC axis surfaces more good non-sequential designs",
             "",
             "Fitness = focused-throughput (fold-blind), **held fixed**. Only BC1 (the diversity",
             "axis) changes; BC2 = thinness throughout. NS := ≥2 TIR on the dominant path.",
             "`good` = focused-thru ≥ 0.5, `high` = ≥ 0.9. *Families* = distinct dominant",
             "interaction-signatures (binning-independent).", "",
             "| BC1 axis | cells | cov% | NS cells | good fam | high fam | good cells | junk% | best fold spot | (focthru,depth) |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        c = r["champ"]
        spot = f"{c['spot']:.2f}" if c else "--"
        fd = f"({c['fitness']:.2f}, {int(c['tir_count'])})" if c else "--"
        lines.append(f"| {r['bc']} | {r['cells']} | {r['cov']*100:.0f} | {r['ns_cells']} | "
                     f"{r['fam_good']} | {r['fam_high']} | {r['good_ns_cells']} | "
                     f"{r['junk_frac']*100:.0f}% | {spot} | {fd} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
