"""
E1 analysis for the PERPENDICULAR prism-fold discovery-control run — twin of analyze_disc_e1.py.
Reads ns_probe/results_disc_prism/{cmame,random,restart_ls}/seed*/archive.json and reports, per
arm, mean +/- 95% CI over seeds of the five metrics + Mann-Whitney U on # high-eff fold families
(A vs B0) + the E2a 50k random-hit rate. Because the descriptor/fitness are byte-identical to the
collinear fold, the same metric code applies -- only geometry differs (forced vs optional fold).

    PYTHONPATH=.:ns_probe python ns_probe/analyze_disc_prism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
DISC = REPO / "ns_probe" / "results_disc_prism"
MODES = ["cmame", "random", "restart_ls"]
LABEL = {"cmame": "A  CMA-ME", "random": "B0 random", "restart_ls": "B1 restart-LS"}
NS_TIR = 2
HIGH = 0.9


def high_fold_families(cells):
    return {c["dom_signature"] for c in cells
            if c.get("tir_count", 0) >= NS_TIR and c["fitness"] >= HIGH}


def evals_to_k_families(good_pool, k):
    seen, order = set(), sorted(good_pool, key=lambda r: r["first_seen"])
    for r in order:
        if r["tir_count"] >= NS_TIR and r["fitness"] >= HIGH:
            seen.add(r["dom_signature"])
            if len(seen) >= k:
                return r["first_seen"]
    return None


def deep_fold_reach(cells):
    deep = [c["fitness"] for c in cells if c.get("tir_count", 0) >= 4]
    return max(deep) if deep else 0.0


def metrics(d):
    cells = d["cells"]
    fams = high_fold_families(cells)
    hi = [c for c in cells if c.get("tir_count", 0) >= NS_TIR and c["fitness"] >= HIGH]
    best_spot = min((c["spot"] for c in hi), default=np.nan)
    qd = float(sum(c["fitness"] for c in cells))
    cov = d["history"][-1]["coverage"] if d["history"] else len(cells) / np.prod(d["meta"]["bc_bins"])
    gp = d.get("good_pool", [])
    return dict(
        fam_high=len(fams), best_spot=best_spot, qd=qd, coverage=cov,
        e2k3=evals_to_k_families(gp, 3), e2k5=evals_to_k_families(gp, 5),
        deep=deep_fold_reach(cells),
        n_good_fold=d.get("hit_rate", {}).get("n_good_fold", len(gp)))


def ci95(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, 0, 0)
    m = v.mean()
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return (m, h, len(v), len(v))


def fmt(m, h, n, tot, unit=""):
    if np.isnan(m):
        return f"  n/a  (0/{tot})"
    reached = f" ({n}/{tot})" if n < tot else ""
    return f"{m:7.2f} +/- {h:5.2f}{unit}{reached}"


def main():
    per_mode = {}
    for mode in MODES:
        seeds = sorted((DISC / mode).glob("seed*/archive.json"))
        per_mode[mode] = [metrics(json.loads(p.read_text())) for p in seeds]
        print(f"{LABEL[mode]}: {len(per_mode[mode])} seeds")

    print("\n=== E1 head-to-head, PERPENDICULAR prism-fold (mean +/- 95% CI over seeds) ===")
    print("Primary metric = # eps-distinct high-eff fold families (fam_high). Fold FORCED by geometry.\n")
    keys = [("fam_high", "# high-eff fold families", ""),
            ("best_spot", "best fold spot (min, sharper=smaller)", ""),
            ("qd", "QD-score (Sum fitness / filled cells)", ""),
            ("coverage", "coverage (frac cells filled)", ""),
            ("deep", "deep-fold reach (best focthru @ tir>=4)", ""),
            ("e2k3", "evals-to-3-families", ""),
            ("e2k5", "evals-to-5-families", ""),
            ("n_good_fold", "# good folds seen (foc>=0.5)", "")]
    hdr = f"{'metric':<40}" + "".join(f"{LABEL[m]:>26}" for m in MODES)
    print(hdr); print("-" * len(hdr))
    for key, name, unit in keys:
        row = f"{name:<40}"
        for m in MODES:
            row += f"{fmt(*ci95([r[key] for r in per_mode[m]]), unit):>26}"
        print(row)

    a = np.array([r["fam_high"] for r in per_mode["cmame"]], float)
    b0 = np.array([r["fam_high"] for r in per_mode["random"]], float)
    b1 = np.array([r["fam_high"] for r in per_mode["restart_ls"]], float)
    print("\n=== headline test: fam_high, A (CMA-ME) vs B0 (random) ===")
    print(f"  A  fam_high per seed: {a.astype(int).tolist()}  mean {a.mean():.2f}")
    print(f"  B0 fam_high per seed: {b0.astype(int).tolist()}  mean {b0.mean():.2f}")
    print(f"  B1 fam_high per seed: {b1.astype(int).tolist()}  mean {b1.mean():.2f}")
    if len(a) and len(b0) and (a.std() + b0.std() > 0):
        u, p = mannwhitneyu(a, b0, alternative="greater")
        print(f"  Mann-Whitney U (A > B0, one-sided): U={u:.1f}  p={p:.4f}")
    else:
        print("  Mann-Whitney U: degenerate -- report means directly")

    print("\n=== ladder reading ===")
    aqd = np.mean([r["qd"] for r in per_mode["cmame"]])
    b0qd = np.mean([r["qd"] for r in per_mode["random"]])
    b1qd = np.mean([r["qd"] for r in per_mode["restart_ls"]])
    print(f"  mean fam_high:  A={a.mean():.2f}  B1={b1.mean():.2f}  B0={b0.mean():.2f}")
    print(f"  mean QD-score:  A={aqd:.2f}  B1={b1qd:.2f}  B0={b0qd:.2f}")
    print("  Interpret: B0~=A -> enumeration (geometry confines); A>B1>B0 -> full QD stack earns it.")

    pool = DISC / "random50k" / "archive.json"
    if pool.exists():
        hr = json.loads(pool.read_text())["hit_rate"]
        print("\n=== E2a: 50k random-draw hit rate (the manifold's measure) ===")
        print(f"  evals={hr['n_evals']}  valid={hr['n_valid']} ({100*hr['frac_valid']:.1f}% of draws)")
        print(f"  folded (tir>=2):        {hr['n_fold']} ({100*hr['frac_fold_of_valid']:.2f}% of valid)")
        print(f"  good folds (foc>=0.5):  {hr['n_good_fold']} ({100*hr['frac_good_fold_of_valid']:.3f}% of valid)")
        print(f"  HIGH folds (foc>=0.9):  {hr['n_high_fold']} ({100*hr['frac_high_fold_of_valid']:.3f}% of valid)")

    out = DISC / "e1_summary.json"
    summary = {m: {k: [r[k] for r in per_mode[m]] for k, _, _ in keys} for m in MODES}
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
