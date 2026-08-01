"""E1 discovery-control analysis: is CMA-ME more than multi-start?

Replaces analyze_disc_e1.py (collinear) and analyze_disc_prism.py (perpendicular),
which were the same 140 lines twice with a different input directory.

    python -m nslens.analysis.discovery_control collinear
    python -m nslens.analysis.discovery_control prism

NOTE ON THE INVARIANT. Unlike the QD driver, these are REPORTS: the numbers must be
identical, the prose need not be. The twins had drifted into seven cosmetic
differences -- n/a wording, the ladder lines printed in a different order, three
reworded interpretation strings -- plus one real latent bug: ci95's empty case
returned (nan, nan, nan, 0) in one twin and (nan, nan, 0, 0) in the other. Harmless
only because fmt short-circuits on nan before reading that field.

So the prose is unified here and the gate (tests/test_analysis_equivalence.py) compares
the machine-readable summary each script writes, metric by metric and seed by seed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]

MODES = ["cmame", "random", "restart_ls"]
LABEL = {"cmame": "A  CMA-ME", "random": "B0 random", "restart_ls": "B1 restart-LS"}

NS_TIR = 2        # a design counts as folded at >= 2 TIR bounces
HIGH = 0.9        # "high-efficiency" focused-throughput threshold
DEEP_TIR = 4      # "deep" fold topologies

METRIC_KEYS = [
    ("fam_high", "# high-eff fold families"),
    ("best_spot", "best fold spot (min, sharper=smaller)"),
    ("qd", "QD-score (Sum fitness / filled cells)"),
    ("coverage", "coverage (frac cells filled)"),
    ("deep", "deep-fold reach (best focthru @ tir>=4)"),
    ("e2k3", "evals-to-3-families"),
    ("e2k5", "evals-to-5-families"),
    ("n_good_fold", "# good folds seen (foc>=0.5)"),
]


@dataclass(frozen=True)
class Arena:
    """One discovery-control geometry."""

    name: str
    results_dir: Path
    title: str
    #: substantive, not cosmetic: in the prism arena the fold is forced by geometry
    note: str
    interpret: str
    summary_name: str = "e1_summary.json"


def arenas() -> Dict[str, Arena]:
    base = REPO / "ns_probe"
    return {
        "collinear": Arena(
            name="collinear",
            results_dir=base / "results_disc",
            title="E1 head-to-head (mean +/- 95% CI over seeds)",
            note="Primary metric = # eps-distinct high-eff fold families (fam_high). "
                 "Folding is OPTIONAL here -- straight transmission also reaches the sensor.",
            interpret="B0~=A -> enumeration; B1~=A>B0 -> local optimizer; "
                      "A>B1>B0 -> full QD stack.",
        ),
        "prism": Arena(
            name="prism",
            results_dir=base / "results_disc_prism",
            title="E1 head-to-head, PERPENDICULAR prism-fold "
                  "(mean +/- 95% CI over seeds)",
            note="Primary metric = # eps-distinct high-eff fold families (fam_high). "
                 "Fold FORCED by geometry -- straight transmission cannot reach the sensor.",
            interpret="B0~=A -> enumeration (geometry confines); "
                      "A>B1>B0 -> full QD stack earns it.",
        ),
    }


# ------------------------------------------------------------------ metrics
def high_fold_families(cells) -> set:
    """Distinct dominant interaction-signatures among elites with >=2 TIR and foc>=0.9.
    Counting FAMILIES rather than cells is the point: it asks how many qualitatively
    different fold topologies were found, not how many grid cells were filled."""
    return {c["dom_signature"] for c in cells
            if c.get("tir_count", 0) >= NS_TIR and c["fitness"] >= HIGH}


def evals_to_k_families(good_pool, k: int) -> Optional[float]:
    """Eval index at which the K-th DISTINCT high-eff fold family appears, or None if
    the budget ran out first. This is the search-EFFICIENCY metric."""
    seen = set()
    for r in sorted(good_pool, key=lambda r: r["first_seen"]):
        if r["tir_count"] >= NS_TIR and r["fitness"] >= HIGH:
            seen.add(r["dom_signature"])
            if len(seen) >= k:
                return r["first_seen"]
    return None


def deep_fold_reach(cells) -> float:
    """Best focused throughput among cells with >= 4 TIR bounces (rare, hard)."""
    deep = [c["fitness"] for c in cells if c.get("tir_count", 0) >= DEEP_TIR]
    return max(deep) if deep else 0.0


def metrics(d: dict) -> dict:
    cells = d["cells"]
    hi = [c for c in cells if c.get("tir_count", 0) >= NS_TIR and c["fitness"] >= HIGH]
    cov = (d["history"][-1]["coverage"] if d["history"]
           else len(cells) / np.prod(d["meta"]["bc_bins"]))
    gp = d.get("good_pool", [])
    return dict(
        fam_high=len(high_fold_families(cells)),
        best_spot=min((c["spot"] for c in hi), default=np.nan),
        qd=float(sum(c["fitness"] for c in cells)),
        coverage=cov,
        deep=deep_fold_reach(cells),
        e2k3=evals_to_k_families(gp, 3),
        e2k5=evals_to_k_families(gp, 5),
        n_good_fold=d.get("hit_rate", {}).get("n_good_fold", len(gp)),
    )


def ci95(v):
    """(mean, half-width, n_used, n_used). Entries that are None or NaN are dropped --
    e2k3/e2k5 are None when a seed never reached K families, and dropping them is what
    makes the (n/total) annotation meaningful."""
    v = np.asarray([x for x in v
                    if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, 0, 0)
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return (v.mean(), h, len(v), len(v))


def fmt(m, h, n, tot, unit="") -> str:
    if np.isnan(m):
        return f"  n/a  (0/{tot})"
    reached = f" ({n}/{tot})" if n < tot else ""
    return f"{m:7.2f} +/- {h:5.2f}{unit}{reached}"


# ------------------------------------------------------------------ report
def load_arm(arena: Arena, mode: str) -> List[dict]:
    seeds = sorted((arena.results_dir / mode).glob("seed*/archive.json"))
    return [metrics(json.loads(p.read_text())) for p in seeds]


def report(arena: Arena, *, log=print) -> dict:
    per_mode = {}
    for mode in MODES:
        per_mode[mode] = load_arm(arena, mode)
        log(f"{LABEL[mode]}: {len(per_mode[mode])} seeds")

    log(f"\n=== {arena.title} ===")
    log(arena.note + "\n")

    hdr = f"{'metric':<40}" + "".join(f"{LABEL[m]:>26}" for m in MODES)
    log(hdr)
    log("-" * len(hdr))
    for key, name in METRIC_KEYS:
        row = f"{name:<40}"
        for m in MODES:
            vals = [r[key] for r in per_mode[m]]
            row += f"{fmt(*ci95(vals), ''):>26}"
        log(row)

    a = np.array([r["fam_high"] for r in per_mode["cmame"]], float)
    b0 = np.array([r["fam_high"] for r in per_mode["random"]], float)
    b1 = np.array([r["fam_high"] for r in per_mode["restart_ls"]], float)
    log("\n=== headline test: fam_high, A (CMA-ME) vs B0 (random) ===")
    log(f"  A  fam_high per seed: {a.astype(int).tolist()}  mean {a.mean():.2f}")
    log(f"  B0 fam_high per seed: {b0.astype(int).tolist()}  mean {b0.mean():.2f}")
    log(f"  B1 fam_high per seed: {b1.astype(int).tolist()}  mean {b1.mean():.2f}")
    if len(a) and len(b0) and (a.std() + b0.std() > 0):
        u, p = mannwhitneyu(a, b0, alternative="greater")
        log(f"  Mann-Whitney U (A > B0, one-sided): U={u:.1f}  p={p:.4f}")
    else:
        log("  Mann-Whitney U: degenerate (identical/constant) -- report means directly")

    log("\n=== ladder reading ===")
    log(f"  mean fam_high:  A={a.mean():.2f}  B1={b1.mean():.2f}  B0={b0.mean():.2f}")
    qds = {m: np.mean([r["qd"] for r in per_mode[m]]) for m in MODES}
    log(f"  mean QD-score:  A={qds['cmame']:.2f}  B1={qds['restart_ls']:.2f}  "
        f"B0={qds['random']:.2f}")
    log(f"  Interpret: {arena.interpret}")

    # E2a is present only where a 50k random-draw pool was run
    pool = arena.results_dir / "random50k" / "archive.json"
    if pool.exists():
        hr = json.loads(pool.read_text())["hit_rate"]
        log("\n=== E2a: 50k random-draw hit rate (the manifold's measure) ===")
        log(f"  evals={hr['n_evals']}  valid={hr['n_valid']} "
            f"({100*hr['frac_valid']:.1f}% of draws)")
        log(f"  folded (tir>=2):        {hr['n_fold']} "
            f"({100*hr['frac_fold_of_valid']:.2f}% of valid)")
        log(f"  good folds (foc>=0.5):  {hr['n_good_fold']} "
            f"({100*hr['frac_good_fold_of_valid']:.3f}% of valid)")
        log(f"  HIGH folds (foc>=0.9):  {hr['n_high_fold']} "
            f"({100*hr['frac_high_fold_of_valid']:.3f}% of valid)")

    summary = {m: {k: [r[k] for r in per_mode[m]] for k, _ in METRIC_KEYS}
               for m in MODES}
    out = arena.results_dir / arena.summary_name
    out.write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out}")
    return summary


def main(argv=None) -> int:
    reg = arenas()
    ap = argparse.ArgumentParser(prog="python -m nslens.analysis.discovery_control")
    ap.add_argument("arena", nargs="?", default="collinear", choices=sorted(reg))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    report(reg[args.arena], log=(lambda *a: None) if args.quiet else print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
