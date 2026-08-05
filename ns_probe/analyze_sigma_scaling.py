"""The single plot for the sigma-scaling experiment: how many good designs each arm found.

THE METRIC. x = a spot-RMS bar s (log, mm). y = the number of archive cells whose elite
passes the flux floor AND focuses to spot <= s. Read it as "how many distinct designs
this good or better did the search find", swept over every definition of "this good" so
no single threshold has to be defended.

WHY NOT FITNESS. In the feasible region the probe's fitness collapses to `1 - spot_rms`
exactly: every flux-passing unfolded cell in the frozen archive has flux == 1.0000, so
the flux term is constant and corr(fitness, -spot_rms) == 1.000. Plotting a fitness
threshold would therefore plot blur in disguise, on an axis nobody can read, and -- worse
-- a window of fitness >= 0 contains ONLY unfolded designs (the best folded cell in the
frozen archive scores -4.32). Improving folded designs from 4.4 mm to 1.5 mm of blur, a
large optical win, would not move a fitness-thresholded curve at all. Spot RMS in mm
spans the whole archive, so progress anywhere is visible.

WHY CELLS AND NOT SAMPLES. y counts occupied archive cells, so two designs only both
score if they sit in different (device-depth, axial-reversal) behaviour cells. Finding
thirty variations of one layout scores 1. That is the "diverse AND good" reading.

The folded series is broken out separately because the unfolded region is near its
ceiling (17 of 24 depth bins were already good in the frozen run) while the folded region
-- the actual research target -- had nothing feasible at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLOOR_FITNESS = -900.0   # below this the design never cleared the flux floor
FOLDED_REVS = 2.0        # BC2 >= 2 axial reversals == a genuinely folded design
ARM_STYLE = {
    "isotropic":  dict(color="#B4413C", label="isotropic (baseline)"),
    "normalized": dict(color="#1F77B4", label="normalized (per-parameter)"),
}


def load_runs(root: Path):
    """-> {arm: [ (spot, revs) arrays per seed ]}, keeping only flux-passing cells."""
    out: dict[str, list] = {}
    for d in sorted(root.glob("*_seed*")):
        f = d / "archive.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        arm = payload["meta"].get("param_scaling", "isotropic")
        cells = [c for c in payload["cells"] if c["fitness"] > FLOOR_FITNESS]
        spot = np.array([c["spot_rms"] for c in cells], dtype=float)
        revs = np.array([c["mean_reversals"] for c in cells], dtype=float)
        out.setdefault(arm, []).append((spot, revs))
    return out


def profile(spot: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Count of cells with spot <= s, for each s in grid."""
    return np.array([(spot <= s).sum() for s in grid], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str,
                    default=str(REPO / "ns_probe" / "results_sigma_scaling"))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.results)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no archives under {root}")

    grid = np.logspace(np.log10(0.05), np.log10(50.0), 220)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    summary = {}
    for arm, seeds in runs.items():
        style = ARM_STYLE.get(arm, dict(color="grey", label=arm))
        allc = np.stack([profile(s, grid) for s, _ in seeds])
        fold = np.stack([profile(s[r >= FOLDED_REVS], grid) for s, r in seeds])

        med, lo, hi = np.median(allc, 0), np.percentile(allc, 25, 0), np.percentile(allc, 75, 0)
        ax.plot(grid, med, lw=2.4, color=style["color"], label=f"{style['label']}  — all")
        ax.fill_between(grid, lo, hi, color=style["color"], alpha=0.16, lw=0)
        ax.plot(grid, np.median(fold, 0), lw=2.0, ls="--", color=style["color"],
                label=f"{style['label']}  — folded only (≥{FOLDED_REVS:g} reversals)")

        summary[arm] = dict(
            n_seeds=len(seeds),
            good_at_1mm=float(np.median(allc[:, np.searchsorted(grid, 1.0)])),
            good_at_0p2mm=float(np.median(allc[:, np.searchsorted(grid, 0.2)])),
            folded_at_1mm=float(np.median(fold[:, np.searchsorted(grid, 1.0)])),
            folded_any=float(np.median(fold[:, -1])),
            best_spot=float(np.median([s.min() for s, _ in seeds])),
        )

    ax.set_xscale("log")
    ax.set_xlabel("spot-RMS bar  $s$  (mm, log)  —  stricter ←→ looser")
    ax.set_ylabel("archive cells with spot ≤ $s$   (distinct good designs)")
    ax.set_title("More good designs per unit of search:\nper-parameter vs shared CMA step scaling",
                 fontsize=13, loc="left")
    ax.grid(alpha=0.25, which="both", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.margins(x=0)
    fig.text(0.99, 0.01, "median over seeds; band = IQR. solid = all designs, dashed = folded only",
             ha="right", fontsize=8, color="#555")
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "sigma_scaling.png"
    fig.savefig(out, dpi=170)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
