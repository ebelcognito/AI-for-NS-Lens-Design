"""
Scatter plot: c1 (x) vs a chosen measure (y), coloured by fitness.

Usage
-----
  # Latest run, measure m2 (default):
  python optical_rl/qd/plot_singlet_param_scatter.py

  # Pick measure:
  python optical_rl/qd/plot_singlet_param_scatter.py --measure m1
  python optical_rl/qd/plot_singlet_param_scatter.py --measure m2

  # Explicit archive (still saves next to that archive unless --save is set):
  python optical_rl/qd/plot_singlet_param_scatter.py --archive path/to/archive.json

  # Also open an interactive window:
  python optical_rl/qd/plot_singlet_param_scatter.py --show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "models"


def latest_singlet_archive() -> Path:
    """Return the archive.json from the most-recently-modified singlet_cmamega run."""
    candidates = sorted(
        RESULTS_DIR.glob("singlet_cmamega_*/archive.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        sys.exit(
            f"No singlet_cmamega_*/archive.json found under {RESULTS_DIR}.\n"
            "Pass --archive <path> explicitly."
        )
    return candidates[-1]


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (c1, measures, fitness) arrays from an archive.json."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    c1_list, m1_list, m2_list, fit_list = [], [], [], []
    for row in rows:
        try:
            theta = row["theta"]
            measures = row["measures"]
            fitness = float(row["fitness"])
            if not (np.isfinite(theta) and np.all(np.isfinite(measures)) and np.isfinite(fitness)):
                continue
            c1_list.append(float(theta[0]))
            m1_list.append(float(measures[0]))
            m2_list.append(float(measures[1]))
            fit_list.append(fitness)
        except Exception:
            continue
    return (
        np.asarray(c1_list),
        np.column_stack([m1_list, m2_list]) if m1_list else np.empty((0, 2)),
        np.asarray(fit_list),
    )


MEASURE_LABELS = {
    "m1": ("m1  (Zernike defocus / vertical coma)", 0),
    "m2": ("m2  (Zernike primary spherical)", 1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="c1 vs measure scatter, coloured by fitness")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to archive.json (default: latest singlet_cmamega run)",
    )
    parser.add_argument(
        "--measure",
        choices=["m1", "m2"],
        default="m2",
        help="Which measure to plot on the y-axis (default: m2)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: <run_folder>/param_scatter_c1_vs_<measure>.png "
            "next to archive.json)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving",
    )
    args = parser.parse_args()

    archive_path = args.archive if args.archive else latest_singlet_archive()
    print(f"Loading archive: {archive_path}")

    c1, measures, fitness = load_archive(archive_path)
    if len(c1) == 0:
        sys.exit("Archive is empty or contained no valid entries.")

    measure_label, measure_idx = MEASURE_LABELS[args.measure]
    y = measures[:, measure_idx]

    # Fitness is stored as negative (minimisation), so flip sign for intuitive colouring
    # (higher = better spot quality).
    color_values = -fitness  # now higher means better

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    sc = ax.scatter(
        c1,
        y,
        c=color_values,
        cmap="plasma",
        s=6,
        alpha=0.75,
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Fitness  (higher = better)", color="#cccccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    ax.set_xlabel("c1  (front surface curvature)", color="#cccccc", fontsize=11)
    ax.set_ylabel(measure_label, color="#cccccc", fontsize=11)
    ax.set_title(
        f"Archive parameter scatter — {archive_path.parent.name}",
        color="#e0e0e0",
        fontsize=12,
        pad=10,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
    ax.yaxis.offsetText.set_color("#aaaaaa")

    ax.grid(color="#2a2a3e", linewidth=0.5, linestyle="--")

    n = len(c1)
    ax.text(
        0.01, 0.99, f"n = {n:,} individuals",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8, color="#888888",
    )

    fig.tight_layout()

    out_path = (
        args.save.resolve()
        if args.save
        else (archive_path.resolve().parent / f"param_scatter_c1_vs_{args.measure}.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
