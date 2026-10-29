"""Create a compact experiment report and standalone learning-curve figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    seeds = config["seeds"]
    strategies = ("uniform", "fixed", "adaptive")
    colors = dict(uniform="#607d8b", fixed="#d68a21", adaptive="#167e68")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for strategy in strategies:
        curves = []
        for seed in seeds:
            directory = root / f"seed{seed}"
            bootstrap = json.loads((directory / "bootstrap_validation.json").read_text())
            score = np.mean([r["early"]["success"] for r in bootstrap])
            current = score
            curve = [score]
            history = json.loads((directory / "history.json").read_text())
            for row in history:
                if row["strategy"] != strategy:
                    continue
                if row["promoted"]:
                    current = row["success_rate"]
                curve.append(current)
            curves.append(curve)
        curves = np.array(curves)
        x = config["bootstrap"] + config["batch"] * np.arange(config["rounds"] + 1)
        axes[0].plot(x, 100 * curves.mean(axis=0), "o-", color=colors[strategy], label=strategy)
        axes[0].fill_between(x, 100 * curves.min(axis=0), 100 * curves.max(axis=0),
                             alpha=.10, color=colors[strategy])
    axes[0].set(title="Promoted policy: validation families", xlabel="Generated training tasks",
                ylabel="Success after 2 candidate calls (%)", ylim=(0, 105))
    axes[0].legend(frameon=False)
    names = ("bootstrap", "uniform", "fixed", "adaptive", "random", "reset")
    y = [100 * summary["test"][name]["success_rate"] for name in names]
    axes[1].bar(names, y, color=[colors.get(name, "#afb7c1") for name in names])
    for i, value in enumerate(y):
        axes[1].text(i, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    axes[1].set(title="Untouched test families: dense audit", ylabel="Success after 2 candidate calls (%)",
                ylim=(0, 105))
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(f"Verified 2-D freeform repair • {len(seeds)} training seeds • 30 µm overall RMS limit", fontsize=12)
    fig.savefig(root / "learning_curve.png", dpi=180)
    fig.savefig(root / "learning_curve.svg")
    plt.close(fig)

    lines = ["# Verified freeform repair experiment", "",
             "This is a bounded 2-D transmitting freeform-lens experiment. It tests whether new",
             "simulator-generated experience improves a small learned optical-response model.",
             "It does not establish folded-route discovery or 3-D optical-engineering competence.", "",
             "![Learning and held-out results](learning_curve.png)", "",
             "| Method | Test success, 2 candidates | Mean RMS, 2 candidates (mm) | Test success, 8 candidates |",
             "|---|---:|---:|---:|"]
    for name in (*names, "teacher"):
        m, full = summary["test"][name], summary["test_eight_calls"][name]
        lines.append(f"| {name} | {m['success_rate']:.1%} | {m['mean_rms']:.5f} | {full['success_rate']:.1%} |")
    broyden_path = root / "broyden_baseline/summary.json"
    if broyden_path.exists():
        broyden = json.loads(broyden_path.read_text())
        lines.append(f"| Finite differences + Broyden | — | — | {broyden['metrics']['success_rate']:.1%} |")
    lines += ["", "The numerical teacher needs four finite-difference probes before proposing its",
              "Newton correction. Its two-call prefix is not a completed optimization step; the",
              "eight-call column is the relevant completed-teacher baseline. All methods also",
              "receive one initial measurement. Training and verification costs are additional.", "",
              "| Adaptive compared with | Success change (percentage points) | Family-bootstrap 95% interval |",
              "|---|---:|---:|"]
    for name in ("bootstrap", "uniform", "fixed"):
        comparison = summary["adaptive_vs"][name]
        lo, hi = comparison["success_delta_95ci"]
        lines.append(f"| {name} | {100 * comparison['success_rate_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "Intervals resample whole test families, not individual perturbations, and are",
              "conditional on the fitted policies. Inspect the separate seed results as well.", "",
              "| Seed | Bootstrap test success | Adaptive test success | Selected adaptive round |",
              "|---|---:|---:|---:|"]
    for result in summary["seeds"]:
        lines.append(f"| {result['seed']} | {result['test']['bootstrap']['success_rate']:.1%} | "
                     f"{result['test']['adaptive']['success_rate']:.1%} | {result['selection']['adaptive']['round']} |")
    lines += ["", "**Verification and artifacts.** `manifest.json` fixes the configuration and source hashes;",
              "`source_snapshot/` preserves the actual code. Each seed contains complete attempted",
              "evaluations, task-generation records, training examples, checkpoints, validation",
              "history, selection decisions, independent-grid test audits, and call accounting.", "",
              "The learner never receives the solvability witness or exact corruption vector.",
              "All primary rays must satisfy the energy and route gates. Final audits use 9 fields",
              "and 31 pupil rays rather than the 3-by-5 optimization grid, with tighter branch",
              "pruning. The audit uses the same physics implementation, not an external tracer.", ""]
    audit_path = root / "audit.json"
    if audit_path.exists():
        report = json.loads(audit_path.read_text())
        lines += [f"Record/provenance/replay audit: **{'PASS' if report['passed'] else 'FAIL'}**.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return root / "REPORT.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(summarize(args.root))


if __name__ == "__main__":
    main()
