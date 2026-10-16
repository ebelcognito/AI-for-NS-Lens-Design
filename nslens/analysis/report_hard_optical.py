"""Summarize completed hard optical environment sweeps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["restart", "reuse_only", "reuse_random", "optical_residual"]
PAIRS = [("optical_residual", "reuse_only"),
         ("optical_residual", "reuse_random")]


def outcome_counts(paired, value):
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for left, right in paired:
        a, b = value(left), value(right)
        if a is None or b is None:
            continue
        if a > b:
            counts["wins"] += 1
        elif a < b:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    counts["decisive_pairs"] = counts["wins"] + counts["losses"]
    return counts


def load_results(out):
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    rows = {}
    for path in sorted(out.glob("*__*__*.json")):
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not all(key in row for key in ("job", "audit", "selected_coarse")):
            continue
        job = row["job"]
        rows[(job["task"], job["arm"], int(job["seed"]))] = row
    expected = protocol.get("jobs", [])
    missing = [job for job in expected
               if (job["task"], job["arm"], int(job["seed"])) not in rows]
    if missing:
        raise RuntimeError(f"Experiment is incomplete: {len(missing)} of "
                           f"{len(expected)} result files are missing")
    refs_path = out / "feasibility_references.json"
    refs = json.loads(refs_path.read_text()) if refs_path.is_file() else {}
    return protocol, rows, refs


def paired_summary(rows, tasks, seeds):
    output = {}
    for task in tasks:
        output[task] = {}
        for left_method, right_method in PAIRS:
            paired = []
            for seed in seeds:
                left = rows.get((task, left_method, seed))
                right = rows.get((task, right_method, seed))
                if left is not None and right is not None:
                    paired.append((left, right))
            success = outcome_counts(
                paired, lambda r: int(bool(r["audit"].get("success", False))))
            # Higher transformed merit means lower (better) final coarse loss.
            merit = outcome_counts(
                paired, lambda r: -float(r["selected_coarse"]["loss"]))
            output[task][f"{left_method}_vs_{right_method}"] = {
                "paired_runs": len(paired), "success": success,
                "final_coarse_merit": merit,
            }
    output["all_tasks_pooled_descriptive"] = {}
    for left_method, right_method in PAIRS:
        paired = [(rows[(task, left_method, seed)],
                   rows[(task, right_method, seed)])
                  for task in tasks for seed in seeds
                  if (task, left_method, seed) in rows
                  and (task, right_method, seed) in rows]
        output["all_tasks_pooled_descriptive"][
            f"{left_method}_vs_{right_method}"] = {
                "paired_runs": len(paired),
                "success": outcome_counts(
                    paired, lambda r: int(bool(r["audit"].get("success", False)))),
                "final_coarse_merit": outcome_counts(
                    paired, lambda r: -float(r["selected_coarse"]["loss"])),
            }
    return output


def plot_sweep(path, rows, tasks, seeds):
    x = np.arange(len(tasks))
    width = 0.19
    fig, ax = plt.subplots(figsize=(max(11, 1.8 * len(tasks)), 6.2))
    for i, method in enumerate(METHODS):
        fractions = []
        for task in tasks:
            found = [rows[(task, method, seed)]["audit"].get("success", False)
                     for seed in seeds if (task, method, seed) in rows]
            fractions.append(float(np.mean(found)) if found else np.nan)
        bars = ax.bar(x + (i - 1.5) * width, fractions, width, label=method)
        for bar, value in zip(bars, fractions):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Successful dense audits / runs")
    ax.set_ylim(0, 1.14)
    ax.set_xticks(x, tasks, rotation=18, ha="right")
    ax.set_title("Hard optical environment sweep: success fraction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_text(out, protocol, rows, refs, tasks, seeds, pairs):
    lines = ["Hard optical environment sweep", "",
             "Success requires the dense audit's 1 mm imaging threshold and package cap (where present).",
             "Routing pass requires at least 99% worst-field support and no sampled mirror crossings. Error-only pass means worst-field mapping RMS <= 1 mm, regardless of routing. Imaging pass is the joint routing+error condition; success additionally requires package pass.",
             "Failure columns are overlapping flags, not mutually exclusive categories.", "",
             "Pass rates and failure counts", "",
             "| Task | Method | Success | Error-only pass | Imaging pass | Routing failures | Imaging failures | Package failures | Known feasible reference |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for task in tasks:
        ref = refs.get(task, {})
        known = "yes" if ref.get("known_feasible") else "no witness found"
        checked = ref.get("checked", 0)
        for method in METHODS:
            chosen = [rows[(task, method, seed)] for seed in seeds
                      if (task, method, seed) in rows]
            n = len(chosen)
            if not n:
                continue
            audits = [r["audit"] for r in chosen]
            success = sum(bool(a.get("success")) for a in audits)
            error_only = sum(a.get("worst_mapping_rms_mm") is not None
                             and a["worst_mapping_rms_mm"] <= 1.0 for a in audits)
            imaging_pass = sum(bool(a.get("imaging_pass", {}).get("1.0"))
                               for a in audits)
            route_fail = sum(not bool(a.get("routing_pass")) for a in audits)
            imaging_fail = sum(not bool(a.get("imaging_pass", {}).get("1.0"))
                               for a in audits)
            package_fail = sum(not bool(a.get("package_pass", True)) for a in audits)
            lines.append(f"| {task} | {method} | {success}/{n} ({success/n:.1%}) | "
                         f"{error_only}/{n} | {imaging_pass}/{n} | {route_fail}/{n} | "
                         f"{imaging_fail}/{n} | {package_fail}/{n} | "
                         f"{known} ({checked} references) |")
            if task in ("compact160", "compact120"):
                rp = [not bool(a.get("routing_pass")) for a in audits]
                ip = [not bool(a.get("imaging_pass", {}).get("1.0")) for a in audits]
                pp = [not bool(a.get("package_pass", True)) for a in audits]
                lines.append(f"  - {method} overlapping compact failure intersections: "
                             f"routing∩package={sum(r and p for r, p in zip(rp, pp))}/{n}; "
                             f"imaging∩package={sum(i and p for i, p in zip(ip, pp))}/{n}; "
                             f"routing∩imaging={sum(r and i for r, i in zip(rp, ip))}/{n}.")
    lines += ["", "Paired optical_residual comparisons", "",
              "For binary success, a win is success by optical_residual and failure by its paired comparator; for coarse merit, a win is lower selected_coarse loss. Ties are exact.",
              "Donors are paired by seed within each task. The six task families share specifications and donor priors, so counts across task families are correlated; no significance claim is made.", ""]
    for task in tasks:
        for label, counts in pairs.get(task, {}).items():
            lines.append(f"- {task}, {label}: success {counts['success']['wins']}/"
                         f"{counts['success']['ties']}/{counts['success']['losses']} wins/ties/losses "
                         f"(n={counts['paired_runs']}); "
                         f"coarse merit {counts['final_coarse_merit']['wins']}/"
                         f"{counts['final_coarse_merit']['ties']}/"
                         f"{counts['final_coarse_merit']['losses']} wins/ties/losses.")
    for label, counts in pairs.get("all_tasks_pooled_descriptive", {}).items():
        lines.append(f"- Pooled descriptive {label}: success "
                     f"{counts['success']['wins']}/{counts['success']['ties']}/"
                     f"{counts['success']['losses']} wins/ties/losses; coarse merit "
                     f"{counts['final_coarse_merit']['wins']}/"
                     f"{counts['final_coarse_merit']['ties']}/"
                     f"{counts['final_coarse_merit']['losses']} wins/ties/losses. "
                     "These pooled pairs are correlated across task families and are descriptive only.")
    budget = protocol.get("budget", "Budget details unavailable.")
    init = protocol.get("initialization", "Initialization details unavailable.")
    residual = protocol.get("residual", "Residual method details unavailable.")
    objective = protocol.get("objective", "Objective details unavailable.")
    limits = protocol.get("enclosure", "") + " " + protocol.get("scope", "")
    lines += ["", "Budget and method", "", str(budget), str(init), str(residual), str(objective),
              "Source-control donors carry prior development cost that is not counted in online run budgets; the restart arm receives no such prior. Interpret transferred-arm advantages with that caveat.", "",
              "Model limits", "", limits.strip(),
              "The enclosure is an axis-aligned 2D packaging proxy for ideal finite parabolas and a target-span detector segment. It does not clip rays at a finite detector and excludes mounts, thickness, clearance, 3D, diffraction, coatings, and tolerances.", ""]
    (out / "RESULTS.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Completed experiment directory")
    args = parser.parse_args()
    out = Path(args.out)
    protocol, rows, refs = load_results(out)
    task_names = list(protocol.get("tasks", {}))
    seeds = sorted({int(job["seed"]) for job in protocol.get("jobs", [])})
    pairs = paired_summary(rows, task_names, seeds)
    (out / "paired_comparisons.json").write_text(
        json.dumps(pairs, indent=2, allow_nan=False) + "\n")
    make_text(out, protocol, rows, refs, task_names, seeds, pairs)
    plot_sweep(out / "sweep.png", rows, task_names, seeds)


if __name__ == "__main__":
    main()
