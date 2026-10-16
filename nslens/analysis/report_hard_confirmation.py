"""Report the paired hard optical confirmation experiment."""

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ["reuse_only", "optical_residual", "direction_random"]
TASKS = ["field8", "aperture42"]
DONOR_GROUPS = {
    "all_donors_0_7": list(range(8)),
    "original_donors_0_5": list(range(6)),
    "unused_hard_sweep_donors_6_7": [6, 7],
}
COMPARATORS = ["reuse_only", "direction_random"]


def _job_key(job):
    return (str(job["task"]), str(job["arm"]), int(job["seed"]),
            int(job["repeat"]))


def load_results(out):
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    jobs = protocol.get("jobs", [])
    if len(jobs) != 144:
        raise AssertionError(f"Expected 144 protocol jobs, found {len(jobs)}")

    expected = {_job_key(job) for job in jobs}
    if len(expected) != len(jobs):
        raise AssertionError("Protocol contains duplicate jobs")
    rows = {}
    for key in sorted(expected):
        task, arm, seed, repeat = key
        path = out / f"{task}__{arm}__{seed}__r{repeat}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        if _job_key(row["job"]) != key:
            raise AssertionError(f"Job metadata mismatch in {path}")
        if int(row.get("coarse_calls", -1)) != 2056:
            raise AssertionError(f"Expected 2056 coarse calls in {path}")
        if int(row.get("dense_calls", -1)) != 1:
            raise AssertionError(f"Expected one dense audit in {path}")
        rows[key] = row
    missing = sorted(expected - rows.keys())
    if missing:
        raise RuntimeError(f"Experiment is incomplete: {len(missing)} of 144 jobs missing; first missing: {missing[0]}")

    # Reject unlisted result files so accidental stale outputs cannot enter the report.
    for path in out.glob("*__*__*__r*.json"):
        try:
            row = json.loads(path.read_text())
            key = _job_key(row["job"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if key not in expected:
            raise AssertionError(f"Unlisted result job in {path}")
    return protocol, rows


def _success(row):
    return int(bool(row["audit"].get("success", False)))


def _rate_entry(selected):
    successes = [_success(row) for row in selected]
    routed = [bool(row["audit"].get("routing_pass", False)) for row in selected]
    rms = [row["audit"].get("worst_mapping_rms_mm") for row in selected]
    finite_rms = [float(value) for value in rms
                  if value is not None and np.isfinite(float(value))]
    return {
        "successes": sum(successes),
        "total": len(selected),
        "rate": float(np.mean(successes)) if successes else None,
        "routing_passes": sum(routed),
        "median_worst_mapping_rms_mm": float(np.median(finite_rms)) if finite_rms else None,
        "finite_rms_count": len(finite_rms),
    }


def summarize_rates(rows):
    summary = {
        "design": {
            "tasks": TASKS,
            "arms": ARMS,
            "donor_groups": DONOR_GROUPS,
            "repeats_per_donor_task_arm": 3,
            "donors": 8,
            "jobs": 144,
            "coarse_calls_per_job": 2056,
            "dense_calls_per_job": 1,
        },
        "success_rates": {},
    }
    for group, donors in DONOR_GROUPS.items():
        summary["success_rates"][group] = {}
        for task in TASKS:
            summary["success_rates"][group][task] = {}
            for arm in ARMS:
                selected = [
                    row for (t, a, seed, _repeat), row in rows.items()
                    if t == task and a == arm and seed in donors
                ]
                summary["success_rates"][group][task][arm] = _rate_entry(selected)
        summary["success_rates"][group]["pooled_descriptive"] = {}
        for arm in ARMS:
            selected = [
                row for (_task, a, seed, _repeat), row in rows.items()
                if a == arm and seed in donors
            ]
            summary["success_rates"][group]["pooled_descriptive"][arm] = _rate_entry(selected)
    return summary


def plot_comparison(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = summary["success_rates"]["all_donors_0_7"]
    x = np.arange(len(TASKS))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, arm in enumerate(ARMS):
        values = [rates[task][arm]["rate"] for task in TASKS]
        bars = ax.bar(x + (i - 1) * width, values, width, label=arm)
        for bar, task in zip(bars, TASKS):
            item = rates[task][arm]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    (item["rate"] or 0) + 0.025,
                    f"{item['successes']}/{item['total']}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Dense-audit successes / 24 jobs")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x, TASKS)
    ax.set_title("Hard optical confirmation: all eight donors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "confirmations.png", dpi=180)
    plt.close(fig)


def paired_comparisons(rows):
    """Pair each method within donor, repeat, and task; bootstrap donor blocks."""
    result = {
        "effect": "optical_residual success fraction minus comparator success fraction",
        "bootstrap": {
            "resamples": 10000,
            "seed": 93,
            "unit": "donor; all tasks and repeats for a sampled donor stay together",
            "interval": "percentile 95%",
        },
        "groups": {},
    }
    rng = np.random.default_rng(93)
    for group, donors in DONOR_GROUPS.items():
        result["groups"][group] = {"by_task": {}, "pooled_descriptive": {}}
        for task in TASKS:
            result["groups"][group]["by_task"][task] = {}
        for comparator in COMPARATORS:
            for task in TASKS:
                pairs = []
                donor_deltas = {}
                for donor in donors:
                    donor_pairs = []
                    for repeat in range(1, 4):
                        left = rows.get((task, "optical_residual", donor, repeat))
                        right = rows.get((task, comparator, donor, repeat))
                        if left is None or right is None:
                            continue
                        delta = _success(left) - _success(right)
                        pairs.append(delta)
                        donor_pairs.append(delta)
                    if donor_pairs:
                        donor_deltas[donor] = float(np.mean(donor_pairs))
                entry = {
                    "paired_jobs": len(pairs),
                    "mean_paired_success_difference": float(np.mean(pairs)) if pairs else None,
                    "optical_wins": sum(x > 0 for x in pairs),
                    "ties": sum(x == 0 for x in pairs),
                    "optical_losses": sum(x < 0 for x in pairs),
                    "donor_block_differences": {
                        str(d): v for d, v in donor_deltas.items()
                    },
                }
                if len(donors) >= 3 and donor_deltas:
                    blocks = np.array([donor_deltas[d] for d in donors if d in donor_deltas])
                    boot = np.mean(rng.choice(blocks, size=(10000, len(blocks)), replace=True), axis=1)
                    entry["donor_block_bootstrap_95"] = np.quantile(boot, [.025, .975]).tolist()
                else:
                    entry["donor_block_bootstrap_95"] = None
                    entry["interval_note"] = "Only two donor blocks; no interval reported."
                result["groups"][group]["by_task"][task][
                    f"optical_residual_vs_{comparator}"
                ] = entry

            # The pooled descriptive point estimate weights every paired task-repeat equally.
            pooled_by_donor = {}
            pooled_pairs = []
            for donor in donors:
                donor_values = []
                for task in TASKS:
                    for repeat in range(1, 4):
                        left = rows.get((task, "optical_residual", donor, repeat))
                        right = rows.get((task, comparator, donor, repeat))
                        if left is not None and right is not None:
                            value = _success(left) - _success(right)
                            pooled_pairs.append(value)
                            donor_values.append(value)
                if donor_values:
                    pooled_by_donor[donor] = float(np.mean(donor_values))
            entry = {
                "paired_jobs": len(pooled_pairs),
                "mean_paired_success_difference": float(np.mean(pooled_pairs)) if pooled_pairs else None,
                "optical_wins": sum(x > 0 for x in pooled_pairs),
                "ties": sum(x == 0 for x in pooled_pairs),
                "optical_losses": sum(x < 0 for x in pooled_pairs),
                "donor_block_differences": {str(d): v for d, v in pooled_by_donor.items()},
                "descriptive_only": True,
                "task_families_and_repeats_are_not_independent_donor_evidence": True,
            }
            if len(donors) >= 3 and pooled_by_donor:
                blocks = np.array([pooled_by_donor[d] for d in donors if d in pooled_by_donor])
                boot = np.mean(rng.choice(blocks, size=(10000, len(blocks)), replace=True), axis=1)
                entry["donor_block_bootstrap_95"] = np.quantile(boot, [.025, .975]).tolist()
            else:
                entry["donor_block_bootstrap_95"] = None
                entry["interval_note"] = "Only two donor blocks; no interval reported."
            result["groups"][group]["pooled_descriptive"][
                f"optical_residual_vs_{comparator}"
            ] = entry
    return result


def write_results_text(out, protocol, summary, paired):
    lines = [
        "HARD OPTICAL CONFIRMATION",
        "",
        "Frozen confirmation on two task variants (field8 and aperture42), three arms, eight donors, and three repeats per donor/task/arm. The arms are reuse_only, optical_residual, and direction_random. Each of the 144 jobs is checked for exactly 2056 coarse calls and one dense audit.",
        "Success is read from the final dense audit. Methods and settings are fixed for this run; this report performs no fitting or method selection.",
        "",
        "SUCCESS COUNTS (successes/total)",
        "",
        "| Donor group | Task | Reuse only | Optical residual | Direction random |",
        "|---|---|---:|---:|---:|",
    ]
    rates = summary["success_rates"]
    for group in DONOR_GROUPS:
        for task in TASKS:
            cells = []
            for arm in ARMS:
                v = rates[group][task][arm]
                cells.append(f"{v['successes']}/{v['total']} ({v['rate']:.1%})")
            lines.append(f"| {group} | {task} | " + " | ".join(cells) + " |")
        cells = []
        for arm in ARMS:
            v = rates[group]["pooled_descriptive"][arm]
            cells.append(f"{v['successes']}/{v['total']} ({v['rate']:.1%})")
        lines.append(f"| {group} | pooled descriptive | " + " | ".join(cells) + " |")

    lines += [
        "",
        "PAIRED SUCCESS DIFFERENCES",
        "",
        "Differences are optical_residual minus comparator in success fraction, paired on donor, repeat, and task. Confidence intervals resample donors as blocks and preserve all tasks and repeats for each sampled donor. The pooled row is descriptive because tasks and repeats do not add independent donor evidence. No p-values are computed.",
        "",
    ]
    for group, entries in paired["groups"].items():
        lines.append(f"{group}:")
        for task in TASKS:
            for label, item in entries["by_task"][task].items():
                ci = item["donor_block_bootstrap_95"]
                interval = (f"95% donor-block bootstrap [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                            if ci is not None else item["interval_note"])
                lines.append(
                    f"- {task}, {label}: difference {item['mean_paired_success_difference']:+.3f}; "
                    f"{item['optical_wins']} optical wins, {item['ties']} ties, "
                    f"{item['optical_losses']} losses (n={item['paired_jobs']}); {interval}."
                )
        for label, item in entries["pooled_descriptive"].items():
            ci = item["donor_block_bootstrap_95"]
            interval = (f"95% donor-block bootstrap [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                        if ci is not None else item["interval_note"])
            lines.append(
                f"- pooled descriptive, {label}: difference {item['mean_paired_success_difference']:+.3f}; "
                f"{item['optical_wins']} optical wins, {item['ties']} ties, "
                f"{item['optical_losses']} losses (n={item['paired_jobs']}); {interval}."
            )
    lines += [
        "",
        "ROUTING AND MAPPING ERROR",
        "",
        "summary.json also reports routing-pass counts and median finite worst-field mapping RMS by task, method, and donor group. These RMS values exclude missing-ray penalties; routing counts therefore matter when interpreting a low median RMS.",
        "",
        "All-donor task details:",
        "| Task | Method | Routing passes/24 | Median finite worst-field RMS (mm) | Finite RMS n |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        for arm in ARMS:
            value = rates["all_donors_0_7"][task][arm]
            median = (f"{value['median_worst_mapping_rms_mm']:.4g}"
                      if value["median_worst_mapping_rms_mm"] is not None else "n/a")
            lines.append(f"| {task} | {arm} | {value['routing_passes']}/{value['total']} | "
                         f"{median} | {value['finite_rms_count']} |")
    lines += [
        "",
        "INTERPRETATION",
        "",
        "The original donor group contains six donors; donors 6 and 7 were unused in the earlier hard sweep. The latter subgroup has only two independent donor blocks, so no confidence interval is reported for it. The eight-donor intervals are exploratory and should be read cautiously. Repeats improve within-donor characterization but do not increase the donor sample size.",
        "",
        "PROTOCOL",
        "",
        json.dumps({k: v for k, v in protocol.items() if k != "jobs"}, indent=2),
        "",
    ]
    (out / "RESULTS.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Completed confirmation run directory")
    args = parser.parse_args()
    out = Path(args.out)
    protocol, rows = load_results(out)
    summary = summarize_rates(rows)
    paired = paired_comparisons(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, allow_nan=False) + "\n")
    write_results_text(out, protocol, summary, paired)
    plot_comparison(out, summary)


if __name__ == "__main__":
    main()
