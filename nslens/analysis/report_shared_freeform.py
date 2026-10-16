"""Report cached-pool, shared-freeform refinement comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = ["field8", "aperture42"]
ARMS = ["reuse_only", "optical_residual", "direction_random"]
REFINERS = ["last_parabolic", "joint_parabolic", "last_freeform"]
DONOR_GROUPS = {
    "all_donors_0_7": list(range(8)),
    "original_donors_0_5": list(range(6)),
    "unused_hard_sweep_donors_6_7": [6, 7],
}
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 93


def _source_job_key(job):
    return (str(job["task"]), str(job["arm"]), int(job["seed"]),
            int(job["repeat"]))


def _protocol_key(job):
    return (str(job["source"]), str(job["source_hash"]), str(job["refiner"]))


def _expected_result_path(out, job):
    source_stem = Path(job["source"]).stem
    return out / f"{source_stem}__{job['refiner']}.json"


def _validate_ledger(row, path):
    if int(row.get("additional_coarse_calls", -1)) != 1044:
        raise AssertionError(f"Expected 1044 additional coarse calls in {path}")
    if int(row.get("total_search_coarse_calls", -1)) != 3100:
        raise AssertionError(f"Expected 3100 total coarse calls in {path}")
    if int(row.get("dense_calls", -1)) != 1:
        raise AssertionError(f"Expected one dense audit in {path}")
    refinements = row.get("refinements")
    if not isinstance(refinements, dict) or len(refinements) != 4:
        raise AssertionError(f"Expected four refinement counts in {path}")
    per_count = []
    for count, result in refinements.items():
        calls = int(result.get("coarse_calls", -1))
        if calls != 261:
            raise AssertionError(
                f"Expected 261 coarse calls for count {count} in {path}, found {calls}")
        per_count.append(calls)
    if sum(per_count) != int(row["additional_coarse_calls"]):
        raise AssertionError(f"Refinement call ledger does not sum to 1044 in {path}")


def load_results(out):
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    jobs = protocol.get("jobs", [])
    if len(jobs) != 432:
        raise AssertionError(f"Expected 432 protocol jobs, found {len(jobs)}")
    expected = {_protocol_key(job) for job in jobs}
    if len(expected) != 432:
        raise AssertionError("Protocol contains duplicate jobs")
    rows = {}
    loaded = set()
    for job in jobs:
        key = _protocol_key(job)
        source, source_hash, refiner = key
        if refiner not in REFINERS:
            raise AssertionError(f"Unexpected refiner in protocol: {refiner}")
        path = _expected_result_path(out, job)
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        source_job = row.get("source_job", {})
        task, arm, seed, repeat = _source_job_key(source_job)
        if task not in TASKS or arm not in ARMS:
            raise AssertionError(f"Unexpected source job metadata in {path}")
        result_job = row.get("job", {})
        if str(result_job.get("source")) != source:
            raise AssertionError(f"Source path mismatch in {path}")
        if str(result_job.get("refiner")) != refiner:
            raise AssertionError(f"Refiner metadata mismatch in {path}")
        if str(result_job.get("source_hash")) != source_hash:
            raise AssertionError(f"Source hash mismatch in {path}")
        _validate_ledger(row, path)
        row_key = (task, arm, seed, repeat, refiner)
        if row_key in rows:
            raise AssertionError(f"Duplicate source/refiner result in {path}")
        rows[row_key] = row
        loaded.add(key)
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(
            f"Experiment is incomplete: {len(missing)} of 432 jobs missing; first missing: {missing[0]}")
    expected_names = {_expected_result_path(out, job).name for job in jobs}
    for path in out.glob("*__*.json"):
        if path.name in expected_names:
            continue
        # Ignore unrelated report products and only reject JSON files that look
        # like refiner outputs carrying this experiment's metadata.
        try:
            extra = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "source_job" in extra and "refiner" in extra and "source_hash" in extra:
            raise AssertionError(f"Unlisted refiner result in {path}")
    return protocol, rows


def _success(row):
    return int(bool(row["audit"].get("success", False)))


def _finite_values(rows, key):
    values = []
    for row in rows:
        value = row["audit"].get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _rate_entry(selected):
    successes = [_success(row) for row in selected]
    audit = [row["audit"] for row in selected]
    rms = _finite_values(selected, "worst_mapping_rms_mm")
    support = _finite_values(selected, "worst_support")
    changed = sum(int(row["selected_n"] != row["source_original_selected_n"])
                  for row in selected)
    original_passes = sum(
        bool(row.get("source_initial_audit", {}).get("success", False))
        for row in selected)
    rescued = sum(
        (not bool(row.get("source_initial_audit", {}).get("success", False)))
        and _success(row) for row in selected)
    return {
        "successes": int(sum(successes)),
        "total": len(selected),
        "rate": float(np.mean(successes)) if successes else None,
        "routing_passes": sum(bool(item.get("routing_pass", False)) for item in audit),
        "median_worst_mapping_rms_mm": float(np.median(rms)) if rms else None,
        "finite_rms_count": len(rms),
        "median_worst_support": float(np.median(support)) if support else None,
        "finite_support_count": len(support),
        "selected_count_changed": changed,
        "source_original_passes": original_passes,
        "rescued_original_failures": rescued,
    }


def summarize_rates(rows):
    summary = {
        "design": {
            "tasks": TASKS,
            "sampling_arms": ARMS,
            "refiners": REFINERS,
            "donor_groups": DONOR_GROUPS,
            "repeats_per_donor_task_arm_refiner": 3,
            "donors": 8,
            "jobs": 432,
            "additional_coarse_calls_per_job": 1044,
            "total_search_coarse_calls_per_job": 3100,
            "dense_calls_per_job": 1,
        },
        "success_rates": {},
    }
    for group, donors in DONOR_GROUPS.items():
        group_rates = summary["success_rates"][group] = {}
        for task in TASKS:
            group_rates[task] = {}
            for arm in ARMS:
                group_rates[task][arm] = {}
                for refiner in REFINERS:
                    selected = [
                        row for (t, a, seed, _repeat, r), row in rows.items()
                        if t == task and a == arm and seed in donors and r == refiner
                    ]
                    group_rates[task][arm][refiner] = _rate_entry(selected)
        group_rates["pooled_descriptive"] = {}
        for arm in ARMS:
            group_rates["pooled_descriptive"][arm] = {}
            for refiner in REFINERS:
                selected = [
                    row for (_task, a, seed, _repeat, r), row in rows.items()
                    if a == arm and seed in donors and r == refiner
                ]
                group_rates["pooled_descriptive"][arm][refiner] = _rate_entry(selected)
    return summary


def plot_comparison(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = summary["success_rates"]["all_donors_0_7"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    x = np.arange(len(REFINERS))
    for column, arm in enumerate(ARMS):
        for row_index, task in enumerate(TASKS):
            ax = axes[row_index, column]
            values = [rates[task][arm][refiner]["rate"] for refiner in REFINERS]
            bars = ax.bar(x, [v or 0 for v in values], color=["#9aa5b1", "#5a8fbd", "#4d9b72"])
            for bar, refiner in zip(bars, REFINERS):
                item = rates[task][arm][refiner]
                ax.text(bar.get_x() + bar.get_width() / 2,
                        (item["rate"] or 0) + 0.025,
                        f"{item['successes']}/{item['total']}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(f"{task} · {arm}")
            ax.set_xticks(x, ["last\nparabolic", "joint\nparabolic", "last\nfreeform"])
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylim(0, 1.15)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dense-audit success rate")
    fig.suptitle("Shared-freeform refiners on the cached confirmation pool (8 donors)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


def _paired_entry(rows, donors, left_refiner=None, right_refiner=None,
                  left_arm=None, right_arm=None, task=None, rng=None):
    per_donor = {}
    pairs = []
    for donor in donors:
        donor_values = []
        tasks = [task] if task else TASKS
        for task_value in tasks:
            for arm_value in ([left_arm] if left_arm else ARMS):
                arm_right = right_arm or arm_value
                for repeat in range(1, 4):
                    left = rows.get((task_value, arm_value, donor, repeat, left_refiner))
                    right = rows.get((task_value, arm_right, donor, repeat, right_refiner))
                    if left is not None and right is not None:
                        value = _success(left) - _success(right)
                        pairs.append(value)
                        donor_values.append(value)
        if donor_values:
            per_donor[donor] = float(np.mean(donor_values))
    entry = {
        "paired_jobs": len(pairs),
        "mean_paired_success_difference": float(np.mean(pairs)) if pairs else None,
        "left_wins": sum(x > 0 for x in pairs),
        "ties": sum(x == 0 for x in pairs),
        "left_losses": sum(x < 0 for x in pairs),
        "donor_block_differences": {str(k): v for k, v in per_donor.items()},
    }
    if len(donors) >= 3 and len(per_donor) == len(donors):
        block_values = np.array([per_donor[d] for d in donors])
        boot = np.mean(rng.choice(block_values, size=(BOOTSTRAPS, len(donors)),
                                  replace=True), axis=1)
        entry["donor_block_bootstrap_95"] = np.quantile(boot, [.025, .975]).tolist()
    else:
        entry["donor_block_bootstrap_95"] = None
        entry["interval_note"] = "Only two donor blocks; no interval reported."
    return entry


def paired_comparisons(rows):
    """Bootstrap donor blocks while retaining tasks and repeats within donors."""
    result = {
        "effect": "left method success fraction minus right method success fraction",
        "bootstrap": {
            "resamples": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "unit": "donor; tasks and repeats remain together within sampled donor",
            "interval": "percentile 95%",
        },
        "groups": {},
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for group, donors in DONOR_GROUPS.items():
        group_result = result["groups"][group] = {
            "refiners_by_sampling_arm": {},
            "sampling_arms_last_freeform": {},
        }
        for arm in ARMS:
            group_result["refiners_by_sampling_arm"][arm] = {}
            for comparator in ("joint_parabolic", "last_parabolic"):
                group_result["refiners_by_sampling_arm"][arm][
                    f"last_freeform_vs_{comparator}"] = {
                        task: _paired_entry(
                            rows, donors, "last_freeform", comparator,
                            left_arm=arm, task=task, rng=rng)
                        for task in TASKS
                    }
                group_result["refiners_by_sampling_arm"][arm][
                    f"last_freeform_vs_{comparator}"]["pooled_descriptive"] = _paired_entry(
                        rows, donors, "last_freeform", comparator,
                        left_arm=arm, rng=rng)
        for comparator in ("reuse_only", "direction_random"):
            label = f"optical_residual_vs_{comparator}"
            group_result["sampling_arms_last_freeform"][label] = {
                task: _paired_entry(
                    rows, donors, "last_freeform", "last_freeform",
                    left_arm="optical_residual", right_arm=comparator,
                    task=task, rng=rng)
                for task in TASKS
            }
            group_result["sampling_arms_last_freeform"][label][
                "pooled_descriptive"] = _paired_entry(
                    rows, donors, "last_freeform", "last_freeform",
                    left_arm="optical_residual", right_arm=comparator, rng=rng)
    return result


def _fmt_rate(value):
    rate = value["rate"]
    return f"{value['successes']}/{value['total']} ({rate:.1%})" if rate is not None else "n/a"


def _fmt_comparison(item):
    ci = item["donor_block_bootstrap_95"]
    interval = (f"95% donor-block bootstrap [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                if ci is not None else item["interval_note"])
    return (f"difference {item['mean_paired_success_difference']:+.3f}; "
            f"{item['left_wins']} wins, {item['ties']} ties, "
            f"{item['left_losses']} losses (n={item['paired_jobs']}); {interval}.")


def write_results_text(out, protocol, summary, paired):
    rates = summary["success_rates"]
    lines = [
        "CACHED-POOL SHARED-FREEFORM REFINER COMPARISON", "",
        "This comparison re-refines 144 already selected source tasks: two task families, three sampling arms, eight donors, and three repeats. Each source is run through last_parabolic, joint_parabolic, and last_freeform, for 432 jobs. Per job, the ledger requires 1,044 additional coarse calls (four counts × 261), 3,100 total search calls including 2,056 inherited calls, and one dense audit.",
        "The source pool was selected before this refiner comparison. These results measure refinement on that selected target pool and the same donor population; they do not provide new independent discovery data.",
        "",
        "SUCCESS COUNTS (successes/total)", "",
        "Each cell reports dense-audit successes and rate. Tables split by task and sampling arm.", "",
        "| Donor group | Task | Sampling arm | Last parabolic | Joint parabolic | Last freeform |",
        "|---|---|---|---:|---:|---:|",
    ]
    for group, donors in DONOR_GROUPS.items():
        for task in TASKS:
            for arm in ARMS:
                values = [rates[group][task][arm][r] for r in REFINERS]
                lines.append(f"| {group} (n={len(donors)}) | {task} | {arm} | " +
                             " | ".join(_fmt_rate(v) for v in values) + " |")
        for arm in ARMS:
            values = [rates[group]["pooled_descriptive"][arm][r] for r in REFINERS]
            lines.append(f"| {group} | pooled descriptive | {arm} | " +
                         " | ".join(_fmt_rate(v) for v in values) + " |")
    lines += ["", "PAIRED DONOR-BLOCK COMPARISONS", "",
              "Refiner effects compare last_freeform with each parabolic refiner within the same sampling arm, task, donor, and repeat. Sampling effects compare optical_residual with reuse_only or direction_random using last_freeform on both sides. Intervals resample donor blocks and keep all tasks and repeats together. Pooled estimates are descriptive. The two-donor subgroup has no interval.", ""]
    for group, entries in paired["groups"].items():
        lines.append(f"{group}:")
        for arm, comparisons in entries["refiners_by_sampling_arm"].items():
            for label, by_task in comparisons.items():
                if label == "pooled_descriptive":
                    continue
                for task, item in by_task.items():
                    if task == "pooled_descriptive":
                        continue
                    lines.append(f"- {arm}, {task}, {label}: {_fmt_comparison(item)}")
                lines.append(f"- {arm}, pooled descriptive, {label}: "
                             f"{_fmt_comparison(by_task['pooled_descriptive'])}")
        for label, by_task in entries["sampling_arms_last_freeform"].items():
            for task, item in by_task.items():
                if task != "pooled_descriptive":
                    lines.append(f"- last_freeform, {task}, {label}: {_fmt_comparison(item)}")
            lines.append(f"- last_freeform, pooled descriptive, {label}: "
                         f"{_fmt_comparison(by_task['pooled_descriptive'])}")
    lines += ["", "ROUTING AND MAPPING ERROR", "",
              "summary.json gives routing-pass counts, median finite worst-field mapping RMS, and median worst support by task, arm, refiner, and donor group. RMS summaries omit missing-ray penalties; interpret them alongside routing counts.", "",
              "All-donor routing and RMS:",
              "| Task | Sampling arm | Refiner | Routing passes/24 | Median finite worst-field RMS (mm) | Finite RMS n |",
              "|---|---|---|---:|---:|---:|"]
    for task in TASKS:
        for arm in ARMS:
            for refiner in REFINERS:
                value = rates["all_donors_0_7"][task][arm][refiner]
                median = (f"{value['median_worst_mapping_rms_mm']:.4g}"
                          if value["median_worst_mapping_rms_mm"] is not None else "n/a")
                lines.append(f"| {task} | {arm} | {refiner} | "
                             f"{value['routing_passes']}/{value['total']} | {median} | "
                             f"{value['finite_rms_count']} |")
    lines += ["", "SOURCE-POOL PASS AND SELECTED COUNT", "",
              "The source_original_passes count records the cached source audit before this refinement. It is a lower-budget starting point (2,056 inherited coarse calls) and is descriptive only. selected_count_changed counts jobs where the chosen mirror count after refinement differs from the source's original choice. rescued_original_failures counts source failures that pass the final dense audit.",
              "", "| Task | Sampling arm | Refiner | Source original passes/24 | Selected count changed/24 | Rescued original failures/24 |",
              "|---|---|---|---:|---:|---:|"]
    for task in TASKS:
        for arm in ARMS:
            for refiner in REFINERS:
                value = rates["all_donors_0_7"][task][arm][refiner]
                lines.append(f"| {task} | {arm} | {refiner} | "
                             f"{value['source_original_passes']}/{value['total']} | "
                             f"{value['selected_count_changed']}/{value['total']} | "
                             f"{value['rescued_original_failures']}/{value['total']} |")
    lines += ["", "INTERPRETATION AND LIMITS", "",
              "The latest refiner's point estimates are compared with older parabolic baselines under a lower per-job refinement budget. Treat that comparison as descriptive, not causal: the first-stage pool is cached and selected, and the per-job search budget differs from earlier work. The paired comparisons within this report use matched source jobs and the same refiner or sampling arm as specified above.",
              "The donor groups contain six earlier donors and two donors unused in the earlier hard sweep. Repeats characterize within-donor variation; they do not raise the independent donor count. This is a 2D optical model and does not establish 3D performance or include sagittal aberrations, diffraction, thickness, coatings, or manufacturing tolerances.",
              "", "PROTOCOL", "",
              json.dumps({k: v for k, v in protocol.items() if k != "jobs"}, indent=2), ""]
    (out / "RESULTS.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Completed shared-freeform run directory")
    args = parser.parse_args()
    out = Path(args.out)
    protocol, rows = load_results(out)
    summary = summarize_rates(rows)
    paired = paired_comparisons(rows)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2, allow_nan=False) + "\n")
    write_results_text(out, protocol, summary, paired)
    plot_comparison(out, summary)


if __name__ == "__main__":
    main()
