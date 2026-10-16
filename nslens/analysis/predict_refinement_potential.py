"""Leave-one-donor-out prediction of final freeform refinement quality."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


DEFAULT_SOURCE = Path("experiments/shared_freeform_comparison_20261006")
COUNTS = (3, 4, 5, 6)


def finite_positive(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def load_rows(source):
    rows, excluded = [], []
    for path in sorted(source.glob("*__last_freeform.json")):
        try:
            obj = json.loads(path.read_text())
            job = obj["source_job"]
            donor = int(job["seed"])
            task, arm, repeat = str(job["task"]), str(job["arm"]), int(job["repeat"])
            for n in COUNTS:
                entry = obj["refinements"][str(n)]
                initial = finite_positive(entry["initial"]["worst_mapping_rms_mm"])
                final = finite_positive(entry["coarse"]["worst_mapping_rms_mm"])
                comp = entry.get("compatibility") or {}
                unexplained = finite_positive(comp.get("rms_unexplained_slope"))
                required = finite_positive(comp.get("rms_required_slope_correction"))
                ratio = unexplained / required if unexplained is not None and required is not None else None
                vals = (initial, final, unexplained, required, ratio)
                if any(v is None or not math.isfinite(v) for v in vals):
                    excluded.append({"file": path.name, "n": n, "reason": "missing_or_nonfinite_positive_measure"})
                    continue
                rows.append(dict(file=path.name, task=task, arm=arm, donor=donor, repeat=repeat, n=n,
                                 initial_rms_mm=initial, final_rms_mm=final,
                                 rms_unexplained_slope=unexplained,
                                 rms_required_slope_correction=required, slope_ratio=ratio))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            excluded.append({"file": path.name, "reason": f"invalid_record:{type(exc).__name__}"})
    return rows, excluded


def ridge_predict(x_train, y_train, x_test, alpha=10.0):
    """Ridge after train-only standardization; intercept is not penalized."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    a = (x_train - mean) / scale
    b = (x_test - mean) / scale
    design = np.column_stack((np.ones(len(a)), a))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.column_stack((np.ones(len(b)), b)) @ beta


def linear_predict(x_train, y_train, x_test):
    """Train-fold calibrated univariate comparison, with an intercept."""
    design = np.column_stack((np.ones(len(x_train)), x_train))
    beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return np.column_stack((np.ones(len(x_test)), x_test)) @ beta


def metric_summary(y, pred, initial_log):
    rho = spearmanr(y, pred).statistic if len(y) > 1 else float("nan")
    improvement = initial_log - y
    predicted_improvement = initial_log - pred
    irho = (spearmanr(improvement, predicted_improvement).statistic
            if len(y) > 1 and np.ptp(predicted_improvement) > 0 else float("nan"))
    rho = float(rho) if math.isfinite(float(rho)) else None
    irho = float(irho) if math.isfinite(float(irho)) else None
    return {"heldout_log_mae": float(np.mean(np.abs(y - pred))),
            "spearman_final_quality": rho,
            "spearman_improvement": irho}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, excluded = load_rows(args.source)
    if not rows:
        raise SystemExit("no usable rows")

    tasks = sorted({r["task"] for r in rows})
    donors = sorted({r["donor"] for r in rows})
    y = np.log(np.array([r["final_rms_mm"] for r in rows]))
    log_initial = np.log(np.array([r["initial_rms_mm"] for r in rows]))
    log_unexplained = np.log(np.array([r["rms_unexplained_slope"] for r in rows]))
    log_required = np.log(np.array([r["rms_required_slope_correction"] for r in rows]))
    log_ratio = np.log(np.array([r["slope_ratio"] for r in rows]))
    task_cols = np.column_stack([[float(r["task"] == t) for r in rows] for t in tasks[1:]]) if len(tasks) > 1 else np.empty((len(rows), 0))
    base_x = np.column_stack((log_initial, np.array([r["n"] for r in rows], dtype=float), task_cols))
    aug_x = np.column_stack((base_x, log_unexplained, log_required, log_ratio))

    predictions = {name: np.full(len(rows), np.nan) for name in ("raw_initial", "compatibility_unexplained", "ridge_baseline", "ridge_augmented")}
    for donor in donors:
        test = np.array([r["donor"] == donor for r in rows])
        train = ~test
        assert not ({r["donor"] for i, r in enumerate(rows) if train[i]} & {r["donor"] for i, r in enumerate(rows) if test[i]})
        predictions["raw_initial"][test] = log_initial[test]
        predictions["compatibility_unexplained"][test] = linear_predict(log_unexplained[train], y[train], log_unexplained[test])
        predictions["ridge_baseline"][test] = ridge_predict(base_x[train], y[train], base_x[test])
        predictions["ridge_augmented"][test] = ridge_predict(aug_x[train], y[train], aug_x[test])

    n3_mask = np.array([r["n"] == 3 for r in rows])
    prediction_metrics = {
        "all_counts": {name: metric_summary(y, pred, log_initial) for name, pred in predictions.items()},
        "n3": {name: metric_summary(y[n3_mask], pred[n3_mask], log_initial[n3_mask])
               for name, pred in predictions.items()},
    }
    summary = {"source": str(args.source), "rows": len(rows), "excluded_count": len(excluded),
               "excluded": excluded, "donors": donors, "tasks": tasks,
               "protocol": "Leave-one-donor-out across 8 shared donor families; fixed ridge alpha=10; train-fold feature scaling; intercept unpenalized; no tuning.",
               "prediction_metrics": prediction_metrics}

    methods = ["raw_initial", "compatibility_unexplained", "ridge_baseline", "ridge_augmented"]
    scopes = {"n3": lambda r: r["n"] == 3, "all_counts": lambda r: True}
    selection = {}
    donor_selection = []
    for scope, keep in scopes.items():
        idxs = [i for i, r in enumerate(rows) if keep(r)]
        groups = {}
        for i in idxs:
            r = rows[i]
            groups.setdefault((r["task"], r["donor"], r["n"]), []).append(i)
        outcomes = {m: [] for m in methods + ["initial_rms_baseline", "random_expected"]}
        by_donor = {}
        group_count = 0
        for (task, donor, n), ids in sorted(groups.items()):
            if len(ids) != 9:
                raise ValueError(f"expected 9 arm/repeat candidates in {(task, donor, n)}, got {len(ids)}")
            group_count += 1
            truth = y[ids]
            oracle = float(np.min(truth))
            chosen_by_method = {}
            for method in methods:
                selected = ids[int(np.argmin(predictions[method][ids]))]
                val = float(y[selected])
                selected_rms = float(rows[selected]["final_rms_mm"])
                outcomes[method].append((val, val - oracle, selected_rms, selected_rms - math.exp(oracle)))
                chosen_by_method[method] = (rows[selected]["arm"], rows[selected]["repeat"], float(rows[selected]["final_rms_mm"]))
            ib = ids[int(np.argmin(log_initial[ids]))]
            iv = float(y[ib]); initial_rms = float(rows[ib]["final_rms_mm"])
            outcomes["initial_rms_baseline"].append((iv, iv - oracle, initial_rms, initial_rms - math.exp(oracle)))
            random_rms = float(np.mean(np.exp(truth)))
            outcomes["random_expected"].append((float(np.mean(truth)), float(np.mean(truth) - oracle), random_rms, random_rms - math.exp(oracle)))
            bucket = by_donor.setdefault(donor, {m: [] for m in outcomes})
            for m in outcomes:
                bucket[m].append(outcomes[m][-1])
            donor_selection.append({"scope": scope, "task": task, "donor": donor, "n": n,
                                    "selected": chosen_by_method,
                                    "oracle_final_rms_mm": float(np.exp(oracle))})
        selection[scope] = {"groups": group_count, "methods": {
            m: {"mean_selected_final_rms_mm": float(np.mean([x[2] for x in values])),
                "mean_oracle_regret_log": float(np.mean([x[1] for x in values])),
                "mean_oracle_regret_rms_mm": float(np.mean([x[3] for x in values]))}
            for m, values in outcomes.items()},
            "per_donor": {str(d): {m: {"mean_selected_final_rms_mm": float(np.mean([x[2] for x in vals])),
                                        "mean_oracle_regret_log": float(np.mean([x[1] for x in vals]))}
                                   for m, vals in dm.items()} for d, dm in by_donor.items()}}
    summary["selection"] = selection
    summary["per_donor_selection_groups"] = donor_selection

    with (args.out / "predictions.csv").open("w", newline="") as f:
        fields = ["source", "n", "donor", "task", "arm", "repeat", "initial_rms_mm", "final_rms_mm",
                  "rms_unexplained_slope", "rms_required_slope_correction", "slope_ratio"] + [f"pred_log_final_{m}" for m in methods]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            outrow = {k: v for k, v in row.items() if k != "file"}
            outrow["source"] = row["file"]
            for m in methods:
                outrow[f"pred_log_final_{m}"] = float(predictions[m][i])
            writer.writerow(outrow)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
