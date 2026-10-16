"""Summarize the selected-layout freeform refinement experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nslens.analysis import final_freeform as ff


ARMS = ("last_parabolic", "last_freeform", "joint_parabolic", "joint_freeform")
SEEDS = (91000000, 91000001, 91000002)
CHECKPOINTS = ("256", "1024")
SOURCE = Path("experiments/hard_confirmation_20261006/field8__optical_residual__1__r1.json")


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_range(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values:
        return None
    return {"median": float(np.median(values)), "min": float(np.min(values)),
            "max": float(np.max(values))}


def mapping_from_fields(rows):
    """Recover RMS mapping error from per-field spot RMS and centroid error."""
    values = []
    for row in rows:
        spot, centroid = row.get("rms_spot_mm"), row.get("centroid_error_mm")
        if spot is not None and centroid is not None:
            values.append(float(np.hypot(spot, centroid)))
    return values


def summarize_run(path):
    result = read_json(path)
    if result.get("coarse_calls") != 1029 or result.get("dense_calls") != 2:
        raise ValueError(f"Unexpected call counts in {path}: "
                         f"{result.get('coarse_calls')} coarse, {result.get('dense_calls')} dense")
    if set(result.get("checkpoints", {})) != set(CHECKPOINTS):
        raise ValueError(f"Missing checkpoint audits in {path}")
    checks = {}
    for checkpoint in CHECKPOINTS:
        audit = result["checkpoints"][checkpoint]["audit"]
        checks[checkpoint] = {
            "mapping_rms_mm": audit.get("worst_mapping_rms_mm"),
            "support": audit.get("worst_support"),
            "crossing": bool(audit.get("crossing")),
            "coefficients_mm": list(audit.get("freeform_coefficients_mm", [])),
            "max_higher_order_departure_mm": audit.get("max_freeform_departure_mm"),
            "max_added_slope": audit.get("max_added_slope"),
        }
    coeff = checks["1024"]["coefficients_mm"]
    bound_hit = any(abs(float(c)) >= 2.0-1e-6 for c in coeff)
    slope = result["slope_compatibility"]
    return {"arm": result["job"]["arm"], "seed": result["job"]["seed"],
            "seconds": result.get("seconds"), "checkpoints": checks,
            "freeform_coefficient_boundary_hit": bool(bound_hit),
            "slope_fit": {
                "rms_required_slope_correction": slope.get("rms_required_slope_correction"),
                "rms_unexplained_slope": slope.get("rms_unexplained_slope"),
                "sample_count": len(slope.get("samples", [])),
                "meaning": slope.get("meaning"),
            }}


def aggregate(runs):
    groups = {}
    for arm in ARMS:
        arm_runs = [r for r in runs if r["arm"] == arm]
        if len(arm_runs) != 3:
            raise ValueError(f"Expected three seeds for {arm}; got {len(arm_runs)}")
        checkpoint_summary = {}
        for c in CHECKPOINTS:
            checkpoint_summary[c] = {
                "worst_mapping_rms_mm": finite_range(
                    r["checkpoints"][c]["mapping_rms_mm"] for r in arm_runs),
                "worst_support": finite_range(r["checkpoints"][c]["support"] for r in arm_runs),
                "crossing_count": sum(r["checkpoints"][c]["crossing"] for r in arm_runs),
                "freeform_coefficient_boundary_hits": sum(
                    any(abs(float(x)) >= 2.0-1e-6 for x in r["checkpoints"][c]["coefficients_mm"])
                    for r in arm_runs),
            }
        groups[arm] = {
            "seeds": [r["seed"] for r in arm_runs],
            "checkpoints": checkpoint_summary,
            "slope_fit_rms_required_correction": finite_range(
                r["slope_fit"]["rms_required_slope_correction"] for r in arm_runs),
            "slope_fit_rms_unexplained": finite_range(
                r["slope_fit"]["rms_unexplained_slope"] for r in arm_runs),
        }
    return groups


def layout_geometry(genes, n, coeff, task, fields=5, pupils=13):
    result, model, outlines, paths, good = ff.trace(
        genes, n, coeff, fields=fields, pupils=pupils, paths=True, **task)
    return result, model, outlines, paths, good


def make_figure(out, source, final_result, dense_initial):
    n, task = source["selected_n"], source["task"]
    initial_genes = np.asarray(source["selected_coarse"]["genes"], dtype=float)
    final_audit = final_result["checkpoints"]["1024"]["audit"]
    final_genes = np.asarray(final_audit["genes"], dtype=float)
    final_coeff = np.asarray(final_audit["freeform_coefficients_mm"], dtype=float)
    zero = np.zeros(4)
    before = layout_geometry(initial_genes, n, zero, task)
    after = layout_geometry(final_genes, n, final_coeff, task)
    all_xy = [p for _, _, outlines, paths, _ in (before, after)
              for p in outlines + [np.asarray(path) for path in paths if len(path)] if len(p)]
    joined = np.vstack(all_xy)
    lo, hi = joined.min(axis=0), joined.max(axis=0)
    pad = np.maximum((hi-lo)*0.04, 5.)
    limits = (lo-pad, hi+pad)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.4, 0.8))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    labels = ((f"Selected layout before refinement — dense RMS {dense_initial['worst_mapping_rms_mm']:.3f} mm", before),
              (f"Seed 91000000: last_freeform at 1024 — dense RMS {final_audit['worst_mapping_rms_mm']:.3f} mm", after))
    for ax, (title, data), final in zip(axes, labels, (False, True)):
        result, model, outlines, paths, good = data
        for j, points in enumerate(outlines):
            ax.plot(points[:, 0], points[:, 1], color=("#c33" if final and j == n-1 else "#46566b"),
                    linewidth=(2.3 if j == n-1 else 1.3), zorder=3)
        for path, valid in zip(paths, good):
            if len(path) < 2:
                continue
            p = np.asarray(path)
            ax.plot(p[:, 0], p[:, 1], color="#2685a6" if valid else "#aaa",
                    alpha=0.38 if valid else 0.20, linewidth=0.55, zorder=1)
        target_span = task["focal"]*np.tan(np.deg2rad(task["half_field_deg"])) + 1.
        sensor = np.array([model["image"]-target_span*model["image_tangent"],
                           model["image"]+target_span*model["image_tangent"]])
        ax.plot(sensor[:, 0], sensor[:, 1], color="black", linewidth=2.0,
                label="detector target span")
        dense_support = (dense_initial if not final else final_audit)["worst_support"]
        ax.set_title(f"{title}\n15 fields × 101 pupils; support {dense_support:.3f}")
        ax.set_xlim(limits[0][0], limits[1][0]); ax.set_ylim(limits[0][1], limits[1][1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.grid(alpha=0.18)
        ax.annotate("last mirror", xy=outlines[-1][len(outlines[-1])//2],
                    xytext=(8, 10), textcoords="offset points", fontsize=8,
                    color="#a22" if final else "#334")

    profile_ax = fig.add_subplot(grid[1, 0])
    u = np.linspace(-30, 30, 401)
    z = u/30.
    higher_order = sum(c*z**k for k, c in enumerate(final_coeff, 3))
    curvature_change = (after[1]["curves"][-1]-before[1]["curves"][-1])*u*u
    total_change = curvature_change+higher_order
    profile_ax.plot(u, higher_order, color="#c33", linewidth=2,
                    label="higher-order departure")
    profile_ax.plot(u, total_change, color="#394f7b", linewidth=1.7, linestyle="--",
                    label="total sag change, including curvature")
    profile_ax.axhline(0, color="#555", linewidth=.7)
    profile_ax.set_title("Added higher-order sag on the final mirror")
    profile_ax.set_xlabel("local coordinate u (mm)"); profile_ax.set_ylabel("departure from refined parabola (mm)")
    profile_ax.grid(alpha=.22); profile_ax.legend(frameon=False)

    rms_ax = fig.add_subplot(grid[1, 1])
    dense = dense_initial
    field_x, initial_rms, refined_rms = [], [], []
    for row0, row1 in zip(dense["fields"], final_audit["fields"]):
        m0, m1 = mapping_from_fields([row0]), mapping_from_fields([row1])
        if m0 and m1:
            field_x.append(float(row0["field_deg"]))
            initial_rms.append(m0[0]); refined_rms.append(m1[0])
    rms_ax.plot(field_x, initial_rms, "o-", label="initial")
    rms_ax.plot(field_x, refined_rms, "o-", label="refined")
    rms_ax.set_title("15-field mapping RMS")
    rms_ax.set_xlabel("field (degrees)"); rms_ax.set_ylabel("RMS mapping error (mm)")
    rms_ax.grid(alpha=.22); rms_ax.legend(frameon=False)
    fig.suptitle("Single selected five-mirror layout: before and after freeform refinement", fontsize=15)
    fig.savefig(out/"before_after.png", dpi=170)
    plt.close(fig)


def markdown_report(summary):
    lines = ["# Selected-layout freeform refinement results", "",
             "This is one selected 2D optical layout, not a population-level test of a potential predictor. "
             "The audits report mapping performance after refinement; no layout pass/fail verdict is assigned.", "",
             f"Initial selected layout dense worst mapping RMS: **{summary['initial_worst_mapping_rms_mm']:.3f} mm**.", "",
             "Each seed used 1,029 coarse trace calls (including five initialization traces) and two dense audits. "
             "Dense checkpoints used 15 fields × 101 pupils.", "",
             "| Arm | Checkpoint | Worst mapping RMS, median [min, max] mm | Worst support, median [min, max] | Crossings | ±2 mm higher-order coefficient hits |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for c in CHECKPOINTS:
            s = summary["arms"][arm]["checkpoints"][c]
            m, support = s["worst_mapping_rms_mm"], s["worst_support"]
            lines.append(f"| {arm} | {c} | {m['median']:.4f} [{m['min']:.4f}, {m['max']:.4f}] | "
                         f"{support['median']:.3f} [{support['min']:.3f}, {support['max']:.3f}] | "
                         f"{s['crossing_count']}/3 | {s['freeform_coefficient_boundary_hits']}/3 |")
    lines.extend(["", "The coefficient-hit column counts runs with any higher-order final-mirror coefficient at its ±2 mm bound. "
                  "Zero coefficients in parabolic arms do not count as hits.", "",
                  "## Frozen-hit slope fit", "",
                  "The initialization fit compares desired slope changes at the original ray-hit locations. "
                  "Its residual is a frozen-hit compatibility diagnostic only: changing the surface moves those intersections, "
                  "so the fit is not a bound on attainable imaging quality.", "",
                  "| Arm | Required correction RMS, median [range] | Unexplained RMS, median [range] |",
                  "|---|---:|---:|"])
    for arm in ARMS:
        g = summary["arms"][arm]
        a, b = g["slope_fit_rms_required_correction"], g["slope_fit_rms_unexplained"]
        lines.append(f"| {arm} | {a['median']:.5f} [{a['min']:.5f}, {a['max']:.5f}] | "
                     f"{b['median']:.5f} [{b['min']:.5f}, {b['max']:.5f}] |")
    lines.extend(["", "The figure compares the selected initial layout with seed 91000000's last_freeform 1024-evaluation audit. "
                  "Both ray-layout panels use the same axis limits; their titles use the 15-field × 101-pupil dense audits. "
                  "The sag plot separates the added cubic-through-sixth-order departure from the total change, which also "
                  "includes the quadratic-curvature change.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="completed final_freeform_20261006 experiment directory")
    args = parser.parse_args()
    out = args.out
    expected = {f"{arm}__{seed}.json" for arm in ARMS for seed in SEEDS}
    missing = sorted(name for name in expected if not (out/name).is_file())
    if missing:
        raise SystemExit("Missing 12-run experiment results: " + ", ".join(missing))
    runs = [summarize_run(out/f"{arm}__{seed}.json") for arm in ARMS for seed in SEEDS]
    if len(runs) != 12:
        raise ValueError(f"Expected 12 run records, got {len(runs)}")
    completion = read_json(out/"completion.json")
    if completion.get("runs") != 12 or completion.get("coarse_calls") != 1029*12:
        raise ValueError(f"Experiment completion record does not match 12 × 1029: {completion}")
    source = read_json(SOURCE)
    dense_initial = read_json(out/"initial_dense.json")
    initial_rms = float(dense_initial["worst_mapping_rms_mm"])
    summary = {
        "purpose": "Single selected-layout 2D refinement; not population-potential evidence.",
        "initial_worst_mapping_rms_mm": initial_rms,
        "run_count": len(runs), "coarse_trace_calls_per_run": 1029,
        "dense_audits_per_run": 2, "runs": runs,
        "arms": aggregate(runs),
        "frozen_hit_slope_fit_interpretation": (
            "Compatibility diagnostic at fixed original ray hits only; not a bound on attainable imaging quality."),
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    (out/"RESULTS.txt").write_text(markdown_report(summary)+"\n")
    chosen = read_json(out/"last_freeform__91000000.json")
    make_figure(out, source, chosen, dense_initial)
    print(f"Wrote {out/'summary.json'}, {out/'RESULTS.txt'}, and {out/'before_after.png'}")


if __name__ == "__main__":
    main()
