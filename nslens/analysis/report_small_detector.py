"""Render the fixed detector layout and tabulate all paired small-detector tasks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    jobs = protocol["jobs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, name in zip(axes, ("base", "wide")):
        brief = next(j["brief"] for j in jobs if j["brief_id"] == name)
        bottom, top = protocol["detector_layout"]["setup_y_bounds"]
        sx, sy = brief["source_center"]
        beam = brief["beam_width_mm"]
        dy, dh = brief["detector_center_y"], brief["detector_half_height"]
        ax.axhline(bottom, color="gray", ls="--")
        ax.axhline(top, color="gray", ls="--")
        ax.axhline(dy, color="#7958a3", ls=":")
        ax.plot([sx-beam/2, sx+beam/2], [sy, sy], color="#11946c", lw=5)
        ax.annotate("Source", (sx, sy), xytext=(sx+8, sy-5))
        ax.annotate("", (sx, sy-15), (sx, sy-2), arrowprops=dict(arrowstyle="->", color="#11946c"))
        ax.plot([brief["detector_x"]]*2, [dy-dh, dy+dh], color="#7958a3", lw=5)
        ax.text(70, dy-14, f"Detector: {2*dh:g} mm\ncentre (120, {dy:g}) mm")
        ax.annotate("", (10, dy), (10, bottom), arrowprops=dict(arrowstyle="<->"))
        ax.text(13, (bottom+dy)/2, f"H/3 = {(top-bottom)/3:g} mm", fontsize=9)
        ax.set(xlim=(0, 140), ylim=(bottom-5, top+10), xlabel="x (mm)", ylabel="y (mm)",
               title=f"{beam:g} mm input beam · n = 1.60")
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
    ratio = protocol["detector_layout"]["beam_ratio"]
    bottom, top = protocol["detector_layout"]["setup_y_bounds"]
    fig.suptitle(f"Fixed environment: detector height = {ratio:g} × beam width\nReference height {top-bottom:g} mm; geometry evolves inside the existing search bounds")
    fig.tight_layout()
    fig.savefig(args.run / "environment.png", dpi=170)
    plt.close(fig)
    if not (args.run / "comparison.json").exists():
        return
    comparison = json.loads((args.run / "comparison.json").read_text())
    rows = json.loads((args.run / "summary.json").read_text())
    arms = ["naive", "legacy_route", "pilot", "portfolio"]
    lines = ["# Small-detector results", "", "See environment.png and protocol.json for the fixed layout.", "",
             "| Method | Successful runs | Audited passing designs | Sum of within-run routes |", "|---|---:|---:|---:|"]
    for arm in arms:
        t = comparison["totals"][arm]
        lines.append(f"| {arm} | {t['successes']}/{t['runs']} ({t['successes']/t['runs']:.1%}) | {t['audited_designs']} | {t['audited_routes']} |")
    lines += ["", "| Beam / corners | naive | legacy_route | pilot | portfolio |", "|---|---:|---:|---:|---:|"]
    for brief in ("base", "wide"):
        for k in (3, 4, 5):
            cells = []
            for arm in arms:
                group = [r for r in rows if r["brief_id"] == brief and r["corners"] == k and r["arm"] == arm]
                cells.append(f"{sum(r['audited_success'] for r in group)}/{len(group)}")
            lines.append(f"| {brief} / {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons", ""]
    for c in comparison["comparisons"]:
        lo, hi = np.array(c["seed_block_bootstrap_95"])*100
        lines.append(f"- portfolio minus {c['control']}: {100*c['mean_difference']:+.1f} percentage points; 95% seed-block interval [{lo:+.1f}, {hi:+.1f}]; one-sided seed sign p={c['one_sided_seed_sign_p']:.4g}.")
    lines += ["", "Primary control is naive; other comparisons are exploratory. Six seeds limit precision.",
              "legacy_route is the historical ranking/restart behavior adapted to common route niches, not the original grid driver.",
              "Verified means denser sampling with the same tracer, not independent-engine certification.", "",
              "## Search effort", "", "First-success numbers below refer to coarse search passes, conditional on finding one; they are not dense first-success times."]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        first = [r["first_coarse_success_rays"] for r in group if r["first_coarse_success_rays"] is not None]
        q = [r["best_audited_quality"] for r in group if r["audited_success"]]
        lines.append(f"- {arm}: median rays to first coarse pass = {float(np.median(first)) if first else 'none'}; median best audited throughput among successful runs = {float(np.median(q)) if q else 'none'}; actual search rays = {sum(r['search_source_rays'] for r in group)}.")
    (args.run / "RESULTS.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
