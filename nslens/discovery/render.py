"""Render traced discovery candidates; failed families are labeled explicitly."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .folded import RoutingBrief, evaluate, ranking
from .polygon import PolygonCodec
from .run import write


def render(root):
    summaries = json.loads((root / "summary.json").read_text())
    counts = sorted({s["corners"] for s in summaries})
    fig, axes = plt.subplots(1, len(counts), figsize=(6*len(counts), 7), squeeze=False)
    verified = []
    for ax, k in zip(axes[0], counts):
        options = []
        for summary in summaries:
            if summary["corners"] != k:
                continue
            folder = Path(summary["config"]["out"])
            archive = json.loads((folder / "archive.json").read_text())
            for audit in archive["audits"]:
                options.append((audit["result"], summary, True))
            best = json.loads((folder / "best_candidate.json").read_text())
            if best:
                options.append((best, summary, False))
        if not options:
            ax.set_title(f"{k} corners: no evaluated candidate")
            continue
        candidate, summary, was_audited = max(options, key=lambda pair: ranking(pair[0]))
        codec = PolygonCodec(**summary["config"]["codec"])
        brief = RoutingBrief(**summary["config"]["brief"])
        f = sorted(brief.fields_deg)
        brief = replace(brief, rays_per_field=summary["config"]["audit_rays"],
                        fields_deg=tuple(sorted(set(f + [(a+b)/2 for a, b in zip(f, f[1:])]))))
        result = evaluate(codec, candidate["genes"], brief, return_trace=True, geometry_samples=1024)
        if not isinstance(result, tuple):
            ax.set_title(f"{k} corners: geometry failed recheck")
            verified.append(result)
            continue
        row, body, trace = result
        verified.append(dict(corners=k, arm=summary["arm"], seed=summary["seed"], result=row))
        boundary = body.sample(1200)[1]
        boundary = np.vstack((boundary, boundary[0]))
        ax.fill(boundary[:, 0], boundary[:, 1], color="#d7eafa", alpha=.8)
        ax.plot(boundary[:, 0], boundary[:, 1], color="#153d56", lw=2)
        ax.scatter(body.corners[:, 0], body.corners[:, 1], color="#153d56", s=20)
        for j, point in enumerate(body.corners):
            ax.annotate(str(j), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
        rays = brief.rays()
        mid = len(brief.fields_deg)//2
        source_ids = np.arange(mid*brief.rays_per_field, (mid+1)*brief.rays_per_field)
        # Plot a few strongest delivered folded branches from the central angular
        # field. Full field/ray acceptance always comes from the stored audit.
        step = max(1, brief.rays_per_field//8)
        for sid in source_ids[::step]:
            choices = [r for r in trace.exits if r.source_id == sid and r.events]
            folded = [r for r in choices if sum(e.kind == "tir" for e in r.events) >= brief.min_tir]
            if not choices:
                continue
            ray = max(folded or choices, key=lambda r: r.energy)
            path = [rays[sid].origin, *(e.point for e in ray.events)]
            origin, direction = np.asarray(ray.origin), np.asarray(ray.direction)
            detector_axis, detector_position, _, _, detector_sign = brief.detector()
            component = direction[detector_axis]
            distance = ((detector_position-origin[detector_axis])/component
                        if component*detector_sign > 1e-9 else 25.)
            path.append(origin + max(0., distance)*direction)
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], lw=.9, alpha=.75,
                    color="#cf5d27" if folded else "#777777")
        origins = np.asarray([r.origin for r in rays[:brief.rays_per_field]])
        ax.plot(origins[:, 0], origins[:, 1], color="#11946c", lw=4)
        detector_axis, detector_position, detector_center, detector_half, _ = brief.detector()
        if detector_axis == 0:
            detector_points = ([detector_position]*2,
                               [detector_center-detector_half, detector_center+detector_half])
        else:
            detector_points = ([detector_center-detector_half, detector_center+detector_half],
                               [detector_position]*2)
        ax.plot(*detector_points, color="#7958a3", lw=3)
        label = ("DENSE AUDIT PASS" if row["feasible"] and was_audited else
                 "POST-RUN DENSE PASS" if row["feasible"] else
                 "EXPLORATORY — DOES NOT PASS")
        ax.set_title(f"{k} corners · {label}\nworst-field folded energy {row['quality']:.1%}; root support {row['support']:.1%}", fontsize=10)
        scene = np.vstack((boundary, origins,
                           np.column_stack(detector_points)))
        margin = max(5., .08*np.ptp(scene, axis=0).max())
        ax.set_xlim(scene[:, 0].min()-margin, scene[:, 0].max()+margin)
        ax.set_ylim(scene[:, 1].min()-margin, scene[:, 1].max()+margin)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(alpha=.15)
    fig.suptitle("Finite-width folded-path discovery · analytic circular faces\nSelected examples; central-field paths shown. Not imaging or compaction certification.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(root / "discovery_examples.png", dpi=160)
    plt.close(fig)
    write(root / "render_verification.json", verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    render(args.run)


if __name__ == "__main__":
    main()
