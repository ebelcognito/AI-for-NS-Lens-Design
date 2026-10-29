"""Render the stress-verified representatives from a discovery stock run."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nslens.discovery.folded import RoutingBrief, evaluate
from nslens.discovery.polygon import PolygonCodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summaries = {row["unit"]: row for row in json.loads((args.run / "summary.json").read_text())}
    checks = json.loads((args.run / "replay_and_stress.json").read_text())["stress_examples"]
    checks = [row for row in checks if row["feasible"]]
    if not checks:
        raise RuntimeError("no stress-verified representatives")
    columns = 2
    rows = int(np.ceil(len(checks) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 5.5 * rows), squeeze=False)
    for ax, check in zip(axes.flat, checks):
        summary = summaries[check["unit"]]
        base = RoutingBrief(**summary["config"]["brief"])
        fields = sorted(base.fields_deg)
        brief = replace(base, rays_per_field=101,
                        fields_deg=tuple(np.linspace(fields[0], fields[-1], 9)))
        result, body, trace = evaluate(PolygonCodec(**summary["config"]["codec"]),
                                       check["result"]["genes"], brief,
                                       return_trace=True, geometry_samples=4096)
        boundary = body.sample(1200)[1]
        boundary = np.vstack((boundary, boundary[0]))
        ax.fill(boundary[:, 0], boundary[:, 1], color="#d7eafa")
        ax.plot(boundary[:, 0], boundary[:, 1], color="#153d56", lw=1.7)
        ax.scatter(body.corners[:, 0], body.corners[:, 1], color="#153d56", s=12)
        detector_axis, detector_position, detector_center, detector_half, detector_sign = brief.detector()
        if detector_axis == 0:
            detector = np.asarray(((detector_position, detector_center-detector_half),
                                   (detector_position, detector_center+detector_half)))
        else:
            detector = np.asarray(((detector_center-detector_half, detector_position),
                                   (detector_center+detector_half, detector_position)))
        ax.plot(detector[:, 0], detector[:, 1], color="#7958a3", lw=3)
        rays = brief.rays()
        central = len(brief.fields_deg) // 2
        ids = range(central*brief.rays_per_field, (central+1)*brief.rays_per_field, 12)
        path_points = []
        for source_id in ids:
            candidates = []
            for ray in trace.exits:
                if ray.source_id != source_id:
                    continue
                direction = np.asarray(ray.direction)
                if direction[detector_axis] * detector_sign <= 1e-9:
                    continue
                tir = [event for event in ray.events if event.kind == "tir"]
                if len(tir) < brief.min_tir or any(event.critical_margin_deg < brief.min_tir_margin_deg for event in tir):
                    continue
                distance = ((detector_position-ray.origin[detector_axis]) /
                            direction[detector_axis])
                end = np.asarray(ray.origin) + distance*direction
                if distance > 0 and abs(end[1-detector_axis]-detector_center) <= detector_half:
                    candidates.append((ray, end))
            if candidates:
                ray, end = max(candidates, key=lambda item: item[0].energy)
                points = np.asarray([rays[source_id].origin,
                                     *(event.point for event in ray.events), end])
                ax.plot(points[:, 0], points[:, 1], color="#d4642b", lw=.9, alpha=.8)
                path_points.append(points)
        source = np.asarray([ray.origin for ray in rays[central*101:(central+1)*101]])
        ax.plot(source[:, 0], source[:, 1], color="#11946c", lw=3)
        scene = np.vstack((boundary, detector, source, *path_points))
        margin = max(4., .07*np.ptp(scene, axis=0).max())
        ax.set_xlim(scene[:, 0].min()-margin, scene[:, 0].max()+margin)
        ax.set_ylim(scene[:, 1].min()-margin, scene[:, 1].max()+margin)
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
        ax.set_title(f"{summary['brief_id']} · {summary['corners']} corners\n"
                     f"909-ray pass · throughput {result['quality']:.1%} · support {result['support']:.1%}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    for ax in axes.flat[len(checks):]:
        ax.axis("off")
    fig.suptitle("Stress-verified diverse folded designs", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.run / "stress_verified_stock.png", dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
