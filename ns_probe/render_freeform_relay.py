"""Render freeform relay elites: the continuous boundary plus the rays it routes.

The legacy renderers draw a four-face `Element` from the engine's raySequences.  A
`FreeformBody2D` has neither, so this walks the Stage-1 tracer directly: one input ray
at a time, so each drawn polyline can start at its true eyebox origin.

    python ns_probe/render_freeform_relay.py runs/relay_pilot/seed11 --top 3 --diverse

`--diverse` picks the best elite from each distinct interaction sequence rather than the
best three overall, which otherwise tend to be near-neighbours in one route family.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nslens.freeform_body import (                                    # noqa: E402
    ArcPolygonCodec,
    DisplayTarget,
    FreeformBodyCodec,
    FreeformRelayProblem,
    VerticalBlocker,
    collimated_eyebox_rays,
    trace_bundle,
)
from nslens.freeform_body.relay import (                              # noqa: E402
    EyeboxSpec,
    ObjectPlane,
    TARGET_OFFSETS,
)

FIELD_COLORS = ("#1f77b4", "#2ca02c", "#d62728")


def interaction_sequence(elite: dict) -> tuple:
    return ast.literal_eval(elite["route_key"])[0][0]


def select(elites: list, count: int, diverse: bool) -> list:
    if not diverse:
        return sorted(elites, key=lambda e: -e["fitness"])[:count]
    best: dict[tuple, dict] = {}
    for elite in elites:
        seq = interaction_sequence(elite)
        if seq not in best or elite["fitness"] > best[seq]["fitness"]:
            best[seq] = elite
    return sorted(best.values(), key=lambda e: -e["fitness"])[:count]


def ray_polylines(problem: FreeformRelayProblem, body, *, stride: int = 2,
                  keep_per_ray: int = 4):
    """One polyline per routed exit: eyebox origin, every interaction, then the plane.

    Each is tagged with whether it lands in a target window.  At a few percent
    throughput the *brightest* exit is usually not the imaging one, so drawing
    brightest-only shows mostly light that misses -- a misleading picture.
    """
    lines = []
    for index, ray in enumerate(collimated_eyebox_rays(problem.eyebox, problem.field_degs)):
        if index % stride:
            continue
        trace = trace_bundle(body, (ray,), max_interactions=problem.max_interactions,
                             energy_floor=problem.energy_floor)
        routed = sorted((e for e in trace.exits if e.events),
                        key=lambda e: -e.energy)[:keep_per_ray]
        for exit_ray in routed:
            points = [ray.origin] + [event.point for event in exit_ray.events]
            origin = np.asarray(exit_ray.origin)
            direction = np.asarray(exit_ray.direction)
            plane = problem.object_plane
            imaging = False
            approach = float(np.dot(direction, plane.normal))
            t = 40.0
            if approach < -1e-12:
                hit = float(np.dot(np.asarray(plane.center) - origin, plane.normal))
                candidate = hit / approach
                if candidate > 0:
                    t = candidate
                    landing = float(np.dot(origin + t * direction
                                           - np.asarray(plane.center), plane.along))
                    imaging = any(abs(landing - target.offset) <= target.half_width
                                  for target in problem.targets)
                    imaging = (imaging and sum(event.kind == "tir"
                                               for event in exit_ray.events)
                               >= problem.minimum_tir_bounces)
            endpoint = origin + min(t, 400.0) * direction
            blocked_at = (problem.blocker.intersection(origin, endpoint)
                          if problem.blocker is not None else None)
            if blocked_at is not None:
                endpoint = np.asarray(blocked_at)
                imaging = False
            points.append(tuple(endpoint))
            lines.append((ray.source_id, exit_ray.energy, imaging, np.asarray(points)))
    return lines


def assigned_field(index: int, parity: int, count: int) -> int:
    """Which field a display target belongs to under this design's parity.

    Colouring the targets by their own index would claim field i lands on target i,
    which is only true at parity +1.  Half the archive is parity -1.
    """
    return index if parity >= 0 else count - 1 - index


def draw(ax, problem: FreeformRelayProblem, elite: dict, title: str) -> None:
    body = problem.codec.decode(np.asarray(elite["genes"]))
    _, boundary = body.sample(400)

    lines = ray_polylines(problem, body)
    for source_id, energy, imaging, points in lines:
        if imaging:      # drawn above the body instead, so it stays readable
            continue
        ax.plot(points[:, 0], points[:, 1], lw=0.5, color="#9aa5b1",
                alpha=float(np.clip(0.05 + 0.35 * energy, 0.04, 0.35)), zorder=1)

    ax.fill(boundary[:, 0], boundary[:, 1], color="#cfd8e3", alpha=0.85, zorder=2)
    ax.plot(np.append(boundary[:, 0], boundary[0, 0]),
            np.append(boundary[:, 1], boundary[0, 1]),
            color="#26323f", lw=1.4, zorder=3)

    for source_id, energy, imaging, points in lines:
        if not imaging:
            continue
        ax.plot(points[:, 0], points[:, 1], lw=1.3,
                color=FIELD_COLORS[source_id % len(FIELD_COLORS)], alpha=0.9, zorder=6)

    eyebox = problem.eyebox
    eye0, eye1 = eyebox.endpoints
    ax.plot([eye0[0], eye1[0]], [eye0[1], eye1[1]], color="#111", lw=4,
            solid_capstyle="butt", zorder=4)
    ax.annotate("eyebox", (eyebox.x, eyebox.y_center), fontsize=7, ha="right",
                va="center", xytext=(-6, 0), textcoords="offset points")

    if problem.blocker is not None:
        p0, p1 = problem.blocker.endpoints
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#111", lw=6,
                solid_capstyle="butt", zorder=7)
        ax.annotate("opaque stop", (problem.blocker.x, p1[1]), fontsize=7,
                    ha="center", va="bottom")

    plane = problem.object_plane
    span = np.array([plane.point_at(-34.0), plane.point_at(34.0)])
    ax.plot(span[:, 0], span[:, 1], color="#555", lw=0.9, ls="--", zorder=4)
    parity = int(elite.get("parity", 1) or 1)
    for index, target in enumerate(problem.targets):
        field = assigned_field(index, parity, len(problem.targets))
        edge = np.array([plane.point_at(target.offset - target.half_width),
                         plane.point_at(target.offset + target.half_width)])
        ax.plot(edge[:, 0], edge[:, 1],
                color=FIELD_COLORS[field % len(FIELD_COLORS)], lw=4,
                solid_capstyle="butt", zorder=5)
    label_at = plane.point_at(24.0)
    ax.annotate("tilted object plane", label_at, fontsize=7, ha="left", va="bottom")

    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_aspect("equal")
    extent_points = [boundary, span, np.asarray(eyebox.endpoints)]
    if problem.blocker is not None:
        extent_points.append(np.asarray(problem.blocker.endpoints))
    extent = np.concatenate(extent_points)
    ax.set_xlim(float(extent[:, 0].min()) - 8.0,
                float(extent[:, 0].max()) + 8.0)
    ax.set_ylim(float(extent[:, 1].min()) - 8.0,
                float(extent[:, 1].max()) + 8.0)
    ax.tick_params(labelsize=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="directory holding archive.json")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--diverse", action="store_true",
                    help="best per distinct interaction sequence, not best overall")
    ap.add_argument("--chief-ray-only", action="store_true",
                    help="draw only the centred chief ray; archived fitness still "
                         "comes from the run's complete pupil bundle")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads((args.run / "archive.json").read_text())
    spec = payload["spec"]
    field_degs = tuple(spec.get(
        "field_degs", (-spec["half_fov"], 0.0, spec["half_fov"])))
    targets = tuple(DisplayTarget(offset) for offset in
                    spec.get("targets", TARGET_OFFSETS))
    blocker_spec = spec.get("blocker")
    blocker = VerticalBlocker(**blocker_spec) if blocker_spec else None
    eyebox_spec = spec.get("eyebox", [EyeboxSpec().x, EyeboxSpec().half_height])
    if isinstance(eyebox_spec, dict):
        eyebox = EyeboxSpec(x=eyebox_spec["x"],
                            y_center=eyebox_spec["y_center"],
                            half_height=eyebox_spec["half_height"],
                            tilt_deg=eyebox_spec.get("tilt_deg", 0.0),
                            ray_count=spec["pupil_rays"])
    else:  # backward compatibility with [x, half_height] archives
        eyebox = EyeboxSpec(x=eyebox_spec[0], half_height=eyebox_spec[1],
                            tilt_deg=0.0,
                            ray_count=spec["pupil_rays"])
    if args.chief_ray_only:
        eyebox = replace(eyebox, ray_count=1)
    arc_kw = {key: spec[key] for key in ("corner_region", "sag_limit")
              if spec.get(key) is not None}
    problem = FreeformRelayProblem(
        codec=(ArcPolygonCodec(n=spec["index"], **arc_kw)
               if spec.get("geometry", "spline") == "arc"
               else FreeformBodyCodec(n=spec["index"], control_count=spec["controls"],
                                      harmonic_count=spec["harmonics"])),
        eyebox=eyebox,
        field_degs=field_degs,
        object_plane=ObjectPlane(center=tuple(spec.get("object_center", (0.0, 0.0))),
                                 tilt_deg=spec.get("object_tilt", 135.0)),
        targets=targets,
        blocker=blocker,
        minimum_tir_bounces=spec.get("minimum_tir_bounces", 0),
        minimum_field_throughput=spec.get("min_throughput", 1e-3),
    )
    chosen = select(payload["elites"], args.top, args.diverse)

    columns = min(5, len(chosen))
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns,
                             figsize=(4.6 * columns, 4.9 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, elite in zip(axes, chosen):
        seq = interaction_sequence(elite)
        bounces = sum(1 for kind in seq if kind == "tir")
        label = "no fold" if not bounces else f"{bounces} TIR"
        ax.set_facecolor("white")
        draw(ax, problem, elite,
             f"{'→'.join(seq)}\nfitness {elite['fitness']:.4f}   {label}   "
             f"parity {elite['parity']:+d}\n"
             f"per-field throughput "
             f"{[round(t, 3) for t in elite['throughputs']]}\n"
             f"colour = field identity (bar = its target),  grey = light that misses")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle(f"{args.run.name}: top {len(chosen)}"
                 f"{' diverse' if args.diverse else ''} freeform relay designs"
                 f"{' (chief-ray trace)' if args.chief_ray_only else ''}",
                 fontsize=11, y=0.995)
    # Four-line panel titles collide with the suptitle under a plain tight_layout.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out = args.out or (args.run / "top_designs.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    for elite in chosen:
        print(f"  {'→'.join(interaction_sequence(elite)):58s} fitness {elite['fitness']:.5f}")


if __name__ == "__main__":
    main()
