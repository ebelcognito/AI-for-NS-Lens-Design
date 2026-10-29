"""Stage-1 freeform discovery against the fixed-eyebox AR/VR relay.

Unlike the prism runs, the genotype here is a continuous periodic spline with no faces,
no corners and no prescribed path.  The objective states only where light enters (a
fixed eyebox, filled by three collimated fields) and where it must land (three fixed
points on a TILTED object plane).  Whatever route the body invents to connect them is recorded by
the descriptors, never by the fitness.

    python ns_probe/run_freeform_relay.py --evals 2000 --out runs/relay_pilot/seed0

Start small.  A three-field relay is far harder to satisfy than the single-bundle
throughput problems, so watch the first few hundred evaluations before committing a
large budget: if the niche count stays at zero, the geometry or the field spec is wrong
and more evaluations will not fix it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nslens.freeform_body import (
    ArcPolygonCodec,
    DisplayTarget,
    EyeboxSpec,
    FreeformBodyCodec,
    FreeformDiscoverySearch,
    FreeformRelayProblem,
    VerticalBlocker,
    route_signature,
)
from nslens.freeform_body.relay import FIELD_DEGS, ObjectPlane, TARGET_OFFSETS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, required=True,
                    help="hard evaluation budget; the driver never exceeds it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emitters", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.20)
    ap.add_argument("--geometry", choices=("arc", "spline"), default="arc",
                    help="arc = spherical faces on sampled corners (default); "
                         "spline = the older continuous freeform boundary")
    ap.add_argument("--corner-region", type=float, default=None,
                    help="arc geometry: half-width of each corner's sampling box")
    ap.add_argument("--sag-limit", type=float, default=None,
                    help="arc geometry: maximum bulge of any face")
    ap.add_argument("--harmonics", type=int, default=3,
                    help="genotype resolution; more harmonics means more geometric "
                         "detail and a larger search space")
    ap.add_argument("--controls", type=int, default=16)
    ap.add_argument("--index", type=float, default=1.60,
                    help="fixed refractive index; part of the problem, not the genotype")
    ap.add_argument("--fields", type=int, choices=(1, 3), default=1,
                    help="number of source fields: 1 = on-axis only (default); "
                         "3 = -half-fov, on-axis, +half-fov")
    ap.add_argument("--half-fov", type=float, default=abs(FIELD_DEGS[0]),
                    help="three-field mode only: outer field angle in degrees")
    ap.add_argument("--pupil-rays", type=int, default=EyeboxSpec().ray_count,
                    help="rays per field across the fixed eyebox; 1 traces only the "
                         "centred chief ray")
    ap.add_argument("--eyebox-tilt", type=float, default=EyeboxSpec().tilt_deg,
                    help="eyebox rotation in degrees; negative is clockwise "
                         "(default: %(default)s)")
    ap.add_argument("--energy-floor", type=float, default=1e-4,
                    help="Fresnel branches below this are pruned; raising it trades "
                         "faint-ghost fidelity for throughput")
    ap.add_argument("--object-tilt", type=float, default=ObjectPlane().tilt_deg,
                    help="orientation of the object plane in deg; 135 is the tilted "
                         "layout, 90 would make it an ordinary straight relay")
    ap.add_argument("--block-direct", action="store_true",
                    help="place an opaque stop and exclude zero-TIR target paths")
    ap.add_argument("--blocker-x", type=float, default=None,
                    help="blocker x; default is 20%% from object centre to eyebox")
    ap.add_argument("--blocker-y", type=float, default=None,
                    help="blocker centre y; default follows the object-to-eyebox line")
    ap.add_argument("--blocker-half-height", type=float, default=12.0)
    ap.add_argument("--min-throughput", type=float, default=None,
                    help="archive admission floor per field; below it a design still "
                         "steers the search from ladder band 3, it just does not take "
                         "a niche. Defaults to 0.001 with --block-direct, otherwise "
                         "the problem's 0.01 floor")
    ap.add_argument("--max-interactions", type=int, default=16)
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()

    if args.geometry == "arc":
        overrides = {}
        if args.corner_region is not None:
            overrides["corner_region"] = args.corner_region
        if args.sag_limit is not None:
            overrides["sag_limit"] = args.sag_limit
        codec = ArcPolygonCodec(n=args.index, **overrides)
    else:
        codec = FreeformBodyCodec(n=args.index, control_count=args.controls,
                                  harmonic_count=args.harmonics)
    field_degs = ((0.0,) if args.fields == 1
                  else (-args.half_fov, 0.0, args.half_fov))
    target_offsets = ((0.0,) if args.fields == 1 else TARGET_OFFSETS)
    eyebox = EyeboxSpec(ray_count=args.pupil_rays, tilt_deg=args.eyebox_tilt)
    object_plane = ObjectPlane(tilt_deg=args.object_tilt)
    blocker = None
    if args.block_direct:
        default_x = object_plane.center[0] + 0.20 * (eyebox.x - object_plane.center[0])
        blocker_x = default_x if args.blocker_x is None else args.blocker_x
        span_x = eyebox.x - object_plane.center[0]
        fraction = ((blocker_x - object_plane.center[0]) / span_x
                    if abs(span_x) > 1e-12 else 0.20)
        default_y = object_plane.center[1] + fraction * (
            eyebox.y_center - object_plane.center[1])
        blocker_y = default_y if args.blocker_y is None else args.blocker_y
        blocker = VerticalBlocker(blocker_x, blocker_y, args.blocker_half_height)
    minimum_throughput = (args.min_throughput if args.min_throughput is not None
                          else (1e-3 if args.block_direct else None))
    problem = FreeformRelayProblem(
        codec=codec,
        eyebox=eyebox,
        field_degs=field_degs,
        object_plane=object_plane,
        targets=tuple(DisplayTarget(offset) for offset in target_offsets),
        blocker=blocker,
        minimum_tir_bounces=1 if args.block_direct else 0,
        energy_floor=args.energy_floor,
        max_interactions=args.max_interactions,
        **({} if minimum_throughput is None
           else dict(minimum_field_throughput=minimum_throughput)),
    )
    search = FreeformDiscoverySearch(problem, emitters=args.emitters,
                                     popsize=args.popsize, sigma0=args.sigma0,
                                     seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    trace_path = args.out / "trace.jsonl"
    start = time.time()
    reported = 0
    with trace_path.open("w") as trace_file:
        while search.evaluations < args.evals:
            progress = search.step(max_new_evaluations=args.evals - search.evaluations)
            if progress.evaluations - reported < args.report_every:
                continue
            reported = progress.evaluations
            # An empty archive reports -inf, which is not valid strict JSON.
            best = progress.best_fitness
            row = dict(evaluations=progress.evaluations, invalid=progress.invalid,
                       joint_niches=progress.joint_niches,
                       route_families=progress.route_families,
                       shape_families=progress.shape_families,
                       best_fitness=best if np.isfinite(best) else None,
                       seconds=round(time.time() - start, 1))
            trace_file.write(json.dumps(row) + "\n")
            trace_file.flush()
            print(f"  {progress.evaluations:6d} evals  niches={progress.joint_niches:4d}"
                  f"  routes={progress.route_families:4d}"
                  f"  shapes={progress.shape_families:4d}"
                  f"  best={progress.best_fitness:.5f}", flush=True)

    records = search.archive.records
    elites = []
    for record in records:
        reports = getattr(record.evaluation, "field_reports", ())
        trace = record.evaluation.trace_result
        # Store the bundle-dominant route with the elite.  Recovering it later means
        # re-tracing the whole archive, which costs minutes and needs the exact problem
        # settings.  Its signature is selected by total energy over every interacted
        # exit branch sharing that route, not by one brightest branch.
        routed_exits = [ray for ray in trace.exits if ray.events] if trace else []
        dominant_signature = (record.evaluation.route_keys[0]
                              if record.evaluation.route_keys else None)
        dominant_branches = [ray for ray in routed_exits
                             if route_signature(ray) == dominant_signature]
        route = list(dominant_signature[0]) if dominant_signature else []
        elites.append(dict(
            genes=record.genes.tolist(),
            fitness=record.fitness,
            parity=getattr(record.evaluation, "parity", None),
            route_key=repr(record.route_key),
            shape_key=repr(record.shape_key),
            throughputs=[r.throughput for r in reports],
            rms=[r.rms_error for r in reports],
            dominant_route=route,
            dominant_route_branches=len(dominant_branches),
            dominant_route_energy=sum(ray.energy for ray in dominant_branches),
            tir_bounces=sum(1 for kind in route if kind == "tir"),
        ))
    summary = dict(search.archive.summary())
    if not np.isfinite(summary["best_fitness"]):
        summary["best_fitness"] = None
    payload = dict(
        spec=dict(evals=args.evals, seed=args.seed, index=args.index,
                  fields=args.fields, field_degs=list(problem.field_degs),
                  half_fov=args.half_fov, pupil_rays=args.pupil_rays,
                  geometry=args.geometry, harmonics=args.harmonics,
                  controls=args.controls,
                  targets=[target.offset for target in problem.targets],
                  object_tilt=args.object_tilt,
                  blocker=(None if blocker is None else
                           dict(x=blocker.x, y_center=blocker.y_center,
                                half_height=blocker.half_height)),
                  minimum_tir_bounces=problem.minimum_tir_bounces,
                  eyebox=dict(x=problem.eyebox.x,
                              y_center=problem.eyebox.y_center,
                              half_height=problem.eyebox.half_height,
                              tilt_deg=problem.eyebox.tilt_deg),
                  object_center=list(problem.object_plane.center),
                  corner_region=(codec.corner_region
                                 if isinstance(codec, ArcPolygonCodec) else None),
                  sag_limit=(codec.sag_limit
                             if isinstance(codec, ArcPolygonCodec) else None),
                  sigma0=args.sigma0, emitters=args.emitters, popsize=args.popsize,
                  min_throughput=problem.minimum_field_throughput),
        summary=summary,
        elites=elites,
    )
    (args.out / "archive.json").write_text(json.dumps(payload, indent=1))

    elapsed = time.time() - start
    print(f"\n{search.evaluations} evaluations in {elapsed:.0f}s "
          f"({search.evaluations / max(elapsed, 1e-9):.1f} ev/s)")
    print(f"archive: {len(records)} niches, "
          f"{search.archive.route_family_count} route families, "
          f"{search.archive.shape_family_count} shape families")
    print(f"best fitness: {search.archive.best_fitness:.5f}")
    if records:
        bounces = Counter(elite["tir_bounces"] for elite in elites)
        # Reported, never rewarded: folding is an observation about what the search
        # found, not a term in the objective.
        folded = sum(n for turns, n in bounces.items() if turns > 0)
        print(f"TIR bounces on the bundle-dominant route: {dict(sorted(bounces.items()))}"
              f"  ({folded}/{len(elites)} fold)")
    print(f"wrote {args.out / 'archive.json'}")


if __name__ == "__main__":
    main()
