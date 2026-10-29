"""Run a development pilot of 3/4/5-corner folded-path discovery."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from nslens.search.cma_me import CMAES
from .folded import RoutingBrief, evaluate, ranking
from .polygon import PolygonCodec


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_one(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    codec = PolygonCodec(**job["codec"])
    brief = RoutingBrief(**job["brief"])
    rng = np.random.default_rng(job["seed"])
    arm = job["arm"]
    archive, best, emitter = {}, None, None
    stagnation = 0
    incumbent_rank = None
    lifetimes, age, boundary_hits = [], 0, 0
    counts = Counter()
    source_rays, evaluated, restarts = 0, 0, 0
    started = time.monotonic()
    trace_path = out / "candidates.jsonl"
    bounds = (-np.ones(codec.gene_size), np.ones(codec.gene_size))
    with trace_path.open("w") as stream:
        while evaluated < job["evals"]:
            if arm == "random":
                population = rng.uniform(-1, 1, (job["popsize"], codec.gene_size))
            else:
                if emitter is None:
                    start = (archive[sorted(archive)[int(rng.integers(len(archive)))]] ["genes"]
                             if archive else rng.uniform(-1, 1, codec.gene_size))
                    emitter = CMAES(np.asarray(start), job["sigma"], popsize=job["popsize"],
                                    seed=int(rng.integers(0, 2**30)), bounds=bounds,
                                    boundary_handling=job["boundary"])
                    stagnation, age, incumbent_rank = 0, 0, None
                population = emitter.ask()
            rows = []
            added = False
            for genes in population:
                if evaluated >= job["evals"]:
                    break
                # Reserve a full trace before dispatch; no uncharged overshoot. A
                # geometrically invalid candidate ultimately consumes zero rays.
                if job["ray_budget"] is not None and source_rays + len(brief.rays()) > job["ray_budget"]:
                    break
                row = evaluate(codec, genes, brief)
                row["evaluation"] = evaluated
                evaluated += 1
                source_rays += row["source_rays"]
                counts[row["status"]] += 1
                if row["status"] == "geometry":
                    counts["geometry:" + row["reason"]] += 1
                if row["feasible"]:
                    counts["feasible"] += 1
                    key = row["route"]
                    if key not in archive or row["quality"] > archive[key]["quality"]:
                        archive[key] = row
                        added = True
                if best is None or ranking(row) > ranking(best):
                    best = row
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if not rows:
                break
            if arm != "random":
                boundary_hits += int(np.any(emitter.last_clipping_mask[:len(rows)], axis=1).sum())
                if len(rows) != len(population):
                    break  # Hard final partial population; never train on fictitious probes.
                ranks = [ranking(row, feedback=job["feedback"]) for row in rows]
                order = np.asarray(sorted(range(len(rows)), key=lambda i: ranks[i], reverse=True))
                emitter.tell(order)
                age += 1
                generation_best = max(ranks)
                progress = added or incumbent_rank is None or generation_best > incumbent_rank
                incumbent_rank = generation_best if incumbent_rank is None else max(incumbent_rank, generation_best)
                stagnation = 0 if progress else stagnation + 1
                if emitter.restart_needed or stagnation >= job["patience"]:
                    restarts += 1
                    lifetimes.append(age)
                    emitter = None
    if emitter is not None and age:
        lifetimes.append(age)
    # Audit is held out from search feedback, includes intermediate angular fields and
    # a denser full aperture. All archived route representatives are audited, not just
    # the most attractive plot. This cost is separate and explicit.
    fields = sorted(brief.fields_deg)
    dense_fields = tuple(sorted(set(fields + [(a+b)/2 for a, b in zip(fields, fields[1:])])))
    dense = replace(brief, rays_per_field=job["audit_rays"], fields_deg=dense_fields)
    audits = []
    for key in sorted(archive):
        candidate = archive[key]
        result = evaluate(codec, candidate["genes"], dense, geometry_samples=1024)
        audits.append(dict(coarse_evaluation=candidate["evaluation"], coarse_route=key, result=result))
    write(out / "archive.json", dict(candidates=list(archive.values()), audits=audits))
    write(out / "best_candidate.json", best)
    summary = dict(corners=codec.corners, arm=arm, seed=job["seed"], evaluations=evaluated,
                   counts=dict(counts), coarse_routes=len(archive),
                   audited_routes=sum(a["result"]["feasible"] for a in audits),
                   unique_audit_routes=len({a["result"]["route"] for a in audits if a["result"]["feasible"]}),
                   source_rays=source_rays, audit_source_rays=sum(a["result"]["source_rays"] for a in audits),
                   boundary_interventions=boundary_hits, restarts=restarts,
                   emitter_lifetimes=lifetimes, best_quality=best["quality"] if best else 0.,
                   wall_seconds=time.monotonic()-started, config=job)
    write(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON with codec and brief overrides")
    parser.add_argument("--corners", nargs="+", type=int, choices=(3, 4, 5), default=[3, 4, 5])
    parser.add_argument("--arms", nargs="+", choices=("random", "flat", "cma"), default=["random", "cma"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--evals", type=int, default=500, help="hard candidate cap per corner/arm/seed")
    parser.add_argument("--ray-budget", type=int, help="hard discovery source-ray cap per run; audit cost separate")
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=.15)
    parser.add_argument("--patience", type=int, default=5, help="explicit experimental stagnation allowance")
    parser.add_argument("--boundary", choices=("clip", "reflect"), default="reflect")
    parser.add_argument("--audit-rays", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.evals, args.patience, args.workers) < 1 or args.popsize < 2 or not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("positive budgets, sigma, workers, patience and population >= 2 required")
    if args.ray_budget is not None and args.ray_budget < 1:
        parser.error("ray budget must be positive")
    if any(len(set(v)) != len(v) for v in (args.corners, args.arms, args.seeds)):
        parser.error("duplicate run units")
    config = json.loads(args.config.read_text()) if args.config else {}
    if set(config) - {"codec", "brief"}:
        parser.error("config allows only codec and brief keys")
    brief = RoutingBrief(**config.get("brief", {}))
    if args.audit_rays <= brief.rays_per_field:
        parser.error("audit ray count must exceed discovery count")
    jobs = []
    for corners in args.corners:
        codec = PolygonCodec(corners=corners, **config.get("codec", {}))
        for arm in args.arms:
            for seed in args.seeds:
                jobs.append(dict(codec=asdict(codec), brief=asdict(brief), arm=arm, seed=seed,
                                 evals=args.evals, popsize=args.popsize, sigma=args.sigma,
                                 patience=args.patience, boundary=args.boundary,
                                 feedback=arm != "flat", ray_budget=args.ray_budget,
                                 audit_rays=args.audit_rays,
                                 out=str(args.out / f"k{corners}_{arm}_seed{seed}")))
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("polygon.py"), Path(__file__).with_name("folded.py"),
               Path(__file__).parents[1]/"search"/"cma_me.py",
               *sorted((Path(__file__).parents[1]/"freeform_body").glob("*.py"))]
    write(args.out / "manifest.json", dict(stage="development_pilot", jobs=jobs,
          sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          caveat="Folded routing only; no imaging, compaction, or global novelty claim."))
    summaries = []
    if args.workers == 1:
        iterator = map(run_one, jobs)
        for result in iterator:
            summaries.append(result)
            print(result["corners"], result["arm"], result["seed"], result["counts"], flush=True)
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            for result in pool.imap_unordered(run_one, jobs):
                summaries.append(result)
                print(result["corners"], result["arm"], result["seed"], result["counts"], flush=True)
    summaries.sort(key=lambda r: (r["corners"], r["arm"], r["seed"]))
    write(args.out / "summary.json", summaries)


if __name__ == "__main__":
    main()
