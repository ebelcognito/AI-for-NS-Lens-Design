"""Compare restart sources for bounded TIR polygon discovery.

The portfolio arm calls ``nslens.discovery.benchmark.run_job`` unchanged. The
two added arms keep its polygon evaluator, ranking, CMA configuration, call
accounting, candidate log, finalist selector, and dense audit; only the restart
centre source changes (uniform random or a differential mutation of measured
history). Audits never feed search or selection.

Example (4/5-corner polygons, fresh paired seeds, 400 physical calls per job):
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.tir_sampling_study \
      --out experiments/tir_sampling_NEW --seeds 801 802 803 804 805 806 807 808 \
      --corners 4 5 --search-calls 400 --workers 8
"""
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
from scipy.stats import qmc

from nslens.discovery import benchmark
from nslens.discovery.folded import RoutingBrief, evaluate, ranking
from nslens.discovery.polygon import PolygonCodec
from nslens.discovery.portfolio import RestartPool, constraint_rank
from nslens.discovery.run import write
from nslens.search.cma_me import CMAES


VARIANT_POLICY = dict(benchmark.POLICIES["portfolio"])


def _reflect(x):
    return 1. - np.abs(np.remainder(np.asarray(x) + 1., 4.) - 2.)


def _history_restart(history, measured, rng, gene_size):
    """Differential-history restart, with the best measured row as its base."""
    ranked = sorted(measured, key=constraint_rank, reverse=True)
    if not ranked:
        return rng.uniform(-1., 1., gene_size), "uniform_empty_history"
    base = np.asarray(ranked[0]["genes"], dtype=float)
    if len(history) < 2:
        return base.copy(), "best_measured_history"
    ids = rng.choice(len(history), size=2, replace=False)
    a, b = (np.asarray(history[i], dtype=float) for i in ids)
    mutant = base + rng.uniform(.3, .8) * (a-b)
    mask = rng.random(gene_size) < .7
    mask[int(rng.integers(gene_size))] = True
    candidate = np.where(mask, mutant, base)
    return _reflect(candidate), "differential_history"


def run_variant_job(job):
    """Benchmark-compatible runner for the two restart-source ablations."""
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    codec, brief = PolygonCodec(**job["codec"]), RoutingBrief(**job["brief"])
    policy = job["policy"]
    rays_per_call = len(brief.rays())
    max_calls = job["search_calls"]
    max_proposals = max_calls * 4
    rng = np.random.default_rng(job["seed"])
    sobol = qmc.Sobol(codec.gene_size, scramble=True, seed=job["seed"])
    initial_points = 2*sobol.random_base2(int(np.ceil(np.log2(max_proposals+64)))) - 1
    pool = RestartPool(capacity=12)
    feasible, measured, history = [], [], []
    best = emitter = None
    calls = proposals = restarts = boundary_hits = 0
    restart_sources = Counter()
    counts = Counter()
    age = stagnation = 0
    incumbent = None
    lifetimes = []
    restart_events = []
    started = time.monotonic()
    with (out/"candidates.jsonl").open("w") as stream:
        while calls < max_calls and proposals < max_proposals:
            if job.get("deadline_unix") is not None and time.time() >= job["deadline_unix"]:
                break
            global_phase = proposals < policy["initial"]
            if global_phase:
                population = initial_points[proposals:min(proposals+job["popsize"], policy["initial"])]
            else:
                if emitter is None:
                    if job["arm"] == "de_history":
                        start, source = _history_restart(history, measured, rng, codec.gene_size)
                    else:
                        start = rng.uniform(-1., 1., codec.gene_size)
                        source = "uniform"
                    restart_sources[source] += 1
                    restart_events.append(dict(at_proposals=proposals, source=source,
                                               genes=np.asarray(start).tolist()))
                    emitter = CMAES(np.asarray(start, float), policy["sigma"],
                                    popsize=job["popsize"], seed=int(rng.integers(0, 2**30)),
                                    bounds=(-np.ones(codec.gene_size), np.ones(codec.gene_size)),
                                    boundary_handling=policy["boundary"])
                    age = stagnation = 0
                    incumbent = None
                population = emitter.ask()
            rows, archive_progress = [], False
            for genes in population:
                if calls >= max_calls or proposals >= max_proposals:
                    break
                row = evaluate(codec, genes, brief)
                row["evaluation"] = proposals
                proposals += 1
                calls += int(row["source_rays"] > 0)
                row["cumulative_search_rays"] = calls*rays_per_call
                rows.append(row)
                counts[row["status"]] += 1
                if row["status"] == "measured":
                    measured.append(row)
                    history.append(np.asarray(genes, dtype=float).copy())
                if row["feasible"]:
                    counts["feasible"] += 1
                    others = [r["quality"] for r in feasible if r["route"] == row["route"]]
                    archive_progress |= not others or row["quality"] > max(others)
                    feasible.append(row)
                if best is None or constraint_rank(row) > constraint_rank(best):
                    best = row
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            pool.update(rows)
            if emitter is not None and not global_phase:
                boundary_hits += int(np.any(emitter.last_clipping_mask[:len(rows)], axis=1).sum())
                if len(rows) < len(population):
                    break
                ranks = [constraint_rank(r) if policy["pool"] else ranking(r, feedback=policy["feedback"])
                         for r in rows]
                emitter.tell(np.asarray(sorted(range(len(rows)), key=lambda i: ranks[i], reverse=True)))
                age += 1
                batch_best = max(ranks)
                progress = archive_progress or (policy["feedback"] and (incumbent is None or batch_best > incumbent))
                incumbent = batch_best if incumbent is None else max(incumbent, batch_best)
                stagnation = 0 if progress else stagnation+1
                if stagnation >= policy["patience"] or emitter.restart_needed:
                    restarts += 1
                    lifetimes.append(age)
                    emitter = None
    if emitter is not None and age:
        lifetimes.append(age)

    fields = sorted(brief.fields_deg)
    audit_brief = replace(brief, rays_per_field=job["audit_rays"],
                          fields_deg=tuple(sorted(set(fields+[(a+b)/2 for a,b in zip(fields, fields[1:])]))))
    candidates = benchmark.selected_candidates(feasible, job["audit_limit"])
    audits = [dict(coarse_evaluation=row["evaluation"],
                   result=evaluate(codec, row["genes"], audit_brief, geometry_samples=1024))
              for row in candidates]
    accepted = [a["result"] for a in audits if a["result"]["feasible"]]
    summary = dict(unit=job["unit"], arm=job["arm"], seed=job["seed"], corners=codec.corners,
                   brief_id=job["brief_id"], config=job, proposals=proposals, search_calls=calls,
                   search_source_rays=calls*rays_per_call,
                   audit_source_rays=sum(a["result"]["source_rays"] for a in audits),
                   reserved_source_rays=max_calls*rays_per_call+job["audit_limit"]*len(audit_brief.rays()),
                   counts=dict(counts), audited_success=bool(accepted), audited_designs=len(accepted),
                   audited_routes=len({r["route"] for r in accepted}),
                   best_audited_quality=max((r["quality"] for r in accepted), default=0.),
                   first_coarse_success_rays=min((r["cumulative_search_rays"] for r in feasible), default=None),
                   restarts=restarts, restart_sources=dict(restart_sources), restart_events=restart_events,
                   lifetimes=lifetimes, boundary_interventions=boundary_hits,
                   wall_seconds=time.monotonic()-started)
    summary["stop_reason"] = ("call_budget" if calls >= max_calls else "proposal_budget"
                              if proposals >= max_proposals else "deadline")
    write(out/"archive.json", dict(candidates=candidates, audits=audits))
    write(out/"best_candidate.json", best)
    write(out/"summary.json", summary)
    return summary


def _run_job(job):
    return benchmark.run_job(job) if job["arm"] == "portfolio" else run_variant_job(job)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(801, 809)))
    p.add_argument("--corners", type=int, choices=(4, 5), nargs="+", default=[4, 5])
    p.add_argument("--briefs", nargs="+", choices=("base", "wide", "shifted", "right_low",
                   "right_mid", "bottom_mid", "bottom_right", "patent_923", "patent_107"), default=["base"])
    p.add_argument("--search-calls", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--arms", nargs="+", choices=("portfolio","random_restart","de_history"), default=["portfolio","random_restart","de_history"])
    p.add_argument("--stage", choices=("development","confirmation"), default="development")
    a = p.parse_args(argv)
    if min(a.search_calls, a.workers) < 1 or any(len(set(x)) != len(x) for x in (a.seeds, a.corners, a.briefs)):
        p.error("positive budgets and unique seeds, corners, and briefs required")

    brief_options = {
        "base": RoutingBrief(),
        "wide": RoutingBrief(beam_width_mm=6.),
        "shifted": RoutingBrief(source_center=(58., 50.), fields_deg=(-1., 0., 1.)),
        "right_low": RoutingBrief(detector_x=120., detector_center_y=-20., detector_half_height=5.),
        "right_mid": RoutingBrief(detector_x=120., detector_center_y=-2.5, detector_half_height=5.),
        "bottom_mid": RoutingBrief(detector_axis=1, detector_travel_sign=-1,
                                   detector_position=-55., detector_center=60., detector_half_length=5.),
        "bottom_right": RoutingBrief(detector_axis=1, detector_travel_sign=-1,
                                     detector_position=-55., detector_center=80., detector_half_length=5.),
        "patent_923": RoutingBrief(source_center=(55.,18.), aim_deg=-90., detector_x=78.,
                                   detector_center_y=0., detector_half_height=5.,
                                   source_clearance_bounds=(3.,10.), detector_clearance_bounds=(16.,22.)),
        "patent_107": RoutingBrief(beam_width_mm=4.8, source_center=(73.,0.), aim_deg=180.,
                                   detector_axis=1, detector_travel_sign=-1, detector_position=-32.,
                                   detector_center=55., detector_half_length=6.,
                                   source_clearance_bounds=(3.,10.), detector_clearance_bounds=(20.,28.)),
    }
    codecs = {name: {k: PolygonCodec(k) for k in a.corners} for name in brief_options}
    for k in a.corners:
        common = dict(center_x=(51.,59.), center_y=(-4.,4.), radial_fraction=(.75,1.),
                      angular_jitter_fraction=.15, max_sag_ratio=.268,
                      min_radius_mm=8., min_edge_mm=4., n=1.60)
        codecs["patent_923"][k] = PolygonCodec(k, half_width=(6.,10.), half_height=(10.,16.), **common)
        codecs["patent_107"][k] = PolygonCodec(k, half_width=(10.,16.), half_height=(6.,10.), **common)

    root = Path(__file__).resolve().parents[2]
    jobs = []
    arms = tuple(a.arms)
    for name in a.briefs:
        for k in a.corners:
            for seed in a.seeds:
                unit = f"{name}_k{k}_seed{seed}"
                for arm in arms:
                    policy = dict(benchmark.POLICIES["portfolio"])
                    jobs.append(dict(unit=unit, brief_id=name, arm=arm, seed=seed,
                                     codec=asdict(codecs[name][k]), brief=asdict(brief_options[name]),
                                     policy=policy, search_calls=a.search_calls, popsize=10,
                                     audit_rays=25, audit_limit=8,
                                     out=str(a.out/f"{unit}_{arm}")))
    a.out.mkdir(parents=True, exist_ok=False)
    package_root = root/"nslens"
    sources = [Path(__file__), Path(benchmark.__file__), Path(benchmark.__file__).with_name("folded.py"),
               Path(benchmark.__file__).with_name("polygon.py"), Path(benchmark.__file__).with_name("portfolio.py"),
               Path(benchmark.__file__).with_name("run.py"), package_root/"search"/"cma_me.py",
               *sorted((package_root/"freeform_body").glob("*.py"))]
    write(a.out/"protocol.json", dict(
        stage=a.stage, jobs=jobs,
        arms={"portfolio": "unmodified benchmark.run_job retained-candidate baseline",
              "random_restart": "portfolio CMA settings; each restart centre uniform in normalized gene space",
              "de_history": "portfolio CMA settings; each restart centre is a reflected F*(a-b) mutation of all prior measured genes, based on the best measured row"},
        physical_call_budget="At most search_calls valid physical bundles; geometry-invalid proposals consume proposal slots but source_rays=0; max proposals=4*search_calls.",
        paired_seeds="Every arm shares seed, Sobol initial population, codec, brief, CMA settings, evaluator and dense audit selector.",
        audits="Up to 8 route-diverse candidates; 25 rays per field and midpoint-heldout fields; 1024 geometry samples. Audit results never select or restart search.",
        caveat="Development comparison over declared 4/5-corner codecs and seeds; does not establish general optimizer superiority.",
        sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}))
    summaries = []
    with get_context("spawn").Pool(a.workers) as pool:
        for row in pool.imap_unordered(_run_job, jobs):
            summaries.append(row)
            print(row["unit"], row["arm"], "audit", row["audited_designs"],
                  "calls", row["search_calls"], "wall", round(row["wall_seconds"], 1), flush=True)
    write(a.out/"summary.json", sorted(summaries, key=lambda r:(r["unit"], r["arm"])))


if __name__ == "__main__":
    main()
