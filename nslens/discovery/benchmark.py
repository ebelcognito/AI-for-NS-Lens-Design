"""Prospective, equal-source-ray discovery comparisons with capped final audits."""
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

from nslens.search.cma_me import CMAES
from .folded import RoutingBrief, evaluate, ranking
from .polygon import PolygonCodec
from .portfolio import RestartPool, constraint_rank, select_diverse
from .run import write


POLICIES = {
    "legacy_route": dict(feedback=False, boundary="clip", patience=1, sigma=.15, pool=False, initial=0),
    # Same geometry, requirements, and material across all arms.
    "naive": dict(feedback=False, boundary="clip", patience=1, sigma=.15, pool=False, initial=0),
    "pilot": dict(feedback=True, boundary="reflect", patience=5, sigma=.15, pool=False, initial=0),
    "portfolio": dict(feedback=True, boundary="reflect", patience=5, sigma=.15, pool=True, initial=40),
    "portfolio_small": dict(feedback=True, boundary="reflect", patience=8, sigma=.08, pool=True, initial=40),
    "random": dict(feedback=False, boundary="clip", patience=1, sigma=.15, pool=False, initial=0),
}


def legacy_route_rank(row, old):
    """Historical new > improved > unchanged ranking, adapted to route niches."""
    if not row["feasible"]:
        return (0, 0.)
    if old is None:
        return (3, row["quality"])
    if row["quality"] > old:
        return (2, row["quality"] - old)
    return (1, row["quality"])


def selected_candidates(feasible, limit):
    """Identical final selection for all arms: route coverage first, then geometry."""
    groups = {}
    for row in feasible:
        groups.setdefault(row["route"], []).append(row)
    alternatives = {route: select_diverse(rows, limit) for route, rows in groups.items()}
    routes = sorted(alternatives, key=lambda key: constraint_rank(alternatives[key][0]), reverse=True)
    chosen = []
    for index in range(limit):
        for route in routes:
            if index < len(alternatives[route]):
                chosen.append(alternatives[route][index])
                if len(chosen) == limit:
                    return chosen
    return chosen


def run_job(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    codec, brief = PolygonCodec(**job["codec"]), RoutingBrief(**job["brief"])
    policy = job["policy"]
    rays_per_call = len(brief.rays())
    # The common allowed ray budget is search_calls * source rays + reserved audits.
    # Invalid geometry consumes a proposal slot, never a fabricated physical trace.
    max_calls = job["search_calls"]
    max_proposals = max_calls * 4
    rng = np.random.default_rng(job["seed"])
    sobol = qmc.Sobol(codec.gene_size, scramble=True, seed=job["seed"])
    initial_points = 2*sobol.random_base2(int(np.ceil(np.log2(max_proposals+64)))) - 1
    pool = RestartPool(capacity=12)
    feasible, best, emitter = [], None, None
    calls = proposals = restarts = boundary_hits = 0
    restart_sources = Counter()
    counts = Counter()
    age = stagnation = 0
    incumbent = None
    lifetimes = []
    started = time.monotonic()
    with (out/"candidates.jsonl").open("w") as stream:
        while calls < max_calls and proposals < max_proposals:
            if job.get("deadline_unix") is not None and time.time() >= job["deadline_unix"]:
                break
            global_phase = proposals < policy["initial"]
            if job["arm"] == "random":
                population = rng.uniform(-1, 1, (job["popsize"], codec.gene_size))
            elif global_phase:
                population = initial_points[proposals:min(proposals+job["popsize"], policy["initial"])]
            else:
                if emitter is None:
                    # Every fourth restart bypasses the retained pool, using the
                    # feasible route archive or a uniform start when it is empty.
                    start = pool.start() if policy["pool"] and restarts % 4 != 3 else None
                    if start is not None:
                        restart_sources["retained"] += 1
                    elif feasible:
                        routes = {}
                        for row in feasible:
                            if row["route"] not in routes or row["quality"] > routes[row["route"]]["quality"]:
                                routes[row["route"]] = row
                        route = sorted(routes)[int(rng.integers(len(routes)))]
                        start = np.asarray(routes[route]["genes"])
                        restart_sources["feasible_archive"] += 1
                    else:
                        start = rng.uniform(-1, 1, codec.gene_size)
                        restart_sources["uniform"] += 1
                    emitter = CMAES(start, policy["sigma"], popsize=job["popsize"],
                                    seed=int(rng.integers(0, 2**30)),
                                    bounds=(-np.ones(codec.gene_size), np.ones(codec.gene_size)),
                                    boundary_handling=policy["boundary"])
                    age = stagnation = 0
                    incumbent = None
                population = emitter.ask()
            rows, archive_progress = [], False
            legacy_ranks = []
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
                old = max((r["quality"] for r in feasible if r["route"] == row["route"]), default=None)
                legacy_ranks.append(legacy_route_rank(row, old))
                if row["feasible"]:
                    counts["feasible"] += 1
                    others = [r["quality"] for r in feasible if r["route"] == row["route"]]
                    archive_progress |= not others or row["quality"] > max(others)
                    feasible.append(row)
                if best is None or constraint_rank(row) > constraint_rank(best):
                    best = row
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            if policy["pool"]:
                pool.update(rows)
            if emitter is not None and not global_phase and job["arm"] != "random":
                boundary_hits += int(np.any(emitter.last_clipping_mask[:len(rows)], axis=1).sum())
                if len(rows) < len(population):
                    break
                ranks = [constraint_rank(r) if policy["pool"] else ranking(r, feedback=policy["feedback"]) for r in rows]
                if job["arm"] == "legacy_route":
                    ranks = legacy_ranks
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
                          fields_deg=tuple(sorted(set(fields+[(a+b)/2 for a,b in zip(fields,fields[1:])]))))
    candidates = selected_candidates(feasible, job["audit_limit"])
    audits = []
    for row in candidates:
        result = evaluate(codec, row["genes"], audit_brief, geometry_samples=1024)
        audits.append(dict(coarse_evaluation=row["evaluation"], result=result))
    accepted = [a["result"] for a in audits if a["result"]["feasible"]]
    row = dict(unit=job["unit"], arm=job["arm"], seed=job["seed"], corners=codec.corners,
               brief_id=job["brief_id"], config=job, proposals=proposals, search_calls=calls,
               search_source_rays=calls*rays_per_call, audit_source_rays=sum(a["result"]["source_rays"] for a in audits),
               reserved_source_rays=max_calls*rays_per_call+job["audit_limit"]*len(audit_brief.rays()),
               counts=dict(counts), audited_success=bool(accepted), audited_designs=len(accepted),
               audited_routes=len({r["route"] for r in accepted}),
               best_audited_quality=max((r["quality"] for r in accepted), default=0.),
               first_coarse_success_rays=min((r["cumulative_search_rays"] for r in feasible), default=None),
               restarts=restarts, restart_sources=dict(restart_sources), lifetimes=lifetimes,
               boundary_interventions=boundary_hits, wall_seconds=time.monotonic()-started)
    row["stop_reason"] = ("call_budget" if calls >= max_calls else "proposal_budget"
                          if proposals >= max_proposals else "deadline")
    write(out/"archive.json", dict(candidates=candidates, audits=audits))
    write(out/"best_candidate.json", best)
    write(out/"summary.json", row)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stage", choices=("development", "confirmation"), required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--arms", choices=tuple(POLICIES), nargs="+", default=["naive", "pilot", "portfolio", "random"])
    p.add_argument("--corners", type=int, choices=(3,4,5), nargs="+", default=[3,4,5])
    p.add_argument("--briefs", nargs="+",
                   choices=("base", "wide", "shifted", "right_low", "right_mid",
                            "bottom_mid", "bottom_right", "patent_923", "patent_107"),
                   default=["base"])
    p.add_argument("--search-calls", type=int, default=500)
    p.add_argument("--search-rays-per-field", type=int,
                   help="override the brief's coarse pupil sampling density")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--candidate", choices=tuple(POLICIES))
    p.add_argument("--detector-beam-ratio", type=float, help="Full detector height / input beam width")
    p.add_argument("--setup-y-bounds", type=float, nargs=2, default=(-55., 50.))
    p.add_argument("--detector-height-fraction", type=float, default=1/3)
    args = p.parse_args(argv)
    if args.stage == "confirmation" and (args.candidate not in args.arms or "naive" not in args.arms or args.candidate == "naive"):
        p.error("confirmation requires a predeclared candidate and naive control")
    if (min(args.search_calls, args.workers) < 1
            or args.search_rays_per_field is not None and args.search_rays_per_field < 5
            or any(len(set(x)) != len(x) for x in (args.seeds,args.arms,args.corners,args.briefs))):
        p.error("positive counts and unique units required")
    # Variations are declared before results. Receiver and acceptance gates are fixed.
    briefs = {
        "base": RoutingBrief(),
        "wide": RoutingBrief(beam_width_mm=6.),
        "shifted": RoutingBrief(source_center=(58.,50.), fields_deg=(-1.,0.,1.)),
        "right_low": RoutingBrief(detector_x=120., detector_center_y=-20.,
                                   detector_half_height=5.),
        "right_mid": RoutingBrief(detector_x=120., detector_center_y=-2.5,
                                   detector_half_height=5.),
        "bottom_mid": RoutingBrief(detector_axis=1, detector_travel_sign=-1,
                                    detector_position=-55., detector_center=60.,
                                    detector_half_length=5.),
        "bottom_right": RoutingBrief(detector_axis=1, detector_travel_sign=-1,
                                      detector_position=-55., detector_center=80.,
                                      detector_half_length=5.),
        # Layout-scale analogues, not prescription reproductions.  The 923 brief uses
        # an eyebrow-plane source and an 18 mm-class side receiver; the 107 brief
        # rotates the fold into the temple plane and uses its larger 12 mm pupil.
        "patent_923": RoutingBrief(source_center=(55.,18.), aim_deg=-90.,
                                    detector_x=78., detector_center_y=0.,
                                    detector_half_height=5.,
                                    source_clearance_bounds=(3.,10.),
                                    detector_clearance_bounds=(16.,22.)),
        "patent_107": RoutingBrief(beam_width_mm=4.8, source_center=(73.,0.), aim_deg=180.,
                                    detector_axis=1, detector_travel_sign=-1,
                                    detector_position=-32., detector_center=55.,
                                    detector_half_length=6.,
                                    source_clearance_bounds=(3.,10.),
                                    detector_clearance_bounds=(20.,28.)),
    }
    codecs = {name: {k: PolygonCodec(k) for k in args.corners} for name in briefs}
    for k in args.corners:
        common = dict(center_x=(51.,59.), center_y=(-4.,4.), radial_fraction=(.75,1.),
                      angular_jitter_fraction=.15, max_sag_ratio=.268,
                      min_radius_mm=8., min_edge_mm=4., n=1.60)
        codecs["patent_923"][k] = PolygonCodec(k, half_width=(6.,10.),
                                               half_height=(10.,16.), **common)
        codecs["patent_107"][k] = PolygonCodec(k, half_width=(10.,16.),
                                               half_height=(6.,10.), **common)
    if args.search_rays_per_field is not None:
        briefs = {name: replace(brief, rays_per_field=args.search_rays_per_field)
                  for name, brief in briefs.items()}
    if args.detector_beam_ratio is not None:
        bottom, top = args.setup_y_bounds
        if (not np.isfinite([bottom, top, args.detector_beam_ratio, args.detector_height_fraction]).all()
                or bottom >= top or args.detector_beam_ratio <= 0
                or not 0 <= args.detector_height_fraction <= 1):
            p.error("invalid detector ratio or fixed setup bounds/fraction")
        briefs = {name: replace(brief,
                  detector_center_y=bottom+(top-bottom)*args.detector_height_fraction,
                  detector_half_height=args.detector_beam_ratio*brief.beam_width_mm/2)
                  for name, brief in briefs.items()}
    jobs = []
    for name in args.briefs:
        for k in args.corners:
            for seed in args.seeds:
                unit = f"{name}_k{k}_seed{seed}"
                for arm in args.arms:
                    jobs.append(dict(unit=unit, brief_id=name, arm=arm, seed=seed,
                                     codec=asdict(codecs[name][k]), brief=asdict(briefs[name]),
                                     policy=POLICIES[arm], search_calls=args.search_calls,
                                     popsize=10, audit_rays=25, audit_limit=8,
                                     out=str(args.out/f"{unit}_{arm}")))
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[1]
    sources = [*sorted(Path(__file__).parent.glob("*.py")), root/"search"/"cma_me.py",
               *sorted((root/"freeform_body").glob("*.py"))]
    for source in sources:
        destination = args.out/"source_snapshot"/source.relative_to(root.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    write(args.out/"protocol.json", dict(stage=args.stage, candidate=args.candidate, jobs=jobs,
          detector_layout=dict(beam_ratio=args.detector_beam_ratio,
                               setup_y_bounds=args.setup_y_bounds,
                               height_fraction=args.detector_height_fraction,
                               convention="Fixed reference envelope; does not vary with candidate geometry"),
          legacy_baseline="legacy_route adapts historical archive-improvement ranking and immediate restarts to route niches, normalized polygon genes, one emitter and common sigma; not an exact historical grid/driver replay",
          primary="paired run success: at least one dense-audited folded design",
          secondary=["distinct audited routes", "best audited folded throughput", "actual source rays", "wall time"],
          acceptance="Complete confirmation cohort: paired 95% seed-block bootstrap interval above zero and one-sided seed sign p < .05 against naive.",
          statistics=dict(block="seed, averaging all declared corner/brief tasks", bootstrap_samples=20000,
                          bootstrap_seed=20260911, primary_control="naive", secondary_comparisons="exploratory"),
          limitations="Inference over declared tasks/seeds, not all optics; no per-candidate independence claim.",
          sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}))
    summaries = []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            summaries.append(row)
            print(row["unit"], row["arm"], "audit", row["audited_designs"], "routes", row["audited_routes"], flush=True)
    write(args.out/"summary.json", sorted(summaries,key=lambda r:(r["unit"],r["arm"])))


if __name__ == "__main__":
    main()
