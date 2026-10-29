"""Build the ray-path catalogue: the best design ever found for EVERY interaction topology.

The archive keeps one design per (tir_count, thinness) cell. Measured across all 45 traces
that is 4-49 cells holding 804 distinct dominant signatures -- 41.7 topologies compete for
one cell and the loser is deleted. Six topologies that reached fitness > 0.6 were never
archived at all. This script recovers them.

Two passes:

  1. scan runs/*/seed*/trace.jsonl, group by FULL dominant signature, keep the highest-
     fitness genotype for each.
  2. re-evaluate every survivor through the engine under ONE hardened config (below), on a
     single canonical field set, and capture the central-field ray polylines for drawing.

Pass 2 is the point. The trace fitnesses were produced under mixed configs -- different
field sets, different gates, some with det_clearance unset -- so they are not comparable to
each other and some of them are the known exploits (body through the sensor, bulb faces,
knife-edge TIR). A topology only earns a catalogue entry on its HARDENED score.

    python ns_probe/build_catalogue.py --out runs/catalogue.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.engine import Runner                          # noqa: E402
from nslens.qd.driver import ExperimentConfig, _evaluate  # noqa: E402
from nslens.qd.observation import measure                 # noqa: E402
from nslens.qd.problems import FoldProblem                # noqa: E402

#: the yardstick. Every catalogue entry is scored under exactly this, whatever config the
#: run that discovered it used. Gates ON -- see 18_rl_engine_reward_plan.md section 2.1 for
#: why each one is here and which exploit it closes.
CANON_FIELDS = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)


#: A genotype means nothing without the Problem that produced it. `relative_sag` runs
#: carry sag genes as FRACTIONS of the face's own half-chord, not millimetres -- read one
#: of those under the absolute interpretation and a 0.21 gene becomes a 0.21 mm sag, which
#: silently flattens the element and rejects a design that is actually fine. The 3-curved
#: runs carry a tenth gene (`dsag`) that the 9-gene layout drops entirely.
#:
#: So the interpretation is taken from the RUN, never guessed from its directory name:
#: archive.json records meta.experiment (= ExperimentConfig.label) and meta.param_names.
#: Only the gates are overridden, by `harden()` below. A hardcoded run-name set was the
#: previous approach and it silently mis-read every run added after it was written.
def run_problem(run_dir: Path):
    """The Problem that produced `run_dir`, from the run's own recorded metadata."""
    metas = sorted(run_dir.glob("seed*/archive.json"))
    if not metas:
        return FoldProblem(), "no-archive"
    meta = json.loads(metas[0].read_text()).get("meta", {})
    label = meta.get("experiment")
    from nslens.qd import experiments
    for key in experiments.names():
        cfg = experiments.get(key)
        if cfg.label == label:
            return cfg.problem, key
    # Config no longer in the registry (the nd1.510 FOV sweeps). Rebuild what the genome
    # layout tells us and take the pre-relative_sag default, which is what those runs used.
    names = meta.get("param_names")
    return (FoldProblem(param_names=list(names)) if names else FoldProblem()), "fallback"


def harden(problem):
    """Same genome, catalogue gates. This is the yardstick every entry is scored under."""
    return replace(
        problem,
        n_rays=17,
        max_sag_ratio=0.268,     # no bulb faces (legacy permits q up to 5.0)
        det_clearance=12.0,      # body must stop short of the detector plane
    )


def hardened_cfg(problem: FoldProblem) -> ExperimentConfig:
    return ExperimentConfig(
        name="catalogue_hardened",
        problem=problem,
        fitness="focused_throughput",
        descriptors=("tir_bounces", "thinness"),
        fields=CANON_FIELDS,
        field_aggregate="worst",             # not "mean": mean hides off-axis blindness
        require_monotone_convergence=True,
        convergence_gate="clean_pack",       # width-only "packing" admits caustics
        gate_all_fields=True,                # beam metrics at every field, not just axis
        # clean_pack stops at the last face -- it never looks at the exit -> detector leg,
        # so a design can "converge" inside the glass and still land a diverging blur on
        # the sensor. Measured on the first catalogue: 3 of its 13 PASS entries fail this.
        require_exit_convergence=True,
        info_descriptors=("thinness", "crit_margin", "ns_frac", "field_accept"),
    )


# --------------------------------------------------------------------------- pass 1
def scan_traces(root: Path, min_fitness: float, min_seen: int = 10**9, per_sig: int = 6):
    """signature -> the `per_sig` best-scoring genotypes for it, over every trace on disk.

    Not one genotype per signature. Trace fitness is NOT comparable across runs -- each
    was produced under its own field set and gates -- so the highest-trace-fitness
    genotype for a signature is frequently not the one that survives the hardened config.
    Keeping only it means adding a new run can DELETE a passing design from the catalogue:
    measured on the sigma=0.05 arms, 3 of 13 passing entries were evicted that way by
    higher-scoring genotypes that the yardstick then rejected. Pass 2 rescores all of
    them and pass 3 keeps whichever actually passes.
    """
    best = collections.defaultdict(list)
    seen = collections.Counter()
    rows = 0
    for f in sorted(glob.glob(str(root / "runs/*/seed*/trace.jsonl"))):
        run, seed = f.split("/")[-3], f.split("/")[-2]
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            info = r.get("info") or {}
            sig = info.get("dom_signature")
            if not sig or r.get("fitness") is None:
                continue
            rows += 1
            seen[sig] += 1
            cand = best[sig]
            cand.append(dict(
                signature=sig,
                trace_fitness=float(r["fitness"]),
                genotype=[float(v) for v in r["genotype"]],
                tir_count=info.get("tir_count"),
                trace_thinness=info.get("thinness_T"),
                source_run=run, source_seed=seed,
                source_gen=r.get("gen"), source_eval=r.get("eval"),
            ))
            if len(cand) > 8 * per_sig:      # amortised trim, keeps memory flat
                cand.sort(key=lambda c: -c["trace_fitness"])
                del cand[per_sig:]
    for sig, cand in best.items():
        cand.sort(key=lambda c: -c["trace_fitness"])
        del cand[per_sig:]
        for rec in cand:
            rec["times_seen"] = seen[sig]
    # A catalogue is about TOPOLOGY COVERAGE, not just top scores: a path that only ever
    # reaches 0.1 is still a real route through the glass and still belongs in the
    # vocabulary the inverse model is asked to hit. Keep a signature if it scored well
    # OR if the search met it often enough that it is not noise (singletons top out at
    # 0.248 and none exceeds 0.3, so repeat-sightings is the honest anti-noise filter).
    kept = {s: cand for s, cand in best.items()
            if cand[0]["trace_fitness"] >= min_fitness or seen[s] >= min_seen}
    return kept, rows, len(best)


# --------------------------------------------------------------------------- pass 2
def rescore(args):
    """Re-evaluate a chunk of candidates under the hardened config. Returns entries."""
    chunk, = args
    # one problem per SOURCE RUN -- the genome layout follows the run that produced the
    # genotype; only the gates are common. Cached because resolving reads an archive.json.
    cache = {}
    out = []
    with Runner(timeout_s=20) as runner:
        for rec in chunk:
            x = np.array(rec["genotype"], float)
            entry = dict(rec)
            run = rec["source_run"]
            if run not in cache:
                base, src = run_problem(_REPO / "runs" / run)
                cache[run] = (harden(base), src)
            problem, src = cache[run]
            cfg = hardened_cfg(problem)
            entry["relative_sag"] = bool(getattr(problem, "relative_sag", False))
            entry["genome_source"] = src
            res = _evaluate(runner, problem, cfg, x)
            if res is None:
                # Rejected by a gate is NOT a reason to skip the drawing. A topology the
                # hardened config refuses is still a real route through the glass, and
                # seeing WHY it was refused -- glass through the detector plane, a bulb
                # face, a bundle blind at one field -- is most of the value of keeping it.
                entry.update(hardened_fitness=None, verdict="REJECTED")
                _attach_geometry(runner, problem, x, entry)
                out.append(entry)
                continue
            info = res.get("info", {})
            entry.update(
                hardened_fitness=float(res["fitness"]),
                verdict="PASS",
                bc=[float(v) for v in res["bc"]],
                hardened_signature=info.get("dom_signature"),
                per_field=info.get("per_field"),
                worst_focthru=info.get("worst_focthru"),
                mean_focthru=info.get("mean_focthru"),
                thinness=info.get("thinness_T"),
                crit_margin=info.get("crit_margin"),
                ns_frac=info.get("ns_frac"),
                field_accept=info.get("field_accept"),
                spot=info.get("central_spot"),
            )
            _attach_geometry(runner, problem, x, entry)
            out.append(entry)
    return out


def _attach_geometry(runner, problem, x, entry) -> None:
    """Body outline + central-field dominant-bundle polylines, for drawing."""
    p = problem.vec_to_params(x)
    entry["params"] = {k: float(v) for k, v in p.items()}
    try:
        entry["outline"] = problem.element(p).path()
    except Exception:
        entry["outline"] = []
    try:
        raw, _ = runner.simulate(problem.layout(p, field_deg=0.0).to_scene_dict())
        obs = measure(raw, problem.face_of(p), x_sensor=problem.x_sensor,
                      allow_empty=True)
        rays = []
        if obs is not None:
            for r in obs.dominant_rays(min_rays=3):
                rays.append([[round(float(v["x"]), 3), round(float(v["y"]), 3)]
                             for v in r["seq"]])
        entry["rays"] = rays
    except Exception:
        entry["rays"] = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/catalogue.json")
    ap.add_argument("--min-fitness", type=float, default=0.20,
                    help="trace-fitness floor for pass 2. Singleton signatures top out "
                         "at 0.248 and none exceeds 0.3, so this cuts search noise.")
    ap.add_argument("--min-seen", type=int, default=10,
                    help="also keep any signature the search met at least this often, "
                         "whatever it scored -- topology coverage, not just top scores.")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--per-signature", type=int, default=6,
                    help="genotypes rescored per signature. 1 reproduces the old "
                         "behaviour, where a new run could evict a passing design.")
    a = ap.parse_args()

    t0 = time.time()
    cand, rows, n_all = scan_traces(_REPO, a.min_fitness, a.min_seen, a.per_signature)
    n_cand = sum(len(v) for v in cand.values())
    print(f"pass 1: {rows} scored trace rows, {n_all} distinct signatures, "
          f"{len(cand)} kept (fitness >= {a.min_fitness} or seen >= {a.min_seen}x), "
          f"{n_cand} genotypes to rescore (<= {a.per_signature} each)")

    items = sorted((r for v in cand.values() for r in v),
                   key=lambda r: -r["trace_fitness"])
    nw = max(1, min(a.workers, len(items)))
    chunks = [(items[i::nw],) for i in range(nw)]
    from multiprocessing import Pool
    with Pool(nw) as pool:
        scored = [e for part in pool.map(rescore, chunks) for e in part]

    # pass 3: one entry per signature -- the best design that PASSES, or, if none does,
    # the best-trace rejected one so the topology still appears with its reason to look at.
    by_sig = collections.defaultdict(list)
    for e in scored:
        by_sig[e["signature"]].append(e)
    entries = []
    for sig, group in by_sig.items():
        ok = [e for e in group if e["verdict"] == "PASS"]
        pick = max(ok, key=lambda e: e["hardened_fitness"]) if ok else \
            max(group, key=lambda e: e["trace_fitness"])
        pick["n_genotypes_tried"] = len(group)
        pick["n_genotypes_passing"] = len(ok)
        entries.append(pick)

    entries.sort(key=lambda e: -(e["hardened_fitness"] or -1))
    passed = [e for e in entries if e["verdict"] == "PASS"]
    survived = [e for e in passed if e.get("hardened_signature") == e["signature"]]

    out = dict(
        meta=dict(
            built=time.strftime("%Y-%m-%d %H:%M:%S"),
            canonical_fields=list(CANON_FIELDS),
            hardened=dict(field_aggregate="worst", max_sag_ratio=0.268,
                          det_clearance=12.0, convergence_gate="clean_pack",
                          gate_all_fields=True, require_exit_convergence=True,
                          n_rays=17),
            scored_trace_rows=rows,
            distinct_signatures_all=n_all,
            candidates=len(entries),
            genotypes_rescored=len(items),
            per_signature=a.per_signature,
            min_fitness=a.min_fitness, min_seen=a.min_seen,
            passed_hardened=len(passed),
            topology_stable=len(survived),
            wall_seconds=round(time.time() - t0, 1),
        ),
        entries=entries,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"pass 2: {len(items)} genotypes rescored -> {len(passed)}/{len(entries)} "
          f"signatures have a design that passes the hardened config; "
          f"{len(survived)} keep their topology under it")
    print(f"wrote {a.out}  ({Path(a.out).stat().st_size/1e6:.1f} MB, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
