"""Fresh-seed confirmation of selected optical refinement-potential reversals."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

from nslens.analysis.mirror_residual_experiment import refine, save
from nslens.analysis.mirror_spec_experiment import audit, crosses
from nslens.analysis.variable_mirror import trace

REPEATS = 24
BUDGET = 256
SEED_BASE = 60_000_000


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return [math.nan, math.nan]
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _load_pair_jobs(source_out: Path, selected: list[dict]) -> list[dict]:
    jobs = []
    for pair_index, pair in enumerate(selected):
        for side in ("worse_initial", "better_initial"):
            parent_job = pair[side]
            source_path = Path(parent_job["source"])
            source_bytes = source_path.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            expected_hash = parent_job.get("source_hash")
            if expected_hash and source_hash != expected_hash:
                raise ValueError(f"Source hash mismatch for {source_path}")
            source = json.loads(source_bytes)
            n = int(parent_job["n"])
            task = parent_job["task"]
            if source["task"] != task:
                raise ValueError(f"Task mismatch for {source_path}")
            genes = source["coarse"][str(n)]["genes"]
            if len(genes) != 3 * n:
                raise ValueError(f"Expected {3*n} starting genes in {source_path} for n={n}")
            jobs.append({
                "pair_index": pair_index,
                "side": side,
                "repeat": None,
                "source": str(source_path),
                "source_hash": source_hash,
                "source_out": str(source_out),
                "taskname": parent_job["taskname"],
                "task": task,
                "arm": parent_job["arm"],
                "donor_seed": int(parent_job["seed"]),
                "n": n,
                "initial_genes": genes,
                "initial_audit": source["audits"][str(n)],
                "pair": pair,
            })
    jobs = [job | {"repeat": repeat, "seed": SEED_BASE + job["pair_index"] * 1000 + repeat}
            for job in jobs for repeat in range(REPEATS)]
    return jobs


def _assert_fresh_seeds(source_out: Path, jobs: list[dict]) -> None:
    old_seeds = set()
    for path in source_out.glob("*.json"):
        if path.name in {"rank_reversals.json", "protocol.json", "summary.json"}:
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        for trial in record.get("trials", []):
            if "seed" in trial:
                old_seeds.add(int(trial["seed"]))
    overlap = old_seeds.intersection(int(job["seed"]) for job in jobs)
    if overlap:
        raise ValueError(f"Confirmation seeds overlap earlier optimizer trials: {sorted(overlap)}")


def run(job: dict) -> dict:
    started = time.monotonic()
    n = job["n"]
    task = job["task"]
    x = np.asarray(job["initial_genes"], dtype=float)
    calls = 0

    def evaluate(genes):
        nonlocal calls
        row = trace(genes, n, **task)
        calls += 1
        bad = crosses(genes, n, 33)
        row["optical_loss"] = row["loss"]
        row["loss"] += 30000 * bad
        row["crossing"] = bad
        return row

    initial = evaluate(x)
    final = refine(evaluate, x, initial, BUDGET, int(job["seed"]))
    final_audit = audit(final["genes"], n, task)
    assert calls == BUDGET + 1, (calls, BUDGET + 1)
    assert final_audit["crossing"] is not None
    success = bool(final_audit["imaging_pass"]["1.0"])

    record = {
        "job": {k: job[k] for k in ("pair_index", "side", "repeat", "source", "source_hash", "source_snapshot",
                                    "taskname", "arm", "donor_seed", "n", "seed")},
        "task": task,
        "initial_genes": x,
        "final_genes": final["genes"],
        "initial_coarse": initial,
        "initial_audit": job["initial_audit"],
        "initial_audit_inherited": True,
        "final_coarse": final,
        "final_audit": final_audit,
        "success_1mm": success,
        "coarse_calls": calls,
        "dense_calls": 1,
        "seconds": time.monotonic() - started,
    }
    out = Path(job["out"])
    save(out / f"pair_{job['pair_index']:02d}__{job['side']}__rep_{job['repeat']:02d}.json", record)
    return {"pair_index": job["pair_index"], "side": job["side"], "repeat": job["repeat"],
            "seed": job["seed"], "success_1mm": success, "coarse_calls": calls,
            "dense_calls": 1, "seconds": record["seconds"]}


def summarize(results: list[dict], selected: list[dict]) -> dict:
    expected = len(selected) * 2 * REPEATS
    if len(results) != expected:
        raise AssertionError(f"Expected {expected} trial results, received {len(results)}")
    if any(r["coarse_calls"] != BUDGET + 1 or r["dense_calls"] != 1 for r in results):
        raise AssertionError("A trial failed the 257-coarse/1-dense call ledger")
    summaries = []
    for pair_index, pair in enumerate(selected):
        subset = [r for r in results if r["pair_index"] == pair_index]
        by_side = {side: [r for r in subset if r["side"] == side] for side in
                   ("worse_initial", "better_initial")}
        counts = {side: sum(bool(r["success_1mm"]) for r in rows)
                  for side, rows in by_side.items()}
        worse = {r["repeat"]: bool(r["success_1mm"]) for r in by_side["worse_initial"]}
        better = {r["repeat"]: bool(r["success_1mm"]) for r in by_side["better_initial"]}
        better_only = sum(better[i] and not worse[i] for i in range(REPEATS))
        worse_only = sum(worse[i] and not better[i] for i in range(REPEATS))
        discordant = better_only + worse_only
        exact = binomtest(better_only, discordant, p=0.5, alternative="two-sided") if discordant else None
        rate = {side: counts[side] / REPEATS for side in counts}
        summaries.append({
            "pair_index": pair_index,
            "taskname": pair["worse_initial"]["taskname"],
            "donor_seed": pair["worse_initial"]["seed"],
            "n_mirrors": {side: pair[side]["n"] for side in ("worse_initial", "better_initial")},
            "initial_losses": pair["initial_losses"],
            "initial_worst_rms_mm": pair["initial_worst_rms"],
            "successes_out_of_24": counts,
            "success_rate": rate,
            "wilson_95": {side: wilson_interval(counts[side], REPEATS) for side in counts},
            "paired_discordants": {"better_only": better_only, "worse_only": worse_only,
                                   "total": discordant},
            "exact_two_sided_binomtest_p": float(exact.pvalue) if exact else None,
            "pair_effect_better_minus_worse": rate["better_initial"] - rate["worse_initial"],
            "paired_repeat_count": REPEATS,
        })
    return {"pairs": summaries,
            "descriptive_pooled_successes_out_of": {
                side: sum(s["successes_out_of_24"][side] for s in summaries)
                for side in ("worse_initial", "better_initial")},
            "descriptive_pooled_trials": REPEATS * len(summaries),
            "call_ledger": {"trials": len(results), "coarse_calls": sum(r["coarse_calls"] for r in results),
                            "dense_calls": sum(r["dense_calls"] for r in results),
                            "expected_coarse_calls": expected * (BUDGET + 1),
                            "expected_dense_calls": expected}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out", required=True, help="Main optical-potential experiment directory")
    parser.add_argument("--out", required=True, help="New confirmation output directory")
    args = parser.parse_args()
    source_out = Path(args.source_out)
    rank_path = source_out / "rank_reversals.json"
    if not rank_path.is_file():
        parser.error(f"rank-reversal file not found: {rank_path}")
    ranking = json.loads(rank_path.read_text())
    selected = ranking.get("selected", [])
    if not selected:
        print("No selected rank-reversal pairs; nothing created.")
        return 0

    out = Path(args.out)
    if out.exists():
        parser.error(f"output directory already exists: {out}")
    jobs = _load_pair_jobs(source_out, selected)
    _assert_fresh_seeds(source_out, jobs)
    out.mkdir(parents=True, exist_ok=False)
    run_dir = out / "runs"
    run_dir.mkdir()
    snapshot_dir = out / "source_snapshots"
    snapshot_dir.mkdir()
    snapshots = {}
    for job in jobs:
        source_path = Path(job["source"])
        if source_path not in snapshots:
            snapshot_name = f"{source_path.stem}__{job['source_hash'][:12]}{source_path.suffix}"
            shutil.copyfile(source_path, snapshot_dir / snapshot_name)
            snapshots[source_path] = str(Path("source_snapshots") / snapshot_name)
        job["source_snapshot"] = snapshots[source_path]
        job["out"] = str(run_dir)

    protocol = {
        "selected_pairs": selected,
        "source_snapshots": {str(path): relative for path, relative in snapshots.items()},
        "rank_reversals_sha256": hashlib.sha256(rank_path.read_bytes()).hexdigest(),
        "repeats_per_side": REPEATS,
        "coarse_budget": BUDGET,
        "coarse_calls_per_trial": BUDGET + 1,
        "dense_calls_per_trial": 1,
        "seed_rule": "60000000 + pair_index*1000 + repeat; paired sides share a seed; disjoint from prior trial seeds",
        "optimizer": "mirror_residual_experiment.refine; CMA sigma=.03, popsize=8, reflected [-1,1] bounds",
        "objective": "variable_mirror.trace legacy loss plus 30000 for any 33-sample surface crossing; retain best coarse loss including initial parent",
        "endpoint": "mirror_spec_experiment.audit: <=1mm worst-field mapping RMS, >=99% worst-field support, no sampled crossings at 513 surface samples",
        "interpretation": "Posthoc-selected cases receive fresh optimizer seeds but reuse the same selected starting configurations; confirmation is conditional on these cases and does not estimate population-wide generalization.",
        "initial_audit": "Inherited audit from the source specification run; not counted as a new dense call.",
    }
    save(out / "protocol.json", protocol)

    completed = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, jobs):
            completed.append(result)
            print(json.dumps(result), flush=True)
    summary = summarize(completed, selected)
    save(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
