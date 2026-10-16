"""Replay saved optical-potential CMA trials and retain their coarse trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nslens.analysis.mirror_spec_experiment import crosses
from nslens.analysis.variable_mirror import trace
from nslens.search.cma_me import CMAES

SOURCE = Path("experiments/optical_potential_20261005")
POPULATION = 8
BUDGET = 256
CHECKPOINTS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
               112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
               200, 208, 216, 224, 232, 240, 248, 256)


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    return value


def _save(path: Path, value):
    path.write_text(json.dumps(_serial(value), indent=2, allow_nan=False) + "\n")


def jobs(out: Path):
    result = []
    original_seed_universe = set()
    sources = []
    for path in sorted(SOURCE.glob("*__*__*__*.json")):
        if path.name in {"protocol.json", "verification.json"}:
            continue
        try:
            source = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "trials" in source and "parent" in source:
            original_seed_universe.update(int(t["seed"]) for t in source["trials"])
            sources.append(path)
    for path in sources:
        raw = path.read_bytes()
        source = json.loads(raw)
        if "trials" not in source or "parent" not in source:
            continue
        if source["initial_audit"]["imaging_pass"]["1.0"] is not False:
            continue
        job = source["job"]
        result.append({
            "source": str(path),
            "source_hash": hashlib.sha256(raw).hexdigest(),
            "taskname": job["taskname"],
            "task": job["task"],
            "arm": job["arm"],
            "n": int(job["n"]),
            "seed": int(job["seed"]),
            "out": str(out),
        })
    return result


def _check_identity(actual, expected, label):
    actual_genes = np.asarray(actual["genes"], dtype=float)
    expected_genes = np.asarray(expected["genes"], dtype=float)
    if actual_genes.shape != expected_genes.shape or not np.allclose(
            actual_genes, expected_genes, rtol=0.0, atol=1e-10):
        raise AssertionError(f"Replay gene mismatch at {label}")
    if not np.isclose(float(actual["loss"]), float(expected["loss"]),
                      rtol=0.0, atol=1e-10):
        raise AssertionError(f"Replay loss mismatch at {label}")


def run(job):
    started = time.monotonic()
    source_path = Path(job["source"])
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != job["source_hash"]:
        raise ValueError(f"Source changed during replay: {source_path}")
    source = json.loads(source_bytes)
    n, task = job["n"], job["task"]
    initial = source["parent"]["raw"]
    calls = 0

    def evaluate(genes):
        nonlocal calls
        row = trace(genes, n, fields=3, pupils=9, **task)
        row["crossing"] = crosses(genes, n, 33)
        row["loss"] += 30000 * row["crossing"]
        calls += 1
        return row

    trials = []
    for repeat, original in enumerate(source["trials"]):
        seed = int(job["trial_seed_plan"][repeat]["seed"])
        cma = CMAES(np.asarray(initial["genes"], dtype=float), .03,
                    popsize=POPULATION, seed=seed,
                    bounds=(-np.ones(3*n), np.ones(3*n)),
                    boundary_handling="reflect")
        incumbent = initial
        checkpoints = {}
        used = 0
        while used < BUDGET:
            population = cma.ask()
            rows = [evaluate(candidate) for candidate in population]
            cma.tell(np.argsort([row["loss"] for row in rows]))
            candidate = min(rows, key=lambda row: row["loss"])
            if candidate["loss"] < incumbent["loss"]:
                incumbent = candidate
            used += POPULATION
            if used in CHECKPOINTS:
                checkpoints[str(used)] = {"coarse": incumbent}
            if used in (64, 256) and not job["fresh_seeds"]:
                _check_identity(incumbent,
                                original["checkpoints"][str(used)]["coarse"],
                                f"{source_path.name} seed={seed} eval={used}")
        trials.append({"seed": seed,
                       "original_seed": int(original["seed"]),
                       "checkpoints": checkpoints})

    expected_calls = len(source["trials"]) * BUDGET
    assert calls == expected_calls, (calls, expected_calls)
    record = {
        "job": {k: job[k] for k in ("source", "taskname", "task", "arm", "n", "seed")},
        "source_hash": source_hash,
        "seed_mode": "fresh" if job["fresh_seeds"] else "original_replay",
        "source_snapshot": {"path": str(source_path), "sha256": source_hash},
        "task": task,
        "initial_audit": source["initial_audit"],
        "features": source["features"],
        "parent": {"raw": initial},
        "trials": trials,
        "coarse_calls": calls,
        "dense_calls": 0,
        "probe_calls": 0,
        "seconds": time.monotonic() - started,
    }
    out_path = Path(job["out"]) / source_path.name
    _save(out_path, record)
    return {"task": job["taskname"], "arm": job["arm"], "n": n,
            "seed": job["seed"], "coarse_calls": calls,
            "dense_calls": 0, "seconds": record["seconds"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh-seeds", action="store_true",
                         help="Use deterministic new CMA seeds instead of replaying saved seeds.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    selected = jobs(out)
    original_seed_universe = {
        int(t["seed"])
        for path in SOURCE.glob("*__*__*__*.json")
        if path.name not in {"protocol.json", "verification.json"}
        for source in [json.loads(path.read_text())]
        if "trials" in source and "parent" in source
        for t in source["trials"]
    }
    new_seed_universe = set()
    for job in selected:
        original = json.loads(Path(job["source"]).read_text())
        plan = []
        for repeat, trial in enumerate(original["trials"]):
            original_seed = int(trial["seed"])
            if args.fresh_seeds:
                seed_payload = (f"potential-trajectories-fresh-v1/"
                                f"{job['source_hash']}/{job['taskname']}/"
                                f"{job['arm']}/{job['seed']}/{job['n']}/{repeat}")
                seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:16], 16)
            else:
                seed = original_seed
            plan.append({"seed": seed, "original_seed": original_seed})
            if args.fresh_seeds:
                assert seed not in original_seed_universe, f"Fresh seed overlaps original seed: {seed}"
                assert seed not in new_seed_universe, f"Fresh seed collision: {seed}"
                new_seed_universe.add(seed)
        job["trial_seed_plan"] = plan
        job["fresh_seeds"] = args.fresh_seeds
    if args.smoke:
        selected = selected[:1]
    _save(out / "protocol.json", {
        "jobs": selected,
        "purpose": ("Run independent seeded CMA refinements and retain incumbent trajectories." if args.fresh_seeds else
                    "Replay the already-observed CMA trial seeds from the optical-potential dataset and expose incumbent trajectories. These are counterfactual records of existing outcomes, not independent new trials."),
        "seed_mode": "fresh independent seeds" if args.fresh_seeds else "original outcome replay",
        "selection": "Only configurations whose inherited initial_audit.imaging_pass['1.0'] is false are replayed.",
        "replay": "Each saved trial seed starts from the inherited parent.raw genes. CMA-ES uses population 8, sigma 0.03, reflect bounds, and the original coarse variable_mirror.trace task settings. Candidate loss is the legacy trace loss plus 30000 times crosses(genes,n,33). The incumbent includes the inherited parent and every evaluated candidate.",
        "checkpoints": list(CHECKPOINTS),
        "identity_checks": ("Skipped for fresh seeds; deterministic fresh seeds are asserted disjoint from all original trial seeds." if args.fresh_seeds else
                            "At evaluations 64 and 256, replayed incumbent genes and penalized loss must match the original trial checkpoint within absolute tolerance 1e-10."),
        "cost": ("768 coarse traces per configuration (3 fresh seeds x 256 evaluations); no probes or dense audits. Parent, task, initial audit, and features are inherited." if args.fresh_seeds else
                 "768 newly executed coarse traces per configuration (3 existing seeds x 256 evaluations); no probes or dense audits. Parent, task, initial audit, and features are inherited. The 768 traces recompute already-existing outcomes for trajectory retention; they do not increase the independent trial count."),
        "input_configuration_count": len(jobs(out)),
        "replayed_configuration_count": len(selected),
        "original_seed_universe_size": len(original_seed_universe),
        "fresh_seed_count": len(new_seed_universe),
    })
    for path in (Path(__file__), Path("nslens/analysis/variable_mirror.py"),
                 Path("nslens/search/cma_me.py"),
                 Path("nslens/analysis/mirror_spec_experiment.py")):
        shutil.copy2(path, out / path.name)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, selected):
            print(json.dumps(result), flush=True)
    _save(out / "verification.json", {
        "configurations": len(selected),
        "trials_per_configuration": 3,
        "coarse_calls": len(selected) * 3 * BUDGET,
        "dense_calls": 0,
        "probe_calls": 0,
        "identity_checks_per_trial": [64, 256],
        "all_identity_checks_passed": None if args.fresh_seeds else True,
        "identity_checks_skipped": bool(args.fresh_seeds),
        "fresh_seed_count": len(new_seed_universe),
        "fresh_seeds_disjoint_from_original": (not bool(new_seed_universe & original_seed_universe)
                                                if args.fresh_seeds else None),
        "interpretation": ("Independent fresh-seed trajectories; not replayed source outcomes." if args.fresh_seeds else
                           "Counterfactual replay of existing trial outcomes; not new independent optimizer trials."),
    })


if __name__ == "__main__":
    main()
