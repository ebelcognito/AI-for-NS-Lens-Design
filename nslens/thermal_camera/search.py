"""Independent staged CMA search for the 3D thermal-camera model."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np

from nslens.search.cma_me import CMAES
from . import model


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def _retain(archive, candidate, capacity=12):
    genes = candidate["genes"]
    near = next((i for i, row in enumerate(archive)
                 if np.linalg.norm(genes-row["genes"])/np.sqrt(len(genes)) < .08), None)
    if near is None:
        archive.append(candidate)
    elif candidate["metrics"]["loss"] < archive[near]["metrics"]["loss"]:
        archive[near] = candidate
    archive.sort(key=lambda row: row["metrics"]["loss"])
    del archive[capacity:]


def run_seed(config):
    out = Path(config["out"]) / f"seed_{config['seed']}"
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config["seed"])
    started = time.monotonic()
    train = dict(fields=config["fields"], waves=config["waves"], pupils=config["pupils"])
    history, stages, best = [], [], None
    total = 0

    # This is scalar merit optimization; CMA-ME would only be useful for a
    # deliberate multi-descriptor quality-diversity search.
    for stage, dimensions, budget in (("base", model.N_BASE, config["base_evals"]),
                                      ("asphere", model.N_TOTAL, config["freeform_evals"])):
        if budget < config["popsize"]:
            continue
        archive = []
        if best is not None:
            _retain(archive, best)
        emitter = None
        stale = restarts = stage_count = 0
        first_population = []
        while stage_count + config["popsize"] <= budget:
            if emitter is None or emitter.restart_needed or stale >= config["restart_generations"]:
                if archive and rng.random() < .8:
                    parent = archive[int(rng.integers(min(4, len(archive))))]["genes"][:dimensions]
                    sigma = .10 if stage == "asphere" else .18
                else:
                    parent = rng.uniform(-.8, .8, dimensions)
                    sigma = .25
                emitter = CMAES(parent, sigma, popsize=config["popsize"],
                                seed=int(rng.integers(2**31)),
                                bounds=(-np.ones(dimensions), np.ones(dimensions)),
                                boundary_handling="reflect")
                stale = 0
                restarts += 1

            samples = emitter.ask()
            sampled = emitter.sampling_snapshot()
            rows = []
            previous = best["metrics"]["loss"] if best else np.inf
            for sample in samples:
                genes = np.zeros(model.N_TOTAL)
                genes[:dimensions] = sample
                metrics = model.evaluate(genes, **train)
                loss = float(metrics["loss"])
                if not np.isfinite(loss):
                    raise RuntimeError("The evaluator returned a non-finite loss")
                row = dict(genes=genes.copy(), metrics=metrics)
                rows.append(row)
                _retain(archive, row)
                if best is None or loss < best["metrics"]["loss"]:
                    best = row
            if not first_population:
                first_population = [dict(genes=r["genes"].copy(), metrics=r["metrics"]) for r in rows]

            emitter.tell(np.argsort([row["metrics"]["loss"] for row in rows]))
            stage_count += len(rows)
            total += len(rows)
            stale = stale + 1 if best["metrics"]["loss"] >= previous else 0
            rec = dict(stage=stage, evaluations=total, stage_evaluations=stage_count,
                       elapsed_seconds=time.monotonic()-started, restarts=restarts,
                       best_loss=float(best["metrics"]["loss"]), best_metrics=best["metrics"])
            history.append(rec)
            write_json(out/"checkpoint.json", dict(
                seed=config["seed"], genes=best["genes"], metrics=best["metrics"],
                evaluations=total, stage=stage, history=history,
                cma_state=emitter.state_snapshot(), last_sampling=sampled,
                last_population=[dict(genes=r["genes"], metrics=r["metrics"]) for r in rows]))
            if len(history) % 10 == 0:
                print(json.dumps(serial(dict(seed=config["seed"], **rec))), flush=True)

        stages.append(dict(stage=stage, dimensions=dimensions, requested_budget=budget,
                           evaluations=stage_count, restarts=restarts,
                           initial_population=first_population,
                           best=dict(genes=best["genes"].copy(), metrics=dict(best["metrics"]))))

    if best is None:
        raise ValueError("At least one stage must allow a full population")
    audit_sampling = dict(fields=config["audit_fields"], waves=config["audit_waves"],
                          pupils=config["audit_pupils"])
    audit = model.evaluate(best["genes"], **audit_sampling)
    base = next((row["best"] for row in stages if row["stage"] == "base"), None)
    base_audit = model.evaluate(base["genes"], **audit_sampling) if base else None
    result = dict(seed=config["seed"], train_sampling=train, audit_sampling=audit_sampling,
                  evaluations=total, audit_evaluations=1+int(base is not None),
                  elapsed_seconds=time.monotonic()-started, stages=stages,
                  genes=best["genes"], training=best["metrics"], audit=audit,
                  base_audit=base_audit, history=history, decoded=model.decode(best["genes"]),
                  interpretation=("Single-channel axisymmetric Schmidt-Cassegrain prototype; "
                                  "eight-channel steering is an idealized coverage construction. "
                                  "Same-model audit; no optical or manufacturing certification."))
    write_json(out/"result.json", result)
    print(json.dumps(serial(dict(seed=config["seed"], complete=True,
                                 evaluations=total, audit=audit))), flush=True)
    return serial(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--base-evals", type=int, default=1200)
    parser.add_argument("--freeform-evals", type=int, default=2400)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--restart-generations", type=int, default=18)
    parser.add_argument("--fields", type=int, default=3)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--pupils", type=int, default=5)
    parser.add_argument("--audit-fields", type=int, default=9)
    parser.add_argument("--audit-waves", type=int, default=7)
    parser.add_argument("--audit-pupils", type=int, default=9)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    if min(args.base_evals, args.freeform_evals) < 0 or args.popsize < 4 or args.workers < 1:
        parser.error("Budgets must be nonnegative, popsize >= 4 and workers >= 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    configuration = vars(args)
    write_json(out/"configuration.json", configuration)
    shutil.copyfile(Path(model.__file__), out/"model_snapshot.py")
    shutil.copyfile(Path(__file__), out/"search_snapshot.py")
    configs = [dict(configuration, seed=seed) for seed in args.seeds]
    started = time.monotonic()
    if args.workers == 1:
        results = [run_seed(config) for config in configs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(configs))) as pool:
            results = list(pool.map(run_seed, configs))
    summary = dict(configuration=configuration, wall_seconds=time.monotonic()-started,
                   interpretation=("Staged scalar CMA over a fixed idealized Schmidt-Cassegrain "
                                   "architecture; dense same-engine audit, not independent "
                                   "optical verification or fabrication certification."),
                   runs=[{key: result[key] for key in
                          ("seed", "evaluations", "elapsed_seconds", "training", "audit", "base_audit")}
                         for result in results])
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
