"""Warm-start scalar CMA refinement from a saved thermal-camera result.

This command searches only the existing 14-parameter architecture. It does not
run a local gradient or least-squares optimizer and does not claim prescription
recovery or hardware certification.
"""
from __future__ import annotations

import argparse
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


def refinement_evaluate(genes, *, fields=3, waves=1, pupils=7):
    """Evaluate the requested spot merit plus an explicit field-support penalty."""
    metrics = model.evaluate(genes, fields=fields, waves=waves, pupils=pupils)
    support = np.asarray(metrics["per_field_throughput"], dtype=float)
    coverage_penalty = 350.0 * float(np.mean(np.maximum(.62-support, 0.0)**2))
    spot_pixels = float(metrics["max_spot_mm"]) / .012
    spot_penalty = 10.0 * max(spot_pixels-1.0, 0.0)**2
    base_loss = float(metrics["loss"])
    exact_loss = base_loss + coverage_penalty + spot_penalty
    return dict(metrics, base_loss=base_loss, coverage_penalty=coverage_penalty,
                spot_penalty=spot_penalty, spot_pixels=spot_pixels,
                loss=float(exact_loss) if np.isfinite(exact_loss) else 1e8)


def run(source_path, out_path, *, budget=4800, seed=201, popsize=24):
    source_path, out = Path(source_path), Path(out_path)
    source = json.loads(source_path.read_text())
    source_genes = np.asarray(source["genes"], dtype=float).ravel()
    if source_genes.shape != (model.N_TOTAL,) or not np.isfinite(source_genes).all():
        raise ValueError(f"source genes must be a finite vector of length {model.N_TOTAL}")
    if np.any(source_genes < -1) or np.any(source_genes > 1):
        raise ValueError("source genes must lie within [-1, 1]")
    if budget < 0 or popsize < 4:
        raise ValueError("budget must be nonnegative and popsize >= 4")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_path, out/"source_result.json")
    shutil.copyfile(Path(model.__file__), out/"model_snapshot.py")
    shutil.copyfile(Path(__file__), out/"refine_snapshot.py")

    sampling = dict(fields=3, waves=1, pupils=7)
    started = time.monotonic()
    input_metrics = refinement_evaluate(source_genes, **sampling)
    best = dict(genes=source_genes.copy(), metrics=input_metrics)
    history = [dict(generation=0, stage="input", evaluations=0,
                    best_loss=float(input_metrics["loss"]), best_metrics=input_metrics)]
    rng = np.random.default_rng(seed)
    emitter = CMAES(source_genes, .06, popsize=popsize, seed=seed,
                    bounds=(-np.ones(model.N_TOTAL), np.ones(model.N_TOTAL)),
                    boundary_handling="reflect")
    generations = 0
    restarts = 0
    evaluations = 0
    while evaluations + popsize <= budget:
        samples = emitter.ask()
        sampled = emitter.sampling_snapshot()
        rows = []
        previous_best = float(best["metrics"]["loss"])
        for sample in samples:
            metrics = refinement_evaluate(sample, **sampling)
            row = dict(genes=sample.copy(), metrics=metrics)
            rows.append(row)
            if metrics["loss"] < best["metrics"]["loss"]:
                best = row
        emitter.tell(np.argsort([r["metrics"]["loss"] for r in rows]))
        evaluations += len(rows)
        generations += 1

        restarted = False
        if generations % 60 == 0 and evaluations + popsize <= budget:
            restarts += 1
            emitter = CMAES(best["genes"], .04, popsize=popsize,
                            seed=int(rng.integers(2**31)),
                            bounds=(-np.ones(model.N_TOTAL), np.ones(model.N_TOTAL)),
                            boundary_handling="reflect")
            restarted = True

        rec = dict(generation=generations, stage="cma", evaluations=evaluations,
                   elapsed_seconds=time.monotonic()-started, restarts=restarts,
                   restarted=restarted, generation_best_loss=float(min(r["metrics"]["loss"] for r in rows)),
                   previous_best_loss=previous_best,
                   best_loss=float(best["metrics"]["loss"]), best_metrics=best["metrics"])
        history.append(rec)
        write_json(out/"checkpoint.json", dict(
            seed=seed, genes=best["genes"], metrics=best["metrics"],
            evaluations=evaluations, input_evaluations=1, generation=generations,
            restarts=restarts, cma_state=emitter.state_snapshot(),
            last_sampling=sampled,
            last_population=[dict(genes=r["genes"], metrics=r["metrics"]) for r in rows],
            history=history))
        if generations % 10 == 0:
            print(json.dumps(serial(dict(seed=seed, **rec))), flush=True)

    audit_sampling = dict(fields=7, waves=1, pupils=17)
    audit = model.evaluate(best["genes"], **audit_sampling)
    audit_merit = refinement_evaluate(best["genes"], **audit_sampling)
    input_audit = model.evaluate(source_genes, **audit_sampling)
    input_audit_merit = refinement_evaluate(source_genes, **audit_sampling)
    config = dict(source=str(source_path.resolve()), seed=seed, requested_budget=budget,
                  popsize=popsize, training_sampling=sampling, audit_sampling=audit_sampling,
                  input_evaluations=1, cma_evaluations=evaluations,
                  total_objective_evaluations=1+evaluations,
                  exact_merit=("model.evaluate loss + 350*mean(max(0.62-per_field_throughput,0)^2) "
                               "+ 10*max(max_spot_mm/0.012-1,0)^2"))
    write_json(out/"configuration.json", config)
    result = dict(seed=seed, genes=best["genes"], source_genes=source_genes,
                  source_metrics=input_metrics, source_audit=input_audit,
                  source_audit_merit=input_audit_merit,
                  training=best["metrics"], audit=audit, audit_merit=audit_merit,
                  train_sampling=sampling, audit_sampling=audit_sampling,
                  input_evaluations=1, evaluations=evaluations,
                  total_objective_evaluations=1+evaluations,
                  requested_budget=budget, generations=generations, restarts=restarts,
                  elapsed_seconds=time.monotonic()-started, history=history,
                  decoded=model.decode(best["genes"]), configuration=config,
                  interpretation=("Warm-start full-covariance CMA over the assumed single-channel "
                                  "Schmidt-Cassegrain family. The input candidate is retained unless "
                                  "the specified sampled merit improves. Dense audit uses the same "
                                  "geometric engine; no independent optical certification."))
    write_json(out/"result.json", result)
    summary = dict(configuration=config, source_metrics=input_metrics,
                   source_audit_merit=input_audit_merit, result_training=best["metrics"],
                   result_audit_merit=audit_merit, input_evaluations=1,
                   cma_evaluations=evaluations, total_objective_evaluations=1+evaluations,
                   elapsed_seconds=result["elapsed_seconds"])
    write_json(out/"summary.json", summary)
    print(json.dumps(serial(dict(seed=seed, complete=True,
                                 cma_evaluations=evaluations,
                                 total_objective_evaluations=1+evaluations,
                                 source_loss=input_metrics["loss"],
                                 best_loss=best["metrics"]["loss"], audit=audit_merit))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="existing search result.json")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--budget", type=int, default=4800,
                        help="CMA evaluations, in full population batches")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--popsize", type=int, default=24)
    args = parser.parse_args(argv)
    run(args.source, args.out, budget=args.budget, seed=args.seed, popsize=args.popsize)


if __name__ == "__main__":
    main()
