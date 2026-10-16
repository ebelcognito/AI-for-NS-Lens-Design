"""Coarse-to-fine CMA search over XY-freeform thermal-camera surfaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from nslens.search.cma_me import CMAES
from . import freeform_model as model

AUDIT_SAMPLING = dict(fields=7, waves=1, pupils=13)


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


def _evaluate(genes, fields, pupils):
    return model.evaluate(genes, fields=fields, waves=1, pupils=pupils)


def _audit(genes):
    return model.evaluate(genes, **AUDIT_SAMPLING)


def run(source_path, out_path, *, seed=201, stage_evals=2400, popsize=24,
        sigma=.06, radial_control=False):
    source_path, out = Path(source_path), Path(out_path)
    source = json.loads(source_path.read_text())
    source_genes = np.asarray(source["genes"], dtype=float).ravel()
    if source_genes.shape != (14,) or not np.isfinite(source_genes).all():
        raise ValueError("Source result must contain 14 finite radial genes")
    if np.any(source_genes < -1) or np.any(source_genes > 1):
        raise ValueError("Source radial genes must lie in [-1, 1]")
    if stage_evals < 0 or popsize < 4 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("stage-evals must be nonnegative, popsize >= 4 and sigma > 0")
    if not model.STAGES:
        raise ValueError("freeform_model.STAGES must define at least one stage")
    maximum = int(model.N_TOTAL)
    original_specs = [(str(n), int(d), int(f), int(p)) for n, d, f, p in model.STAGES]
    if any(d < 14 or d > maximum for _, d, _, _ in original_specs):
        raise ValueError("Each stage dimension must include the 14 radial genes and fit N_TOTAL")
    if any(f < 1 or p < 1 for _, _, f, p in original_specs):
        raise ValueError("Stage sampling counts must be positive")
    # The control uses identical stage names, sampling, budget and evaluator,
    # but keeps every candidate at the 14 radial genes.
    specs = [(name, 14 if radial_control else dim, fields, pupils)
             for name, dim, fields, pupils in original_specs]
    representation = "radial_control" if radial_control else "xy_freeform"

    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_path, out/"source_result.json")
    shutil.copyfile(Path(model.__file__), out/"freeform_model_snapshot.py")
    shutil.copyfile(Path(model.radial.__file__), out/"radial_model_snapshot.py")
    shutil.copyfile(Path(__file__), out/"freeform_search_snapshot.py")
    import nslens.search.cma_me as cma_module
    shutil.copyfile(Path(cma_module.__file__), out/"cma_me_snapshot.py")

    source_full = np.zeros(maximum)
    source_full[:14] = source_genes
    source_audit = _audit(source_full)
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    history, stages = [], []
    incumbent = source_full.copy()
    cumulative = 0

    for name, dimensions, fields, pupils in specs:
        source_metrics = _evaluate(source_full, fields, pupils)
        inherited_metrics = _evaluate(incumbent, fields, pupils)
        if source_metrics["loss"] <= inherited_metrics["loss"]:
            best = dict(genes=source_full.copy(), metrics=source_metrics)
            baseline_source = "source"
        else:
            best = dict(genes=incumbent.copy(), metrics=inherited_metrics)
            baseline_source = "inherited"
        baseline = dict(selected=best["metrics"], selected_from=baseline_source,
                        source=source_metrics, inherited=inherited_metrics)
        train_sampling = dict(fields=fields, waves=1, pupils=pupils)
        stage_count = generations = restarts = stale = 0
        emitter = None
        first_population = []

        while stage_count + popsize <= stage_evals:
            if emitter is None or emitter.restart_needed or stale >= 60:
                step = sigma if emitter is None else sigma*(2/3)
                emitter = CMAES(best["genes"][:dimensions], step, popsize=popsize,
                                seed=int(rng.integers(2**31)),
                                bounds=(-np.ones(dimensions), np.ones(dimensions)),
                                boundary_handling="reflect")
                stale = 0
                restarts += 1
            samples = emitter.ask()
            sampled = emitter.sampling_snapshot()
            rows = []
            prior_loss = float(best["metrics"]["loss"])
            for sample in samples:
                genes = np.zeros(maximum)
                genes[:dimensions] = sample
                metrics = _evaluate(genes, fields, pupils)
                if not np.isfinite(float(metrics["loss"])):
                    raise RuntimeError("freeform_model.evaluate returned a non-finite loss")
                row = dict(genes=genes, metrics=metrics)
                rows.append(row)
                if metrics["loss"] < best["metrics"]["loss"]:
                    best = row
            if not first_population:
                first_population = [dict(genes=r["genes"].copy(), metrics=r["metrics"]) for r in rows]
            emitter.tell(np.argsort([r["metrics"]["loss"] for r in rows]))
            stage_count += len(rows)
            cumulative += len(rows)
            generations += 1
            stale = stale+1 if best["metrics"]["loss"] >= prior_loss else 0
            rec = dict(stage=name, generation=generations, evaluations=cumulative,
                       stage_evaluations=stage_count, elapsed_seconds=time.monotonic()-started,
                       restarts=restarts, best_loss=float(best["metrics"]["loss"]),
                       best_metrics=best["metrics"])
            history.append(rec)
            write_json(out/"checkpoint.json", dict(
                seed=seed, sigma=sigma, representation=representation, stage=name,
                genes=best["genes"], metrics=best["metrics"], evaluations=cumulative,
                stage_evaluations=stage_count, cma_state=emitter.state_snapshot(),
                last_sampling=sampled,
                last_population=[dict(genes=r["genes"], metrics=r["metrics"]) for r in rows],
                history=history))
            if generations % 10 == 0:
                print(json.dumps(serial(dict(seed=seed, **rec))), flush=True)

        frozen_audit = _audit(best["genes"])
        stage_result = dict(name=name, dimensions=dimensions, representation=representation,
                            training_sampling=train_sampling, requested_budget=stage_evals,
                            evaluations=stage_count, restarts=restarts, baseline=baseline,
                            initial_population=first_population,
                            best=dict(genes=best["genes"].copy(), metrics=best["metrics"]),
                            audit=frozen_audit)
        stages.append(stage_result)
        incumbent = best["genes"].copy()
        write_json(out/"checkpoint.json", dict(seed=seed, sigma=sigma,
                    representation=representation, stage=name, genes=incumbent,
                    metrics=best["metrics"], evaluations=cumulative,
                    stage_evaluations=stage_count, frozen_stage_audit=frozen_audit,
                    history=history))

    final = stages[-1]
    genes = np.asarray(final["best"]["genes"])
    configuration = dict(source=str(source_path.resolve()), seed=seed, sigma=sigma,
                         restart_sigma=sigma*(2/3), representation=representation,
                         stage_evals=stage_evals, popsize=popsize,
                         stages=[dict(name=n, dimensions=d, fields=f, pupils=p) for n, d, f, p in specs],
                         audit_sampling=AUDIT_SAMPLING, source_audit=source_audit)
    result = dict(seed=seed, sigma=sigma, restart_sigma=sigma*(2/3),
                  representation=representation, source=str(source_path.resolve()),
                  source_genes=source_genes, source_audit=source_audit, genes=genes,
                  training=final["best"]["metrics"], audit=final["audit"],
                  train_sampling=final["training_sampling"], audit_sampling=AUDIT_SAMPLING,
                  evaluations=cumulative, stage_evals=stage_evals,
                  elapsed_seconds=time.monotonic()-started, stages=stages,
                  history=history, decoded=model.decode(genes), configuration=configuration,
                  interpretation=("Coarse-to-fine CMA over an assumed Schmidt-Cassegrain family. "
                                  f"Representation={representation}; dense audits use the same tracer."))
    write_json(out/"configuration.json", configuration)
    write_json(out/"result.json", result)
    write_json(out/"summary.json", dict(configuration=configuration,
                   source_audit=source_audit,
                   stages=[{k: row[k] for k in ("name", "evaluations", "baseline", "best", "audit")}
                           for row in stages], evaluations=cumulative,
                   elapsed_seconds=result["elapsed_seconds"]))
    print(json.dumps(serial(dict(seed=seed, sigma=sigma, representation=representation,
                                 complete=True, evaluations=cumulative,
                                 final_training=final["best"]["metrics"], final_audit=final["audit"]))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="radial search result.json")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--stage-evals", type=int, default=2400)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=.06)
    parser.add_argument("--radial-control", action="store_true",
                        help="matched CMA control with only the 14 radial genes active")
    args = parser.parse_args(argv)
    if args.stage_evals < 0 or args.popsize < 4 or not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("stage-evals must be nonnegative, popsize >= 4 and sigma > 0")
    run(args.source, args.out, seed=args.seed, stage_evals=args.stage_evals,
        popsize=args.popsize, sigma=args.sigma, radial_control=args.radial_control)


if __name__ == "__main__":
    main()
