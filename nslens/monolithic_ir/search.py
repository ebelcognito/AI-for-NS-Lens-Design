"""Two-stage CMA-ES search for the monolithic infrared lens model."""
from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np

from nslens.search.cma_me import CMAES
from . import model

TRAIN = dict(fields=3, waves=3, pupils=5)
AUDIT = dict(fields=7, waves=3, pupils=9)


def _serial(x):
    if dataclasses.is_dataclass(x): return _serial(dataclasses.asdict(x))
    if isinstance(x, dict): return {str(k): _serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_serial(v) for v in x]
    if isinstance(x, np.ndarray): return _serial(x.tolist())
    if isinstance(x, np.generic): return _serial(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def _write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_serial(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _genes_from_source(path):
    data = json.loads(Path(path).read_text())
    genes = np.asarray(data["genes"], dtype=float).reshape(-1)
    if genes.shape != (model.N_TOTAL,) or not np.isfinite(genes).all() or np.any(np.abs(genes) > 1):
        raise ValueError(f"source must contain {model.N_TOTAL} finite normalized genes in [-1,1]")
    return data, genes


def _eval(genes):
    out = model.evaluate(genes, **TRAIN)
    loss = float(out["loss"])
    if not np.isfinite(loss): raise ValueError("model.evaluate returned non-finite loss")
    return out


def run(out, *, seed=201, source=None, base_evals=1200, freeform_evals=3600,
        popsize=24, base_sigma=.1, freeform_sigma=.04):
    out = Path(out)
    if out.exists(): raise FileExistsError(out)
    if min(base_evals, freeform_evals) < 0 or popsize < 4:
        raise ValueError("budgets must be nonnegative and popsize at least 4")
    source_data = None
    if source:
        source_data, start = _genes_from_source(source)
    else:
        start = np.zeros(model.N_TOTAL)
    out.mkdir(parents=True)
    if source: shutil.copyfile(source, out / "source_result.json")
    for name, file in (("model_snapshot.py", model.__file__), ("search_snapshot.py", __file__)):
        shutil.copyfile(file, out / name)
    import nslens.search.cma_me as cma
    shutil.copyfile(cma.__file__, out / "cma_me_snapshot.py")

    config = dict(seed=int(seed), source=str(Path(source).resolve()) if source else None,
                  start_genes=start, stages=[dict(name="base", dimensions=model.N_BASE20,
                  budget=base_evals, sigma=base_sigma), dict(name="freeform", dimensions=model.N_TOTAL,
                  budget=freeform_evals, sigma=freeform_sigma)], popsize=popsize,
                  training_sampling=dict(**TRAIN), audit_sampling=dict(**AUDIT),
                  selection="training loss only; audits do not select")
    _write(out / "configuration.json", config)
    started = time.monotonic()
    cumulative = 0
    trajectory, stages = [], []
    incumbent = start.copy()
    incumbent_metrics = _eval(incumbent)
    source_metrics = incumbent_metrics
    if source_data:
        source_data = dict(source=source_data, training=source_metrics,
                           audit=model.evaluate(start, **AUDIT))
    for name, dim, budget, sigma in (("base", model.N_BASE20, base_evals, base_sigma),
                                     ("freeform", model.N_TOTAL, freeform_evals, freeform_sigma)):
        base = np.zeros(model.N_TOTAL); base[:dim] = incumbent[:dim]
        baseline_metrics = _eval(base)
        if baseline_metrics["loss"] < incumbent_metrics["loss"]:
            incumbent, incumbent_metrics = base, baseline_metrics
        stage_start = cumulative
        generations = restarts = stale = 0
        emitter = None
        while cumulative - stage_start + popsize <= budget:
            if emitter is None or emitter.restart_needed or stale >= 60:
                emitter = CMAES(incumbent[:dim], sigma if restarts == 0 else sigma * (2/3),
                                popsize=popsize, seed=int(np.random.SeedSequence([seed, len(stages), restarts]).generate_state(1)[0]),
                                bounds=(-np.ones(dim), np.ones(dim)), boundary_handling="reflect")
                stale = 0; restarts += 1
            pop = emitter.ask()
            sampling = emitter.sampling_snapshot()
            rows = []
            prior = float(incumbent_metrics["loss"])
            for sample in pop:
                g = np.zeros(model.N_TOTAL); g[:dim] = sample
                metrics = _eval(g)
                rows.append((g, metrics))
                if metrics["loss"] < incumbent_metrics["loss"]:
                    incumbent, incumbent_metrics = g, metrics
            emitter.tell(np.argsort([r[1]["loss"] for r in rows]))
            cumulative += len(rows); generations += 1
            stale = stale + 1 if incumbent_metrics["loss"] >= prior else 0
            row = dict(stage=name, generation=generations, evaluations=cumulative,
                       stage_evaluations=cumulative-stage_start,
                       elapsed_seconds=time.monotonic()-started, best_loss=float(incumbent_metrics["loss"]),
                       best_metrics=incumbent_metrics, genes=incumbent.copy(), restarts=restarts)
            trajectory.append(row)
            # Each generation is a durable append-only trajectory record.
            with (out / "trajectory.jsonl").open("a") as f:
                f.write(json.dumps(_serial(row), allow_nan=False) + "\n"); f.flush()
            _write(out / "checkpoint.json", dict(**row, cma_state=emitter.state_snapshot(),
                   last_sampling=sampling, rng_snapshot=emitter.state_snapshot()["rng_state"]))
            _write(out / "trajectory.json", trajectory)
        audit = model.evaluate(incumbent, **AUDIT)
        stages.append(dict(name=name, dimensions=dim, requested_budget=budget,
                           evaluations=cumulative-stage_start, baseline=baseline_metrics,
                           best=dict(genes=incumbent.copy(), metrics=incumbent_metrics), audit=audit))
        _write(out / "checkpoint.json", dict(stage=name, genes=incumbent, metrics=incumbent_metrics,
               evaluations=cumulative, frozen_stage_audit=audit, history=trajectory))
    result = dict(seed=seed, representation="monolithic_xy", source=config["source"],
                  source_training=source_metrics, source_audit=source_data["audit"] if source_data else None,
                  genes=incumbent, decoded=model.decode(incumbent), training=incumbent_metrics,
                  audit=model.evaluate(incumbent, **AUDIT), train_sampling=TRAIN, audit_sampling=AUDIT,
                  evaluations=cumulative, requested_evaluations=base_evals+freeform_evals,
                  stages=stages, trajectory=trajectory, configuration=config,
                  elapsed_seconds=time.monotonic()-started)
    _write(out / "result.json", result)
    _write(out / "summary.json", dict(seed=seed, evaluations=cumulative,
           training=result["training"], audit=result["audit"], source_audit=result["source_audit"]))
    return result


def _worker(args):
    return run(args[1], seed=args[0], source=args[2], base_evals=args[3],
               freeform_evals=args[4], popsize=args[5])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True); p.add_argument("--source")
    p.add_argument("--seed", type=int, default=201); p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stage", choices=("base", "freeform", "both"), default="both")
    p.add_argument("--base-evals", type=int, default=1200); p.add_argument("--freeform-evals", type=int, default=3600)
    p.add_argument("--popsize", type=int, default=24)
    a = p.parse_args(argv)
    be = a.base_evals if a.stage != "freeform" else 0
    fe = a.freeform_evals if a.stage != "base" else 0
    seeds = a.seeds or [a.seed]
    if len(seeds) == 1 or a.workers <= 1:
        for s in seeds:
            run(Path(a.out) / f"seed{s}", seed=s, source=a.source, base_evals=be,
                freeform_evals=fe, popsize=a.popsize)
    else:
        jobs = [(s, str(Path(a.out) / f"seed{s}"), a.source, be, fe, a.popsize) for s in seeds]
        with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as pool:
            list(pool.map(_worker, jobs))


if __name__ == "__main__": main()

