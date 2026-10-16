"""Independent, staged CMA searches for the architecture-conditioned spectrometer.

No published prescription is used. This is scalar CMA with retained-candidate
restarts, not a claim of CMA-ME or unconstrained architecture discovery.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import shutil
from pathlib import Path
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
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')


def retain(archive, candidate, capacity=12):
    """Keep low-loss, separated candidates; this is not a topology catalogue."""
    x = candidate['genes']
    near = next((i for i, row in enumerate(archive)
                 if np.linalg.norm(x-row['genes']) / np.sqrt(len(x)) < .08), None)
    if near is not None:
        if candidate['metrics']['loss'] < archive[near]['metrics']['loss']:
            archive[near] = candidate
    else:
        archive.append(candidate)
    archive.sort(key=lambda row: row['metrics']['loss'])
    del archive[capacity:]


def run_seed(config):
    out = Path(config['out']) / f"seed_{config['seed']}"
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config['seed'])
    started = time.monotonic()
    history = []
    best = None
    total = 0
    stage_results = []
    initial = []
    train = dict(fields=config['fields'], waves=config['waves'], pupils=config['pupils'])
    for stage, dimensions, budget in [('base', model.N_BASE, config['base_evals']),
                                       ('freeform', model.N_TOTAL, config['freeform_evals'])]:
        if budget < config['popsize']:
            continue
        archive = []
        # Each stage carries its own incumbent; an expansion never discards a better base result.
        if best is not None:
            retain(archive, best)
        stage_count = 0
        emitter = None
        stale = 0
        restarts = 0
        while stage_count + config['popsize'] <= budget:
            if emitter is None or emitter.restart_needed or stale >= config['restart_generations']:
                if archive and rng.random() < .8:
                    parent = archive[int(rng.integers(min(4, len(archive))))]['genes'][:dimensions]
                    sigma = .10 if stage == 'freeform' else .18
                else:
                    parent = rng.uniform(-.8, .8, dimensions)
                    sigma = .25
                emitter = CMAES(parent, sigma, popsize=config['popsize'],
                                seed=int(rng.integers(2**31)),
                                bounds=(-np.ones(dimensions), np.ones(dimensions)),
                                boundary_handling='reflect')
                stale = 0
                restarts += 1
            samples = emitter.ask()
            rows = []
            previous = best['metrics']['loss'] if best else np.inf
            for sample in samples:
                genes = np.zeros(model.N_TOTAL)
                genes[:dimensions] = sample
                metrics = model.evaluate(genes, **train)
                if not np.isfinite(metrics['loss']):
                    raise RuntimeError('The evaluator returned a non-finite loss')
                row = dict(genes=genes.copy(), metrics=metrics)
                rows.append(row)
                if not stage_results and len(initial) < config['popsize']:
                    initial.append(row)
                retain(archive, row)
                if best is None or metrics['loss'] < best['metrics']['loss']:
                    best = row
            emitter.tell(np.argsort([r['metrics']['loss'] for r in rows]))
            stage_count += len(samples)
            total += len(samples)
            stale = stale + 1 if best['metrics']['loss'] >= previous else 0
            rec = dict(stage=stage, evaluations=total, stage_evaluations=stage_count,
                       elapsed_seconds=time.monotonic()-started, restarts=restarts,
                       best=best['metrics'])
            history.append(rec)
            # Checkpoints survive interruption; they do not influence other seed runs.
            write_json(out/'checkpoint.json', dict(seed=config['seed'], genes=best['genes'],
                       metrics=best['metrics'], evaluations=total, stage=stage))
            if len(history) % 10 == 0:
                print(json.dumps(serial(dict(seed=config['seed'], **rec))), flush=True)
        stage_results.append(dict(stage=stage, dimensions=dimensions, budget=budget,
                                  evaluations=stage_count, restarts=restarts,
                                  best=dict(genes=best['genes'].copy(), metrics=dict(best['metrics']))))
    if best is None:
        raise ValueError('At least one stage must allow a full population')
    # Audit the final incumbent and the frozen base-stage incumbent, not select on the audit.
    audit_sampling = dict(fields=config['audit_fields'], waves=config['audit_waves'],
                          pupils=config['audit_pupils'])
    audit = model.evaluate(best['genes'], **audit_sampling)
    base = next((s['best'] for s in stage_results if s['stage']=='base'), None)
    base_audit = model.evaluate(base['genes'], **audit_sampling) if base is not None else None
    result = dict(seed=config['seed'], train_sampling=train, audit_sampling=audit_sampling,
                  evaluations=total, audit_evaluations=1+int(base is not None),
                  elapsed_seconds=time.monotonic()-started, stages=stage_results,
                  genes=best['genes'], training=best['metrics'], audit=audit,
                  base_audit=base_audit, initial_population=initial, history=history,
                  decoded=model.decode(best['genes']))
    write_json(out/'result.json', result)
    print(json.dumps(serial(dict(seed=config['seed'], complete=True, evaluations=total,
                                 elapsed_seconds=result['elapsed_seconds'], audit=audit))), flush=True)
    return serial(result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103])
    p.add_argument('--base-evals', type=int, default=1200)
    p.add_argument('--freeform-evals', type=int, default=2400)
    p.add_argument('--popsize', type=int, default=24)
    p.add_argument('--restart-generations', type=int, default=18)
    p.add_argument('--fields', type=int, default=3)
    p.add_argument('--waves', type=int, default=3)
    p.add_argument('--pupils', type=int, default=5)
    p.add_argument('--audit-fields', type=int, default=8)
    p.add_argument('--audit-waves', type=int, default=8)
    p.add_argument('--audit-pupils', type=int, default=10)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args(argv)
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Seeds must be unique to keep independent output streams')
    if min(a.base_evals, a.freeform_evals) < 0 or a.popsize < 4 or a.workers < 1:
        p.error('Budgets must be nonnegative, popsize >=4, workers >=1')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    configuration = vars(a)
    write_json(out/'configuration.json', configuration)
    shutil.copyfile(Path(model.__file__),out/'model_snapshot.py')
    shutil.copyfile(Path(__file__),out/'search_snapshot.py')
    configs = [dict(configuration, seed=seed) for seed in a.seeds]
    started = time.monotonic()
    if a.workers == 1:
        results = [run_seed(c) for c in configs]
    else:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(configs))) as pool:
            results = list(pool.map(run_seed, configs))
    summary = dict(configuration=configuration, wall_seconds=time.monotonic()-started,
                   interpretation='Architecture-conditioned scalar CMA with retained-candidate restarts; no published prescription initialization. Audit is denser same-engine validation, not independent optical verification.',
                   runs=[{k:r[k] for k in ['seed','evaluations','elapsed_seconds','training','audit','base_audit']}
                         for r in results])
    write_json(out/'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
