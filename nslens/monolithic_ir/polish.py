"""Bounded local least-squares polish for a monolithic CMA incumbent."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

from . import model
from .search import _serial, _write, TRAIN, AUDIT


def run(source, out, *, max_nfev=30):
    global model
    source, out = Path(source), Path(out)
    snapshot=source.parent/'model_snapshot.py'
    if snapshot.exists():
        spec=importlib.util.spec_from_file_location('monolithic_polish_frozen_model',snapshot)
        frozen=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=frozen
        spec.loader.exec_module(frozen)
        model=frozen
    data = json.loads(source.read_text())
    genes = np.asarray(data["genes"], dtype=float).reshape(-1)
    if genes.shape != (model.N_TOTAL,) or not np.isfinite(genes).all() or np.any(np.abs(genes) > 1):
        raise ValueError(f"source must contain {model.N_TOTAL} finite normalized genes in [-1,1]")
    if max_nfev < 1: raise ValueError("max_nfev must be positive")
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    for name, file in (("polish_snapshot.py", __file__), ("model_snapshot.py", model.__file__)):
        shutil.copyfile(file, out / name)
    shutil.copyfile(source, out / "source_result.json")
    trace = model.trace(genes, **TRAIN)
    reference = np.asarray(trace["good"], dtype=bool)
    config = dict(source=str(source.resolve()), max_nfev=max_nfev, method="trf",
                  jac="2-point", diff_step=1e-4, x_scale="jac", bounds=[-1, 1],
                  training_sampling=TRAIN, audit_sampling=AUDIT,
                  callback="record accepted solver iterates only")
    _write(out / "configuration.json", config)
    calls = 0
    def fun(g):
        nonlocal calls
        calls += 1
        return model.residuals(g, reference, fields=TRAIN["fields"],
                               waves=TRAIN["waves"], pupils=TRAIN["pupils"])
    initial = fun(genes)
    initial_cost = float(initial @ initial)
    history = []
    _write(out / 'trajectory.json', history)
    started = time.monotonic()
    def callback(x, *args):
        # SciPy calls this with solver iterates; Jacobian probes pass through fun only.
        val = fun(x)
        row = dict(iteration=len(history)+1, genes=np.asarray(x).copy(),
                   sum_squares=float(val @ val), residual_calls=calls,
                   elapsed_seconds=time.monotonic()-started)
        history.append(row)
        _write(out / "trajectory.json", history)
        _write(out / "checkpoint.json", dict(**row, history=history))
    opt = least_squares(fun, np.clip(genes, -1+1e-10, 1-1e-10), bounds=(-1, 1),
                        method="trf", jac="2-point", diff_step=1e-4, x_scale="jac",
                        ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=max_nfev,
                        callback=callback)
    solver_residual = fun(opt.x)
    solver_cost = float(solver_residual @ solver_residual)
    accepted = solver_cost <= initial_cost
    final = opt.x if accepted else genes
    final_trace = model.trace(final, **TRAIN)
    # Count all residual calls, including source, callback evaluations and final solver iterate.
    result = dict(seed=data.get("seed"), representation="monolithic_xy",
                  algorithm="bounded least-squares after CMA; hybrid residual merit",
                  source=str(source.resolve()), genes=final, decoded=model.decode(final),
                  configuration=config, training_sampling=TRAIN, audit_sampling=AUDIT,
                  initial_sum_squares=initial_cost, solver_sum_squares=solver_cost,
                  final_sum_squares=solver_cost if accepted else initial_cost,
                  accepted_solver_iterate=bool(accepted), solver_status=int(opt.status),
                  solver_message=str(opt.message), scipy_nfev=int(opt.nfev),
                  scipy_njev=int(opt.njev) if opt.njev is not None else None,
                  residual_calls=calls, callback_iterations=len(history),
                  residual_count=len(initial), initially_transmitted_mask=reference,
                  initially_transmitted_rays_lost=int(np.sum(reference & ~final_trace["good"])),
                  elapsed_seconds=time.monotonic()-started, before_training=model.evaluate(genes, **TRAIN),
                  training=model.evaluate(final, **TRAIN), before_audit=model.evaluate(genes, **AUDIT),
                  audit=model.evaluate(final, **AUDIT), trajectory=history,
                  interpretation="All-launched-ray polish using the same residual sum of squares as CMA. The source good mask is recorded for loss accounting, not used to discard invalid rays.")
    _write(out / "result.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-nfev", type=int, default=30)
    a = p.parse_args(argv)
    run(a.source, a.out, max_nfev=a.max_nfev)


if __name__ == "__main__": main()


