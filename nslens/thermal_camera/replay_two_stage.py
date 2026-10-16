"""Replay the archived two-stage CMA run and capture incumbent trajectory."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "experiments/thermal_camera_two_stage_v1"
REPLAY_ROOT = ROOT / "experiments/thermal_camera_replay_v1"
RUN_DIR = REPLAY_ROOT / "two_stage_replay_exec_v2"
TRAJECTORY = REPLAY_ROOT / "two_stage_trajectory.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


WORKER_MODEL = None

def _worker_evaluate(genes):
    return WORKER_MODEL.evaluate(genes, fields=5, waves=1, pupils=7)


def replay():
    global WORKER_MODEL
    if TRAJECTORY.exists() or RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite replay artifacts under {REPLAY_ROOT}")
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ORIGINAL / "configuration.json").read_text())
    original = json.loads((ORIGINAL / "result.json").read_text())

    import nslens.thermal_camera
    pkg_name = "nslens.thermal_camera._two_stage_replay_v1"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ORIGINAL)]
    sys.modules[pkg_name] = pkg
    radial = _load_module(pkg_name + ".model", ORIGINAL / "radial_model_snapshot.py")
    optical = _load_module(pkg_name + ".spectrometer_style_model",
                           ORIGINAL / "freeform_model_snapshot.py")
    driver = _load_module(pkg_name + ".two_stage_search",
                          ORIGINAL / "freeform_search_snapshot.py")
    cma = _load_module(pkg_name + ".cma_me_snapshot", ORIGINAL / "cma_me_snapshot.py")
    driver.CMAES = cma.CMAES
    if driver.model is not optical or optical.radial is not radial:
        raise RuntimeError("Snapshot module graph did not resolve to archived model files")

    WORKER_MODEL = optical
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork"))
    raw_evaluate = optical.evaluate
    cached = {}
    class ParallelCMA(cma.CMAES):
        def ask(self):
            samples = super().ask()
            padded = [np.pad(g, (0, optical.N_TOTAL-len(g))) for g in samples]
            values = list(pool.map(_worker_evaluate, padded))
            cached.update((g.tobytes(), value) for g, value in zip(padded, values))
            return samples
    def cached_evaluate(g, **kwargs):
        if kwargs == dict(fields=5, waves=1, pupils=7):
            value = cached.pop(np.asarray(g, float).tobytes(), None)
            if value is not None:
                return value
        return raw_evaluate(g, **kwargs)
    list(pool.map(_worker_evaluate, [np.zeros(optical.N_TOTAL)] * 4))
    optical.evaluate = cached_evaluate
    driver.CMAES = ParallelCMA
    captured = []
    write_json = driver.write_json

    def capture_write_json(path, value):
        p = Path(path)
        if p.name == "checkpoint.json" and "last_population" in value:
            rec = value["history"][-1]
            stage_spec = next(s for s in driver.model.STAGES if s[0] == rec["stage"])
            captured.append(dict(
                stage=rec["stage"], genes=np.asarray(value["genes"], float).copy(),
                representation=value["representation"], evaluations=int(rec["evaluations"]),
                objective=float(rec["best_loss"]), metrics=value["metrics"],
                objective_sampling=dict(fields=int(stage_spec[2]), waves=1, pupils=int(stage_spec[3])),
                generation=int(rec["stage_evaluations"] / config["popsize"])))
            write_json(REPLAY_ROOT/"two_stage_partial_trajectory.json",
                       dict(records=captured, verified=False, status="in progress"))
        return write_json(path, value)

    driver.write_json = capture_write_json
    if RUN_DIR.exists():
        raise FileExistsError(RUN_DIR)
    source_path = ORIGINAL / "source_result.json"
    started = time.monotonic()
    replay_result = driver.run(
        source_path, RUN_DIR, seed=int(config["seed"]),
        base_evals=int(config["base_evals"]), freeform_evals=int(config["freeform_evals"]),
        popsize=int(config["popsize"]), sigma=float(config["sigma"]),
        radial_control=(config.get("representation") == "radial_control"))
    elapsed = time.monotonic() - started
    driver.write_json = write_json
    optical.evaluate = raw_evaluate
    pool.shutdown()

    original_history = original["history"]
    if len(captured) != len(original_history):
        raise RuntimeError(f"Captured {len(captured)} generations; original history has {len(original_history)}")
    loss_diffs, stage_mismatches, eval_mismatches = [], [], []
    for index, (frame, old) in enumerate(zip(captured, original_history)):
        loss_diffs.append(abs(frame["objective"]-float(old["best_loss"])))
        if frame["stage"] != old["stage"]:
            stage_mismatches.append(index)
        if frame["evaluations"] != int(old["evaluations"]):
            eval_mismatches.append(index)
    replay_genes = np.asarray(replay_result["genes"], dtype=float)
    old_genes = np.asarray(original["genes"], dtype=float)
    gene_delta = np.abs(replay_genes-old_genes)
    max_gene_error = float(gene_delta.max(initial=0.))
    max_loss_error = float(max(loss_diffs, default=0.))
    exact = (max_loss_error <= 1e-12 and max_gene_error <= 1e-12
             and not stage_mismatches and not eval_mismatches)

    records = [dict(index=i, stage=frame["stage"], genes=frame["genes"],
                    representation="spectrometer_style_xy", evaluations=frame["evaluations"],
                    objective=frame["objective"], objective_sampling=frame["objective_sampling"],
                    objective_provenance="best incumbent loss from archived spectrometer_style_model.evaluate",
                    generation=frame["generation"])
               for i, frame in enumerate(captured)]
    trajectory = dict(
        schema="thermal_camera_cma_trajectory_v1",
        trajectory_kind="verified_deterministic_replay" if exact else "new_replay_not_verified_as_original",
        representation="spectrometer_style_xy", seed=int(config["seed"]),
        source_run="experiments/thermal_camera_two_stage_v1",
        snapshots=dict(driver="freeform_search_snapshot.py",
                       optical_model="freeform_model_snapshot.py",
                       radial_model="radial_model_snapshot.py",
                       optimizer="cma_me_snapshot.py", configuration="configuration.json"),
        objective=dict(name="archived model.evaluate loss",
                       training_fields_per_axis={str(s[0]): int(s[2]) for s in optical.STAGES},
                       wavelengths=1,
                       pupil_argument={str(s[0]): int(s[3]) for s in optical.STAGES},
                       arithmetic="full covariance CMA; candidates ranked by archived scalar loss; best-so-far incumbent"),
        verification=dict(matches_original_history=exact,
                          original_history_count=len(original_history),
                          replay_generation_count=len(captured),
                          max_history_best_loss_abs_error=max_loss_error,
                          max_final_gene_abs_error=max_gene_error,
                          stage_mismatch_indices=stage_mismatches,
                          evaluation_mismatch_indices=eval_mismatches,
                          original_final_evaluations=int(original["evaluations"]),
                          replay_final_evaluations=int(replay_result["evaluations"]),
                          exact_replay=exact, replay_wall_seconds=elapsed),
        records=records)
    TRAJECTORY.write_text(json.dumps(_plain(trajectory), indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(path=str(TRAJECTORY), records=len(records), exact_replay=exact,
                          max_loss_error=max_loss_error, max_gene_error=max_gene_error,
                          replay_seconds=elapsed)), flush=True)
    return trajectory


if __name__ == "__main__":
    replay()

