"""Parallel seeding experiment for EF50 Zernike CMA-MEGA.

Standalone script — does NOT modify ef50_zernike.py or any existing code.
Uses torch.multiprocessing to run evaluations across multiple processes
and both GPUs, bypassing the GIL entirely.

Each worker process gets its own EF50ZernikeProblem on its assigned GPU.

Usage:
    python -m optical_rl.qd.seed_parallel_experiment --config optical_rl/configs/ef50_zernike.yaml
    python -m optical_rl.qd.seed_parallel_experiment --config optical_rl/configs/ef50_zernike.yaml --workers 16
    python -m optical_rl.qd.seed_parallel_experiment --config optical_rl/configs/ef50_zernike.yaml --seeds 5000
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np


def _available_cpu_count() -> int:
    """Return usable CPU count, respecting cgroup limits and os.sched_getaffinity."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# DeepLens stdout warning filter
# ---------------------------------------------------------------------------
#
# DeepLens's GeoLens.calc_entrance_pupil() prints two unconditional warnings
# when the front-group ray trace fails and it falls back to surfaces[0]:
#
#     "No intersection points found, use the first surface as entrance pupil."
#     "Zero or negative entrance pupil is detected, use the first surface ..."
#
# With strict_entrance_pupil=True these designs are rejected by our worker
# anyway, so the warning text is pure noise that drowns out our own progress
# logs.  We install a thin line-buffered filter on sys.stdout (in both
# workers and the parent) that drops any line containing these phrases and
# passes everything else through unchanged.
#
# Cost: one substring scan per write call.  Negligible vs. ray tracing.
# Set DEEPLENS_KEEP_WARNINGS=1 in the environment to disable the filter
# without editing code (useful for debugging).

_DEEPLENS_WARNING_PATTERNS = (
    "No intersection points found",
    "Zero or negative entrance pupil",
)


class _FilteredStdout:
    """Line-buffered stdout proxy that drops lines matching given patterns."""

    def __init__(self, real_stdout, drop_patterns: tuple[str, ...]):
        self._real = real_stdout
        self._drop = drop_patterns
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p in line for p in self._drop):
                self._real.write(line + "\n")

    def flush(self):
        if self._buf and not any(p in self._buf for p in self._drop):
            self._real.write(self._buf)
            self._buf = ""
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_deeplens_warning_filter() -> None:
    """Install the DeepLens warning filter on sys.stdout for this process.

    Idempotent: safe to call multiple times.  Honors DEEPLENS_KEEP_WARNINGS=1
    in the environment as an opt-out for debugging.
    """
    import sys
    if os.environ.get("DEEPLENS_KEEP_WARNINGS", "").strip() in ("1", "true", "yes"):
        return
    if isinstance(sys.stdout, _FilteredStdout):
        return
    sys.stdout = _FilteredStdout(sys.stdout, _DEEPLENS_WARNING_PATTERNS)


def _available_gpus() -> list[int]:
    """Return list of available CUDA device indices."""
    import torch
    return list(range(torch.cuda.device_count()))


def _fmt_hms(seconds: float) -> str:
    """Format a duration as h:mm:ss (or m:ss for short durations)."""
    if not (seconds == seconds) or seconds == float("inf"):
        return "  --:--"
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:2d}:{s:02d}"


def _pick_worker_count(requested: int | None, num_gpus: int) -> int:
    """Choose worker count: requested value, or auto-scale based on available
    CPUs and GPUs.  Leave headroom for other users."""
    available_cpus = _available_cpu_count()
    if requested is not None:
        return max(1, min(requested, available_cpus))
    # Auto: ~6 workers per GPU (11% util each → ~66% target), cap at half CPUs
    auto = num_gpus * 6
    return max(2, min(auto, available_cpus // 2, 32))


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_fn(gpu_id: int, task_queue, result_queue, config_dict: dict):
    """Worker process: pulls (idx, theta) from task_queue, evaluates, pushes
    (idx, result_tuple) to result_queue.  Sends (idx, None) on error."""
    import torch
    # Drop DeepLens entrance-pupil fallback warnings from this worker's
    # stdout before any lens-building code can fire them.
    install_deeplens_warning_filter()
    # Force this process onto the assigned GPU
    torch.cuda.set_device(gpu_id)

    # Reconstruct config in this process
    from optical_rl.qd.ef50_zernike import EF50ZernikeConfig, EF50ZernikeProblem
    config = EF50ZernikeConfig(**config_dict)

    # Override device to this worker's GPU
    config_for_gpu = EF50ZernikeConfig(
        **{**config_dict, "device": f"cuda:{gpu_id}"}
    )
    problem = EF50ZernikeProblem(config_for_gpu)

    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            break
        idx, theta = item
        try:
            with torch.no_grad():
                r = problem.evaluate(theta, with_grads=False)
            # Send back plain data (no torch tensors across processes)
            result_queue.put((idx, {
                "theta": r.theta,
                "fitness": r.fitness,
                "measures": r.measures,
                "metrics": r.metrics,
            }))
        except Exception as e:
            result_queue.put((idx, None, str(e)))


def _config_to_dict(config) -> dict:
    """Convert a frozen dataclass config to a plain dict for pickling."""
    d = {}
    for field in config.__dataclass_fields__:
        val = getattr(config, field)
        # Convert tuples-of-tuples to nested lists for safe pickling
        if isinstance(val, tuple):
            val = _tuple_to_list(val)
        d[field] = val
    return d


def _tuple_to_list(t):
    if isinstance(t, tuple):
        return tuple(_tuple_to_list(x) for x in t)
    return t


# ---------------------------------------------------------------------------
# Parallel evaluation via multiprocessing
# ---------------------------------------------------------------------------

def _parallel_evaluate(config, config_dict: dict, thetas: np.ndarray,
                       num_workers: int, gpu_ids: list[int],
                       *,
                       progress_label: str | None = None,
                       progress_every: int = 100) -> list[dict]:
    """Evaluate all thetas in parallel across worker processes and GPUs.
    Returns list of result dicts in original theta order.

    If progress_label is set, prints a throttled live counter every
    progress_every results pulled from the worker queue.  The print runs
    in the parent process between queue.get() calls and adds no cost to
    workers, so it does not affect GPU throughput.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")  # spawn = safe for CUDA
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Assign workers round-robin across GPUs
    workers = []
    for w in range(num_workers):
        gpu = gpu_ids[w % len(gpu_ids)]
        p = ctx.Process(target=_worker_fn,
                        args=(gpu, task_queue, result_queue, config_dict))
        p.start()
        workers.append(p)

    # Feed all tasks
    for i, theta in enumerate(thetas):
        task_queue.put((i, theta))

    # Poison pills to shut down workers
    for _ in workers:
        task_queue.put(None)

    # Collect results
    import time as _time
    n_total = len(thetas)
    results = [None] * n_total
    errors = []
    n_done = 0
    n_rejected = 0
    n_worker_err = 0
    t_start = _time.perf_counter()
    for _ in range(n_total):
        item = result_queue.get()
        n_done += 1
        if len(item) == 3:  # error case
            idx, _, err_msg = item
            errors.append((idx, err_msg))
            n_worker_err += 1
        else:
            idx, result_dict = item
            results[idx] = result_dict
            if result_dict["metrics"].get("rejected", 0.0) > 0.0:
                n_rejected += 1
        if progress_label is not None and (
            n_done % progress_every == 0 or n_done == n_total
        ):
            elapsed = _time.perf_counter() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = n_total - n_done
            eta = remaining / rate if rate > 0 else float("inf")
            pct = 100.0 * n_done / n_total
            print(
                f"[{progress_label}] {n_done}/{n_total} ({pct:5.1f}%)  "
                f"elapsed {_fmt_hms(elapsed)}  ETA {_fmt_hms(eta)}  "
                f"rate {rate:6.1f}/s  "
                f"rejected {n_rejected}"
                + (f"  worker_err {n_worker_err}" if n_worker_err else ""),
                flush=True,
            )

    # Wait for workers to finish
    for p in workers:
        p.join()

    if errors:
        print(f"  WARNING: {len(errors)} evaluations failed:")
        for idx, msg in errors[:5]:
            print(f"    theta[{idx}]: {msg}")

    return results


# ---------------------------------------------------------------------------
# Sequential baseline
# ---------------------------------------------------------------------------

def run_sequential(config, seed_count: int, seed_offset: int = 31):
    """Baseline: sequential seeding identical to EF50ZernikeRunner.seed_archive()."""
    from optical_rl.qd.ef50_zernike import EF50ZernikeProblem, EF50Archive2D

    problem = EF50ZernikeProblem(config)

    t0 = time.perf_counter()

    # Calibrate archive bounds
    if config.archive_bounds is not None:
        archive = EF50Archive2D(config.archive_bins, config.archive_bounds,
                                seed=config.cma_seed)
        cal_count = 0
    else:
        samples = problem.sample_theta(config.calibration_count, config.cma_seed + 17)
        m0_vals, m1_vals = [], []
        for theta in samples:
            result = problem.evaluate(theta, with_grads=False)
            m0_vals.append(result.measures[0])
            m1_vals.append(result.measures[1])
        bounds = _percentile_bounds(m0_vals, m1_vals)
        archive = EF50Archive2D(config.archive_bins, bounds, seed=config.cma_seed)
        cal_count = len(samples)
    cal_time = time.perf_counter() - t0

    # Seed
    flat = problem.initial_theta()
    all_thetas = np.vstack([
        flat.reshape(1, -1),
        problem.sample_theta(seed_count, config.cma_seed + seed_offset),
    ])

    t1 = time.perf_counter()
    for theta in all_thetas:
        r = problem.evaluate(theta, with_grads=False)
        archive.insert(r)
    seed_time = time.perf_counter() - t1
    total_time = time.perf_counter() - t0

    return archive.stats(), total_time, cal_time, seed_time, cal_count, len(all_thetas)


# ---------------------------------------------------------------------------
# Parallel run
# ---------------------------------------------------------------------------

def run_parallel(config, config_dict: dict, seed_count: int,
                 num_workers: int, gpu_ids: list[int], seed_offset: int = 31):
    """Parallel seeding using multiprocessing across GPUs."""
    from optical_rl.qd.ef50_zernike import EF50ZernikeProblem, EF50Archive2D, EF50EvalResult

    problem = EF50ZernikeProblem(config)

    t0 = time.perf_counter()

    # Calibrate archive bounds (parallel)
    if config.archive_bounds is not None:
        archive = EF50Archive2D(config.archive_bins, config.archive_bounds,
                                seed=config.cma_seed)
        cal_count = 0
    else:
        samples = problem.sample_theta(config.calibration_count, config.cma_seed + 17)
        cal_results = _parallel_evaluate(config, config_dict, samples,
                                         num_workers, gpu_ids)
        m0_vals = [r["measures"][0] for r in cal_results if r is not None]
        m1_vals = [r["measures"][1] for r in cal_results if r is not None]
        bounds = _percentile_bounds(m0_vals, m1_vals)
        archive = EF50Archive2D(config.archive_bins, bounds, seed=config.cma_seed)
        cal_count = len(samples)
    cal_time = time.perf_counter() - t0

    # Seed (parallel)
    flat = problem.initial_theta()
    all_thetas = np.vstack([
        flat.reshape(1, -1),
        problem.sample_theta(seed_count, config.cma_seed + seed_offset),
    ])

    t1 = time.perf_counter()
    seed_results = _parallel_evaluate(config, config_dict, all_thetas,
                                      num_workers, gpu_ids)
    seed_time = time.perf_counter() - t1
    total_time = time.perf_counter() - t0

    # Insert into archive in original order (deterministic)
    for r in seed_results:
        if r is not None:
            archive.insert(EF50EvalResult(
                theta=r["theta"],
                fitness=r["fitness"],
                measures=r["measures"],
                metrics=r["metrics"],
                grads=None,
            ))

    return archive.stats(), total_time, cal_time, seed_time, cal_count, len(all_thetas), num_workers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_bounds(m0_vals, m1_vals):
    """Compute archive bounds from measure values (2nd/98th percentile + 5% margin)."""
    def _bounds(arr):
        lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
        span = max(hi - lo, 1e-6)
        return (lo - 0.05 * span, hi + 0.05 * span)
    return (
        _bounds(np.asarray(m0_vals, dtype=np.float64)),
        _bounds(np.asarray(m1_vals, dtype=np.float64)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel seeding experiment for EF50 Zernike CMA-MEGA"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (qd: section)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override seed_count from config")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: auto)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids to use (default: all available)")
    parser.add_argument("--skip-sequential", action="store_true",
                        help="Skip sequential baseline (just run parallel)")
    args = parser.parse_args()

    # Load config
    from optical_rl.qd.ef50_zernike import _load_config, EF50ZernikeConfig
    if args.config is not None:
        config = _load_config(args.config)
    else:
        config = EF50ZernikeConfig()

    config_dict = _config_to_dict(config)

    seed_count = args.seeds if args.seeds is not None else config.seed_count
    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus else _available_gpus()
    num_workers = _pick_worker_count(args.workers, len(gpu_ids))
    available_cpus = _available_cpu_count()

    print("=" * 65)
    print("EF50 Parallel Seeding Experiment (multiprocessing)")
    print("=" * 65)
    print(f"  Available CPUs:      {available_cpus}")
    print(f"  Available GPUs:      {gpu_ids} ({len(gpu_ids)} devices)")
    print(f"  Worker processes:    {num_workers}")
    print(f"  Workers per GPU:     {num_workers // len(gpu_ids)} "
          f"(+{num_workers % len(gpu_ids)} extra)")
    print(f"  Seed count:          {seed_count}")
    print(f"  Calibration count:   {config.calibration_count}")
    print(f"  Num rays:            {config.num_rays}")
    print(f"  Wavelengths:         {config.fitness_wavelengths_um}")
    print(f"  Stage:               {config.stage}")
    print("=" * 65)

    # --- Sequential baseline ---
    if not args.skip_sequential:
        print("\n[sequential] Running baseline …")
        seq_stats, seq_total, seq_cal, seq_seed, seq_cal_n, seq_seed_n = \
            run_sequential(config, seed_count)
        seq_evals = seq_cal_n + seq_seed_n
        print(f"[sequential] Calibration:  {seq_cal:.2f}s  ({seq_cal_n} evals)")
        print(f"[sequential] Seeding:      {seq_seed:.2f}s  "
              f"({seq_seed_n} evals, {seq_seed_n / max(seq_seed, 0.01):.1f} evals/s)")
        print(f"[sequential] Total:        {seq_total:.2f}s  "
              f"({seq_evals} total evals)")
        print(f"[sequential] Archive: filled={seq_stats['filled']} "
              f"coverage={seq_stats['coverage']:.4f} "
              f"best={seq_stats['best_fitness']:.6f}")
    else:
        seq_total = None

    # --- Parallel ---
    print(f"\n[parallel]   Running with {num_workers} workers on GPUs {gpu_ids} …")
    par_stats, par_total, par_cal, par_seed, par_cal_n, par_seed_n, actual_w = \
        run_parallel(config, config_dict, seed_count, num_workers, gpu_ids)
    par_evals = par_cal_n + par_seed_n
    print(f"[parallel]   Calibration:  {par_cal:.2f}s  ({par_cal_n} evals)")
    print(f"[parallel]   Seeding:      {par_seed:.2f}s  "
          f"({par_seed_n} evals, {par_seed_n / max(par_seed, 0.01):.1f} evals/s)")
    print(f"[parallel]   Total:        {par_total:.2f}s  "
          f"({par_evals} total evals)")
    print(f"[parallel]   Archive: filled={par_stats['filled']} "
          f"coverage={par_stats['coverage']:.4f} "
          f"best={par_stats['best_fitness']:.6f}")

    # --- Comparison ---
    print("\n" + "=" * 65)
    if seq_total is not None:
        speedup_total = seq_total / par_total if par_total > 0 else float("inf")
        speedup_cal = seq_cal / par_cal if par_cal > 0 else float("inf")
        speedup_seed = seq_seed / par_seed if par_seed > 0 else float("inf")
        print(f"                Sequential    Parallel    Speedup")
        print(f"  Calibration:  {seq_cal:>8.2f}s    {par_cal:>8.2f}s    {speedup_cal:.2f}x")
        print(f"  Seeding:      {seq_seed:>8.2f}s    {par_seed:>8.2f}s    {speedup_seed:.2f}x")
        print(f"  Total:        {seq_total:>8.2f}s    {par_total:>8.2f}s    {speedup_total:.2f}x")
        if seq_stats["filled"] == par_stats["filled"]:
            print("  Archive fill: MATCH")
        else:
            print(f"  Archive fill: DIFFERS (seq={seq_stats['filled']}, "
                  f"par={par_stats['filled']})")
            print("  (Small differences expected — insertion order affects ties)")
    else:
        print(f"  Parallel total: {par_total:.2f}s  "
              f"(cal={par_cal:.2f}s + seed={par_seed:.2f}s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
