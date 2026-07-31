"""Single-process timing benchmark for EF50ZernikeProblem.evaluate().

Settles the diagnosis: is the parallel pool's wallclock dominated by GPU
SM saturation, or by per-eval Python/launch overhead?

Method:
  - Build one EF50ZernikeProblem on cuda:0 (single process, no pool).
  - Time N sequential forward-only evals (with_grads=False).
  - Time M sequential forward+grad evals (with_grads=True).
  - Sync CUDA before/after each eval so GPU async work is captured.
  - Print min/median/mean/max + total wallclock.

Compare the resulting "20 forward-only sequential" total to the pool's
~1.24s steady state:
  - If single-proc ~= pool wallclock => parallelism gives ~0 speedup =>
    bottleneck is per-eval CPU/launch overhead (MPS / more workers
    cannot help; need to reduce per-eval cost or batch differently).
  - If single-proc ~= 2-3x pool wallclock => pool is genuinely getting
    real parallelism, ceiling is GPU SM contention.

Usage:
  python -m optical_rl.qd.bench_eval_singleproc \\
      --config optical_rl/configs/ef50_zernike.yaml \\
      --n-forward 20 --n-grad 5 --gpu 0
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEEPLENS_ROOT = PROJECT_ROOT / "DeepLens"
for _p in [str(PROJECT_ROOT), str(_DEEPLENS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optical_rl.qd.ef50_zernike import EF50ZernikeProblem  # noqa: E402
from optical_rl.qd.ef50_zernike_parallel import _load_config  # noqa: E402


def _find_valid_theta(problem: EF50ZernikeProblem, max_attempts: int = 200) -> np.ndarray:
    """Try the latest archive first, then fall back to random sampling."""
    archives = sorted(
        Path("results/models").glob("ef50_zernike_*/archive_cells.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for arc_path in archives[:5]:
        try:
            with open(arc_path) as f:
                data = json.load(f)
            for cell in data.get("cells", []):
                theta = np.asarray(cell["theta"], dtype=np.float64)
                ev = problem.evaluate(theta, with_grads=False)
                if ev.metrics.get("rejected", 0.0) <= 0.0:
                    print(f"[bench] using theta from {arc_path.name}", flush=True)
                    return theta
        except Exception:
            continue

    rng = np.random.default_rng(0)
    bounds = np.asarray(problem.config.theta_bounds, dtype=np.float64)
    for _ in range(max_attempts):
        theta = rng.uniform(bounds[:, 0], bounds[:, 1])
        ev = problem.evaluate(theta, with_grads=False)
        if ev.metrics.get("rejected", 0.0) <= 0.0:
            print("[bench] using random theta", flush=True)
            return theta
    raise RuntimeError("could not find a non-rejected theta in 200 random tries")


def _summarize(name: str, times_ms: list[float]) -> None:
    n = len(times_ms)
    total = sum(times_ms)
    print(
        f"  {name:15s}  n={n:3d}  total={total:8.1f} ms  "
        f"mean={statistics.mean(times_ms):7.2f}  "
        f"median={statistics.median(times_ms):7.2f}  "
        f"min={min(times_ms):7.2f}  max={max(times_ms):7.2f}",
        flush=True,
    )


class _PhaseProfile:
    """Accumulates wallclock per named phase across many evals."""

    def __init__(self) -> None:
        self.totals_ms: dict = collections.defaultdict(float)
        self.counts: dict = collections.defaultdict(int)

    def add(self, name: str, dt_ms: float) -> None:
        self.totals_ms[name] += dt_ms
        self.counts[name] += 1


def _install_phase_timers(problem, profile: _PhaseProfile, sync) -> None:
    """Monkey-patch hot methods on the problem instance with timing wrappers.

    Each wrapper does sync() at entry/exit so per-phase wallclock includes the
    GPU work it launches. Adds extra synchronizations vs the unwrapped path,
    so the *sum* of phase times will exceed the unwrapped per-eval total
    slightly — phase ratios are still meaningful.
    """
    method_names = [
        "_build_lens_from_theta_tensor",
        "_calc_hfov_deg",
        "_trace_fields_batch",
        "_rms_spot_batch",
        "_intersection_penalty",
        "_zernike_measures",
    ]
    for name in method_names:
        orig = getattr(problem, name, None)
        if orig is None:
            continue

        def make_wrapper(orig_fn, label):
            def wrapped(*args, **kwargs):
                sync()
                t0 = time.perf_counter()
                result = orig_fn(*args, **kwargs)
                sync()
                profile.add(label, (time.perf_counter() - t0) * 1e3)
                return result
            return wrapped

        setattr(problem, name, make_wrapper(orig, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="optical_rl/configs/ef50_zernike.yaml")
    parser.add_argument("--n-forward", type=int, default=20,
                        help="Number of sequential forward-only evals to time")
    parser.add_argument("--n-grad", type=int, default=5,
                        help="Number of sequential forward+grad evals to time")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-rays", type=int, default=None,
                        help="Override num_rays from config")
    parser.add_argument("--wavelengths", type=str, default=None,
                        help="Comma-separated wavelengths in um, e.g. '0.587' or '0.656,0.587,0.486'")
    parser.add_argument("--stage", type=str, default=None, choices=[None, "stage1", "stage2"],
                        help="Override stage (stage1 = on-axis only, stage2 = 3 field points)")
    parser.add_argument("--profile", action="store_true",
                        help="Install per-method timers and print intra-eval breakdown")
    args = parser.parse_args()

    base_config = _load_config(args.config)
    overrides = {
        "device": f"cuda:{args.gpu}",
        "seed_count": 0,
        "calibration_count": 0,
    }
    if args.num_rays is not None:
        overrides["num_rays"] = int(args.num_rays)
    if args.wavelengths is not None:
        overrides["fitness_wavelengths_um"] = tuple(
            float(x) for x in args.wavelengths.split(",") if x.strip()
        )
    if args.stage is not None:
        overrides["stage"] = args.stage
    config = dataclasses.replace(base_config, **overrides)

    print(f"[bench] device={config.device} num_rays={config.num_rays} "
          f"stage={config.stage} wavelengths={config.fitness_wavelengths_um}",
          flush=True)

    problem = EF50ZernikeProblem(config)
    problem._ensure_backend()
    torch = problem._torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=problem._device)

    theta = _find_valid_theta(problem)

    # ---- warmup (JIT, kernel cache, allocator) ----
    print("[bench] warming up (3 evals each, discarded)...", flush=True)
    for _ in range(3):
        problem.evaluate(theta, with_grads=False)
    for _ in range(3):
        problem.evaluate(theta, with_grads=True)
    sync()

    profile = _PhaseProfile() if args.profile else None
    if profile is not None:
        _install_phase_timers(problem, profile, sync)

    # ---- forward-only ----
    print(f"[bench] timing {args.n_forward} sequential forward-only evals...", flush=True)
    fwd_times = []
    for _ in range(args.n_forward):
        sync()
        t0 = time.perf_counter()
        problem.evaluate(theta, with_grads=False)
        sync()
        fwd_times.append((time.perf_counter() - t0) * 1e3)

    # ---- forward + grad ----
    print(f"[bench] timing {args.n_grad} sequential forward+grad evals...", flush=True)
    grad_times = []
    for _ in range(args.n_grad):
        sync()
        t0 = time.perf_counter()
        problem.evaluate(theta, with_grads=True)
        sync()
        grad_times.append((time.perf_counter() - t0) * 1e3)

    # ---- report ----
    print("\n[bench] results (single process, 1 GPU, no worker pool):")
    _summarize("forward-only", fwd_times)
    _summarize("forward+grad", grad_times)

    if profile is not None:
        n_evals = args.n_forward + args.n_grad
        print(f"\n[bench] intra-eval breakdown (avg per eval, across {n_evals} evals):")
        # Sort phases descending by total time
        rows = sorted(
            profile.totals_ms.items(), key=lambda kv: kv[1], reverse=True
        )
        sum_ms = sum(profile.totals_ms.values())
        for name, total_ms in rows:
            count = profile.counts[name]
            per_eval = total_ms / n_evals
            per_call = total_ms / count if count else 0.0
            calls_per_eval = count / n_evals
            pct = 100.0 * total_ms / sum_ms if sum_ms > 0 else 0.0
            print(
                f"  {name:36s}  per_eval={per_eval:7.2f} ms  "
                f"per_call={per_call:7.2f} ms  "
                f"calls/eval={calls_per_eval:5.2f}  pct={pct:5.1f}%",
                flush=True,
            )
        print(
            f"  {'(sum of phases above)':36s}  per_eval={sum_ms/n_evals:7.2f} ms",
            flush=True,
        )

    fwd_total = sum(fwd_times)
    print(
        f"\n[bench] interpretation:\n"
        f"  Single-proc total for {args.n_forward} forward-only evals: {fwd_total:.0f} ms\n"
        f"  Pool wallclock for {args.n_forward} forward-only evals (steady state, your run): ~1240 ms\n",
        flush=True,
    )
    if fwd_total < 200:
        print("  => Single-proc is FASTER than the pool. Parallelism gives NEGATIVE benefit;\n"
              "     IPC + spawn + per-call overhead in the pool dominates.", flush=True)
    elif fwd_total < 1500:
        print("  => Single-proc is comparable to the pool. Parallelism gives ~0 speedup;\n"
              "     bottleneck is per-eval Python/launch overhead, not GPU compute.", flush=True)
    else:
        speedup = fwd_total / 1240.0
        print(f"  => Single-proc is ~{speedup:.1f}x slower than the pool;\n"
              f"     pool IS getting real parallelism, ceiling is GPU SM saturation\n"
              f"     (only way to go faster is reduce per-eval kernel cost).", flush=True)


if __name__ == "__main__":
    main()
