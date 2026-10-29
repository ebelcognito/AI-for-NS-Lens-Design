"""Equal-evaluation toy benchmark for the pretrained EvoTF-ES optimizer.

This is an architecture probe, not an optical-design result.  It runs the official
``evosax.algorithms.EvoTF_ES`` implementation and its bundled SNES-distilled
checkpoint against official SNES and this repository's full-covariance CMA-ES.
All optimizers see the same initial mean, population size, objective, and number of
objective calls.

The task suite contains shifted separable and rotated objectives.  Rotation is an
important stress test because the released EvoTF checkpoint updates a diagonal
Gaussian while our CMA implementation learns a full covariance matrix.

Install the optional external dependency in the Python environment used to run it:

    python -m pip install evosax==0.3.1
    python -m nslens.analysis.run_evotf_toy --seeds 20 --generations 50

The output JSON contains every paired run as well as aggregate statistics, making
it possible to re-summarize the experiment without rerunning it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from nslens.search.cma_me import CMAES


Objective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Task:
    family: str
    seed: int
    dimension: int
    initial_mean: np.ndarray
    objective: Objective


def _random_rotation(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q *= np.sign(np.diag(r))[None, :]
    return q


def make_task(family: str, seed: int, dimension: int) -> Task:
    """Create a deterministic task with optimum value zero at a hidden shift."""
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-2.0, 2.0, size=dimension)
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    initial_mean = shift + 3.0 * direction
    rotation = _random_rotation(rng, dimension)
    condition = np.logspace(0.0, 3.0, dimension)

    def coordinates(x: np.ndarray, *, rotate: bool) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64) - shift
        return z @ rotation if rotate else z

    if family == "sphere":
        objective = lambda x: np.sum(coordinates(x, rotate=False) ** 2, axis=-1)
    elif family == "separable_ellipsoid":
        objective = lambda x: np.sum(
            condition * coordinates(x, rotate=False) ** 2, axis=-1
        )
    elif family == "rotated_ellipsoid":
        objective = lambda x: np.sum(
            condition * coordinates(x, rotate=True) ** 2, axis=-1
        )
    elif family == "rotated_rosenbrock":
        def objective(x: np.ndarray) -> np.ndarray:
            z = coordinates(x, rotate=True) + 1.0
            return np.sum(
                100.0 * (z[..., 1:] - z[..., :-1] ** 2) ** 2
                + (1.0 - z[..., :-1]) ** 2,
                axis=-1,
            )
    elif family == "rotated_rastrigin":
        def objective(x: np.ndarray) -> np.ndarray:
            z = coordinates(x, rotate=True)
            return 10.0 * dimension + np.sum(
                z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=-1
            )
    else:
        raise ValueError(f"unknown task family: {family}")

    return Task(family, seed, dimension, initial_mean, objective)


def _checkpoints(generations: int) -> list[int]:
    points = {1, generations, max(1, generations // 5), max(1, generations // 2)}
    return sorted(points)


def run_cma(task: Task, population_size: int, generations: int) -> dict:
    strategy = CMAES(
        task.initial_mean,
        1.0,
        popsize=population_size,
        seed=task.seed + 10_000,
    )
    best = float(task.objective(task.initial_mean[None, :])[0])
    trace = {0: best}
    checkpoints = set(_checkpoints(generations))
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        population = strategy.ask()
        fitness = np.asarray(task.objective(population), dtype=np.float64)
        best = min(best, float(np.min(fitness)))
        strategy.tell(np.argsort(fitness))
        if generation in checkpoints:
            trace[generation] = best
    return {
        "best": best,
        "initial": trace[0],
        "trace": {str(k): v for k, v in trace.items()},
        "seconds": time.perf_counter() - start,
    }


def make_evotf(population_size: int, dimension: int, generations: int):
    try:
        import jax.numpy as jnp
        from evosax.algorithms import EvoTF_ES
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise SystemExit(
            "EvoTF benchmark requires the official package: "
            "python -m pip install evosax==0.3.1"
        ) from exc
    return EvoTF_ES(
        population_size=population_size,
        solution=jnp.zeros(dimension),
        max_context_len=generations,
    )


def make_snes(population_size: int, dimension: int):
    import jax.numpy as jnp
    from evosax.algorithms import SNES

    return SNES(population_size=population_size, solution=jnp.zeros(dimension))


def run_evosax(task: Task, generations: int, strategy) -> dict:
    import jax
    import jax.numpy as jnp

    params = strategy.default_params
    key = jax.random.key(task.seed + 10_000)
    key, init_key = jax.random.split(key)
    state = strategy.init(init_key, jnp.asarray(task.initial_mean), params)
    best = float(task.objective(task.initial_mean[None, :])[0])
    trace = {0: best}
    checkpoints = set(_checkpoints(generations))
    start = time.perf_counter()
    for generation in range(1, generations + 1):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = strategy.ask(ask_key, state, params)
        population_np = np.asarray(population, dtype=np.float64)
        fitness = np.asarray(task.objective(population_np), dtype=np.float64)
        best = min(best, float(np.min(fitness)))
        state, _ = strategy.tell(
            tell_key, population, jnp.asarray(fitness), state, params
        )
        # Materialize the update so timing includes inference rather than queued work.
        jax.block_until_ready(state.mean)
        if generation in checkpoints:
            trace[generation] = best
    result = {
        "best": best,
        "initial": trace[0],
        "trace": {str(k): v for k, v in trace.items()},
        "seconds": time.perf_counter() - start,
    }
    if hasattr(strategy, "num_network_params"):
        result["network_parameters"] = strategy.num_network_params
    return result


def summarize(records: list[dict], algorithms: list[str]) -> dict:
    summary: dict[str, dict] = {}
    families = sorted({r["family"] for r in records})
    for family in families + ["all"]:
        subset = records if family == "all" else [r for r in records if r["family"] == family]
        family_summary: dict[str, dict] = {}
        for algorithm in algorithms:
            runs = [r[algorithm] for r in subset]
            ratios = np.asarray(
                [max(run["best"], 1e-30) / max(run["initial"], 1e-30) for run in runs]
            )
            family_summary[algorithm] = {
                "runs": len(runs),
                "median_final": float(np.median([run["best"] for run in runs])),
                "median_log10_relative_regret": float(np.median(np.log10(ratios))),
                "geometric_mean_relative_regret": float(
                    math.exp(float(np.mean(np.log(ratios))))
                ),
                "median_seconds": float(np.median([run["seconds"] for run in runs])),
            }
        if "cma" in algorithms and "evotf" in algorithms:
            cma_best = np.asarray([r["cma"]["best"] for r in subset])
            evotf_best = np.asarray([r["evotf"]["best"] for r in subset])
            family_summary["paired_evotf_vs_cma"] = {
                "evotf_wins": int(np.sum(evotf_best < cma_best)),
                "ties": int(np.sum(evotf_best == cma_best)),
                "cma_wins": int(np.sum(cma_best < evotf_best)),
                "median_evotf_over_cma": float(
                    np.median((evotf_best + 1e-30) / (cma_best + 1e-30))
                ),
            }
        if "snes" in algorithms and "evotf" in algorithms:
            snes_best = np.asarray([r["snes"]["best"] for r in subset])
            evotf_best = np.asarray([r["evotf"]["best"] for r in subset])
            family_summary["paired_evotf_vs_snes"] = {
                "evotf_wins": int(np.sum(evotf_best < snes_best)),
                "ties": int(np.sum(evotf_best == snes_best)),
                "snes_wins": int(np.sum(snes_best < evotf_best)),
                "median_evotf_over_snes": float(
                    np.median((evotf_best + 1e-30) / (snes_best + 1e-30))
                ),
            }
        summary[family] = family_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "sphere",
            "separable_ellipsoid",
            "rotated_ellipsoid",
            "rotated_rosenbrock",
            "rotated_rastrigin",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evotf_toy.json"),
    )
    args = parser.parse_args()
    if args.dimension < 2 or args.population_size < 2 or args.generations < 1 or args.seeds < 1:
        parser.error("dimension/population-size must be >=2; generations/seeds must be >=1")

    # Reusing this object is significant: JAX treats the algorithm instance as a
    # static JIT argument.  A fresh instance per task would compile the same graph
    # repeatedly and make the benchmark needlessly slow.
    evotf = make_evotf(args.population_size, args.dimension, args.generations)
    snes = make_snes(args.population_size, args.dimension)
    records = []
    for family_index, family in enumerate(args.families):
        for seed in range(args.seeds):
            task_seed = 1_000 * family_index + seed
            task = make_task(family, task_seed, args.dimension)
            record = {
                "family": family,
                "seed": seed,
                "task_seed": task_seed,
                "cma": run_cma(task, args.population_size, args.generations),
                "snes": run_evosax(task, args.generations, strategy=snes),
                "evotf": run_evosax(task, args.generations, strategy=evotf),
            }
            records.append(record)
            print(
                f"{family:21s} seed={seed:02d} "
                f"CMA={record['cma']['best']:.3e} "
                f"SNES={record['snes']['best']:.3e} "
                f"EvoTF={record['evotf']['best']:.3e}",
                flush=True,
            )

    payload = {
        "meta": {
            "experiment": "official_pretrained_evotf_vs_repository_cma_toy",
            "dimension": args.dimension,
            "population_size": args.population_size,
            "generations": args.generations,
            "objective_calls_per_run": 1 + args.population_size * args.generations,
            "seeds_per_family": args.seeds,
            "families": args.families,
            "evotf_checkpoint": "evosax 0.3.1 bundled 2024_03_SNES_small.pkl",
            "caveat": "Synthetic architecture probe; no optical simulator was used.",
        },
        "summary": summarize(records, ["cma", "snes", "evotf"]),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
