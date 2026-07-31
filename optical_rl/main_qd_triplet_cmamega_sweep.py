from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.cmamega_triplet import (
    DESCRIPTOR_NAMES,
    FixedCookeTripletProblem,
    RunnerConfig,
    TripletCMAMEGARunner,
    TripletCookeConfig,
)
from optical_rl.utils.config import load_config


DEFAULT_SWEEP_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 3), (2, 4))


def _ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def _new_run_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _new_sweep_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"{run_name}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_archive_json(path: Path, runner: TripletCMAMEGARunner) -> None:
    assert runner.archive is not None
    rows = []
    for (i, j), result in runner.archive.cells.items():
        rows.append(
            {
                "cell": [int(i), int(j)],
                "theta": [float(x) for x in result.theta.tolist()],
                "fitness": float(result.fitness),
                "measures": [float(result.measures[0]), float(result.measures[1])],
                "metrics": {
                    **{k: float(v) for k, v in result.metrics.items()},
                    "descriptor_names": list(DESCRIPTOR_NAMES),
                    "descriptor_vector": [float(x) for x in result.descriptor_vector.tolist()],
                },
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _save_archive_plot(path: Path, runner: TripletCMAMEGARunner) -> None:
    assert runner.archive is not None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    bins_x, bins_y = runner.archive.bins
    grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
    xs, ys, cs = [], [], []
    for (i, j), result in runner.archive.cells.items():
        grid[i, j] = float(result.fitness)
        xs.append(float(result.measures[0]))
        ys.append(float(result.measures[1]))
        cs.append(float(result.fitness))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(grid.T, origin="lower", aspect="auto")
    axes[0].set_title("Archive Fitness Heatmap")
    axes[0].set_xlabel("Measure bin 0")
    axes[0].set_ylabel("Measure bin 1")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    if xs:
        sc = axes[1].scatter(xs, ys, c=cs, s=24)
        (x_lo, x_hi), (y_lo, y_hi) = runner.archive.bounds
        axes[1].set_xlim(x_lo, x_hi)
        axes[1].set_ylim(y_lo, y_hi)
        fig.colorbar(sc, ax=axes[1], shrink=0.85)
    axes[1].set_title("Archive Measures")
    axes[1].set_xlabel("descriptor[active0]")
    axes[1].set_ylabel("descriptor[active1]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_pairs_override(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(raw).split(";"):
        token = part.strip()
        if not token:
            continue
        vals = [x.strip() for x in token.split(",") if x.strip()]
        if len(vals) != 2:
            raise ValueError(f"Invalid pair token '{token}'. Expected 'i,j'.")
        out.append((int(vals[0]), int(vals[1])))
    if not out:
        raise ValueError("No valid descriptor pairs found in --pairs.")
    return out


def _pairs_from_cfg(qd_cfg: dict) -> list[tuple[int, int]]:
    raw = qd_cfg.get("sweep_pairs", None)
    if raw is None:
        return list(DEFAULT_SWEEP_PAIRS)
    out: list[tuple[int, int]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        out.append((int(pair[0]), int(pair[1])))
    return out if out else list(DEFAULT_SWEEP_PAIRS)


def _invalid_stats_from_archive(archive_rows: list[dict], best_fitness: float) -> tuple[float | None, float | None]:
    vals: list[float] = []
    best_invalid: float | None = None
    for row in archive_rows:
        metrics = row.get("metrics", {})
        inv = metrics.get("invalid_fraction")
        fit = row.get("fitness")
        if inv is None:
            continue
        inv_f = float(inv)
        vals.append(inv_f)
        if fit is not None and float(fit) == float(best_fitness) and best_invalid is None:
            best_invalid = inv_f
    mean_invalid = float(sum(vals) / len(vals)) if vals else None
    return best_invalid, mean_invalid


def _run_triplet_once(*, qd_cfg: dict, run_dir: Path, config_name: str) -> tuple[dict, dict]:
    bounds = tuple(tuple(float(x) for x in pair) for pair in qd_cfg["theta_bounds"])
    active = tuple(int(x) for x in qd_cfg.get("active_descriptor_indices", [0, 1]))

    problem_cfg = TripletCookeConfig(
        lens_path=str(qd_cfg.get("lens_path", "/home/hisham26/Desktop/RL/DeepLens/datasets/lenses/cooke.json")),
        object_depth_mm=float(qd_cfg.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd_cfg.get("wavelength_um", 0.587)),
        theta_bounds=bounds,
        invalid_tol=float(qd_cfg.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd_cfg.get("invalid_penalty_weight", 5.0)),
        num_rays=int(qd_cfg.get("num_rays", 256)),
        alpha_deg=float(qd_cfg.get("alpha_deg", 0.5)),
        stage2_field_deg=float(qd_cfg.get("stage2_field_deg", 2.0)),
        device=str(qd_cfg.get("device", "auto")),
        active_descriptor_indices=active,
    )
    problem = FixedCookeTripletProblem(problem_cfg)

    archive_bounds = qd_cfg.get("archive_bounds")
    if archive_bounds is not None:
        archive_bounds = (tuple(archive_bounds[0]), tuple(archive_bounds[1]))

    runner_cfg = RunnerConfig(
        theta_bounds=bounds,
        archive_bins=tuple(int(x) for x in qd_cfg.get("archive_bins", [32, 32])),
        archive_bounds=archive_bounds,
        invalid_tol=float(qd_cfg.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd_cfg.get("invalid_penalty_weight", 5.0)),
        cma_eta=float(qd_cfg.get("cma_eta", 0.05)),
        seed_count=int(qd_cfg.get("seed_count", 128)),
        calibration_count=int(qd_cfg.get("calibration_count", 512)),
        initial_sigma=float(qd_cfg.get("initial_sigma", 0.75)),
        population_size=int(qd_cfg.get("population_size", 16)),
        cma_seed=int(qd_cfg.get("seed", 0)),
        log_every=int(qd_cfg.get("log_every", 50)),
        active_descriptor_indices=active,
    )

    runner = TripletCMAMEGARunner(runner_cfg, problem)
    if runner_cfg.archive_bounds is None:
        cal = runner.calibrate_archive_bounds()
        print(f"[triplet-sweep] Calibrated archive bounds: {cal}", flush=True)

    seed_stats = runner.seed_archive()
    print(f"[triplet-sweep] Seed stats: {seed_stats}", flush=True)
    final_stats = runner.run(int(qd_cfg.get("iterations", 5000)))
    print(f"[triplet-sweep] Final stats: {final_stats}", flush=True)

    _save_archive_json(run_dir / "archive.json", runner)
    _save_archive_plot(run_dir / "archive_map.png", runner)
    (run_dir / "summary.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")

    resolved_cfg = {"qd": copy.deepcopy(qd_cfg)}
    (run_dir / f"resolved_{config_name}").write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")

    archive_rows = json.loads((run_dir / "archive.json").read_text(encoding="utf-8"))
    best_invalid, mean_invalid = _invalid_stats_from_archive(archive_rows, float(final_stats.get("best_fitness", float("nan"))))

    run_summary = {
        "active_descriptor_indices": [int(active[0]), int(active[1])],
        "run_dir": str(run_dir),
        "coverage": float(final_stats.get("coverage", float("nan"))),
        "best_fitness": float(final_stats.get("best_fitness", float("nan"))),
        "mean_score": float(final_stats.get("mean_score", float("nan"))),
        "clip_hits_mean": float(final_stats.get("clip_hits_mean", float("nan"))),
        "best_invalid_fraction": best_invalid,
        "mean_invalid_fraction": mean_invalid,
    }
    return final_stats, run_summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a triplet CMA-MEGA sweep over descriptor pairs.")
    p.add_argument(
        "--config",
        default=os.environ.get("RL_CONFIG", "optical_rl/configs/qd_triplet_cmamega.yaml"),
        help="Path to YAML config.",
    )
    p.add_argument(
        "--pairs",
        default=None,
        help="Optional descriptor pair override, e.g. '0,1;0,3;2,4'.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    qd_cfg = copy.deepcopy(cfg["qd"])

    if args.pairs:
        pairs = _parse_pairs_override(args.pairs)
    else:
        pairs = _pairs_from_cfg(qd_cfg)

    _ensure_dirs(qd_cfg["results_dir"])
    sweep_dir = _new_sweep_dir(qd_cfg["results_dir"], qd_cfg.get("run_name", "triplet_cmamega") + "_sweep")

    config_name = Path(args.config).name
    shutil.copy2(args.config, sweep_dir / config_name)

    print(f"[triplet-sweep] Sweep directory: {sweep_dir}", flush=True)
    print(f"[triplet-sweep] Descriptor pairs: {pairs}", flush=True)

    runs: list[dict] = []
    for i, j in pairs:
        run_cfg = copy.deepcopy(qd_cfg)
        run_cfg["active_descriptor_indices"] = [int(i), int(j)]
        run_cfg["run_name"] = f"{qd_cfg.get('run_name', 'triplet_cmamega')}_d{i}{j}"

        run_dir = _new_run_dir(run_cfg["results_dir"], run_cfg["run_name"])
        print(f"[triplet-sweep] Starting pair ({i},{j}) in {run_dir}", flush=True)

        _final_stats, run_summary = _run_triplet_once(qd_cfg=run_cfg, run_dir=run_dir, config_name=config_name)
        runs.append(run_summary)

    best_run = None
    if runs:
        best_run = max(runs, key=lambda row: float(row.get("best_fitness", -float("inf"))))

    sweep_summary = {
        "source_config": str(Path(args.config).resolve()),
        "pairs": [[int(i), int(j)] for i, j in pairs],
        "default_pairs": [[int(i), int(j)] for i, j in DEFAULT_SWEEP_PAIRS],
        "runs": runs,
        "best_run": best_run,
    }

    (sweep_dir / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")
    print(f"[triplet-sweep] Summary saved to: {sweep_dir / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
