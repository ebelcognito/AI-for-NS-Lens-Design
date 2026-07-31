from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.cmamega_ext import CMAMEGARunner, RunnerConfig
from optical_rl.qd.cmamega_ext.descriptors import DESCRIPTOR_NAMES
from optical_rl.qd.cmamega_ext.doublet_starting_point import build_doublet_problem_from_qd_cfg
from optical_rl.utils.config import load_config


def _ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def _new_run_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_archive_json(path: Path, runner: CMAMEGARunner) -> None:
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


def _save_archive_plot(path: Path, runner: CMAMEGARunner) -> None:
    assert runner.archive is not None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    bins_x, bins_y = runner.archive.bins
    grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
    xs = []
    ys = []
    cs = []
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get("RL_CONFIG", "optical_rl/configs/qd_doublet_cmamega.yaml"),
        help="Path to YAML config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    qd_cfg = cfg["qd"]
    _ensure_dirs(qd_cfg["results_dir"])
    run_dir = _new_run_dir(qd_cfg["results_dir"], qd_cfg.get("run_name", "doublet_cmamega"))

    config_dst = run_dir / Path(args.config).name
    shutil.copy2(args.config, config_dst)
    print(f"[doublet-cmamega] Config saved to: {config_dst}", flush=True)

    problem, bounds, active = build_doublet_problem_from_qd_cfg(qd_cfg)

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

    runner = CMAMEGARunner(runner_cfg, problem)
    if runner_cfg.archive_bounds is None:
        cal = runner.calibrate_archive_bounds()
        print(f"[doublet-cmamega] Calibrated archive bounds: {cal}", flush=True)
    seed_stats = runner.seed_archive()
    print(f"[doublet-cmamega] Seed stats: {seed_stats}", flush=True)
    final_stats = runner.run(int(qd_cfg.get("iterations", 5000)))
    print(f"[doublet-cmamega] Final stats: {final_stats}", flush=True)

    _save_archive_json(run_dir / "archive.json", runner)
    _save_archive_plot(run_dir / "archive_map.png", runner)
    (run_dir / "summary.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
