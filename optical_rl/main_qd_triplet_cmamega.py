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

from optical_rl.qd.cmamega_triplet import DESCRIPTOR_NAMES, FixedCookeTripletProblem, RunnerConfig, TripletCMAMEGARunner, TripletCookeConfig
from optical_rl.utils.config import load_config


def _ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def _new_run_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get("RL_CONFIG", "optical_rl/configs/qd_triplet_cmamega.yaml"),
        help="Path to YAML config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    qd_cfg = cfg["qd"]
    _ensure_dirs(qd_cfg["results_dir"])
    run_dir = _new_run_dir(qd_cfg["results_dir"], qd_cfg.get("run_name", "triplet_cmamega"))

    config_dst = run_dir / Path(args.config).name
    shutil.copy2(args.config, config_dst)
    print(f"[triplet-cmamega] Config saved to: {config_dst}", flush=True)

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
        print(f"[triplet-cmamega] Calibrated archive bounds: {cal}", flush=True)
    seed_stats = runner.seed_archive()
    print(f"[triplet-cmamega] Seed stats: {seed_stats}", flush=True)
    final_stats = runner.run(int(qd_cfg.get("iterations", 5000)))
    print(f"[triplet-cmamega] Final stats: {final_stats}", flush=True)

    _save_archive_json(run_dir / "archive.json", runner)
    _save_archive_plot(run_dir / "archive_map.png", runner)
    (run_dir / "summary.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
