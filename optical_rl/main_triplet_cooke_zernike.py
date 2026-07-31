"""Main runner for the Cooke-triplet CMA-MEGA search with Zernike descriptors.

Usage:
  python3 optical_rl/main_triplet_cooke_zernike.py \
      --config optical_rl/configs/triplet_cooke_zernike.yaml
"""

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

from optical_rl.qd.triplet_cooke_zernike import (
    TripletCookeZernikeConfig,
    TripletCookeZernikeRunner,
)
from optical_rl.qd.singlet_zernike_measures import NOLL_TERM_NAMES
from optical_rl.utils.config import load_config


def _new_run_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_archive_json(path: Path, runner: TripletCookeZernikeRunner) -> None:
    assert runner.archive is not None
    rows = []
    for (i, j), result in runner.archive.cells.items():
        rows.append({
            "cell": [int(i), int(j)],
            "theta": [float(x) for x in result.theta.tolist()],
            "fitness": float(result.fitness),
            "measures": [float(result.measures[0]), float(result.measures[1])],
            "metrics": {k: float(v) for k, v in result.metrics.items()},
        })
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _save_archive_plot(
    path: Path, runner: TripletCookeZernikeRunner, config: TripletCookeZernikeConfig
) -> None:
    assert runner.archive is not None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
    except Exception:
        return

    j0, j1 = config.descriptor_zernike_indices
    label0 = f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)"
    label1 = f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)"

    bins_x, bins_y = runner.archive.bins
    import numpy as np
    neg_grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
    xs, ys, neg_cs = [], [], []
    for (i, j), result in runner.archive.cells.items():
        f = float(result.fitness)
        neg_grid[i, j] = -f
        xs.append(float(result.measures[0]))
        ys.append(float(result.measures[1]))
        neg_cs.append(-f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid = neg_grid[np.isfinite(neg_grid) & (neg_grid > 0)]
    norm = LogNorm(vmin=float(valid.min()), vmax=float(valid.max())) if valid.size > 0 else None
    im = axes[0].imshow(neg_grid.T, origin="lower", aspect="auto", norm=norm)
    axes[0].set_title("Archive Fitness Heatmap")
    axes[0].set_xlabel("Measure bin 0")
    axes[0].set_ylabel("Measure bin 1")
    fig.colorbar(im, ax=axes[0], shrink=0.85).set_label("−fitness (log)")

    if xs:
        pos_cs = [v for v in neg_cs if v > 0]
        scatter_norm = LogNorm(vmin=min(pos_cs), vmax=max(pos_cs)) if pos_cs else None
        sc = axes[1].scatter(xs, ys, c=neg_cs, s=24, norm=scatter_norm)
        (x_lo, x_hi), (y_lo, y_hi) = runner.archive.bounds
        axes[1].set_xlim(x_lo, x_hi)
        axes[1].set_ylim(y_lo, y_hi)
        fig.colorbar(sc, ax=axes[1], shrink=0.85).set_label("−fitness (log)")
    axes[1].set_title("Archive (Cooke triplet Zernike)")
    axes[1].set_xlabel(label0)
    axes[1].set_ylabel(label1)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[triplet-zernike] Archive plot saved: {path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "RL_CONFIG", "optical_rl/configs/triplet_cooke_zernike.yaml"
        ),
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing run directory to resume from",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    qd = cfg["qd"]

    run_dir = _new_run_dir(qd["results_dir"], qd.get("run_name", "triplet_cooke_zernike"))
    config_dst = run_dir / Path(args.config).name
    shutil.copy2(args.config, config_dst)
    print(f"[triplet-zernike] Config saved: {config_dst}", flush=True)
    print(f"[triplet-zernike] Run dir:      {run_dir}", flush=True)

    archive_bounds = qd.get("archive_bounds")
    if archive_bounds is not None:
        archive_bounds = (tuple(archive_bounds[0]), tuple(archive_bounds[1]))

    _fl_raw = qd.get("focal_length_target_mm")
    focal_length_target_mm = float(_fl_raw) if _fl_raw is not None else None

    theta_dim = 5 if focal_length_target_mm is not None else 6
    theta_bounds_default = [[-0.06, 0.06]] * theta_dim

    config = TripletCookeZernikeConfig(
        lens_path=str(qd["lens_path"]),
        object_depth_mm=float(qd.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd.get("wavelength_um", 0.587)),
        fitness_wavelengths_um=tuple(
            float(x) for x in qd.get("fitness_wavelengths_um", [0.587])
        ),
        theta_bounds=tuple(tuple(b) for b in qd.get("theta_bounds", theta_bounds_default)),
        stop_surface_index=int(qd.get("stop_surface_index", 4)),
        focal_length_target_mm=focal_length_target_mm,
        descriptor_zernike_indices=tuple(int(x) for x in qd.get("descriptor_zernike_indices", [7, 4])),
        archive_bins=tuple(int(x) for x in qd.get("archive_bins", [64, 64])),
        archive_bounds=archive_bounds,
        invalid_tol=float(qd.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd.get("invalid_penalty_weight", 5.0)),
        cma_eta=float(qd.get("cma_eta", 0.05)),
        seed_count=int(qd.get("seed_count", 128)),
        calibration_count=int(qd.get("calibration_count", 512)),
        num_rays=int(qd.get("num_rays", 256)),
        stage=str(qd.get("stage", "stage1")),
        zernike_field_index=int(qd.get("zernike_field_index", 0)),
        device=str(qd.get("device", "auto")),
        initial_sigma=float(qd.get("initial_sigma", 0.75)),
        population_size=int(qd.get("population_size", 16)),
        cma_seed=int(qd.get("seed", 0)),
        results_dir=str(qd["results_dir"]),
        run_name=str(qd.get("run_name", "triplet_cooke_zernike")),
        log_every=int(qd.get("log_every", 50)),
        checkpoint_every=int(qd.get("checkpoint_every", 50)),
        new_cell_bonus=float(qd.get("new_cell_bonus", 1.0e6)),
    )

    if config.focal_length_target_mm is not None:
        print(f"[triplet-zernike] EFL constraint: {config.focal_length_target_mm} mm  "
              f"(theta is 5-D; c6 derived)", flush=True)
    else:
        print(f"[triplet-zernike] EFL constraint: NONE  (theta is 6-D free)", flush=True)

    print(f"[triplet-zernike] Descriptors: Z{config.descriptor_zernike_indices[0]} "
          f"({NOLL_TERM_NAMES.get(config.descriptor_zernike_indices[0], '?')})  "
          f"Z{config.descriptor_zernike_indices[1]} "
          f"({NOLL_TERM_NAMES.get(config.descriptor_zernike_indices[1], '?')})", flush=True)
    print(f"[triplet-zernike] Fitness stage: {config.stage}  "
          f"Zernike field index: {config.zernike_field_index}", flush=True)

    runner = TripletCookeZernikeRunner(config)
    runner.result_dir = run_dir

    resumed = False
    if args.resume is not None:
        from optical_rl.qd._checkpoint import hash_config, load_checkpoint
        ckpt_path = Path(args.resume) / "checkpoint.pkl"
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path)
            runner.archive = ckpt.archive
            runner.emitter.set_state(ckpt.emitter_state)
            runner.completed = int(ckpt.completed_evaluations)
            runner.generation = int(ckpt.generation)
            runner.restart_count = int(ckpt.restart_count)
            runner.wallclock_seconds = float(ckpt.wallclock_seconds)
            current_hash = hash_config(config)
            if ckpt.config_hash != current_hash:
                print(
                    f"[triplet-zernike] WARNING: config hash drift on resume "
                    f"(checkpoint={ckpt.config_hash[:12]}, "
                    f"current={current_hash[:12]}). Continuing anyway.",
                    flush=True,
                )
            print(
                f"[triplet-zernike] Resumed from {ckpt_path}: "
                f"completed={runner.completed} gen={runner.generation} "
                f"restarts={runner.restart_count} "
                f"best={runner.archive.stats()['best_fitness']:.6f}",
                flush=True,
            )
            resumed = True
        else:
            print(f"[triplet-zernike] No checkpoint.pkl in {args.resume}, starting fresh.",
                  flush=True)

    if not resumed:
        if config.archive_bounds is None:
            bounds = runner.calibrate_archive_bounds()
            print(f"[triplet-zernike] Calibrated archive bounds: {bounds}", flush=True)

        seed_stats = runner.seed_archive()
        print(f"[triplet-zernike] Seed stats: {seed_stats}", flush=True)

    final_stats = runner.run(int(qd.get("iterations", 2000)), resume=resumed)
    print(f"[triplet-zernike] Final stats: {final_stats}", flush=True)

    _save_archive_json(run_dir / "archive.json", runner)
    _save_archive_plot(run_dir / "archive_map.png", runner, config)
    (run_dir / "summary.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
    print(f"[triplet-zernike] Done. Results in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
