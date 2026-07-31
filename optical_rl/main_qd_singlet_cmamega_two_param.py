import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.qd.singlet_cmamega import SingletCMAMEGAConfig
from optical_rl.qd.singlet_cmamega_two_param import TwoParamSingletCMAMEGARunner
from optical_rl.utils.config import load_config


def _ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def _new_run_dir(base_dir: str, run_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_archive_json(path: Path, runner: TwoParamSingletCMAMEGARunner) -> None:
    assert runner.archive is not None
    rows = []
    for (i, j), result in runner.archive.cells.items():
        rows.append(
            {
                "cell": [int(i), int(j)],
                "theta": [float(x) for x in result.theta.tolist()],
                "fitness": float(result.fitness),
                "measures": [float(result.measures[0]), float(result.measures[1])],
                "metrics": {k: float(v) for k, v in result.metrics.items()},
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _save_archive_plot(
    path: Path, runner: TwoParamSingletCMAMEGARunner, config: SingletCMAMEGAConfig
) -> None:
    assert runner.archive is not None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
    except Exception:
        return

    from optical_rl.qd.singlet_cmamega import NOLL_TERM_NAMES  # noqa: PLC0415

    j0, j1 = config.descriptor_zernike_indices
    label0 = f"Z{j0} {NOLL_TERM_NAMES.get(j0, '')} (mm)"
    label1 = f"Z{j1} {NOLL_TERM_NAMES.get(j1, '')} (mm)"

    bins_x, bins_y = runner.archive.bins
    neg_grid = np.full((bins_x, bins_y), np.nan, dtype=np.float64)
    xs = []
    ys = []
    neg_cs = []
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
    cbar0 = fig.colorbar(im, ax=axes[0], shrink=0.85)
    cbar0.set_label("−fitness (log scale)")

    if xs:
        pos_cs = [v for v in neg_cs if v > 0]
        scatter_norm = LogNorm(vmin=min(pos_cs), vmax=max(pos_cs)) if pos_cs else None
        sc = axes[1].scatter(xs, ys, c=neg_cs, s=24, norm=scatter_norm)
        (x_lo, x_hi), (y_lo, y_hi) = runner.archive.bounds
        axes[1].set_xlim(x_lo, x_hi)
        axes[1].set_ylim(y_lo, y_hi)
        cbar1 = fig.colorbar(sc, ax=axes[1], shrink=0.85)
        cbar1.set_label("−fitness (log scale)")
    axes[1].set_title("Archive Measures (2-param singlet)")
    axes[1].set_xlabel(label0)
    axes[1].set_ylabel(label1)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "RL_CONFIG", "optical_rl/configs/qd_singlet_cmamega_two_param.yaml"
        ),
        help="Path to the YAML config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    qd_cfg = cfg["qd"]
    _ensure_dirs(qd_cfg["results_dir"])
    run_dir = _new_run_dir(
        qd_cfg["results_dir"], qd_cfg.get("run_name", "singlet_cmamega_two_param")
    )

    config_dst = run_dir / Path(args.config).name
    shutil.copy2(args.config, config_dst)
    print(f"[singlet-2p-cmamega] Config saved to: {config_dst}", flush=True)

    archive_bounds = qd_cfg.get("archive_bounds")
    if archive_bounds is not None:
        archive_bounds = (tuple(archive_bounds[0]), tuple(archive_bounds[1]))

    raw_focal = qd_cfg.get("focal_length_target_mm")
    focal_length_target_mm = float(raw_focal) if raw_focal is not None else None

    config = SingletCMAMEGAConfig(
        lens_path=str(qd_cfg["lens_path"]),
        object_depth_mm=float(qd_cfg.get("object_depth_mm", -1000.0)),
        wavelength_um=float(qd_cfg.get("wavelength_um", 0.587)),
        theta_bounds=tuple(tuple(x) for x in qd_cfg.get("theta_bounds", [[-0.1, 0.1], [-0.1, 0.1]])),
        focal_length_target_mm=focal_length_target_mm,
        descriptor_zernike_indices=tuple(qd_cfg.get("descriptor_zernike_indices", [4, 11])),
        archive_bins=tuple(qd_cfg.get("archive_bins", [64, 64])),
        archive_bounds=archive_bounds,
        invalid_tol=float(qd_cfg.get("invalid_tol", 0.05)),
        invalid_penalty_weight=float(qd_cfg.get("invalid_penalty_weight", 5.0)),
        cma_eta=float(qd_cfg.get("cma_eta", 0.05)),
        seed_count=int(qd_cfg.get("seed_count", 128)),
        calibration_count=int(qd_cfg.get("calibration_count", 512)),
        num_rays=int(qd_cfg.get("num_rays", 256)),
        alpha_deg=float(qd_cfg.get("alpha_deg", 0.5)),
        stage=str(qd_cfg.get("stage", "stage1")),
        stage2_field_deg=float(qd_cfg.get("stage2_field_deg", 2.0)),
        zernike_field_deg=float(qd_cfg.get("zernike_field_deg", 0.0)),
        device=str(qd_cfg.get("device", "auto")),
        initial_sigma=float(qd_cfg.get("initial_sigma", 0.75)),
        population_size=int(qd_cfg.get("population_size", 16)),
        cma_seed=int(qd_cfg.get("seed", 0)),
        results_dir=str(qd_cfg["results_dir"]),
        run_name=str(qd_cfg.get("run_name", "singlet_cmamega_two_param")),
        log_every=int(qd_cfg.get("log_every", 50)),
    )

    runner = TwoParamSingletCMAMEGARunner(config)

    if config.archive_bounds is None:
        bounds = runner.calibrate_archive_bounds()
        print(f"[singlet-2p-cmamega] Calibrated archive bounds: {bounds}", flush=True)

    seed_stats = runner.seed_archive()
    print(f"[singlet-2p-cmamega] Seed stats: {seed_stats}", flush=True)

    final_stats = runner.run(int(qd_cfg.get("iterations", 2000)))
    print(f"[singlet-2p-cmamega] Final stats: {final_stats}", flush=True)

    _save_archive_json(run_dir / "archive.json", runner)
    _save_archive_plot(run_dir / "archive_map.png", runner, config)
    (run_dir / "summary.json").write_text(json.dumps(final_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
