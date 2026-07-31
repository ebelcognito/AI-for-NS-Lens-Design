import os
import sys
from pathlib import Path

import pytest
import yaml

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.main_stage1_single_surface_curvature_v1 import main


@pytest.mark.skipif(
    os.environ.get("RL_RUN_SMOKE", "0") != "1",
    reason="Set RL_RUN_SMOKE=1 to run PPO smoke training test",
)
def test_training_smoke(tmp_path, monkeypatch):
    config = {
        "env": {
            "world_width": 300,
            "world_height": 300,
            "slab_right_x": 260,
            "slab_thickness": 3,
            "slab_height": 15,
            "slab_center_y": 150,
            "detector_x": 290,
            "detector_center_y": 150,
            "detector_height": 3,
            "ref_index": 1.5,
            "sag_min_mm": -2.0,
            "sag_max_mm": 2.0,
            "action_scale_sag_mm": 0.2,
            "ray_origin_x": 10,
            "ray_target_x": 290,
            "ray_target_y_min": 145,
            "ray_target_y_max": 155,
            "object_depth_mm": 1000,
            "mapping_numerator_mm": 710,
            "max_steps": 32,
            "max_initial_power": 0.0,
            "max_reset_attempts": 50,
            "boundary_penalty": 0.01,
        },
        "training": {
            "algorithm": "PPO",
            "device": "cpu",
            "learning_rate": 0.0003,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "total_timesteps": 2048,
            "ent_coef": 0.01,
            "use_persistent_runner": True,
        },
        "callbacks": {
            "eval_freq": 512,
            "checkpoint_freq": 1024,
            "progress_freq": 256,
            "n_eval_episodes": 2,
            "eval_verbose": 0,
        },
        "logging": {
            "results_dir": str(tmp_path / "results"),
            "logs_dir": str(tmp_path / "results" / "logs"),
            "models_dir": str(tmp_path / "results" / "models"),
            "tensorboard_dir": str(tmp_path / "results" / "tb"),
            "visualizations_dir": str(tmp_path / "results" / "viz"),
        },
        "evaluation": {
            "n_holdout_episodes": 5,
            "seed_start": 1234,
            "success_threshold": 0.60,
        },
        "debug": {
            "verbose_timing": False,
            "timing_log_interval": 100,
        },
    }

    cfg_path = tmp_path / "smoke_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setenv("RL_CONFIG", str(cfg_path))
    main()

    model_runs = sorted((tmp_path / "results" / "models").glob("single_surface_curvature_v1_*"))
    log_runs = sorted((tmp_path / "results" / "logs").glob("single_surface_curvature_v1_*"))
    assert len(model_runs) >= 1
    assert len(log_runs) >= 1

    final_model = model_runs[-1] / "final_model_single_surface_curvature_v1.zip"
    holdout_json = log_runs[-1] / "holdout_eval_single_surface_curvature_v1.json"
    assert final_model.exists()
    assert holdout_json.exists()
