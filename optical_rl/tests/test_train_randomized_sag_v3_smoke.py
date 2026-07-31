import os
import sys
from pathlib import Path

import pytest
import yaml

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.main_stage1_randomized_sag_v3 import main


@pytest.mark.skipif(
    os.environ.get("RL_RUN_SMOKE", "0") != "1",
    reason="Set RL_RUN_SMOKE=1 to run PPO smoke training test",
)
def test_training_smoke(tmp_path, monkeypatch):
    config = {
        "env": {
            "world_width": 300,
            "world_height": 100,
            "source_x": 0,
            "detector_x": 300,
            "quad_center_x": 150,
            "source_y_min": 49,
            "source_y_max": 51,
            "detector_y_min": 62,
            "detector_y_max": 68,
            "slope_min": 0.04,
            "slope_max": 0.06,
            "detector_height": 5,
            "quad_size": 40,
            "ref_index": 1.5,
            "ray_dx": 30,
            "quad_y_min": 20,
            "quad_y_max": 80,
            "sag_depth_frac_min": -0.25,
            "sag_depth_frac_max": 0.25,
            "action_scale_y": 5,
            "action_scale_sag": 0.01,
            "max_steps": 32,
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
            "use_persistent_runner": True,
            "curriculum": [
                {
                    "name": "smoke",
                    "timesteps": 2048,
                    "env_overrides": {},
                }
            ],
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

    final_model = tmp_path / "results" / "models" / "final_model_randomized_sag_v3_dualctrl.zip"
    holdout_json = tmp_path / "results" / "logs" / "holdout_eval_randomized_sag_v3.json"
    assert final_model.exists()
    assert holdout_json.exists()
