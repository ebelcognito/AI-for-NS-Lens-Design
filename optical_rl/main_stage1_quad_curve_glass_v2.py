import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Allow running as a script: `python optical_rl/main_stage1_quad_curve_glass.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.envs.quad_curve_glass_env_v2 import (
    QuadCurveGlassConfigV2,
    QuadCurveGlassEnvV2,
)
from optical_rl.utils.config import load_config


def _ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def _next_tensorboard_run_dir(base_dir: str) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    indices = []
    for d in run_dirs:
        try:
            indices.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_idx = max(indices, default=-1) + 1
    run_path = base / f"run_{next_idx}"
    run_path.mkdir(parents=True, exist_ok=True)
    return str(run_path)


def _new_model_run_dir(base_dir: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"quad_curve_glass_v2_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_env_cfg(base_env_cfg: Dict, training_cfg: Dict) -> QuadCurveGlassConfigV2:
    env_cfg_dict = dict(base_env_cfg)
    env_cfg_dict["use_persistent_runner"] = bool(training_cfg.get("use_persistent_runner", True))
    return QuadCurveGlassConfigV2(**env_cfg_dict)


class ProgressCallback(BaseCallback):
    def __init__(self, every_n_steps=10000):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.every_n_steps == 0:
            elapsed = time.perf_counter() - self._start_time
            print(
                f"[train] step={self.num_timesteps} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get("RL_CONFIG", "optical_rl/configs/stage1_quad_curve_glass_v2.yaml"),
        help="Path to the YAML config (defaults to $RL_CONFIG or stage1_quad_curve_glass_v2.yaml).",
    )
    parser.add_argument(
        "--pretrained-model",
        default=os.environ.get("RL_PRETRAINED_MODEL", None),
        help="Optional path to a .zip model to load before training (also via $RL_PRETRAINED_MODEL).",
    )
    args, _unknown = parser.parse_known_args(argv)

    config_path = args.config
    cfg = load_config(config_path)

    logging_cfg = cfg["logging"]
    _ensure_dirs(
        logging_cfg["results_dir"],
        logging_cfg["logs_dir"],
        logging_cfg["models_dir"],
        logging_cfg["tensorboard_dir"],
        logging_cfg["visualizations_dir"],
    )
    tensorboard_run_dir = _next_tensorboard_run_dir(logging_cfg["tensorboard_dir"])
    model_run_dir = _new_model_run_dir(logging_cfg["models_dir"])

    # Copy config file to model directory for reproducibility
    config_src = Path(config_path)
    config_dst = model_run_dir / config_src.name
    shutil.copy2(config_src, config_dst)
    print(f"[train] Config saved to: {config_dst}", flush=True)

    training_cfg = cfg["training"]
    callbacks_cfg = cfg.get("callbacks", {})
    debug_cfg = cfg.get("debug", {})
    base_env_cfg = cfg["env"].copy()
    base_env_cfg.update(debug_cfg)

    device = training_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env_cfg = _build_env_cfg(base_env_cfg, training_cfg)
    env = Monitor(QuadCurveGlassEnvV2(env_cfg))
    eval_env = Monitor(QuadCurveGlassEnvV2(env_cfg))

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=int(callbacks_cfg.get("eval_freq", 4096)),
        n_eval_episodes=int(callbacks_cfg.get("n_eval_episodes", 5)),
        best_model_save_path=str(model_run_dir),
        log_path=logging_cfg["logs_dir"],
        deterministic=True,
        verbose=int(callbacks_cfg.get("eval_verbose", 0)),
    )
    progress_callback = ProgressCallback(
        every_n_steps=int(callbacks_cfg.get("progress_freq", 10000))
    )

    pretrained_model_path = args.pretrained_model or training_cfg.get("pretrained_model_path")
    if pretrained_model_path:
        pretrained_model_path = str(Path(pretrained_model_path).expanduser())
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"pretrained model not found: {pretrained_model_path}")
        print(f"[train] Loading pretrained model: {pretrained_model_path}", flush=True)
        model = PPO.load(
            pretrained_model_path,
            env=env,
            device=device,
            tensorboard_log=tensorboard_run_dir,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            ent_coef=float(training_cfg.get("ent_coef", 0.0)),
            learning_rate=float(training_cfg["learning_rate"]),
            n_steps=int(training_cfg["n_steps"]),
            batch_size=int(training_cfg["batch_size"]),
            n_epochs=int(training_cfg["n_epochs"]),
            gamma=float(training_cfg["gamma"]),
            verbose=0,
            tensorboard_log=tensorboard_run_dir,
            device=device,
        )

    model.learn(
        total_timesteps=int(training_cfg["total_timesteps"]),
        callback=[eval_callback, progress_callback],
        reset_num_timesteps=False,
        tb_log_name="PPO_quad_curve_glass_v2",
    )

    env.close()
    eval_env.close()

    best_model_path = model_run_dir / "best_model.zip"
    if best_model_path.exists():
        print(f"[train] best model saved to: {best_model_path}", flush=True)
    else:
        print(
            f"[train] warning: best_model.zip not found in {model_run_dir} "
            f"(check eval_freq/total_timesteps).",
            flush=True,
        )


if __name__ == "__main__":
    main()
