import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

# Allow running as a script: `python optical_rl/main_stage1_randomized_sag_v3.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.envs.randomized_sag_slab_env_v3 import (
    RandomizedSagSlabConfig,
    RandomizedSagSlabEnvV3,
)
from optical_rl.utils.config import load_config
from optical_rl.utils.metrics import MetricsLogger


def _ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


class ProgressCallback(BaseCallback):
    def __init__(self, every_n_steps=5000, verbose_timing=False):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.verbose_timing = verbose_timing
        self.rollout_start_time = None
        self.cumulative_rollout_time = 0.0

    def _on_step(self) -> bool:
        # if self.num_timesteps % self.every_n_steps == 0:
        #     print(f"step {self.num_timesteps}", flush=True)
        return True

    def _on_rollout_start(self) -> None:
        if self.verbose_timing:
            self.rollout_start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        if self.verbose_timing and self.rollout_start_time is not None:
            rollout_time = time.perf_counter() - self.rollout_start_time
            self.cumulative_rollout_time += rollout_time

    def _on_training_end(self) -> None:
        if self.verbose_timing:
            print(
                f"\n[PPO] Training complete. Total rollout time: {self.cumulative_rollout_time:.1f}s",
                flush=True,
            )


def _build_env_cfg(base_env_cfg: Dict, training_cfg: Dict) -> RandomizedSagSlabConfig:
    env_cfg_dict = dict(base_env_cfg)
    env_cfg_dict["use_persistent_runner"] = bool(training_cfg.get("use_persistent_runner", True))
    return RandomizedSagSlabConfig(**env_cfg_dict)


def _evaluate_holdout(model: PPO, env_cfg: RandomizedSagSlabConfig, seed_start: int, n_episodes: int, success_threshold: float) -> Dict:
    env = RandomizedSagSlabEnvV3(env_cfg)
    episode_returns: List[float] = []
    final_powers: List[float] = []
    best_powers: List[float] = []
    clipped_y_rate: List[float] = []
    clipped_sag_rate: List[float] = []
    clip_streak_y_final: List[float] = []
    clip_streak_sag_final: List[float] = []

    try:
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed_start + i)
            done = False
            total_reward = 0.0
            best_power = float(env.current_power)
            clipped_y_steps = 0.0
            clipped_sag_steps = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                best_power = max(best_power, float(env.current_power))
                clipped_y_steps += float(env.was_clipped_y)
                clipped_sag_steps += float(env.was_clipped_sag)
                steps += 1

            episode_returns.append(total_reward)
            final_powers.append(float(env.current_power))
            best_powers.append(best_power)
            denom = max(steps, 1)
            clipped_y_rate.append(clipped_y_steps / denom)
            clipped_sag_rate.append(clipped_sag_steps / denom)
            clip_streak_y_final.append(float(env.clip_streak_y))
            clip_streak_sag_final.append(float(env.clip_streak_sag))
    finally:
        env.close()

    best_arr = np.array(best_powers, dtype=np.float64)
    summary = {
        "n_episodes": int(n_episodes),
        "seed_start": int(seed_start),
        "success_threshold": float(success_threshold),
        "mean_episode_return": float(np.mean(episode_returns)),
        "mean_final_power": float(np.mean(final_powers)),
        "success_rate": float(np.mean(best_arr >= success_threshold)),
        "best_power_p10": float(np.percentile(best_arr, 10)),
        "best_power_p50": float(np.percentile(best_arr, 50)),
        "best_power_p90": float(np.percentile(best_arr, 90)),
        "mean_clipped_y_rate": float(np.mean(clipped_y_rate)),
        "mean_clipped_sag_rate": float(np.mean(clipped_sag_rate)),
        "mean_final_clip_streak_y": float(np.mean(clip_streak_y_final)),
        "mean_final_clip_streak_sag": float(np.mean(clip_streak_sag_final)),
    }
    return summary


def main():
    config_path = os.environ.get(
        "RL_CONFIG", "optical_rl/configs/stage1_randomized_sag_v3.yaml"
    )
    cfg = load_config(config_path)

    logging_cfg = cfg["logging"]
    _ensure_dirs(
        logging_cfg["results_dir"],
        logging_cfg["logs_dir"],
        logging_cfg["models_dir"],
        logging_cfg["tensorboard_dir"],
        logging_cfg["visualizations_dir"],
    )

    # Copy config file to model directory for reproducibility
    config_src = Path(config_path)
    config_dst = Path(logging_cfg["models_dir"]) / config_src.name
    shutil.copy2(config_src, config_dst)
    print(f"[train] Config saved to: {config_dst}", flush=True)

    metrics = MetricsLogger(
        log_dir=logging_cfg["tensorboard_dir"],
        csv_path=Path(logging_cfg["logs_dir"]) / "episode_metrics_randomized_sag_v3.csv",
    )

    training_cfg = cfg["training"]
    callbacks_cfg = cfg.get("callbacks", {})
    debug_cfg = cfg.get("debug", {})
    base_env_cfg = cfg["env"].copy()
    base_env_cfg.update(debug_cfg)

    eval_freq = int(
        callbacks_cfg.get("eval_freq", max(training_cfg["n_steps"] * 4, 1000))
    )
    checkpoint_freq = int(
        callbacks_cfg.get("checkpoint_freq", max(training_cfg["n_steps"] * 20, 50000))
    )
    progress_freq = int(
        callbacks_cfg.get("progress_freq", max(training_cfg["n_steps"], 50000))
    )
    n_eval_episodes = int(callbacks_cfg.get("n_eval_episodes", 10))
    eval_verbose = int(callbacks_cfg.get("eval_verbose",0))

    device = training_cfg.get("device", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env_cfg = _build_env_cfg(base_env_cfg, training_cfg)
    env = RandomizedSagSlabEnvV3(env_cfg)
    eval_env = RandomizedSagSlabEnvV3(env_cfg)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=logging_cfg["models_dir"],
        log_path=logging_cfg["logs_dir"],
        deterministic=True,
        verbose=eval_verbose,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=logging_cfg["models_dir"],
        name_prefix="model",
    )
    progress_callback = ProgressCallback(
        every_n_steps=progress_freq,
        verbose_timing=debug_cfg.get("verbose_timing", False),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        ent_coef=training_cfg.get("ent_coef", 0.01),
        learning_rate=training_cfg["learning_rate"],
        n_steps=training_cfg["n_steps"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
        gamma=training_cfg["gamma"],
        verbose=0,
        tensorboard_log=logging_cfg["tensorboard_dir"],
        device=device,
    )

    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, progress_callback],
        reset_num_timesteps=False,
        tb_log_name="PPO_randomized_sag_v3",
    )

    env.close()
    eval_env.close()

    model.save(str(Path(logging_cfg["models_dir"]) / "final_model_randomized_sag_v3_dualctrl"))

    # Single rollout summary metric in CSV logger for parity with existing scripts.
    eval_env_cfg = _build_env_cfg(base_env_cfg, training_cfg)
    eval_env = RandomizedSagSlabEnvV3(eval_env_cfg)
    obs, _ = eval_env.reset(seed=123)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
    metrics.log_episode(0, total_reward, eval_env_cfg.max_steps)
    metrics.close()
    eval_env.close()

    holdout_cfg = cfg.get("evaluation", {})
    holdout_summary = _evaluate_holdout(
        model,
        eval_env_cfg,
        seed_start=int(holdout_cfg.get("seed_start", 10000)),
        n_episodes=int(holdout_cfg.get("n_holdout_episodes", 200)),
        success_threshold=float(holdout_cfg.get("success_threshold", 0.60)),
    )
    holdout_summary["total_timesteps"] = int(training_cfg["total_timesteps"])
    holdout_path = Path(logging_cfg["logs_dir"]) / "holdout_eval_randomized_sag_v3.json"
    holdout_path.write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")

    print("Holdout summary:")
    print(json.dumps(holdout_summary, indent=2))


if __name__ == "__main__":
    main()
