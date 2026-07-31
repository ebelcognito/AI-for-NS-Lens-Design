import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_rl.envs.dual_surface_curvature_env_v2 import (
    DualSurfaceCurvatureConfigV2,
    DualSurfaceCurvatureEnvV2,
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


def _build_env_cfg(base_env_cfg: Dict, training_cfg: Dict) -> DualSurfaceCurvatureConfigV2:
    env_cfg_dict = dict(base_env_cfg)
    env_cfg_dict["use_persistent_runner"] = bool(training_cfg.get("use_persistent_runner", True))
    return DualSurfaceCurvatureConfigV2(**env_cfg_dict)


def _evaluate_holdout(
    model: PPO,
    env_cfg: DualSurfaceCurvatureConfigV2,
    seed_start: int,
    n_episodes: int,
    success_threshold: float,
) -> Dict:
    env = DualSurfaceCurvatureEnvV2(env_cfg)
    episode_returns: List[float] = []
    final_powers: List[float] = []
    best_powers: List[float] = []

    try:
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed_start + i)
            done = False
            total_reward = 0.0
            best_power = float(env.current_power)

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                best_power = max(best_power, float(env.current_power))

            episode_returns.append(total_reward)
            final_powers.append(float(env.current_power))
            best_powers.append(best_power)
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
    }
    return summary


def main():
    config_path = os.environ.get(
        "RL_CONFIG", "optical_rl/configs/stage1_dual_surface_curvature_v2.yaml"
    )
    cfg = load_config(config_path)

    logging_cfg = cfg["logging"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dual_surface_curvature_v2_{timestamp}"

    root_results_dir = Path(logging_cfg["results_dir"])
    root_logs_dir = Path(logging_cfg["logs_dir"])
    root_models_dir = Path(logging_cfg["models_dir"])
    root_tb_dir = Path(logging_cfg["tensorboard_dir"])
    root_viz_dir = Path(logging_cfg["visualizations_dir"])

    run_logs_dir = root_logs_dir / run_name
    run_models_dir = root_models_dir / run_name
    run_tb_dir = root_tb_dir / run_name
    run_viz_dir = root_viz_dir / run_name

    _ensure_dirs(
        root_results_dir,
        root_logs_dir,
        root_models_dir,
        root_tb_dir,
        root_viz_dir,
        run_logs_dir,
        run_models_dir,
        run_tb_dir,
        run_viz_dir,
    )
    print(f"[train] Run directory: {run_name}", flush=True)

    config_src = Path(config_path)
    config_dst = run_models_dir / config_src.name
    shutil.copy2(config_src, config_dst)
    print(f"[train] Config saved to: {config_dst}", flush=True)

    metrics = MetricsLogger(
        log_dir=run_tb_dir,
        csv_path=run_logs_dir / "episode_metrics_dual_surface_curvature_v2.csv",
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
    eval_verbose = int(callbacks_cfg.get("eval_verbose", 0))

    device = training_cfg.get("device", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env_cfg = _build_env_cfg(base_env_cfg, training_cfg)
    env = DualSurfaceCurvatureEnvV2(env_cfg)
    eval_env = DualSurfaceCurvatureEnvV2(env_cfg)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(run_models_dir),
        log_path=str(run_logs_dir),
        deterministic=True,
        verbose=eval_verbose,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(run_models_dir),
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
        tensorboard_log=str(run_tb_dir),
        device=device,
    )

    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, progress_callback],
        reset_num_timesteps=False,
        tb_log_name="PPO_dual_surface_curvature_v2",
    )

    env.close()
    eval_env.close()

    model.save(str(run_models_dir / "final_model_dual_surface_curvature_v2"))

    eval_env_cfg = _build_env_cfg(base_env_cfg, training_cfg)
    eval_env = DualSurfaceCurvatureEnvV2(eval_env_cfg)
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
    holdout_summary["run_name"] = run_name
    holdout_summary["total_timesteps"] = int(training_cfg["total_timesteps"])
    holdout_path = run_logs_dir / "holdout_eval_dual_surface_curvature_v2.json"
    holdout_path.write_text(json.dumps(holdout_summary, indent=2), encoding="utf-8")

    print("Holdout summary:")
    print(json.dumps(holdout_summary, indent=2))


if __name__ == "__main__":
    main()
