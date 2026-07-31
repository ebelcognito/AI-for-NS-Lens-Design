import os
import shutil
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from optical_rl.envs.angled_ray_env import AngledRayConfig, AngledRayEnv
from optical_rl.utils.config import load_config
from optical_rl.utils.metrics import MetricsLogger


def _ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)



def test_random_baseline(env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"Random baseline: {mean:.3f} ± {std:.3f}")
    return mean


class ProgressCallback(BaseCallback):
    def __init__(self, every_n_steps=500, verbose_timing=False):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.verbose_timing = verbose_timing
        
        # Timing tracking
        self.rollout_start_time = None
        self.rollout_end_time = None
        self.training_start_time = None
        self.training_end_time = None
        self.rollout_count = 0
        
        self.cumulative_rollout_time = 0.0
        self.cumulative_training_time = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_n_steps == 0:
            print(f"step {self.num_timesteps}", flush=True)
        return True
    
    def _on_rollout_start(self) -> None:
        """Called before collecting rollout data."""
        if self.verbose_timing:
            self.rollout_start_time = time.perf_counter()
    
    def _on_rollout_end(self) -> None:
        """Called after rollout collection, before training update."""
        if self.verbose_timing and self.rollout_start_time is not None:
            self.rollout_end_time = time.perf_counter()
            rollout_time = self.rollout_end_time - self.rollout_start_time
            self.cumulative_rollout_time += rollout_time
            
            # Training update happens after this callback returns
            self.training_start_time = time.perf_counter()
            
    def _on_training_end(self) -> None:
        """Called at the end of training (not per-update)."""
        if self.verbose_timing:
            print(f"\n[PPO] Training complete:", flush=True)
            print(f"  Total rollout time: {self.cumulative_rollout_time:.1f}s", flush=True)
            print(f"  Total training time: {self.cumulative_training_time:.1f}s", flush=True)
            if self.cumulative_rollout_time + self.cumulative_training_time > 0:
                total = self.cumulative_rollout_time + self.cumulative_training_time
                rollout_pct = 100 * self.cumulative_rollout_time / total
                training_pct = 100 * self.cumulative_training_time / total
                print(f"  Rollout: {rollout_pct:.1f}% | Training: {training_pct:.1f}%", flush=True)


def main():
    config_path = os.environ.get(
        "RL_CONFIG", "optical_rl/configs/stage1_angled.yaml"
    )
    cfg = load_config(config_path)
    # Merge debug/training options into env config if present
    env_config_dict = cfg["env"].copy()
    if "debug" in cfg:
        env_config_dict.update(cfg["debug"])
    env_config_dict["use_persistent_runner"] = bool(
        cfg.get("training", {}).get("use_persistent_runner", True)
    )
    
    env_cfg = AngledRayConfig(**env_config_dict)
    env = AngledRayEnv(env_cfg)
    eval_env = AngledRayEnv(env_cfg)

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
        csv_path=Path(logging_cfg["logs_dir"]) / "episode_metrics.csv",
    )
    # random_baseline = test_random_baseline(env)  # skipped for now
    random_baseline = None
    training_cfg = cfg["training"]
    callbacks_cfg = cfg.get("callbacks", {})

    # Sensible defaults tied to rollout length.
    eval_freq = int(callbacks_cfg.get("eval_freq", max(training_cfg["n_steps"] * 4, 1000)))
    checkpoint_freq = int(
        callbacks_cfg.get("checkpoint_freq", max(training_cfg["n_steps"] * 20, 5000))
    )
    progress_freq = int(
        callbacks_cfg.get("progress_freq", max(training_cfg["n_steps"], 500))
    )
    n_eval_episodes = int(callbacks_cfg.get("n_eval_episodes", 5))
    eval_verbose = int(callbacks_cfg.get("eval_verbose", 1))

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
    )
    
    # Get debug config for verbose timing
    debug_cfg = cfg.get("debug", {})
    verbose_timing = debug_cfg.get("verbose_timing", False)
    progress_callback = ProgressCallback(
        every_n_steps=progress_freq,
        verbose_timing=verbose_timing,
    )
    device = training_cfg.get("device", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=training_cfg["learning_rate"],
        n_steps=training_cfg["n_steps"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
        gamma=training_cfg["gamma"],
        verbose=1,
        tensorboard_log=logging_cfg["tensorboard_dir"],
        device=device,
    )
    print("Starting training...")
    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, progress_callback],
    )
    print("trained model")
    print("\nEvaluating trained policy...")
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    metrics.log_episode(0, total_reward, env_cfg.max_steps)
    metrics.close()

    print(f"Trained policy: {total_reward:.3f}")
    if random_baseline is not None and random_baseline > 0:
        improvement = (total_reward - random_baseline) / random_baseline * 100
        print(f"Improvement over random: {improvement:.1f}%")


if __name__ == "__main__":
    main()
