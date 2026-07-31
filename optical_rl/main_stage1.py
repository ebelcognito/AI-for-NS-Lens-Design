import os
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from optical_rl.envs.simple_steering_env import SimpleSteeringConfig, SimpleSteeringEnv
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


def main():
    config_path = os.environ.get(
        "RL_CONFIG", "optical_rl/configs/stage1_baseline.yaml"
    )
    cfg = load_config(config_path)

    env_cfg = SimpleSteeringConfig(**cfg["env"])
    env = SimpleSteeringEnv(env_cfg)
    eval_env = SimpleSteeringEnv(env_cfg)

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

    random_baseline = test_random_baseline(env)

    training_cfg = cfg["training"]
    callbacks_cfg = cfg.get("callbacks", {})

    eval_freq = int(callbacks_cfg.get("eval_freq", max(training_cfg["n_steps"] * 4, 1000)))
    checkpoint_freq = int(
        callbacks_cfg.get("checkpoint_freq", max(training_cfg["n_steps"] * 20, 5000))
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
        callback=[eval_callback, checkpoint_callback],
    )

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

    improvement = (
        (total_reward - random_baseline) / random_baseline * 100
        if random_baseline > 0
        else 0.0
    )
    print(f"Trained policy: {total_reward:.3f}")
    print(f"Improvement over random: {improvement:.1f}%")


if __name__ == "__main__":
    main()
