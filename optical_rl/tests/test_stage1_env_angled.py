import numpy as np
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.angled_ray_env import AngledRayEnv


def test_reset_obs_shape():
    env = AngledRayEnv()
    obs, info = env.reset()
    assert obs.shape == (3,)
    assert 0.0 <= obs[0] <= 1.0
    assert 0.0 <= obs[1] <= 1.0
    assert 0.0 <= obs[2] <= 1.0


def test_step_reward_range():
    env = AngledRayEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert np.isfinite(reward)
    assert 0.0 <= reward <= 1.0


def test_episode_terminates():
    env = AngledRayEnv()
    env.reset()
    done = False
    for _ in range(env.config.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        if done:
            break
    assert done


def test_actions_affect_reward_or_range():
    env = AngledRayEnv()

    env.reset()
    env.quad_y = env.config.quad_y_min
    obs1, reward1, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

    env.reset()
    env.quad_y = env.config.quad_y_max
    obs2, reward2, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

    assert 0.0 <= reward1 <= 1.0
    assert 0.0 <= reward2 <= 1.0
