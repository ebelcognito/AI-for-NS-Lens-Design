import numpy as np
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.randomized_slab_env_v2 import (
    RandomizedSlab2DConfig,
    RandomizedSlab2DEnv,
)


def test_observation_shape_and_ranges():
    env = RandomizedSlab2DEnv()
    obs, _ = env.reset()
    assert obs.shape == (4,)
    assert 0.0 <= obs[0] <= 1.0
    assert 0.0 <= obs[1] <= 1.0
    assert 0.0 <= obs[2] <= 1.0
    assert 0.0 <= obs[3] <= 1.0


def test_action_space_shape():
    env = RandomizedSlab2DEnv()
    sample = env.action_space.sample()
    assert sample.shape == (2,)


def test_reset_samples_ranges():
    env = RandomizedSlab2DEnv()
    env.reset()
    assert env.config.source_y_min <= env.episode_source_y <= env.config.source_y_max
    assert env.config.detector_y_min <= env.episode_detector_y <= env.config.detector_y_max
    assert env.config.slope_min <= env.episode_slope <= env.config.slope_max
    assert env.config.quad_x_min <= env.quad_x <= env.config.quad_x_max
    assert env.config.quad_y_min <= env.quad_y <= env.config.quad_y_max


def test_geometry_constant_within_episode():
    env = RandomizedSlab2DEnv()
    env.reset()
    source_y = env.episode_source_y
    detector_y = env.episode_detector_y
    slope = env.episode_slope
    for _ in range(5):
        env.step(np.array([0.3, -0.4], dtype=np.float32))
        assert env.episode_source_y == source_y
        assert env.episode_detector_y == detector_y
        assert env.episode_slope == slope


def test_quad_xy_clipping():
    env = RandomizedSlab2DEnv()
    env.reset()
    for _ in range(20):
        env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.quad_x <= env.config.quad_x_max
    assert env.quad_y <= env.config.quad_y_max
    for _ in range(40):
        env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert env.quad_x >= env.config.quad_x_min
    assert env.quad_y >= env.config.quad_y_min


def test_reward_equals_power_and_finite():
    env = RandomizedSlab2DEnv()
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert 0.0 <= reward <= 1.0
    assert abs(reward - env.current_power) < 1e-9


def test_episode_terminates():
    env = RandomizedSlab2DEnv()
    env.reset()
    done = False
    for _ in range(env.config.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        if done:
            break
    assert done


def test_values_vary_across_resets():
    env = RandomizedSlab2DEnv()
    seen = set()
    for _ in range(6):
        env.reset()
        seen.add(
            (
                round(env.episode_source_y, 3),
                round(env.episode_detector_y, 3),
                round(env.episode_slope, 4),
                round(env.quad_x, 3),
                round(env.quad_y, 3),
            )
        )
    assert len(seen) > 1


def test_persistent_runner_start_stop():
    cfg = RandomizedSlab2DConfig(use_persistent_runner=True, max_steps=2, verbose_timing=False)
    env = RandomizedSlab2DEnv(cfg)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    env.close()
    assert env._runner is None
