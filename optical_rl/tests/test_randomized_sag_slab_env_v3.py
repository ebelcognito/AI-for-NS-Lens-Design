import numpy as np
import sys
from pathlib import Path
from types import MethodType

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.randomized_sag_slab_env_v3 import (
    RandomizedSagSlabConfig,
    RandomizedSagSlabEnvV3,
)


def test_observation_shape_and_ranges():
    env = RandomizedSagSlabEnvV3()
    obs, _ = env.reset()
    assert obs.shape == (13,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_action_space_shape():
    env = RandomizedSagSlabEnvV3()
    sample = env.action_space.sample()
    assert sample.shape == (2,)


def test_reset_samples_ranges():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    assert env.config.source_y_min <= env.episode_source_y <= env.config.source_y_max
    assert env.config.detector_y_min <= env.episode_detector_y <= env.config.detector_y_max
    assert env.config.slope_min <= env.episode_slope <= env.config.slope_max
    assert env.config.quad_y_min <= env.quad_y <= env.config.quad_y_max
    assert (
        env.config.sag_depth_frac_min
        <= env.episode_sag_depth_frac
        <= env.config.sag_depth_frac_max
    )


def test_geometry_constant_within_episode():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    source_y = env.episode_source_y
    detector_y = env.episode_detector_y
    slope = env.episode_slope
    sag_start = env.episode_sag_depth_frac

    for _ in range(5):
        env.step(np.array([0.2, 0.7], dtype=np.float32))
        assert env.episode_source_y == source_y
        assert env.episode_detector_y == detector_y
        assert env.episode_slope == slope
    assert env.episode_sag_depth_frac != sag_start


def test_quad_y_clipping():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    for _ in range(20):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.quad_y <= env.config.quad_y_max
    for _ in range(40):
        env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert env.quad_y >= env.config.quad_y_min


def test_sag_clipping():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    for _ in range(100):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.episode_sag_depth_frac <= env.config.sag_depth_frac_max
    for _ in range(200):
        env.step(np.array([0.0, -1.0], dtype=np.float32))
    assert env.episode_sag_depth_frac >= env.config.sag_depth_frac_min


def test_reward_is_finite():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert reward <= 1.0


def test_boundary_penalty_applies():
    cfg = RandomizedSagSlabConfig(
        action_scale_y=100.0,
        boundary_penalty=0.05,
        max_steps=4,
        verbose_timing=False,
        use_persistent_runner=False,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    env.reset(seed=7)
    env.quad_y = env.config.quad_y_max - 0.001
    env.episode_sag_depth_frac = 0.0
    _, reward_clipped, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    _, reward_unclipped, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert reward_clipped < reward_unclipped + 1e-9


def test_episode_terminates():
    env = RandomizedSagSlabEnvV3()
    env.reset()
    done = False
    for _ in range(env.config.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        if done:
            break
    assert done


def test_early_termination_no_improve():
    cfg = RandomizedSagSlabConfig(
        max_steps=500,
        no_improve_patience=3,
        clip_streak_patience=999,
        improve_epsilon=1e-6,
        early_term_enabled=True,
        verbose_timing=False,
        use_persistent_runner=False,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    env.reset(seed=123)
    done = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated
        if done:
            break
    assert done


def test_early_termination_clip_streak_y():
    cfg = RandomizedSagSlabConfig(
        max_steps=500,
        no_improve_patience=999,
        clip_streak_patience=3,
        action_scale_y=100.0,
        early_term_enabled=True,
        verbose_timing=False,
        use_persistent_runner=False,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    env.reset(seed=42)
    env.quad_y = env.config.quad_y_max - 0.001
    done = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
        done = terminated or truncated
        if done:
            break
    assert done


def test_early_termination_clip_streak_sag():
    cfg = RandomizedSagSlabConfig(
        max_steps=500,
        no_improve_patience=999,
        clip_streak_patience=3,
        action_scale_sag=1.0,
        early_term_enabled=True,
        verbose_timing=False,
        use_persistent_runner=False,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    env.reset(seed=42)
    env.episode_sag_depth_frac = env.config.sag_depth_frac_max - 1e-4
    done = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        done = terminated or truncated
        if done:
            break
    assert done


def test_values_vary_across_resets():
    env = RandomizedSagSlabEnvV3()
    seen = set()
    for _ in range(6):
        env.reset()
        seen.add(
            (
                round(env.episode_source_y, 3),
                round(env.episode_detector_y, 3),
                round(env.episode_slope, 4),
                round(env.quad_y, 3),
                round(env.episode_sag_depth_frac, 3),
            )
        )
    assert len(seen) > 1


def test_persistent_runner_start_stop():
    cfg = RandomizedSagSlabConfig(
        use_persistent_runner=True,
        max_steps=2,
        verbose_timing=False,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    obs, _ = env.reset()
    assert obs.shape == (13,)
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    env.close()
    assert env._runner is None


def _attach_constant_power_simulator(env, power_value: float):
    def _simulate_const(self):
        return {"detectors": [{"power": power_value}]}, {"total_ms": 0.0}

    env._simulate = MethodType(_simulate_const, env)


def test_high_power_suppresses_early_term():
    cfg = RandomizedSagSlabConfig(
        use_persistent_runner=False,
        max_steps=30,
        early_term_enabled=True,
        no_improve_patience=50,
        clip_streak_patience=50,
        good_enough_power_threshold=0.90,
        good_power_patience=3,
        max_initial_power=1.0,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    _attach_constant_power_simulator(env, 0.95)
    env.reset(seed=1)

    done = False
    last_info = {}
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        done = terminated or truncated
        last_info = info
        if done:
            break

    assert not done
    assert last_info["high_power_confirmed"] is True


def test_below_threshold_early_term_still_applies():
    cfg = RandomizedSagSlabConfig(
        use_persistent_runner=False,
        max_steps=100,
        early_term_enabled=True,
        no_improve_patience=2,
        clip_streak_patience=99,
        improve_epsilon=1e-6,
        good_enough_power_threshold=0.90,
        good_power_patience=3,
        max_initial_power=1.0,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    _attach_constant_power_simulator(env, 0.89)
    env.reset(seed=2)

    done = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated
        if done:
            break

    assert done


def test_high_power_requires_patience():
    cfg = RandomizedSagSlabConfig(
        use_persistent_runner=False,
        max_steps=100,
        early_term_enabled=True,
        no_improve_patience=50,
        clip_streak_patience=99,
        improve_epsilon=1e-6,
        good_enough_power_threshold=0.90,
        good_power_patience=20,
        max_initial_power=1.0,
    )
    env = RandomizedSagSlabEnvV3(cfg)
    _attach_constant_power_simulator(env, 0.95)
    env.reset(seed=3)

    info_before = {}
    done = False
    for _ in range(19):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated
        info_before = info
        if done:
            break

    assert not done
    assert info_before["high_power_confirmed"] is False

    _, _, terminated, truncated, info_after = env.step(np.array([0.0, 0.0], dtype=np.float32))
    done = terminated or truncated
    assert not done
    assert info_after["high_power_confirmed"] is True
