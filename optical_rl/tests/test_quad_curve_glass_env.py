import numpy as np
import sys
from pathlib import Path
from types import MethodType

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.quad_curve_glass_env import QuadCurveGlassConfig, QuadCurveGlassEnv


def _attach_constant_power_simulator(env, power_value: float):
    def _simulate_const(self):
        return {"detectors": [{"power": power_value}], "error": None}, {"total_ms": 0.0}

    env._simulate = MethodType(_simulate_const, env)


def test_action_space_shape():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False))
    try:
        sample = env.action_space.sample()
        assert sample.shape == (12,)
    finally:
        env.close()


def test_observation_shape_and_ranges():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False, max_steps=10))
    try:
        _attach_constant_power_simulator(env, power_value=1.0)
        obs, _ = env.reset()
        assert obs.shape == (18,)
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
    finally:
        env.close()


def test_reset_is_fixed():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False))
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert np.allclose(obs1[:16], obs2[:16])
    finally:
        env.close()


def test_step_clips_points():
    cfg = QuadCurveGlassConfig(
        use_persistent_runner=False,
        action_scale_xy=1000.0,
        max_steps=5,
    )
    env = QuadCurveGlassEnv(cfg)
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset()
        for _ in range(3):
            env.step(np.ones((12,), dtype=np.float32))
        for x, y in env.points:
            assert 0.0 <= x <= cfg.world_width
            assert 0.0 <= y <= cfg.world_height
    finally:
        env.close()


def test_reward_finite():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False, max_steps=3))
    try:
        _attach_constant_power_simulator(env, power_value=1.7)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros((12,), dtype=np.float32))
        assert np.isfinite(reward)
    finally:
        env.close()


def test_step_handles_simulation_timeout():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False, max_steps=20))
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset()

        def _simulate_timeout(self):
            raise TimeoutError("timed out")

        env._simulate = MethodType(_simulate_timeout, env)
        _, reward, terminated, truncated, info = env.step(np.zeros((12,), dtype=np.float32))
        assert reward == 0.0
        assert terminated
        assert not truncated
        assert info["simulation_ok"] is False
    finally:
        env.close()


def test_corners_stay_in_quadrants_and_sag_bounded():
    cfg = QuadCurveGlassConfig(
        use_persistent_runner=False,
        action_scale_xy=1000.0,
        action_scale_sag=10.0,
    )
    env = QuadCurveGlassEnv(cfg)
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset()
        for _ in range(5):
            env.step(np.ones((12,), dtype=np.float32))

        for idx, (x, y) in enumerate(env.corners):
            xmin, xmax, ymin, ymax = env._quadrant_bounds(idx)
            assert xmin <= x <= xmax
            assert ymin <= y <= ymax

        assert env.corners[1][0] <= env.config.right_side_x_max
        assert env.corners[2][0] <= env.config.right_side_x_max

        for sag in env.sags:
            assert env.config.sag_min <= sag <= env.config.sag_max
    finally:
        env.close()


def test_scene_contains_four_rays():
    env = QuadCurveGlassEnv(QuadCurveGlassConfig(use_persistent_runner=False))
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset()
        scene = env._build_scene()
        ray_count = sum(1 for obj in scene["objs"] if obj.get("type") == "SingleRay")
        assert ray_count == 4
    finally:
        env.close()
