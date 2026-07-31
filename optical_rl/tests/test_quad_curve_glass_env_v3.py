import numpy as np
import sys
from pathlib import Path
from types import MethodType

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.quad_curve_glass_env_v3 import (
    QuadCurveGlassConfigV3,
    QuadCurveGlassEnvV3,
)


def _attach_constant_power_simulator(env, power_value: float):
    def _simulate_const(self):
        return {"detectors": [{"power": power_value}], "error": None}, {"total_ms": 0.0}

    env._simulate = MethodType(_simulate_const, env)


def _random_bundle_cfg(**overrides):
    cfg = QuadCurveGlassConfigV3(
        use_persistent_runner=False,
        ray_mode="bundle",
        ray_count=5,
        ray_bundle_sampling="random_uniform_angle",
        ray_bundle_target_x=290.0,
        ray_bundle_angle_min_target_x=290.0,
        ray_bundle_angle_min_target_y=130.0,
        ray_bundle_angle_max_target_x=290.0,
        ray_bundle_angle_max_target_y=170.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _ray_angles(env):
    source_x = float(env.config.source_x)
    source_y = float(env.config.source_y)
    return [
        float(np.arctan2(target_y - source_y, target_x - source_x))
        for target_x, target_y in env.ray_targets
    ]


def test_scene_contains_five_rays_in_random_uniform_angle_bundle():
    env = QuadCurveGlassEnvV3(_random_bundle_cfg())
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset(seed=7)
        scene = env._build_scene()
        ray_count = sum(1 for obj in scene["objs"] if obj.get("type") == "SingleRay")
        assert ray_count == 5
    finally:
        env.close()


def test_random_uniform_angle_bundle_targets_stay_within_requested_span():
    cfg = _random_bundle_cfg()
    env = QuadCurveGlassEnvV3(cfg)
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset(seed=11)
        angle_min, angle_max = env._compute_angle_bounds(cfg)
        for angle in _ray_angles(env):
            assert angle_min <= angle <= angle_max
    finally:
        env.close()


def test_random_uniform_angle_bundle_varies_across_episode_seeds():
    env = QuadCurveGlassEnvV3(_random_bundle_cfg())
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        seen = set()
        for seed in [1, 2, 3, 4, 5]:
            env.reset(seed=seed)
            seen.add(tuple((round(x, 6), round(y, 6)) for x, y in env.ray_targets))
        assert len(seen) > 1
    finally:
        env.close()


def test_random_uniform_angle_bundle_is_reproducible_for_same_seed():
    cfg = _random_bundle_cfg()
    env_a = QuadCurveGlassEnvV3(cfg)
    env_b = QuadCurveGlassEnvV3(cfg)
    try:
        _attach_constant_power_simulator(env_a, power_value=0.0)
        _attach_constant_power_simulator(env_b, power_value=0.0)
        env_a.reset(seed=23)
        env_b.reset(seed=23)
        assert np.allclose(np.array(env_a.ray_targets), np.array(env_b.ray_targets))
    finally:
        env_a.close()
        env_b.close()


def test_deterministic_bundle_mode_still_matches_v2_behavior():
    cfg = QuadCurveGlassConfigV3(
        use_persistent_runner=False,
        ray_mode="bundle",
        ray_count=5,
        ray_bundle_sampling="deterministic",
        ray_bundle_target_x=290.0,
        ray_bundle_y_min=140.0,
        ray_bundle_y_max=160.0,
    )
    env = QuadCurveGlassEnvV3(cfg)
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset(seed=5)
        expected_ys = np.linspace(140.0, 160.0, num=5, dtype=np.float64)
        expected_targets = [(290.0, float(y)) for y in expected_ys]
        assert np.allclose(np.array(env.ray_targets), np.array(expected_targets))
    finally:
        env.close()


def test_explicit_mode_behavior_unchanged_in_v3():
    cfg = QuadCurveGlassConfigV3(
        use_persistent_runner=False,
        ray_mode="explicit",
        ray1_target_x=290.0,
        ray1_target_y=170.0,
        ray2_target_x=290.0,
        ray2_target_y=130.0,
        ray3_target_x=290.0,
        ray3_target_y=180.0,
        ray4_target_x=290.0,
        ray4_target_y=120.0,
    )
    env = QuadCurveGlassEnvV3(cfg)
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset(seed=3)
        assert env.ray_targets == [
            (290.0, 170.0),
            (290.0, 130.0),
            (290.0, 180.0),
            (290.0, 120.0),
        ]
    finally:
        env.close()


def test_reset_keeps_same_bundle_within_episode():
    env = QuadCurveGlassEnvV3(_random_bundle_cfg(max_steps=3))
    try:
        _attach_constant_power_simulator(env, power_value=0.0)
        env.reset(seed=19)
        before_step = np.array(env.ray_targets, dtype=np.float64)
        env.step(np.zeros((12,), dtype=np.float32))
        after_step = np.array(env.ray_targets, dtype=np.float64)
        assert np.allclose(before_step, after_step)
    finally:
        env.close()
