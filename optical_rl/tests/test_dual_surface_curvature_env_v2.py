import numpy as np
import sys
from pathlib import Path
from types import MethodType

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.envs.dual_surface_curvature_env_v2 import (
    DualSurfaceCurvatureConfigV2,
    DualSurfaceCurvatureEnvV2,
)


def test_observation_shape_and_ranges():
    env = DualSurfaceCurvatureEnvV2()
    obs, _ = env.reset()
    assert obs.shape == (12,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_action_space_shape():
    env = DualSurfaceCurvatureEnvV2()
    sample = env.action_space.sample()
    assert sample.shape == (2,)


def test_ray_mapping_formula_applied_exactly():
    cfg = DualSurfaceCurvatureConfigV2(
        ray_target_y_min=150.0,
        ray_target_y_max=150.0001,
        mapping_numerator_mm=710.0,
        object_depth_mm=1000.0,
    )
    env = DualSurfaceCurvatureEnvV2(cfg)
    env.reset(seed=42)
    expected_origin = env.episode_target_y * 710.0 / 1000.0
    assert abs(env.episode_origin_y - expected_origin) < 1e-9


def test_scene_has_two_curved_faces_only():
    env = DualSurfaceCurvatureEnvV2()
    env.reset(seed=1)
    scene = env._build_scene()
    glass = next(obj for obj in scene["objs"] if obj["type"] == "Glass")
    path = glass["path"]
    assert len(path) == 6
    assert sum(1 for p in path if p["arc"]) == 2


def test_independent_clipping_and_non_intersection_guard():
    cfg = DualSurfaceCurvatureConfigV2(
        action_scale_front_sag_mm=10.0,
        action_scale_back_sag_mm=10.0,
        front_sag_min_mm=-100.0,
        front_sag_max_mm=100.0,
        back_sag_min_mm=-100.0,
        back_sag_max_mm=100.0,
        max_steps=2,
    )
    env = DualSurfaceCurvatureEnvV2(cfg)
    env.reset(seed=5)
    env.step(np.array([1.0, -1.0], dtype=np.float32))

    front_lo, front_hi = env._effective_front_sag_bounds()
    back_lo, back_hi = env._effective_back_sag_bounds()

    assert front_lo <= env.front_sag_mm <= front_hi
    assert back_lo <= env.back_sag_mm <= back_hi

    scene = env._build_scene()
    glass = next(obj for obj in scene["objs"] if obj["type"] == "Glass")
    path = glass["path"]
    right_arc_x = float(path[2]["x"])
    left_arc_x = float(path[5]["x"])
    assert right_arc_x > left_arc_x


def test_reset_rejects_initial_power_above_threshold():
    cfg = DualSurfaceCurvatureConfigV2(max_initial_power=0.0, max_reset_attempts=5)
    env = DualSurfaceCurvatureEnvV2(cfg)

    counter = {"n": 0}

    def _simulate_controlled(self):
        counter["n"] += 1
        if counter["n"] == 1:
            return {"detectors": [{"power": 0.2}]}, {"total_ms": 0.0}
        return {"detectors": [{"power": 0.0}]}, {"total_ms": 0.0}

    env._simulate = MethodType(_simulate_controlled, env)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,)
    assert counter["n"] >= 2
    assert env.current_power <= cfg.max_initial_power + 1e-9


def test_movement_penalty_reduces_reward():
    cfg = DualSurfaceCurvatureConfigV2(
        max_steps=3,
        movement_penalty_coef=1.0,
        action_scale_front_sag_mm=0.1,
        action_scale_back_sag_mm=0.1,
    )
    env = DualSurfaceCurvatureEnvV2(cfg)

    def _simulate_const(self):
        return {"detectors": [{"power": 0.5}]}, {"total_ms": 0.0}

    env._simulate = MethodType(_simulate_const, env)
    env.reset(seed=0)

    _, reward_move, _, _, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    _, reward_stay, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert reward_move < reward_stay


def test_reward_is_finite_and_episode_terminates():
    cfg = DualSurfaceCurvatureConfigV2(max_steps=5)
    env = DualSurfaceCurvatureEnvV2(cfg)
    env.reset(seed=2)

    done = False
    for _ in range(cfg.max_steps + 2):
        _, reward, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert np.isfinite(reward)
        done = terminated or truncated
        if done:
            break
    assert done
