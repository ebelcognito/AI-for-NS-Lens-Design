from dataclasses import dataclass
from typing import Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_single_ray_with_dual_curved_slab


@dataclass
class DualSurfaceCurvatureConfigV2:
    world_width: float = 300.0
    world_height: float = 300.0

    slab_right_x: float = 260.0
    slab_thickness: float = 3.0
    slab_height: float = 15.0
    slab_center_y: float = 150.0

    detector_x: float = 290.0
    detector_center_y: float = 150.0
    detector_height: float = 3.0

    ref_index: float = 1.5

    front_sag_min_mm: float = -2.0
    front_sag_max_mm: float = 2.0
    action_scale_front_sag_mm: float = 0.10

    back_sag_min_mm: float = -2.0
    back_sag_max_mm: float = 2.0
    action_scale_back_sag_mm: float = 0.10

    ray_origin_x: float = 10.0
    ray_target_x: float = 290.0
    ray_target_y_min: float = 145.0
    ray_target_y_max: float = 155.0
    object_depth_mm: float = 1000.0
    mapping_numerator_mm: float = 710.0

    max_steps: int = 100
    max_initial_power: float = 0.0
    max_reset_attempts: int = 100
    boundary_penalty: float = 0.01
    movement_penalty_coef: float = 0.001

    verbose_timing: bool = False
    timing_log_interval: int = 500
    use_persistent_runner: bool = False
    timeout_s: float = 5.0


class DualSurfaceCurvatureEnvV2(gym.Env):
    """
    2D single-ray environment with fixed slab and detector geometry.

    Both vertical slab face curvatures are controlled via two sag parameters:
    - front/left face sag
    - back/right face sag
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[DualSurfaceCurvatureConfigV2, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = DualSurfaceCurvatureConfigV2()
        elif isinstance(config, dict):
            self.config = DualSurfaceCurvatureConfigV2(**config)
        else:
            self.config = config

        if self.config.ray_target_y_min >= self.config.ray_target_y_max:
            raise ValueError("ray_target_y_min must be < ray_target_y_max")
        if self.config.object_depth_mm <= 0:
            raise ValueError("object_depth_mm must be > 0")

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 12, dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32,
        )

        self.front_sag_mm = 0.0
        self.back_sag_mm = 0.0
        self.episode_target_y = 0.0
        self.episode_origin_y = 0.0
        self.step_count = 0
        self.current_power = 0.0
        self.was_clipped_front = 0.0
        self.was_clipped_back = 0.0

        self._runner = None
        self.total_steps = 0
        self.cumulative_simulation_time_ms = 0.0
        self.cumulative_env_logic_time_ms = 0.0
        self.cumulative_step_time_ms = 0.0

        if self.config.use_persistent_runner:
            self._runner = create_runner(
                persistent=True,
                timeout_s=float(self.config.timeout_s),
                verbose=self.config.verbose_timing,
            )
            self._runner.start()

    @property
    def slab_left_x(self) -> float:
        return float(self.config.slab_right_x - self.config.slab_thickness)

    def _effective_front_sag_bounds(self):
        eps = 1e-3
        safe_min = max(float(self.config.front_sag_min_mm), -self.slab_left_x + eps)
        safe_max = min(float(self.config.front_sag_max_mm), float(self.config.slab_thickness) - eps)
        if safe_min >= safe_max:
            raise ValueError(
                f"Invalid front sag bounds after geometry constraints: {safe_min} >= {safe_max}"
            )
        return safe_min, safe_max

    def _effective_back_sag_bounds(self):
        eps = 1e-3
        safe_min = max(
            float(self.config.back_sag_min_mm),
            -float(self.config.slab_thickness) + eps,
        )
        safe_max = min(
            float(self.config.back_sag_max_mm),
            float(self.config.world_width) - float(self.config.slab_right_x) - eps,
        )
        if safe_min >= safe_max:
            raise ValueError(
                f"Invalid back sag bounds after geometry constraints: {safe_min} >= {safe_max}"
            )
        return safe_min, safe_max

    def _apply_coupled_sag_clipping(self, front_candidate: float, back_candidate: float):
        eps = 1e-3
        front_lo, front_hi = self._effective_front_sag_bounds()
        back_lo, back_hi = self._effective_back_sag_bounds()

        front_hi_coupled = min(
            front_hi,
            float(self.config.slab_right_x) + back_hi - eps - self.slab_left_x,
        )
        if front_hi_coupled < front_lo:
            raise ValueError(
                f"Infeasible coupled sag bounds: front_hi_coupled={front_hi_coupled} < front_lo={front_lo}"
            )

        front = float(np.clip(front_candidate, front_lo, front_hi_coupled))
        back_lo_coupled = max(
            back_lo,
            self.slab_left_x + front + eps - float(self.config.slab_right_x),
        )
        back = float(np.clip(back_candidate, back_lo_coupled, back_hi))
        return front, back

    def _sample_episode_ray(self):
        self.episode_target_y = float(
            self.np_random.uniform(
                self.config.ray_target_y_min,
                self.config.ray_target_y_max,
            )
        )
        self.episode_origin_y = float(
            self.episode_target_y
            * float(self.config.mapping_numerator_mm)
            / float(self.config.object_depth_mm)
        )

    def _build_scene(self):
        return scene_single_ray_with_dual_curved_slab(
            ray_origin_x=float(self.config.ray_origin_x),
            ray_origin_y=float(self.episode_origin_y),
            ray_target_x=float(self.config.ray_target_x),
            ray_target_y=float(self.episode_target_y),
            slab_right_x=float(self.config.slab_right_x),
            slab_thickness=float(self.config.slab_thickness),
            slab_center_y=float(self.config.slab_center_y),
            slab_height=float(self.config.slab_height),
            left_sag_mm=float(self.front_sag_mm),
            right_sag_mm=float(self.back_sag_mm),
            detector_x=float(self.config.detector_x),
            detector_center_y=float(self.config.detector_center_y),
            detector_half_height=float(self.config.detector_height) / 2.0,
            ref_index=float(self.config.ref_index),
        )

    def _simulate(self):
        scene = self._build_scene()
        if self._runner is not None:
            return self._runner.simulate(scene, episode=0, step=self.step_count)
        return run_simulation(
            scene,
            timeout_s=float(self.config.timeout_s),
            verbose=self.config.verbose_timing,
        )

    def _simulate_safe(self):
        try:
            result, timing = self._simulate()
            return True, result, timing
        except Exception as exc:
            if self.config.verbose_timing:
                print(f"[DualSurfaceCurvatureEnvV2] Simulation failed: {exc}", flush=True)
            return False, {"detectors": [], "error": str(exc)}, {"total_ms": 0.0}

    @staticmethod
    def _extract_power(result) -> float:
        if not isinstance(result, dict):
            return 0.0
        if result.get("error"):
            return 0.0
        detectors = result.get("detectors") or []
        if not detectors:
            return 0.0
        try:
            return float(detectors[0].get("power", 0.0))
        except Exception:
            return 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        front_lo, front_hi = self._effective_front_sag_bounds()
        back_lo, back_hi = self._effective_back_sag_bounds()

        accepted = False
        for attempt in range(int(self.config.max_reset_attempts)):
            self._sample_episode_ray()
            if attempt == 0:
                self.front_sag_mm = 0.0
                self.back_sag_mm = 0.0
            else:
                front_hi_coupled = min(
                    front_hi,
                    float(self.config.slab_right_x) + back_hi - 1e-3 - self.slab_left_x,
                )
                front_seed = float(self.np_random.uniform(front_lo, front_hi_coupled))
                back_seed = float(self.np_random.uniform(back_lo, back_hi))
                self.front_sag_mm, self.back_sag_mm = self._apply_coupled_sag_clipping(
                    front_seed, back_seed
                )

            self.step_count = 0
            self.was_clipped_front = 0.0
            self.was_clipped_back = 0.0

            ok, result, _timing = self._simulate_safe()
            if not ok:
                self.current_power = 0.0
                continue

            self.current_power = float(self._extract_power(result))
            if self.current_power <= float(self.config.max_initial_power):
                accepted = True
                break

        if not accepted and self.config.verbose_timing:
            print(
                f"Warning: Could not find initial state with power <= {self.config.max_initial_power} "
                f"after {self.config.max_reset_attempts} attempts. "
                f"Using last sampled state with power={self.current_power:.3f}",
                flush=True,
            )

        return self._get_obs(), {}

    def step(self, action):
        step_start = time.perf_counter()
        logic_start = time.perf_counter()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        old_front = float(self.front_sag_mm)
        old_back = float(self.back_sag_mm)

        proposed_front = old_front + float(action[0]) * float(self.config.action_scale_front_sag_mm)
        proposed_back = old_back + float(action[1]) * float(self.config.action_scale_back_sag_mm)

        new_front, new_back = self._apply_coupled_sag_clipping(proposed_front, proposed_back)

        self.was_clipped_front = 1.0 if abs(new_front - proposed_front) > 1e-12 else 0.0
        self.was_clipped_back = 1.0 if abs(new_back - proposed_back) > 1e-12 else 0.0

        self.front_sag_mm = new_front
        self.back_sag_mm = new_back

        delta_front = abs(new_front - old_front)
        delta_back = abs(new_back - old_back)

        ok, result, timing = self._simulate_safe()
        self.current_power = float(self._extract_power(result))

        reward = (
            self.current_power
            - float(self.config.boundary_penalty) * (self.was_clipped_front + self.was_clipped_back)
            - float(self.config.movement_penalty_coef) * float(delta_front + delta_back)
        )
        self.step_count += 1

        terminated = self.step_count >= int(self.config.max_steps)
        if not ok:
            terminated = True
        truncated = False

        logic_time_ms = (time.perf_counter() - logic_start) * 1000 - timing["total_ms"]
        step_time_ms = (time.perf_counter() - step_start) * 1000

        self.total_steps += 1
        self.cumulative_simulation_time_ms += timing["total_ms"]
        self.cumulative_env_logic_time_ms += logic_time_ms
        self.cumulative_step_time_ms += step_time_ms

        if (
            self.config.verbose_timing
            and self.total_steps % int(self.config.timing_log_interval) == 0
        ):
            avg_step = self.cumulative_step_time_ms / self.total_steps
            avg_sim = self.cumulative_simulation_time_ms / self.total_steps
            avg_logic = self.cumulative_env_logic_time_ms / self.total_steps
            print(
                f"[env] Step {self.total_steps}: avg={avg_step:.1f}ms/step "
                f"(sim: {avg_sim:.1f}ms, env logic: {avg_logic:.1f}ms) | "
                f"cumulative: {self.cumulative_step_time_ms/1000:.1f}s",
                flush=True,
            )

        info = {
            "simulation_ok": bool(ok),
            "episode_target_y": float(self.episode_target_y),
            "episode_origin_y": float(self.episode_origin_y),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    @staticmethod
    def _norm(val: float, lo: float, hi: float) -> float:
        return (val - lo) / (hi - lo)

    def _get_obs(self):
        front_lo, front_hi = self._effective_front_sag_bounds()
        back_lo, back_hi = self._effective_back_sag_bounds()

        front_norm = self._norm(self.front_sag_mm, front_lo, front_hi)
        back_norm = self._norm(self.back_sag_mm, back_lo, back_hi)
        target_norm = self._norm(
            self.episode_target_y,
            float(self.config.ray_target_y_min),
            float(self.config.ray_target_y_max),
        )
        origin_min = (
            float(self.config.ray_target_y_min)
            * float(self.config.mapping_numerator_mm)
            / float(self.config.object_depth_mm)
        )
        origin_max = (
            float(self.config.ray_target_y_max)
            * float(self.config.mapping_numerator_mm)
            / float(self.config.object_depth_mm)
        )
        origin_norm = self._norm(self.episode_origin_y, origin_min, origin_max)

        front_dist_lower = self._norm(self.front_sag_mm, front_lo, front_hi)
        front_dist_upper = self._norm(front_hi - self.front_sag_mm, 0.0, front_hi - front_lo)
        back_dist_lower = self._norm(self.back_sag_mm, back_lo, back_hi)
        back_dist_upper = self._norm(back_hi - self.back_sag_mm, 0.0, back_hi - back_lo)
        progress = float(self.step_count / max(int(self.config.max_steps), 1))

        return np.array(
            [
                float(np.clip(front_norm, 0.0, 1.0)),
                float(np.clip(back_norm, 0.0, 1.0)),
                float(np.clip(self.current_power, 0.0, 1.0)),
                float(np.clip(progress, 0.0, 1.0)),
                float(np.clip(target_norm, 0.0, 1.0)),
                float(np.clip(origin_norm, 0.0, 1.0)),
                float(np.clip(front_dist_lower, 0.0, 1.0)),
                float(np.clip(front_dist_upper, 0.0, 1.0)),
                float(np.clip(back_dist_lower, 0.0, 1.0)),
                float(np.clip(back_dist_upper, 0.0, 1.0)),
                float(self.was_clipped_front),
                float(self.was_clipped_back),
            ],
            dtype=np.float32,
        )

    def close(self):
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
        super().close()
