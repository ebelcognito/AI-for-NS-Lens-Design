from dataclasses import dataclass
from typing import Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_randomized_ray_with_sag_slab_at_y


@dataclass
class RandomizedSagSlabConfig:
    world_width: float = 300
    world_height: float = 100
    source_x: float = 0
    detector_x: float = 300
    quad_center_x: float = 150
    source_y_min: float = 40
    source_y_max: float = 60
    detector_y_min: float = 30
    detector_y_max: float = 70
    slope_min: float = -50.0 / 300.0
    slope_max: float = 50.0 / 300.0
    sag_depth_frac_min: float = -0.5
    sag_depth_frac_max: float = 0.5
    detector_height: float = 5
    quad_size: float = 40
    ref_index: float = 1.5
    ray_dx: float = 30
    quad_y_min: float = 20
    quad_y_max: float = 80
    action_scale_y: float = 5
    action_scale_sag: float = 0.01
    max_steps: int = 64
    max_initial_power: float = 1.0  # Filter initial conditions: reject if power > this
    max_reset_attempts: int = 100  # Max resamples to avoid infinite loop
    boundary_penalty: float = 0.01
    early_term_enabled: bool = True
    no_improve_patience: int = 120
    clip_streak_patience: int = 60
    improve_epsilon: float = 1e-4
    good_enough_power_threshold: float = 0.90
    good_power_patience: int = 20
    verbose_timing: bool = False
    timing_log_interval: int = 500
    use_persistent_runner: bool = False


class RandomizedSagSlabEnvV3(gym.Env):
    """
    2D control slab steering with per-episode randomized source/detector/slope/sag.
    Action: [delta_y, delta_sag] in [-1, 1]^2.
    Observation:
      [quad_y_norm, detector_power, progress, source_y_norm, detector_y_norm,
       slope_norm, sag_depth_norm, dist_y_lower_norm, dist_y_upper_norm,
       dist_sag_lower_norm, dist_sag_upper_norm, was_clipped_y, was_clipped_sag].
    Reward: detector power.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[RandomizedSagSlabConfig, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = RandomizedSagSlabConfig()
        elif isinstance(config, dict):
            self.config = RandomizedSagSlabConfig(**config)
        else:
            self.config = config

        if self.config.sag_depth_frac_min >= self.config.sag_depth_frac_max:
            raise ValueError("sag_depth_frac_min must be < sag_depth_frac_max")

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 13, dtype=np.float32),
            high=np.array([1.0] * 13, dtype=np.float32),
            dtype=np.float32,
        )

        self.quad_y = float(self.config.quad_y_min)
        self.step_count = 0
        self.current_power = 0.0

        self.episode_source_y = 0.0
        self.episode_detector_y = 0.0
        self.episode_slope = 0.0
        self.episode_sag_depth_frac = 0.0
        self.best_power_so_far = 0.0
        self.steps_since_improve = 0
        self.clip_streak_y = 0
        self.clip_streak_sag = 0
        self.last_clip_dir_y = 0
        self.last_clip_dir_sag = 0
        self.was_clipped_y = 0.0
        self.was_clipped_sag = 0.0
        self.high_power_streak = 0
        self.high_power_confirmed = False

        self._runner = None
        self.total_steps = 0
        self.cumulative_simulation_time_ms = 0.0
        self.cumulative_env_logic_time_ms = 0.0
        self.cumulative_step_time_ms = 0.0

        if self.config.use_persistent_runner:
            self._runner = create_runner(
                persistent=True,
                timeout_s=5.0,
                verbose=self.config.verbose_timing,
            )
            self._runner.start()

    def _build_scene(self):
        return scene_randomized_ray_with_sag_slab_at_y(
            source_x=self.config.source_x,
            source_y=self.episode_source_y,
            slope=self.episode_slope,
            detector_x=self.config.detector_x,
            detector_center_y=self.episode_detector_y,
            detector_half_height=self.config.detector_height / 2.0,
            quad_center_x=self.config.quad_center_x,
            quad_center_y=self.quad_y,
            quad_size=self.config.quad_size,
            sag_depth_frac=self.episode_sag_depth_frac,
            ref_index=self.config.ref_index,
            ray_dx=self.config.ray_dx,
        )

    def _simulate(self):
        scene = self._build_scene()
        if self._runner is not None:
            return self._runner.simulate(scene, episode=0, step=self.step_count)
        return run_simulation(scene, verbose=self.config.verbose_timing)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Resample initial conditions until we get acceptable initial power
        for attempt in range(self.config.max_reset_attempts):
            self.episode_source_y = float(
                self.np_random.uniform(self.config.source_y_min, self.config.source_y_max)
            )
            self.episode_detector_y = float(
                self.np_random.uniform(self.config.detector_y_min, self.config.detector_y_max)
            )
            self.episode_slope = float(
                self.np_random.uniform(self.config.slope_min, self.config.slope_max)
            )
            self.episode_sag_depth_frac = float(
                self.np_random.uniform(
                    self.config.sag_depth_frac_min, self.config.sag_depth_frac_max
                )
            )
            self.quad_y = float(
                self.np_random.uniform(self.config.quad_y_min, self.config.quad_y_max)
            )
            self.step_count = 0

            result, _ = self._simulate()
            self.current_power = float(result["detectors"][0]["power"])

            # Accept if initial power is below threshold (or threshold disabled with 1.0)
            if self.current_power <= self.config.max_initial_power:
                break

            # If we exhausted attempts, warn and accept anyway
            if attempt == self.config.max_reset_attempts - 1:
                print(f"Warning: Could not find initial conditions with power <= {self.config.max_initial_power} "
                      f"after {self.config.max_reset_attempts} attempts. Accepting power={self.current_power:.3f}")

        self.best_power_so_far = self.current_power
        self.steps_since_improve = 0
        self.clip_streak_y = 0
        self.clip_streak_sag = 0
        self.last_clip_dir_y = 0
        self.last_clip_dir_sag = 0
        self.was_clipped_y = 0.0
        self.was_clipped_sag = 0.0
        self.high_power_streak = 0
        self.high_power_confirmed = False

        return self._get_obs(), {}

    def step(self, action):
        step_start = time.perf_counter()
        logic_start = time.perf_counter()

        proposed_y = self.quad_y + float(action[0]) * self.config.action_scale_y
        new_y = float(np.clip(proposed_y, self.config.quad_y_min, self.config.quad_y_max))
        self.was_clipped_y = 1.0 if abs(new_y - proposed_y) > 1e-12 else 0.0
        clip_dir_y = 0
        if self.was_clipped_y > 0:
            clip_dir_y = -1 if proposed_y < self.config.quad_y_min else 1
        if clip_dir_y != 0 and clip_dir_y == self.last_clip_dir_y:
            self.clip_streak_y += 1
        elif clip_dir_y != 0:
            self.clip_streak_y = 1
        else:
            self.clip_streak_y = 0
        self.last_clip_dir_y = clip_dir_y
        self.quad_y = new_y

        proposed_sag = (
            self.episode_sag_depth_frac + float(action[1]) * self.config.action_scale_sag
        )
        new_sag = float(
            np.clip(
                proposed_sag,
                self.config.sag_depth_frac_min,
                self.config.sag_depth_frac_max,
            )
        )
        self.was_clipped_sag = 1.0 if abs(new_sag - proposed_sag) > 1e-12 else 0.0
        clip_dir_sag = 0
        if self.was_clipped_sag > 0:
            clip_dir_sag = -1 if proposed_sag < self.config.sag_depth_frac_min else 1
        if clip_dir_sag != 0 and clip_dir_sag == self.last_clip_dir_sag:
            self.clip_streak_sag += 1
        elif clip_dir_sag != 0:
            self.clip_streak_sag = 1
        else:
            self.clip_streak_sag = 0
        self.last_clip_dir_sag = clip_dir_sag
        self.episode_sag_depth_frac = new_sag

        result, timing = self._simulate()
        self.current_power = float(result["detectors"][0]["power"])

        if self.current_power >= self.config.good_enough_power_threshold:
            self.high_power_streak += 1
        else:
            self.high_power_streak = 0
        self.high_power_confirmed = self.high_power_streak >= self.config.good_power_patience

        reward = self.current_power - self.config.boundary_penalty * (
            self.was_clipped_y + self.was_clipped_sag
        )
        if self.current_power > self.best_power_so_far + self.config.improve_epsilon:
            self.best_power_so_far = self.current_power
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1

        self.step_count += 1
        terminated = self.step_count >= self.config.max_steps
        if (
            self.config.early_term_enabled
            and not terminated
            and not self.high_power_confirmed
        ):
            terminated = (
                self.steps_since_improve >= self.config.no_improve_patience
                or self.clip_streak_y >= self.config.clip_streak_patience
                or self.clip_streak_sag >= self.config.clip_streak_patience
            )
        truncated = False

        logic_time_ms = (time.perf_counter() - logic_start) * 1000 - timing["total_ms"]
        step_time_ms = (time.perf_counter() - step_start) * 1000

        self.total_steps += 1
        self.cumulative_simulation_time_ms += timing["total_ms"]
        self.cumulative_env_logic_time_ms += logic_time_ms
        self.cumulative_step_time_ms += step_time_ms

        if (
            self.config.verbose_timing
            and self.total_steps % self.config.timing_log_interval == 0
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
            "high_power_streak": self.high_power_streak,
            "high_power_confirmed": self.high_power_confirmed,
            "steps_since_improve": self.steps_since_improve,
            "clip_streak_y": self.clip_streak_y,
            "clip_streak_sag": self.clip_streak_sag,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _norm(self, val: float, lo: float, hi: float) -> float:
        return (val - lo) / (hi - lo)

    def _get_obs(self):
        quad_y_norm = self._norm(self.quad_y, self.config.quad_y_min, self.config.quad_y_max)
        source_y_norm = self._norm(
            self.episode_source_y, self.config.source_y_min, self.config.source_y_max
        )
        detector_y_norm = self._norm(
            self.episode_detector_y,
            self.config.detector_y_min,
            self.config.detector_y_max,
        )
        slope_norm = self._norm(self.episode_slope, self.config.slope_min, self.config.slope_max)
        sag_depth_norm = self._norm(
            self.episode_sag_depth_frac,
            self.config.sag_depth_frac_min,
            self.config.sag_depth_frac_max,
        )
        dist_y_lower_norm = self._norm(
            self.quad_y, self.config.quad_y_min, self.config.quad_y_max
        )
        dist_y_upper_norm = self._norm(
            self.config.quad_y_max - self.quad_y,
            0.0,
            self.config.quad_y_max - self.config.quad_y_min,
        )
        dist_sag_lower_norm = self._norm(
            self.episode_sag_depth_frac,
            self.config.sag_depth_frac_min,
            self.config.sag_depth_frac_max,
        )
        dist_sag_upper_norm = self._norm(
            self.config.sag_depth_frac_max - self.episode_sag_depth_frac,
            0.0,
            self.config.sag_depth_frac_max - self.config.sag_depth_frac_min,
        )
        progress = self.step_count / self.config.max_steps

        return np.array(
            [
                quad_y_norm,
                self.current_power,
                progress,
                source_y_norm,
                detector_y_norm,
                slope_norm,
                sag_depth_norm,
                dist_y_lower_norm,
                dist_y_upper_norm,
                dist_sag_lower_norm,
                dist_sag_upper_norm,
                self.was_clipped_y,
                self.was_clipped_sag,
            ],
            dtype=np.float32,
        )

    def close(self):
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
        super().close()
