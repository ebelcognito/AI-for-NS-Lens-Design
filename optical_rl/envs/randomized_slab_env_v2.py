from dataclasses import dataclass
from typing import Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_randomized_ray_with_slab_at_xy


@dataclass
class RandomizedSlab2DConfig:
    world_width: float = 300
    world_height: float = 100
    source_x: float = 0
    detector_x: float = 300
    source_y_min: float = 40
    source_y_max: float = 60
    detector_y_min: float = 30
    detector_y_max: float = 70
    slope_min: float = -50.0 / 300.0
    slope_max: float = 50.0 / 300.0
    detector_height: float = 5
    quad_size: float = 40
    ref_index: float = 1.5
    ray_dx: float = 30
    quad_x_min: float = 40
    quad_x_max: float = 260
    quad_y_min: float = 20
    quad_y_max: float = 80
    action_scale_x: float = 8
    action_scale_y: float = 5
    max_steps: int = 64
    verbose_timing: bool = False
    timing_log_interval: int = 500
    use_persistent_runner: bool = False


class RandomizedSlab2DEnv(gym.Env):
    """
    2D slab steering environment with per-episode randomized source, detector, and slope.
    Action: [delta_x, delta_y] in [-1, 1].
    Observation: [quad_x_norm, quad_y_norm, detector_power, progress].
    Reward: detector power.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[RandomizedSlab2DConfig, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = RandomizedSlab2DConfig()
        elif isinstance(config, dict):
            self.config = RandomizedSlab2DConfig(**config)
        else:
            self.config = config

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.quad_x = float(self.config.quad_x_min)
        self.quad_y = float(self.config.quad_y_min)
        self.step_count = 0
        self.current_power = 0.0

        self.episode_source_y = 0.0
        self.episode_detector_y = 0.0
        self.episode_slope = 0.0

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
        return scene_randomized_ray_with_slab_at_xy(
            source_x=self.config.source_x,
            source_y=self.episode_source_y,
            slope=self.episode_slope,
            detector_x=self.config.detector_x,
            detector_center_y=self.episode_detector_y,
            detector_half_height=self.config.detector_height / 2.0,
            quad_center_x=self.quad_x,
            quad_center_y=self.quad_y,
            quad_size=self.config.quad_size,
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
        self.episode_source_y = float(
            self.np_random.uniform(self.config.source_y_min, self.config.source_y_max)
        )
        self.episode_detector_y = float(
            self.np_random.uniform(self.config.detector_y_min, self.config.detector_y_max)
        )
        self.episode_slope = float(
            self.np_random.uniform(self.config.slope_min, self.config.slope_max)
        )
        self.quad_x = float(
            self.np_random.uniform(self.config.quad_x_min, self.config.quad_x_max)
        )
        self.quad_y = float(
            self.np_random.uniform(self.config.quad_y_min, self.config.quad_y_max)
        )
        self.step_count = 0

        result, _ = self._simulate()
        self.current_power = float(result["detectors"][0]["power"])
        return self._get_obs(), {}

    def step(self, action):
        step_start = time.perf_counter()
        logic_start = time.perf_counter()

        dx = float(action[0]) * self.config.action_scale_x
        dy = float(action[1]) * self.config.action_scale_y
        self.quad_x = float(
            np.clip(self.quad_x + dx, self.config.quad_x_min, self.config.quad_x_max)
        )
        self.quad_y = float(
            np.clip(self.quad_y + dy, self.config.quad_y_min, self.config.quad_y_max)
        )

        result, timing = self._simulate()
        self.current_power = float(result["detectors"][0]["power"])

        reward = self.current_power
        self.step_count += 1
        terminated = self.step_count >= self.config.max_steps
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

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        quad_x_norm = (self.quad_x - self.config.quad_x_min) / (
            self.config.quad_x_max - self.config.quad_x_min
        )
        quad_y_norm = (self.quad_y - self.config.quad_y_min) / (
            self.config.quad_y_max - self.config.quad_y_min
        )
        progress = self.step_count / self.config.max_steps
        return np.array(
            [quad_x_norm, quad_y_norm, self.current_power, progress], dtype=np.float32
        )

    def close(self):
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
        super().close()
