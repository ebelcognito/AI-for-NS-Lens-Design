from dataclasses import dataclass
from typing import Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_angled_ray_with_quad_at_y


@dataclass
class AngledRayConfig:
    quad_y_min: float = 0
    quad_y_max: float = 60
    action_scale: float = 5
    max_steps: int = 20
    quad_center_x: float = 50
    quad_size: float = 40
    detector_x: float = 100
    detector_center_y: float = 70
    detector_half_height: float = 30
    ref_index: float = 1.5
    # Timing/debug options
    verbose_timing: bool = False
    timing_log_interval: int = 500
    use_persistent_runner: bool = False


class AngledRayEnv(gym.Env):
    """
    1D steering environment using an angled ray and a movable quad in Y.
    Action: delta Y in [-1, 1].
    Observation: [quad_y_norm, detector_power, progress].
    Reward: detector power.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[AngledRayConfig, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = AngledRayConfig()
        elif isinstance(config, dict):
            self.config = AngledRayConfig(**config)
        else:
            self.config = config

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.quad_y = float(self.config.quad_y_min)
        self.step_count = 0
        self.current_power = 0.0
        self._runner = None
        
        # Timing tracking
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.quad_y = self.np_random.uniform(
            self.config.quad_y_min, self.config.quad_y_max
        )
        self.step_count = 0
        self.current_power = 0.0
        return self._get_obs(), {}

    def step(self, action):
        step_start = time.perf_counter()
        
        # Env logic (action processing, scene building)
        logic_start = time.perf_counter()
        delta = float(action[0]) * self.config.action_scale
        self.quad_y = float(
            np.clip(self.quad_y + delta, self.config.quad_y_min, self.config.quad_y_max)
        )

        scene = scene_angled_ray_with_quad_at_y(
            quad_center_y=self.quad_y,
            quad_center_x=self.config.quad_center_x,
            quad_size=self.config.quad_size,
            detector_x=self.config.detector_x,
            detector_center_y=self.config.detector_center_y,
            detector_half_height=self.config.detector_half_height,
            ref_index=self.config.ref_index,
        )
        
        # Simulation call (the expensive part)
        if self._runner is not None:
            result, timing = self._runner.simulate(
                scene,
                episode=0,
                step=self.step_count,
            )
        else:
            result, timing = run_simulation(scene, verbose=self.config.verbose_timing)
        self.current_power = float(result["detectors"][0]["power"])
        
        # More env logic (reward computation)
        reward = self.current_power
        self.step_count += 1
        terminated = self.step_count >= self.config.max_steps
        truncated = False
        
        logic_time_ms = (time.perf_counter() - logic_start) * 1000 - timing['total_ms']
        step_time_ms = (time.perf_counter() - step_start) * 1000
        
        # Update cumulative timing
        self.total_steps += 1
        self.cumulative_simulation_time_ms += timing['total_ms']
        self.cumulative_env_logic_time_ms += logic_time_ms
        self.cumulative_step_time_ms += step_time_ms
        
        # Print summary at intervals
        if self.config.verbose_timing and self.total_steps % self.config.timing_log_interval == 0:
            avg_step = self.cumulative_step_time_ms / self.total_steps
            avg_sim = self.cumulative_simulation_time_ms / self.total_steps
            avg_logic = self.cumulative_env_logic_time_ms / self.total_steps
            print(f"[env] Step {self.total_steps}: avg={avg_step:.1f}ms/step "
                  f"(sim: {avg_sim:.1f}ms, env logic: {avg_logic:.1f}ms) | "
                  f"cumulative: {self.cumulative_step_time_ms/1000:.1f}s", flush=True)

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        quad_y_norm = (self.quad_y - self.config.quad_y_min) / (
            self.config.quad_y_max - self.config.quad_y_min
        )
        progress = self.step_count / self.config.max_steps
        return np.array([quad_y_norm, self.current_power, progress], dtype=np.float32)

    def close(self):
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
        super().close()
