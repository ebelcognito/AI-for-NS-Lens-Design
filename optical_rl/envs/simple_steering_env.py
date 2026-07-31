from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import run_simulation


@dataclass
class SimpleSteeringConfig:
    source_x: float = 50
    source_y: float = 150
    detector_x: float = 350
    detector_y: float = 150
    detector_height: float = 20
    quad_size: float = 50
    quad_x_min: float = 100
    quad_x_max: float = 300
    action_scale: float = 10
    max_steps: int = 20
    ref_index: float = 1.5


class SimpleSteeringEnv(gym.Env):
    """
    Minimal 1D steering environment.
    Action: delta X in [-1, 1].
    Observation: [quad_x_norm, detector_power, progress].
    Reward: detector power.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[SimpleSteeringConfig, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = SimpleSteeringConfig()
        elif isinstance(config, dict):
            self.config = SimpleSteeringConfig(**config)
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

        self.quad_x = float(self.config.quad_x_min)
        self.step_count = 0
        self.current_power = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.quad_x = self.np_random.uniform(
            self.config.quad_x_min, self.config.quad_x_max
        )
        self.step_count = 0
        self.current_power = 0.0
        return self._get_obs(), {}

    def step(self, action):
        delta = float(action[0]) * self.config.action_scale
        self.quad_x = float(
            np.clip(self.quad_x + delta, self.config.quad_x_min, self.config.quad_x_max)
        )

        result, _ = run_simulation(self._build_scene())
        self.current_power = float(result["detectors"][0]["power"])

        reward = self.current_power
        self.step_count += 1
        terminated = self.step_count >= self.config.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        quad_x_norm = (self.quad_x - self.config.quad_x_min) / (
            self.config.quad_x_max - self.config.quad_x_min
        )
        progress = self.step_count / self.config.max_steps
        return np.array([quad_x_norm, self.current_power, progress], dtype=np.float32)

    def _build_scene(self):
        half = self.config.quad_size / 2
        detector_half = self.config.detector_height / 2
        quad_center_y = self.config.source_y
        return {
            "version": 5,
            "objs": [
                {
                    "type": "SingleRay",
                    "p1": {"x": self.config.source_x, "y": self.config.source_y},
                    "p2": {
                        "x": self.config.source_x + 50,
                        "y": self.config.source_y,
                    },
                },
                {
                    "type": "Glass",
                    "path": [
                        {
                            "x": self.quad_x - half,
                            "y": quad_center_y - half,
                            "arc": False,
                        },
                        {
                            "x": self.quad_x + half,
                            "y": quad_center_y - half,
                            "arc": False,
                        },
                        {
                            "x": self.quad_x + half,
                            "y": quad_center_y + half,
                            "arc": False,
                        },
                        {
                            "x": self.quad_x - half,
                            "y": quad_center_y + half,
                            "arc": False,
                        },
                    ],
                    "refIndex": self.config.ref_index,
                    "partialReflect": True,
                },
                {
                    "type": "Detector",
                    "p1": {
                        "x": self.config.detector_x,
                        "y": self.config.detector_y - detector_half,
                    },
                    "p2": {
                        "x": self.config.detector_x,
                        "y": self.config.detector_y + detector_half,
                    },
                },
            ],
        }
