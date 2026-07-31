from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optical_rl.utils.ray_optics_bridge import create_runner, run_simulation
from optical_rl.utils.scene_builder import scene_ray_bundle_with_custom_glass


Point = Tuple[float, float]


@dataclass
class QuadCurveGlassConfigV3:
    world_width: float = 300.0
    world_height: float = 300.0

    source_x: float = 10.0
    source_y: float = 150.0

    detector_x: float = 290.0
    detector_center_y: float = 150.0
    detector_y_min: float = 100.0
    detector_y_max: float = 200.0
    detector_height: float = 5.0

    ray_mode: str = "explicit"  # explicit|bundle
    ray_count: int = 200
    ray_bundle_target_x: float = 290.0
    ray_bundle_y_min: float = 120.0
    ray_bundle_y_max: float = 180.0
    ray_bundle_sampling: str = "deterministic"
    ray_bundle_angle_min_target_x: float = 290.0
    ray_bundle_angle_min_target_y: float = 130.0
    ray_bundle_angle_max_target_x: float = 290.0
    ray_bundle_angle_max_target_y: float = 170.0

    ray1_target_x: float = 290.0
    ray1_target_y: float = 200.0
    ray2_target_x: float = 290.0
    ray2_target_y: float = 100.0
    ray3_target_x: float = 290.0
    ray3_target_y: float = 250.0
    ray4_target_x: float = 290.0
    ray4_target_y: float = 50.0

    ref_index: float = 1.5

    action_scale_xy: float = 5.0
    action_scale_sag: float = 0.05
    max_steps: int = 200
    max_initial_power: float = 0.0
    max_reset_attempts: int = 100
    reset_jitter_xy: float = 20.0
    reset_jitter_sag: float = 0.1
    center_x: float = 150.0
    center_y: float = 150.0
    quadrant_margin: float = 5.0
    right_side_x_max: float = 270.0
    sag_min: float = -0.35
    sag_max: float = 0.35

    # Optional scaling knobs. If left as None, bundle mode defaults to per-ray normalization.
    power_norm_divisor: Optional[float] = None
    reward_scale: Optional[float] = None

    verbose_timing: bool = False
    timing_log_interval: int = 5000
    use_persistent_runner: bool = False
    timeout_s: float = 5.0


class QuadCurveGlassEnvV3(gym.Env):
    """
    2D environment with fixed source/detector and two rays, controlling a single Glass
    object defined by 8 points (4 corners + 4 arc control points).

    Action: 12D deltas in [-1,1]:
      - 8 for corner point (dx, dy) updates
      - 4 for per-edge sag updates (top/bottom sag actions are ignored)
    Observation: 18D in [0,1]:
      - 16 normalized point coords (x/world_width, y/world_height)
      - normalized power: clip(detector_power/4, 0, 1)
      - progress: step/max_steps
    Reward: raw detector power (may exceed 1 due to multiple rays).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Union[QuadCurveGlassConfigV3, dict, None] = None):
        super().__init__()
        if config is None:
            self.config = QuadCurveGlassConfigV3()
        elif isinstance(config, dict):
            self.config = QuadCurveGlassConfigV3(**config)
        else:
            self.config = config

        self.ray_targets: List[Point] = self._initial_ray_targets(self.config)
        self._power_norm_divisor = self._resolve_power_norm_divisor(
            self.config, self.ray_targets
        )
        self._reward_scale = self._resolve_reward_scale(self.config, self.ray_targets)

        self.action_space = spaces.Box(
            low=np.full((12,), -1.0, dtype=np.float32),
            high=np.full((12,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros((18,), dtype=np.float32),
            high=np.ones((18,), dtype=np.float32),
            dtype=np.float32,
        )

        self.corners: List[Point] = []
        self.sags: List[float] = []
        self.points: List[Point] = []
        self.detector_center_y = float(self.config.detector_center_y)
        self.step_count = 0
        self.current_power = 0.0

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

    @staticmethod
    def _resolve_ray_mode(cfg: QuadCurveGlassConfigV3) -> str:
        ray_mode = str(getattr(cfg, "ray_mode", "explicit")).lower().strip()
        if ray_mode not in ("explicit", "bundle"):
            raise ValueError(f"Unsupported ray_mode: {ray_mode}")
        return ray_mode

    @staticmethod
    def _resolve_bundle_sampling(cfg: QuadCurveGlassConfigV3) -> str:
        sampling = str(getattr(cfg, "ray_bundle_sampling", "deterministic")).lower().strip()
        if sampling not in ("deterministic", "random_uniform_angle"):
            raise ValueError(f"Unsupported ray_bundle_sampling: {sampling}")
        return sampling

    @staticmethod
    def _validate_ray_count(cfg: QuadCurveGlassConfigV3) -> int:
        ray_count = int(cfg.ray_count)
        if ray_count <= 0:
            raise ValueError(f"ray_count must be > 0, got {ray_count}")
        return ray_count

    @staticmethod
    def _build_explicit_ray_targets(cfg: QuadCurveGlassConfigV3) -> List[Point]:
        return [
            (float(cfg.ray1_target_x), float(cfg.ray1_target_y)),
            (float(cfg.ray2_target_x), float(cfg.ray2_target_y)),
            (float(cfg.ray3_target_x), float(cfg.ray3_target_y)),
            (float(cfg.ray4_target_x), float(cfg.ray4_target_y)),
        ]

    @classmethod
    def _build_deterministic_bundle_targets(cls, cfg: QuadCurveGlassConfigV3) -> List[Point]:
        ray_count = cls._validate_ray_count(cfg)
        ys = np.linspace(
            float(cfg.ray_bundle_y_min),
            float(cfg.ray_bundle_y_max),
            num=ray_count,
            dtype=np.float64,
        )
        target_x = float(cfg.ray_bundle_target_x)
        return [(target_x, float(y)) for y in ys]

    @classmethod
    def _initial_ray_targets(cls, cfg: QuadCurveGlassConfigV3) -> List[Point]:
        ray_mode = cls._resolve_ray_mode(cfg)
        if ray_mode == "explicit":
            return cls._build_explicit_ray_targets(cfg)

        sampling = cls._resolve_bundle_sampling(cfg)
        if sampling == "deterministic":
            return cls._build_deterministic_bundle_targets(cfg)

        cls._validate_ray_count(cfg)
        return []

    @staticmethod
    def _compute_angle_bounds(cfg: QuadCurveGlassConfigV3) -> Tuple[float, float]:
        source_x = float(cfg.source_x)
        source_y = float(cfg.source_y)
        min_angle = float(
            np.arctan2(
                float(cfg.ray_bundle_angle_min_target_y) - source_y,
                float(cfg.ray_bundle_angle_min_target_x) - source_x,
            )
        )
        max_angle = float(
            np.arctan2(
                float(cfg.ray_bundle_angle_max_target_y) - source_y,
                float(cfg.ray_bundle_angle_max_target_x) - source_x,
            )
        )
        return tuple(sorted((min_angle, max_angle)))

    @classmethod
    def _validate_random_uniform_angle_bundle(cls, cfg: QuadCurveGlassConfigV3) -> None:
        target_x = float(cfg.ray_bundle_target_x)
        source_x = float(cfg.source_x)
        if target_x <= source_x:
            raise ValueError(
                f"ray_bundle_target_x must be > source_x for angle sampling, got "
                f"{target_x} <= {source_x}"
            )
        for bound_x in (
            float(cfg.ray_bundle_angle_min_target_x),
            float(cfg.ray_bundle_angle_max_target_x),
        ):
            if bound_x <= source_x:
                raise ValueError(
                    f"angle boundary target x must be > source_x for angle sampling, got "
                    f"{bound_x} <= {source_x}"
                )
        cls._validate_ray_count(cfg)

    def _sample_random_uniform_angle_bundle(self) -> List[Point]:
        self._validate_random_uniform_angle_bundle(self.config)
        angle_min, angle_max = self._compute_angle_bounds(self.config)
        ray_count = self._validate_ray_count(self.config)
        target_x = float(self.config.ray_bundle_target_x)
        source_x = float(self.config.source_x)
        source_y = float(self.config.source_y)
        dx = target_x - source_x
        angles = self.np_random.uniform(angle_min, angle_max, size=ray_count)
        return [(target_x, float(source_y + dx * np.tan(angle))) for angle in angles]

    def _refresh_ray_targets_for_episode(self) -> None:
        ray_mode = self._resolve_ray_mode(self.config)
        if ray_mode == "explicit":
            self.ray_targets = self._build_explicit_ray_targets(self.config)
            return

        sampling = self._resolve_bundle_sampling(self.config)
        if sampling == "deterministic":
            self.ray_targets = self._build_deterministic_bundle_targets(self.config)
            return

        self.ray_targets = self._sample_random_uniform_angle_bundle()

    @staticmethod
    def _resolve_power_norm_divisor(
        cfg: QuadCurveGlassConfigV3, ray_targets: List[Point]
    ) -> float:
        if cfg.power_norm_divisor is not None:
            return float(cfg.power_norm_divisor)
        ray_mode = QuadCurveGlassEnvV3._resolve_ray_mode(cfg)
        if ray_mode == "bundle":
            return float(max(QuadCurveGlassEnvV3._validate_ray_count(cfg), 1))
        return 4.0

    @staticmethod
    def _resolve_reward_scale(
        cfg: QuadCurveGlassConfigV3, ray_targets: List[Point]
    ) -> float:
        if cfg.reward_scale is not None:
            return float(cfg.reward_scale)
        ray_mode = QuadCurveGlassEnvV3._resolve_ray_mode(cfg)
        if ray_mode == "bundle":
            return 1.0 / float(max(QuadCurveGlassEnvV3._validate_ray_count(cfg), 1))
        return 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._refresh_ray_targets_for_episode()
        self.current_power = 0.0
        accepted = False
        for attempt in range(int(self.config.max_reset_attempts)):
            self.corners, self.sags = self._initial_state_for_attempt(attempt)
            self.detector_center_y = float(
                self.np_random.uniform(
                    self.config.detector_y_min,
                    self.config.detector_y_max,
                )
            )
            self.points = self._build_path_points(self.corners, self.sags)
            self.step_count = 0

            ok, result, _timing = self._simulate_safe()
            if not ok:
                self.current_power = 0.0
                continue

            self.current_power = float(self._extract_power(result))
            if self.current_power <= float(self.config.max_initial_power):
                accepted = True
                break

        if not accepted:
            if self.config.verbose_timing:
                print(
                    f"Warning: Could not find initial state with power <= "
                    f"{self.config.max_initial_power} after {self.config.max_reset_attempts} attempts. "
                    f"Using last sampled state with power={self.current_power:.3f}",
                    flush=True,
                )
        return self._get_obs(), {}

    def step(self, action):
        step_start = time.perf_counter()
        logic_start = time.perf_counter()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (12,):
            raise ValueError(f"Expected action shape (12,), got {action.shape}")

        updated_corners: List[Point] = []
        for i in range(4):
            dx = float(action[2 * i]) * self.config.action_scale_xy
            dy = float(action[2 * i + 1]) * self.config.action_scale_xy
            x, y = self.corners[i]
            x, y = self._clip_corner_to_quadrant(i, x + dx, y + dy)
            updated_corners.append((x, y))
        self.corners = updated_corners

        updated_sags: List[float] = []
        for i in range(4):
            dsag = float(action[8 + i]) * self.config.action_scale_sag
            sag = float(
                np.clip(
                    self.sags[i] + dsag,
                    self.config.sag_min,
                    self.config.sag_max,
                )
            )
            updated_sags.append(sag)
        updated_sags[0] = 0.0
        updated_sags[2] = 0.0
        self.sags = updated_sags
        self.points = self._build_path_points(self.corners, self.sags)

        ok, result, timing = self._simulate_safe()
        self.current_power = float(self._extract_power(result))

        reward = float(self.current_power) * float(self._reward_scale)
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

        info = {"simulation_ok": bool(ok)}
        return self._get_obs(), reward, terminated, truncated, info

    def _default_corners(self) -> List[Point]:
        return [
            (130.0, 130.0),  # TL
            (170.0, 130.0),  # TR
            (170.0, 170.0),  # BR
            (130.0, 170.0),  # BL
        ]

    def _default_sags(self) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]

    def _quadrant_bounds(self, corner_idx: int) -> Tuple[float, float, float, float]:
        m = float(self.config.quadrant_margin)
        cx = float(self.config.center_x)
        cy = float(self.config.center_y)
        w = float(self.config.world_width)
        h = float(self.config.world_height)
        if corner_idx == 0:  # TL
            return m, cx - m, m, cy - m
        if corner_idx == 1:  # TR
            return cx + m, min(w - m, float(self.config.right_side_x_max)), m, cy - m
        if corner_idx == 2:  # BR
            return cx + m, min(w - m, float(self.config.right_side_x_max)), cy + m, h - m
        if corner_idx == 3:  # BL
            return m, cx - m, cy + m, h - m
        raise ValueError(f"Invalid corner index: {corner_idx}")

    def _clip_corner_to_quadrant(self, corner_idx: int, x: float, y: float) -> Point:
        xmin, xmax, ymin, ymax = self._quadrant_bounds(corner_idx)
        return (
            float(np.clip(x, xmin, xmax)),
            float(np.clip(y, ymin, ymax)),
        )

    def _build_path_points(
        self,
        corners: List[Point],
        sags: List[float],
    ) -> List[Point]:
        if len(corners) != 4 or len(sags) != 4:
            raise ValueError("Expected 4 corners and 4 sags")

        center = np.array([self.config.center_x, self.config.center_y], dtype=np.float64)
        path: List[Point] = []

        for edge_idx in range(4):
            p0 = np.array(corners[edge_idx], dtype=np.float64)
            p1 = np.array(corners[(edge_idx + 1) % 4], dtype=np.float64)
            edge = p1 - p0
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-6:
                edge_len = 1e-6

            midpoint = (p0 + p1) / 2.0
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            normal_len = float(np.linalg.norm(normal))
            if normal_len < 1e-6:
                normal = np.array([0.0, 0.0], dtype=np.float64)
            else:
                normal /= normal_len

            to_center = center - midpoint
            if float(np.dot(normal, to_center)) < 0:
                normal = -normal

            if edge_idx in (0, 2):
                sag = 0.0
            else:
                sag = float(np.clip(sags[edge_idx], self.config.sag_min, self.config.sag_max))
            ctrl = midpoint + normal * (sag * edge_len)

            ctrl_x = float(np.clip(ctrl[0], 0.0, self.config.world_width))
            ctrl_y = float(np.clip(ctrl[1], 0.0, self.config.world_height))

            path.append((float(p0[0]), float(p0[1])))
            path.append((ctrl_x, ctrl_y))

        return path

    def _initial_state_for_attempt(self, attempt: int) -> Tuple[List[Point], List[float]]:
        corners = self._default_corners()
        sags = self._default_sags()
        if attempt == 0:
            return corners, sags

        xy_jitter = float(self.config.reset_jitter_xy)
        sag_jitter = float(self.config.reset_jitter_sag)
        if xy_jitter <= 0 and sag_jitter <= 0:
            return corners, sags

        jittered_corners: List[Point] = []
        for corner_idx, (x, y) in enumerate(corners):
            jx = (
                float(self.np_random.uniform(-xy_jitter, xy_jitter))
                if xy_jitter > 0
                else 0.0
            )
            jy = (
                float(self.np_random.uniform(-xy_jitter, xy_jitter))
                if xy_jitter > 0
                else 0.0
            )
            jittered_corners.append(
                self._clip_corner_to_quadrant(corner_idx, x + jx, y + jy)
            )

        jittered_sags: List[float] = []
        for sag in sags:
            js = (
                float(self.np_random.uniform(-sag_jitter, sag_jitter))
                if sag_jitter > 0
                else 0.0
            )
            jittered_sags.append(
                float(np.clip(sag + js, self.config.sag_min, self.config.sag_max))
            )

        return jittered_corners, jittered_sags

    def _build_scene(self):
        return scene_ray_bundle_with_custom_glass(
            source_x=float(self.config.source_x),
            source_y=float(self.config.source_y),
            ray_targets=self.ray_targets,
            detector_x=float(self.config.detector_x),
            detector_center_y=float(self.detector_center_y),
            detector_half_height=float(self.config.detector_height) / 2.0,
            points=self.points,
            ref_index=float(self.config.ref_index),
        )

    def _simulate(self):
        scene = self._build_scene()
        if self._runner is not None:
            return self._runner.simulate(scene, episode=0, step=self.step_count)
        return run_simulation(
            scene, timeout_s=float(self.config.timeout_s), verbose=self.config.verbose_timing
        )

    def _simulate_safe(self):
        try:
            result, timing = self._simulate()
            return True, result, timing
        except Exception as exc:
            if self.config.verbose_timing:
                print(f"[QuadCurveGlassEnvV3] Simulation failed: {exc}", flush=True)
            return False, {"detectors": [], "error": str(exc)}, {"total_ms": 0.0}

    def _extract_power(self, result) -> float:
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

    def _get_obs(self):
        coords = []
        for x, y in self.points:
            coords.append(float(np.clip(x / self.config.world_width, 0.0, 1.0)))
            coords.append(float(np.clip(y / self.config.world_height, 0.0, 1.0)))

        denom = float(max(self._power_norm_divisor, 1e-9))
        power_norm = float(np.clip(self.current_power / denom, 0.0, 1.0))
        progress = float(self.step_count / max(int(self.config.max_steps), 1))
        return np.array(coords + [power_norm, progress], dtype=np.float32)

    def close(self):
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
        super().close()
