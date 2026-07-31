import os
import sys
import time
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.utils.ray_optics_bridge import run_simulation
from optical_rl.utils.scene_builder import (
    scene_direct,
    scene_with_glass,
    scene_with_quad_at_x,
)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def test_basic_scene():
    max_sim_ms = _env_float("RL_MAX_SIM_MS", 600.0)
    result, elapsed_s = run_simulation(scene_direct())
    power = result["detectors"][0]["power"]
    assert 0 <= power <= 1
    assert power > 0
    assert (elapsed_s * 1000.0) <= max_sim_ms


def test_refraction_sanity():
    min_delta = _env_float("RL_MIN_POWER_DELTA", 0.05)
    result_direct, _ = run_simulation(scene_direct())
    result_glass, _ = run_simulation(scene_with_glass())
    power_direct = result_direct["detectors"][0]["power"]
    power_glass = result_glass["detectors"][0]["power"]
    assert abs(power_direct - power_glass) >= min_delta


def test_position_control():
    epsilon = _env_float("RL_POWER_EPS", 1e-6)
    powers = []
    # Use positions that change ray path: blocking (50), in-path (150), beyond detector (400)
    for quad_x in [400, 140, 200, 260]:
        result, _ = run_simulation(scene_with_quad_at_x(quad_x))
        powers.append(result["detectors"][0]["power"])

    distinct = []
    for p in powers:
        if all(abs(p - d) > epsilon for d in distinct):
            distinct.append(p)

    assert len(distinct) >= 2
