import json
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.utils.ray_optics_bridge import run_simulation
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner
from optical_rl.utils.scene_builder import scene_direct


def test_single_call_matches_oneshot():
    scene = scene_direct()
    result_oneshot, _ = run_simulation(scene)

    with PersistentRayOpticsRunner() as runner:
        result_persistent, _ = runner.simulate(scene)

    det_one = result_oneshot["detectors"][0]
    det_persist = result_persistent["detectors"][0]
    assert abs(det_one["power"] - det_persist["power"]) < 1e-9
    assert abs(det_one["normal"] - det_persist["normal"]) < 1e-9
    assert abs(det_one["shear"] - det_persist["shear"]) < 1e-9


def test_multiple_calls():
    scene = scene_direct()
    with PersistentRayOpticsRunner() as runner:
        for step in range(20):
            result, timing = runner.simulate(scene, episode=1, step=step)
            assert "detectors" in result
            assert timing["total_ms"] > 0


def test_error_recovery_after_invalid_json_line():
    with PersistentRayOpticsRunner() as runner:
        assert runner._process is not None
        assert runner._process.stdin is not None

        runner._process.stdin.write("{not-valid-json}\n")
        runner._process.stdin.flush()
        raw_response = runner._read_response_line()
        response = json.loads(raw_response)
        assert response.get("error") == "invalid_json"

        result, _ = runner.simulate(scene_direct(), episode=2, step=1)
        assert result["detectors"]


def test_graceful_shutdown():
    runner = PersistentRayOpticsRunner()
    runner.start()
    assert runner._process is not None
    pid = runner._process.pid
    runner.simulate(scene_direct())
    runner.stop()
    assert runner._process is None
    assert pid is not None
