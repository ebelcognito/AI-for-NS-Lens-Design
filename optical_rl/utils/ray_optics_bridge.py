import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_TIMEOUT_S = 5


class OneShotRunner:
    """Compatibility wrapper matching the persistent runner interface."""

    def __init__(self, timeout_s=DEFAULT_TIMEOUT_S, verbose=False):
        self.timeout_s = timeout_s
        self.verbose = verbose

    def start(self):
        return None

    def stop(self):
        return None

    def restart(self):
        return None

    def simulate(self, scene_dict, episode=0, step=0):
        return run_simulation(
            scene_dict,
            timeout_s=self.timeout_s,
            verbose=self.verbose,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _runner_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "ray-optics" / "integrations" / "runner.js"


def run_simulation(scene_dict, timeout_s=DEFAULT_TIMEOUT_S, verbose=False):
    """Send scene to ray-optics Node runner and return parsed JSON with timing breakdown.
    
    Args:
        scene_dict: Scene configuration dictionary
        timeout_s: Subprocess timeout in seconds
        verbose: If True, print detailed timing breakdown
        
    Returns:
        tuple: (parsed_result, timing_dict) where timing_dict contains:
            - 'json_encode_ms': Time to encode scene to JSON (ms)
            - 'subprocess_ms': Time to spawn and run Node.js subprocess (ms)
            - 'json_decode_ms': Time to decode result from JSON (ms)
            - 'total_ms': Total time (ms)
    """
    runner = _runner_path()
    if not runner.exists():
        raise FileNotFoundError(f"runner.js not found at {runner}")

    # Time JSON encoding
    t0 = time.perf_counter()
    scene_json = json.dumps(scene_dict)
    json_encode_time = time.perf_counter() - t0
    
    # Time subprocess execution
    t1 = time.perf_counter()
    result = subprocess.run(
        ["node", str(runner)],
        input=scene_json.encode(),
        capture_output=True,
        timeout=timeout_s,
        check=False,
        env=os.environ.copy(),
    )
    subprocess_time = time.perf_counter() - t1

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore").strip()
        raise RuntimeError(f"ray-optics runner failed (code {result.returncode}): {stderr}")

    # Time JSON decoding
    t2 = time.perf_counter()
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        stderr = result.stderr.decode(errors="ignore").strip()
        stdout = result.stdout.decode(errors="ignore").strip()
        raise RuntimeError(
            f"Failed to parse runner output as JSON. stdout={stdout} stderr={stderr}"
        ) from exc
    json_decode_time = time.perf_counter() - t2
    
    total_time = time.perf_counter() - t0
    
    timing = {
        'json_encode_ms': json_encode_time * 1000,
        'subprocess_ms': subprocess_time * 1000,
        'json_decode_ms': json_decode_time * 1000,
        'total_ms': total_time * 1000
    }
    
    if verbose:
        print(f"[run_simulation] JSON encode: {timing['json_encode_ms']:.2f}ms | "
              f"subprocess: {timing['subprocess_ms']:.2f}ms | "
              f"JSON decode: {timing['json_decode_ms']:.2f}ms | "
              f"total: {timing['total_ms']:.2f}ms", flush=True)
        # Level 1 JS-side timing (from runner.js)
        js_timing = parsed.get("timing")
        if js_timing:
            print(f"[runner.js] json_parse: {js_timing.get('json_parse_ms', 0):.2f}ms | "
                  f"scene_create: {js_timing.get('scene_create_ms', 0):.2f}ms | "
                  f"scene_load: {js_timing.get('scene_load_ms', 0):.2f}ms | "
                  f"simulator_create: {js_timing.get('simulator_create_ms', 0):.2f}ms | "
                  f"simulation_run: {js_timing.get('simulation_run_ms', 0):.2f}ms | "
                  f"detector_extract: {js_timing.get('detector_extract_ms', 0):.2f}ms | "
                  f"json_stringify: {js_timing.get('json_stringify_ms', 0):.2f}ms | "
                  f"total_js: {js_timing.get('total_ms', 0):.2f}ms", flush=True)

    return parsed, timing


def create_runner(persistent=True, **kwargs):
    """Factory for simulation runners with a shared simulate() interface."""
    if persistent:
        from .ray_optics_bridge_persistent import PersistentRayOpticsRunner

        return PersistentRayOpticsRunner(**kwargs)
    return OneShotRunner(**kwargs)
