import atexit
import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_TIMEOUT_S = 5.0


class PersistentRayOpticsRunner:
    """Persistent Node.js process for ray-optics simulations over NDJSON."""

    def __init__(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        verbose: bool = False,
        log_every_n: int = 5000,
    ):
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.log_every_n = max(int(log_every_n), 1)
        self._process: Optional[subprocess.Popen] = None
        self._request_count = 0
        self._atexit_registered = False

    def _get_runner_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "ray-optics" / "integrations" / "runner_persistent.js"

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._is_running():
            return

        runner = self._get_runner_path()
        if not runner.exists():
            raise FileNotFoundError(f"runner_persistent.js not found at {runner}")

        self._process = subprocess.Popen(
            ["node", str(runner)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        if self.verbose:
            print("[PersistentRunner] Node.js process started", flush=True)

    def stop(self) -> None:
        if self._process is None:
            return

        process = self._process

        if process.poll() is None:
            try:
                if process.stdin:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                process.wait(timeout=2.0)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except Exception:
                    process.kill()
                    process.wait(timeout=1.0)

        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

        self._process = None

        if self.verbose:
            print(
                f"[PersistentRunner] Stopped after {self._request_count} requests",
                flush=True,
            )

    def restart(self) -> None:
        self.stop()
        self.start()

    def _read_response_line(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Node process not running. Call start() first.")

        ready, _, _ = select.select([self._process.stdout], [], [], self.timeout_s)
        if not ready:
            raise TimeoutError(
                f"Timed out waiting for Node response after {self.timeout_s:.2f}s"
            )

        line = self._process.stdout.readline()
        if line:
            return line

        stderr = ""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read().strip()
        raise RuntimeError(f"Node process terminated unexpectedly: {stderr}")

    def simulate(
        self,
        scene_dict: Dict[str, Any],
        episode: int = 0,
        step: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        if not self._is_running():
            raise RuntimeError("Node process not running. Call start() first.")

        assert self._process is not None
        assert self._process.stdin is not None

        t0 = time.perf_counter()

        request = {
            "episode": episode,
            "step": step,
            "scene": scene_dict,
        }

        t1 = time.perf_counter()
        request_line = json.dumps(request) + "\n"
        self._process.stdin.write(request_line)
        self._process.stdin.flush()
        json_encode_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        try:
            response_line = self._read_response_line()
        except TimeoutError:
            self.restart()
            raise
        subprocess_time = time.perf_counter() - t2

        t3 = time.perf_counter()
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse runner response as JSON: {response_line.strip()}"
            ) from exc
        json_decode_time = time.perf_counter() - t3

        self._request_count += 1
        total_time = time.perf_counter() - t0

        if response.get("error"):
            raise RuntimeError(
                f"Simulation failed: {response.get('details', response['error'])}"
            )

        result = response.get("result", response)

        timing = {
            "json_encode_ms": json_encode_time * 1000,
            "subprocess_ms": subprocess_time * 1000,
            "json_decode_ms": json_decode_time * 1000,
            "total_ms": total_time * 1000,
        }

        if self.verbose and self._request_count % self.log_every_n == 0:
            print(
                f"[simulate] Request {self._request_count} | "
                f"encode: {timing['json_encode_ms']:.2f}ms | "
                f"roundtrip: {timing['subprocess_ms']:.2f}ms | "
                f"decode: {timing['json_decode_ms']:.2f}ms | "
                f"total: {timing['total_ms']:.2f}ms",
                flush=True,
            )

            js_timing = result.get("timing")
            if js_timing:
                print(
                    f"[runner_persistent.js] simulation: "
                    f"{js_timing.get('simulation_run_ms', 0):.2f}ms | "
                    f"total_js: {js_timing.get('total_ms', 0):.2f}ms",
                    flush=True,
                )

        return result, timing

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
