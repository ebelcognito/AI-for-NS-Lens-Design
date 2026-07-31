"""Long-lived multiprocessing worker pool for the CMA-MEGA candidate loop.

Wraps the same `_worker_fn` used by `seed_parallel_experiment.py`, but keeps
the workers alive across many `evaluate()` calls. This is required for the
CMA-MEGA inner loop, which evaluates ~lambda candidates per generation for
hundreds of generations: re-spawning workers per call would pay the (very
expensive) PyTorch + DeepLens + EF50ZernikeProblem startup cost on every
generation and dominate wallclock.

The seeding path keeps using the existing per-call `_parallel_evaluate`
because seeding is one-shot — the spawn cost amortizes cleanly over the
single big batch of thousands of evaluations.

Single-threaded use only. evaluate() blocks until all results are in.
shutdown() is idempotent. Use as a context manager for safety:

    with PersistentEvalPool(config, cfg_dict, 20, [0, 1]) as pool:
        for gen in range(N):
            results = pool.evaluate(candidate_batch)
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class PersistentEvalPool:
    """Persistent worker pool for EF50 lens evaluation."""

    def __init__(
        self,
        config: Any,
        config_dict: dict,
        num_workers: int,
        gpu_ids: List[int],
    ):
        import torch.multiprocessing as mp
        from optical_rl.qd.seed_parallel_experiment import _worker_fn

        if int(num_workers) <= 0:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if not gpu_ids:
            raise ValueError("gpu_ids must contain at least one device id")

        self._ctx = mp.get_context("spawn")  # spawn = safe for CUDA
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._workers: list = []
        self._num_workers = int(num_workers)
        self._gpu_ids = list(gpu_ids)
        self._shutdown_called = False

        # Spawn N persistent worker processes, round-robin across GPUs.
        # Each worker pulls (idx, theta) tuples from task_queue, runs
        # problem.evaluate(theta, with_grads=False), and pushes
        # (idx, result_dict) (or (idx, None, err_msg)) onto result_queue.
        # The worker stays alive until it pulls a `None` poison pill.
        for w in range(self._num_workers):
            gpu = self._gpu_ids[w % len(self._gpu_ids)]
            p = self._ctx.Process(
                target=_worker_fn,
                args=(gpu, self._task_queue, self._result_queue, config_dict),
                daemon=False,
            )
            p.start()
            self._workers.append(p)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def gpu_ids(self) -> List[int]:
        return list(self._gpu_ids)

    def evaluate(self, thetas: np.ndarray) -> List[Optional[dict]]:
        """Dispatch all thetas to the worker pool, return results in order.

        Returns a list of length len(thetas). Each entry is either a
        dict {theta, fitness, measures, metrics} from the worker, or
        None if that worker raised an exception (which is also logged).
        """
        if self._shutdown_called:
            raise RuntimeError("PersistentEvalPool has been shut down")
        n = len(thetas)
        if n == 0:
            return []

        # Feed all tasks first, then collect — keeps queue depth bounded
        # by n and lets workers start processing as soon as the first
        # task lands on the queue.
        for i, theta in enumerate(thetas):
            self._task_queue.put((i, theta))

        results: List[Optional[dict]] = [None] * n
        errors: list = []
        for _ in range(n):
            item = self._result_queue.get()
            if len(item) == 3:  # (idx, None, err_msg) error case
                idx, _none, err_msg = item
                errors.append((idx, err_msg))
            else:                # (idx, result_dict) success case
                idx, result_dict = item
                results[idx] = result_dict

        if errors:
            print(
                f"[eval_pool] WARNING: {len(errors)} worker errors this batch:",
                flush=True,
            )
            for idx, msg in errors[:5]:
                print(f"  theta[{idx}]: {msg}", flush=True)

        return results

    def shutdown(self) -> None:
        """Send poison pills, join workers, drain queues. Idempotent."""
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # One poison pill per worker. Each worker exits its `while True`
        # loop when it pulls a None from the task queue.
        for _ in self._workers:
            try:
                self._task_queue.put(None)
            except Exception:
                pass

        for p in self._workers:
            try:
                p.join(timeout=10.0)
                if p.is_alive():
                    # Worker stuck on something — force terminate.
                    p.terminate()
                    p.join(timeout=2.0)
            except Exception:
                pass
        self._workers.clear()

    def __enter__(self) -> "PersistentEvalPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.shutdown()
        return False

    def __del__(self):  # best-effort safety net
        try:
            self.shutdown()
        except Exception:
            pass
