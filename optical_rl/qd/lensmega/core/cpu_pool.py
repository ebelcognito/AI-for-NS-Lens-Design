from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..problem.base import EvalResult


def _worker_loop(
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    config_dict: Dict[str, Any],
) -> None:
    """Long-lived CPU worker: builds its own ``LensProblem`` from a config
    dict, then loops on ``(idx, theta)`` tasks until a poison pill (``None``)
    arrives.  Lives in a separate process; never shares GPU state."""
    # Lazy imports so the parent does not pay them at spawn time.
    from optical_rl.qd.lensmega.config.schema import RunConfig
    from optical_rl.qd.lensmega.run import build_all

    # Keep intra-op threading low inside each worker; the parallelism comes
    # from having N workers, not from torch's own threadpools.  Without this,
    # 16 workers × torch default threads oversubscribes the CPU heavily and
    # scales sub-linearly.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = RunConfig.model_validate(config_dict)
    # Force CPU regardless of what the YAML asked for; this pool is CPU-only.
    cfg.io.device = "cpu"

    problem, _desc, _arch, _emit, _seeder = build_all(cfg)
    problem._ensure_backend()

    while True:
        item = task_q.get()
        if item is None:
            return
        idx, theta = item
        try:
            r = problem.evaluate(
                np.asarray(theta, dtype=np.float64), with_grads=False
            )
            result_q.put((idx, r, None))
        except Exception as e:  # pragma: no cover
            result_q.put((idx, None, repr(e)))


class CPUWorkerPool:
    """Persistent CPU worker pool for population + seeding evaluation.

    One process per worker, kept alive across CMA-MEGA generations so the
    spawn cost (lens load + pinv_Z build ~200-500ms) is paid once.  Each
    worker owns its own ``LensProblem`` instance constructed inside the
    process; no GPU state crosses the boundary, so scaling is linear in CPU
    cores up to the point of contention.
    """

    def __init__(
        self,
        num_workers: int,
        config_dict: Dict[str, Any],
    ) -> None:
        assert num_workers >= 1
        self.num_workers = int(num_workers)
        ctx = mp.get_context("spawn")
        self._task_q: mp.Queue = ctx.Queue()
        self._result_q: mp.Queue = ctx.Queue()
        self._workers = []
        for _ in range(self.num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(self._task_q, self._result_q, config_dict),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        self._alive = True

    def evaluate(self, thetas) -> List[Optional[EvalResult]]:
        """Dispatch a batch of thetas across workers, return in original order.

        Failures (worker exceptions, cancellations) are returned as ``None``
        at the corresponding index; callers treat those the same way they
        treat a rejected design.
        """
        arr = np.asarray(thetas, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        n = int(arr.shape[0])
        if n == 0:
            return []
        for i in range(n):
            self._task_q.put((i, arr[i]))
        results: List[Optional[EvalResult]] = [None] * n
        collected = 0
        while collected < n:
            idx, r, err = self._result_q.get()
            if err is not None:
                results[idx] = None
            else:
                results[idx] = r
            collected += 1
        return results

    def shutdown(self) -> None:
        if not self._alive:
            return
        for _ in self._workers:
            try:
                self._task_q.put(None)
            except Exception:
                pass
        for p in self._workers:
            try:
                p.join(timeout=5.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        self._alive = False

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.shutdown()
        except Exception:
            pass


def auto_worker_count(requested: Optional[int]) -> int:
    """Resolve ``parallel_workers`` from IOCfg to an integer.

    Rules:
      - ``None``  -> ``max(1, cpu_count - 2)`` capped at 32
      - ``0``     -> 1 (serial fallback; caller treats 1 as "no pool")
      - ``N > 0`` -> ``N``
    """
    if requested is None:
        cpus = os.cpu_count() or 4
        return max(1, min(32, cpus - 2))
    n = int(requested)
    if n < 1:
        return 1
    return n
