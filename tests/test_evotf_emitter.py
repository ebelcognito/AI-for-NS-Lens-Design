"""Contracts for the released EvoTF emitter and its dependency-free driver hook."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.qd.driver import ExperimentConfig, run
from nslens.search.cma_me import CMAES
from nslens.search.evotf_emitter import (
    EVOSAX_VERSION,
    EvoTFEmitter,
    stable_rank_fitness,
)


def _official_runtime_available() -> bool:
    if importlib.util.find_spec("jax") is None or importlib.util.find_spec("evosax") is None:
        return False
    try:
        return metadata.version("evosax") == EVOSAX_VERSION
    except metadata.PackageNotFoundError:
        return False


class RankFitnessTests(unittest.TestCase):
    def test_best_first_order_becomes_rank_percentiles(self):
        order = np.array([2, 0, 3, 1])
        rank = stable_rank_fitness(order)
        np.testing.assert_allclose(rank, [1 / 3, 1, 0, 2 / 3])
        self.assertEqual(list(np.argsort(rank)), list(order))
        self.assertLessEqual(float(np.max(rank)), 1.0)

    def test_invalid_order_is_rejected(self):
        for order in ([0, 0, 1], [0, 2], [[0, 1]]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                stable_rank_fitness(order, 3)


class _DummyRunner:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _DummyProblem:
    param_names = ("x", "y")
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    focus_w = 1.0
    calls = 0

    def evaluate(self, runner, cfg, x, pert_rng=None):
        del runner, cfg, pert_rng
        self.calls += 1
        x = np.asarray(x)
        return {"fitness": -float(np.sum(x * x)), "bc": tuple(x), "info": {}}


class _FakeEvoTF(CMAES):
    """Dependency-free emitter used only to exercise driver selection/accounting."""
    def __init__(self, *args, max_context_len=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_context_len = max_context_len
        self._restart_reason = None

    def note_restart(self, reason):
        self._restart_reason = reason

    def state_summary(self):
        return {
            "generation": self.gen,
            "ask_calls": self.gen,
            "tell_calls": self.gen,
            "ask_seconds": 0.0,
            "inference_seconds": 0.0,
            "clipped_candidates": 0,
            "clipped_values": 0,
            "clipped_by_gene": [0, 0],
            "fallback_events": 0,
            "fallback_candidates": 0,
            "nonfinite_events": 0,
            "restart_events": int(self._restart_reason is not None),
            "numerical_failure": None,
            "restart_reason": self._restart_reason,
        }


class DriverHookTests(unittest.TestCase):
    def _config(self, emitter):
        problem = _DummyProblem()
        return ExperimentConfig(
            name="dummy", problem=problem, descriptors=("tir_bounces", "thinness"),
            bc_bins=(2, 2), bc_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            evals=8, emitters=1, emitter=emitter, popsize=4, sigma0=0.2,
            param_scaling="normalized", cell_bc_keys=("x", "y"),
        )

    def test_default_emitter_remains_cma(self):
        cfg = self._config("cma")
        with tempfile.TemporaryDirectory() as directory, patch(
                "nslens.qd.driver.Runner", _DummyRunner):
            payload = run(cfg, Path(directory), log=lambda message: None)
        self.assertEqual(payload["meta"]["emitter"], "cma")
        self.assertEqual(payload["meta"]["evals"], 8)
        self.assertEqual(cfg.problem.calls, 8)
        self.assertIsNone(payload["meta"]["emitter_telemetry"])

    def test_evotf_selection_uses_same_candidate_budget(self):
        cfg = self._config("evotf")
        with tempfile.TemporaryDirectory() as directory, patch(
                "nslens.qd.driver.Runner", _DummyRunner), patch(
                "nslens.search.evotf_emitter.EvoTFEmitter", _FakeEvoTF):
            payload = run(cfg, Path(directory), log=lambda message: None)
        self.assertEqual(payload["meta"]["emitter"], "evotf")
        self.assertEqual(payload["meta"]["evals"], 8)
        self.assertEqual(cfg.problem.calls, 8)
        self.assertGreaterEqual(payload["meta"]["emitter_telemetry"]["lifecycles"], 1)


@unittest.skipUnless(_official_runtime_available(), "optional evosax==0.3.1 not installed")
class OfficialEvoTFTests(unittest.TestCase):
    def _emitter(self, seed=7, sigma=2.0):
        return EvoTFEmitter(
            np.full(3, 0.5), sigma, popsize=4, seed=seed,
            bounds=(np.zeros(3), np.ones(3)), max_context_len=4,
        )

    def test_bounds_determinism_and_tell(self):
        left, right = self._emitter(), self._emitter()
        first_left, first_right = left.ask(), right.ask()
        np.testing.assert_array_equal(first_left, first_right)
        self.assertTrue(np.all((0.0 <= first_left) & (first_left <= 1.0)))
        np.testing.assert_array_equal(left.last_clipping_mask, right.last_clipping_mask)
        self.assertTrue(np.any(left.last_clipping_mask))
        order = np.array([2, 0, 3, 1])
        left.tell(order)
        right.tell(order)
        np.testing.assert_array_equal(left.ask(), right.ask())
        np.testing.assert_array_equal(left.mean, right.mean)

    def test_nonfinite_population_falls_back_and_requests_restart(self):
        import jax.numpy as jnp

        emitter = self._emitter(sigma=0.2)
        emitter._state = emitter._state.replace(std=jnp.full(3, jnp.nan))
        population = emitter.ask()
        self.assertTrue(np.all(np.isfinite(population)))
        self.assertTrue(emitter.restart_needed)
        self.assertTrue(np.all(emitter.last_fallback_mask))
        json.dumps(emitter.state_summary(), allow_nan=False)

    def test_state_summary_is_json_serializable(self):
        emitter = self._emitter(sigma=0.2)
        emitter.ask()
        emitter.tell([0, 1, 2, 3])
        summary = emitter.state_summary()
        json.dumps(summary, allow_nan=False)
        self.assertEqual(summary["evosax_version"], EVOSAX_VERSION)
        self.assertEqual(summary["ask_calls"], 1)
        self.assertEqual(summary["tell_calls"], 1)
        self.assertTrue(summary["state_finite"])


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("ALL EVOTF EMITTER CHECKS PASS")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
