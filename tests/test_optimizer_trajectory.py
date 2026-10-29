"""Replay, recorder-equivalence, schema, and lifecycle split tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.qd.driver import ExperimentConfig, run  # noqa: E402
from nslens.rl.data.build_optimizer_trajectories import build  # noqa: E402
from nslens.rl.data.schema import validate_optimizer_transition  # noqa: E402
from nslens.search.cma_me import CMAES  # noqa: E402


class _Runner:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Problem:
    param_names = ("x", "y")
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    focus_w = 1.0
    x_sensor = 2.0
    n_rays = 3
    beam_w = 1.0
    relative_sag = False
    max_sag_ratio = None
    det_clearance = None

    def evaluate(self, runner, cfg, x, pert_rng=None):
        del runner, cfg, pert_rng
        x = np.asarray(x)
        return {
            "fitness": 1.0 - float(np.sum(x * x)),
            "bc": tuple(x),
            "info": {
                "dom_signature": "T@A -> TIR@T -> T@B",
                "per_field": [0.2, 0.3], "efficiency": 0.5,
                "crit_margin": 1.2,
            },
        }


def _config(seed=0, transitions=False):
    return ExperimentConfig(
        name="trajectory_dummy", problem=_Problem(),
        descriptors=("tir_bounces", "thinness"),
        bc_bins=(3, 3), bc_bounds=((-1.0, 1.0), (-1.0, 1.0)),
        cell_bc_keys=("x", "y"), evals=8, seed=seed,
        emitters=1, popsize=4, sigma0=0.2, param_scaling="normalized",
        trace=True, optimizer_transitions=transitions,
    )


class OptimizerTrajectoryTests(unittest.TestCase):
    def test_cma_snapshot_replays_next_population_exactly(self):
        bounds = (np.zeros(3), np.ones(3))
        emitter = CMAES(
            np.full(3, 0.5), 0.2, popsize=6, seed=42, bounds=bounds)
        emitter.ask()
        emitter.tell(np.arange(6))
        snapshot = emitter.state_snapshot()
        restored = CMAES.from_state_snapshot(snapshot, bounds=bounds)
        np.testing.assert_array_equal(emitter.ask(), restored.ask())
        self.assertEqual(emitter.sampling_snapshot(), restored.sampling_snapshot())

    def test_recording_does_not_move_search(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "nslens.qd.driver.Runner", _Runner):
            root = Path(directory)
            plain = run(_config(transitions=False), root / "plain", log=lambda _: None)
            recorded = run(_config(transitions=True), root / "recorded", log=lambda _: None)
            self.assertEqual(plain["cells"], recorded["cells"])
            self.assertEqual(plain["history"], recorded["history"])
            self.assertEqual(
                (root / "plain" / "trace.jsonl").read_text(),
                (root / "recorded" / "trace.jsonl").read_text())
            rows = [json.loads(line) for line in
                    (root / "recorded" / "optimizer_transitions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not validate_optimizer_transition(row) for row in rows))
        self.assertTrue(all(len(row["candidates"]) == 4 for row in rows))
        self.assertEqual(rows[0]["state_after"]["generation"], 1)
        self.assertEqual(rows[0]["bounds"], [[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_equal(
            [candidate["search_coordinates"] for candidate in rows[0]["candidates"]],
            rows[0]["sampling"]["population"])
        replay_bounds = (
            np.asarray([bound[0] for bound in rows[0]["bounds"]]),
            np.asarray([bound[1] for bound in rows[0]["bounds"]]),
        )
        replay = CMAES.from_state_snapshot(
            rows[0]["state_before"], bounds=replay_bounds)
        np.testing.assert_array_equal(
            replay.ask(), rows[0]["sampling"]["population"])
        replay.tell(np.asarray(rows[0]["ranking_order"]))
        self.assertEqual(replay.state_snapshot(), rows[0]["state_after"])

    def test_builder_splits_whole_lifecycles_and_adds_targets(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "nslens.qd.driver.Runner", _Runner):
            root = Path(directory)
            paths = []
            for seed in range(3):
                out = root / f"seed{seed}"
                run(_config(seed=seed, transitions=True), out, log=lambda _: None)
                paths.append(out / "optimizer_transitions.jsonl")
            manifest = build(paths, root / "dataset", hindsight=2)
            seen = {}
            for split in ("train", "validation", "test"):
                path = root / "dataset" / "held_seed" / f"{split}.jsonl"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    lifecycle = row["source_lifecycle_id"]
                    if lifecycle in seen:
                        self.assertEqual(seen[lifecycle], split)
                    else:
                        seen[lifecycle] = split
                    self.assertIn("teacher_target", row)
                    self.assertIn("hindsight_target", row)
            self.assertEqual(manifest["complete_generations"], 6)
            self.assertTrue(manifest["checks"]["whole_lifecycle_splits"])


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("ALL OPTIMIZER TRAJECTORY CHECKS PASS")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
