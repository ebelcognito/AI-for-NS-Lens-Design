"""Physical, information-boundary, and budget checks for the learning loop.

Run with: python -m unittest discover -s tests -p test_freeform_repair_loop.py
"""
from dataclasses import replace
import unittest

import numpy as np

from nslens.analysis.run_repair_curriculum import make_specs, make_task, paired_summary, train_worker
from nslens.analysis.benchmark_repair_broyden import broyden_solve
from nslens.learning.freeform_repair import (
    ImagingSpec, Recorder, RepairPolicy, ResponsePolicy, curriculum_weights, evaluate, features,
    solve, teacher_step,
)


class FreeformRepairTests(unittest.TestCase):
    def setUp(self):
        self.spec = ImagingSpec("unit/train", "train")

    def test_physics_energy_and_full_pupil(self):
        v = evaluate(self.spec, np.zeros(4))
        self.assertTrue(v.physical_valid)
        self.assertLess(v.closure_error, 1e-10)
        self.assertGreater(v.min_ray_energy, .8)
        self.assertEqual(v.route_fraction, 1.)
        clipped = evaluate(replace(self.spec, pupil_half=40.), np.zeros(4))
        self.assertFalse(clipped.physical_valid)
        self.assertFalse(clipped.success)

    def test_local_teacher_improves_unseen_dense_grid(self):
        x = np.array([.10, .02, -.08, .03])
        rec = Recorder(self.spec, "test", "unit")
        before = rec(x)
        end, _ = teacher_step(x, before, rec)
        self.assertEqual(len(rec.records), 9)
        dense_before = evaluate(self.spec, x, "dense")
        dense_after = evaluate(self.spec, end, "dense")
        self.assertTrue(dense_after.physical_valid)
        self.assertLess(dense_after.rms, dense_before.rms)

    def test_policy_cannot_access_witness_or_corruption(self):
        family = dict(spec=self.spec.__dict__, witness=[0., .1, -.2, 0.])
        task = make_task(family, 1, np.random.default_rng(1), "unit")
        self.assertNotIn("witness", task)
        self.assertNotIn("corruption", task)
        v = evaluate(self.spec, task["start"])
        # IDs/splits are metadata, never numeric policy features.
        twin = replace(self.spec, family_id="secret/heldout", split="test")
        np.testing.assert_array_equal(features(self.spec, task["start"], v),
                                      features(twin, task["start"], v))
        with self.assertRaises(ValueError):
            features(self.spec, task["start"], evaluate(self.spec, task["start"], "dense"))

    def test_heldout_training_is_rejected(self):
        for cls in (RepairPolicy, ResponsePolicy):
            with self.assertRaises(ValueError):
                cls().fit([dict(split="test", features=[0.], action=[0.] * 4)])

    def test_equal_candidate_allowance(self):
        for algorithm in ("learned", "random", "reset", "teacher"):
            rec = Recorder(self.spec, "unit", "unit")
            solve(self.spec, [.03, 0., 0., 0.], rec, policy=RepairPolicy(), algorithm=algorithm)
            self.assertEqual(len(rec.records), 9)
            self.assertEqual(sum(r["verdict"]["rays"] for r in rec.records), 135)

    def test_broyden_reuses_probes_within_the_same_allowance(self):
        rec = Recorder(self.spec, "unit", "unit")
        x = [.1, .02, -.08, .03]
        end, _ = broyden_solve(self.spec, x, rec)
        self.assertEqual(len(rec.records), 9)
        after = evaluate(self.spec, end, "dense")
        self.assertTrue(after.physical_valid)
        self.assertLess(after.rms, evaluate(self.spec, x, "dense").rms)

    def test_training_records_all_attempts(self):
        task = dict(spec=self.spec.__dict__, start=[.1, .02, -.08, .03],
                    severity=1, task_id="unit", seed=1)
        result = train_worker((task, RepairPolicy(), "unit"))
        self.assertEqual(len(result["records"]), 10)
        self.assertIsNotNone(result["example"])
        self.assertLess(result["example"]["after_rms"], result["example"]["before_rms"])
        self.assertEqual(len(result["example"]["jacobian"]), 60)

    def test_learned_response_repairs_a_different_start(self):
        task = dict(spec=self.spec.__dict__, start=[.1, .02, -.08, .03],
                    severity=1, task_id="unit", seed=1)
        example = train_worker((task, ResponsePolicy(), "unit"))["example"]
        policy = ResponsePolicy().fit([example])
        x = [-.08, -.02, .1, -.03]
        rec = Recorder(self.spec, "unseen_start", "unit")
        end, _, _ = solve(self.spec, x, rec, policy=policy)
        self.assertTrue(evaluate(self.spec, end, "dense").physical_valid)
        self.assertLess(evaluate(self.spec, end, "dense").rms,
                        evaluate(self.spec, x, "dense").rms)

    def test_split_families_and_curriculum_coverage(self):
        specs = make_specs(17, (4, 3, 2))
        self.assertEqual(len({s["family_id"] for s in specs}), 9)
        for strategy in ("uniform", "fixed", "adaptive"):
            weights = curriculum_weights(strategy, [0, 5, 10, 5], [10] * 4, 1, 3)
            self.assertTrue(np.all(weights > 0))
            self.assertAlmostEqual(sum(weights), 1.)
        weights = curriculum_weights("adaptive", [0, 5, 10, 5], [10] * 4, 1, 3)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])

    def test_statistics_pair_tasks_and_cluster_siblings(self):
        left = [dict(task_id=str(i), family_id=str(i // 4),
                     audit=dict(success=True, rms=.1)) for i in range(8)]
        right = [dict(task_id=str(i), family_id=str(i // 4),
                      audit=dict(success=False, rms=.2)) for i in range(8)]
        summary = paired_summary(left, right)
        self.assertEqual(summary["families"], 2)
        self.assertEqual(summary["success_rate_delta"], 1.)
        with self.assertRaises(ValueError):
            paired_summary(left, right[:-1])


if __name__ == "__main__":
    unittest.main()
