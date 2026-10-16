import unittest
from unittest.mock import patch

import numpy as np

from nslens.analysis import routing_repair_teacher as teacher


class FixedCMA:
    def __init__(self, mean, *args, **kwargs):
        self.mean = np.asarray(mean).copy()
        self.calls = 0

    def ask(self):
        self.calls += 1
        rows = np.tile(self.mean, (8, 1))
        rows[0, 0] = .99
        return rows

    def tell(self, order):
        pass


def fake_evaluate(state, task):
    state = np.asarray(state)
    return dict(loss=float((state[0] - .99) ** 2), worst_support=.8,
                crossing=False, ray_good=np.ones(1), ray_field_deg=np.zeros(1),
                ray_image_mm=np.zeros(1), parity=1, ray_route=np.zeros((1, 3)))


class RoutingRepairTeacherTests(unittest.TestCase):
    @patch.object(teacher, 'CMAES', FixedCMA)
    @patch.object(teacher, 'evaluate_state', side_effect=fake_evaluate)
    @patch.object(teacher, 'features', side_effect=lambda state, row, task: np.asarray(state).copy())
    def test_exact_budget_and_parent_features(self, feature_mock, evaluate_mock):
        task = dict(focal=75., aperture=35., half_field_deg=8.)
        result = teacher.collect(dict(seed=17, task=task, task_name='field8'), budget=10)

        self.assertEqual(result['coarse_calls'], 10)
        self.assertEqual(evaluate_mock.call_count, 10)
        self.assertEqual(result['diagnostics']['children'], 9)
        self.assertEqual(result['task'], 'field8')
        self.assertLess(result['final_loss'], result['initial_loss'])
        self.assertTrue(result['transitions'])
        initial_state = np.r_[teacher.initial(3, np.random.default_rng(17)), np.zeros(4)]
        for transition in result['transitions']:
            np.testing.assert_array_equal(transition['features'], initial_state)
            np.testing.assert_array_equal(transition['parent'], initial_state)
            np.testing.assert_array_equal(transition['child'][9:], np.zeros(4))
            np.testing.assert_array_equal(transition['delta'], transition['child'][:9] - initial_state[:9])
            self.assertEqual(transition['task'], 'field8')
            self.assertLess(transition['child_loss'], transition['parent_loss'])
            self.assertGreaterEqual(transition['child_support'], transition['parent_support'] - .01)

    def test_budget_one_counts_only_initial_evaluation(self):
        with patch.object(teacher, 'evaluate_state', side_effect=fake_evaluate) as evaluate_mock:
            result = teacher.collect(dict(seed=1, task=dict(focal=75., aperture=42., half_field_deg=6.),
                                          task_name='aperture42'), budget=1)
        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertEqual(result['coarse_calls'], 1)
        self.assertEqual(result['transitions'], [])

    def test_invalid_job_and_budget_rejected(self):
        with self.assertRaises(ValueError):
            teacher.collect({'seed': 2}, budget=1)
        with self.assertRaises(ValueError):
            teacher.collect({'seed': 2, 'task': {}}, budget=0)


if __name__ == '__main__':
    unittest.main()
