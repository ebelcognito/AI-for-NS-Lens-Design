import unittest

import numpy as np

from nslens.analysis.routing_transition_policy import TransitionPolicy


class TransitionPolicyTests(unittest.TestCase):
    def test_proposals_preserve_trailing_state_and_bound_steps(self):
        X = np.array([[0.0], [1.0], [2.0]])
        D = np.zeros((3, 9))
        D[0, 0] = 0.4
        D[1, 1] = -0.2
        D[2, 2] = 0.1
        policy = TransitionPolicy().fit(X, D)
        state = np.r_[np.zeros(9), [0.123, -0.456, 0.789, -0.321]]

        proposals = policy.propose([0.0], state, count=5, radius=0.15)

        self.assertEqual(proposals.shape, (5, 13))
        np.testing.assert_array_equal(proposals[:, 9:], np.tile(state[9:], (5, 1)))
        self.assertTrue(np.all(proposals >= -1.0))
        self.assertTrue(np.all(proposals <= 1.0))
        self.assertTrue(np.all(np.linalg.norm(proposals[:, :9] - state[:9], axis=1) <= 0.15 + 1e-12))

    def test_retrieves_modes_instead_of_averaging_conflicting_moves(self):
        X = np.array([[0.0], [0.01]])
        D = np.zeros((2, 9))
        D[0, 0] = 0.1
        D[1, 0] = -0.1
        policy = TransitionPolicy().fit(X, D)

        first = policy.propose([0.005], np.zeros(13), count=2, radius=0.2)
        again = policy.propose([0.005], np.zeros(13), count=2, radius=0.2)

        self.assertEqual(first[0, 0], 0.1)
        self.assertEqual(first[1, 0], -0.1)
        self.assertFalse(np.any(np.all(np.isclose(first[:, :9], 0), axis=1)))
        np.testing.assert_array_equal(first, again)

    def test_duplicate_moves_are_removed_and_short_results_pad_with_state(self):
        X = np.array([[0.0], [1.0], [2.0]])
        D = np.zeros((3, 9))
        D[0, 0] = D[1, 0] = 0.1
        state = np.zeros(13)
        proposals = TransitionPolicy().fit(X, D).propose([0.0], state, count=3)
        self.assertEqual(proposals.shape, (3, 13))
        self.assertEqual(np.count_nonzero(proposals[0]), 1)
        np.testing.assert_array_equal(proposals[1:], np.zeros((2, 13)))

    def test_invalid_training_and_proposal_shapes_are_rejected(self):
        policy = TransitionPolicy()
        with self.assertRaises(ValueError):
            policy.fit(np.array([1.0, 2.0]), np.zeros((2, 9)))
        with self.assertRaises(ValueError):
            policy.fit(np.zeros((2, 1)), np.zeros((1, 9)))
        with self.assertRaises(ValueError):
            policy.fit(np.array([[np.nan]]), np.zeros((1, 9)))
        policy.fit(np.zeros((1, 1)), np.zeros((1, 9)))
        with self.assertRaises(ValueError):
            policy.propose([0.0, 1.0], np.zeros(13))
        with self.assertRaises(ValueError):
            policy.propose([0.0], np.zeros(12))
        with self.assertRaises(ValueError):
            policy.propose([0.0], np.r_[np.zeros(12), 1.1])
        with self.assertRaises(ValueError):
            policy.propose([0.0], np.zeros(13), count=0)


if __name__ == "__main__":
    unittest.main()
