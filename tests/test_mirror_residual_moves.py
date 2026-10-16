import unittest

import numpy as np

from nslens.analysis.mirror_residual_moves import (
    action_groups, jacobian, observe, propose,
)
from nslens.analysis.variable_mirror import baseline_genes, trace


class MirrorResidualMoveTests(unittest.TestCase):
    def test_observation_matches_direct_detector_mapping(self):
        genes = baseline_genes(3)
        task = {"focal": 50.0, "aperture": 27.78, "half_field_deg": 4.0}
        observed = observe(genes, 3, task)
        row, mirror, _outlines, paths, good = trace(
            genes, 3, fields=3, pupils=9, paths=True, **task)
        image = np.array([(p[-1] - mirror["image"]) @ mirror["image_tangent"]
                          if good[i] else 0.0 for i, p in enumerate(paths)])
        fields = np.repeat(np.linspace(-4, 4, 3), 9)
        targets = row["parity"] * 50 * np.tan(np.deg2rad(fields))
        expected = np.where(good, (image - targets) /
                            max(50 * np.tan(np.deg2rad(4)), 1), 0)
        np.testing.assert_allclose(observed["residual"], expected)
        np.testing.assert_array_equal(observed["mask"], good)
        self.assertEqual(observed["parity"], row["parity"])
        self.assertEqual(observed["features"].shape, (15,))

    def test_missing_rays_are_zero_but_never_marked_valid(self):
        genes = baseline_genes(3)
        obs = observe(genes, 3, {"aperture": 50.0})
        self.assertEqual(obs["residual"].shape, (27,))
        self.assertTrue(np.all(obs["residual"][~obs["mask"]] == 0))
        J, valid = jacobian(genes, obs, lambda x: observe(x, 3, {"aperture": 50.0}))
        self.assertEqual(J.shape, (27, 9))
        self.assertFalse(np.any(valid[~obs["mask"]]))

    def test_jacobian_uses_bounded_denominator_and_common_masks(self):
        x = np.array([1.0, 0.0])
        calls = []

        def callback(z):
            calls.append(z.copy())
            return {"residual": np.array([z[0] + 2*z[1], z[0] - z[1]]),
                    "mask": np.array([True, len(calls) % 2 == 1])}

        parent = {"residual": np.array([1., 1.]), "mask": np.ones(2, bool)}
        J, valid = jacobian(x, parent, callback, h=0.2)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(J[0, 0], 1.0)
        self.assertFalse(valid[1, 0])
        self.assertTrue(np.all(valid[0]))

    def test_proposals_reduce_residual_respect_group_radius_and_bounds(self):
        x = np.array([0.0, 0.0, 0.99])
        J = np.array([[1., 0., 10.], [0., 1., 10.]])
        residual = np.array([1., -2.])
        step = propose(x, residual, J, np.array([0, 1]), radius=0.1)
        self.assertEqual(step[2], 0.)
        self.assertLessEqual(np.linalg.norm(step), 0.1 + 1.e-12)
        self.assertLess(np.linalg.norm(residual + J @ step), np.linalg.norm(residual))
        bounded = propose(np.array([0.99, 0.]), np.array([-1.]),
                          np.array([[1., 0.]]), np.array([0]), radius=2.)
        self.assertLessEqual(0.99 + bounded[0], 1.0 + 1.e-12)
        self.assertEqual(bounded[1], 0.)
        groups = action_groups(3)
        np.testing.assert_array_equal(groups["angles"], [0, 1, 2])
        np.testing.assert_array_equal(groups["lengths"], [3, 4, 5])
        np.testing.assert_array_equal(groups["curves"], [6, 7, 8])
        self.assertEqual(len(groups["all"]), 9)


if __name__ == "__main__":
    unittest.main()
