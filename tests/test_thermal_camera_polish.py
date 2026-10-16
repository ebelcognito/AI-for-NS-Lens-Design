"""Regression tests for the fixed-reference post-CMA residual vector."""
import unittest
from unittest.mock import patch
import numpy as np

from nslens.thermal_camera import spectrometer_style_model as model
from nslens.thermal_camera.polish import residuals


class ThermalCameraPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.genes = np.zeros(model.N_TOTAL)
        cls.reference_trace = model.trace(cls.genes, fields=3, waves=1, pupils=3)
        cls.reference_good = cls.reference_trace['good'].copy()

    def test_residual_shape_is_fixed_finite_and_matches_candidates(self):
        base = residuals(self.genes, self.reference_good, fields=3, pupils=3)
        candidate = self.genes.copy()
        candidate[0] = .1
        changed = residuals(candidate, self.reference_good, fields=3, pupils=3)
        self.assertEqual(base.shape, changed.shape)
        self.assertGreater(base.size, 0)
        self.assertTrue(np.isfinite(base).all())
        self.assertTrue(np.isfinite(changed).all())
        self.assertGreater(np.linalg.norm(base-changed), 1e-6)

    def test_losing_an_initially_valid_ray_increases_residual_cost(self):
        reference = self.reference_good
        trace = dict(self.reference_trace)
        lost = dict(trace)
        lost_good = trace['good'].copy()
        # Remove one initially valid central-field ray while leaving its image
        # coordinates unchanged, isolating the fixed-mask loss penalties.
        field = len(reference)//2
        ray = int(np.flatnonzero(reference[field, 0])[0])
        lost_good[field, 0, ray] = False
        lost['good'] = lost_good
        with patch.object(model, 'trace', return_value=trace):
            baseline = residuals(self.genes, reference, fields=3, pupils=3)
        with patch.object(model, 'trace', return_value=lost):
            after_loss = residuals(self.genes, reference, fields=3, pupils=3)
        self.assertEqual(baseline.shape, after_loss.shape)
        self.assertGreater(float(after_loss @ after_loss), float(baseline @ baseline))


if __name__ == '__main__':
    unittest.main()
