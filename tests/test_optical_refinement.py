"""Small-budget invariants for shared optical refinement."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from nslens.analysis import optical_refinement as refinement  # noqa: E402
from nslens.analysis import final_freeform as ff  # noqa: E402


GENES = np.array([
    0.6184022185817588, 0.3738782049314582, -0.5933151268969421,
    -0.5075446650387713, 0.5423782661084044, -0.3145991896314273,
    -0.25608358433075673, -0.48424361598386945, -0.42595325194201794,
    -0.8431832152639132, -0.10764401476866554, 0.0016940064966006266,
    -0.21891028881371086, -0.05426793005869776, -0.2275414120958611,
])
N = 5
TASK = dict(focal=75.0, aperture=35.0, half_field_deg=8.0)


class OpticalRefinementTests(unittest.TestCase):
    def test_small_budget_last_freeform_preserves_layout_and_bounds_coefficients(self):
        result = refinement.refine_design(
            GENES, N, TASK, seed=731, refiner="last_freeform", budget=13)

        self.assertEqual(result["coarse_calls"], 13)
        np.testing.assert_array_equal(result["initial"]["genes"], GENES)
        np.testing.assert_array_equal(
            result["initial"]["freeform_coefficients_mm"], np.zeros(4))
        np.testing.assert_array_equal(result["coarse"]["genes"][:-1], GENES[:-1])
        self.assertGreaterEqual(result["coarse"]["genes"][-1], -1.0)
        self.assertLessEqual(result["coarse"]["genes"][-1], 1.0)
        coefficients = np.asarray(result["coarse"]["freeform_coefficients_mm"])
        self.assertEqual(coefficients.shape, (4,))
        self.assertTrue(np.all(np.isfinite(coefficients)))
        self.assertTrue(np.all(np.abs(coefficients) <= 2.0))
        self.assertLessEqual(result["coarse"]["loss"], result["initial"]["loss"])

        # The detector location is defined by vertices/leg lengths, so changing
        # only the last curvature gene must leave the detector fixed.
        before = ff.decode(GENES, N, np.zeros(4))["image"]
        after = ff.decode(result["coarse"]["genes"], N, coefficients)["image"]
        np.testing.assert_array_equal(after, before)

    def test_invalid_slope_fit_falls_back_and_still_spends_the_fit_trace(self):
        real_fit = refinement.slope_fit
        fit_traces = []

        def invalid_after_real_trace(genes, n, task, free):
            # The compatibility helper includes one 5x21 trace even when its
            # constraints are unusable.
            real_fit(genes, n, task, free)
            fit_traces.append(True)
            raise ValueError("no usable slope constraints")

        with patch.object(refinement, "slope_fit", side_effect=invalid_after_real_trace):
            result = refinement.refine_design(
                GENES, N, TASK, seed=732, refiner="last_parabolic", budget=13)

        self.assertEqual(result["coarse_calls"], 13)
        self.assertEqual(fit_traces, [True])
        self.assertFalse(result["compatibility"]["available"])
        self.assertIn("no usable slope constraints", result["compatibility"]["reason"])
        self.assertLessEqual(result["coarse"]["loss"], result["initial"]["loss"])
        np.testing.assert_array_equal(result["coarse"]["genes"][:-1], GENES[:-1])
        np.testing.assert_array_equal(
            result["coarse"]["freeform_coefficients_mm"], np.zeros(4))
