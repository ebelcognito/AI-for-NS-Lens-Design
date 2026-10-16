"""State invariants for three-mirror freeform refinement."""
import unittest
from unittest.mock import patch

import numpy as np

from nslens.analysis import adapter_refinement as adapter


class AdapterRefinementTests(unittest.TestCase):
    def test_evaluation_keeps_starting_freeform_parameters_and_checks_bounds(self):
        state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
                          0.25, 0.1, -0.2, 0.3, -0.4])
        captured = {}

        def trace(genes, n, coefficients, **kwargs):
            captured.update(genes=np.array(genes), n=n,
                            coefficients=np.array(coefficients), kwargs=kwargs)
            return {"loss": 2.0}

        with patch.object(adapter.ff, "trace", side_effect=trace), \
                patch.object(adapter, "geometry_crossing", return_value=False):
            row = adapter.evaluate_state(state, {"focal": 50.0})

        np.testing.assert_array_equal(captured["genes"], state[:9])
        np.testing.assert_array_equal(captured["coefficients"], state[9:] * 2.0)
        self.assertEqual(captured["n"], 3)
        np.testing.assert_array_equal(row["state"], state)
        for invalid in (np.zeros(12), np.r_[np.zeros(12), np.nan],
                        np.r_[np.zeros(12), 1.01]):
            with self.assertRaises(ValueError):
                adapter.evaluate_state(invalid, {})

    def test_refine_spends_budget_preserves_geometry_and_incumbent(self):
        start = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
                          0.2, 0.15, -0.1, 0.05, -0.2])
        seen = []

        def fake_evaluate(state, task):
            state = np.asarray(state).copy()
            seen.append(state)
            # Starting point remains incumbent with loss 0; later trials are worse.
            loss = 0.0 if np.array_equal(state, start) else 1.0
            return {"state": state.tolist(), "loss": loss}

        with patch.object(adapter, "evaluate_state", side_effect=fake_evaluate):
            result = adapter.refine_state(start, {}, seed=23, budget=19)

        self.assertEqual(len(seen), 19)
        self.assertEqual(result["coarse_calls"], 19)
        np.testing.assert_array_equal(result["initial"]["state"], start)
        np.testing.assert_array_equal(result["coarse"]["state"], start)
        np.testing.assert_array_equal(result["state"], start)
        for state in seen:
            np.testing.assert_array_equal(state[:8], start[:8])
            self.assertTrue(np.all(np.isfinite(state)))
            self.assertTrue(np.all(state >= -1.0))
            self.assertTrue(np.all(state <= 1.0))

    def test_budget_one_returns_initial_without_sampling(self):
        start = np.zeros(13)
        with patch.object(adapter, "evaluate_state", return_value={
                "state": start.tolist(), "loss": 3.0}) as evaluate:
            result = adapter.refine_state(start, {}, seed=1, budget=1)
        evaluate.assert_called_once()
        self.assertEqual(result["coarse_calls"], 1)
        self.assertEqual(result["coarse"], result["initial"])


if __name__ == "__main__":
    unittest.main()
