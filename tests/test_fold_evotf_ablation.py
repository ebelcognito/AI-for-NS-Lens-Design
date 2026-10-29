"""Dependency-free paired-summary checks for the EvoTF online comparison."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.analysis.summarize_fold_evotf_ablation import summarize  # noqa: E402


class FoldEvoTFAblationTests(unittest.TestCase):
    def test_pairing_uses_complete_seed_units_and_orients_lower_metrics(self):
        records = []
        for seed, cma_coverage, optical_coverage, cma_invalid, optical_invalid in (
                (2, 0.1, 0.2, 0.5, 0.25),
                (7, 0.2, 0.1, 0.5, 0.75)):
            records.extend((
                {"arm": "cma", "seed": seed, "metrics": {
                    "coverage": cma_coverage, "invalid_rate": cma_invalid}},
                {"arm": "optical_full", "seed": seed, "metrics": {
                    "coverage": optical_coverage, "invalid_rate": optical_invalid}},
            ))
        payload = {"schema_version": "nslens.fold_evotf_ablation.v1",
                   "records": records}
        result = summarize(payload)
        paired = result["paired_differences"]["optical_full_minus_cma"]
        self.assertEqual(paired["coverage"]["direction"], "higher")
        self.assertAlmostEqual(paired["coverage"]["median_improvement"], 0.0)
        self.assertEqual(paired["coverage"]["wins"], 1)
        self.assertEqual(paired["invalid_rate"]["direction"], "lower")
        self.assertAlmostEqual(paired["invalid_rate"]["median_improvement"], 0.0)
        self.assertEqual(paired["invalid_rate"]["wins"], 1)


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("ALL FOLD EVOTF ABLATION CHECKS PASS")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
