"""Independent search seeds, not rays or shapes, are the uncertainty blocks."""
import unittest
from nslens.discovery.summarize_benchmark import paired_comparison


def fixture():
    rows = []
    for seed in range(6):
        for k in (3, 4, 5):
            for arm in ("portfolio", "naive"):
                rows.append(dict(unit=f"{seed}_{k}", seed=seed, arm=arm,
                    config=dict(codec=k, brief="fixed", search_calls=400, popsize=10,
                                audit_rays=25, audit_limit=8), reserved_source_rays=11800,
                    audited_success=arm == "portfolio"))
    return rows


class ComparisonTests(unittest.TestCase):
    def test_shapes_do_not_inflate_independent_seed_count(self):
        result = paired_comparison(fixture(), "portfolio", "naive")
        self.assertEqual(result["paired_units"], 18)
        self.assertEqual(result["seed_blocks"], 6)
        self.assertEqual(result["one_sided_seed_sign_p"], 1/64)
        self.assertEqual(result["seed_block_bootstrap_95"], [1., 1.])

    def test_missing_control_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison(fixture()[:-1], "portfolio", "naive")

    def test_changed_physics_or_budget_is_rejected(self):
        for key in ("codec", "brief", "search_calls", "audit_limit"):
            rows = fixture()
            rows[0]["config"][key] = "changed"
            with self.assertRaises(ValueError):
                paired_comparison(rows, "portfolio", "naive")

    def test_ties_are_not_evidence_of_improvement(self):
        rows = fixture()
        for row in rows:
            row["audited_success"] = False
        result = paired_comparison(rows, "portfolio", "naive")
        self.assertEqual(result["one_sided_seed_sign_p"], 1.)
        self.assertEqual(result["seed_block_bootstrap_95"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
