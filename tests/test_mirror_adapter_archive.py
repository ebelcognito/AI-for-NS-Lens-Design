import unittest
from nslens.analysis.train_archive_adapter import families

class ArchiveFamilies(unittest.TestCase):
    def test_multiple_occurrences_union_ancestry(self):
        row={'occurrences':[{'source':'experiments/potential_trajectories_fresh_20261005/field__source_reuse__3__4102.json'}, {'source':'experiments/shared_freeform_comparison_20261006/field8__reuse_only__6__r3__last_freeform.json'}]}
        self.assertEqual(families(row),{2,6})

    def test_unresolved_summary_is_not_a_family(self):
        self.assertEqual(families({'occurrences':[{'source':'experiments/a/summary.json'}]}),set())

    def test_direct_donor_mapping(self):
        self.assertEqual(families({'occurrences':[{'source':'experiments/a/cross_count_3107.json'}]}),{7})
