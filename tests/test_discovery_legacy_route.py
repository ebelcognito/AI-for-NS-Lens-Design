import unittest
from nslens.discovery.benchmark import legacy_route_rank


class LegacyRouteTests(unittest.TestCase):
    def test_new_beats_improved_and_unchanged(self):
        def row(q):
            return dict(feasible=True, quality=q)
        self.assertGreater(legacy_route_rank(row(.3), None), legacy_route_rank(row(.9), .4))
        self.assertGreater(legacy_route_rank(row(.5), .4), legacy_route_rank(row(.9), .95))

    def test_improvement_ranks_by_delta_and_failures_flatten(self):
        self.assertGreater(legacy_route_rank(dict(feasible=True, quality=.6), .3),
                           legacy_route_rank(dict(feasible=True, quality=.9), .8))
        self.assertEqual(legacy_route_rank(dict(feasible=False, quality=.9), None), (0, 0.))
