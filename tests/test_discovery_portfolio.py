"""Stepping-stone retention, comparable selection, and source-ray budget tests."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nslens.discovery.benchmark import POLICIES, run_job, selected_candidates
from nslens.discovery.folded import RoutingBrief
from nslens.discovery.polygon import PolygonCodec
from nslens.discovery.portfolio import RestartPool, constraint_rank, select_diverse


def candidate(x, violations=(1., .8), feasible=False, route=None):
    return dict(genes=[x, 0.], status="measured", feasible=feasible,
                violations=dict(enumerate(violations)), progress=(0., 0., 1.),
                quality=.5 if feasible else 0., support=.8 if feasible else 0., route=route)


class PortfolioTests(unittest.TestCase):
    def test_second_worst_violation_breaks_worst_violation_plateau(self):
        a, b = candidate(0, (1., .8)), candidate(.1, (1., .2))
        self.assertGreater(constraint_rank(b), constraint_rank(a))
        self.assertGreater(constraint_rank(candidate(0, feasible=True)), constraint_rank(b))

    def test_pool_retains_failed_candidates_and_rotates_starts(self):
        pool = RestartPool(capacity=2)
        pool.update([candidate(-.8), candidate(.8), candidate(0)])
        self.assertEqual(len(pool.rows), 2)
        first, second = pool.start(), pool.start()
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(all(not row["feasible"] for row in pool.rows))

    def test_duplicate_candidates_do_not_inflate_diversity(self):
        rows = select_diverse([candidate(0)]*20+[candidate(.9)], 4)
        self.assertEqual(len(rows), 2)

    def test_common_audit_selector_covers_routes_before_alternatives(self):
        rows = [candidate(x, feasible=True, route="A") for x in (0,.2,.4)]
        rows.append(candidate(.6, feasible=True, route="B"))
        selected = selected_candidates(rows, 2)
        self.assertEqual({r["route"] for r in selected}, {"A", "B"})

    def test_every_arm_obeys_same_ray_reservation_with_invalid_geometry(self):
        codec, brief = PolygonCodec(3), RoutingBrief()
        for arm in ("naive", "portfolio", "random"):
            counter = [0]
            def fake(codec, genes, brief, **kwargs):
                counter[0] += 1
                valid = counter[0] % 2 == 0
                row = candidate(0)
                row.update(genes=list(genes), source_rays=27 if valid else 0,
                           status="measured" if valid else "geometry")
                return row
            with tempfile.TemporaryDirectory() as tmp:
                job = dict(out=str(Path(tmp)/"run"), codec=asdict(codec), brief=asdict(brief),
                           policy=POLICIES[arm], arm=arm, seed=1, unit="test", brief_id="base",
                           search_calls=3, popsize=10, audit_limit=8, audit_rays=25)
                with patch("nslens.discovery.benchmark.evaluate", side_effect=fake):
                    row = run_job(job)
                self.assertEqual(row["search_source_rays"], 81)
                self.assertEqual(row["proposals"], 6)
                self.assertEqual(row["audit_source_rays"], 0)
                self.assertEqual(row["reserved_source_rays"], 81+8*125)


if __name__ == "__main__":
    unittest.main()
