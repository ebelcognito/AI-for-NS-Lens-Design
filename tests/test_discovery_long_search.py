"""Wall-clock stopping preserves results and the declared optical task grid."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nslens.discovery.benchmark import POLICIES, run_job
from nslens.discovery.folded import RoutingBrief
from nslens.discovery.long_search import task_grid
from nslens.discovery.polygon import PolygonCodec


class LongSearchTests(unittest.TestCase):
    def test_expired_deadline_starts_no_population_and_saves_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = dict(out=str(Path(tmp)/'run'), codec=asdict(PolygonCodec(3)),
                       brief=asdict(RoutingBrief()), policy=POLICIES['portfolio'],
                       arm='portfolio', seed=1, unit='test', brief_id='base',
                       search_calls=400, popsize=10, audit_limit=8, audit_rays=25,
                       deadline_unix=0.)
            with patch('nslens.discovery.benchmark.evaluate') as evaluate:
                row = run_job(job)
            evaluate.assert_not_called()
            self.assertEqual(row['stop_reason'], 'deadline')
            self.assertEqual(row['search_calls'], 0)
            self.assertTrue((Path(job['out'])/'summary.json').exists())

    def test_task_grid_preserves_acceptance_and_fixed_receivers(self):
        tasks = task_grid()
        self.assertEqual(len(tasks), 216)
        self.assertEqual(len({t['brief_id'] for t in tasks}), 216)
        for task in tasks:
            brief = RoutingBrief(**task['brief'])
            PolygonCodec(**task['codec'])
            self.assertEqual(brief.min_tir, RoutingBrief().min_tir)
            self.assertEqual(brief.min_folded_throughput, RoutingBrief().min_folded_throughput)
            self.assertEqual(brief.min_root_fraction, RoutingBrief().min_root_fraction)
            if task['brief_id'].startswith('right_1p5'):
                self.assertEqual(brief.detector()[3]*2, brief.beam_width_mm*1.5)
            if task['brief_id'].startswith('bottom'):
                self.assertEqual(brief.detector()[:3], (1, -55., 60.))

    def test_task_grid_caps_all_detector_lengths_and_enables_spot_objective(self):
        for task in task_grid(max_detector_beam_ratio=2., spot_weight=1.):
            brief = RoutingBrief(**task['brief'])
            self.assertLessEqual(2*brief.detector()[3], 2*brief.beam_width_mm)
            self.assertEqual(brief.spot_weight, 1.)


if __name__ == '__main__':
    unittest.main()
