"""Ensure widened tasks cannot silently use legacy angles or spend on other probes."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from nslens.workbench.stock_probe import dump, read, initialize, step, finalize, measure
from test_stock_probe import fake_measure

class AngularProbeTests(unittest.TestCase):
    @patch('nslens.workbench.stock_probe.measure', side_effect=fake_measure)
    def test_field_only_budget(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            dump(p/'task.json',dict(task_id='angular',label='test',seed=2,design=dict(genes=[.5]*14,corners=3),
                 batch_limits={'field':6},full_fov_deg={'field':12},instructions='Widen to 12 degrees.'))
            initialize(p)
            self.assertEqual(read(p/'request.json')['remaining_batches'],{'field':6})
            action=dict(probe='field',group='curvature',sigma=.03,parent='best',reason='test')
            with self.assertRaises(ValueError):step(p,action|{'probe':'repair'})
            for _ in range(6):step(p,action)
            with self.assertRaises(ValueError):step(p,action)
            self.assertEqual(finalize(p)['budget']['search']['evaluations'],240)

    def test_target_angles_reach_evaluator(self):
        from dataclasses import asdict
        from nslens.discovery.folded import RoutingBrief
        from nslens.discovery.polygon import PolygonCodec
        d=dict(codec=asdict(PolygonCodec(corners=3)),
               brief=asdict(RoutingBrief(beam_width_mm=6., detector_half_length=7.5)))
        task=dict(design=d,full_fov_deg={'field':12},initial_dimensions=[100,100])
        with patch('nslens.workbench.stock_probe.evaluate',return_value=dict(feasible=False)) as evaluator:
            measure(task,[0.]*14,'field')
        brief=evaluator.call_args.args[2]
        self.assertEqual((brief.fields_deg[0],brief.fields_deg[-1]),(-6.,6.))
        self.assertEqual(brief.beam_width_mm,6.)
        self.assertEqual(brief.detector_half_length,d['brief']['detector_half_length'])

if __name__=='__main__':unittest.main()
