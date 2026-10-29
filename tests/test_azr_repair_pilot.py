import unittest
from unittest.mock import patch
import numpy as np

from nslens.analysis.azr_repair_pilot import measure, public_observation, reward, indices


class PilotTests(unittest.TestCase):
    def test_solver_observation_hides_witness_and_generator(self):
        task=dict(config={},start=[.1],initial={},witness=[.2],style=3,
                  task_id='secret',family_id='secret',split='test',initial_audit={'secret':1})
        public=public_observation(task)
        self.assertFalse(set(public)&{'witness','style','task_id','family_id','split','initial_audit'})

    def test_material_and_stricter_requirements_preserved(self):
        config=dict(codec=dict(corners=3,n=1.5), brief=dict(min_folded_throughput=.9,
                    min_root_fraction=.85,fields_deg=[-2,0,2],min_tir=3))
        result=dict(feasible=False,status='measured',fields=[dict(folded_throughput=.7)],
                    support=.8,violations={},source_rays=125)
        with patch('nslens.analysis.azr_repair_pilot.evaluate',return_value=result) as evaluator:
            value=measure(config,np.zeros(14),'audit')
        codec,_,brief=evaluator.call_args.args
        self.assertEqual(codec.n,1.5)
        self.assertEqual(brief.min_folded_throughput,.9)
        self.assertEqual(brief.min_root_fraction,.85)
        self.assertEqual(brief.min_tir,3)
        self.assertEqual((min(brief.fields_deg),max(brief.fields_deg)),(-2,2))
        self.assertFalse(value['success'])

    def test_invalid_physics_cannot_get_reward(self):
        r=dict(measured=True,success=True,throughput=1.,support=1.,violations={'unresolved':.01})
        self.assertEqual(reward(r),0.)

    def test_solver_parameter_partition(self):
        for k in (3,4,5):
            a=set(indices(k,'curvature'));b=set(indices(k,'shape_pose'))
            self.assertFalse(a&b)
            self.assertEqual(a|b,set(indices(k,'all')))


if __name__=='__main__':unittest.main()
