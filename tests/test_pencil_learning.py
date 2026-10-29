import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from nslens.workbench.pencil_loop import PencilWorkbench,pencil_evaluate,observation
from nslens.workbench.pencil_screen import pencil_config
from nslens.workbench.folded_domain import PARAMETERS
from nslens.workbench.train_pencil_loop import select_tasks


def task():
    t=json.loads((Path(__file__).parent/'fixtures/folded_workbench_task.json').read_text())
    layout,req=pencil_config(min_folds=1)
    req.update(target_centroids_deg=[1.,2.,3.],centroid_tolerance_deg=.15)
    t.update(profile='pencil-angle-v1',layout=layout,requirements=req,
             constraints=dict(editable=list(PARAMETERS),candidate_budget=12,max_delta_mm=4.))
    return t


class PencilLearningTests(unittest.TestCase):
    def test_contract_rejects_changed_budget_width_and_target_tolerance(self):
        original=task();PencilWorkbench.validate_task(original)
        for key,value in [('centroid_tolerance_deg',20.),('min_tir_bounces',0)]:
            t=copy.deepcopy(original);t['requirements'][key]=value
            with self.assertRaises(ValueError):PencilWorkbench.validate_task(t)
        t=copy.deepcopy(original);t['layout']['source_points'][0][0]=40.
        with self.assertRaises(ValueError):PencilWorkbench.validate_task(t)
        t=copy.deepcopy(original);t['constraints']['candidate_budget']=48
        with self.assertRaises(ValueError):PencilWorkbench.validate_task(t)
    def test_angle_target_is_an_acceptance_gate(self):
        t=task();v=dict(success=True,physical_valid=True,merit=0.,fields=[dict(centroid_deg=x) for x in [1.,2.,3.2]])
        with patch('nslens.workbench.pencil_loop.evaluate',return_value=v):
            result=pencil_evaluate(t,t['start'],'audit')
        self.assertFalse(result['success']);self.assertGreater(result['merit'],0)
    def test_adaptive_has_coverage_and_uses_only_training_feedback(self):
        tasks=[dict(task_id='a',split='train'),dict(task_id='b',split='train')]
        previous={'a':dict(success=False),'b':dict(success=True)}
        selected,weights=select_tasks('adaptive',tasks,previous,np.random.default_rng(2))
        np.testing.assert_allclose(weights,[.875,.125]);self.assertEqual(len(selected),4)
        tasks[0]['split']='test'
        with self.assertRaises(ValueError):select_tasks('adaptive',tasks,previous,np.random.default_rng(2))
    def test_policy_prompt_has_no_split_or_family_identifiers(self):
        public=observation(task(),{})['task']
        self.assertFalse({'task_id','family_id','split','witness'}&set(public))

if __name__=='__main__':unittest.main()
