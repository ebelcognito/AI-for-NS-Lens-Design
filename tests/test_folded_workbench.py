import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from nslens.workbench.folded_domain import surface_body,evaluate
from nslens.workbench.folded_tools import FoldedWorkbench
from nslens.freeform_body.trace import TraceResult,ExitRay


def task():return json.loads((Path(__file__).parent/'fixtures/folded_workbench_task.json').read_text())


class FoldedWorkbenchTests(unittest.TestCase):
 def test_surface_is_closed_and_freeform_edits_are_real(self):
  t=task();body=surface_body(t['base'],t['start'])
  for derivative in [0,1,2]:np.testing.assert_allclose(body.evaluate(0,derivative),body.evaluate(1,derivative),atol=1e-10)
  self.assertTrue(body.geometry_report(sample_count=512).valid)
  changed=t['start'].copy();changed[0]+=.2
  other=surface_body(t['base'],changed)
  self.assertGreater(np.max(np.abs(body.sample(512)[1]-other.sample(512)[1])),.001)
 def test_straight_through_energy_cannot_satisfy_folded_task(self):
  exits=tuple(ExitRay(i,(100.,0.),(1.,0.),.5,.5,()) for i in range(51))
  trace=TraceResult(exits,(),51.,0.,0.,0.)
  t=task()
  with patch('nslens.workbench.folded_domain.trace_bundle',return_value=trace):v=evaluate(t['base'],t['start'])
  self.assertFalse(v['success'])
  self.assertTrue(all(f['folded_throughput']==0 for f in v['fields']))
 def test_all_coordinate_trials_are_charged_and_requirements_locked(self):
  wb=FoldedWorkbench(task())
  wb.call('enable',dict(faces=['A'],powers=[2]))
  self.assertIn('error',wb.call('optimize',dict(max_evaluations=49,step_mm=.25)))
  self.assertEqual(wb.used,0)
  wb.call('optimize',dict(max_evaluations=6,step_mm=.25))
  self.assertEqual(wb.used,6);self.assertEqual(len(wb.records),7)
  self.assertIn('error',wb.call('perturb',dict(parameter='B_p2',delta_mm=.1)))
  for _ in range(2):self.assertIn('verification',wb.call('verify',{}))
  self.assertIn('error',wb.call('verify',{}))
  final=wb.finalize();n=len(wb.records)
  self.assertEqual(final,wb.finalize());self.assertEqual(n,len(wb.records))
  self.assertEqual(wb.records[-1]['verdict']['rays'],183)
  self.assertFalse(final['submitted'])
 def test_task_constraints_are_enforced_before_tracing(self):
  t=task();t['constraints']=dict(editable=['A_p2','A_p4'],candidate_budget=4,max_delta_mm=.1)
  wb=FoldedWorkbench(t)
  self.assertIn('error',wb.call('enable',dict(faces=['B'],powers=[2])))
  self.assertEqual(wb.enabled,[])
  wb.call('enable',dict(faces=['A'],powers=[2,4]))
  self.assertIn('error',wb.call('perturb',dict(parameter='A_p2',delta_mm=.2)))
  self.assertIn('error',wb.call('optimize',dict(max_evaluations=5,step_mm=.25)))
  self.assertEqual(wb.used,0)
  wb.call('optimize',dict(max_evaluations=4,step_mm=.25))
  self.assertEqual(wb.used,4)
  for row in wb.records:
   np.testing.assert_allclose(row['coefficients'][2:],t['start'][2:])
   self.assertLessEqual(max(abs(np.array(row['coefficients'])-t['start'])),.100000001)
  self.assertIn('error',wb.call('optimize',dict(max_evaluations=2,step_mm=.25)))
 def test_invalid_candidate_cannot_replace_best(self):
  wb=FoldedWorkbench(task());original=wb.best_x.copy()
  invalid=copy.deepcopy(wb.current);invalid.update(physical_valid=False,merit=-100)
  wb.accept(wb.x+.01,invalid)
  np.testing.assert_equal(wb.best_x,original)
 def test_pencil_profile_is_separate_and_respects_field_width(self):
  from nslens.workbench.pencil_screen import pencil_config
  from nslens.workbench.folded_domain import LAYOUT,REQUIREMENTS,rays_for
  before=copy.deepcopy(LAYOUT);requirements_before=copy.deepcopy(REQUIREMENTS)
  layout,requirements=pencil_config()
  self.assertEqual(layout['source_points'][-1][0]-layout['source_points'][0][0],1.)
  self.assertEqual(requirements['min_tir_bounces'],2)
  self.assertEqual(requirements['min_fov_deg'],0.)
  rays,count=rays_for('audit',layout)
  self.assertEqual(len(rays),3*count)
  self.assertLessEqual(max(r.origin[0] for r in rays)-min(r.origin[0] for r in rays),1.)
  self.assertEqual(LAYOUT,before);self.assertEqual(REQUIREMENTS,requirements_before)
  with self.assertRaises(ValueError):pencil_config(width_mm=2.)
  with self.assertRaises(ValueError):pencil_config(min_folds=0)
 def test_bad_constraints_rejected(self):
  for constraints in [dict(editable=['A_p2'],candidate_budget=True,max_delta_mm=1.),
                      dict(editable=['A_p2'],candidate_budget=4,max_delta_mm=float('nan')),
                      dict(editable=['invented'],candidate_budget=4,max_delta_mm=1.)]:
   t=task();t['constraints']=constraints
   with self.assertRaises(ValueError):FoldedWorkbench(t)
 def test_hidden_answers_and_changed_requirements_rejected(self):
  t=task();t['witness']=t['start']
  with self.assertRaises(ValueError):FoldedWorkbench(t)
  t=task();t['requirements']['min_tir_bounces']=0
  with self.assertRaises(ValueError):FoldedWorkbench(t)


if __name__=='__main__':unittest.main()
