import copy,unittest
import numpy as np
from nslens.workbench.optical_capability_map import validate_edits,validate_cma,rank,agreement,random_action,BUDGET

class CapabilityMapTests(unittest.TestCase):
 def test_edit_contract_bounds_and_budget(self):
  action=dict(edits=[dict(parameters=[0],deltas=[.02]) for _ in range(BUDGET)],rationale='test')
  validate_edits(action,14)
  for mutation in ['count','id','nan','step','duplicate','extra']:
   a=copy.deepcopy(action)
   if mutation=='count':a['edits'].pop()
   elif mutation=='id':a['edits'][0]['parameters']=[14]
   elif mutation=='nan':a['edits'][0]['deltas']=[float('nan')]
   elif mutation=='step':a['edits'][0]['deltas']=[.11]
   elif mutation=='duplicate':a['edits'][0]=dict(parameters=[0,0],deltas=[.01,.01])
   else:a['detector_position']=0
   with self.assertRaises(ValueError):validate_edits(a,14)
 def test_cma_contract(self):
  validate_cma(dict(parameters=[0,4,13],sigma=.025,rationale='test'),14)
  for a in [dict(parameters=[],sigma=.025,rationale=''),dict(parameters=[0],sigma=float('inf'),rationale=''),dict(parameters=[True],sigma=.025,rationale=''),dict(parameters=[0],sigma=.2,rationale='')]:
   with self.assertRaises(ValueError):validate_cma(a,14)
 def test_random_baseline_reproducible_and_valid(self):
  task=dict(start=[0.]*14)
  self.assertEqual(random_action(task,1),random_action(task,1))
  self.assertNotEqual(random_action(task,1),random_action(task,2))
  validate_edits(random_action(task,1),14)
 def test_bad_energy_ledger_cannot_outrank_valid_physics(self):
  a=dict(status='measured',feasible=False,violations=dict(closure=0.,spot_rms=2.),worst_folded_throughput=.9,support=1.,worst_spot_rms=1.5)
  b={**a,'violations':dict(closure=.01,spot_rms=0.),'worst_spot_rms':.1}
  self.assertGreater(rank(a),rank(b))
 def test_capacity_check_detects_per_field_change(self):
  row=dict(status='measured',worst_folded_throughput=.8,worst_spot_rms=.2,support=1.,closure=0.,unresolved=0.,pruned=0.,fields=[dict(folded_throughput=.8,spot_rms=.2,root_fraction=1.,detector_centroid=2.)])
  other=copy.deepcopy(row);self.assertTrue(agreement(row,other));other['fields'][0]['detector_centroid']+=.1;self.assertFalse(agreement(row,other))
if __name__=='__main__':unittest.main()
