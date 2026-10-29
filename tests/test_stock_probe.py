import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from nslens.workbench.stock_probe import validate,indices,rank,dump,read,initialize,step,finalize


def fake_measure(task, genes, probe, **kw):
 quality=float(1-np.mean(np.asarray(genes)**2))
 return dict(genes=list(genes),status='measured',feasible=True,routing_feasible=True,quality=quality,support=1.,
             violations={},package_violation=0.,dimensions=[10.,10.],source_rays=909 if kw.get('audit') else 225)

class StockProbeTests(unittest.TestCase):
 def test_reject_unbounded_actions(self):
  action=dict(probe='repair',group='all',sigma=.03,parent='best',reason='test')
  validate(action)
  for change in [dict(sigma=float('nan')),dict(sigma=True),dict(sigma=.5),dict(group='shell'),dict(command='x')]:
   with self.assertRaises(ValueError):validate(action|change)

 def test_package_failure_cannot_beat_joint_feasibility(self):
  good=fake_measure({},[.5],'repair')
  bad=good|dict(feasible=False,package_violation=.1,quality=1.)
  self.assertGreater(rank(good),rank(bad))

 def test_indices_cover_actual_codec(self):
  for k in [3,4,5]:
   parts=[indices(k,g).tolist() for g in ['shape','curvature','placement_size']]
   self.assertEqual(sorted(sum(parts,[])),list(range(5+3*k)))

 @patch('nslens.workbench.stock_probe.measure',side_effect=fake_measure)
 def test_budget_checkpoint_and_audit_isolation(self,_):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);genes=[.5]*14
   dump(folder/'task.json',dict(task_id='test',label='bad',seed=4,design=dict(genes=genes,corners=3)))
   initialize(folder)
   with self.assertRaises(ValueError):finalize(folder)
   action=dict(probe='repair',group='curvature',sigma=.03,parent='best',reason='test')
   step(folder,action)
   state=read(folder/'state.json')
   self.assertEqual(len(state['records']),43)
   for r in state['records'][3:]:self.assertEqual(r['result']['genes'][:11],genes[:11])
   step(folder,action)
   with self.assertRaises(ValueError):step(folder,action)
   for probe in ['compact','field']:
    for _ in range(2):step(folder,action|dict(probe=probe))
   before=read(folder/'state.json')
   self.assertFalse(any(r['stage'].startswith('audit') for r in before['records']))
   outcome=finalize(folder)
   self.assertEqual(outcome['budget']['search'],dict(evaluations=240,source_rays=54000))
   self.assertTrue(all(v['replay_exact'] and v['capacity_agrees'] for v in outcome['outcomes'].values()))
   self.assertEqual(finalize(folder),outcome)
   with self.assertRaises(ValueError):step(folder,action)

if __name__=='__main__':unittest.main()
