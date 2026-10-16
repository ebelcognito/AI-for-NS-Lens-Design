import unittest
import json
import numpy as np
from nslens.jewellery.discovery import cell,add,ranks,run_arm

def metric(x=None):
    if x is None: x=np.zeros(6)
    return dict(flash_area=.015+.01*(x[0]+1),activity=.003+.002*(x[1]+1),brightness=.08+.01*x[2],lower_brightness=.05,residual=0.,flash_count=5.,dark_fraction=.5)

class DiscoveryTests(unittest.TestCase):
    def test_invalid_and_out_of_range(self):
        for patch in [dict(brightness=.02),dict(residual=.01),dict(flash_area=.2),dict(activity=float('nan'))]:
            self.assertIsNone(cell(dict(metric(),**patch)))
    def test_elite_replacement_and_ranking(self):
        a=dict(genes=[0]*6,metrics=metric()); archive={}; self.assertTrue(add(archive,a))
        worse=dict(genes=[0]*6,metrics=dict(metric(),brightness=.06))
        new=dict(genes=[0]*6,metrics=dict(metric(),flash_area=.08))
        self.assertFalse(add(archive,worse)); self.assertEqual(ranks([worse,new],archive).tolist(),[1,0])
    def test_equal_budgets_and_archive_restart(self):
        for arm in ['random','portfolio','archive']:
            r=run_arm(arm,17,96,evaluator=metric)
            self.assertEqual(len(r['history']),96)
            json.dumps(r)
            if arm=='archive': self.assertTrue(any(x['parent_cell'] is not None for x in r['restarts']))
            again=run_arm(arm,17,96,evaluator=metric)
            self.assertEqual(r,again)
if __name__=='__main__': unittest.main()
