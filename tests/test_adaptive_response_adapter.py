import unittest
from unittest.mock import patch
import numpy as np
from nslens.analysis import adaptive_response_adapter as adapter

class AdaptiveResponseTests(unittest.TestCase):
    def test_exact_budget_and_quadratic_response_improvement(self):
        count=[0]
        def evaluate(x,task):
            count[0]+=1;e=np.asarray(x)-.02
            return dict(loss=float(e@e),crossing=False,worst_support=1.,parity=1,ray_image_mm=(10*e).tolist(),ray_field_deg=[0.]*13,ray_good=[True]*13,ray_route=[[0,1,2]]*13)
        def local(x,task,budget,seed,sigma=.03):
            rows=[evaluate(x,task) for _ in range(budget)]
            return dict(state=x,coarse=rows[0],coarse_calls=budget)
        for budget in (1,9,28,29,65,129):
            count[0]=0
            with patch.object(adapter,'evaluate_state',side_effect=evaluate),patch.object(adapter,'local',side_effect=local):
                r=adapter.search(np.zeros(13),{'focal':75},budget,100)
            self.assertEqual(count[0],budget);self.assertEqual(r['coarse_calls'],budget)
            self.assertLessEqual(r['coarse']['loss'],13*.02**2+1e-12)
            if budget>=29:self.assertLess(r['coarse']['loss'],1e-5)
            self.assertTrue(np.all(abs(r['state'])<=1))

if __name__=='__main__':unittest.main()
