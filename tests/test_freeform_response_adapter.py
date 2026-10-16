import unittest
import numpy as np
from nslens.analysis.freeform_response_adapter import propose_response

class ResponseTests(unittest.TestCase):
    def test_linear_response_reduces_error_and_accounts_all_probes(self):
        x=np.zeros(13);calls=[]
        def evaluate(v):
            calls.append(v.copy());return dict(parity=1,ray_good=[True]*13,ray_field_deg=[0.]*13,ray_image_mm=(10*(v-.01)).tolist(),ray_route=[[0,1,2]]*13)
        parent=evaluate(x);calls.clear()
        ps,probes,diag=propose_response(x,parent,{'focal':75.},evaluate)
        self.assertEqual(len(calls),26);self.assertEqual(len(probes),26)
        self.assertEqual(diag['common_rays'],13)
        self.assertLess(np.linalg.norm(ps[-1]-.01),np.linalg.norm(x-.01))
        self.assertTrue(np.all(abs(ps)<=1))
        self.assertTrue(np.all(np.linalg.norm(ps-x,axis=1)<=.15+1e-12))
    def test_route_changes_remove_constraints(self):
        x=np.zeros(13)
        parent=dict(parity=1,ray_good=[True],ray_field_deg=[0.],ray_image_mm=[1.],ray_route=[[0,1,2]])
        def evaluate(v):return dict(parent,ray_route=[[2,1,0]])
        ps,_,diag=propose_response(x,parent,{'focal':75.},evaluate)
        self.assertEqual(diag['common_rays'],0)
        np.testing.assert_array_equal(ps,np.zeros((4,13)))

if __name__=='__main__':unittest.main()
