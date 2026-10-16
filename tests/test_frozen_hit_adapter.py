import unittest
import numpy as np
from nslens.analysis.frozen_hit_adapter import propose_slopes

class FrozenHitTests(unittest.TestCase):
    def test_preserves_layout_and_bounds_nonzero_freeform(self):
        x=np.array([-.8,.9,-.9,0,0,0,.01,.01,.01,.1,-.2,.3,-.1])
        proposals,info=propose_slopes(x,dict(focal=75.,aperture=35.,half_field_deg=8.))
        self.assertEqual(proposals.shape,(4,13));self.assertEqual(info['observation_calls'],1)
        np.testing.assert_array_equal(proposals[:,:8],np.tile(x[:8],(4,1)))
        self.assertTrue(np.all(abs(proposals)<=1))

if __name__=='__main__':unittest.main()
