"""Independent geometric check of the new packaging constraint."""
import unittest
import numpy as np
from nslens.analysis.hard_optical_environments import envelope, TASKS
from nslens.analysis.variable_mirror import decode

class EnvelopeTests(unittest.TestCase):
 def test_extrema_against_dense_surface_sampling(self):
  rng=np.random.default_rng(93);task=TASKS['combined']
  for n in range(3,7):
   for _ in range(10):
    g=rng.uniform(-1,1,3*n);m=decode(g,n);u=np.linspace(-30,30,10001)
    pts=[v+u[:,None]*t+(c*u*u)[:,None]*normal for v,t,normal,c in zip(m['vertices'],m['tangents'],m['normals'],m['curves'])]
    half=task['focal']*np.tan(np.deg2rad(task['half_field_deg']))+1
    pts.append(np.array([m['image']-half*m['image_tangent'],m['image']+half*m['image_tangent']]))
    sampled=np.ptp(np.concatenate(pts),axis=0);exact=envelope(g,n,task)
    self.assertTrue(np.all(exact>=sampled-1e-10))
    np.testing.assert_allclose(exact,sampled,atol=1e-5,rtol=0)


class RandomDirectionTests(unittest.TestCase):
 def test_exact_norm_group_and_bounds(self):
  from nslens.analysis.hard_optical_environments import random_direction
  rng=np.random.default_rng(77)
  for d in [9,12,15,18]:
   for _ in range(100):
    x=rng.uniform(-1,1,d);x[:3]=[1.,-1.,.999]
    group=np.arange(d//3,d);norm=rng.uniform(.00001,.1)
    step=random_direction(x,group,norm,rng)
    self.assertTrue(np.all(np.abs(x+step)<=1+1e-12))
    self.assertAlmostEqual(np.linalg.norm(step),norm,places=12)
    np.testing.assert_array_equal(step[:d//3],0)

if __name__=='__main__':unittest.main()
