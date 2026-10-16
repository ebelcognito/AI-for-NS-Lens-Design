import unittest
import numpy as np
from nslens.three_mirror.refine3d import residual_vector
from nslens.three_mirror import model3d as m
from tests.test_three_mirror_3d import sample

class PerRayResidualTests(unittest.TestCase):
 def test_objective_is_spot_variance_plus_mapping_anchor(self):
  rng=np.random.default_rng(418);xy=rng.normal(size=(5,13,2));target=rng.normal(size=(5,2));w=3.
  r=residual_vector(xy,target,w);cent=xy.mean(axis=1)
  expected=np.mean(np.sum((xy-cent[:,None])**2,axis=-1))+w*w*np.mean(np.sum((cent-target)**2,axis=-1))
  self.assertAlmostEqual(r@r,expected,places=12)
 def test_coordinate_gradient_matches_independent_finite_difference(self):
  rng=np.random.default_rng(19);xy=rng.normal(size=(3,11,2));target=rng.normal(size=(3,2));direction=rng.normal(size=xy.shape);w=3.
  center=xy.mean(axis=1);analytic=2*(xy-center[:,None])/33+2*w*w*(center-target)[:,None]/33
  def cost(a):r=residual_vector(a,target,w);return r@r
  fd=(cost(xy+1e-6*direction)-cost(xy-1e-6*direction))/2e-6
  self.assertAlmostEqual(fd,float(np.sum(analytic*direction)),places=7)
 def test_ray_jacobian_direction_is_stable_on_fixed_route(self):
  from copy import deepcopy
  layout=m.lift_sample(sample());fields=m.field_grid((2,1),(3,3));p=m.disk_pupil(15,7);o,d=m.launch_rays(layout,fields,p)
  ref=m.trace_rays(layout,o,d);self.assertTrue(ref['good'].all());target=ref['image_mm'].reshape(9,len(p),2).mean(axis=1)
  delta=np.random.default_rng(12).normal(size=(3,12))
  indices=[k for k,power in enumerate(m.POWERS) if sum(power)<=4]
  def f(h):
   changed=deepcopy(layout)
   for j in range(3):changed['surfaces'][j]['coefficients'][indices]+=h*delta[j]
   r=m.trace_rays(changed,o,d);np.testing.assert_array_equal(r['route'],ref['route']);self.assertTrue(r['good'].all())
   return residual_vector(r['image_mm'].reshape(9,len(p),2),target)
  a=(f(1e-5)-f(-1e-5))/2e-5;b=(f(5e-6)-f(-5e-6))/1e-5
  np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-5)
if __name__=='__main__':unittest.main()
