"""Pre-slit obstruction screens use the prescribed launch cone, not fold pose."""
import unittest
import numpy as np
from nslens.spectrometer.entrance_access import upstream_access


def plane(z, radius=30., x=0.):
    return dict(vertex=np.array([x,0.,z]),basis=np.eye(3),coeff=np.zeros(9),aperture=radius)


def layout(surfaces=()):
    return dict(surfaces=list(surfaces),input_length=60.,image=np.array([100.,0.,-10.]),image_x=np.array([1.,0.,0.]),image_y=np.array([0.,1.,0.]),image_normal=np.array([0.,0.,1.]),detector_bounds=np.array([[-20.,-20.],[20.,20.]]))


class EntranceAccessTests(unittest.TestCase):
    def test_plane_before_after_working_distance_and_downstream(self):
        for z,blocked in [(-10.,True),(-25.,False),(10.,False)]:
            with self.subTest(z=z):
                r=upstream_access(layout([plane(z)]),distance_mm=20.)
                self.assertEqual(r['blocked_rays'],r['rays'] if blocked else 0)
                self.assertEqual(r['passed'],not blocked)
        self.assertTrue(upstream_access(layout([plane(-10.,radius=.1,x=25.)]))['passed'])

    def test_finite_detector_can_block_every_ray(self):
        m=layout();m['image']=np.array([0.,0.,-10.]);r=upstream_access(m)
        self.assertEqual(r['detector_blocked_rays'],r['rays']);self.assertEqual(r['blocked_fraction'],1.)
        m['image'][0]=100.;self.assertTrue(upstream_access(m)['passed'])

    def test_newton_and_polynomial_roots_agree(self):
        m=layout([plane(-10.,radius=3.,x=1.)]);a=upstream_access(m);b=upstream_access(m,exact=True)
        self.assertGreater(a['blocked_rays'],0);self.assertLess(a['blocked_rays'],a['rays'])
        for key in ['blocked_rays','surface_blocked_rays','detector_blocked_rays']:
            self.assertEqual(a[key],b[key])
        self.assertAlmostEqual(a['nearest_obstruction_mm'],b['nearest_obstruction_mm'],places=10)

    def test_input_cone_independent_of_first_fold_position(self):
        blocker=plane(-10.,radius=3.,x=1.)
        a=upstream_access(layout([plane(3.),blocker]),input_length=60.)
        b=upstream_access(layout([plane(35.,x=15.),blocker]),input_length=60.)
        self.assertEqual(a['blocked_rays'],b['blocked_rays'])
        self.assertEqual(a['surface_blocked_rays'][1],b['surface_blocked_rays'][1])
        self.assertAlmostEqual(a['nearest_obstruction_mm'],b['nearest_obstruction_mm'])
        # The independently prescribed object-side chief-ray slope depends on L.
        c=upstream_access(layout([plane(35.),blocker]),input_length=10.)
        self.assertNotEqual(a['blocked_rays'],c['blocked_rays'])

if __name__=='__main__':unittest.main()
