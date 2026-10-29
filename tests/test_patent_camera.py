"""Independent geometry and physical-route checks for the patent approximation."""
import unittest
import numpy as np
from nslens.patent_camera.model import PatentScene, Surface, codev_mode, meridional_zernike
from nslens.patent_camera.run import spherical_prescription, chief


class PatentCameraTest(unittest.TestCase):
    def test_spherical_intersection_and_normal(self):
        s=Surface(dict(name='test',origin_zy_mm=[0,0],tilt_deg=0,
                       kind='asphere',radius_mm=10,coefficients={}),0,1.5)
        s.limits=(-4,4);s.face_id=0;s.prepare(128)
        hit=s.intersect(np.array([-2.,3.]),np.array([1.,0.]),1e-8)
        z=10-np.sqrt(100-9)
        np.testing.assert_allclose(hit.point,[z,3],atol=1e-12)
        # Sphere's normal must be parallel to the vector from its center.
        radial=np.array(hit.point)-[10,0]
        self.assertAlmostEqual(abs(np.dot(radial/10,hit.outward_normal)),1.,places=12)

    def test_standard_zernike_convention(self):
        self.assertEqual(codev_mode(13),(4,0,'cos'))
        self.assertEqual(codev_mode(61),(10,0,'cos'))
        u=np.array([-.7,0,.3])
        np.testing.assert_allclose(meridional_zernike(3,1)(u),u)
        np.testing.assert_allclose(meridional_zernike(5,1)(u),2*u*u-1)
        np.testing.assert_allclose(meridional_zernike(13,1)(u),6*u**4-6*u*u+1)

    def test_chief_physics_and_overlap_guard(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            PatentScene().validate()
        s=PatentScene(spherical_prescription());s.validate()
        r,_=chief(s,0)
        self.assertTrue(r['route_correct'])
        self.assertTrue(r['hits_sensor'])
        self.assertEqual(sum(e['kind']=='tir' for e in r['events']),2)
        self.assertEqual(sum(e['kind']=='mirror' for e in r['events']),2)
        self.assertTrue(all(e['critical_margin_deg']>0 for e in r['events'] if e['kind']=='tir'))
        self.assertLess(abs(r['energy_closure_residual']),1e-12)
        self.assertGreater(r['dominant_branch_power_fraction'],.7)
        self.assertLess(r['dominant_branch_power_fraction'],1)

if __name__=='__main__':unittest.main()
