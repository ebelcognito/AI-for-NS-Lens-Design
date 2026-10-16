import unittest
import numpy as np
from nslens.three_mirror import model3d as m

class OpaqueDetectorTests(unittest.TestCase):
    def layout(self):
        return dict(surfaces=[dict(vertex=np.array([0.,0.,z]),basis=np.eye(3),radius=20.,coefficients=np.zeros(len(m.POWERS))) for z in [-10.,-20.,-30.]],detector=dict(vertex=np.zeros(3),basis=np.eye(3),bounds_mm=np.array([[-1.,-1.],[1.,1.]])))
    def test_back_face_blocks_before_mirror(self):
        r=m.trace_rays(self.layout(),np.array([[0.,0.,1.]]),np.array([[0.,0.,-1.]]))
        self.assertEqual(r['hits'][0],0)
        self.assertFalse(r['good'][0])
        self.assertTrue(np.isnan(r['image_mm'][0]).all())
    def test_outside_rectangle_does_not_block(self):
        r=m.trace_rays(self.layout(),np.array([[2.,0.,1.]]),np.array([[0.,0.,-1.]]))
        self.assertGreater(r['hits'][0],0)
    def test_front_face_records_but_does_not_accept_incomplete_route(self):
        r=m.trace_rays(self.layout(),np.array([[0.,0.,-1.]]),np.array([[0.,0.,1.]]))
        np.testing.assert_allclose(r['image_mm'][0],[0.,0.])
        self.assertFalse(r['good'][0])
