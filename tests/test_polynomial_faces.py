"""Numerical geometry gates for the freeform refinement experiment."""
import unittest
import numpy as np
from nslens.freeform_body.arc_polygon import ArcPolygonBody
from nslens.freeform_body.polynomial_faces import PolynomialFaceBody
from nslens.discovery.polygon import exact_extent


class PolynomialFacesTest(unittest.TestCase):
    corners=np.array([[0.,0.],[20.,0.],[9.,30.]])
    sags=np.array([1.,.7,-.3])

    def test_zero_departure_numerical_intersections_match_circles(self):
        arc=ArcPolygonBody(self.corners,self.sags,1.6)
        ff=PolynomialFaceBody(self.corners,self.sags,1.6,np.zeros((3,3)),force_numerical=True)
        rng=np.random.default_rng(11)
        for _ in range(120):
            o=rng.uniform([-10,-10],[30,40]);d=rng.normal(size=2)
            a=arc.all_intersections(o,d);b=ff.all_intersections(o,d)
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):
                np.testing.assert_allclose(x.point,y.point,atol=2e-9)
                np.testing.assert_allclose(x.outward_normal,y.outward_normal,atol=2e-9)
        np.testing.assert_allclose(ff.extent(),exact_extent(arc),atol=1e-10)

    def test_endpoint_clamping_and_normal_derivatives(self):
        c=np.array([[.2,.1,-.12],[-.1,.2,.15],[.1,-.1,.05]])
        ff=PolynomialFaceBody(self.corners,self.sags,1.6,c)
        arc=ArcPolygonBody(self.corners,self.sags,1.6)
        for i in range(3):
            for f in [0.,1.]:
                np.testing.assert_allclose(ff._point_on(i,f),arc._point_on(i,f),atol=1e-11)
                np.testing.assert_allclose(ff._normal_on(i,f),arc._normal_on(i,f),atol=1e-11)
            for f in [.15,.5,.85]:
                h=1e-5;t=(ff._point_on(i,f+h)-ff._point_on(i,f-h))/(2*h)
                np.testing.assert_allclose(t,ff._local(i,f,1),rtol=1e-8,atol=1e-8)
                self.assertAlmostEqual(float(t@ff._normal_on(i,f)),0,places=7)

    def test_nonzero_scan_convergence_and_ray_residuals(self):
        c=np.array([[.2,.1,-.12],[-.1,.2,.15],[.1,-.1,.05]])
        a=PolynomialFaceBody(self.corners,self.sags,1.6,c,face_scan=32)
        b=PolynomialFaceBody(self.corners,self.sags,1.6,c,face_scan=256)
        rng=np.random.default_rng(19)
        for _ in range(120):
            o=rng.uniform([-10,-10],[30,40]);d=rng.normal(size=2);d/=np.linalg.norm(d)
            aa=a.all_intersections(o,d);bb=b.all_intersections(o,d)
            self.assertEqual(len(aa),len(bb))
            for x,y in zip(aa,bb):
                np.testing.assert_allclose(x.point,y.point,atol=1e-9)
                np.testing.assert_allclose(x.point,o+x.t*d,atol=1e-9)

if __name__=='__main__':unittest.main()
