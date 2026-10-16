"""Analytic checks for the exploratory mirror tracer."""
import unittest
import numpy as np
from nslens.three_mirror.pilot import intersections, decode, trace


class MirrorTests(unittest.TestCase):
    def test_plane_forward_and_aperture(self):
        m=dict(vertices=np.array([[0.,0.]]),normals=np.array([[1.,0.]]),
               tangents=np.array([[0.,1.]]),curves=np.array([0.]),widths=np.array([2.]))
        o=np.array([[-5.,1.],[-5.,3.],[5.,0.]])
        d=np.tile([1.,0.],(3,1))
        hit=intersections(o,d,m)[:,0]
        self.assertAlmostEqual(hit[0],5.)
        self.assertTrue(np.isinf(hit[1:]).all())

    def test_parabola_focus(self):
        # x=y^2/(4f), parallel rays from +x reflect through (f,0).
        f=20.; y=np.linspace(-10,10,21)
        m=dict(vertices=np.array([[0.,0.]]),normals=np.array([[1.,0.]]),
               tangents=np.array([[0.,1.]]),curves=np.array([1/(4*f)]),widths=np.array([11.]))
        o=np.column_stack([np.full(len(y),30.),y]); d=np.tile([-1.,0.],(len(y),1))
        t=intersections(o,d,m)[:,0]; point=o+t[:,None]*d
        np.testing.assert_allclose(point[:,0],y*y/(4*f),atol=1e-12)
        n=np.column_stack([np.ones(len(y)),-y/(2*f)]); n/=np.linalg.norm(n,axis=1)[:,None]
        reflected=d-2*np.sum(d*n,axis=1)[:,None]*n
        at_focus=point+((f-point[:,0])/reflected[:,0])[:,None]*reflected
        np.testing.assert_allclose(at_focus[:,1],0.,atol=1e-12)

    def test_scaffold_reflection(self):
        m=decode(np.linspace(-.5,.5,12)); incoming=np.array([1.,0.])
        for i in range(3):
            outgoing=incoming-2*np.dot(incoming,m['normals'][i])*m['normals'][i]
            next_point=m['vertices'][i+1] if i<2 else m['image']
            np.testing.assert_allclose(m['vertices'][i]+m['lengths'][i]*outgoing,next_point,atol=1e-12)
            incoming=outgoing

    def test_lost_rays_penalized(self):
        r=trace(np.zeros(12))
        self.assertTrue(np.isfinite(r['loss']))
        self.assertGreaterEqual(r['loss'],0.)
        self.assertTrue(0<=r['worst_support']<=1)

if __name__=='__main__': unittest.main()
