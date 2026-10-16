import unittest
import numpy as np
from nslens.jewellery.model import interface, intersect, geometry, render, rotation, unit

class JewelleryPhysics(unittest.TestCase):
    def test_normal_fresnel(self):
        _,t,r,tir=interface(np.array([[0,0,-1.]]),np.array([[0,0,1.]]),1.,1.6)
        self.assertAlmostEqual(r[0],(.6/2.6)**2,14)
        np.testing.assert_allclose(t,[[0,0,-1.]])
        self.assertFalse(tir[0])

    def test_snell_and_critical(self):
        a=np.deg2rad([20,38,40,60])
        d=np.column_stack([np.sin(a),np.zeros(4),-np.cos(a)])
        _,t,r,tir=interface(d,np.tile([0,0,1.],(4,1)),1.6,1.)
        np.testing.assert_array_equal(tir,[False,False,True,True])
        np.testing.assert_allclose(t[:2,0],1.6*np.sin(a[:2]))
        np.testing.assert_allclose(r[2:],1.)

    def test_slab_total_and_reciprocity(self):
        d=unit([[.3,0,-1.]])
        _,inside,r,_=interface(d,[[0,0,1]],1.,1.6)
        _,exit_d,r2,_=interface(inside,[[0,0,1]],1.6,1.)
        np.testing.assert_allclose(exit_d,d,atol=1e-14)
        np.testing.assert_allclose(r,r2,atol=1e-14)
        # At normal incidence scalar polarization averaging is exact.
        R=(.6/2.6)**2
        transmission=sum((1-R)**2*R**(2*k) for k in range(30))
        self.assertAlmostEqual(transmission,(1-R)/(1+R),14)

    def test_intersection_rotation(self):
        n,h=geometry(np.zeros(6)); o=np.array([[0,0,15.],[20,0,15.]])
        d=np.tile([0,0,-1.],(2,1)); t,_,valid=intersect(o,d,n,h)
        q=rotation(23,39)
        rt,_,rv=intersect(o@q,d@q,n@q,h)
        np.testing.assert_array_equal(valid,[True,False]); np.testing.assert_array_equal(valid,rv)
        np.testing.assert_allclose(t,rt)

    def test_cube_entry_exit_parallel_and_edge(self):
        normals=np.vstack([np.eye(3),-np.eye(3)])
        offsets=np.ones(6)
        origins=np.array([[0,0,3.],[0,0,0.],[2,0,3.],[1,1,3.]])
        directions=np.tile([0,0,-1.],(4,1))
        t,n,valid=intersect(origins,directions,normals,offsets)
        np.testing.assert_array_equal(valid,[True,True,False,True])
        np.testing.assert_allclose(t[[0,1,3]],[2,1,2])
        np.testing.assert_allclose(n[:2],[[0,0,1],[0,0,-1]])

    def test_interface_rotates_with_scene(self):
        d=unit([[.2,.3,-1.]])
        n=np.array([[0.,0.,1.]])
        q=rotation(32,-17)
        r,t,w,tir=interface(d,n,1.,1.6)
        rr,tt,ww,ttir=interface(d@q,n@q,1.,1.6)
        np.testing.assert_allclose(rr,r@q,atol=1e-14)
        np.testing.assert_allclose(tt,t@q,atol=1e-14)
        np.testing.assert_allclose(ww,w,atol=1e-14)
        np.testing.assert_array_equal(ttir,tir)

    def test_random_extreme_shapes_preserve_ledger(self):
        rng=np.random.default_rng(202)
        for _ in range(24):
            g=rng.uniform(-1,1,6)
            im,m,a=render(g,12,(15,0),'constant',bounces=48)
            self.assertTrue(np.isfinite(im).all())
            self.assertLess(a['closure_error'],1e-12)
            self.assertLessEqual(float(im.max()),1+1e-12)
            self.assertAlmostEqual(float(im[m].mean())+a['residual'],1.,places=12)

    def test_white_environment_and_closure(self):
        for g in [np.zeros(6),np.array([.4,-.3,.2,.5,-.4,.6])]:
            im,m,a=render(g,24,(11,7),'constant',bounces=100)
            self.assertLess(a['closure_error'],1e-12)
            self.assertLess(a['residual'],1e-5)
            self.assertLess(np.mean(abs(im[m]-1)),1e-5)

if __name__=='__main__': unittest.main()
