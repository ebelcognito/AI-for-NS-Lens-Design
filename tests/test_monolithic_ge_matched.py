import unittest
import numpy as np
from nslens.monolithic_ir import model_ge_matched as m

class MatchedGeTests(unittest.TestCase):
    def test_specs(self):
        np.testing.assert_allclose(m.index(np.array([.008,.012])),[4.0067,4.0029],atol=1e-12)
        self.assertAlmostEqual(m.FOCAL_MM/(2*m.PUPIL_RADIUS),1.4)
        for g in [np.zeros(35),np.random.default_rng(3).uniform(-.3,.3,35)]:
            d=m.decode(g)
            self.assertAlmostEqual(sum(np.linalg.norm(d.surfaces[i+1].center-d.surfaces[i].center) for i in range(3)),58)
            self.assertEqual([s.radial is None for s in d.surfaces],[False,True,False,False])

    def test_gradients_and_rotational_aspheres(self):
        d=m.decode(np.random.default_rng(4).uniform(-.2,.2,35))
        for s in d.surfaces:
            x,y,h=s.center[0]+.6,.4,1e-5
            z,dx,dy=m.sag(s,x,y)
            self.assertAlmostEqual(dx,(m.sag(s,x+h,y)[0]-m.sag(s,x-h,y)[0])/(2*h),delta=1e-6)
            self.assertAlmostEqual(dy,(m.sag(s,x,y+h)[0]-m.sag(s,x,y-h)[0])/(2*h),delta=1e-6)
            if s.radial is not None:
                c=1/np.sqrt(1+s.tilt**2);sn=c*s.tilt
                u=c*(x-s.center[0])+sn*(z-s.center[2]);w=c*(z-s.center[2])-sn*(x-s.center[0]);r2=u*u+y*y
                A,B,C=s.radial
                self.assertAlmostEqual(w,A*r2+B*r2*r2+C*r2**3,places=8)

    def test_sampling_and_merit(self):
        g=np.zeros(35);t=m.trace(g,fields=(5,3),waves=3,pupils=2)
        self.assertEqual(t['good'].shape,(15,3,16))
        np.testing.assert_allclose(t['wavelengths_mm'],[.008,.010,.012])
        r=m.residuals(g,fields=(5,3),pupils=2)
        self.assertTrue(np.isfinite(r).all())
        self.assertAlmostEqual(float(r@r),m.evaluate(g,fields=(5,3),pupils=2)['loss'])

if __name__=='__main__':unittest.main()
