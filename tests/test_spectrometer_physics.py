"""Independent analytic physics gates for the 3-D spectrometer prototype."""
import unittest
from unittest.mock import patch
import numpy as np
from nslens.spectrometer.model import sag, intersect, reflect, diffract, grating_phase_length, geometry_metrics, sag_hessian, curvature_violations
from nslens.spectrometer.physics_audit import exact_intersect


class SpectrometerPhysicsTests(unittest.TestCase):
    def test_hessian_matches_independent_gradient_finite_differences(self):
        c=np.array([.01,-.008,.0002,-.0003,2e-5,-1e-5,3e-5,1e-6,-2e-6])
        x=np.array([-4.7,.2,3.1]);y=np.array([2.3,-.8,-3.3]);h=1e-5
        value=sag_hessian(x,y,c)
        dx=(np.array(sag(x+h,y,c)[1:])-np.array(sag(x-h,y,c)[1:]))/(2*h)
        dy=(np.array(sag(x,y+h,c)[1:])-np.array(sag(x,y-h,c)[1:]))/(2*h)
        np.testing.assert_allclose(value[:,:,0],dx.T,rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(value[:,:,1],dy.T,rtol=1e-8,atol=1e-10)

    def test_curvature_gate_rejects_nonconvex_grating(self):
        surfaces=[]
        for sign in [-1.,1.,-1.]:
            c=np.zeros(9);c[:2]=sign*np.array([.01,.02])
            surfaces.append(dict(coeff=c,aperture=2.))
        with patch('nslens.spectrometer.model.decode',return_value=dict(surfaces=surfaces)):
            np.testing.assert_array_equal(curvature_violations(np.zeros(36)),0.)
            surfaces[1]['coeff'][0]=-.01
            self.assertGreater(curvature_violations(np.zeros(36)).max(),.019)

    def test_sag_normals_match_finite_differences(self):
        c = np.array([.01, -.008, .0002, -.0003, 2e-5, -1e-5, 3e-5, 1e-6, -2e-6])
        x=np.array([-.9,.2,1.1]); y=np.array([.3,-.8,1.3]); h=1e-5
        z,dx,dy=sag(x,y,c)
        np.testing.assert_allclose(dx,(sag(x+h,y,c)[0]-sag(x-h,y,c)[0])/(2*h),rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(dy,(sag(x,y+h,c)[0]-sag(x,y-h,c)[0])/(2*h),rtol=1e-8,atol=1e-10)

    def test_planar_intersection_and_aperture(self):
        surface=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)
        origins=np.array([[.3,.4,5.],[3.,0.,5.],[0.,0.,-1.]])
        dirs=np.tile([0.,0.,-1.],(3,1))
        points,normals,t,valid=intersect(origins,dirs,surface)
        np.testing.assert_array_equal(valid,[True,False,False])
        np.testing.assert_allclose(points[0],[.3,.4,0.],atol=1e-9)
        self.assertAlmostEqual(t[0],5.,places=9)
        np.testing.assert_allclose(abs(normals[0]),[0.,0.,1.],atol=1e-10)

    def test_curved_intersection_residual_and_normal(self):
        c=np.zeros(9);c[:2]=[.02,.03]
        surface=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=c,aperture=3.)
        origins=np.array([[.3,.4,5.],[-.9,.7,6.]])
        dirs=np.tile([0.,0.,-1.],(2,1))
        points,normals,t,valid=intersect(origins,dirs,surface)
        self.assertTrue(valid.all())
        z,zx,zy=sag(points[:,0],points[:,1],c)
        np.testing.assert_allclose(points[:,2],z,atol=1e-9)
        expected=np.column_stack([-zx,-zy,np.ones(2)]);expected/=np.linalg.norm(expected,axis=1)[:,None]
        np.testing.assert_allclose(np.abs(np.sum(normals*expected,axis=1)),1.,atol=1e-10)
        np.testing.assert_allclose(points,origins+t[:,None]*dirs,atol=1e-9)

    def test_reflection_is_unit_length_and_involutive(self):
        d=np.array([[.3,.4,-np.sqrt(.75)]])
        n=np.array([[0.,0.,1.]])
        out=reflect(d,n)
        np.testing.assert_allclose(out,[[.3,.4,np.sqrt(.75)]],atol=1e-12)
        np.testing.assert_allclose(reflect(out,n),d,atol=1e-12)

    def test_parabolic_focus_and_reference_sphere_wavefront(self):
        focal=10.;coeff=np.zeros(9);coeff[:2]=1/(4*focal)
        surface=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=coeff,aperture=5.)
        x,y=np.meshgrid(np.linspace(-3,3,7),np.linspace(-3,3,7))
        origins=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,20.)])
        directions=np.tile([0.,0.,-1.],(len(origins),1))
        points,normals,lengths,valid=intersect(origins,directions,surface)
        self.assertTrue(valid.all())
        outgoing=reflect(directions,normals)
        focus=np.array([0.,0.,focal])
        distance=(focal-points[:,2])/outgoing[:,2]
        image=points+distance[:,None]*outgoing
        np.testing.assert_allclose(image,np.broadcast_to(focus,image.shape),atol=1e-10)
        # Same geometric reference-sphere expression used by evaluate, with no grating.
        wavepath=lengths+np.linalg.norm(focus-points,axis=1)
        np.testing.assert_allclose(wavepath,30.,atol=1e-10)
        rms_waves=np.std(wavepath)/.0005
        self.assertLess(rms_waves,1e-8)

    def test_grating_law_and_phase_cancel_for_plane_wave(self):
        wavelength=.0008; density=300.; order=1
        d=np.array([[.1,.2,-np.sqrt(.95)]])
        n=np.array([[0.,0.,1.]]);tangent=np.array([[1.,0.,0.]])
        out,valid=diffract(d,n,tangent,wavelength,order=order,density=density)
        self.assertTrue(np.asarray(valid).all())
        np.testing.assert_allclose(out[0,:2],[.1+order*wavelength*density,.2],atol=1e-12)
        self.assertGreater(out[0,2],0.)
        np.testing.assert_allclose(np.linalg.norm(out,axis=1),1.,atol=1e-12)
        points=np.array([[-2.,.7,0.],[0.,-.4,0.],[3.,1.2,0.]])
        phase=points@d[0]+grating_phase_length(points[:,0],wavelength,order=order,density=density)-points@out[0]
        np.testing.assert_allclose(phase,0.,atol=1e-12)

    def test_grating_uses_projected_phase_gradient_without_renormalizing(self):
        n=np.array([[.6,0.,.8]]); d=-n.copy(); axis=np.array([1.,0.,0.])
        out,valid=diffract(d,n,axis,.001,density=300.)
        self.assertTrue(valid.all())
        gradient=axis-n*np.sum(axis*n,axis=-1,keepdims=True)
        tangential=out-n*np.sum(out*n,axis=-1,keepdims=True)
        np.testing.assert_allclose(tangential,.3*gradient,atol=1e-12)

    def test_polynomial_root_audit_finds_nearest_of_two_hits(self):
        c=np.zeros(9);c[0]=1.
        surface=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=c,aperture=3.)
        origins=np.array([[-2.,0.,1.],[-2.,0.,4.],[0.,0.,5.]])
        dirs=np.array([[1.,0.,0.],[1.,0.,0.],[0.,0.,-1.]])
        points,normals,t,valid=exact_intersect(origins,dirs,surface)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(t,[1.,4.,5.],atol=1e-10)
        np.testing.assert_allclose(points[:,2],points[:,0]**2,atol=1e-10)

    def test_geometry_diagnostics_report_uncertified_clearance(self):
        metrics=geometry_metrics(np.zeros(36),samples=9)
        self.assertFalse(metrics['clearance_certified'])
        self.assertGreater(metrics['sampled_aabb_volume_cm3'],0.)
        self.assertEqual(len(metrics['surfaces']),3)
        for row in metrics['surfaces']:
            self.assertGreaterEqual(row['sampled_max_abs_sag_mm'],0.)
            self.assertGreaterEqual(row['sampled_max_slope'],0.)
            self.assertGreater(row['sampled_min_abs_principal_radius_mm'],0.)

    def test_zero_order_equals_reflection_and_evanescent_is_rejected(self):
        d=np.array([[.1,0.,-np.sqrt(.99)]])
        n=np.array([[0.,0.,1.]]);tangent=np.array([[1.,0.,0.]])
        out,valid=diffract(d,n,tangent,.0008,order=0,density=300.)
        self.assertTrue(np.asarray(valid).all())
        np.testing.assert_allclose(out,reflect(d,n),atol=1e-12)
        _,valid=diffract(d,n,tangent,.001,order=1,density=2000.)
        self.assertFalse(np.asarray(valid).any())

if __name__=='__main__':
    unittest.main()
