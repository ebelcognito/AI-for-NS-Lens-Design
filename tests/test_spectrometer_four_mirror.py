"""Independent regression checks for folding and its audit machinery."""
import unittest
import numpy as np
from nslens.spectrometer import model
from nslens.spectrometer.four_mirror_audit import audit_paths, dense_pupil, sphere_separation
from nslens.spectrometer.physics_audit import exact_intersect


class FoldAuditTests(unittest.TestCase):
    def test_improper_isometry_preserves_sag_grating_and_phase(self):
        # A reflection has negative determinant; do not rebuild basis with cross products.
        normal=model.unit(np.array([.4,.7,.3])); q=np.eye(3)-2*np.outer(normal,normal)
        shift=np.array([7.,-3.,2.]); coeff=np.zeros(9);coeff[:2]=[.01,.02]
        surf=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=coeff,aperture=5.)
        folded=dict(surf,vertex=shift,basis=q)
        origins=np.array([[.1,.4,4.],[-1.,.2,4.]])
        direction=model.unit(np.array([[.03,.01,-1.],[-.02,.04,-1.]]))
        p,n,t,good=exact_intersect(origins,direction,surf)
        pp,nn,tt,gg=exact_intersect(origins@q.T+shift,direction@q.T,folded)
        self.assertTrue(good.all() and gg.all());self.assertLess(np.linalg.det(folded['basis']),0.)
        np.testing.assert_allclose(pp,p@q.T+shift,atol=1e-12)
        np.testing.assert_allclose(tt,t,atol=1e-12)
        d,_=model.diffract(direction,n,surf['basis'][:,0],.0008)
        dd,_=model.diffract(direction@q.T,nn,folded['basis'][:,0],.0008)
        np.testing.assert_allclose(dd,d@q.T,atol=1e-12)
        np.testing.assert_allclose(model.grating_phase_length(((pp-shift)@q)[:,0],.0008),model.grating_phase_length(p[:,0],.0008),atol=1e-12)

    def test_audit_accepts_specular_flat_path(self):
        s=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)
        p=np.array([[[0.,0.,1.]],[[0.,0.,0.]],[[0.,0.,2.]]])
        result=audit_paths(p,[s],grating_index=-1)
        self.assertEqual(result['accepted_fraction'],1.)
        p[-1,0,0]=.1
        self.assertEqual(audit_paths(p,[s],grating_index=-1)['invalid_path_rays'],1)

    def test_audit_rejects_unintended_finite_surface_on_segment(self):
        # First mirror at z=0, second at z=1: the latter blocks entrance from z=2.
        s0=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)
        s1=dict(s0,vertex=np.array([0.,0.,1.]))
        p=np.array([[[0.,0.,2.]],[[0.,0.,0.]],[[0.,0.,1.]],[[0.,0.,.5]]])
        result=audit_paths(p,[s0,s1],grating_index=-1)
        self.assertEqual(result['obstructed_rays'],1)
        self.assertEqual(result['segments'][0]['blockers'][0]['surface'],1)

    def test_folded_layout_preserves_imaging_but_detects_early_detector_hit(self):
        from nslens.spectrometer import four_mirror as fold
        from nslens.spectrometer.four_mirror_audit import audit_layout
        b=fold.load_baseline(fields=3,waves=3,pupils=5)
        layout=fold.build_layout([.5,90,0,.5,90,180],b)
        radii=[s['aperture'] for s in layout['surfaces']]
        result=audit_layout(layout,b,fields=3,waves=3,radial_samples=2,azimuth_samples=8)
        self.assertLess(result['maximum_image_coordinate_error_mm'],1e-9)
        self.assertLess(result['maximum_total_geometric_path_error_mm'],1e-9)
        self.assertLess(result['maximum_grating_phase_length_error_mm'],1e-9)
        self.assertEqual(result['good_mask_disagreements'],0)
        self.assertGreater(result['early_detector_interceptions']['blocked_rays'],0)
        self.assertFalse(result['passed'])
        self.assertEqual(radii,[s['aperture'] for s in layout['surfaces']])
        self.assertTrue(all(row['outside_rays']==0 for row in result['fold_segment_fractions']))

    def test_dense_pupil_includes_center_and_whole_rim(self):
        p=dense_pupil(5,32);r=np.linalg.norm(p,axis=1)
        self.assertEqual(np.sum(np.isclose(r,1.)),32)
        self.assertEqual(np.sum(r==0),1)
        self.assertTrue(np.all(r<=1.+1e-12))

    def test_covered_surface_gap_is_conservative_for_parallel_disks(self):
        from nslens.spectrometer.four_mirror_audit import covered_surface_separation
        s=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)
        layout=dict(surfaces=[s,dict(s,vertex=np.array([0.,0.,5.]))],
                    image=np.array([0.,0.,10.]),image_x=np.array([1.,0.,0.]),
                    image_y=np.array([0.,1.,0.]),detector_bounds=np.array([[-1.,-1.],[1.,1.]]))
        result=covered_surface_separation(layout,radial_intervals=8,azimuth_samples=32,detector_intervals=8)
        pair=result['pairs'][0]
        self.assertAlmostEqual(pair['sampled_gap_mm'],5.)
        self.assertGreater(pair['conservative_gap_lower_bound_mm'],4.)
        self.assertLess(pair['conservative_gap_lower_bound_mm'],5.)
        self.assertTrue(result['all_optical_surfaces_and_detector_disjoint_certified'])

    def test_negative_sphere_gap_is_inconclusive(self):
        s=dict(vertex=np.zeros(3),basis=np.eye(3),coeff=np.zeros(9),aperture=2.)
        result=sphere_separation([s,dict(s,vertex=np.array([0.,0.,1.]))])
        self.assertFalse(result['all_optical_surfaces_disjoint_certified'])
        self.assertLess(result['pairs'][0]['conservative_gap_lower_bound_mm'],0.)
        self.assertIn('inconclusive',result['limitation'])

if __name__=='__main__':unittest.main()
