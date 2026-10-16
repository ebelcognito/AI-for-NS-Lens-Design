"""Physics and parameterization gates for the five-surface powered system."""
import unittest
import numpy as np
from nslens.spectrometer import model, four_mirror as fold, powered_four_mirror as powered
from nslens.spectrometer.physics_audit import exact_intersect
from nslens.spectrometer.four_mirror_audit import audit_trace_pair

PARAMS=[.6202153620335927,82.8469428562683,180.,.8027559088239182,89.6438726288061,0.]


class PoweredFourMirrorTests(unittest.TestCase):
    def test_flat_limit_matches_old_isometry_and_launch_cone(self):
        g=powered.seed_genes(PARAMS);layout=powered.decode(g)
        b=fold.load_baseline(fields=3,waves=3,pupils=5);old=fold.build_layout(PARAMS,b)
        for s,a in zip(old['surfaces'],powered.APERTURES):s['aperture']=float(a)
        old['detector_bounds']=layout['detector_bounds'].copy()
        a=fold.trace(old,b,fields=3,waves=3,pupils=5);r=powered.trace(g,fields=3,waves=3,pupils=5)
        self.assertEqual(r['paths'].shape[0],7)
        np.testing.assert_array_equal(a['good'],r['good'])
        np.testing.assert_allclose(a['paths'],r['paths'],atol=1e-10)
        np.testing.assert_allclose(a['xy'],r['xy'],atol=1e-10)
        np.testing.assert_allclose(a['phase'],r['phase'],atol=1e-10)
        incoming=model.unit(r['paths'][1]-r['paths'][0]).reshape(3,3,-1,3)
        u=np.linspace(-1,1,5);x,y=np.meshgrid(u,u);mask=x*x+y*y<=1+1e-12
        expected=model.unit(np.column_stack([x[mask]/7.6,y[mask]/7.6,np.ones(mask.sum())]))
        np.testing.assert_allclose(incoming[1,0],expected,atol=1e-12)

    def test_both_added_mirror_powers_change_ray_paths_and_are_fixed_masks(self):
        g=powered.seed_genes(PARAMS);original=powered.trace(g,fields=3,waves=3,pupils=5)
        for gene,surface in [(21,0),(23,4)]:
            changed=g.copy();changed[gene]=.01
            r=powered.trace(changed,fields=3,waves=3,pupils=5)
            self.assertGreater(np.max(np.abs(r['xy']-original['xy'])),1e-3)
            self.assertAlmostEqual(r['model']['surfaces'][surface]['coeff'][0],.00012)
            np.testing.assert_array_equal([s['aperture'] for s in r['model']['surfaces']],powered.APERTURES)

    def test_coarse_preserves_inherited_freeforms_and_fine_releases_all_five(self):
        g=powered.seed_genes(PARAMS);base=powered.decode(g)
        for j in [1,2,3]:
            np.testing.assert_array_equal(powered.decode(g[:powered.N_BASE])['surfaces'][j]['coeff'][2:],base['surfaces'][j]['coeff'][2:])
        for gene,surface in [(28,1),(35,2),(42,3),(49,0),(56,4)]:
            fine=g.copy();fine[gene]=.1;m=powered.decode(fine)
            self.assertNotEqual(m['surfaces'][surface]['coeff'][2],base['surfaces'][surface]['coeff'][2])
            np.testing.assert_array_equal(m['surfaces'][surface]['coeff'][:2],base['surfaces'][surface]['coeff'][:2])

    def test_independent_polynomial_trace_of_small_powered_seed(self):
        g=powered.seed_genes(PARAMS);g[21:25]=[.001,-.001,.001,-.001]
        sampling=dict(fields=3,waves=3,pupils=3)
        nominal=powered.trace(g,**sampling);exact=powered.trace(g,intersect_fn=exact_intersect,**sampling)
        audit=audit_trace_pair(nominal,exact,include_surface_certificate=False)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['good_mask_disagreements'],0)
        self.assertLess(audit['maximum_common_image_coordinate_disagreement_mm'],1e-9)

    def test_independent_curvature_audit_distinguishes_four_powered_from_two_flats(self):
        from nslens.spectrometer.four_mirror_audit import powered_curvature_audit
        g=powered.seed_genes(PARAMS)
        a=powered_curvature_audit(powered.decode(g),radial_samples=3,azimuth_samples=8)
        self.assertFalse(a['all_four_mirrors_powered'])
        g[21:25]=[.1,.2,-.1,-.2]
        a=powered_curvature_audit(powered.decode(g),radial_samples=3,azimuth_samples=8)
        self.assertTrue(a['all_four_mirrors_powered'])
        self.assertTrue(a['all_expected_signs_preserved'])
        self.assertAlmostEqual(a['surfaces'][0]['minimum_hessian_eigenvalue_per_mm'],.0024)
        self.assertAlmostEqual(a['surfaces'][0]['maximum_hessian_eigenvalue_per_mm'],.0048)
        self.assertAlmostEqual(a['surfaces'][4]['minimum_hessian_eigenvalue_per_mm'],-.0048)

    def test_curvature_gate_catches_reversed_grating_and_invalid_pupil(self):
        m=powered.decode(powered.seed_genes(PARAMS))
        self.assertEqual(powered.curvature_metrics(m)['sign_violation'],0.)
        m['surfaces'][2]['coeff']=-m['surfaces'][2]['coeff']
        self.assertGreater(powered.curvature_metrics(m)['sign_violation'],.001)
        with self.assertRaises(ValueError):powered.trace(powered.seed_genes(PARAMS),pupil_xy=[[1.1,0.]])

if __name__=='__main__':unittest.main()
