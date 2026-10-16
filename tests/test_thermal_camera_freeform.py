"""Regression checks for nested XY-freeform thermal-camera surfaces."""
import unittest
import numpy as np

from nslens.thermal_camera import model as radial
from nslens.thermal_camera import freeform_model as freeform
from nslens.thermal_camera.root_audit import roots


class ThermalCameraFreeformTests(unittest.TestCase):
    def test_analytic_xy_slopes_match_finite_differences(self):
        desc = {'radial': [2e-4, -4e-7], 'scale': 2.7,
                'xy': [(2, 0, .013), (1, 1, -.021), (0, 3, .008), (3, 1, .004)]}
        x, y, h = .37, -.42, 1e-6
        z, zx, zy = radial.sag_xy(x, y, -.04, -.8, desc)
        xp = radial.sag_xy(x+h, y, -.04, -.8, desc)[0]
        xm = radial.sag_xy(x-h, y, -.04, -.8, desc)[0]
        yp = radial.sag_xy(x, y+h, -.04, -.8, desc)[0]
        ym = radial.sag_xy(x, y-h, -.04, -.8, desc)[0]
        self.assertTrue(np.isfinite(z))
        self.assertAlmostEqual(zx, (xp-xm)/(2*h), delta=2e-8)
        self.assertAlmostEqual(zy, (yp-ym)/(2*h), delta=2e-8)

    def test_xy_freeform_is_asymmetric_and_not_radially_symmetric(self):
        desc = {'radial': [], 'scale': 2., 'xy': [(2, 0, .02), (0, 2, -.01), (1, 1, .015)]}
        self.assertNotAlmostEqual(radial.sag_xy(.4,.1,0.,0.,desc)[0],
                                  radial.sag_xy(.1,.4,0.,0.,desc)[0], places=8)

    def test_zero_extension_preserves_radial_trace_and_metrics(self):
        genes = np.zeros(freeform.BASE_DIM)
        pupil = np.array([[0.,0.],[.35,0.],[0.,.35],[-.35,0.]])
        base = radial.trace(genes, fields=2, waves=1, pupils=1, pupil_xy=pupil)
        embedded = freeform.trace(genes, fields=2, waves=1, pupils=1, pupil_xy=pupil)
        np.testing.assert_array_equal(base['good'], embedded['good'])
        np.testing.assert_allclose(base['xy'], embedded['xy'], atol=1e-12, rtol=0)
        a = radial.evaluate(genes, fields=2, waves=1, pupils=1, pupil_xy=pupil)
        b = freeform.evaluate(genes, fields=2, waves=1, pupils=1, pupil_xy=pupil)
        for key in ('throughput','rms_spot_mm','max_spot_mm','mapping_rms_mm','fitted_efl_mm'):
            self.assertAlmostEqual(a[key], b[key], delta=1e-12, msg=key)
        self.assertAlmostEqual(b['radial_merit_component'], a['loss'], delta=1e-12)

    def test_active_stage_prefix_is_exactly_preserved_when_extended(self):
        rng = np.random.default_rng(7)
        short = np.zeros(freeform.STAGES[0][1])
        short[:freeform.BASE_DIM] = np.linspace(-.2,.2,freeform.BASE_DIM)
        short[freeform.BASE_DIM:] = rng.uniform(-.2,.2,len(short)-freeform.BASE_DIM)
        extended = np.pad(short,(0,freeform.STAGES[1][1]-len(short)))
        a,b = freeform.decode(short),freeform.decode(extended)
        for name in freeform.SURFACES:
            np.testing.assert_array_equal(a[name][:3],b[name][:3])
            x,y=np.array([-.5,.1,.7]),np.array([.2,-.4,.3])
            za=radial.sag_xy(x,y,a[name][1],a[name][2],a[name][3])[0]
            zb=radial.sag_xy(x,y,b[name][1],b[name][2],b[name][3])[0]
            np.testing.assert_array_equal(za,zb)
        pup=np.array([[0.,0.],[.3,0.],[0.,-.3]])
        ta=freeform.trace(short,fields=2,waves=1,pupils=1,pupil_xy=pup)
        tb=freeform.trace(extended,fields=2,waves=1,pupils=1,pupil_xy=pup)
        np.testing.assert_array_equal(ta['good'],tb['good'])
        np.testing.assert_array_equal(ta['xy'],tb['xy'])

    def test_each_appended_gene_changes_its_surface_sag(self):
        zeros=np.zeros(freeform.N_TOTAL)
        points=np.array([[.31,.27],[-.43,.22],[.19,-.36]])
        base=freeform.decode(zeros)
        for offset,(name,_,_) in enumerate(freeform.PARAMETERS):
            with self.subTest(surface=name,parameter=offset):
                genes=zeros.copy(); genes[freeform.BASE_DIM+offset]=.6
                altered=freeform.decode(genes)
                before=radial.sag_xy(points[:,0],points[:,1],*base[name][1:])[0]
                after=radial.sag_xy(points[:,0],points[:,1],*altered[name][1:])[0]
                self.assertFalse(np.array_equal(before,after))

    def test_sampled_geometry_guard_penalizes_excessive_surface_slope(self):
        genes=np.zeros(freeform.N_TOTAL)
        genes[23]=-20.  # Outside CMA's [-1,1] box to exercise the physical guard.
        result=freeform.evaluate(genes,fields=1,waves=1,pupils=1)
        geometry=result['freeform_geometry']
        self.assertTrue(geometry['sampled_only'])
        self.assertEqual(set(geometry['surfaces']),set(freeform.SURFACES))
        self.assertGreater(geometry['surfaces']['primary']['max_slope'],1.)
        self.assertGreater(result['geometry_penalty'],0.)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertGreater(geometry['optical_depth_mm'],0.)
        self.assertGreater(geometry['minimum_plate_thickness_mm'],0.)

    def test_freeform_surface_hessian_matches_finite_differences_of_slopes(self):
        desc={'radial':[1.7e-4,-2e-7],'scale':2.4,
              'xy':[(2,0,.014),(1,1,-.009),(0,3,.006),(2,2,.003)]}
        surface=(0.,-.035,-.7,desc)
        x,y,h=.28,-.33,2e-5
        hxx,hxy,hyy=freeform.sag_hessian(x,y,surface)
        slope=lambda xx,yy: radial.sag_xy(xx,yy,surface[1],surface[2],desc)[1:]
        zx_p=slope(x+h,y)[0]; zx_m=slope(x-h,y)[0]
        zy_p=slope(x,y+h)[1]; zy_m=slope(x,y-h)[1]
        zy_xp=slope(x+h,y)[1]; zy_xm=slope(x-h,y)[1]
        self.assertAlmostEqual(hxx,(zx_p-zx_m)/(2*h),delta=2e-7)
        self.assertAlmostEqual(hxy,(zy_xp-zy_xm)/(2*h),delta=2e-7)
        self.assertAlmostEqual(hyy,(zy_p-zy_m)/(2*h),delta=2e-7)

    def test_independent_root_audit_handles_pure_xy_sag(self):
        desc={'radial':[],'scale':1.,'xy':[(2,0,.5)]}
        hits=roots(np.zeros(3),np.array([1.,0.,1.]),(0.,0.,0.,desc))
        self.assertEqual(len(hits),1)
        self.assertAlmostEqual(hits[0],2.,places=10)


if __name__ == '__main__':
    unittest.main()
