"""Independent physics and numerical checks for the experimental IR model."""
import unittest

import numpy as np

from nslens.monolithic_ir import model
from nslens.thermal_camera.model import reflect, snell, unit


class MonolithicIRPhysicsTests(unittest.TestCase):
    def test_ig4_dispersion_at_ten_microns(self):
        self.assertAlmostEqual(float(model.index(0.010)), 2.6091, delta=0.0002)
        self.assertGreater(float(model.index(0.008)), float(model.index(0.014)))

    def test_snell_round_trip_and_reflection(self):
        d = unit(np.array([0.12, -0.07, 1.0]))
        normal = unit(np.array([0.02, -0.01, 1.0]))
        inside, ok = snell(d, normal, 1.0, 2.6)
        self.assertTrue(bool(ok))
        outside, ok = snell(inside, normal, 2.6, 1.0)
        self.assertTrue(bool(ok))
        np.testing.assert_allclose(outside, d, atol=1e-12)
        reflected = reflect(d, normal)
        self.assertAlmostEqual(np.linalg.norm(reflected), 1.0, places=12)
        self.assertLess(np.dot(reflected, normal), 0.0)

    def test_sag_derivatives_match_finite_differences(self):
        surface = model.Surface(
            center=np.array([0.3, 0.0, 0.1]), tilt=0.02,
            power=np.array([0.003, -0.002]),
            higher=np.array([0.01, -0.02, 0.015, 0.01, -0.01, 0.007, 0.003]),
            radius=3.0, name="test")
        x, y, h = 1.13, -0.71, 1e-6
        z, zx, zy = model.sag(surface, x, y)
        dx = (model.sag(surface, x+h, y)[0] - model.sag(surface, x-h, y)[0])/(2*h)
        dy = (model.sag(surface, x, y+h)[0] - model.sag(surface, x, y-h)[0])/(2*h)
        self.assertAlmostEqual(float(zx), float(dx), delta=2e-8)
        self.assertAlmostEqual(float(zy), float(dy), delta=2e-8)
        self.assertTrue(np.isfinite(z))

    def test_nominal_objective_is_finite_and_reports_real_ray_counts(self):
        genes = np.zeros(model.N_TOTAL)
        result = model.evaluate(genes, fields=2, waves=3, pupils=3)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["launched_rays"], 0)
        self.assertGreater(result["valid_rays"], 0)
        self.assertLessEqual(result["valid_rays"], result["launched_rays"])
        self.assertAlmostEqual(result["throughput"], result["valid_rays"]/result["launched_rays"])

    def test_good_rays_stay_inside_intermediate_body(self):
        tr = model.trace(np.zeros(model.N_TOTAL), fields=2, waves=2, pupils=3)
        self.assertTrue(np.isfinite(tr["paths"]).all())
        self.assertTrue(np.isfinite(tr["escape"]).all())
        self.assertTrue(np.isfinite(tr["overflow"]).all())
        self.assertGreater(tr["good"].sum(), 0)
        self.assertTrue(np.all(tr["escape"][tr["good"]] < 1e-6))
        self.assertTrue(np.all(tr["overflow"][tr["good"]] < 1e-7))

    def test_nominal_body_constraints_have_positive_clearance(self):
        m = model.decode(np.zeros(model.N_TOTAL))
        e, m1, m2, ex = m.surfaces
        x, y = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(-5, 5, 21))
        front, back = model.body(m, x, y)
        self.assertGreater(m2.center[0]-m2.radius-e.center[0]-e.radius, 0)
        self.assertGreater(ex.center[0]-ex.radius-m1.center[0]-m1.radius, 0)
        self.assertTrue(np.all(back-front > 0))

    def test_representation_conversions_preserve_physical_rays(self):
        from nslens.monolithic_ir import model_wide, model_expanded
        from nslens.monolithic_ir.refine_run import convert
        g=np.random.default_rng(19).uniform(-.7,.7,48)
        h=convert(g,'freeform'); k=convert(h,'geometry')
        traces=[mod.trace(v,fields=2,waves=2,pupils=2) for mod,v in ((model,g),(model_wide,h),(model_expanded,k))]
        for tr in traces[1:]:
            np.testing.assert_allclose(tr['paths'],traces[0]['paths'],rtol=1e-8,atol=1e-7)
            np.testing.assert_array_equal(tr['good'],traces[0]['good'])

    def test_optimizer_and_polish_share_each_variants_merit(self):
        from nslens.monolithic_ir import model_wide, model_expanded
        for mod in (model,model_wide,model_expanded):
            g=np.zeros(48)
            r=mod.residuals(g,fields=2,waves=2,pupils=2)
            self.assertAlmostEqual(float(r@r),mod.evaluate(g,fields=2,waves=2,pupils=2)['loss'],places=8)


if __name__ == "__main__":
    unittest.main()

