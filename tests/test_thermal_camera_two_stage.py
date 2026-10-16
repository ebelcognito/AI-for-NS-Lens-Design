"""Checks for the spectrometer-style two-stage XY camera basis."""
import unittest
import numpy as np

from nslens.spectrometer import model as spectrometer
from nslens.thermal_camera import model as radial
from nslens.thermal_camera import spectrometer_style_model as camera


class ThermalCameraTwoStageTests(unittest.TestCase):
    def test_basis_and_stage_dimensions_match_spectrometer(self):
        self.assertEqual(camera.TERMS, spectrometer.TERMS)
        self.assertEqual(camera.BASE_DIM, 14)
        self.assertEqual(camera.N_BASE, 20)
        self.assertEqual(camera.N_TOTAL, 41)
        self.assertEqual(camera.STAGES[0][1], 20)
        self.assertEqual(camera.STAGES[1][1], 41)

    def test_twenty_gene_stage_zero_extends_to_identical_41_gene_trace(self):
        rng = np.random.default_rng(23)
        stage_base = rng.uniform(-.1, .1, camera.N_BASE)
        full = np.pad(stage_base, (0, camera.N_TOTAL-camera.N_BASE))
        pupils = np.array([[0., 0.], [.3, 0.], [0., -.3], [-.2, .2]])
        short_trace = camera.trace(stage_base, fields=2, waves=1, pupils=1, pupil_xy=pupils)
        full_trace = camera.trace(full, fields=2, waves=1, pupils=1, pupil_xy=pupils)
        np.testing.assert_array_equal(short_trace['good'], full_trace['good'])
        np.testing.assert_array_equal(short_trace['xy'], full_trace['xy'])
        short_model = camera.decode(stage_base)
        full_model = camera.decode(full)
        for surface in camera.SURFACES:
            np.testing.assert_array_equal(short_model[surface][3]['xy'], full_model[surface][3]['xy'])

    def test_fifth_order_coefficients_are_active_on_each_surface(self):
        baseline_genes = np.zeros(camera.N_TOTAL)
        baseline = camera.decode(baseline_genes)
        points = np.array([[.37, .28], [-.42, .31], [.22, -.39]])
        fifth = [(index, parameter) for index, parameter in enumerate(camera.PARAMETERS)
                 if sum(parameter[1:]) == 5]
        self.assertEqual(len(fifth), 6)
        for offset, (surface, i, j) in fifth:
            with self.subTest(surface=surface, term=(i, j)):
                genes = baseline_genes.copy()
                genes[camera.BASE_DIM+offset] = .7
                altered = camera.decode(genes)
                before = radial.sag_xy(points[:, 0], points[:, 1], *baseline[surface][1:])[0]
                after = radial.sag_xy(points[:, 0], points[:, 1], *altered[surface][1:])[0]
                self.assertFalse(np.array_equal(before, after))

    def test_decoded_analytic_slopes_match_finite_differences(self):
        genes = np.zeros(camera.N_TOTAL)
        target = next(index for index, term in enumerate(camera.PARAMETERS) if term == ('primary', 5, 0))
        genes[camera.BASE_DIM+target] = .6
        surface = camera.decode(genes)['primary']
        x, y, h = .31, -.27, 1e-6
        _, zx, zy = camera.sag_xy(x, y, surface[1], surface[2], surface[3])
        xp = camera.sag_xy(x+h, y, surface[1], surface[2], surface[3])[0]
        xm = camera.sag_xy(x-h, y, surface[1], surface[2], surface[3])[0]
        yp = camera.sag_xy(x, y+h, surface[1], surface[2], surface[3])[0]
        ym = camera.sag_xy(x, y-h, surface[1], surface[2], surface[3])[0]
        self.assertAlmostEqual(zx, (xp-xm)/(2*h), delta=2e-8)
        self.assertAlmostEqual(zy, (yp-ym)/(2*h), delta=2e-8)


if __name__ == '__main__':
    unittest.main()
