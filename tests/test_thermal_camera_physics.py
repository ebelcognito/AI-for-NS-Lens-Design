"""Focused physical checks for the analytic thermal-camera ray scaffold."""
import unittest
import numpy as np
from nslens.thermal_camera import model
from nslens.thermal_camera.array_audit import CENTERS, steering_bundle
from nslens.thermal_camera.root_audit import roots


class ThermalCameraPhysicsTests(unittest.TestCase):
    def test_snell_scalar_law_and_total_internal_reflection(self):
        theta = np.deg2rad(30.0)
        d = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
        out, valid = model.snell(d, np.array([[0.0, 0.0, 1.0]]), 1.0, 1.5)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(np.arcsin(out[0, 0]), np.arcsin(np.sin(theta)/1.5), places=12)
        inside = np.array([[np.sin(np.deg2rad(60)), 0.0, np.cos(np.deg2rad(60))]])
        _, valid = model.snell(inside, np.array([[0.0, 0.0, 1.0]]), 1.5, 1.0)
        self.assertFalse(valid[0])

    def test_reflection_and_conic_focus(self):
        self.assertTrue(np.allclose(model.reflect([[0, 0, 1]], [[0, 0, 1]]), [[0, 0, -1]]))
        # Paraboloid c=-.1 has paraxial focus five millimetres behind z=10.
        o, d = np.array([[1., 0., 0.]]), np.array([[0., 0., 1.]])
        p, n, _, valid, _ = model._surface(o, d, 10., -.1, -1., np.zeros(3))
        self.assertTrue(valid[0])
        reflected = model.reflect(d, n)
        t = (5.-p[0, 2])/reflected[0, 2]
        self.assertAlmostEqual(p[0, 0]+t*reflected[0, 0], 0., places=10)

    def test_obscuration_stays_in_launched_ray_denominator(self):
        pupil = np.array([[0., 0.], [.5, 0.], [-.5, 0.], [0., .5], [0., -.5]])
        r = model.trace(np.zeros(model.N_TOTAL), fields=1, waves=1, pupils=1, pupil_xy=pupil)
        self.assertEqual(r['good'].shape, (1, 1, 5))
        self.assertEqual(r['launched'], 5)
        self.assertFalse(r['good'][0, 0, 0])
        self.assertAlmostEqual(float(r['good'].mean()), np.count_nonzero(r['good'])/r['launched'])
        e = model.evaluate(np.zeros(model.N_TOTAL), fields=1, waves=1, pupils=1, pupil_xy=pupil)
        self.assertTrue(np.isfinite(e['loss']))
        self.assertTrue(np.isfinite(e['throughput']))
        self.assertAlmostEqual(e['throughput'], float(r['good'].mean()))

    def test_injected_prism_rays_and_output_shapes(self):
        fields = np.array([[0., 0.], [1., -1.]])
        pupils = np.array([[0., 0.], [.5, 0.], [-.5, 0.]])
        origins = np.zeros((6, 3))
        directions = np.tile([0., 0., 1.], (6, 1))
        r = model.trace(np.zeros(model.N_TOTAL), fields=2, waves=1, pupils=1,
                        field_xy=fields, pupil_xy=pupils,
                        entrance_o=origins, entrance_d=directions)
        self.assertEqual(r['xy'].shape, (2, 1, 3, 2))
        self.assertEqual(r['good'].shape, (2, 1, 3))
        self.assertEqual(r['fields_deg'].shape, (2, 2))

    def test_every_gene_affects_decoded_physics(self):
        def signature(m):
            return np.concatenate([np.array(m['primary'][:3], float), m['primary'][3],
                                   np.array(m['secondary'][:3], float), m['secondary'][3],
                                   np.array(m['plate'][:3], float), m['plate'][3],
                                   np.array([m['secondary_radius'], m['hole_radius'],
                                             m['primary_vertex_z'], m['secondary_vertex_z'],
                                             m['plate_thickness'], m['back_focus']])])
        base = signature(model.decode(np.zeros(model.N_TOTAL)))
        for i in range(model.N_TOTAL):
            g = np.zeros(model.N_TOTAL)
            g[i] = .7
            self.assertFalse(np.array_equal(base, signature(model.decode(g))), f'gene {i} is inert')

    def test_all_eight_steering_prisms_send_local_chiefs_along_positive_z(self):
        pupils = np.array([[0., 0.], [.3, 0.], [-.3, .2]])
        for center in CENTERS:
            with self.subTest(center=center.tolist()):
                bundle = steering_bundle(center, np.zeros((1, 2)), pupils)
                self.assertTrue(np.all(bundle['valid']))
                self.assertTrue(np.all(np.isfinite(bundle['entrance_o'])))
                self.assertTrue(np.all(np.isfinite(bundle['entrance_d'])))
                self.assertTrue(np.all(bundle['entrance_d'][:, 2] > 0))
                self.assertLess(np.max(np.abs(bundle['entrance_d'][:, :2])), 1e-12)
                self.assertGreater(bundle['prism']['maximum_thickness_mm'], 0.)
                self.assertTrue(np.isfinite(bundle['prism']['maximum_thickness_mm']))
                exit_z = bundle['prism_paths'][1, :, 2]
                self.assertTrue(np.all(exit_z < 0.))

    def test_independent_root_audit_matches_plane_and_paraboloid_roots(self):
        # Plane z=3 with d_z=.8 has t=3/.8.
        plane = roots(np.zeros(3), np.array([.6, 0., .8]), (3., 0., 0., []))
        self.assertEqual(len(plane), 1)
        self.assertAlmostEqual(plane[0], 3./.8, places=10)
        # For z=10-.05*x^2 and ray (0,0,11)+t*(1,0,-1), the
        # intersection equation is .05*t^2-t+1=0, with two positive roots.
        parabola = roots(np.array([0., 0., 11.]), np.array([1., 0., -1.]),
                         (10., -.1, -1., []))
        expected = sorted(np.roots([.05, -1., 1.]).tolist())
        self.assertEqual(len(parabola), 2)
        self.assertTrue(np.allclose(parabola, expected, atol=1e-9, rtol=0))
        self.assertAlmostEqual(parabola[0], (1.-np.sqrt(.8))/.1, places=9)


if __name__ == '__main__':
    unittest.main()
