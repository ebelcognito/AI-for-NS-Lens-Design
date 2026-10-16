"""Fast shared intersection backend versus retained scalar all-root reference."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from nslens.spectrometer.intersections import intersect
from nslens.spectrometer.physics_audit import exact_intersect
from nslens.spectrometer import model
from nslens.three_mirror import model3d
from tests.test_three_mirror_3d import sample


class BatchedIntersectionTests(unittest.TestCase):
    def assert_same(self, o, d, s, chunk_size=4096):
        actual = intersect(o, d, s, chunk_size=chunk_size)
        reference = exact_intersect(o, d, s)
        np.testing.assert_array_equal(actual[3], reference[3])
        good = reference[3]
        for a, b in zip(actual[:3], reference[:3]):
            np.testing.assert_allclose(a[good], b[good], rtol=1e-8, atol=1e-7)
        return actual

    def test_all_degrees_and_arbitrary_poses(self):
        rng = np.random.default_rng(720)
        for degree in (0, 2, 3, 4, 5, 6):
            terms = tuple(p for p in model3d.POWERS if sum(p) <= max(degree, 2))
            coeff = np.array([rng.uniform(-1, 1)/30**sum(p) for p in terms])
            if degree == 0: coeff[:] = 0
            s = dict(vertex=np.array([11., -7., 5.]),
                     basis=Rotation.from_rotvec([.4, -.2, .3]).as_matrix(),
                     aperture=30., coeff=coeff, terms=terms)
            origin = np.column_stack((rng.uniform(-35, 35, (127, 2)), rng.uniform(5, 25, 127)))
            direction = model.unit(np.column_stack((rng.uniform(-.5, .5, (127, 2)), -np.ones(127))))
            self.assert_same(origin@s['basis'].T+s['vertex'], direction@s['basis'].T, s, 17)

    def test_default_nine_term_surface(self):
        rng = np.random.default_rng(23)
        s = model.decode(np.zeros(36))['surfaces'][0]
        s['coeff'][2:] = np.array([1e-5, -2e-5, 2e-7, -1e-7, 1e-7, 1e-9, 2e-9])
        local_o = np.column_stack((rng.uniform(-18, 18, (256, 2)), np.full(256, 20.)))
        local_d = model.unit(np.column_stack((rng.normal(0, .2, (256, 2)), -np.ones(256))))
        self.assert_same(local_o@s['basis'].T+s['vertex'], local_d@s['basis'].T, s)

    def test_mixed_ray_degrees_parallel_and_tangent(self):
        s = dict(vertex=np.zeros(3), basis=np.eye(3), aperture=10., coeff=np.array([1.]), terms=((2, 0),))
        o = np.array([[-2., 0., 1.], [0., 0., 1.], [0., 0., 0.],
                      [-2., 0., 0.], [0., 12., 1.], [2., 0., 1.]])
        d = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.],
                      [1., 0., 0.], [0., 0., -1.], [1., 0., 0.]])
        r = self.assert_same(o, d, s, 2)
        np.testing.assert_array_equal(r[3], [True, True, False, True, False, False])
        self.assertAlmostEqual(r[2][0], 1.)

    def test_aperture_rim_and_self_hit_rejection(self):
        s = dict(vertex=np.zeros(3), basis=np.eye(3), aperture=1., coeff=np.zeros(9))
        o = np.array([[1., 0., 1.], [1.+1e-6, 0., 1.], [0., 0., 1e-8], [0., 0., -1.]])
        d = np.tile([0., 0., -1.], (len(o), 1))
        r = self.assert_same(o, d, s)
        np.testing.assert_array_equal(r[3], [True, False, False, False])

    def test_small_high_order_coefficient_is_not_trimmed(self):
        s = dict(vertex=np.zeros(3), basis=np.eye(3), aperture=100.,
                 coeff=np.array([1e-15]), terms=((6, 0),))
        o = np.array([[0., 0., 1e-6]])
        d = np.array([[1., 0., 0.]])
        r = self.assert_same(o, d, s)
        self.assertTrue(r[3][0])
        self.assertAlmostEqual(r[2][0], (1e9)**(1/6), places=7)

    def test_full_routing_and_detector_coordinates(self):
        layout = model3d.lift_sample(sample())
        layout = model3d.pose_delta(layout, 1, (0., 0., .08), (.001, -.001, .002))
        fields = model3d.field_grid((4., 3.), (3, 3))
        pupil = model3d.disk_pupil(27.78, 9)
        o, d = model3d.launch_rays(layout, fields, pupil)
        fast = model3d.trace_rays(layout, o, d)
        ref = model3d.trace_rays(layout, o, d, intersect_fn=exact_intersect)
        np.testing.assert_array_equal(fast['good'], ref['good'])
        np.testing.assert_array_equal(fast['route'], ref['route'])
        np.testing.assert_allclose(fast['image_mm'][ref['good']], ref['image_mm'][ref['good']], atol=1e-7)

    def test_empty_batch(self):
        s = model.decode(np.zeros(36))['surfaces'][0]
        r = intersect(np.empty((0, 3)), np.empty((0, 3)), s)
        self.assertEqual(r[0].shape, (0, 3))
        self.assertEqual(r[3].shape, (0,))


if __name__ == '__main__':
    unittest.main()
