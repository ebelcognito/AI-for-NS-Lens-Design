"""3D physics checks independent of the new tracer's batch root construction."""
import unittest
from copy import deepcopy

import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial.transform import Rotation

from nslens.analysis import final_freeform as old
from nslens.three_mirror import model3d as m3


GENES = [-.6969515450861835, .740862516388769, .9788451876775675,
         .821563296174245, .7536500089869658, -.9773388808470015,
         .0148250856104013, -.0669003227022188, -.25565007167528253]


def sample():
    return dict(model='variable_mirror_n3', genes=GENES.copy(),
                task=dict(focal=50., aperture=27.78, half_field_deg=4.),
                freeform_coefficients_mm=[.13, -.08, .05, .015])


class ThreeMirror3DTests(unittest.TestCase):
    def test_planar_slice_matches_old_freeform_rays(self):
        s = sample()
        fields = np.column_stack((np.linspace(-4, 4, 7), np.zeros(7)))
        pupils = np.column_stack((np.linspace(-13.89, 13.89, 31), np.zeros(31)))
        ref = old.trace(s['genes'], 3, s['freeform_coefficients_mm'], fields=7, pupils=31)
        for mode in ('equal_curvature', 'cylindrical'):
            result = m3.evaluate(m3.lift_sample(s, mode), fields, pupils)
            good = result['ray_good']
            np.testing.assert_array_equal(good, ref['ray_good'])
            self.assertTrue(good.all())
            np.testing.assert_allclose(result['ray_image_mm'][good, 0], np.array(ref['ray_image_mm'])[good], atol=2e-8)
            np.testing.assert_allclose(result['ray_image_mm'][good, 1], 0, atol=1e-12)
            np.testing.assert_array_equal(result['ray_route'][good], np.array(ref['ray_route'])[good])

    def test_legacy_planar_slice(self):
        from nslens.three_mirror import pilot
        s = sample()
        s['model'] = 'legacy_three_mirror'
        s['genes'] = [(180*GENES[0]+150)/28, (180*GENES[1]-180)/58,
                      (180*GENES[2]+180)/58]+GENES[3:]+[0., 0., 0.]
        s['freeform_coefficients_mm'] = [0.]*4
        f = np.column_stack((np.linspace(-4, 4, 5), np.zeros(5)))
        p = np.column_stack((np.linspace(-13.89, 13.89, 21), np.zeros(21)))
        ref = pilot.trace(s['genes'], fields=5, pupils=21)
        result = m3.evaluate(m3.lift_sample(s), f, p)
        self.assertEqual(result['worst_support'], ref['worst_support'])
        np.testing.assert_allclose([v['rms_spot_mm'] for v in result['fields']],
                                   [v['rms_spot_mm'] for v in ref['fields']], atol=2e-8)

    def test_general_surface_against_scalar_polynomial_oracle(self):
        rng = np.random.default_rng(314)
        surface = m3.lift_sample(sample())['surfaces'][0]
        surface['basis'] = Rotation.from_rotvec([.2, -.35, .17]).as_matrix()
        surface['vertex'] = np.array([9., -12., 4.])
        surface['coefficients'] = rng.uniform(-.35, .35, len(m3.POWERS))
        local_o = np.column_stack((rng.uniform(-20, 20, (30, 2)), np.full(30, 12.)))
        local_d = m3.unit(np.column_stack((rng.uniform(-.25, .25, (30, 2)), -np.ones(30))))
        origin = local_o @ surface['basis'].T+surface['vertex']
        direction = local_d @ surface['basis'].T
        expected = []
        for o, d in zip(local_o, local_d):
            u, v, w = [Polynomial([o[k], d[k]]) for k in range(3)]
            equation = -w
            for (i, j), c in zip(m3.POWERS, surface['coefficients']):
                equation += c*(u/30)**i*(v/30)**j
            roots = equation.roots()
            candidates = [r.real for r in roots if abs(r.imag) < 1e-7 and r.real > 1e-7
                          and u(r.real)**2+v(r.real)**2 <= surface['radius']**2]
            expected.append(min(candidates, default=np.inf))
        np.testing.assert_allclose(m3.intersect_surface(origin, direction, surface), expected, atol=1e-7, rtol=1e-8)

    def test_normals_are_orthogonal_to_surface_tangents(self):
        s = m3.lift_sample(sample())['surfaces'][2]
        s['coefficients'][m3.POWERS.index((1, 2))] = .4
        uv = np.array([[7., -5.], [-9., 6.]])
        def point(x):
            return np.column_stack((x, m3.surface_sag(s, x))) @ s['basis'].T+s['vertex']
        normal = m3.surface_normal(s, point(uv))
        for axis in range(2):
            h = np.eye(2)[axis]*1e-5
            tangent = (point(uv+h)-point(uv-h))/(2e-5)
            np.testing.assert_allclose(np.sum(tangent*normal, axis=1), 0, atol=1e-9)

    def test_rigid_motion_invariance_of_nonplanar_rays(self):
        layout = m3.lift_sample(sample())
        layout = m3.pose_delta(layout, 1, (0., 0., .1), (.001, -.001, .002))
        f = m3.field_grid((2., 1.), (3, 3))
        p = m3.disk_pupil(12., 7)
        o, d = m3.launch_rays(layout, f, p)
        ref = m3.trace_rays(layout, o, d)
        rotation = Rotation.from_rotvec([.23, -.18, .35]).as_matrix()
        translation = np.array([14., -8., 23.])
        changed = deepcopy(layout)
        for obj in changed['surfaces']+[changed['detector']]:
            obj['vertex'] = rotation @ obj['vertex']+translation
            obj['basis'] = rotation @ obj['basis']
        got = m3.trace_rays(changed, o @ rotation.T+translation, d @ rotation.T)
        self.assertGreater(ref['received'], 0)
        np.testing.assert_array_equal(got['good'], ref['good'])
        np.testing.assert_allclose(got['image_mm'][got['good']], ref['image_mm'][ref['good']], atol=1e-7)

    def test_disk_mask_and_nearest_root(self):
        s = dict(vertex=np.zeros(3), basis=np.eye(3), radius=2., coefficients=np.zeros(len(m3.POWERS)))
        s['coefficients'][m3.POWERS.index((2, 0))] = 900.  # sag = u²
        o = np.array([[-2., 0., 1.], [0., 3., 1.]])
        d = np.array([[1., 0., 0.], [0., 0., -1.]])
        got = m3.intersect_surface(o, d, s)
        self.assertAlmostEqual(got[0], 1.)  # two roots at 1 and 3; choose first
        self.assertTrue(np.isinf(got[1]))

    def test_wrong_route_is_not_good_imaging(self):
        layout = m3.lift_sample(sample())
        layout['detector']['vertex'] = np.array([-1000., 0., 0.])
        layout['detector']['basis'] = np.column_stack(([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]))
        r = m3.trace_rays(layout, [[-1100., 0., 0.]], [[1., 0., 0.]])
        self.assertTrue(np.isfinite(r['image_mm']).all())
        self.assertEqual(r['received'], 0)

    def test_residual_statistics_and_ray_accounting(self):
        layout = m3.lift_sample(sample())
        f, p = m3.field_grid((4., 3.), (3, 3)), m3.disk_pupil(27.78, 9)
        r = m3.evaluate(layout, f, p)
        self.assertEqual(r['launched_rays'], len(f)*len(p))
        self.assertEqual(r['geometric_survival'], float(np.mean(r['ray_good'])))
        for row in r['fields']:
            if row['rms_spot_mm'] is not None:
                self.assertAlmostEqual(row['rms_spot_mm'], np.linalg.norm(row['rms_xy_mm']))


if __name__ == '__main__':
    unittest.main()
