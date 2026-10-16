"""Independent numerical checks for the final-mirror freeform tracer."""
import unittest

import numpy as np
from numpy.polynomial import Polynomial

from nslens.analysis import final_freeform as ff
from nslens.analysis import variable_mirror as base


def polynomial_root_oracle(o, d, m, j):
    """Solve each ray/sag crossing as a scalar polynomial in ray distance."""
    hits = []
    for origin, direction in zip(o, d):
        delta = origin - m["vertices"][j]
        u0 = float(delta @ m["tangents"][j])
        v0 = float(delta @ m["normals"][j])
        du = float(direction @ m["tangents"][j])
        dv = float(direction @ m["normals"][j])
        t = Polynomial([0., 1.])
        u = u0 + du*t
        v = v0 + dv*t
        sag = m["curves"][j]*u*u
        if j == len(m["curves"])-1:
            for power, coeff in zip(range(3, 7), m["ff_coeff"]):
                sag = sag + coeff*(u/30.)**power
        roots = (sag-v).roots()
        candidates = []
        for root in roots:
            if abs(root.imag) > 1e-7*(1+abs(root.real)):
                continue
            distance = float(root.real)
            local_u = u0 + distance*du
            if distance > 1e-7 and abs(local_u) <= m["widths"][j] + 1e-9:
                candidates.append(distance)
        hits.append(min(candidates, default=np.inf))
    return np.asarray(hits)


class FinalFreeformNumericalTests(unittest.TestCase):
    def test_zero_coefficients_match_base_intersections_and_trace(self):
        zero = np.zeros(4)
        rng = np.random.default_rng(631)
        for n in range(3, 7):
            genes = base.baseline_genes(n)
            old_model = base.decode(genes, n)
            model = ff.decode(genes, n, zero)
            origins = rng.normal(size=(24, 2))*75.
            angles = rng.uniform(-np.pi, np.pi, len(origins))
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            np.testing.assert_array_equal(
                ff.intersections(origins, directions, model),
                base.intersections(origins, directions, old_model))
            a = base.trace(genes, n, fields=3, pupils=5, paths=True)
            b = ff.trace(genes, n, zero, fields=3, pupils=5, paths=True)
            self.assertEqual(a[0], {k: v for k, v in b[0].items()
                                   if k not in ("freeform_coefficients_mm", "ray_image_mm",
                                                "ray_good", "ray_field_deg", "ray_route")})
            np.testing.assert_array_equal(a[4], b[4])
            for ap, bp in zip(a[3], b[3]):
                np.testing.assert_allclose(ap, bp, rtol=0, atol=1e-12)

    def test_curved_degree_six_intersections_match_polynomial_oracle(self):
        genes = base.baseline_genes(4).copy()
        genes[8:12] = [0.55, -0.4, 0.7, -0.6]
        coefficients = np.array([1.1, -2.4, 3.3, 2.0])
        model = ff.decode(genes, 4, coefficients)
        j = 3
        vertex, tangent, normal = (model[k][j] for k in
                                   ("vertices", "tangents", "normals"))
        # Rays begin above or below the profile and have mixed tangential
        # components, yielding multiple real roots for some rays.
        us = np.array([-29., -21., -8., 0., 11., 23., 29.])
        local_origins = np.column_stack((us, np.full(len(us), 35.)))
        origins = vertex + local_origins[:, :1]*tangent + local_origins[:, 1:]*normal
        local_dirs = np.array([[.35, -1.], [-.55, -1.], [.1, -1.],
                               [0., -1.], [.7, -1.], [-.25, -1.], [.45, -1.]])
        local_dirs /= np.linalg.norm(local_dirs, axis=1, keepdims=True)
        directions = local_dirs[:, :1]*tangent + local_dirs[:, 1:]*normal
        # Add a vertical ray and a ray whose forward crossing lies outside
        # the finite clear width.
        origins = np.vstack((origins, vertex + 35.*normal))
        directions = np.vstack((directions, normal))
        far_u = 45.
        origins = np.vstack((origins, vertex + far_u*tangent + 35.*normal))
        directions = np.vstack((directions, -normal))
        expected = polynomial_root_oracle(origins, directions, model, j)
        actual = ff.intersections(origins, directions, model)[:, j]
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8,
                                   equal_nan=True)
        self.assertTrue(np.isinf(actual[-1]))


if __name__ == "__main__":
    unittest.main()
