import numpy as np
import unittest

from nslens.analysis import refinement_probe_3d as probe
from nslens.monolithic_ir import model as mono
from nslens.thermal_camera import freeform_model as thermal
from nslens.thermal_camera.model import reflect, snell, unit


def test_thermal_secondary_analytic_gradient_jacobian_matches_finite_difference():
    genes = np.zeros(thermal.N_TOTAL)
    decoded = thermal.decode(genes)
    x, y = np.array([-.7, .4]), np.array([.35, -.8])
    jac_x, jac_y = probe._thermal_gradient_jacobian(x, y)
    surface = decoded["secondary"]
    for col, gene_index in enumerate(probe._thermal_active_indices()):
        shifted = genes.copy(); shifted[gene_index] = 1e-6
        updated = thermal.decode(shifted)["secondary"]
        _, sx, sy = thermal.radial.sag_xy(x, y, *surface[1:])
        _, tx, ty = thermal.radial.sag_xy(x, y, *updated[1:])
        np.testing.assert_allclose((tx - sx) / 1e-6, jac_x[:, col], rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose((ty - sy) / 1e-6, jac_y[:, col], rtol=2e-6, atol=1e-10)


def test_mono_exit_analytic_gradient_jacobian_matches_finite_difference():
    genes = np.zeros(mono.N_TOTAL)
    base_surface = mono.decode(genes).surfaces[3]
    x, y = np.array([-.8, .65]), np.array([.4, -.3])
    jac_x, jac_y = probe._mono_gradient_jacobian(x, y)
    active = probe._mono_active_indices()
    for col, gene_index in enumerate(active):
        shifted = genes.copy(); shifted[gene_index] = 1e-6
        surface = mono.decode(shifted).surfaces[3]
        _, sx, sy = mono.sag(base_surface, base_surface.center[0] + x, y)
        _, tx, ty = mono.sag(surface, surface.center[0] + x, y)
        np.testing.assert_allclose((tx - sx) / 1e-6, jac_x[:, col], rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose((ty - sy) / 1e-6, jac_y[:, col], rtol=2e-6, atol=1e-10)


def test_reflection_required_normal_reproduces_outgoing_ray():
    incoming = unit(np.array([.18, -.07, 1.]))
    outgoing = unit(np.array([-.12, .03, 1.]))
    sx, sy = probe._reflection_required_slopes(incoming, outgoing)
    normal = unit(np.array([-sx, -sy, 1.]))
    np.testing.assert_allclose(reflect(incoming, normal), outgoing, atol=1e-12)


def test_refraction_required_normal_reproduces_outgoing_ray():
    incoming = unit(np.array([.18, -.07, 1.]))
    normal = unit(np.array([-.08, .025, 1.]))
    outgoing, valid = snell(incoming, normal, 1.5, 1.0)
    assert bool(valid)
    sx, sy = probe._refraction_required_slopes(incoming, outgoing, 1.5)
    fitted_normal = unit(np.array([-sx, -sy, 1.]))
    reproduced, valid = snell(incoming, fitted_normal, 1.5, 1.0)
    assert bool(valid)
    np.testing.assert_allclose(reproduced, outgoing, atol=1e-12)


def test_prepare_initial_metrics_match_observe_on_both_tracers():
    for environment, size in (("thermal", thermal.N_TOTAL), ("mono", mono.N_TOTAL)):
        context = probe.prepare(environment, np.zeros(size), fields=2, pupils=2)
        repeated = probe.observe(context, context["genes"])
        assert repeated == context["initial"]
        assert context["trace"] is not None
        assert context["active_indices"].size == (12 if environment == "thermal" else 9)


def test_thermal_probe_uses_secondary_surface_hits():
    context = probe.prepare("thermal", np.zeros(thermal.N_TOTAL), fields=2, pupils=2)
    hit = context["trace"]["paths"][4]
    _, c, k, coeff = context["trace"]["model"]["secondary"]
    z, _, _ = thermal.radial.sag_xy(hit[:, 0], hit[:, 1], c, k, coeff)
    np.testing.assert_allclose(hit[:, 2], context["trace"]["model"]["secondary"][0] + z,
                               atol=2e-6, rtol=0)


def test_missing_rays_do_not_count_as_small_mapping_error():
    context = probe.prepare('thermal', np.zeros(thermal.N_TOTAL), fields=2, pupils=2)
    trace = dict(context['trace'])
    trace['good'] = np.zeros_like(trace['good'])
    metrics = probe._measure('thermal', trace, context['parity'])
    assert metrics['worst_mapping_rms_mm'] is None
    assert metrics['worst_support'] == 0
    assert metrics['loss'] >= 1100


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))
