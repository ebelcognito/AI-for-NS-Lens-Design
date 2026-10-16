import numpy as np
import unittest

from nslens.analysis.variable_mirror import baseline_genes, trace


VALID_GENES = np.array([
    -0.42074889966353735, -0.4636612883066068, -0.41043454602682894,
    0.7839100938539674, -0.10932265245493844, 0.6688289132489258,
    -0.018093967369051334, -0.21739615254454436, -0.2580389171464168,
])


class VariableMirrorSpecTests(unittest.TestCase):
  def test_default_task_is_identical_to_explicit_default_spec(self):
    genes = baseline_genes(3)
    self.assertEqual(trace(genes, 3), trace(
        genes, 3, focal=50.0, aperture=27.78, half_field_deg=4.0))


  def test_focal_changes_mapping_error_without_changing_good_ray_paths(self):
    default = trace(VALID_GENES, 3, fields=3, pupils=9, paths=True)
    custom = trace(VALID_GENES, 3, fields=3, pupils=9, paths=True, focal=60.0)
    self.assertTrue(np.array_equal(default[4], custom[4]))
    for default_path, custom_path in zip(default[3], custom[3]):
        self.assertEqual(len(default_path), len(custom_path))
        for default_point, custom_point in zip(default_path, custom_path):
            self.assertTrue(np.array_equal(default_point, custom_point))
    self.assertNotEqual(default[0]["fields"][0]["centroid_error_mm"],
                        custom[0]["fields"][0]["centroid_error_mm"])
    self.assertNotEqual(default[0]["worst_mapping_rms_mm"],
                        custom[0]["worst_mapping_rms_mm"])
    self.assertTrue(np.all(default[4]))
    images = np.array([
        (path[-1] - default[1]["image"]) @ default[1]["image_tangent"]
        for path in default[3]
    ])
    for result, focal in ((default[0], 50.0), (custom[0], 60.0)):
        fields = np.repeat(np.linspace(-4.0, 4.0, 3), 9)
        targets = focal * np.tan(np.deg2rad(fields))
        errors = images - result["parity"] * targets
        expected = max(np.sqrt(np.mean(errors[i:i+9] ** 2))
                       for i in range(0, len(images), 9))
        self.assertAlmostEqual(result["worst_mapping_rms_mm"], expected)


  def test_trace_rejects_invalid_specification(self):
    invalid = [
        {"focal": 0}, {"focal": np.inf}, {"focal": np.nan},
        {"aperture": -1}, {"aperture": np.inf}, {"half_field_deg": 0},
        {"half_field_deg": np.nan}, {"fields": 0}, {"fields": 1.5},
        {"fields": True}, {"pupils": 0}, {"pupils": 2.5},
    ]
    for kwargs in invalid:
        with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
            trace(baseline_genes(3), 3, **kwargs)


  def test_metrics_report_missing_field_data_as_none(self):
    result = trace(VALID_GENES, 3, fields=2, pupils=2, aperture=50.0)
    self.assertTrue(any(row["support"] == 0 for row in result["fields"]))
    self.assertIsNone(result["worst_rms_spot_mm"])
    self.assertIsNone(result["worst_centroid_error_mm"])
    self.assertIsNone(result["worst_mapping_rms_mm"])


if __name__ == "__main__":
    unittest.main()
