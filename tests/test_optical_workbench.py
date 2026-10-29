from dataclasses import asdict
import unittest

import numpy as np

from nslens.learning.freeform_repair import ImagingSpec
from nslens.workbench.domain import surface_body
from nslens.workbench.tools import Workbench
from nslens.workbench.model_baseline import run_episode


def task():
    spec = ImagingSpec("unit", "train")
    return dict(task_id="unit/task", family_id="unit", split="train", spec=asdict(spec),
                start=[0.]*6, objective="Correct image errors with fixed requirements.")


class WorkbenchTests(unittest.TestCase):
    def test_surface_controls_are_independent_at_anchors(self):
        spec = ImagingSpec("unit", "train")
        a = surface_body(spec, [0.]*6)
        b = surface_body(spec, [.5,0.,0.,0.,0.,0.])
        # Right-hand anchors are unchanged when only the entrance side is edited.
        u = np.array([0., 1/32, 2/32, 30/32, 31/32])
        np.testing.assert_allclose(a.evaluate(u), b.evaluate(u), atol=1e-10)
        self.assertGreater(np.linalg.norm(a.evaluate(14/32)-b.evaluate(14/32)), .001)

    def test_locked_parameters_and_requirements(self):
        wb = Workbench(task())
        initial = wb.x.copy()
        self.assertIn("error", wb.call("perturb", dict(parameter="exit_p2", delta_mm=.1)))
        np.testing.assert_array_equal(wb.x, initial)
        self.assertIn("error", wb.call("enable", dict(surface="exit", powers=[2], index=1.8)))
        wb.call("enable", dict(surface="entrance", powers=[2]))
        self.assertIn("error", wb.call("perturb", dict(parameter="exit_p2", delta_mm=.1)))
        wb.call("optimize", dict(max_evaluations=8))
        np.testing.assert_array_equal(wb.x[1:], initial[1:])

    def test_budget_preflight_is_atomic_and_all_probes_are_counted(self):
        wb = Workbench(task(), budget=8)
        wb.call("enable", dict(surface="both", powers=[2,3,4]))
        self.assertIn("error", wb.call("optimize", dict(max_evaluations=9)))
        self.assertEqual(wb.used, 0)
        wb.call("optimize", dict(max_evaluations=8))
        self.assertLessEqual(wb.used, 8)
        self.assertEqual(wb.used, len(wb.records)-1)
        self.assertEqual(sum(r["source"]=="jacobian" for r in wb.records), 6)

    def test_verification_is_limited_and_final_audit_idempotent(self):
        wb = Workbench(task())
        for _ in range(2):
            self.assertIn("verification", wb.call("verify", {}))
        self.assertIn("error", wb.call("verify", {}))
        first = wb.finalize()
        count = len(wb.records)
        self.assertEqual(first, wb.finalize())
        self.assertEqual(count, len(wb.records))
        self.assertFalse(first["submitted"])
        self.assertFalse(first["success"])
        self.assertEqual(wb.records[-1]["verdict"]["rays"], 279)

    def test_public_task_rejects_hidden_answer_channels(self):
        t = task()
        t["witness"] = [0.]*6
        with self.assertRaises(ValueError):
            Workbench(t)

    def test_bad_tool_calls_consume_action_allowance(self):
        wb = Workbench(task(), max_actions=2)
        self.assertIn("error", wb.call("enable", dict(surface="both",powers=[True])))
        self.assertIn("error", wb.call("perturb", dict(parameter="exit_p2",delta_mm=float("nan"))))
        self.assertIn("error", wb.call("inspect", {}))
        self.assertEqual(wb.actions, 2)
        self.assertEqual(wb.used, 0)

    def test_actual_function_call_protocol_without_an_api(self):
        commands = [("inspect",{}), ("enable",dict(surface="both", powers=[2,3,4])),
                    ("optimize",dict(max_evaluations=24)), ("verify",{}),
                    ("finish",dict(summary="Submit measured candidate."))]
        import json
        def fake(payload):
            i = sum(item.get("type")=="function_call" for item in payload["input"])
            if i:
                self.assertEqual(payload["input"][-1]["type"], "function_call_output")
            name,args = commands[i]
            return dict(output=[dict(type="function_call",name=name,arguments=json.dumps(args),call_id=f"c{i}")],
                        usage=dict(input_tokens=10,output_tokens=5))
        result = run_episode(task(),fake,model="unit-test-fixture")
        self.assertEqual(result["model_requests"],5)
        self.assertEqual(result["usage"]["input_tokens"],50)
        self.assertTrue(result["outcome"]["success"])
        self.assertEqual(result["status"],"submitted")
        self.assertNotIn("hidden_final_audit", json.dumps(result["conversation"]))

    def test_transport_failure_never_fabricates_a_model_result(self):
        def fail(payload):
            raise RuntimeError("secret error text must not leak")
        result = run_episode(task(),fail,model="unit-test-fixture")
        self.assertEqual(result["model_requests"],0)
        self.assertEqual(result["status"],"transport_error:RuntimeError")
        self.assertFalse(result["outcome"]["success"])
        self.assertNotIn("secret error", str(result))


if __name__ == "__main__":
    unittest.main()
