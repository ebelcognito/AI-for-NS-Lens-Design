import copy
import unittest
from unittest.mock import patch

from nslens.analysis.cma_signal_probe import fork_rng
from nslens.workbench.cma_midpoint_pilot import continuation, fixed_start
from nslens.workbench.interactive_cma_pilot import initial_state, run_batch
from tests.test_interactive_cma_pilot import fake_measure


class SignalProbeTests(unittest.TestCase):
    @patch('nslens.workbench.interactive_cma_pilot.measure', side_effect=fake_measure)
    def test_independent_repeats_preserve_distribution_and_original_state(self, _):
        task = dict(start=[.1] * 14, config={}, initial=fake_measure({}, [.1] * 14))
        original = run_batch(task, initial_state(task), fixed_start(task), 17)['state']
        saved = copy.deepcopy(original)
        a, b = fork_rng(original, 100), fork_rng(original, 101)
        x = run_batch(task, a, continuation(1.), 0)
        y = run_batch(task, a, continuation(1.), 99)
        z = run_batch(task, b, continuation(1.), 0)
        self.assertEqual(original, saved)
        self.assertEqual(x['records'], y['records'])
        self.assertNotEqual(x['records'], z['records'])
        a['optimizer']['snapshot']['rng_state'] = original['optimizer']['snapshot']['rng_state']
        self.assertEqual(a, original)


if __name__ == '__main__':
    unittest.main()
