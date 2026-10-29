import copy
import unittest
from unittest.mock import patch

from nslens.workbench.cma_midpoint_pilot import (
    continuation, evaluate_action, fixed_start,
)
from nslens.workbench.interactive_cma_pilot import initial_state, run_batch
from tests.test_interactive_cma_pilot import fake_measure


class MidpointTests(unittest.TestCase):
    @patch('nslens.workbench.cma_midpoint_pilot.finalize', return_value={'success': False})
    @patch('nslens.workbench.interactive_cma_pilot.measure', side_effect=fake_measure)
    def test_fork_identity_and_deterministic_continuation(self, measure, finalize):
        task = dict(start=[.1]*14, config={}, initial=fake_measure({}, [.1]*14))
        first = run_batch(task, initial_state(task), fixed_start(task), 17)
        saved = copy.deepcopy(first['state'])
        a = evaluate_action(task, first['state'], continuation(1.), 17)
        b = evaluate_action(task, first['state'], continuation(1.), 17)
        c = evaluate_action(task, first['state'], continuation(.5), 17)
        self.assertEqual(first['state'], saved)
        self.assertEqual(a['batch']['records'], b['batch']['records'])
        self.assertEqual(a['midpoint_hash'], c['midpoint_hash'])
        self.assertNotEqual(a['batch']['records'], c['batch']['records'])
        for arm in (a, b, c):
            self.assertEqual(arm['batch']['state']['candidates'], 32)
            self.assertEqual(len(arm['batch']['records']), 16)
        self.assertEqual(a['batch']['generations'][0]['before'],
                         first['generations'][-1]['after'])


if __name__ == '__main__':
    unittest.main()
