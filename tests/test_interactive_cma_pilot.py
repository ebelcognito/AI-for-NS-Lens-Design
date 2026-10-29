import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from nslens.workbench.interactive_cma_pilot import (
    initial_state, observation, run_batch, validate_action, digest,
    run_branch, read,
)


def fake_measure(config, genes, *args):
    value = float(.7 - .1*np.sum(np.square(genes)))
    return dict(success=False, measured=True, throughput=value, support=.7,
                fields=[], violations={}, reason=None, route=None, rays=27)


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(config=dict(codec=dict(corners=3), brief=dict(aim_deg=-90,
                          min_folded_throughput=.8, min_root_fraction=.8)),
                         start=[.1]*14, initial=fake_measure({}, [.1]*14),
                         family_id='private_family', witness=[.2]*14, style='secret')
        self.action = dict(operation='restart', parameters=[0,12], scales=[.025,.04],
                           radius=.1, rationale='test')

    def test_private_fields_absent(self):
        obs = observation(self.task, initial_state(self.task))
        self.assertNotIn('private_family', str(obs))
        self.assertNotIn('secret', str(obs))
        self.assertNotIn('witness', str(obs))

    def test_invalid_control_rejected(self):
        for field, value in [('parameters',[14]), ('parameters',[0,0]),
                             ('scales',[float('nan'),.04]), ('radius',0)]:
            bad = {**self.action, field:value}
            with self.assertRaises(ValueError):
                validate_action(bad, 14, False)
        with self.assertRaises(ValueError):
            validate_action(dict(operation='continue',sigma_multiplier=1.,rationale='x'),14,False)

    @patch('nslens.workbench.interactive_cma_pilot.finalize', return_value={'success':False})
    @patch('nslens.workbench.interactive_cma_pilot.measure', side_effect=fake_measure)
    def test_exhausted_model_does_not_execute_fallback(self, measure_mock, final_mock):
        class FailedQwen:
            def decide(self, *args):
                raise RuntimeError('model failed three valid-action attempts: test')
        task = {**self.task, 'task_id':'dev_test'}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_branch((root,task,self.action,0,FailedQwen()))
            branch = read(root/'episodes/dev_test/branch_0.json')
            self.assertEqual(branch['status'],'model_failure')
            self.assertEqual(len(branch['actions']),1)
            self.assertEqual(measure_mock.call_count,16)
            final_mock.assert_called_once()

    @patch('nslens.workbench.interactive_cma_pilot.measure', side_effect=fake_measure)
    def test_budget_frozen_genes_replay_and_continue(self, mock):
        state = initial_state(self.task)
        original = copy.deepcopy(state)
        a = run_batch(self.task, state, self.action, 17)
        b = run_batch(self.task, state, self.action, 17)
        self.assertEqual(state, original)
        self.assertEqual(a['records'], b['records'])
        self.assertEqual(digest(a['state']), digest(b['state']))
        self.assertEqual(len(a['records']),16)
        for row in a['records']:
            for i, value in enumerate(row['genes']):
                if i not in (0,12): self.assertEqual(value,.1)
                self.assertLessEqual(abs(value-.1),.1+1e-12)
        c = run_batch(self.task, a['state'], dict(operation='continue',sigma_multiplier=1.,rationale='x'),18)
        self.assertEqual(c['generations'][0]['before'], a['generations'][-1]['after'])
        self.assertEqual(c['state']['candidates'],32)
        self.assertEqual(c['state']['optimizer']['snapshot']['generation'],4)
        with self.assertRaises(ValueError):
            run_batch(self.task,c['state'],self.action,19)


if __name__ == '__main__': unittest.main()
