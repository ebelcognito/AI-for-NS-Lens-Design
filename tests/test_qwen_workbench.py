import unittest
from nslens.workbench.qwen_baseline import chat_messages, parse_generation


class QwenProtocolTests(unittest.TestCase):
    def test_native_tool_round_trip(self):
        raw = 'Use the optimizer.\n<tool_call>{"name":"optimize","arguments":{"max_evaluations":24}}</tool_call>'
        outputs, errors = parse_generation(raw, 1)
        self.assertEqual(errors, [])
        payload = dict(instructions='system', input=[dict(role='user',content='task'), *outputs,
            dict(type='function_call_output',call_id='qwen_1_0',output='{"ok":true}')])
        messages = chat_messages(payload)
        self.assertEqual([m['role'] for m in messages], ['system','user','assistant','tool'])
        self.assertEqual(messages[2]['tool_calls'][0]['function']['arguments'], {'max_evaluations':24})
        self.assertEqual(messages[3]['content'], '{"ok":true}')

    def test_next_input_is_reserved_before_generation(self):
        from dataclasses import asdict
        from nslens.learning.freeform_repair import ImagingSpec
        from nslens.workbench.model_baseline import run_episode
        class TooLarge:
            def input_token_count(self, payload):
                return 101
            def __call__(self, payload):
                raise AssertionError("oversized request must not be sent")
        spec = ImagingSpec("unit", "train")
        task = dict(task_id="unit/task", family_id="unit", split="train", spec=asdict(spec),
                    start=[0.]*6, objective="Repair image")
        result = run_episode(task, TooLarge(), model="fixture", max_tokens=100)
        self.assertEqual(result["status"], "token_limit")
        self.assertEqual(result["model_requests"], 0)
        self.assertFalse(result["outcome"]["submitted"])

    def test_promotion_rejects_hidden_regressions(self):
        from nslens.workbench.compare_qwen import promotion
        baseline=dict(path='base',config={'max_tokens':64000},episodes=12,successful=12,
                      mean_rms_mm=.003,mean_candidate_calls=21,mean_ray_evaluations=847,mean_total_tokens=30000)
        candidate=dict(baseline,path='adapter',mean_total_tokens=25000)
        self.assertTrue(promotion(baseline,candidate)['promote'])
        candidate['successful']=11
        self.assertFalse(promotion(baseline,candidate)['promote'])
        candidate['successful']=12
        candidate['mean_rms_mm']=.004
        self.assertFalse(promotion(baseline,candidate)['promote'])
        candidate['mean_rms_mm']=.003
        candidate['config']={'max_tokens':128000}
        self.assertFalse(promotion(baseline,candidate)['promote'])

    def test_training_rejects_validation_rows_before_tokenization(self):
        import json
        from nslens.workbench.train_qwen import examples
        row=dict(messages=[{'role':'system','content':''},
                           {'role':'user','content':json.dumps({'split':'validation','family_id':'held-out'})}],
                 metadata={'family_id':'held-out'})
        with self.assertRaisesRegex(ValueError,'boundary'):
            examples([row],None)

    def test_teacher_selection_uses_complete_training_outcomes(self):
        from nslens.workbench.train_qwen import select_training_rows
        episodes={}
        for strategy,errors in [('robust',[.001,.002]),('fragile',[.0001,.2])]:
            for i,error in enumerate(errors):
                episodes[(str(i),strategy)]=dict(strategy=strategy,task={'task_id':str(i),'split':'train'},
                    outcome={'success':error<.03,'final_audit':{'rms':error}})
        rows=[{'metadata':{'strategy':s}} for s in ('robust','fragile')]
        selected,detail=select_training_rows(rows,episodes,'robust_plan')
        self.assertEqual(detail['chosen'],'robust')
        self.assertEqual(selected,[rows[0]])
        episodes[('0','robust')]['task']['split']='test'
        with self.assertRaisesRegex(ValueError,'training'):
            select_training_rows(rows,episodes,'robust_plan')

    def test_malformed_calls_never_execute(self):
        for raw in ['<tool_call>{"name":"inspect","arguments":{}}',
                    '<tool_call>{"name":"perturb","arguments":{"delta_mm":NaN}}</tool_call>',
                    '<tool_call>{"name":"inspect","arguments":{}}</tool_call><tool_call>bad</tool_call>']:
            outputs, errors = parse_generation(raw, 1)
            self.assertTrue(errors)
            self.assertFalse(any(o['type']=='function_call' for o in outputs))


if __name__ == '__main__':
    unittest.main()
