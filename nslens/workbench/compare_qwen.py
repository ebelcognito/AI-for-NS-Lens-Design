"""Compare actual model runs without interpreting unsubmitted designs as success."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from nslens.analysis.run_repair_curriculum import write_json
from .qwen_baseline import parse_generation


def summarize(root):
    config=json.loads((root/'config.json').read_text())
    episodes=[json.loads(p.read_text()) for p in sorted(root.glob('episode*.json'))]
    assert len(episodes)==len(config['selected_tasks'])
    assert [e['task']['task_id'] for e in episodes]==config['selected_tasks']
    assert all(e['replay_passed'] for e in episodes)
    for episode in episodes:
        calls=[]
        for response in episode['response_log']:
            parsed, errors=parse_generation(response['raw_generation'],0)
            assert errors==response['parse_errors']
            native=[(c['name'],json.loads(c['arguments'])) for c in parsed if c['type']=='function_call']
            saved=[(c['name'],json.loads(c['arguments'])) for c in response['output'] if c['type']=='function_call']
            assert native==saved
            calls.extend(native)
        assert calls==[(e['tool'],e['arguments']) for e in episode['events']]
        for key in ('input_tokens','output_tokens'):
            assert episode['usage'][key]==sum(r['usage'][key] for r in episode['response_log'])
    assert all(sum(e['usage'].values())<=config['max_tokens'] for e in episodes)
    return dict(path=str(root),episodes=len(episodes),families=len({e['task']['family_id'] for e in episodes}),
        successful=sum(e['outcome']['success'] for e in episodes),
        submitted=sum(e['outcome']['submitted'] for e in episodes),
        passing_final_optics=sum(e['outcome']['final_audit']['success'] for e in episodes),
        mean_candidate_calls=statistics.mean(e['outcome']['candidate_calls'] for e in episodes),
        mean_ray_evaluations=statistics.mean(e['outcome']['nominal_ray_evaluations'] for e in episodes),
        mean_dense_checks=statistics.mean(e['outcome']['verification_calls'] for e in episodes),
        mean_rms_mm=statistics.mean(e['outcome']['final_audit']['rms'] for e in episodes),
        mean_total_tokens=statistics.mean(sum(e['usage'].values()) for e in episodes),
        invalid_tool_calls=sum(e['outcome']['invalid_tool_calls'] for e in episodes),
        parse_errors=sum(len(r['parse_errors']) for e in episodes for r in e['response_log']),
        all_replays_passed=True,config=config)


def promotion(baseline, candidate):
    """Conservative pilot gate; a gain in one metric cannot hide another regression."""
    same_budget=baseline['config']['max_tokens']==candidate['config']['max_tokens']
    metrics=('mean_rms_mm','mean_candidate_calls','mean_ray_evaluations','mean_total_tokens')
    no_regression=all(candidate[k] <= baseline[k]+1e-12 for k in metrics)
    meaningful_gain=any(candidate[k] < .99*baseline[k] for k in metrics)
    full_success=candidate['successful']==baseline['successful']==baseline['episodes']==candidate['episodes']
    return dict(promote=bool(same_budget and no_regression and meaningful_gain and full_success),
        checks=dict(same_token_budget=same_budget, all_tasks_successful=full_success,
                    no_mean_metric_regression=no_regression, at_least_one_percent_gain=meaningful_gain),
        metrics=list(metrics),scope='exploratory four-family pilot, not a general engineering certification',
        baseline=baseline['path'],candidate=candidate['path'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--promotion-out',type=Path)
    a=p.parse_args()
    reports=[summarize(root) for root in a.runs]
    expected=reports[0]['config']['selected_tasks']
    assert all(r['config']['selected_tasks']==expected for r in reports)
    reference=reports[0]['config']
    for report in reports[1:]:
        cfg=report['config']
        for key in ('model','revision','seed','decoding','thinking','dtype','tasks_sha256','max_requests','max_output_tokens_per_request'):
            assert cfg[key]==reference[key], f'mismatched comparison: {key}'
        for path in ('nslens/workbench/domain.py','nslens/workbench/tools.py','nslens/workbench/model_baseline.py'):
            assert cfg['sources'][path]==reference['sources'][path],f'changed environment/runner: {path}'
    # Different token ceilings are an explicitly labeled budget ablation.
    for report in reports:
        report['token_ceiling']=report['config']['max_tokens']
    write_json(a.out,reports)
    if a.promotion_out:
        assert len(reports)==2
        write_json(a.promotion_out,promotion(*reports))
    for report in reports:
        print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))


if __name__=='__main__':
    main()
