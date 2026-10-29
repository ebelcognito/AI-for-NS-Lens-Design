"""Report completed repair self-play results without checkpoint selection."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from nslens.analysis.azr_repair_pilot import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run;corpus=root/'corpus';learning=root/'learning'
    if not (learning/'completion.json').exists():raise ValueError('Learning/evaluation not complete')
    summary=json.loads((learning/'summary.json').read_text())
    tasks=json.loads((corpus/'tasks.json').read_text())
    admission=json.loads((corpus/'admission.json').read_text())
    generation=[json.loads(p.read_text()) for p in (corpus/'private_generation').glob('*.json')]
    outcomes=[json.loads(p.read_text()) for p in (corpus/'private_outcomes').glob('*.json')]
    costs=dict(admission_rays=sum(x[k]['rays'] for x in admission for k in ('coarse','audit','stress','capacity')),
               generation_rays=sum(x['result']['rays'] for row in generation for x in row['attempts']),
               counterfactual_solver_rays=sum(r['rays'] for r in outcomes),
               counterfactual_solver_evaluations=sum(r['evaluations'] for r in outcomes))
    replay=corpus/'replay_check.json'
    costs['replay_rays']=json.loads(replay.read_text())['additional_rays'] if replay.exists() else 0
    protocol=json.loads((corpus/'protocol.json').read_text())
    drift=[]
    for path,digest in protocol['sources'].items():
        p=Path(path)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:drift.append(path)
    # Paired intervals are clustered by starting family, not by corruption or decoding seed.
    base=json.loads((learning/'test_frozen.json').read_text())
    b={r['task_id']:r for r in base};deltas={}
    for arm in ('uniform','learned_proposer'):
        grouped=collections.defaultdict(list)
        for seed in (8123,8124):
            for r in json.loads((learning/f'test_{arm}_seed{seed}_round2.json').read_text()):
                grouped[r['family_id']].append(float(r['success'])-float(b[r['task_id']]['success']))
        v=np.array([np.mean(x) for x in grouped.values()])
        rng=np.random.default_rng(31)
        means=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
        deltas[arm]=dict(families=len(v),success_rate_delta=float(v.mean()),
                         family_bootstrap_95ci=np.quantile(means,[.025,.975]).tolist(),
                         caveat='Exploratory interval from only four test families.')
    write(root/'report.json',dict(costs=costs,source_drift=drift,comparisons_to_frozen=deltas,summary=summary))
    lines=['# Optical repair / bounded AZR-style pilot results','',
           'Quality floor: 80% worst-field folded throughput and 80% root support; fixed n=1.50.',
           'This tests quality recovery, not thickness reduction. Ten starting families produced '+str(len(tasks))+' degraded tasks.',
           'Test results cover twelve corruptions of four held-out brief/range families. All checkpoints are reported; none was selected on test performance.','',
           '| System | Test successes | Expected success | Mean worst-field throughput |',
           '|---|---:|---:|---:|']
    for key,value in summary.items():
        if not key.startswith('test_') or key=='test_oracle':continue
        lines.append(f"| {key[5:]} | {value['successes']}/{value['tasks']} | {value['expected_success']:.1%} | {value['mean_throughput']:.1%} |")
    oracle=summary['test_oracle']
    lines.extend(['',f"At least one numerical plan repaired {oracle['successes']}/{oracle['tasks']} test tasks. This all-plan oracle is an upper bound, not a matched-budget competitor.",'',
                  '## Interpretation boundaries','',
                  '- The proposer selects among six cached perturbation recipes. The solver selects among six CMA configurations; neither produces unrestricted geometry edits.',
                  '- Qwen adapters are updated using exact finite-action optical reward gradients. This is an AZR-style feasibility experiment, not the original AZR algorithm.',
                  '- Uniform and learned-proposer arms have equal solver updates; proposer overhead is extra and recorded.',
                  '- The generated pool and counterfactual cache are reused. Future experiments must test fresh task generation and more independent families.',
                  '- There is no external-engine or manufacturing validation.','',
                  '## Physical computation','',f"Admission rays: {costs['admission_rays']:,}; degradation screening rays: {costs['generation_rays']:,}; numerical repair/cache rays: {costs['counterfactual_solver_rays']:,}.",'',
                  'Training and inference cost per arm is in `learning/*/cost.json`. Cached evaluations were actually executed and are not free data.',
                  '',f'Source drift relative to corpus launch: {drift}.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,comparisons=deltas,costs=costs),indent=2))


if __name__=='__main__':main()
