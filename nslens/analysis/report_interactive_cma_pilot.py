"""Audit and summarize the interactive development pilot without retraining."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nslens.workbench.interactive_cma_pilot import (
    SEED, initial_state, run_batch, finalize, read, report, write,
)


def audit_replay(root, task):
    saved = read(root/'episodes'/task['task_id']/'branch_0.json')
    state = initial_state(task)
    rays, started = 0, time.monotonic()
    for i, action in enumerate(saved['actions']):
        batch = run_batch(task, state, action, SEED+i)
        for key in ('state', 'event', 'records', 'generations'):
            if batch[key] != saved['batches'][i][key]:
                raise AssertionError(f'replay mismatch: batch {i}, {key}')
        rays += batch['event']['candidate_rays']
        state = batch['state']
    outcome = finalize(task, state)
    for key in saved['outcome']:
        if key != 'audit_seconds' and outcome[key] != saved['outcome'][key]:
            raise AssertionError(f'audit replay mismatch: {key}')
    return dict(task_id=task['task_id'], branch=0, exact=True,
                evaluations=35, source_rays=outcome['source_rays'],
                elapsed_seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    root = args.root
    tasks = read(root/'tasks.json')
    summary = report(root, tasks)
    if args.replay:
        write(root/'replay.json', audit_replay(root, tasks[0]))
    generation = [read(f) for f in (root/'private_generation').glob('*.json')]
    gen_cost = dict(evaluations=sum(len(g['attempts']) for g in generation),
                    rays=sum(a['result']['rays'] for g in generation for a in g['attempts']))
    write(root/'generation_cost.json', gen_cost)
    counts = Counter()
    actions = Counter()
    for row in summary['rows']:
        counts.update(row['followup_operations'])
        for a in row['initial_actions']:
            actions[tuple(a['parameters'])] += 1
    all_branches = [read(p) for p in (root/'episodes').glob('*/branch_*.json')]
    invalid = sum(b['event']['invalid_candidates'] for r in all_branches for b in r['batches'])
    audit_losses = sum(r['batches'][-1]['state']['result']['success'] and not r['outcome']['success']
                       for r in all_branches)
    wins = sum(r['successes'][0] and not r['baseline_success'] for r in summary['rows'])
    losses = sum(r['baseline_success'] and not r['successes'][0] for r in summary['rows'])
    extra = dict(followup_operations=dict(counts), distinct_initial_parameter_subsets=len(actions),
                 invalid_qwen_candidates=invalid, search_successes_failing_audits=audit_losses,
                 first_branch_only_wins=wins, baseline_only_wins=losses)
    write(root/'diagnostics.json', extra)
    if summary['rows']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values = np.array([[r['baseline_reward']]+r['rewards'] for r in summary['rows']])
        flags = np.array([[r['baseline_success']]+r['successes'] for r in summary['rows']])
        fig, ax = plt.subplots(figsize=(8, max(5, len(values)*.24)))
        im = ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        for i in range(len(values)):
            for j in range(5):
                ax.text(j,i,'PASS' if flags[i,j] else f'{values[i,j]:.2f}',
                        ha='center',va='center',fontsize=7,color='white' if values[i,j]>.7 else 'black')
        ax.set_xticks(range(5),['Fixed CMA','Qwen 1','Qwen 2','Qwen 3','Qwen 4'])
        ax.set_yticks(range(len(values)),[r['task_id'] for r in summary['rows']],fontsize=7)
        ax.set_title('Development repair outcomes | 32 candidates per column per task')
        fig.colorbar(im, ax=ax, label='Verified reward (PASS requires dense audits)')
        fig.tight_layout()
        fig.savefig(root/'outcomes.png',dpi=150)
        plt.close(fig)
    n=summary['completed_tasks']
    replay=read(root/'replay.json') if (root/'replay.json').exists() else None
    text=f'''# Interactive Qwen/CMA development pilot

Status: {"complete" if summary['complete'] else "partial"}; {n}/{len(tasks)} tasks.
Model: local `qwen3.8-27b`, Q8_0 GGUF. Model weights were **not trained**.

## Results

| Measure | Result |
|---|---:|
| Individual Qwen continuations verified | {summary['verified_qwen_repairs']}/{summary['qwen_continuations']} |
| First listed Qwen continuation verified | {summary['first_branch_repairs']}/{n} |
| Fixed all-parameter CMA verified | {summary['fixed_baseline_repairs']}/{n} |
| At least one of four Qwen continuations verified | {summary['any_of_four_repairs']}/{n} |
| Tasks with mixed pass/fail across Qwen branches | {summary['tasks_with_success_disagreement']} |
| Tasks with verified reward spread >= 0.05 | {summary['tasks_with_reward_gap_005']} |
| Candidate trajectory preference pairs | {summary['trajectory_preference_pairs']} |

The four-branch result spends four times the numerical budget of one branch or the
fixed baseline; it is a diagnostic ceiling, not a fair single-policy comparison.
First-branch versus baseline paired discordances: {wins} Qwen-only wins, {losses} baseline-only wins.

## Interface and data

32 fresh degraded starts drawn from {summary['task_families']} previously admitted train/validation
families. Existing test families were excluded. Witnesses establish feasibility;
the model never receives witnesses, corruption recipes, family labels or sibling outcomes.
Related starts remain related: this is not a 32-independent-family benchmark.

The model first proposes four alternative parameter subsets, per-parameter scales and
trust-region radii in one response. Each alternative gets 16 CMA candidates, followed
by a separate model call that sees only its own feedback and chooses another 16-candidate
batch. Restart resets CMA around the best retained geometry; continue preserves covariance,
mean, paths and RNG and can multiply sigma. All optical requirements are immutable.

Observations contain normalized genes, physical corners/sags, per-field throughput,
root support, detector centroid/RMS, energy violations and optimization history.
They contain no images or surface-specific loss/TIR-margin maps. Free-text diagnoses
are hypotheses; surface-cause claims are not independently established by this report.
No direct edits or separately charged probes are included in this version.

Follow-up actions: {dict(counts)}. Distinct initial parameter subsets: {len(actions)}.
Invalid Qwen candidates: {invalid}. Provisional search passes rejected by final audit: {audit_losses}.
Every continuation includes 5x25 audit and 7x51 stress sampling; the stress run is repeated
at 400,000 branch capacity and must agree exactly with the 200,000-capacity result.

## Costs and validation

- Fresh task generation: {gen_cost['evaluations']} evaluations, {gen_cost['rays']} source rays,
  including rejected attempts. Original witness-admission computation is reused, not rerun.
- Qwen plus baseline numerical search: {summary['candidate_evaluations']} candidate evaluations;
  {summary['audit_evaluations']} final audits; {summary['solver_source_rays']} total source rays.
- Model: {summary['model_requests']} requests, {summary['input_tokens']} input tokens,
  {summary['output_tokens']} output tokens (including reasoning); {summary['rejected_model_responses']} rejected responses.
- Summed model request latency: {summary['model_seconds']/60:.1f} minutes. Concurrent requests
  overlap; this sum is not wall-clock runtime or GPU-hours.
- Numerical replay: {json.dumps(replay)}.

## What the preference file means

`trajectory_preferences.json` contains pairs whose dense-audit reward differs by at
least 0.05. These compare complete two-action continuations from the same initial state.
They are candidate training evidence, not independent samples, and not a ready-to-use
single-turn DPO dataset: a sequential trainer must retain the intervening observations.
Each strategy has only one numerical seed. Differences may reflect search luck as well
as the control choice; no causal or trained-policy improvement is established.

Artifacts: `summary.json`, `diagnostics.json`, `tasks.json`, `model_calls/`, `episodes/`,
`private_generation/`, `trajectory_preferences.json`, `source_snapshot/`, `outcomes.png`.
'''
    (root/'RESULTS.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__ == '__main__':main()
