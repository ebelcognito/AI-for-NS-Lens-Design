"""Additional paired diagnostics and a task-level chart for the midpoint pilot."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from nslens.workbench.interactive_cma_pilot import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    rows = [read(p) for p in sorted((root/'episodes').glob('*/result.json'))]
    actions = Counter()
    per_task = []
    midpoint_groups = {'provisionally_passing': [], 'still_failing': []}
    invalid = audit_losses = 0
    for row in rows:
        midpoint = read(root/'episodes'/row['key']/'midpoint.json')
        group = 'provisionally_passing' if midpoint['state']['result']['success'] else 'still_failing'
        midpoint_groups[group].append(row)
        action = row['arms']['qwen']['action']
        if action:
            label = (f'continue sigma x{action["sigma_multiplier"]}'
                     if action['operation'] == 'continue' else 'restart')
            actions[label] += 1
            arm = read(root/'episodes'/row['key']/'qwen.json')
            invalid += arm['batch']['event']['invalid_candidates']
            audit_losses += bool(arm['batch']['state']['result']['success'] and
                                 not arm['outcome']['success'])
    for task_id in sorted({r['task_id'] for r in rows}):
        group = [r for r in rows if r['task_id'] == task_id]
        counts = {name: sum(r['arms'][name]['outcome']['success'] for r in group)
                  for name in ['fixed', 'half', 'double', 'qwen']}
        delta = [r['arms']['qwen']['outcome']['reward'] - r['arms']['fixed']['outcome']['reward']
                 for r in group]
        per_task.append(dict(task_id=task_id, family_id=group[0]['family_id'], seeds=len(group),
                             successes=counts, qwen_minus_fixed_rewards=delta))
    groups = {name: dict(cases=len(group), verified_successes={
        arm: sum(r['arms'][arm]['outcome']['success'] for r in group)
        for arm in ['fixed', 'half', 'double', 'qwen']}) for name, group in midpoint_groups.items()}
    extra = dict(action_counts=dict(actions), midpoint_subgroups=groups,
                 subgroup_scope='Exploratory; midpoint success uses coarse search measurements.',
                 invalid_qwen_candidates=invalid,
                 provisional_qwen_passes_rejected_by_audit=audit_losses,
                 per_task=per_task,
                 tasks_qwen_more_successes=sum(t['successes']['qwen'] > t['successes']['fixed'] for t in per_task),
                 tasks_fixed_more_successes=sum(t['successes']['qwen'] < t['successes']['fixed'] for t in per_task))
    write(root/'diagnostics.json', extra)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['fixed', 'half', 'double', 'qwen']
    values = np.array([[t['successes'][n] for n in names] for t in per_task])
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(values, vmin=0, vmax=3, cmap='YlGnBu', aspect='auto')
    for i, task in enumerate(per_task):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]}/{task["seeds"]}', ha='center', va='center',
                    color='white' if values[i,j] >= 2 else 'black', fontsize=8)
    ax.set_xticks(range(4), ['Fixed continuation', 'Half step', 'Double step', 'Frozen Qwen'])
    ax.set_yticks(range(len(per_task)), [t['task_id'] for t in per_task], fontsize=8)
    ax.set_title('Verified repairs across three numerical seeds\nSame midpoint; 16 further candidates per arm')
    fig.tight_layout()
    fig.savefig(root/'outcomes.png', dpi=160)
    plt.close(fig)
    print(extra['action_counts'])


if __name__ == '__main__':
    main()
