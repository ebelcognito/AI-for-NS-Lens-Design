"""Check task separation, curriculum feedback, native decisions and training budgets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .pencil_loop import write
from .train_pencil_loop import select_tasks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('run',type=Path)
    a=p.parse_args();tasks=json.loads((a.corpus/'tasks.json').read_text());by_id={t['task_id']:t for t in tasks}
    config=json.loads((a.run/'config.json').read_text());eligible=[by_id[k] for k in config['eligible_train']]
    budgets=[];targets=[];adapters=[]
    for arm in ['uniform','fixed','adaptive']:
        previous=json.loads((a.run/'baseline_development.json').read_text())
        for round_index in [1,2]:
            selection=json.loads((a.run/arm/f'selection{round_index}.json').read_text())
            chosen,weights=select_tasks(arm,eligible,{r['task_id']:r for r in previous},np.random.default_rng(20260913+round_index))
            assert selection['tasks']==[t['task_id'] for t in chosen] and selection['weights']==weights
            assert all(by_id[k]['split']=='train' for k in selection['feedback_task_ids'])
            root=a.run/arm/f'round{round_index}'
            training=json.loads((root/'training.json').read_text())
            assert sorted(r['task_id'] for r in training['selection'])==sorted(selection['tasks'])
            for selected in training['selection']:
                assert by_id[selected['task_id']]['split']=='train'
                teacher=json.loads((a.corpus/'private_plans'/f'{selected["task_id"]}_plan{selected["plan_id"]}.json').read_text())
                assert teacher['outcome']['success']
            assert len(training['steps'])==8
            budgets.append(training['padded_training_tokens']);targets.append(training['supervised_tokens'])
            adapters.append(dict(arm=arm,round=round_index,sha256=hashlib.sha256((root/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest()))
            previous=json.loads((root/'development.json').read_text())
    assert len(set(budgets))==1 and len(set(targets))==1
    evaluated=0
    files=list(a.run.glob('test_*.json'))+[a.run/'baseline_development.json']+list(a.run.glob('*/round*/development.json'))
    for f in files:
        for row in json.loads(f.read_text()):
            task=by_id[row['task_id']]
            assert (task['split']=='test')==f.name.startswith('test_')
            response=row['response'];calls=[c for c in response['output'] if c['type']=='function_call']
            if row['plan_id'] is not None:
                assert len(calls)==1 and calls[0]['name']=='execute_plan'
                assert json.loads(calls[0]['arguments'])==dict(plan_id=row['plan_id'])
                oracle=json.loads((a.corpus/'private_plans'/f'{task["task_id"]}_plan{row["plan_id"]}.json').read_text())
                assert row['outcome']==oracle['outcome'] and row['success']==oracle['outcome']['success']
            else:assert not row['success'] and row['outcome'] is None
            evaluated+=1
    write(a.run/'audit.json',dict(passed=True,evaluated_decisions=evaluated,adapters=adapters,
        per_round_padded_tokens=budgets[0],per_round_supervised_tokens=targets[0],
        caveat='Split checks establish prescription separation, not independent evolutionary ancestry.'))
    print('Training audit passed:',evaluated,'decisions, six adapters',flush=True)

if __name__=='__main__':main()
