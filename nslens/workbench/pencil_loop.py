"""Bounded Qwen planner curriculum on physically verified narrow-field folded repairs.

The learned module selects a numerical repair plan. It does not propose new topology.
All counterfactual optical outcomes are private to the experiment controller.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .folded_domain import evaluate,PARAMETERS
from .folded_tools import FoldedWorkbench
from .pencil_screen import pencil_config
from .tools import tool,canonical_hash

PLANS=[dict(face=face,step_mm=step) for face in ['all','A','B','T'] for step in [.25,.5]]
INSTRUCTIONS='''You choose a numerical repair plan for a 2-D folded freeform optical element.
The source field spans at most 1 mm with a finite angular fan. Every useful ray must
undergo TIR. Meet the fixed per-field output-angle targets within their tolerances,
throughput and angular-spread requirements. Use initial measured diagnostics and the
surface prescription to choose exactly one execute_plan call. The eight available plans
spend the same 12 candidate traces. Each performs bounded coordinate search, restores
its best coarse prescription, and submits it for a hidden dense audit. Plan identifiers
are indices in the supplied plans array. You cannot inspect alternate outcomes or
change the task requirements. This is repair-plan selection, not topology discovery.'''
TOOLS=[tool('execute_plan','Choose one numerical repair plan; consumes the entire 12-candidate budget.',
            {'plan_id':dict(type='integer',minimum=0,maximum=7)})]


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class PencilWorkbench(FoldedWorkbench):
    @staticmethod
    def validate_task(task):
        if set(task)!={'task_id','family_id','split','base','start','requirements','layout','constraints','profile'}:
            raise ValueError('unexpected pencil task fields')
        if task['profile']!='pencil-angle-v1':raise ValueError('unknown pencil profile')
        layout,req=pencil_config(min_folds=1)
        supplied=task['requirements']
        if task['layout']!=layout or set(supplied)!=set(req)|{'target_centroids_deg','centroid_tolerance_deg'}:
            raise ValueError('pencil contract mismatch')
        if any(supplied[k]!=v for k,v in req.items()):raise ValueError('fixed requirement changed')
        target=np.asarray(supplied['target_centroids_deg'])
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('invalid target')
        if supplied['centroid_tolerance_deg']!=.15:raise ValueError('invalid tolerance')
        if task['constraints']!=dict(editable=list(PARAMETERS),candidate_budget=12,max_delta_mm=4.):
            raise ValueError('fixed planner budget or access changed')
    def evaluate(self,x,fidelity):
        return pencil_evaluate(self.task,x,fidelity)


def pencil_evaluate(task,x,fidelity):
    v=evaluate(task['base'],x,fidelity,layout=task['layout'],requirements=task['requirements'])
    if v['fields']:
        errors=[f['centroid_deg']-target for f,target in zip(v['fields'],task['requirements']['target_centroids_deg'])]
        tol=task['requirements']['centroid_tolerance_deg']
        v['centroid_errors_deg']=errors
        v['success']=bool(v['success'] and max(map(abs,errors))<=tol)
        v['merit']+=10*sum(max(0,abs(e)/tol-1) for e in errors)
        if not v['success']:v['reason']='pencil routing or target-angle gate'
    return v


def observation(task,initial):
    # Deliberately exclude split/family/task identifiers from the policy prompt.
    public={k:copy.deepcopy(v) for k,v in task.items() if k not in ['split','family_id','task_id']}
    return dict(task=public,initial_diagnostics=initial,plans=PLANS)


def plan_episode(job):
    task,index=job;wb=PencilWorkbench(task);plan=PLANS[index]
    faces=['A','B','T','D'] if plan['face']=='all' else [plan['face']]
    wb.call('enable',dict(faces=faces,powers=[2,4]))
    wb.call('optimize',dict(max_evaluations=12,step_mm=plan['step_mm']))
    wb.call('restore_best',{});wb.call('finish',dict(summary='Fixed numerical repair plan.'))
    outcome=wb.finalize()
    return dict(task_id=task['task_id'],plan_id=index,events=wb.events,evaluations=wb.records,outcome=outcome)


def replay(task,episode):
    wb=PencilWorkbench(task)
    for event in episode['events']:
        assert canonical_hash(wb.state())==event['before_hash']
        assert wb.call(event['tool'],event['arguments'])==event['output']
        assert canonical_hash(wb.state())==event['after_hash']
    assert wb.finalize()==episode['outcome']
    assert wb.records==episode['evaluations']
    return True


def build(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    seeds=[r for r in json.loads(Path('runs/pencil_singlefold_screen_20260910_v1/screen.json').read_text()) if r['passed']]
    # Fixed assignment before planner rollouts: first two train, third validation, fourth test.
    tasks=[];private=[];attempts=[]
    rng=np.random.default_rng(20260914)
    for si,seed in enumerate(seeds):
        split=['train','train','validation','test'][si]
        layout,requirements=pencil_config(min_folds=1)
        requirements['target_centroids_deg']=[f['centroid_deg'] for f in seed['audit']['fields']]
        requirements['centroid_tolerance_deg']=.15
        accepted=0
        # Bound total screening cost, and never filter on Qwen outcomes.
        for attempt in range(36):
            x=np.array(seed['coefficients']);face=['A','B','T'][attempt%3]
            i=PARAMETERS.index(face+'_p2');delta=float(rng.choice([-1,1])*rng.uniform(.45,1.5))
            x[i]=np.clip(x[i]+delta,-2,2)
            task=dict(task_id=f'pencil-{si}-{accepted}',family_id=f'archive-{seed["cell"][0]}-{seed["cell"][1]}',
                split=split,profile='pencil-angle-v1',base=seed['base'],start=x.tolist(),layout=layout,
                requirements=requirements,constraints=dict(editable=list(PARAMETERS),candidate_budget=12,max_delta_mm=4.))
            initial=pencil_evaluate(task,x,'coarse')
            attempts.append(dict(seed=si,attempt=attempt,coefficients=x.tolist(),verdict=initial))
            if not initial['physical_valid'] or initial['success']:continue
            dense=pencil_evaluate(task,x,'audit')
            if dense['success']:continue
            if any(t['base']==task['base'] and t['start']==task['start'] for t in tasks):continue
            tasks.append(copy.deepcopy(task))
            private.append(dict(task_id=task['task_id'],witness=seed['coefficients'],face=face,initial=initial,
                witness_audit=pencil_evaluate(task,seed['coefficients'],'audit'),
                witness_convergence=pencil_evaluate(task,seed['coefficients'],'convergence')))
            assert private[-1]['witness_audit']['success'] and private[-1]['witness_convergence']['success']
            accepted+=1
            if accepted==4:break
        print('seed',si,split,'tasks',accepted,flush=True)
    write(out/'tasks.json',tasks);write(out/'private_generation.json',dict(witnesses=private,attempts=attempts))
    write(out/'protocol.json',dict(seed=20260914,rounds=2,arms=['uniform','fixed','adaptive'],
        train_tasks_per_round=4,optimizer_updates_per_round=8,training_sequence_length=4096,
        plan_calls=12,plans=PLANS,task_sha256=hashlib.sha256((out/'tasks.json').read_bytes()).hexdigest(),
        scope='Bounded planner; within archival design space, unresolved ancestry. One training seed pilot.',
        selection='Adaptive 75% mass on current training failures with a successful teacher; 25% uniform coverage.',
        cost='Common exhaustive teacher cache charged once to every arm; same planner scoring and optimizer-update caps.'))
    jobs=[(t,i) for t in tasks for i in range(len(PLANS))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(plan_episode,jobs):
            write(out/'private_plans'/f'{row["task_id"]}_plan{row["plan_id"]}.json',row)
            print(row['task_id'],row['plan_id'],row['outcome']['success'],flush=True)
    write(out/'build_complete.json',dict(tasks=len(tasks),episodes=len(jobs)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args();build(a.out,a.workers)

if __name__=='__main__':main()
