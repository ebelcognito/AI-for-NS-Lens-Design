"""Fresh-seed confirmation and optical-direction ablation; no parameter tuning."""
import argparse,json,shutil,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nslens.analysis.hard_optical_environments import run,TASKS,SOURCE
from nslens.analysis.mirror_residual_experiment import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 arms=['reuse_only','optical_residual','direction_random'];tasks=['field8','aperture42']
 jobs=[dict(task=t,arm=arm,seed=s,repeat=r,out=str(out)) for t in tasks for r in [1,2,3] for s in range(8) for arm in arms]
 if a.smoke:jobs=[dict(task='field8',arm='direction_random',seed=6,repeat=1,out=str(out))]
 old=json.loads(Path('experiments/hard_optical_20261006/protocol.json').read_text())
 protocol=old|dict(jobs=jobs,tasks={k:TASKS[k] for k in tasks},arms=arms,purpose='Confirm exploratory gains on two selected difficult optical specifications using fresh optimizer seeds, plus two unused-in-prior-hard-sweep donors. Conditional on task selection; no independent task-generalization claim.',repeats=[1,2,3],donors=list(range(8)),original_donors=list(range(6)),new_to_hard_sweep_donors=[6,7],seeding='Add repeat*10000000 to old optimizer and random proposal seed formulas. All repeat values positive, disjoint from original sweep. Same optimizer seed per task/donor/repeat/round across arms.',direction_ablation='Same residual Jacobian, group and magnitude prescription,4 proposals, and trace cost as optical method. Replace each direction with Gaussian direction constrained inward for coordinates within one step norm of bounds; normalize to exact optical step norm at that arm\'s own parent. Curves/all groups and radii unchanged. Histories diverge across methods; step norms need not match across diverged parents. Derivative probes deliberately charged to this mechanistic ablation. Refinement-only control spends them on optimization.',scope='Fixed analytic optical mutation, not a learned model. Six old plus two new-to-hard-sweep donor families;3 fresh optimizer repeats each. Report donor-block uncertainty and new-donor subset separately. Main selection remains coarse-only and one dense audit per search.',source_cost=sum(json.loads((SOURCE/f'cross_count_{3100+s}.json').read_text())['coarse_calls'] for s in range(8)),source_sha256={str(SOURCE/f'cross_count_{3100+s}.json'):hashlib.sha256((SOURCE/f'cross_count_{3100+s}.json').read_bytes()).hexdigest() for s in range(8)})
 save(out/'protocol.json',protocol)
 for f in [Path(__file__),Path('nslens/analysis/hard_optical_environments.py'),Path('nslens/analysis/mirror_residual_moves.py'),Path('nslens/analysis/variable_mirror.py'),Path('nslens/search/cma_me.py')]:shutil.copy(f,out/f.name)
 with ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(run,jobs):print(json.dumps(result),flush=True)
 save(out/'completion.json',dict(runs=len(jobs),coarse_calls=2056*len(jobs),selected_dense_audits=len(jobs)))
if __name__=='__main__':main()
