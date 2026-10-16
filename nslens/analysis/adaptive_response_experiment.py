"""Fresh equal-budget comparison for adaptive optical response proposal reuse."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,shutil,time
from pathlib import Path
import joblib
import numpy as np
from nslens.analysis.adaptive_response_adapter import search
from nslens.analysis.adapter_refinement import refine_state
from nslens.analysis.mirror_adapter_pilot import save,TASKS
from nslens.analysis.response_adapter_experiment import run as run_control
from nslens.analysis.routing_transition_experiment import prepare_start
from nslens.analysis.optical_refinement import audit


def run(job):
    s,name,arm,out,policy_path=job
    if arm not in ('adaptive_response','adaptive_repair'):return run_control(job)
    task=TASKS[name];d=s['donor'];seed=181000+d*100
    warm=refine_state(s['state'],task,seed=seed,budget=65)
    policy=joblib.load(policy_path)['retrieval'] if arm=='adaptive_repair' else None
    r=search(warm['state'],task,257,seed,policy)
    final=refine_state(r['state'],task,seed=seed+900,budget=65);x=np.asarray(final['state']);dense=audit(x[:9],3,2*x[9:],task)
    save(Path(out)/f'{d}_{name}_{arm}.json',dict(donor=d,task=name,arm=arm,initial_loss=warm['coarse']['loss'],state=x,coarse=final['coarse'],audit=dense,history=r['history'],online_calls=322,warmup_calls=65))
    return dict(donor=d,task=name,arm=arm,loss=final['coarse']['loss'],mapping=dense['worst_mapping_rms_mm'],success=dense['threshold_reached'],routing=dense['routing_pass'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--starts');p.add_argument('--donor-start',type=int,default=12);p.add_argument('--donors',type=int,default=8);p.add_argument('--workers',type=int,default=6);p.add_argument('--policy',default='experiments/staged_repair_policy_v2_20261006/policies.joblib');p.add_argument('--arms',nargs='+',default=['adaptive_response','adaptive_repair','response_repair','segmented_cma','local_cma']);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.monotonic()
    protocol=dict(arms=a.arms,policy=a.policy,policy_sha256=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),starts=a.starts,donor_start=a.donor_start,donors=a.donors,status='development' if a.starts else 'fresh donor confirmation with settings frozen before outcomes',
        mechanism='Finite-difference13-state Jacobian26calls, bounded damped least squares, secant update after accepted route-preserving moves, trust radius.15 adapting within.001..4, at most8 updates between probes; three rejections trigger refresh plus16-call CMA escape. Low support:64 local CMA calls, or8 learned repairs+56 local CMA. Exact257-call search budget.',
        budget='65 common warmup+257 adaptive search+65 final freeform=387 calls per run, all5x21. All probes, line trials and fallback counted. Primary dense audit separate. Models/policy were fixed before fresh donors.')
    files=[Path(__file__),Path('nslens/analysis/adaptive_response_adapter.py'),Path('nslens/analysis/staged_repair_policy.py'),Path('nslens/analysis/response_adapter_experiment.py'),Path('nslens/analysis/routing_transition_experiment.py'),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/analysis/optical_refinement.py'),Path('nslens/search/cma_me.py')]
    protocol['code_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f in files:shutil.copy2(f,out/f.name)
    save(out/'protocol.json',protocol)
    if a.starts:starts=[s for s in json.loads(Path(a.starts).read_text()) if s['kind']=='fresh']
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:starts=list(pool.map(prepare_start,[('fresh',d,str(out)) for d in range(a.donor_start,a.donor_start+a.donors)]))
    save(out/'starts.json',starts);jobs=[(s,t,arm,str(out),a.policy) for s in starts for t in TASKS for arm in a.arms]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[]
        for i,r in enumerate(pool.map(run,jobs),1):
            results.append(r)
            if i%len(a.arms)==0:print(f'{i}/{len(jobs)} evaluated',flush=True)
    summary={arm:dict(runs=sum(r['arm']==arm for r in results),passes=sum(r['success'] for r in results if r['arm']==arm),routing=sum(r['routing'] for r in results if r['arm']==arm),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in results if r['arm']==arm]))) for arm in a.arms}
    save(out/'results.json',results);save(out/'summary.json',dict(arms=summary,actual_warmup_calls=len(jobs)*65,online_calls=len(jobs)*322,source_preparation_calls=0 if a.starts else len(starts)*1028,dense_audits=len(jobs),seconds=time.monotonic()-t0));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
