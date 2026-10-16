"""Evaluate cheap ray-to-slope proposals with the same search/refinement ledger."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from nslens.analysis.adapter_refinement import evaluate_state,refine_state
from nslens.analysis.frozen_hit_adapter import propose_slopes
from nslens.analysis.mirror_adapter_pilot import save,TASKS,features
from nslens.analysis.routing_transition_experiment import local,prepare_start
from nslens.analysis.response_adapter_experiment import run as run_control
from nslens.analysis.optical_refinement import audit


def run(job):
    start,name,arm,out,policy_path=job
    if arm not in ('slope','slope_repair'):return run_control(job)
    task=TASKS[name];d=start['donor'];seed=181000+d*100
    warm=refine_state(start['state'],task,seed=seed,budget=65)
    state=np.asarray(warm['state']);best=evaluate_state(state,task);first=best;calls=1;history=[]
    policy=joblib.load(policy_path)['retrieval'] if arm=='slope_repair' else None
    for round_id in range(4):
        before=best['loss'];before_support=best['worst_support'];used=0;diags=[];proposal_losses=[]
        if arm=='slope_repair' and best['worst_support']<.99:
            proposed=policy.propose(features(state,best,task),state,count=8,radius=.15)
            for x in proposed:
                row=evaluate_state(x,task);used+=1;proposal_losses.append(row['loss'])
                if row['loss']<best['loss']:state=np.asarray(x);best=row
            diags.append(dict(mode='learned_repair',proposals=8))
        else:
            for _ in range(4):
                proposed,diag=propose_slopes(state,task);used+=diag['observation_calls'];diags.append(diag)
                for x in proposed:
                    row=evaluate_state(x,task);used+=1;proposal_losses.append(row['loss'])
                    if row['loss']<best['loss']:state=np.asarray(x);best=row
        after_proposals=best['loss'];candidate=local(state,task,64-used,seed+round_id);used+=candidate['coarse_calls']
        if candidate['coarse']['loss']<best['loss']:state=np.asarray(candidate['state']);best=candidate['coarse']
        assert used==64;calls+=used
        history.append(dict(round=round_id,before=before,before_support=before_support,after_proposals=after_proposals,after=best['loss'],after_support=best['worst_support'],proposal_losses=proposal_losses,diagnostics=diags,calls=used))
    assert calls==257
    final=refine_state(state,task,seed=seed+900,budget=65);state=np.asarray(final['state']);calls+=65
    dense=audit(state[:9],3,2*state[9:],task)
    save(Path(out)/f'{d}_{name}_{arm}.json',dict(donor=d,task=name,arm=arm,initial_loss=first['loss'],initial_support=first['worst_support'],state=state,coarse=final['coarse'],audit=dense,history=history,online_calls=calls,warmup_calls=65))
    return dict(donor=d,task=name,arm=arm,loss=final['coarse']['loss'],mapping=dense['worst_mapping_rms_mm'],success=dense['threshold_reached'],routing=dense['routing_pass'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--starts');p.add_argument('--donor-start',type=int,default=12);p.add_argument('--donors',type=int,default=8)
    p.add_argument('--policy',default='experiments/staged_repair_policy_v2_20261006/policies.joblib');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--arms',nargs='+',default=['slope','slope_repair','response_repair','segmented_cma']);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.monotonic()
    protocol=dict(arms=a.arms,policy=a.policy,policy_sha256=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),starts=a.starts,donor_start=a.donor_start,donors=a.donors,
        budget='65 common warmup+257 discovery+65 final freeform=387 traces5x21 per arm. Four64-call discovery rounds. Dense audits separate.',
        slope='4 repeated current-surface frozen-hit fits per round, each1 observation+4 bounded line-search candidates=20 calls, then44 full-state CMA calls. Required normals aim valid rays at ideal field mapping; bounded curvature+freeform corrections preserve layout. Analytic local refinement, not learned discovery.',
        slope_repair='If worst-field support<.99 at round start,8 learned routing-repair candidates +56 CMA calls; otherwise slope branch.',
        learned_policy='Staged nearest neighbors; routing repair radius.3 with half/full steps;676 strict support-increasing historical moves from six old families and64 new training searches.',
        status='development' if a.starts else 'fresh independent confirmation; settings fixed before outcomes')
    files=[Path(__file__),Path('nslens/analysis/frozen_hit_adapter.py'),Path('nslens/analysis/response_adapter_experiment.py'),Path('nslens/analysis/freeform_response_adapter.py'),Path('nslens/analysis/staged_repair_policy.py'),Path('nslens/analysis/routing_transition_experiment.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/optical_refinement.py'),Path('nslens/search/cma_me.py')]
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
