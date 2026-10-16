"""Matched controls for freeform-aware local optical response and learned repair."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from nslens.analysis.adapter_refinement import evaluate_state, refine_state
from nslens.analysis.freeform_response_adapter import propose_response
from nslens.analysis.mirror_adapter_pilot import save, TASKS, features
from nslens.analysis.routing_transition_experiment import local, prepare_start
from nslens.analysis.optical_refinement import audit

DEFAULT_ARMS=('response','response_repair','retrieval','segmented_cma','local_cma','random_restart_cma')


def run(job):
    start,name,arm,out,policy_path=job;task=TASKS[name];d=start['donor'];seed=181000+d*100
    warm=refine_state(start['state'],task,seed=seed,budget=65)
    state=np.asarray(warm['state']);best=evaluate_state(state,task);first=best;calls=1;history=[]
    policy=joblib.load(policy_path)['retrieval'] if arm in ('response_repair','retrieval') else None
    rng=np.random.default_rng(seed)
    if arm=='local_cma':
        candidate=local(state,task,256,seed);calls+=256;state=np.asarray(candidate['state']);best=candidate['coarse']
    else:
        for round_id in range(4):
            before=best['loss'];before_support=best['worst_support'];used=0;meta={};proposal_losses=[];best_proposal=before
            proposals=[];probes=[]
            if arm=='random_restart_cma':
                from nslens.analysis.mirror_count_experiment import initial
                candidate=local(np.r_[initial(3,rng),np.zeros(4)],task,64,seed+round_id,sigma=.12);used=64
            else:
                if arm in ('response','response_repair'):
                    if arm=='response_repair' and best['worst_support']<.99:
                        proposals=policy.propose(features(state,best,task),state,count=8,radius=.15);meta['mode']='learned_repair'
                    else:
                        proposals,probes,meta=propose_response(state,best,task,lambda v:evaluate_state(v,task));used+=len(probes);meta['mode']='optical_response'
                elif arm=='retrieval':
                    proposals=policy.propose(features(state,best,task),state,count=8,radius=.15);meta['mode']='retrieval'
                for x,row in probes:
                    if row['loss']<best['loss']:state=np.asarray(x);best=row
                for x in proposals:
                    row=evaluate_state(x,task);used+=1;proposal_losses.append(row['loss'])
                    if row['loss']<best['loss']:state=np.asarray(x);best=row
                best_proposal=best['loss']
                candidate=local(state,task,64-used,seed+round_id);used+=candidate['coarse_calls']
            if candidate['coarse']['loss']<best['loss']:state=np.asarray(candidate['state']);best=candidate['coarse']
            calls+=used;assert used==64
            history.append(dict(round=round_id,before=before,before_support=before_support,after_proposals=best_proposal,after=best['loss'],after_support=best['worst_support'],proposal_losses=proposal_losses,diagnostics=meta,calls=used))
    assert calls==257
    refined=refine_state(state,task,seed=seed+900,budget=65);calls+=65;state=np.asarray(refined['state'])
    dense=audit(state[:9],3,2*state[9:],task)
    r=dict(donor=d,task=name,arm=arm,initial_loss=first['loss'],initial_support=first['worst_support'],state=state,coarse=refined['coarse'],audit=dense,history=history,online_calls=calls,warmup_calls=65)
    save(Path(out)/f'{d}_{name}_{arm}.json',r)
    return dict(donor=d,task=name,arm=arm,loss=refined['coarse']['loss'],mapping=dense['worst_mapping_rms_mm'],success=dense['threshold_reached'],routing=dense['routing_pass'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--policy',default='experiments/routing_transition_adapter_20261006/policies.joblib');p.add_argument('--starts');p.add_argument('--donor-start',type=int,default=4);p.add_argument('--donors',type=int,default=8);p.add_argument('--arms',nargs='+',default=list(DEFAULT_ARMS));a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    protocol=dict(arms=a.arms,tasks=TASKS,policy=a.policy,policy_sha256=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),
        budget='Identical65 warmup+257 discovery+65 final freeform=387 traces5x21; all probes counted. Four64-call rounds except continuous local CMA. Dense audit separate.',
        response='Fixed parent parity, central differences h1e-4 in all13 normalized parameters;26 probes plus4 bounded damped least-squares proposals, then34 CMA calls per round. Common support and identical ray routes required for derivative rows. Candidate groups optical6:13 and all13, radii.03/.15, damping1e-3.',
        learned_repair='If worst-field support<.99, response_repair uses8 retrieved geometry moves then56 CMA calls; otherwise optical-response branch. Learned geometry actions preserve current freeform coefficients.',
        controls='segmented_cma gives4*64 calls with same CMA reseeding and no learned proposals; local_cma uses continuous256 calls plus parent; random_restart_cma uses4 independent physical-prior starts*64 and incumbent guard.',
        starts=a.starts or dict(donor_start=a.donor_start,donors=a.donors,source_seeds=[57000+d*10 for d in range(a.donor_start,a.donor_start+a.donors)]),
        status='development' if a.starts else 'fresh donor evaluation with method settings fixed before outcomes',limits='2D ideal optics; repeated tasks within donor correlated. Learned branch must beat response-only and matched controls to establish learned-policy contribution.')
    files=[Path(__file__),Path('nslens/analysis/freeform_response_adapter.py'),Path('nslens/analysis/routing_transition_experiment.py'),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/optical_refinement.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/search/cma_me.py')]
    protocol['code_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f in files:shutil.copy2(f,out/f.name)
    save(out/'protocol.json',protocol)
    if a.starts:starts=[s for s in json.loads(Path(a.starts).read_text()) if s['kind']=='fresh']
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:starts=list(pool.map(prepare_start,[('fresh',d,str(out)) for d in range(a.donor_start,a.donor_start+a.donors)]))
    save(out/'starts.json',starts)
    jobs=[(s,t,arm,str(out),a.policy) for s in starts for t in TASKS for arm in a.arms];results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(run,jobs),1):
            results.append(r)
            if i%len(a.arms)==0:print(f'{i}/{len(jobs)} evaluated',flush=True)
    summary={arm:dict(runs=sum(r['arm']==arm for r in results),passes=sum(r['success'] for r in results if r['arm']==arm),routing=sum(r['routing'] for r in results if r['arm']==arm),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in results if r['arm']==arm]))) for arm in a.arms}
    save(out/'results.json',results);save(out/'summary.json',dict(arms=summary,actual_warmup_calls=len(jobs)*65,online_calls=len(jobs)*322,source_preparation_calls=0 if a.starts else len(starts)*1028,dense_audits=len(jobs),seconds=time.monotonic()-started));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
