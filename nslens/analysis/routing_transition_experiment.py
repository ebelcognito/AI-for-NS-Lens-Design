"""Trajectory-conditioned geometry proposals with matched online optical budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from nslens.analysis.adapter_refinement import evaluate_state, refine_state
from nslens.analysis.mirror_adapter_pilot import features, save, TASKS
from nslens.analysis.mirror_count_experiment import initial
from nslens.analysis.optical_refinement import audit
from nslens.analysis.routing_transition_policy import TransitionPolicy
from nslens.search.cma_me import CMAES

ARMS=('retrieval','neural','unconditioned','random_direction','local_cma','random_restart_cma')

class NeuralMoves:
    """Classify move modes, then select an actual teacher move in each mode."""
    def fit(self,X,D):
        self.scale=StandardScaler().fit(X);self.X=self.scale.transform(X);self.D=np.asarray(D)
        k=min(24,len(D))
        self.cluster=KMeans(n_clusters=k,random_state=42,n_init=10).fit(D)
        self.labels=self.cluster.labels_
        self.net=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=200,random_state=42)
        self.net.fit(self.X,self.labels)
        return self
    def propose(self,obs,state,count=8,radius=.15):
        z=self.scale.transform([obs])[0];prob=self.net.predict_proba([z])[0]
        classes=self.net.classes_[np.argsort(-prob)]
        moves=[]
        for c in classes:
            ids=np.flatnonzero(self.labels==c)
            idx=ids[np.argmin(np.sum((self.X[ids]-z)**2,axis=1))]
            d=self.D[idx].copy();d*=min(1.,radius/max(np.linalg.norm(d),1e-15))
            x=np.asarray(state).copy();x[:9]=np.clip(x[:9]+d,-1,1);moves.append(x)
            if len(moves)==count:break
        while len(moves)<count:moves.append(np.asarray(state).copy())
        return np.asarray(moves)


def raw_transitions():
    rows=[];seen=set();sources={}
    for folder in ('potential_trajectories_20261005','potential_trajectories_fresh_20261005'):
        for path in sorted((Path('experiments')/folder).glob('*__3__*.json')):
            raw=path.read_bytes();r=json.loads(raw);j=r['job']
            if j['n']!=3 or j['seed'] not in range(4100,4106):continue
            sources[str(path)]=hashlib.sha256(raw).hexdigest()
            for ti,trial in enumerate(r['trials']):
                prev=r['parent']['raw']
                for step,entry in sorted(trial['checkpoints'].items(),key=lambda kv:int(kv[0])):
                    child=entry['coarse'];a=np.asarray(prev['genes']);b=np.asarray(child['genes'])
                    key=(tuple(a),tuple(b),tuple(sorted(r['task'].items())))
                    if np.linalg.norm(b-a)>1e-10 and key not in seen:
                        seen.add(key);rows.append(dict(parent=np.r_[a,np.zeros(4)],child=np.r_[b,np.zeros(4)],task=r['task'],donor=j['seed']-4100,source=str(path),trial=ti,checkpoint=int(step)))
                    prev=child
    return rows,sources


def trace_pair(row):
    parent=evaluate_state(row['parent'],row['task']);child=evaluate_state(row['child'],row['task'])
    good=child['loss']<parent['loss'] and not child['crossing'] and child['worst_support']>=parent['worst_support']-.01
    return dict(**row,features=features(row['parent'],parent,row['task']),delta=np.asarray(row['child'])[:9]-np.asarray(row['parent'])[:9],accepted=bool(good),parent_loss=parent['loss'],child_loss=child['loss'],parent_support=parent['worst_support'],child_support=child['worst_support'])


def local(state,task,budget,seed,sigma=.03,parabolic=False):
    state=np.asarray(state).copy();best=evaluate_state(state,task);beststate=state.copy();calls=1
    size=9 if parabolic else 13
    c=CMAES(state[:size],sigma,popsize=8,seed=seed,bounds=(-np.ones(size),np.ones(size)),boundary_handling='reflect')
    while calls<budget:
        pop=c.ask();rows=[]
        for v in pop[:min(8,budget-calls)]:
            x=state.copy();x[:size]=v;r=evaluate_state(x,task);rows.append(r);calls+=1
            if r['loss']<best['loss']:best=r;beststate=x
        if len(rows)==8:c.tell(np.argsort([r['loss'] for r in rows]))
    return dict(state=beststate,coarse=best,coarse_calls=calls)


def prepare_start(job):
    kind,donor,out=job
    if kind=='archive':
        path=Path(f'experiments/mirror_count_geometry_20261005/cross_count_{3100+donor}.json')
        state=np.r_[json.loads(path.read_text())['coarse']['3']['genes'],np.zeros(4)]
        return dict(kind=kind,donor=donor,state=state,preparation_calls=0,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    task=dict(focal=50.,aperture=27.78,half_field_deg=4.);best=None
    for repeat in range(4):
        seed=57000+donor*10+repeat;rng=np.random.default_rng(seed)
        candidate=local(np.r_[initial(3,rng),np.zeros(4)],task,257,seed,sigma=.12,parabolic=True)
        if best is None or candidate['coarse']['loss']<best['coarse']['loss']:best=candidate
    return dict(kind=kind,donor=donor,state=best['state'],preparation_calls=1028,source_task=task,source_coarse=best['coarse'])


def run_case(job):
    start,name,arm,out=job;task=TASKS[name];donor=start['donor'];seed=71000+donor*100+(10000 if start['kind']=='fresh' else 0)
    rng=np.random.default_rng(seed)
    warm=refine_state(start['state'],task,seed=seed,budget=65)
    state=np.asarray(warm['state']);parent=evaluate_state(state,task);best=parent;calls=1;history=[]
    policies=joblib.load(Path(out)/'policies.joblib') if arm in ('retrieval','neural','unconditioned','random_direction') else None
    if arm=='local_cma':
        r=local(state,task,256,seed);calls+=r['coarse_calls'];state=np.asarray(r['state']);best=r['coarse']
    else:
        for round_id in range(4):
            before=best['loss'];proposal_rows=[]
            if arm=='random_restart_cma':
                x=np.r_[initial(3,rng),np.zeros(4)]
                r=local(x,task,64,seed+round_id,sigma=.12);calls+=64
                if r['coarse']['loss']<best['loss']:state=np.asarray(r['state']);best=r['coarse']
            else:
                obs=features(state,best,task)
                policy=policies['neural' if arm=='neural' else 'retrieval']
                proposed=policy.propose(obs,state,count=8,radius=.15)
                if arm=='unconditioned':
                    ds=policies['deltas'][rng.integers(len(policies['deltas']),size=8)]
                    proposed=[]
                    for d in ds:
                        d=d*min(1.,.15/max(np.linalg.norm(d),1e-15));x=state.copy();x[:9]=np.clip(x[:9]+d,-1,1);proposed.append(x)
                elif arm=='random_direction':
                    matched=[]
                    for x in proposed:
                        d=rng.normal(size=9);d*=np.linalg.norm(x[:9]-state[:9])/max(np.linalg.norm(d),1e-15)
                        # Clip both arms to the same box; post-clipping norms recorded.
                        y=state.copy();y[:9]=np.clip(y[:9]+d,-1,1);matched.append(y)
                    proposed=matched
                reference=state.copy()
                for x in proposed:
                    row=evaluate_state(x,task);calls+=1
                    proposal_rows.append(dict(loss=row['loss'],support=row['worst_support'],step_norm=float(np.linalg.norm(x[:9]-reference[:9]))))
                    if row['loss']<best['loss']:best=row;state=np.asarray(x)
                r=local(state,task,56,seed+round_id);calls+=56
                if r['coarse']['loss']<best['loss']:state=np.asarray(r['state']);best=r['coarse']
            history.append(dict(round=round_id,before=before,after=best['loss'],support=best['worst_support'],proposals=proposal_rows))
    assert calls==257
    refined=refine_state(state,task,seed=seed+900,budget=65);calls+=65
    state=np.asarray(refined['state']);dense=audit(state[:9],3,2*state[9:],task)
    result=dict(kind=start['kind'],donor=donor,task=name,arm=arm,online_calls=calls,warmup_calls=65,
                initial_loss=parent['loss'],initial_support=parent['worst_support'],coarse=refined['coarse'],state=state,audit=dense,history=history)
    save(Path(out)/f'{start["kind"]}_{donor}_{name}_{arm}.json',result)
    return dict(kind=start['kind'],donor=donor,task=name,arm=arm,success=dense['threshold_reached'],routing=dense['routing_pass'],loss=refined['coarse']['loss'],mapping=dense['worst_mapping_rms_mm'],online_calls=calls)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    pairs,sources=raw_transitions()
    protocol=dict(training_sources=sources,raw_transition_count=len(pairs),train_donors=list(range(6)),archive_test_donors=[6,7],fresh_test_seeds=[57000,57010,57020,57030],
        arms=ARMS,tasks=TASKS,models='Standardized optical nearest-neighbor move retrieval; KMeans24 delta modes + MLP64,32 classifier, seed42,200 max epochs. Preserve concrete move modes, no vector averaging.',
        teacher='Adjacent changed incumbent checkpoints from original and fresh trajectory runs, current3-mirror model only. Retrace endpoints5x21; retain lower-merit, noncrossing moves with support drop<=.01. Exact parent-child-task duplicates removed. No held-out examples used.',
        proposal='Eight geometry deltas norm capped .15; preserve freeform coefficients. Four rounds of8 proposals +56 full-state local CMA calls. Random-direction control uses retrieved step norm before box clipping; logs actual norms. Unconditioned samples same teacher library.',
        budget='Per arm65 common warmup +257 discovery +65 final freeform=387 calls; all5x21 rays. Local CMA uses1 parent+256 calls. True random restart uses1 parent+4*(64 calls), fresh layout prior each restart and sigma.12. Final dense audit excluded from selection.',
        starts='Archive donor6/7 plus4 fresh independent source-task donors, each best of4 random-prior parabolic CMA257 searches. Fresh donor development1028 calls each. All arms receive identical source donor and common warmup.',
        limits='Development study; archive holdouts previously inspected. Four new source donors provide limited fresh evidence. Random restart comparator receives same incumbent guard. Geometric2D only; no Qwen backbone; no global-optimum claim.')
    files=[Path(__file__),Path('nslens/analysis/routing_transition_policy.py'),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/final_freeform.py'),Path('nslens/analysis/mirror_adapter_pilot.py'),Path('nslens/search/cma_me.py')]
    protocol['code_hashes']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    for x in files:shutil.copy2(x,out/x.name)
    save(out/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:labeled=list(pool.map(trace_pair,pairs,chunksize=16))
    save(out/'transitions.json',labeled);accepted=[r for r in labeled if r['accepted']]
    X=np.asarray([r['features'] for r in accepted]);D=np.asarray([r['delta'] for r in accepted])
    if len(D)<24:raise ValueError('Too few verified teacher transitions')
    print(f'Transitions:{len(labeled)}; accepted:{len(accepted)}',flush=True)
    policies=dict(retrieval=TransitionPolicy().fit(X,D),neural=NeuralMoves().fit(X,D),deltas=D)
    joblib.dump(policies,out/'policies.joblib')
    # Worker processes import this module, so pickle must refer to its importable class.
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        starts=list(pool.map(prepare_start,[('archive',d,str(out)) for d in (6,7)]+[('fresh',d,str(out)) for d in range(4)]))
    save(out/'starts.json',starts)
    jobs=[(s,t,arm,str(out)) for s in starts for t in TASKS for arm in ARMS]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(run_case,jobs),1):
            results.append(r)
            if i%6==0:print(f'{i}/{len(jobs)} evaluated',flush=True)
    summary={kind:{arm:dict(runs=sum(r['kind']==kind and r['arm']==arm for r in results),passes=sum(r['success'] for r in results if r['kind']==kind and r['arm']==arm),routing_passes=sum(r['routing'] for r in results if r['kind']==kind and r['arm']==arm),mean_log_loss=float(np.mean([np.log1p(r['loss']) for r in results if r['kind']==kind and r['arm']==arm]))) for arm in ARMS} for kind in ('archive','fresh')}
    save(out/'results.json',results);save(out/'summary.json',dict(results=summary,training_traces=2*len(labeled),accepted_transitions=len(accepted),source_preparation_calls=4112,online_calls=len(jobs)*322,warmup_calls=len(jobs)*65,dense_audits=len(jobs),seconds=time.monotonic()-started))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    from nslens.analysis.routing_transition_experiment import main as run_main
    run_main()
