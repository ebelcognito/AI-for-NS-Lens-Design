"""Isolated, budget-accounted optical diagnostics and restart-selection study."""
from __future__ import annotations
import argparse, json, time, hashlib, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from nslens.search.cma_me import CMAES

D=12
WARMUP=-1
SEED_OFFSET=0
from nslens.three_mirror import pilot as mirror

def serial(x):
    if isinstance(x,dict): return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    if isinstance(x,np.ndarray): return serial(x.tolist())
    if isinstance(x,np.generic): return serial(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def save(path,x):
    Path(path).write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n')

def ev(x,dense=False):
    r=mirror.trace(x,fields=11 if dense else 3,pupils=101 if dense else 9)
    return dict(loss=r['loss'],throughput=float(np.mean([f['support'] for f in r['fields']])),
                min_throughput=r['worst_support'],rms_spot_mm=float(np.mean([f['rms_spot_mm'] or 0 for f in r['fields']])),
                dispersion_mm_per_um=0.,slit_magnification=0.,raw=r)

def residual(x):
    r,m,outlines,histories,good=mirror.trace(x,fields=3,pupils=9,paths=True)
    field=np.repeat(np.linspace(-4,4,3),9); target=mirror.FOCAL*np.tan(np.deg2rad(field))
    image=np.array([(h[-1]-m['image'])@m['image_tangent'] for h in histories])
    error=np.where(good,np.clip(image-r['parity']*target,-100,100),100.)
    return np.r_[error,100*(~good)]/np.sqrt(len(good))
def reflect(x): return 1-abs(np.remainder(x+1,4)-2)

class Ledger:
    def __init__(self): self.x=[]; self.y=[]; self.metrics=[]; self.curve=[]
    def evaluate(self,x):
        m=ev(x); self.x.append(np.array(x).copy()); self.y.append(m['loss']); self.metrics.append(m)
        self.curve.append(min(self.curve[-1] if self.curve else np.inf,m['loss']))
        return m['loss']
    def best(self): return self.x[int(np.argmin(self.y))]

def refine(ledger,x,budget,seed,sigma=.14):
    c=CMAES(x,sigma,popsize=12,seed=seed,bounds=(-np.ones(D),np.ones(D)),boundary_handling='reflect')
    bestx=np.array(x); best=np.inf
    for _ in range(budget//12):
        xs=c.ask(); ys=[ledger.evaluate(v) for v in xs]
        c.tell(np.argsort(ys))
        i=int(np.argmin(ys))
        if ys[i]<best: best=ys[i]; bestx=xs[i].copy()
        if c.restart_needed: c=CMAES(bestx,.08,popsize=12,seed=seed+len(ledger.y),bounds=(-np.ones(D),np.ones(D)),boundary_handling='reflect')
    return bestx,best

def diagnostic_job(args):
    idx,out=args; rng=np.random.default_rng(91000+idx+SEED_OFFSET); begun=time.monotonic()
    x=rng.uniform(-.8,.8,D); warm=Ledger()
    warm_budget=(120 if idx%2 else 0) if WARMUP<0 else WARMUP
    if warm_budget: x,_=refine(warm,x,warm_budget,92000+idx+SEED_OFFSET,.22)
    m=ev(x); r=residual(x)
    columns=[]; residual_calls=1; probe_survival=[]
    for j in range(D):
        xp=x.copy(); xm=x.copy(); xp[j]=min(1,x[j]+.01); xm[j]=max(-1,x[j]-.01)
        rp=residual(xp); rm=residual(xm)
        columns.append((rp-rm)/(xp[j]-xm[j])); residual_calls+=2
        # Separate evaluation is explicitly charged; no hidden free traces.
        probe_survival.extend([ev(xp)['throughput'],ev(xm)['throughput']])
    J=np.array(columns).T
    lo=np.maximum(-.08,-1-x); hi=np.minimum(.08,1-x)
    fit=lsq_linear(J,-r,bounds=(lo,hi),max_iter=60)
    s=np.linalg.svd(J,compute_uv=False); pred=float(np.dot(r+J@fit.x,r+J@fit.x))
    corrected=ev(x+fit.x)
    features=[m['loss'],m['throughput'],m['rms_spot_mm'],0.,0.,float(r@r),pred,float(np.linalg.norm(fit.x)),float(s[-1]/max(s[0],1e-12)),float(np.std(probe_survival)),corrected['loss']]
    endpoints=[]
    for rep in range(2):
        ledger=Ledger(); bx,bv=refine(ledger,x,480,100000+idx*2+rep+SEED_OFFSET)
        if m['loss']<bv: bx=x; bv=m['loss']
        endpoints.append(dict(loss=bv,genes=bx,audit=ev(bx,True),curve=ledger.curve))
    row=dict(id=idx,split='train' if idx<32 else 'test',warm=bool(warm_budget),warm_budget=warm_budget,seed_offset=SEED_OFFSET,genes=x,initial=m,features=features,corrected=corrected,endpoints=endpoints,
             costs=dict(warmup=len(warm.y),diagnostic_traces=1+residual_calls+2*D+1,refinement=960,dense_audits=2),seconds=time.monotonic()-begun)
    save(Path(out)/f'diagnostic_{idx:03}.json',row)
    return row

def diagnostic_report(rows,out):
    names=['initial_loss','throughput','spot','unused_constant','unused_constant2','residual_norm_squared','predicted_residual_floor','correction_norm','singular_ratio','survival_std','corrected_loss']
    X=np.array([r['features'] for r in rows]); y=np.array([min(e['loss'] for e in r['endpoints']) for r in rows]); tr=np.array([r['split']=='train' for r in rows]); te=~tr
    predictions={'initial_loss':X[te,0], 'residual_floor':X[te,6], 'corrected_loss':X[te,10]}
    for key,cols in [('initial_only',[0]),('physical_diagnostics',list(range(len(names))))]:
        f=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,max_features=1.,random_state=7,n_jobs=1)
        f.fit(X[tr][:,cols],y[tr]); predictions[key]=f.predict(X[te][:,cols])
    yt=y[te]; pairs=np.triu_indices(len(yt),1); dy=(yt[:,None]-yt[None,:])[pairs]; report={}
    for key,p in predictions.items():
        dp=(p[:,None]-p[None,:])[pairs]; mask=dy!=0
        report[key]=dict(spearman=float(spearmanr(p,yt).statistic),pair_accuracy=float(np.mean((np.sign(dp[mask])==np.sign(dy[mask]))+.5*(dp[mask]==0))),selected_best4_actual_mean=float(np.mean(yt[np.argsort(p)[:4]])))
    save(Path(out)/'diagnostic_summary.json',dict(features=names,train_count=int(tr.sum()),test_count=int(te.sum()),scores=report,test_targets=yt,predictions=predictions,interpretation='Held-out initializations of ONE architecture/task. Local residual proxy differs from scalar CMA loss. No diffraction-limit or global-optimum certification. All-zero field spots when support is absent are not good imaging. Two feature slots are constants and provide no information.'))
    return report

def proposals(rng,ledger,n):
    X=np.array(ledger.x); y=np.array(ledger.y); order=np.argsort(y); elite=X[order[:max(4,len(X)//10)]]
    base=elite[rng.integers(len(elite),size=n)]; a=X[rng.integers(len(X),size=n)]; b=X[rng.integers(len(X),size=n)]
    trial=base+rng.uniform(.3,.8,(n,1))*(a-b)
    mask=rng.random((n,D))<.7; trial=np.where(mask,trial,base)
    trial=reflect(trial+rng.normal(0,.02,(n,D)))
    trial[:max(1,n//4)]=rng.uniform(-.8,.8,(max(1,n//4),D))
    return trial

def search_job(args):
    arm,seed,budget,out=args; begun=time.monotonic(); rng=np.random.default_rng(seed); ledger=Ledger(); starts=[]
    # Shared measured initial design pool in every arm.
    for x in rng.uniform(-.8,.8,(24,D)): ledger.evaluate(x)
    cycle=0
    if arm=='race':
        entrants=[]
        for j in np.argsort(ledger.y)[:8]:
            entrants.append(dict(cma=CMAES(ledger.x[j],.14,popsize=12,seed=seed*100+int(j),bounds=(-np.ones(D),np.ones(D)),boundary_handling='reflect'),best=ledger.y[j],gene=ledger.x[j].copy()))
        def advance(e,calls):
            for _ in range(calls//12):
                if e['cma'].restart_needed:
                    e['cma']=CMAES(e['gene'],.08,popsize=12,seed=seed+len(ledger.y),bounds=(-np.ones(D),np.ones(D)),boundary_handling='reflect')
                xs=e['cma'].ask(); ys=[ledger.evaluate(v) for v in xs]; e['cma'].tell(np.argsort(ys)); j=int(np.argmin(ys))
                if ys[j]<e['best']: e['best']=ys[j]; e['gene']=xs[j].copy()
        for e in entrants: advance(e,min(120,((budget-len(ledger.y))//12)*12))
        entrants=sorted(entrants,key=lambda e:e['best'])[:2]
        for e in entrants: advance(e,min(480,((budget-len(ledger.y))//12)*12))
        winner=min(entrants,key=lambda e:e['best']); advance(winner,((budget-len(ledger.y))//12)*12)
        while len(ledger.y)<budget: ledger.evaluate(rng.uniform(-.8,.8,D))
    if arm=='continuous':
        refine(ledger,ledger.best(),((budget-len(ledger.y))//12)*12,seed*100,.14)
        while len(ledger.y)<budget: ledger.evaluate(rng.uniform(-.8,.8,D))
    while len(ledger.y)<budget:
        screen=min(24,budget-len(ledger.y)); before=len(ledger.y)
        if arm=='random': candidates=rng.uniform(-.8,.8,(screen,D))
        elif arm=='retained':
            if cycle%4==3: candidates=rng.uniform(-.8,.8,(screen,D))
            else:
                X=np.array(ledger.x); elite=X[np.argsort(ledger.y)[:min(12,len(X))]]
                candidates=reflect(elite[rng.integers(len(elite),size=screen)]+rng.normal(0,.12,(screen,D)))
        elif arm=='de': candidates=proposals(rng,ledger,screen)
        elif arm=='surrogate_de' and screen<=4: candidates=rng.uniform(-.8,.8,(screen,D))
        elif arm=='surrogate_de':
            pool=proposals(rng,ledger,768)
            f=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=3,max_features=.8,random_state=seed+cycle,n_jobs=1)
            f.fit(np.array(ledger.x),np.array(ledger.y)); ps=np.array([t.predict(pool) for t in f.estimators_]); scores=ps.mean(0)-.5*ps.std(0)
            selected=[]
            for i in np.argsort(scores):
                if not selected or np.min(np.linalg.norm(pool[selected]-pool[i],axis=1))/np.sqrt(D)>.03: selected.append(int(i))
                if len(selected)==screen-4: break
            candidates=np.concatenate([pool[selected],rng.uniform(-.8,.8,(4,D))])[:screen]
        else: raise ValueError(arm)
        vals=[ledger.evaluate(x) for x in candidates]; parent=candidates[int(np.argmin(vals))]
        starts.append(dict(call=before,parent=parent,screen_best=min(vals)))
        remaining=budget-len(ledger.y); local=min(216,(remaining//12)*12)
        if local: refine(ledger,parent,local,seed*100+cycle,.14)
        while 0<budget-len(ledger.y)<12: ledger.evaluate(rng.uniform(-.8,.8,D))
        cycle+=1
    bx=ledger.best(); audit=ev(bx,True)
    row=dict(arm=arm,seed=seed,calls=len(ledger.y),dense_audit_calls=1,seconds=time.monotonic()-begun,best_loss=min(ledger.y),genes=bx,audit=audit,curve=ledger.curve,starts=starts)
    target=Path(out)/f'{arm}_{seed}'; np.savez_compressed(str(target)+'.npz',X=ledger.x,y=ledger.y)
    save(str(target)+'.json',row); return row

def configure(aperture,warmup=-1,seed_offset=0):
    global WARMUP,SEED_OFFSET
    mirror.APERTURE=aperture
    WARMUP=warmup; SEED_OFFSET=seed_offset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--phase',choices=['diagnostics','search'],required=True); p.add_argument('--workers',type=int,default=4); p.add_argument('--count',type=int,default=48); p.add_argument('--seeds',type=int,nargs='+',default=list(range(301,309))); p.add_argument('--budget',type=int,default=2400); p.add_argument('--arms',nargs='+',choices=['random','retained','de','surrogate_de','continuous','race'],default=['random','retained','de','surrogate_de']); p.add_argument('--aperture',type=float,default=27.78); p.add_argument('--warmup',type=int,default=-1); p.add_argument('--seed-offset',type=int,default=0); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    protocol=dict(arguments=vars(a),dimensions=D,coarse=dict(fields=3,pupils=9),dense=dict(fields=11,pupils=101),model_sha256=hashlib.sha256(Path(mirror.__file__).read_bytes()).hexdigest(),notes='12 variables; ordinary three-mirror 2D cross-section, finite parabolas, nearest-hit tracing. No grating. No diffraction or sagittal imaging model. Every optical call charged; offline diagnostics reported separately. No archive cells. Equal total optical budgets; per-branch allocation depends on the arm. Dense audit never selects candidates. New seed studies only, no published prescription.')
    save(out/(a.phase+'_protocol.json'),protocol); shutil.copyfile(__file__,out/(a.phase+'_script.py')); shutil.copyfile(mirror.__file__,out/'model_snapshot.py')
    jobs=[(i,str(out)) for i in range(a.count)] if a.phase=='diagnostics' else [(arm,s,a.budget,str(out)) for s in a.seeds for arm in a.arms]
    fn=diagnostic_job if a.phase=='diagnostics' else search_job
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=configure,initargs=(a.aperture,a.warmup,a.seed_offset)) as pool:
        for row in pool.map(fn,jobs):
            rows.append(row); print(json.dumps(serial({k:row[k] for k in ['id','arm','seed','seconds','best_loss'] if k in row})),flush=True)
    if a.phase=='diagnostics' and len(rows)>32: print(diagnostic_report(rows,out),flush=True)
    else:
        summary={}
        for arm in a.arms:
            rr=[r for r in rows if r['arm']==arm]; summary[arm]=dict(median_loss=float(np.median([r['best_loss'] for r in rr])),median_audit_loss=float(np.median([r['audit']['loss'] for r in rr])),median_throughput=float(np.median([r['audit']['throughput'] for r in rr])),total_calls=sum(r['calls'] for r in rr),seconds=sum(r['seconds'] for r in rr))
        save(out/'search_summary.json',summary); print(summary,flush=True)
if __name__=='__main__': main()
