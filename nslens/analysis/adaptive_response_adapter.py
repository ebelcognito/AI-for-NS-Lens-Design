"""Budgeted trust-region optical response with secant reuse and repair fallback."""
import numpy as np
from scipy.optimize import lsq_linear
from nslens.analysis.adapter_refinement import evaluate_state
from nslens.analysis.freeform_response_adapter import residual,propose_response
from nslens.analysis.routing_transition_experiment import local
from nslens.analysis.mirror_adapter_pilot import features


def search(state,task,budget,seed,repair_policy=None):
    if budget<1:raise ValueError('budget must be positive')
    state=np.asarray(state,float).copy();best=evaluate_state(state,task);calls=1;history=[];rng=np.random.default_rng(seed)
    radius=.15;J=None;mask=None;model_state=state.copy();model_row=best;updates=0;failures=0
    def evaluate(x):
        nonlocal calls,best,state
        r=evaluate_state(x,task);calls+=1
        if r['loss']<best['loss']:best=r;state=np.asarray(x).copy()
        return r
    while calls<budget:
        remaining=budget-calls
        if best['worst_support']<.99 or best['crossing']:
            start_calls=calls;before=best['loss'];base=state.copy()
            if repair_policy is not None and remaining>=9:
                for x in repair_policy.propose(features(base,best,task),base,count=8,radius=.15):evaluate(x)
            allocation=min(56 if repair_policy is not None else 64,budget-calls)
            if allocation:
                r=local(state,task,allocation,seed+calls);calls+=allocation
                if r['coarse']['loss']<best['loss']:best=r['coarse'];state=np.asarray(r['state'])
            history.append(dict(mode='routing_repair',before=before,after=best['loss'],calls=calls-start_calls))
            J=None;continue
        if J is None:
            if remaining<28:
                r=local(state,task,remaining,seed+calls,sigma=.01);calls+=remaining
                if r['coarse']['loss']<best['loss']:best=r['coarse'];state=np.asarray(r['state'])
                break
            base=state.copy();parent=best;parity=parent['parity'];rbase,good=residual(parent,task,parity)
            A=np.zeros((len(rbase),13));valid=np.tile(good[:,None],(1,13));route=np.asarray(parent['ray_route']);h=1e-4
            for j in range(13):
                xp=base.copy();xm=base.copy();xp[j]=min(1,base[j]+h);xm[j]=max(-1,base[j]-h)
                rp=evaluate(xp);rm=evaluate(xm);ep,gp=residual(rp,task,parity);em,gm=residual(rm,task,parity)
                common=good&gp&gm&np.all(np.asarray(rp['ray_route'])==route,axis=1)&np.all(np.asarray(rm['ray_route'])==route,axis=1)
                valid[:,j]=common;A[common,j]=(ep[common]-em[common])/(xp[j]-xm[j])
            mask=np.all(valid,axis=1);J=A[mask];model_state=base;model_row=parent;updates=0;failures=0
            if not mask.any():J=None;continue
        e,_=residual(model_row,task,model_row['parity']);e=e[mask]
        lo=np.maximum(-radius,-1-model_state);hi=np.minimum(radius,1-model_state)
        matrix=np.vstack([J,np.sqrt(1e-3)*np.eye(13)]);rhs=np.r_[-e,np.zeros(13)]
        d=lsq_linear(matrix,rhs,bounds=(lo,hi),tol=1e-8,max_iter=100).x
        d*=min(1.,radius/max(np.linalg.norm(d),1e-15));x=np.clip(model_state+d,-1,1);d=x-model_state
        before=model_row['loss'];row=evaluate(x);enew,good=residual(row,task,model_row['parity']);same_route=np.all(np.asarray(row['ray_route'])[mask]==np.asarray(model_row['ray_route'])[mask])
        predicted=float(e@e-(e+J@d)@(e+J@d));actual=float(e@e-enew[mask]@enew[mask])
        ratio=actual/max(predicted,1e-15)
        accept=row['loss']<model_row['loss'] and row['worst_support']>=.99 and not row['crossing'] and np.all(good[mask]) and same_route
        history.append(dict(mode='response_step',before=before,after=row['loss'],accepted=bool(accept),predicted=predicted,actual=actual,ratio=ratio,radius=radius,calls=1))
        if accept:
            change=enew[mask]-e;den=float(d@d)
            if den>1e-14:J+=np.outer(change-J@d,d)/den
            model_state=x;model_row=row;updates+=1;failures=0
            if ratio>.75:radius=min(.4,radius*1.5)
            elif ratio<.25:radius=max(.001,radius*.5)
            if updates>=8 or ratio<.05:J=None
        else:
            failures+=1;radius=max(.001,radius*.5)
            if failures>=3 or np.linalg.norm(d)<1e-7:
                J=None
                # A short local escape uses only the remaining counted calls.
                allocation=min(16,budget-calls)
                if allocation:
                    r=local(state,task,allocation,seed+calls,sigma=.01);calls+=allocation
                    if r['coarse']['loss']<best['loss']:best=r['coarse'];state=np.asarray(r['state'])
    assert calls==budget
    return dict(state=state,coarse=best,coarse_calls=calls,history=history)
