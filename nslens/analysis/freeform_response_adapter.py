"""Local optical inverse-response proposals for the complete freeform state.

Finite-difference probes are explicit evaluator calls. Missing or rerouted rays
never become fictitious zero-error constraints. The parent parity stays fixed.
"""
import numpy as np
from scipy.optimize import lsq_linear


def residual(row, task, parity):
    good=np.asarray(row['ray_good'],bool)
    target=parity*task['focal']*np.tan(np.deg2rad(row['ray_field_deg']))
    return np.where(good,(np.asarray(row['ray_image_mm'])-target)/10.,0.),good


def propose_response(state,parent,task,evaluate,h=1e-4):
    """26 bounded probes + four candidate vectors; caller evaluates candidates.

    Return proposals, probe observations and diagnostics. Does not run CMA.
    """
    x=np.asarray(state,float)
    if x.shape!=(13,) or not np.all(np.isfinite(x)) or np.any(abs(x)>1):raise ValueError('Expected bounded13-state')
    if not np.isfinite(h) or h<=0:raise ValueError('h must be positive')
    base,mask=residual(parent,task,parent['parity']);J=np.zeros((len(base),13));valid=np.tile(mask[:,None],(1,13));probes=[]
    route=np.asarray(parent['ray_route'])
    for j in range(13):
        plus=x.copy();minus=x.copy();plus[j]=min(1.,x[j]+h);minus[j]=max(-1.,x[j]-h)
        rp=evaluate(plus);rm=evaluate(minus);probes.extend([(plus,rp),(minus,rm)])
        ep,gp=residual(rp,task,parent['parity']);em,gm=residual(rm,task,parent['parity'])
        common=mask & gp & gm & np.all(np.asarray(rp['ray_route'])==route,axis=1) & np.all(np.asarray(rm['ray_route'])==route,axis=1)
        valid[:,j]=common;J[common,j]=(ep[common]-em[common])/(plus[j]-minus[j])
    common=np.all(valid,axis=1);A=J[common];b=base[common];proposals=[]
    for group in (np.arange(6,13),np.arange(13)):
        for radius in (.03,.15):
            delta=np.zeros(13)
            if len(b):
                # Bounded damped least squares respects active geometry limits.
                block=A[:,group];damping=1e-3
                mat=np.vstack([block,np.sqrt(damping)*np.eye(len(group))]);rhs=np.r_[-b,np.zeros(len(group))]
                lo=np.maximum(-radius,-1-x[group]);hi=np.minimum(radius,1-x[group])
                step=lsq_linear(mat,rhs,bounds=(lo,hi),tol=1e-8,max_iter=100).x
                step*=min(1.,radius/max(np.linalg.norm(step),1e-15));delta[group]=step
            proposals.append(np.clip(x+delta,-1,1))
    return np.asarray(proposals),probes,dict(probe_calls=len(probes),common_rays=int(common.sum()),total_rays=len(base),jacobian_norm=float(np.linalg.norm(A)),initial_squared_residual=float(b@b))
