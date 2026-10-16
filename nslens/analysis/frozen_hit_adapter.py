"""Convert desired outgoing rays into bounded final-surface slope corrections.

A frozen-intersection local model; every proposed shape must be retraced. This
is an optical proposal/refinement mechanism, not a learned global sampler.
"""
import numpy as np
from scipy.optimize import lsq_linear
from nslens.analysis import final_freeform as ff


def propose_slopes(state,task):
    x=np.asarray(state,float)
    if x.shape!=(13,) or not np.all(np.isfinite(x)) or np.any(abs(x)>1):raise ValueError('Expected bounded13-state')
    r,m,_,paths,good=ff.trace(x[:9],3,2*x[9:],fields=5,pupils=21,paths=True,**task)
    rows=[];targets=[]
    for i,path in enumerate(paths):
        if not good[i] or r['ray_route'][i][2]!=2 or len(path)<3:continue
        hit=np.asarray(path[-2]);incoming=ff.unit(hit-np.asarray(path[-3]))
        desired_target=m['image']+r['parity']*task['focal']*np.tan(np.deg2rad(r['ray_field_deg'][i]))*m['image_tangent']
        outgoing=ff.unit(desired_target-hit);normal=incoming-outgoing;den=normal@m['normals'][2]
        if abs(den)<1e-8:continue
        u=float((hit-m['vertices'][2])@m['tangents'][2]);z=u/30
        required=float(-(normal@m['tangents'][2])/den)
        current=float(ff.slope(m,2,u))
        rows.append([.05*u]+[2*k*z**(k-1)/30 for k in range(3,7)])
        targets.append(required-current)
    delta=np.zeros(5)
    if rows:
        A=np.asarray(rows);b=np.asarray(targets)
        # Penalize large normalized surface changes only weakly.
        fit=lsq_linear(np.vstack([A,1e-5*np.eye(5)]),np.r_[b,np.zeros(5)],bounds=(-1-x[8:],1-x[8:]),tol=1e-10,max_iter=100)
        delta=fit.x
    proposals=[]
    for fraction in (1.,.5,.25,.125):
        v=x.copy();v[8:]=np.clip(v[8:]+fraction*delta,-1,1);proposals.append(v)
    return np.asarray(proposals),dict(observation_calls=1,constraints=len(rows),correction_norm=float(np.linalg.norm(delta)))
