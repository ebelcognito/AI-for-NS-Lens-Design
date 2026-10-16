"""Shared budgeted optical refinement; last-mirror freeform is the default.

Input layouts use the existing parabolic3n-gene representation. Output rows retain
that geometry plus four physical-mm final-mirror shape coefficients. Dense audits
are separate and must never choose optimizer incumbents.
"""
import numpy as np
from scipy.optimize import lsq_linear
from nslens.search.cma_me import CMAES
from nslens.three_mirror.audit import crossing
from nslens.analysis import final_freeform as ff

REFINERS=('last_parabolic','joint_parabolic','last_freeform')
DEFAULT_REFINER='last_freeform'
DEFAULT_BUDGET=261

def geometry_crossing(g,n,a,samples):
 m=ff.decode(g,n,a);u=np.linspace(-30,30,samples)
 shapes=[v+u[:,None]*t+ff.sag(m,j,u)[:,None]*normal for j,(v,t,normal) in enumerate(zip(m['vertices'],m['tangents'],m['normals']))]
 return any(crossing(shapes[i],shapes[j]) for i in range(n) for j in range(i))

def audit(g,n,a,task,fields=15,pupils=101,samples=513):
 r=ff.trace(g,n,a,fields=fields,pupils=pupils,**task);r['crossing']=geometry_crossing(g,n,a,samples);r['routing_pass']=r['worst_support']>=.99 and not r['crossing'];r['threshold_reached']=bool(r['routing_pass'] and r['worst_mapping_rms_mm'] is not None and r['worst_mapping_rms_mm']<=1)
 m=ff.decode(g,n,a);u=np.linspace(-30,30,1001);z=u/30;departure=sum(c*z**k for k,c in enumerate(a,3));departure_slope=sum(k*c*z**(k-1)/30 for k,c in enumerate(a,3));r['max_freeform_departure_mm']=float(np.max(abs(departure)));r['max_added_slope']=float(np.max(abs(departure_slope)))
 return r

def slope_fit(g,n,task,free):
 r,m,_,paths,good=ff.trace(g,n,np.zeros(4),fields=5,pupils=21,paths=True,**task)
 samples=[]
 for i,path in enumerate(paths):
  if not good[i] or r['ray_route'][i][n-1]!=n-1:continue
  hit=np.array(path[-2]);incoming=ff.unit(hit-np.array(path[-3]));field=r['ray_field_deg'][i];target=m['image']+r['parity']*task['focal']*np.tan(np.deg2rad(field))*m['image_tangent'];desired=ff.unit(target-hit);normal=incoming-desired
  den=normal@m['normals'][-1]
  if abs(den)<1e-8:continue
  u=float((hit-m['vertices'][-1])@m['tangents'][-1]);required=float(-(normal@m['tangents'][-1])/den);samples.append(dict(u=u,required_slope=required,current_slope=float(2*m['curves'][-1]*u),field=field))
 if not samples:raise ValueError('No valid last-mirror slope constraints')
 u=np.array([s['u'] for s in samples]);z=u/30;y=np.array([s['required_slope']-s['current_slope'] for s in samples]);A=(2*.025*u)[:,None]
 if free:A=np.column_stack([A]+[k*z**(k-1)/30 for k in range(3,7)])
 lo=np.r_[-1-g[-1],[-2.]*4] if free else np.array([-1-g[-1]]);hi=np.r_[1-g[-1],[2.]*4] if free else np.array([1-g[-1]])
 fit=lsq_linear(A,y,bounds=(lo,hi),tol=1e-10).x
 return fit,dict(samples=samples,fit=fit,rms_required_slope_correction=float(np.sqrt(np.mean(y*y))),rms_unexplained_slope=float(np.sqrt(np.mean((y-A@fit)**2))),meaning='Frozen-hit slope compatibility diagnostic only: moving the surface changes intersections. Not a bound on attainable imaging quality; physical coefficient bounds enforced.')

def refine_design(genes,n,task,seed,refiner=DEFAULT_REFINER,budget=DEFAULT_BUDGET):
 """Refine every candidate before ranking; budget includes fit and line-search traces."""
 if refiner not in REFINERS:raise ValueError(f'Unknown refiner: {refiner}')
 if not isinstance(budget,int) or budget<5:raise ValueError('budget must be an integer >=5')
 base=np.asarray(genes,dtype=float).copy()
 if base.shape!=(3*n,) or not np.all(np.isfinite(base)) or np.any(abs(base)>1+1e-12):raise ValueError('Expected bounded finite parabolic genes')
 free=refiner=='last_freeform';indices=np.arange(3*n) if refiner=='joint_parabolic' else np.array([3*n-1]);x=np.r_[base[indices],np.zeros(4)] if free else base[indices].copy();calls=0
 def unpack(v):
  g=base.copy();g[indices]=v[:len(indices)];a=2*np.array(v[-4:]) if free else np.zeros(4);return g,a
 def evaluate(v):
  nonlocal calls
  g,a=unpack(v);r=ff.trace(g,n,a,fields=5,pupils=21,**task);calls+=1;r['crossing']=geometry_crossing(g,n,a,65);r['optical_loss']=r['loss'];r['loss']+=30000*r['crossing'];r['parameters']=np.asarray(v).tolist();return r
 initial=evaluate(x);best=initial
 try:
  correction,compatibility=slope_fit(base,n,task,free);compatibility['available']=True
 except (ValueError,np.linalg.LinAlgError) as exc:
  correction=np.zeros(5 if free else 1);compatibility=dict(available=False,reason=str(exc))
 calls+=1 # slope_fit always traces before fitting or reporting missing constraints.
 for fraction in [1.,.5,.25]:
  candidate=x.copy();candidate[np.flatnonzero(indices==3*n-1)[0]]+=fraction*correction[0]
  if free:candidate[-4:]=fraction*correction[1:]/2
  row=evaluate(np.clip(candidate,-1,1))
  if row['loss']<best['loss']:best=row
 fitted=best;history=[dict(calls=calls,loss=best['loss'])]
 cma=CMAES(np.asarray(best['parameters']),.03,popsize=8,seed=seed,bounds=(-np.ones(len(x)),np.ones(len(x))),boundary_handling='reflect')
 while calls<budget:
  population=cma.ask();rows=[evaluate(v) for v in population[:min(8,budget-calls)]]
  if len(rows)==8:cma.tell(np.argsort([r['loss'] for r in rows]))
  candidate=min(rows,key=lambda r:r['loss'])
  if candidate['loss']<best['loss']:best=candidate
  history.append(dict(calls=calls,loss=best['loss']))
 assert calls==budget
 if refiner!='joint_parabolic':np.testing.assert_array_equal(np.asarray(best['genes'])[:-1],base[:-1])
 return dict(refiner=refiner,seed=int(seed),initial=initial,fitted=fitted,coarse=best,coarse_calls=calls,compatibility=compatibility,history=history)
