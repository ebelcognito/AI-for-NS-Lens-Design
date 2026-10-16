"""Predeclared follow-up: target specified focus/mapping rather than freeze errors."""
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from nslens.analysis.optical_mutation_experiment import refine,geometry_ok
from nslens.analysis import compensated_mirror as m
from nslens.analysis.refinement_sampling_study import save
ROOT=Path('experiments/optical_mutation_20261005')
OUT=Path('experiments/optical_target_repair_20261005')

def run(path):
 r=json.loads(Path(path).read_text());x=np.array(r['job']['genes']);J=np.array(r['J']);parent=r['parent'];parent['protected']=np.array(parent['protected']);active=9;calls=0;dense_calls=0;start=time.monotonic()
 def obs(v,dense=False):
  nonlocal calls,dense_calls
  if dense:dense_calls+=1
  else:calls+=1
  return m.observe(v,dense=dense)
 # Nominal map: zero on-axis centroid, +/-50mm focal mapping, zero paraxial pupil slope.
 target=np.array([0.,1. if parent['protected'][1]>=0 else -1.,0.]);err=target-parent['protected'];G=J@J.T+1e-6*np.eye(3);actions=[]
 for i,a in enumerate(r['actions']):
  raw=np.array(a['raw_delta']);repair=raw.copy();repair[:9]+=J.T@np.linalg.solve(G,err-J@raw[:9]);repair[9:]=0.
  norm=np.linalg.norm(repair)
  if norm>0:repair*=np.linalg.norm(raw)/norm
  scale=1.
  for j in range(9):
   if repair[j]>0:scale=min(scale,max(0,(1-x[j])/repair[j]))
   elif repair[j]<0:scale=min(scale,max(0,(-1-x[j])/repair[j]))
  repair*=scale;raw*=np.linalg.norm(repair)/max(np.linalg.norm(raw),1e-30)
  assert np.isclose(np.linalg.norm(raw),np.linalg.norm(repair),atol=1e-12)
  ends={}
  for tag,step,extra in [('target_repair',repair,0),('raw_matched',raw,18)]:
   child=obs(x+step);_,(g,o)=refine(obs,x+step,child,48+extra,800000+r['job']['seed']*20+i,9)
   if parent['loss']<o['loss']:g=x;o=parent
   end=dict(genes=g,loss=o['loss'])
   if r['job']['split']=='test':end.update(audit=obs(g,True),geometry_ok=geometry_ok(g))
   ends[tag]=end
  actions.append(dict(family=a['family'],radius=a['radius'],features=a['features'],norm=float(np.linalg.norm(repair)),endpoints=ends))
 row=dict(job=r['job'],parent=r['parent'],actions=actions,new_coarse_calls=calls,new_dense_calls=dense_calls,charged_existing_probes_per_decision=18,seconds=time.monotonic()-start)
 save(OUT/Path(path).name,row);return row

def main():
 OUT.mkdir(exist_ok=False,parents=True);save(OUT/'protocol.json',dict(parent_source=str(ROOT),target='[0,sign(current field slope),0] for protected centroid/focal mapping/pupil slope',budget='Target:18 already-measured probes+1 child+48 refinement. Control:1 child+66 refinement. Both inherit parent measurement; actual new dataset calls and reused probe provenance reported separately.',steps='Norm matched after bounds. Same 6 raw proposals and local-CMA seeds as earlier protocol. Fixed apertures. Independent test parents1302..1324; no test-driven hyperparameter tuning.',interpretation='Follow-up after preservation did not clearly outperform cost-matched control; not an independent fresh-parent confirmation.'))
 import shutil;shutil.copyfile(__file__,OUT/'script_snapshot.py')
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(run,sorted(ROOT.glob('mirror_0_*.json'))):print(r['job']['split'],r['job']['seed'],r['seconds'],flush=True)
if __name__=='__main__':main()
