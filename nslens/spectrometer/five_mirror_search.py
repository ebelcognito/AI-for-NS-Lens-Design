"""Broad insertion search followed by genuine coarse-to-fine CMA."""
import json,time,shutil,hashlib
from pathlib import Path
import numpy as np
from . import powered_four_mirror as old, powered_five_mirror as pm, four_mirror as fold
from .four_mirror_search import package_metrics,optimized_package
from .search import write_json
OUT=Path('experiments/spectrometer_powered_five_mirror_cma_v1')
SOURCE=Path('experiments/spectrometer_powered_four_mirror_cma_v3/result.json')

def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();rng=np.random.default_rng(715)
 inherited=np.asarray(json.loads(SOURCE.read_text())['genes']);a=np.arange(32)*2*np.pi/32
 pup=np.vstack([np.zeros((1,2)),np.column_stack([np.cos(a),np.sin(a)]),.65*np.column_stack([np.cos(a),np.sin(a)])])
 ref=old.trace(inherited,fields=5,waves=7,pupil_xy=pup);rows=[];counts=[0]*6
 with (OUT/'geometry_population.jsonl').open('w') as log:
  for i in range(2400):
   seg=i%6;params=[rng.uniform(.15,.85),rng.uniform(45,135),float(rng.choice([0,180])) if i<1800 else rng.uniform(0,360)]
   m=pm.insert(ref['model'],seg,params,paths=ref['paths']);screen=fold.screen_layout(m,max_rays=35)
   row=dict(evaluation=i+1,segment=seg,params=params,screen=screen,aperture=m['surfaces'][seg]['aperture'])
   if screen['valid']:
    counts[seg]+=1
    origin=ref['paths'][0];direction=-fold.model.unit(ref['paths'][1]-origin);blocked=np.zeros(len(origin),bool)
    for surface in m['surfaces']:
     _,_,distance,valid=fold.model.intersect(origin,direction,surface);blocked|=valid&(distance<=20)
    distance,valid=fold.detector_intersection(origin,direction,m);blocked|=valid&(distance<=20)
    row['entrance_20mm_blocked_fraction']=float(blocked.mean())
    geo=package_metrics(m,inherited[25:28]*90,samples=7);row['volume_cm3']=geo['volume_cm3'];rows.append(row)
   log.write(json.dumps(row)+'\n')
   if (i+1)%400==0:print('geometry',i+1,'feasible',counts,flush=True)
 rows.sort(key=lambda r:r['volume_cm3']);verified=[]
 # Keep each insertion slot represented in detailed validation, plus global leaders.
 selected=rows[:30]
 for seg in range(6):selected += [r for r in rows if r['segment']==seg][:4]
 unique={r['evaluation']:r for r in selected}
 for row in unique.values():
  pm.configure(row['segment'],row['aperture']);g=pm.seed_genes(inherited,row['params']);r=pm.trace(g,fields=7,waves=9,pupil_xy=pup)
  if r['good'].all():
   geo=optimized_package(r['model'],seed=715,iterations=12);g[25:28]=np.asarray(geo['rotation_xyz_deg'])/90
   row={**row,'genes':g,'geometry':geo,'metrics':pm.evaluate(g,fields=3,waves=5,pupil_xy=pup)};verified.append(row)
 verified.sort(key=lambda r:r['metrics']['loss'])
 qualified=[r for r in verified if r['entrance_20mm_blocked_fraction']==0]
 if qualified:verified=qualified+[r for r in verified if r['entrance_20mm_blocked_fraction']>0]
 write_json(OUT/'geometry_search.json',dict(sampled=2400,feasible_by_segment=counts,coarse_feasible=len(rows),detailed_candidates=len(unique),verified=verified,elapsed_seconds=time.monotonic()-start,source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),bounds=dict(fraction=[.15,.85],turn_deg=[45,135],azimuth_deg=[0,360]),aperture_policy='Inherited five masks unchanged; added mask full 5x7x65-ray footprint plus 1 mm, frozen before refinement'))
 if not verified:raise RuntimeError('No valid six-surface seed')
 best=verified[0];pm.configure(best['segment'],best['aperture']);g=np.asarray(best['genes']);write_json(OUT/'initial_genes_source.json',best)
 # Central finite differences only scale CMA covariance; no gradient optimizer.
 steps=[]
 for j in range(pm.N_TOTAL):
  d=np.zeros(pm.N_TOTAL);d[j]=1e-4
  plus=pm.trace(g+d,fields=3,waves=5,pupil_xy=pup)['xy'];minus=pm.trace(g-d,fields=3,waves=5,pupil_xy=pup)['xy']
  sensitivity=float(np.sqrt(np.mean(((plus-minus)/2e-4)**2)))
  steps.append(float(np.clip(.0005/max(sensitivity,1e-12),1e-6,.05)))
 config=dict(out=str(OUT),segment=best['segment'],aperture=best['aperture'],initial_genes=g.tolist(),fields=3,waves=5,pupils=5,rim_points=16,base_evals=1200,freeform_evals=2400,popsize=24,restart_generations=100,base_sigma=.025,fine_sigma=.015,initial_perturbation=0.,sensitivity_steps=steps,audit_fields=7,audit_waves=9,audit_pupils=9,finalists=4,source=str(SOURCE),entrance_20mm_blocked_fraction=best['entrance_20mm_blocked_fraction'],algorithm='staged_full_covariance_cma_es',local_refinement=False)
 write_json(OUT/'configuration.json',config)
 for name in ['powered_five_mirror.py','five_mirror_search.py','powered_five_mirror_cma.py']:
  shutil.copyfile(Path(__file__).parent/name,OUT/(name[:-3]+'_snapshot.py'))
 from .powered_five_mirror_cma import run_seed
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_seed,[{**config,'seed':s} for s in [701,702,703]]))
 best=min(results,key=lambda r:r['best']['metrics']['loss'])
 write_json(OUT/'result.json',dict(configuration=config,evaluations=sum(r['evaluations'] for r in results),best_seed=best['seed'],genes=best['best']['genes'],best=best['best'],audit=best['audit'],geometry=best['geometry'],layout=best['layout'],seed_results=[dict(seed=r['seed'],best=r['best'],audit=r['audit'],geometry=r['geometry']) for r in results],elapsed_seconds=time.monotonic()-start,local_refinement=False))
 from .five_mirror_select import select
 select(OUT)
 print('COMPLETE',best['seed'],best['audit'],flush=True)
if __name__=='__main__':main()
