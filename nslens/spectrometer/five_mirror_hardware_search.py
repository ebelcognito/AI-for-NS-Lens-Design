"""Hardware-constrained broad insertion search and coarse-to-fine CMA."""
import json,time,shutil,hashlib
from pathlib import Path
import numpy as np
from . import powered_four_mirror as old, powered_five_mirror as pm, four_mirror as fold
from .four_mirror_search import package_metrics,optimized_package
from .search import write_json
from .slit_hardware import hardware_check, SETTINGS
OUT=Path('experiments/spectrometer_powered_five_mirror_cma_v2_hardware')
SOURCE=Path('experiments/spectrometer_powered_four_mirror_cma_v3/result.json')

def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();rng=np.random.default_rng(816)
 inherited=np.asarray(json.loads(SOURCE.read_text())['genes']);a=np.arange(32)*2*np.pi/32
 pup=np.vstack([np.zeros((1,2)),np.column_stack([np.cos(a),np.sin(a)]),.65*np.column_stack([np.cos(a),np.sin(a)])])
 ref=old.trace(inherited,fields=5,waves=7,pupil_xy=pup);rows=[];counts=[0]*6
 with (OUT/'geometry_population.jsonl').open('w') as log:
  for i in range(9000):
   parent_genes=inherited.copy()
   if i>=3000:
    parent_genes[15:21]=np.clip(inherited[15:21]+rng.normal(0,.18 if i<6000 else .45,6),-1,1)
    # Parent may self-obstruct; the new fold can remove that obstruction.
    ref=old.trace(parent_genes,fields=5,waves=7,pupil_xy=pup)
   seg=i%6;params=[rng.uniform(.15,.85),rng.uniform(45,135),float(rng.choice([0,180])) if i%4 else rng.uniform(0,360)]
   m=pm.insert(ref['model'],seg,params,paths=ref['paths']);screen=fold.screen_layout(m,max_rays=35)
   row=dict(evaluation=i+1,segment=seg,params=params,parent_genes=parent_genes.tolist(),screen=screen,aperture=m['surfaces'][seg]['aperture'])
   if screen['valid']:
    counts[seg]+=1
    hardware=hardware_check(m,paths=m['nominal_paths'],dense=False);row['hardware']=hardware
    if hardware['passed']:
     geo=package_metrics(m,parent_genes[25:28]*90,samples=7);row['volume_cm3']=geo['volume_cm3'];rows.append(row)
   log.write(json.dumps(row)+'\n')
   if (i+1)%400==0:print('geometry',i+1,'optical_feasible',counts,'hardware_feasible',len(rows),flush=True)
 rows.sort(key=lambda r:r['volume_cm3']);verified=[]
 # Keep each insertion slot represented in detailed validation, plus global leaders.
 selected=rows[:30]
 for seg in range(6):selected += [r for r in rows if r['segment']==seg][:4]
 unique={r['evaluation']:r for r in selected}
 dense_pup=np.vstack([np.zeros((1,2)),*[radius*np.column_stack([np.cos(a),np.sin(a)]) for radius in [.2,.4,.6,.8,1.]]])
 for row in unique.values():
  pm.configure(row['segment'],row['aperture']);g=pm.seed_genes(np.asarray(row['parent_genes']),row['params']);r=pm.trace(g,fields=13,waves=17,pupil_xy=dense_pup)
  hardware=hardware_check(r['model'],paths=r['paths'],dense=True)
  if r['good'].all() and hardware['passed']:
   geo=optimized_package(r['model'],seed=715,iterations=12);g[25:28]=np.asarray(geo['rotation_xyz_deg'])/90
   row={**row,'genes':g,'geometry':geo,'metrics':pm.evaluate(g,fields=3,waves=5,pupil_xy=pup,hardware_constraints=True),'dense_hardware':hardware};verified.append(row)
 verified.sort(key=lambda r:r['metrics']['loss'])
 write_json(OUT/'geometry_search.json',dict(sampled=9000,feasible_by_segment=counts,coarse_feasible=len(rows),detailed_candidates=len(unique),verified=verified,elapsed_seconds=time.monotonic()-start,source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),bounds=dict(fraction=[.15,.85],turn_deg=[45,135],azimuth_deg=[0,360]),aperture_policy='Inherited five masks unchanged; added mask full 5x7x65-ray footprint plus 1 mm, frozen before refinement'))
 if not verified:raise RuntimeError('No hardware-qualified six-surface seed; search record saved, CMA not launched')
 best=verified[0];pm.configure(best['segment'],best['aperture']);g=np.asarray(best['genes']);write_json(OUT/'initial_genes_source.json',best)
 # Central finite differences only scale CMA covariance; no gradient optimizer.
 steps=[]
 for j in range(pm.N_TOTAL):
  d=np.zeros(pm.N_TOTAL);d[j]=1e-4
  plus=pm.trace(g+d,fields=3,waves=5,pupil_xy=pup)['xy'];minus=pm.trace(g-d,fields=3,waves=5,pupil_xy=pup)['xy']
  sensitivity=float(np.sqrt(np.mean(((plus-minus)/2e-4)**2)))
  steps.append(float(np.clip(.0005/max(sensitivity,1e-12),1e-6,.05)))
 config=dict(out=str(OUT),segment=best['segment'],aperture=best['aperture'],initial_genes=g.tolist(),fields=3,waves=5,pupils=5,rim_points=16,base_evals=1200,freeform_evals=2400,popsize=24,restart_generations=100,base_sigma=.025,fine_sigma=.015,initial_perturbation=0.,sensitivity_steps=steps,audit_fields=7,audit_waves=9,audit_pupils=9,finalists=4,source=str(SOURCE),hardware_constraints=True,hardware_settings=SETTINGS,algorithm='staged_full_covariance_cma_es',local_refinement=False)
 write_json(OUT/'configuration.json',config)
 for name in ['powered_five_mirror.py','five_mirror_hardware_search.py','powered_five_mirror_cma.py','slit_hardware.py']:
  shutil.copyfile(Path(__file__).parent/name,OUT/(name[:-3]+'_snapshot.py'))
 from .powered_five_mirror_cma import run_seed
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_seed,[{**config,'seed':s} for s in [801,802,803]]))
 best=min(results,key=lambda r:r['best']['metrics']['loss'])
 write_json(OUT/'result.json',dict(configuration=config,evaluations=sum(r['evaluations'] for r in results),best_seed=best['seed'],genes=best['best']['genes'],best=best['best'],audit=best['audit'],geometry=best['geometry'],layout=best['layout'],seed_results=[dict(seed=r['seed'],best=r['best'],audit=r['audit'],geometry=r['geometry']) for r in results],elapsed_seconds=time.monotonic()-start,local_refinement=False))
 from .five_mirror_hardware_select import select
 select(OUT)
 print('COMPLETE',best['seed'],best['audit'],flush=True)
if __name__=='__main__':main()
