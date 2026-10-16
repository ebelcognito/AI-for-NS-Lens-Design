"""Select archived final candidates using held-out rays and upstream access."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
from . import powered_five_mirror as pm
from .entrance_access import upstream_access
from .search import write_json

def select(out):
 out=Path(out);summary=json.loads((out/'result.json').read_text());config=summary['configuration'];pm.configure(config['segment'],config['aperture'])
 candidates=[]
 for path in sorted(out.glob('seed_*/result.json')):
  r=json.loads(path.read_text())
  inherited=json.loads((path.parent/'inherited_seed.json').read_text())
  for row in [r['best'],inherited,*[a for a in r['stages'][-1]['archive'][:8]]]:
   layout=pm.decode(row['genes']);access=upstream_access(layout,input_length=layout['input_length'],distance_mm=20.,fields=13,pupil_rings=5,pupil_azimuths=32)
   if access['passed']:
    angle=np.arange(32)*2*np.pi/32;pupil=np.vstack([np.zeros((1,2)),*[radius*np.column_stack([np.cos(angle),np.sin(angle)]) for radius in [.25,.5,.75,1.]]])
    audit=pm.evaluate(row['genes'],fields=9,waves=11,pupil_xy=pupil)
    if audit['throughput']==1 and audit['curvature']['sign_violation']==0:
     candidates.append(dict(seed=r['seed'],row=row,access=access,audit=audit))
 if candidates:
  chosen=min(candidates,key=lambda r:r['row']['metrics']['loss']);row=chosen['row']
  summary.update(best_seed=chosen['seed'],genes=row['genes'],best=row,audit=chosen['audit'],geometry=pm.geometry_metrics(row['genes']),layout=pm.decode(row['genes']),entrance_access=chosen['access'],selection='Lowest training loss among final archived candidates passing held-out finite ray trace, sampled original curvature signs and 20mm upstream access; independent audit still required')
 else:summary['selection']='No archived candidate passed held-out optical and upstream access selection; raw best retained as rejected/provisional'
 summary['insertion_configuration']={k:config[k] for k in ['segment','aperture']}
 summary['selection_candidates_passed']=len(candidates)
 write_json(out/'result.json',summary)
 for name in ['powered_five_mirror.py','five_mirror_search.py','powered_five_mirror_cma.py','five_mirror_select.py']:
  shutil.copyfile(Path(__file__).parent/name,out/(name[:-3]+'_snapshot.py'))
 paths=[Path(__file__).parent/name for name in ['powered_five_mirror.py','five_mirror_search.py','powered_five_mirror_cma.py','five_mirror_select.py','powered_four_mirror.py','four_mirror.py','model.py']]+[Path('nslens/search/cma_me.py')]
 write_json(out/'provenance_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 print('selection',len(candidates),'seed',summary['best_seed'],'volume',summary['geometry']['volume_cm3'],flush=True)
 return summary
if __name__=='__main__':
 import sys
 select(sys.argv[1])
