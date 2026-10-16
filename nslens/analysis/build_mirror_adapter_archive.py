"""Re-trace and label current-model archive records under one refinement budget.

Preserves provenance for later grouped splits. Does not mix legacy encodings or
interpret historical optimizer losses as comparable potential labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from nslens.analysis.adapter_refinement import evaluate_state, refine_state
from nslens.analysis.mirror_adapter_pilot import features, target, save


def label(job):
    path,out,budget=job
    p=Path(path);raw=p.read_bytes();r=json.loads(raw)
    state=np.r_[r['genes'],np.asarray(r['freeform_coefficients_mm'])/2]
    if np.any(abs(state)>1):
        return dict(id=r['id'],excluded='outside bounded adapter domain')
    obs=evaluate_state(state,r['task'])
    refined=refine_state(state,r['task'],seed=24680,budget=budget)
    row=dict(id=r['id'],model=r['model'],task=r['task'],state=state,
             source_sha256=hashlib.sha256(raw).hexdigest(),occurrences=r['occurrences'],
             features=features(state,obs,r['task']),target=target(refined['coarse']),
             observation=obs,refined=refined,calls=1+budget)
    save(Path(out)/'labels'/p.name,row)
    return dict(id=r['id'],calls=1+budget,initial_loss=obs['loss'],refined_loss=refined['coarse']['loss'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='experiments/all_three_mirror_samples_20261006');p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--budget',type=int,default=33);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'labels').mkdir()
    paths=[p for p in sorted((Path(args.source)/'samples').glob('*.json')) if json.loads(p.read_text())['model']=='variable_mirror_n3']
    save(out/'protocol.json',dict(source=args.source,records=len(paths),budget=args.budget,seed=24680,
        purpose='Uniform potential labels for archive training; split must group ancestry before model fitting.',
        features='mirror_adapter_pilot.features; observed 5x21 rays with masks, routing, geometry, task and crossing',
        code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('nslens/analysis/adapter_refinement.py'),Path('nslens/analysis/mirror_adapter_pilot.py'),Path('nslens/analysis/final_freeform.py')]}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[]
        for i,r in enumerate(pool.map(label,[(str(p),str(out),args.budget) for p in paths],chunksize=8),1):
            rows.append(r)
            if i%200==0:print(f'{i}/{len(paths)} labeled',flush=True)
    save(out/'manifest.json',rows)
    print(json.dumps(dict(records=len(rows),calls=sum(r.get('calls',0) for r in rows),excluded=sum('excluded' in r for r in rows))),flush=True)

if __name__=='__main__':main()
