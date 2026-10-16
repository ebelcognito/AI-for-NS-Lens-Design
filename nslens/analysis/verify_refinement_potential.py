"""Verify provenance and join held-out predictions to independent dense audits."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUT = Path('experiments/refinement_potential_20261006')


def main():
    protocol=json.loads((OUT/'audit_protocol.json').read_text())
    designs=defaultdict(set)
    for path,digest in protocol['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        data=json.loads(Path(path).read_text())
        for n,r in data['refinements'].items():
            key=(n,data['source_job']['task'],tuple(r['initial']['genes']))
            designs[key].add(data['source_job']['seed'])
    assert all(len(d)==1 for d in designs.values()), 'Exact designs leaked across donors'
    predictions=list(csv.DictReader((OUT/'prediction/predictions.csv').open()))
    assert len(predictions)==576
    assert len({(r['source'],r['n']) for r in predictions})==576
    audits=[json.loads(l) for l in (OUT/'dense_audits.jsonl').read_text().splitlines()]
    assert len(audits)==144
    lookup={Path(r['source']).name:r['audit'] for r in audits}
    rows=[r for r in predictions if r['n']=='3']
    groups=defaultdict(list)
    for r in rows:
        r['dense']=lookup[r['source']]
        groups[(r['donor'],r['task'])].append(r)
    assert len(groups)==16 and all(len(v)==9 for v in groups.values())
    methods={}
    for key in [k for k in rows[0] if k.startswith('pred_log_final_')]:
        selected=[min(g,key=lambda r:float(r[key])) for g in groups.values()]
        methods[key]=dict(
            spearman_dense=float(spearmanr([float(r[key]) for r in rows],[r['dense']['worst_mapping_rms_mm'] for r in rows]).statistic),
            mean_selected_dense_rms=float(np.mean([r['dense']['worst_mapping_rms_mm'] for r in selected])),
            passes=sum(r['dense']['threshold_reached'] for r in selected),
            feasible=sum(r['dense']['routing_pass'] for r in selected))
    online=[]
    for p in sorted(Path('experiments/online_freeform_comparison_20261006').glob('*__*__*__r4.json')):
        data=json.loads(p.read_text())
        online.extend(data['history'])
    assert len(online)==384
    associations={}
    for name,history in [('all_counts',online),('n3',[r for r in online if r['n']==3])]:
        y=[r['refined_loss'] for r in history]
        associations[name]=dict(n=len(history),
            discovery_merit_spearman=float(spearmanr([r['discovery_best_loss'] for r in history],y).statistic),
            compatibility_spearman=float(spearmanr([r['compatibility']['rms_unexplained_slope'] for r in history],y).statistic))
    scripts=['audit_refinement_potential.py','predict_refinement_potential.py','report_refinement_potential.py','verify_refinement_potential.py']
    hashes={f:hashlib.sha256((Path('nslens/analysis')/f).read_bytes()).hexdigest() for f in scripts}
    result=dict(source_hashes_verified=len(protocol['sources']),
                exact_design_cross_donor_overlap=0,prediction_rows=576,dense_audits=144,
                learned_dense_selection=methods,online_descriptive_associations=associations,
                code_sha256=hashes,
                online_caveat='Separate optimizer trajectories, same eight donor families; coarse merit target differs from dense worst RMS. No independent fitted-probe validation is possible from stored online histories.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    for script in scripts:
        (OUT/script).write_bytes((Path('nslens/analysis')/script).read_bytes())
    with (OUT/'RESULTS.txt').open('a') as f:
        f.write('\nHELD-OUT LEARNED PREDICTORS, DENSE THREE-MIRROR SELECTION\n')
        for key,v in methods.items():f.write(f"{key}: mean RMS {v['mean_selected_dense_rms']:.4f} mm, {v['passes']}/16 passes, {v['feasible']}/16 feasible\n")
        f.write('\nONLINE HISTORY CROSS-CHECK (same donor families; coarse merit outcome)\n')
        for k,v in associations.items():f.write(f'{k}: {v}\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
