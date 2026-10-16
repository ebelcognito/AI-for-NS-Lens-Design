"""Verify experiment provenance, frozen geometry, bounds, and evaluation budgets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    protocol=json.loads((out/'protocol.json').read_text())
    source_hashes={j['source']:j['source_hash'] for j in protocol['jobs']}
    for path,digest in {**source_hashes,**protocol['code_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    # Preserve distinct files with identical basenames (both environments have model.py).
    for path in protocol['code_sha256']:
        dest=out/'source_snapshots'/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
    rows=[json.loads(p.read_text()) for p in out.glob('*__*__*.json')]
    assert len(rows)==len(protocol['jobs'])==48
    identifiers=set()
    for r in rows:
        j=r['job'];identifier=(j['environment'],j['donor_seed'],j['variant'])
        assert identifier not in identifiers;identifiers.add(identifier)
        g=np.array(j['genes']);active=r['active_indices']
        frozen=[i for i in range(len(g)) if i not in active]
        for h in [r['fitted']['genes']]+[a['genes'] for a in r['arms'].values()]:
            h=np.array(h)
            np.testing.assert_array_equal(h[frozen],g[frozen])
            assert np.all(np.abs(h)<=1+1e-12)
        assert r['actual_training_traces']==775 and r['dense_audits']==5
        for name,arm in r['arms'].items():
            assert arm['effective_training_calls']==260
            assert arm['history'][-1]['calls']==260
            start=r['initial']['loss'] if name=='cold_cma' else r['fitted']['metrics']['loss']
            losses=np.array([start]+[a['loss'] for a in arm['history']])
            assert np.all(np.diff(losses)<=1e-10)
            a=arm['audit'];assert 0<=a['valid_rays']<=a['launched_rays']
            assert abs(a['throughput']-a['valid_rays']/a['launched_rays'])<1e-12
    result=dict(cases=48,source_files_verified=len(source_hashes),code_files_verified=len(protocol['code_sha256']),
                frozen_nonactive_genes=True,coefficient_bounds=True,incumbent_retention=True,
                effective_calls_per_arm=260,actual_training_traces=48*775,dense_audits=48*5,
                note='Numerical invariants and provenance verified; not a global root or continuous geometry certificate.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ['report_refinement_probe_3d.py','verify_refinement_probe_3d.py']:
        (out/name).write_bytes((Path('nslens/analysis')/name).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':main()
