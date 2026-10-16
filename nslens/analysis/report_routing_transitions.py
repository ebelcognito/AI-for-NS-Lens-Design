"""Verify and summarize the fixed trajectory-proposal pilot without model tuning."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import joblib
from nslens.analysis.mirror_adapter_pilot import save, TASKS
from nslens.analysis.optical_refinement import audit


def extra_check(path):
    r=json.loads(Path(path).read_text());x=np.asarray(r['state'])
    dense=audit(x[:9],3,2*x[9:],TASKS[r['task']],fields=31,pupils=201,samples=1025)
    return dict(file=str(path),primary_pass=r['audit']['threshold_reached'],extra_pass=dense['threshold_reached'],worst_mapping_rms_mm=dense['worst_mapping_rms_mm'],worst_support=dense['worst_support'],crossing=dense['crossing'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    protocol=json.loads((out/'protocol.json').read_text());summary=json.loads((out/'summary.json').read_text())
    for f,h in protocol['training_sources'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    for f,h in protocol['code_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    ts=json.loads((out/'transitions.json').read_text());accepted=[r for r in ts if r['accepted']]
    assert len(ts)*2==summary['training_traces']
    assert all(r['donor'] in range(6) for r in ts)
    assert all(r['child_loss']<r['parent_loss'] and r['child_support']>=r['parent_support']-.01 for r in accepted)
    models=joblib.load(out/'policies.joblib')
    sample=accepted[0];state=np.asarray(sample['parent']);state[-4:]=[.1,-.2,.3,-.4]
    for name in ('retrieval','neural'):
        x=models[name].propose(sample['features'],state)
        assert x.shape==(8,13) and np.all(abs(x)<=1)
        np.testing.assert_array_equal(x[:,9:],np.tile(state[9:],(8,1)))
        assert np.all(np.linalg.norm(x[:,:9]-state[:9],axis=1)<=.15+1e-12)
    results=json.loads((out/'results.json').read_text());paths=[];diagnostics=[]
    for row in results:
        path=out/f'{row["kind"]}_{row["donor"]}_{row["task"]}_{row["arm"]}.json';r=json.loads(path.read_text())
        assert r['online_calls']==322 and r['warmup_calls']==65
        assert r['coarse']['loss']<=r['initial_loss']+1e-9
        for h in r['history']:assert h['after']<=h['before']+1e-9
        if r['audit']['threshold_reached']:paths.append(str(path))
        if r['arm'] in ('retrieval','neural','unconditioned','random_direction'):
            count=sum(len(h['proposals']) for h in r['history']);assert count==32
            improve=sum(p['loss']<h['before'] for h in r['history'] for p in h['proposals'])
            diagnostics.append(dict(kind=r['kind'],donor=r['donor'],task=r['task'],arm=r['arm'],improving_proposals=improve,proposals=count))
    with ProcessPoolExecutor(max_workers=4) as pool:checks=list(pool.map(extra_check,paths))
    verification=dict(source_hashes_verified=len(protocol['training_sources']),code_hashes_verified=len(protocol['code_hashes']),runs_verified=len(results),training_pairs=len(ts),accepted_pairs=len(accepted),extra_dense_calls=len(checks),extra_passes=sum(r['extra_pass'] for r in checks),neural_iterations=int(models['neural'].net.n_iter_),neural_training_score=float(models['neural'].net.score(models['neural'].X,models['neural'].labels)))
    save(out/'verification.json',verification);save(out/'extra_dense.json',checks);save(out/'proposal_diagnostics.json',diagnostics)
    # Paired counts are descriptive, not independence-adjusted significance tests.
    paired={}
    for kind in ('archive','fresh'):
        baseline={(r['donor'],r['task']):r for r in results if r['kind']==kind and r['arm']=='local_cma'}
        paired[kind]={}
        for arm in protocol['arms']:
            rr=[r for r in results if r['kind']==kind and r['arm']==arm]
            paired[kind][arm]=dict(lower_merit_than_local_cma=sum(r['loss']<baseline[r['donor'],r['task']]['loss']-1e-9 for r in rr),higher_merit_than_local_cma=sum(r['loss']>baseline[r['donor'],r['task']]['loss']+1e-9 for r in rr))
    save(out/'paired.json',paired)
    lines=['Routing-aware transition adapter —2026-10-06','',f'Training: {len(ts)} deduplicated chronological transitions; {len(accepted)} survived fresh optical verification. Six training donor groups. Model: nearest-neighbor whole-move retrieval and neural classification over24 learned geometry-move modes. Existing freeform state preserved during proposal.','',
           'Evaluation: two previously used archive donors and four new source-task donors, three target specifications each. Six arms,108 runs, one optimizer repeat. Same65-call warmup +257 discovery +65 final refinement per run. Initial and final-freeform traces included. All online traces5x21. Random restart uses four fresh geometry-prior starts, not random directions. Offline teacher development is inherited; retracing costs counted separately.','',
           'RESULTS (passes / runs, routing passes, mean log1p merit; lower merit is better):']
    for kind,arms in summary['results'].items():
        lines.append(kind.upper())
        for arm,r in arms.items():lines.append(f"  {arm}: {r['passes']}/{r['runs']}; routing{r['routing_passes']}; mean log merit{r['mean_log_loss']:.4f}")
    lines.extend(['',f'Primary dense passes surviving31x201 extra audit: {verification["extra_passes"]}/{len(checks)}.',
                  f'New trace costs: teacher verification{summary["training_traces"]}; fresh donor development{summary["source_preparation_calls"]}; online{summary["online_calls"]}; actual repeated warmup{summary["warmup_calls"]}; primary dense{summary["dense_audits"]}; extra dense{len(checks)}.',
                  '',f'Neural training hit its fixed200-iteration cap (actual{verification["neural_iterations"]}); convergence warning was recorded. Training classification accuracy{verification["neural_training_score"]:.3f} is a fit diagnostic, not held-out optical performance.',
                  '', 'Limits: four fresh donor families, two familiar archive families, correlated specifications; no statistically established global search advantage. Teacher moves came from parabolic CMA and span only six source families. Freeform coefficients are preserved but the learned actions do not change them; shared full-state CMA and final refinement do. Random-direction lengths match retrieval before boundary clipping, so post-clipping lengths can differ. No Qwen backbone was trained. All results concern ideal2D geometric optics, not manufacturing or diffraction.'])
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
