"""Audit fixed optical-adapter comparisons and report donor-level uncertainty."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from nslens.analysis.mirror_adapter_pilot import save,TASKS
from nslens.analysis.optical_refinement import audit


def check(path):
    r=json.loads(Path(path).read_text());x=np.asarray(r['state'])
    result=audit(x[:9],3,2*x[9:],TASKS[r['task']],fields=31,pupils=201,samples=1025)
    return dict(file=str(path),donor=r['donor'],task=r['task'],arm=r['arm'],primary_mapping=r['audit']['worst_mapping_rms_mm'],extra_mapping=result['worst_mapping_rms_mm'],extra_pass=result['threshold_reached'],extra_support=result['worst_support'],extra_crossing=result['crossing'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    protocol=json.loads((out/'protocol.json').read_text());summary=json.loads((out/'summary.json').read_text());rows=json.loads((out/'results.json').read_text())
    for f,h in protocol['code_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(protocol['policy']).read_bytes()).hexdigest()==protocol['policy_sha256']
    paths=[];direct=[]
    for row in rows:
        path=out/f'{row["donor"]}_{row["task"]}_{row["arm"]}.json';r=json.loads(path.read_text())
        assert r['online_calls']==322 and r['warmup_calls']==65 and r['coarse']['loss']<=r['initial_loss']+1e-9
        for h in r['history']:
            assert h['calls']==64 and h['after']<=h['before']+1e-9
            direct.append(dict(donor=r['donor'],task=r['task'],arm=r['arm'],round=h['round'],log_gain_proposals=float(np.log1p(h['before'])-np.log1p(h['after_proposals'])),log_gain_cma=float(np.log1p(h['after_proposals'])-np.log1p(h['after']))))
        if r['audit']['threshold_reached']:paths.append(str(path))
    with ProcessPoolExecutor(max_workers=4) as pool:extra=list(pool.map(check,paths))
    save(out/'extra_dense.json',extra);save(out/'direct_gains.json',direct)
    base={(r['donor'],r['task']):r for r in rows if r['arm']=='segmented_cma'};donors=sorted({r['donor'] for r in rows});paired={};rng=np.random.default_rng(700)
    for arm in protocol['arms']:
        rr=[r for r in rows if r['arm']==arm]
        by_donor=[np.mean([float(r['success'])-float(base[r['donor'],r['task']]['success']) for r in rr if r['donor']==d]) for d in donors]
        samples=np.mean(np.asarray(by_donor)[rng.integers(len(donors),size=(10000,len(donors)))],axis=1)
        paired[arm]=dict(merit_wins=sum(r['loss']<base[r['donor'],r['task']]['loss']-1e-9 for r in rr),merit_losses=sum(r['loss']>base[r['donor'],r['task']]['loss']+1e-9 for r in rr),pass_rate_difference=float(np.mean(by_donor)),donor_bootstrap_95_percentile_interval=np.quantile(samples,[.025,.975]).tolist(),donor_pass_differences=by_donor)
    save(out/'paired.json',paired)
    verification=dict(runs=len(rows),online_calls=sum(322 for r in rows),warmup_calls=sum(65 for r in rows),primary_audits=len(rows),extra_audits=len(extra),extra_survivors=sum(r['extra_pass'] for r in extra),hashes_verified=True,donors=len(donors))
    save(out/'verification.json',verification)
    lines=['Optical-adapter comparison —2026-10-06','',f'Status: {protocol["status"]}. {len(donors)} donor families; three correlated specifications per donor.','',
           'All arms receive identical warmed source designs. Per run65 warmup +257 discovery +65 final freeform=387 training-grid trace calls. Probe calls included. Final15x101 audit does not choose incumbents. Extra31x201 audit of primary successes only,1025 surface crossing samples.','',
           'Results: passes/runs, routing passes, mean log1p merit (lower better).']
    for arm,r in summary['arms'].items():lines.append(f"{arm}: {r['passes']}/{r['runs']}; routing {r['routing']}; merit {r['mean_log_loss']:.5f}")
    lines.extend(['',f'Extra dense pass survival: {verification["extra_survivors"]}/{len(extra)}.',
                  '', 'Against matched segmented CMA; percentile intervals resample entire donor families and are exploratory with this donor count:'])
    for arm,r in paired.items():lines.append(f"{arm}: merit wins/losses {r['merit_wins']}/{r['merit_losses']}; pass-rate difference {r['pass_rate_difference']:.3f}, donor-bootstrap95% interval {r['donor_bootstrap_95_percentile_interval']}")
    lines.extend(['',f'Actual new run costs: source development {summary["source_preparation_calls"]}; warmup {summary["actual_warmup_calls"]}; discovery/final {summary["online_calls"]}; primary dense {summary["dense_audits"]}; extra dense {len(extra)}. Prior teacher/model costs are separately logged and are not free.',
                  '', 'Interpretation: response-only and slope-only are physics-based local correction mechanisms. A hybrid improvement over CMA alone cannot establish learned sampling value; inspect its response-only/slope-only control. Random-restart CMA uses four64-call restarts and the same incumbent guard, so conclusions concern this finite budget, not every restart policy. The teacher repair bank is676 support-improving moves; new data came from64 independent training searches,16448 calls. The staged learned policy tests four whole moves at half/full size, caps full norm .3, and preserves freeform coefficients during those moves.',
                  '', 'Limits: ideal2D finite mirrors, sampled geometric audits, fixed3-mirror encoding, no3D/manufacturing/tolerance/diffraction guarantee. Repeated tasks within a donor are not independent trials. Frozen-hit corrections and Jacobians are local models, not bounds on achievable quality.'])
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(verification,indent=2));print(json.dumps(paired,indent=2))

if __name__=='__main__':main()
