"""Prospective short-probe/refinement comparison in two existing 3D tracers."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nslens.analysis import refinement_probe_3d as probe
from nslens.search.cma_me import CMAES
from nslens.thermal_camera.freeform_search import serial


def save(path, data):
    Path(path).write_text(json.dumps(serial(data), indent=2, allow_nan=False)+'\n')


def candidates():
    rows=[]
    for environment,seeds in [('thermal',range(101,104)),('monolithic',range(301,304))]:
        for family,seed in enumerate(seeds):
            path=(Path(f'experiments/thermal_camera_pilot_v1/seed_{seed}/result.json')
                  if environment=='thermal' else Path(f'experiments/monolithic_ir_pilot_v2/seed{seed}/result.json'))
            data=json.loads(path.read_text())
            genes=np.array(data['genes'],dtype=float)
            assert len(genes)==(14 if environment=='thermal' else 48)
            if environment=='thermal':
                genes=np.pad(genes,(0,50-len(genes)))
                from nslens.thermal_camera import freeform_model
                active=np.array([freeform_model.BASE_DIM+i for i,(name,_,_) in enumerate(freeform_model.PARAMETERS) if name=='secondary'])
                # Perturb only the inherited radial layout/asphere variables.
                mutable=np.arange(14)
            else:
                active=np.r_[14,15,np.arange(41,48)]
                mutable=np.array([i for i in range(48) if i not in active])
            for variant,scale in enumerate([0,.01,.01,.03,.03,.06,.06,.10]):
                rng=np.random.default_rng(611000+seed*100+variant)
                g=genes.copy()
                g[mutable]=np.clip(g[mutable]+rng.normal(0,scale,len(mutable)),-1,1)
                np.testing.assert_array_equal(g[active],genes[active])
                rows.append(dict(environment=environment,family=family,donor_seed=seed,
                    variant=variant,perturbation_sigma=scale,genes=g,
                    source=str(path),source_hash=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def run(job):
    start=time.monotonic()
    context=probe.prepare(job['environment'],job['genes'],fields=3,pupils=3)
    base=np.asarray(context['genes']).copy()
    active=np.asarray(context['active_indices'])
    frozen=np.array([i for i in range(len(base)) if i not in active])
    initial=context['initial']
    traces=1
    try:
        delta,compatibility=probe.fit_probe(context)
    except (ValueError,np.linalg.LinAlgError) as exc:
        delta=np.zeros_like(base)
        compatibility=dict(available=False,reason=str(exc))
    best_g=base.copy();best=initial
    trials=[]
    for scale in [1.,.5,.25]:
        g=np.clip(base+scale*delta,-1,1)
        row=probe.observe(context,g);traces+=1
        trials.append(dict(scale=scale,metrics=row))
        if row['loss']<best['loss']:
            best_g=g.copy();best=row
    fitted=dict(genes=best_g.copy(),metrics=best)
    arms={}
    for name,sigma,warm in [('probe_cma',.03,True),('probe_wide_cma',.15,True),('cold_cma',.03,False)]:
        g0=best_g.copy() if warm else base.copy()
        incumbent=best if warm else initial
        incumbent_g=g0.copy()
        calls=4 if warm else 1
        cma=CMAES(g0[active],sigma,popsize=8,
                  seed=612000+job['donor_seed']*100+job['variant'],
                  bounds=(-np.ones(len(active)),np.ones(len(active))),boundary_handling='reflect')
        history=[]
        while calls<260:
            population=cma.ask();values=[]
            for sample in population[:min(8,260-calls)]:
                g=base.copy();g[active]=sample
                row=probe.observe(context,g);calls+=1;traces+=1
                values.append(row['loss'])
                if row['loss']<incumbent['loss']:
                    incumbent=row;incumbent_g=g.copy()
            if len(values)==8:
                cma.tell(np.argsort(values))
            history.append(dict(calls=calls,loss=incumbent['loss']))
        np.testing.assert_array_equal(incumbent_g[frozen],base[frozen])
        assert np.all(abs(incumbent_g)<=1+1e-12)
        arms[name]=dict(genes=incumbent_g,coarse=incumbent,history=history,
                        effective_training_calls=calls,sigma=sigma)
    assert traces==775
    # Independent denser ray grids never select an incumbent or a candidate.
    initial_audit=probe.observe(context,base,fields=5,pupils=8)
    fitted_audit=probe.observe(context,best_g,fields=5,pupils=8)
    for result in arms.values():
        result['audit']=probe.observe(context,result['genes'],fields=5,pupils=8)
    result=dict(job=job,initial=initial,initial_audit=initial_audit,
                compatibility=compatibility,trials=trials,fitted=fitted,
                fitted_audit=fitted_audit,arms=arms,
                actual_training_traces=traces,dense_audits=5,
                active_indices=active,seconds=time.monotonic()-start)
    name=f"{job['environment']}__{job['donor_seed']}__{job['variant']}.json"
    save(Path(job['out'])/name,result)
    return dict(case=name,seconds=result['seconds'],initial=initial['worst_mapping_rms_mm'],
                fitted=best['worst_mapping_rms_mm'],final=arms['probe_cma']['audit']['worst_mapping_rms_mm'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    jobs=candidates()
    if args.smoke:jobs=[j for j in jobs if j['family']==0 and j['variant']==0]
    for j in jobs:j['out']=str(out)
    paths=[Path(__file__),Path(probe.__file__),Path('nslens/thermal_camera/model.py'),
           Path('nslens/thermal_camera/freeform_model.py'),Path('nslens/monolithic_ir/model.py'),Path('nslens/search/cma_me.py')]
    save(out/'protocol.json',dict(jobs=jobs,code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        purpose='Test transfer of short final-surface physical correction to genuine 3D ray bundles.',
        population='Two environments, three saved donors each, baseline plus seven deterministic upstream perturbations. No quality filtering. Families are local perturbations, not independent global architectures.',
        surface='Thermal secondary:12 XY coefficients degree2..4. Monolithic exit refraction:2 quadratic powers+7 higher coefficients. Other genes and detector fixed.',
        probe='Initial ray trace reused for bounded two-component slope fit; retrace correction at1,.5,.25. Four actual traces.',
        arms='Probe+256 CMA sigma.03; probe+256 CMA sigma.15; cold CMA259+initial. Effective260 calls each; common initial/probe calls shared for actual ledger775 per case.',
        objective='Mapping error to fixed signed focal targets, pixel normalized, with lost-ray penalties; differs from native camera spot-size objective. Worst field/wavelength RMS and throughput reported separately.',
        training='3x3 fields,36 pupil samples,1 wavelength thermal/3 monolithic.',
        audit='5x5 fields,256 pupil samples,1 wavelength thermal/3 monolithic; no audit feedback.',
        limits='Ideal engineering models; Newton/sequential intersections and sampled geometry/body checks. Not3D three-mirror reproduction or fabrication certification. One CMA random seed per candidate and arm.'))
    for p in paths:(out/p.name).write_bytes(p.read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run,jobs):print(json.dumps(serial(row)),flush=True)
    save(out/'completion.json',dict(cases=len(jobs),actual_training_traces=775*len(jobs),dense_audits=5*len(jobs)))


if __name__=='__main__':main()
