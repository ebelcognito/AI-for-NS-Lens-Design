"""Summarize prospective 3D probe experiments without pooling environments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import numpy as np
from scipy.stats import spearmanr


def finite(value):
    return value is not None and np.isfinite(value)


def corr(x,y):
    mask=np.array([finite(a) and finite(b) for a,b in zip(x,y)])
    a,b=np.asarray(x,dtype=float)[mask],np.asarray(y,dtype=float)[mask]
    return dict(n=int(mask.sum()),rho=float(spearmanr(a,b).statistic)
                if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None)


def mean(values):
    return float(np.mean(values)) if all(finite(v) for v in values) else None


def score(metrics):
    v=metrics['worst_mapping_rms_mm']
    return v if finite(v) else float('inf')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    rows=[json.loads(p.read_text()) for p in sorted(out.glob('*__*__*.json'))]
    assert len(rows)==48
    summaries={}
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ei,environment in enumerate(('thermal','monolithic')):
        subset=[r for r in rows if r['job']['environment']==environment]
        assert len(subset)==24
        initial=[r['initial']['worst_mapping_rms_mm'] for r in subset]
        short=[r['fitted']['metrics']['worst_mapping_rms_mm'] for r in subset]
        slope=[r['compatibility'].get('rms_unexplained_slope') for r in subset]
        warm=[r['arms']['probe_cma']['audit']['worst_mapping_rms_mm'] for r in subset]
        cold=[r['arms']['cold_cma']['audit']['worst_mapping_rms_mm'] for r in subset]
        correlations={}
        for name,predictor in [('initial',initial),('short_probe',short),('slope',slope)]:
            correlations[name]=dict(to_probe_cma=corr(predictor,warm),to_cold_cma=corr(predictor,cold))
        common=[i for i in range(len(subset)) if all(finite(v[i]) for v in (initial,short,slope,warm,cold))]
        paired_correlations={name:dict(to_probe_cma=corr([x[i] for i in common],[warm[i] for i in common]),
                                      to_cold_cma=corr([x[i] for i in common],[cold[i] for i in common]))
                             for name,x in [('initial',initial),('short_probe',short),('slope',slope)]}
        arm_summary={}
        for arm in ('probe_cma','probe_wide_cma','cold_cma'):
            data=[r['arms'][arm] for r in subset]
            ratios=[(r['fitted']['metrics']['loss']-a['coarse']['loss'])/max(r['fitted']['metrics']['loss'],1e-12)
                    for r,a in zip(subset,data)]
            optical_ratios=[1-a['audit']['worst_mapping_rms_mm']/r['fitted_audit']['worst_mapping_rms_mm']
                for r,a in zip(subset,data) if finite(a['audit']['worst_mapping_rms_mm'])
                and finite(r['fitted_audit']['worst_mapping_rms_mm']) and r['fitted_audit']['worst_mapping_rms_mm']>0]
            arm_summary[arm]=dict(
                mean_dense_rms_mm=mean([a['audit']['worst_mapping_rms_mm'] for a in data]),
                median_dense_rms_mm=float(np.median([score(a['audit']) for a in data])),
                mean_dense_loss=float(np.mean([a['audit']['loss'] for a in data])),
                mean_dense_throughput=float(np.mean([a['audit']['throughput'] for a in data])),
                min_dense_worst_support=float(min(a['audit']['worst_support'] for a in data)),
                empty_rms_cases=sum(a['audit']['worst_mapping_rms_mm'] is None for a in data),
                physical_geometry_failures=sum(not a['audit']['physical_geometry'] for a in data),
                support_retention_cases=sum(a['audit']['worst_support']>=r['initial_audit']['worst_support']-.02 for r,a in zip(subset,data)),
                median_dense_rms_improvement_relative_to_probe=float(np.median(optical_ratios)) if optical_ratios else None,
                median_coarse_merit_improvement_relative_to_probe=float(np.median(ratios)))
        selection={}
        for predictor in ('initial','slope','short_probe'):
            picks=[];oracles=[];details=[]
            for family in range(3):
                ids=[i for i,r in enumerate(subset) if r['job']['family']==family]
                values={'initial':initial,'slope':slope,'short_probe':short}[predictor]
                chosen=min(ids,key=lambda i:values[i] if finite(values[i]) else float('inf'))
                oracle=min(ids,key=lambda i:warm[i] if finite(warm[i]) else float('inf'))
                picks.append(warm[chosen]);oracles.append(warm[oracle])
                details.append(dict(family=family,chosen_variant=subset[chosen]['job']['variant'],
                    oracle_variant=subset[oracle]['job']['variant'],selected_dense_rms_mm=warm[chosen],
                    oracle_dense_rms_mm=warm[oracle],selected_dense_worst_support=subset[chosen]['arms']['probe_cma']['audit']['worst_support']))
            selection[predictor]=dict(mean_selected_rms_mm=mean(picks),oracle_mean_rms_mm=mean(oracles),
                exact_oracle_picks=sum(d['chosen_variant']==d['oracle_variant'] for d in details),families=details)
        summaries[environment]=dict(cases=24,donor_families=3,correlations=correlations,
            common_finite_case_correlations=paired_correlations,
            initial_mean_dense_rms_mm=mean([r['initial_audit']['worst_mapping_rms_mm'] for r in subset]),
            probe_mean_dense_rms_mm=mean([r['fitted_audit']['worst_mapping_rms_mm'] for r in subset]),
            median_saturated_fit_coefficients=float(np.median([r['compatibility'].get('bound_saturated',0) for r in subset])),
            arms=arm_summary,selection=selection)
        pixel=.012 if environment=='thermal' else .017
        for axis,x,title in zip(axes[ei],(initial,short),('Initial mapping RMS','Four-call probe RMS')):
            ok=[i for i in common if x[i]>0 and warm[i]>0]
            axis.scatter(np.array(x,dtype=float)[ok]/pixel,np.array(warm,dtype=float)[ok]/pixel,
                         c=[subset[i]['job']['family'] for i in ok],cmap='viridis',s=25)
            axis.set(xscale='log',yscale='log',xlabel=f'{title} (pixels)',ylabel='Dense final RMS (pixels)',
                     title=f'{environment}: rank correlation {corr([x[i] for i in ok],[warm[i] for i in ok])["rho"]:.3f} (n={len(ok)})')
            for coordinate in (axis.xaxis,axis.yaxis):
                coordinate.set_major_locator(LogLocator(base=10,subs=(1,2,3,5)))
                coordinate.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
                coordinate.set_minor_formatter(NullFormatter())
            axis.grid(alpha=.15)
    fig.suptitle('3D final-surface correction: 24 local layouts per environment')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    summary=dict(environments=summaries,actual_training_traces=sum(r['actual_training_traces'] for r in rows),
                 dense_audits=sum(r['dense_audits'] for r in rows),
                 caveats=['Three saved donors per environment, seven local perturbations each; no global architecture coverage.',
                          'Worst RMS calculated on valid rays; throughput and empty fields reported separately.',
                          'Four-call probe uses two-component gradients of one sag surface; frozen hits are approximate.',
                          'Full objective is mapping plus loss penalties, not the native camera spot objective.',
                          'RMS correlations do not prove feasible good imaging or a superior discovery sampler.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['3D PHYSICAL CORRECTION PROBE: PROSPECTIVE PILOT','48 candidate layouts, two environments, three donor families each.']
    for env,s in summaries.items():
        lines.extend(['',env.upper(),f"Initial / probe mean dense RMS: {s['initial_mean_dense_rms_mm']} / {s['probe_mean_dense_rms_mm']} mm"])
        for name,v in s['correlations'].items():lines.append(f"{name}: rho to warm final {v['to_probe_cma']['rho']}; rho to cold final {v['to_cold_cma']['rho']}")
        for name,v in s['common_finite_case_correlations'].items():lines.append(f"Common {v['to_probe_cma']['n']} cases, {name}: rho to warm final {v['to_probe_cma']['rho']}; to cold final {v['to_cold_cma']['rho']}")
        for name,a in s['arms'].items():lines.append(f"{name}: mean RMS {a['mean_dense_rms_mm']} mm; mean throughput {a['mean_dense_throughput']:.3f}; median RMS improvement vs probe {a['median_dense_rms_improvement_relative_to_probe']}; median merit improvement vs probe {a['median_coarse_merit_improvement_relative_to_probe']:.4%}; support retained {a['support_retention_cases']}/24")
        for name,a in s['selection'].items():lines.append(f"Select by {name}: {a['mean_selected_rms_mm']} mm, oracle matches {a['exact_oracle_picks']}/3")
        lines.append('Empty-field final outcomes: '+str({name:a['empty_rms_cases'] for name,a in s['arms'].items()}))
    lines.extend(['',*summary['caveats']])
    if (out/'capacity_diagnostics.json').exists():
        capacity=json.loads((out/'capacity_diagnostics.json').read_text())
        lines.append('\nFROZEN-HIT CAPACITY DIAGNOSTIC (48 additional initial traces)')
        for env,s in capacity['summary'].items():
            lines.append(f"{env}: median saturated coefficients {s['bound_saturated']}/{s['active_parameters']}; bounded/unbounded fraction of squared slope correction explained {s['bounded_explained_fraction']:.3f}/{s['unbounded_explained_fraction']:.3f}")
        lines.append(capacity['interpretation'])
    if (out/'extreme_ray_diagnostic.json').exists():
        lines.append('\nMonolithic 301 variant7: coarse worst RMS0.700mm versus dense64.046mm. A traced ray exits nearly parallel to the detector plane (direction z=0.000445); surface sag and Snell reconstruction were checked. See extreme_ray_diagnostic.json. This is an audit failure, not a good design.')
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('\n'.join(lines))


if __name__=='__main__':main()
