"""Physical, untrained screening diagnostics on a fully audited cached cohort."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr

OUT = Path('experiments/refinement_potential_20261006')


def main():
    rows = []
    for line in (OUT / 'dense_audits.jsonl').read_text().splitlines():
        record = json.loads(line)
        data = json.loads(Path(record['source']).read_text())
        r, j, a = data['refinements']['3'], data['source_job'], record['audit']
        c = r['compatibility']
        rows.append(dict(source=record['source'], donor=j['seed'], task=j['task'],
                         arm=j['arm'], repeat=j['repeat'],
                         initial=r['initial']['worst_mapping_rms_mm'],
                         initial_loss=r['initial']['loss'],
                         compatibility=c['rms_unexplained_slope'],
                         fitted=r['fitted']['worst_mapping_rms_mm'],
                         fitted_loss=r['fitted']['loss'],
                         final_loss=r['coarse']['loss'],
                         coarse_final=r['coarse']['worst_mapping_rms_mm'],
                         final=a['worst_mapping_rms_mm'],
                         feasible=a['routing_pass'], success=a['threshold_reached']))
    assert len(rows) == 144
    assert all(np.isfinite(r[k]) for r in rows for k in ('initial','fitted','compatibility','final'))
    with (OUT / 'dense_screening_rows.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    y = np.array([r['final'] for r in rows])
    groups = [[i for i,r in enumerate(rows) if r['task']==t and r['donor']==d]
              for d in range(8) for t in ('field8','aperture42')]
    assert all(len(g)==9 for g in groups)
    oracle = np.array([min(y[g]) for g in groups])
    random = np.array([np.mean(y[g]) for g in groups])
    methods = {}
    selected = {}
    for key in ('initial','initial_loss','compatibility','fitted','fitted_loss'):
        x = np.array([r[key] for r in rows])
        indices = [min(g, key=lambda i:x[i]) for g in groups]
        scores = y[indices]
        selected[key] = scores
        methods[key] = dict(spearman_final=float(spearmanr(x,y).statistic),
            mean_selected_rms=float(np.mean(scores)), mean_oracle_regret=float(np.mean(scores-oracle)),
            successes=sum(rows[i]['success'] for i in indices),
            feasible=sum(rows[i]['feasible'] for i in indices),
            oracle_matches=int(np.sum(np.isclose(scores,oracle,rtol=0,atol=1e-10))),
            selected_sources=[rows[i]['source'] for i in indices])
    rng = np.random.default_rng(61006)
    bootstrap_ids = rng.integers(0,8,size=(20000,8))
    for key in ('compatibility','fitted'):
        delta=(selected['initial']-selected[key]).reshape(8,2).mean(axis=1)
        methods[key]['improvement_over_initial_rms'] = dict(
            mean=float(delta.mean()), donor_means=delta.tolist(),
            donor_bootstrap_95_percent_interval=np.quantile(delta[bootstrap_ids].mean(axis=1),[.025,.975]).tolist())
    relative=np.array([(r['fitted_loss']-r['final_loss'])/max(r['fitted_loss'],1e-12) for r in rows])
    digest=[hashlib.sha256(json.dumps(json.loads(Path(r['source']).read_text())['refinements']['3']['initial']['genes']).encode()).hexdigest() for r in rows]
    summary=dict(n=144,donors=8,pools=16,candidates_per_pool=9,
        unique_initial_gene_vectors=len(set(digest)),
        feasible=sum(r['feasible'] for r in rows),successes=sum(r['success'] for r in rows),
        methods=methods,random_expected_rms=float(random.mean()),oracle_mean_rms=float(oracle.mean()),
        random_expected_successes=float(sum(np.mean([rows[i]['success'] for i in g]) for g in groups)),
        median_relative_merit_gain_after_five_calls=float(np.median(relative)),
        identical_fitted_final_merit=sum(r['fitted_loss']==r['final_loss'] for r in rows),
        initial_to_final_worst_coarse_rms_worsened=sum(r['coarse_final']>r['initial'] for r in rows),
        screening_cost=dict(initial_rms=1,compatibility_including_initial=2,fitted_probe_including_initial=5,
                            full_refinement=261,nine_full_refinements=2349,
                            nine_probes_plus_one_continuation=9*5+256,
                            nine_probes_plus_full_selected_restart=9*5+261),
        limitations=['Retrospective cached search endpoints; only eight donor families and two specifications.',
                    'No learned fitting in physical screening methods. Dense outcomes never entered scores.',
                    'The five-call probe is the initialization of the endpoint refiner: early stopping/allocation evidence, not independent prediction of global potential.',
                    'The oracle minimizes RMS among recorded endpoints; it is not a global optimum.',
                    'Cost arithmetic excludes common discovery, training, and audit costs; online savings not demonstrated.',
                    'Worst-field RMS differs from mean-square optimization merit; inspect feasibility separately.'])
    (OUT/'dense_summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    colors=['tab:blue' if r['task']=='field8' else 'tab:orange' for r in rows]
    for axis,key,title in zip(ax,('initial','compatibility','fitted'),('Initial worst-field RMS','Unexplained slope','Five-call probe RMS')):
        axis.scatter([r[key] for r in rows],y,c=colors,s=16,alpha=.65)
        axis.set(xscale='log',yscale='log',xlabel=title,ylabel='Dense final RMS (mm)',
                 title=f"Rank correlation: {methods[key]['spearman_final']:.3f}")
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=.15)
    ax[0].legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=t)
                          for c,t in [('tab:blue','field8'),('tab:orange','aperture42')]],fontsize=8)
    fig.suptitle('144 cached three-mirror records; final outcome after 261 calls')
    fig.savefig(OUT/'screening.png',dpi=160);plt.close(fig)
    text=['THREE-MIRROR REFINEMENT POTENTIAL: RETROSPECTIVE SCREENING',
          f"144 records ({len(set(digest))} unique initial gene vectors); 8 donor families; 16 pools of 9 candidates; all endpoints densely audited.",
          'Method | rank correlation | selected mean RMS mm | passes /16 | feasible /16 | oracle matches /16']
    for key,v in methods.items():
        text.append(f"{key} | {v['spearman_final']:.4f} | {v['mean_selected_rms']:.4f} | {v['successes']} | {v['feasible']} | {v['oracle_matches']}")
    text.extend(['',f"Random expected RMS: {random.mean():.4f}; recorded-outcome oracle: {oracle.mean():.4f}",
                 f"Median merit improvement from the remaining 256 CMA calls: {100*np.median(relative):.4f}%",
                 '',*summary['limitations']])
    (OUT/'RESULTS.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__':
    main()
