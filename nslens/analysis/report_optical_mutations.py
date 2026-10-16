"""Audited report for optical compensation and numerical action selection."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesRegressor
from nslens.analysis.refinement_sampling_study import save

ROOT=Path('experiments/optical_mutation_20261005')
REPAIR=Path('experiments/optical_target_repair_20261005')

def main():
 data=json.loads((ROOT/'analysis.json').read_text());rows=[json.loads(p.read_text()) for p in ROOT.glob('*_*_*.json') if p.name.startswith(('mirror_','tir_'))];assert len(rows)==93
 totals={k:sum(r['ledger'][k] for r in rows) for k in rows[0]['ledger']};norm_errors=[]
 for r in rows:
  assert r['ledger']['coarse_attempts']==1+r['probe_calls']+6*(2+96+r['probe_calls'])
  for a in r['actions']:norm_errors.append(abs(np.linalg.norm(a['raw_delta'])-np.linalg.norm(a['compensated_delta'])))
 assert max(norm_errors)<1e-12
 # Exploratory transfer: same19-feature optical state/action schema, train one polygon family only.
 transfer={}
 for source,target in [(4,5),(5,4)]:
  train=[r for r in rows if r['job']['backend']=='tir' and r['job']['split']=='train' and r['job']['corners']==source];test=[r for r in rows if r['job']['backend']=='tir' and r['job']['split']=='test' and r['job']['corners']==target]
  X=np.array([a['features'] for r in train for a in r['actions']]);y=np.array([r['parent']['loss']-a['endpoints']['compensated']['loss'] for r in train for a in r['actions']]);f=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=5,max_features=.8,random_state=93,n_jobs=1).fit(X,y)
  learned=[];random=[]
  for r in test:
   aa=r['actions'];i=int(np.argmax(f.predict([a['features'] for a in aa])));learned.append(aa[i]['endpoints']['compensated']['audit']['loss']);random.append(np.mean([a['endpoints']['compensated']['audit']['loss'] for a in aa]))
  delta=np.array(learned)-random;transfer[f'{source}_to_{target}']=dict(train_parents=len(train),test_parents=len(test),mean_delta=float(delta.mean()),wins=int(sum(delta<0)),losses=int(sum(delta>0)))
 save(ROOT/'family_transfer.json',transfer)
 repair=[json.loads(p.read_text()) for p in REPAIR.glob('mirror_0_test_*.json')];rv={k:np.array([np.mean([a['endpoints'][k]['audit']['loss'] for a in r['actions']]) for r in repair]) for k in ['target_repair','raw_matched']};rd=rv['target_repair']-rv['raw_matched'];rng=np.random.default_rng(831);boot=np.mean(rng.choice(rd,(20000,len(rd)),replace=True),axis=1)
 rs=dict(test_parents=len(repair),mean_loss={k:float(v.mean()) for k,v in rv.items()},paired_mean_difference=float(rd.mean()),bootstrap95=np.quantile(boot,[.025,.975]),wins=int(sum(rd<0)),losses=int(sum(rd>0)),sign_p=float(binomtest(sum(rd<0),sum(rd!=0)).pvalue))
 save(REPAIR/'analysis.json',rs)
 lines=['OPTICALLY GROUNDED MUTATION EXPERIMENT — 2026-10-05','',
 'DECISION',
 'The local compensation primitive works: it changes the design while disturbing selected optical quantities much less. This did not translate into a reliable gain over a cost-matched uncompensated control. The learned selector is exploratory and does not justify replacing the current search policy.',
 'Scope: ordinary three-mirror2D cross-sections and curved4/5-sided TIR bodies. No spectrometer. No text/LLM proposals, no archive-cell objective, and no claim of diffraction-limited or globally optimal designs.', '',
 'WHAT WAS TESTED',
 '93 parent configurations:48 training and45 held-out. Every parent generated6 paired raw/compensated proposals (558 pairs). Mirror parents:24 train/23 test. TIR:12 train and11 test per corner family. Smoke parents1301/1501 excluded from held-out analysis.',
 'Each proposal changes curvature, fold/angle, or spacing/shape parameters. Two predeclared normalized radii(.025,.075) per family. Both arms have equal actual Euclidean mutation norm after bounds. Mirror apertures remain fixed.',
 'Mirror protected vector: normalized on-axis centroid, field-image slope (effective focal mapping), and on-axis pupil-image slope. The Jacobian has3 rows and9 active columns.',
 'TIR protected vector:3 normalized detector centroids plus source and detector clearances.19 fixed physical features support the same learned action interface for quads and pentagons. This is centroid/clearance preservation, not a claim of exact paraxial power or TIR-angle-margin conservation.',
 'Compensation is a damped local nullspace projection: d_comp=d-J^T(JJ^T+lambda I)^-1 Jd, then matched norm and bounds. Columns crossing invalid observations are masked. No network is needed to compute compensation.',
 'TIR acceptance gates are unchanged, but spot_weight=1 adds a finite-bundle detector-spot preference to quality. The scalar search merit is max violation+0.1sum violations-0.01quality; it is not identical to the historical lexicographic portfolio ranking. All arms here share it.', '',
 'COST CONTROL',
 '18 mirror or34/40 TIR sensitivity probes per parent. Compensated proposal: probes+1 child+48 CMA refinement evaluations. Cost-matched raw proposal:1 child+(48+probe_count) refinement evaluations. Parent observation is shared. A raw48 arm isolates the mechanism without cost matching.',
 'Actual dataset collection reuses a parent Jacobian across6 counterfactual actions; reported standalone decision costs conservatively charge the full Jacobian to the compensated choice. The raw norm-matching control is given the matched radius without charging it for discovering that radius; this favors the control.',
 'Training data collection is NOT free: all actual calls are separately recorded in each parent ledger. Any future training amortization requires enough subsequent searches; this study does not establish that break-even point.',
 f'Actual primary dataset totals: {totals}.',
 'Counterfactual outcomes were collected offline for every action. Random selection is reported as its exact expectation over6 actions, not as if it obtained6 candidate outcomes for free. Inner refinement uses CMA popsize8/sigma.015, fixed across arms, with common per-action random seeds.', '',
 'HELD-OUT RESULTS (dense loss: lower is better)',]
 for backend,r in data.items():
  lines.append('\n'+backend.upper())
  raw=r['protected_drift']['raw']['median'];comp=r['protected_drift']['compensated']['median'];lines.append(f'  Median protected drift: raw {raw:.6g}, compensated {comp:.6g}; ratio {raw/max(comp,1e-30):.2f}x. Conditional on valid measurable protected outputs; see analysis.json for counts.')
  for k in ['random_raw48','random_raw_matched','random_compensated','learned_scalar','learned_physical','batch_raw','batch_compensated']:
   z=r['methods'][k];lines.append(f"  {k:23s} mean loss={z['mean_dense_loss']:.8g}; audit acceptance={z['mean_audit_acceptance']:.4f}")
  for k in ['random_compensated_vs_random_raw_matched','learned_physical_vs_random_compensated','batch_compensated_vs_batch_raw']:
   z=r['paired'][k];lines.append(f"  {k}: paired delta={z['mean_loss_delta']:+.7g},95% bootstrap={z['bootstrap95_seed_blocks']}, seed-block wins/losses={z['block_wins']}/{z['block_losses']}, sign p={z['sign_p']:.4g}.")
 lines+=['', 'NUMERICAL LEARNING TEST',
 'ExtraTrees160trees/minleaf5/max_features.8, fixed before the main run. The physical model sees parent optical features and action features (family, radius, actual norm, predicted protected change and alignment). Scalar baseline sees current loss and action family/size only. Training labels are guarded coarse loss improvements after48 refinement calls.',
 'Models train only on train-parent ancestries. Dense audit results never determine labels, action selection or hyperparameters. Saved mirror_selector.joblib and tir_selector.joblib contain the fitted numerical selectors and training provenance.',
 'Secondary batch analysis (post-hoc): select the best immediate coarse child among6, then refine only that child. Compensated cost=J+6+48 and raw cost=6+48+J, plus common parent observation. This tests amortized batch use without an outcome oracle. It was not an independently confirmed online rollout.',
 'Exploratory cross-family transfer (no new simulator calls): '+json.dumps(transfer), '',
 'FOCUSED FOLLOW-UP: FIX ERRORS RATHER THAN PRESERVE THEM',
 'The mirror follow-up targets protected values[0,sign(current field slope),0]: centered image, nominal +/-50mm mapping and zero on-axis pupil slope. It adds a local least-squares correction before norm/bound matching.',
 'Same parent cohorts, same paired mutation norms and standalone evaluation budgets. This is a hypothesis-driven follow-up on reused parents, not fresh-parent confirmation.',
 'Results: '+json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in rs.items()}), '',
 'PHYSICAL AUDITS AND LIMITS',
 'Mirror:11 fields x41 pupil rays; sampled mirror intersections at513 points/surface. Audit acceptance requires>=99% worst-field support and no sampled proper mirror crossing. This is NOT an imaging-specification pass or proof of mechanical clearance.',
 'TIR:5 fields x25 rays and1024 geometry samples. Audit acceptance uses the existing energy, distinct-source support, TIR-margin, closure and pruning gates. All these models are2D. No sagittal imaging, full wavefront model, tolerance robustness or diffraction-limit claim.',
 'TIR mean loss is affected by a few infeasible parents; negative losses indicate favorable scalar routing/spot merit. Reported means are not percentages of optical quality. Seed-block bootstrap keeps quad/pentagon units with the same seed together; sample sizes remain small and comparisons are exploratory.',
 'The same corpus is reused across the reported policy comparisons; confidence intervals and p-values are not multiplicity-adjusted. A preservation win is not a global-search win. No multi-step learned search rollout was established.', '',
 'VERIFICATION',
 '4 existing mirror tests and10 existing polygon tests passed. Synthetic linear cases verified norm preservation, nullspace cancellation and bounds. All93 primary ledgers reconciled; all558 paired mutation norms matched within1e-12; aperture changes were zero.',
 'Worker code was reviewed; dense TIR sampling, bounded finite differences and mirror incidence wrapping were corrected before the main run. Original evaluator/default search files were not changed.', '',
 'REPRODUCE',
 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.optical_mutation_experiment --out NEW_DIRECTORY --workers6',
 'Replace --workers6 with --workers 6. Exact parent sources and hashes are in protocol.json; run requires the preserved prior-parent experiment files.',
 'Analysis: .venv/bin/python -m nslens.analysis.optical_mutation_experiment --out experiments/optical_mutation_20261005 --analyze-only',
 'Follow-up: nslens.analysis.targeted_mirror_mutation. Its default output directory must not exist before running.', '',
 'NEXT TEST WORTH DOING',
 'Reduce sensitivity cost by reusing a locally validated Jacobian across several generations, with explicit invalidation after path changes. Compare against giving the same evaluations to ordinary refinement. Collect more independent parent families before expanding the learned selector.', '']
 (ROOT/'RESULTS.txt').write_text('\n'.join(lines).replace('--workers6','--workers 6').replace('Replace --workers 6 with --workers 6. ',''))
 save(ROOT/'verification.json',dict(parents=len(rows),action_pairs=len(norm_errors),max_norm_difference=max(norm_errors),ledger_totals=totals,physics_tests_passed=14))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(13,4))
 for i,b in enumerate(['mirror','tir']):
  r=data[b];ax[i].bar(['Raw','Compensated'],[r['protected_drift'][k]['median'] for k in ['raw','compensated']]);ax[i].set(yscale='log',title=b+' protected drift',ylabel='Median normalized drift')
 colors=['tab:blue','tab:orange']
 for i,b in enumerate(['mirror','tir']):
  rr=data[b]['parent_results'];delta=np.array([r['scores']['random_compensated']-r['scores']['random_raw_matched'] for r in rr]);ax[2].scatter(np.full(len(delta),i)+np.linspace(-.08,.08,len(delta)),delta,alpha=.65,color=colors[i])
 ax[2].axhline(0,color='black',lw=1);ax[2].set(xticks=[0,1],xticklabels=['Mirror','TIR'],ylabel='Compensated minus cost-matched raw loss',title='Held-out parent outcomes')
 fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=170)
 source=ROOT/'final_sources';source.mkdir(exist_ok=True);hashes={}
 for f in ['compensated_mirror','compensated_tir','optical_mutation_experiment','targeted_mirror_mutation','report_optical_mutations']:
  p=Path('nslens/analysis')/(f+'.py');shutil.copyfile(p,source/p.name);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [Path('nslens/three_mirror/pilot.py'),Path('nslens/discovery/folded.py'),Path('nslens/discovery/polygon.py'),Path('nslens/search/cma_me.py')]:
  shutil.copyfile(p,source/str(p).replace('/','__'));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(ROOT/'final_source_hashes.json',hashes);print(ROOT/'RESULTS.txt')
if __name__=='__main__':main()
