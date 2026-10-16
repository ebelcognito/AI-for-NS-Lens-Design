"""Build a readable final experimental record from completed immutable run files."""
import json
from pathlib import Path
import numpy as np
from nslens.analysis.refinement_sampling_study import save

ROOT=Path('experiments/refinement_sampling_20261005')
def read(p):return json.loads((ROOT/p).read_text())
def main():
 dev=read('mirror_audit.json'); conf=read('mirror_confirmation/mirror_audit.json');race=read('mirror_racing_confirmation/mirror_audit.json')
 td=read('tir_search/analysis.json');tc=read('tir_confirmation/analysis.json');tr=read('tir_racing/analysis.json')
 initial=read('mirror_diagnostics/diagnostic_summary.json');working=read('mirror_working_diagnostics/diagnostic_summary.json');tir_diag=read('tir_diagnostics/summary.json')
 lines=['OPTICAL REFINEMENT AND SAMPLE-INFORMED SEARCH — 2026-10-05','',
 'Goal: obtain one strong design. Compare selection of the next starting point and allocation of local refinement effort. No archive-cell objective and no text/LLM proposal generation.', '',
 'MAIN FINDINGS',
 '1. Simple local response proxies did not establish a reliable estimate of best achievable performance. The measured target was finite-budget CMA performance, never a global optimum.',
 '2. History-difference proposals improved over weak random restarts in development, but failed to replace the retained-candidate TIR baseline on fresh confirmation seeds.',
 '3. In the ordinary-mirror model, longer refinement and racing produced lower median loss than frequent retained perturbation/restarts. Racing was not a statistically established winner over uninterrupted CMA, and support/clearance failures remain material.',
 '4. All methods remain experimental. Existing production/default search policies were not changed.', '',
 'TIR RESULTS (4/5-sided curved polygons; unchanged acceptance gates)',
 'Each job: 400 physical 27-source-ray bundle evaluations, at most 1600 total proposals; equal reservation of up to eight 125-ray dense audits. Geometry-invalid proposals are recorded but are not fabricated physical calls.',
 'The portfolio comparator calls nslens.discovery.benchmark.run_job unchanged. Finalist selection/audits are shared. Success means folded-energy, root-support, TIR-margin, closure/pruning constraints pass; it does not mean good imaging.', '']
 for label,data in [('Development, seeds801–808',td),('Frozen DE confirmation, seeds1201–1212',tc),('Prospective racing transfer, seeds1501–1512',tr)]:
  lines.append(label)
  for k,aa in data['stats'].items():
   for arm,r in aa.items():lines.append(f"  {k} corners | {arm:16s} | verified {r['successes']}/{r['n']} | mean verified quality {r['mean_verified_quality']:.4f}")
  for base,r in data['paired'].items():lines.append(f"  Versus {base}: success-rate change {r['success_rate_difference']:+.3f}; 95% seed-block bootstrap {r['bootstrap95_by_seed']}; blocks={r['seed_blocks']}. Quality difference {r['mean_quality_difference']:+.4f}.")
  lines.append('')
 lines += ['ORDINARY THREE-MIRROR RESULTS','12 normalized geometry/power/aperture variables; true nearest-surface tracing but only a 2D cross-section with finite ideal parabolic mirrors. No sagittal imaging, diffraction, coatings, thickness or tolerance model.',
 'Search: 3 fields × 9 rays; frozen selected design audit: 21 fields × 201 rays, plus sampled proper mirror intersections at 513 points/surface. Qualification below means no sampled proper intersection and >=99% worst-field support, NOT optical specification compliance.',
 'Every arm gets4800 coarse evaluations including initial24-point pool and screening. All arms use the same CMA implementation. Reported time was measured under parallel CPU contention; no isolated speed claim.', '']
 for label,data in [('Development, seeds301–312',dev),('DE confirmation + later long-CMA control, seeds601–624',conf),('Frozen racing confirmation, seeds1301–1324',race)]:
  lines.append(label)
  for arm,r in data['summary'].items():lines.append(f"  {arm:15s} | n={r['n']:2d} | median audit loss {r['median_dense_loss']:.6f} | geometry/support qualified {r['qualified']}/{r['n']}")
  for key in ['de_vs_retained','race_vs_continuous','race_vs_retained']:
   if key in data['comparisons']:
    r=data['comparisons'][key];lines.append(f"  {key}: loss wins {r['wins']}/{r['n']}, sign p={r['two_sided_sign_p']:.4g}; median paired delta {r['median_paired_loss_difference']:+.6f}, bootstrap95 {r['bootstrap_95_median_difference']}.")
  lines.append('')
 lines += ['WHAT THE SAMPLERS ACTUALLY DO',
 'random: screen24 uniform candidate starts, refine the best for216 evaluations; repeat. This is a deliberately matched restart control, not a claimed optimal restart schedule.',
 'retained (mirror control): screen perturbations of the best previously evaluated samples, with every fourth screen global; same local CMA block. It is a controlled comparator, not a verbatim production mirror runner.',
 'de (mirror): create candidates around elite historical designs with F*(a-b) differences from full history, crossover and small jitter, plus25% uniform candidates; screen24, same local CMA.',
 'surrogate_de (mirror development only): ExtraTrees fit to all prior scalar losses ranks a pool of768 mixed candidates, using mean minus0.5 tree dispersion; retain explicit random exploration. Tree dispersion is not calibrated uncertainty. This was slower and did not improve DE development median.',
 'de_history (TIR): change only CMA restart centers to reflected best+F*(a-b) with coordinate crossover; same current constraints, initial40 Sobol proposals, sigma, population, stagnation policy and dense audit.',
 'continuous (mirror): start from the best of the shared24 measured points and spend the remaining budget on one CMA trajectory, with numerical-failure restart only.',
 'race (mirror): best 8 initial starts receive120 evaluations each; best 2 continue for480 each; the current winner consumes the remainder. CMA state is preserved across stages.',
 'race (TIR): same40 Sobol proposals; four diverse measured starts get 4 generations each; best 2 get 5 additional generations; continue best to400 physical calls. Ranking uses constraint_rank. Eliminated branches still contribute final candidates. This tests allocation, not a clean ablation of restart source.', '',
 'DIAGNOSTICS',
 'Mirror initial study:48 starts (24 uniform,24 with120-call warmup),32 train/16 held-out; every start gets two independent480-call CMA refinements. Targets are the best of the two, and separate post-hoc trajectory analysis predicts the independent second replicate.',
 '45/48 starts in the separate480-call-warmup cohort had full coarse support. This guards against interpreting missing-ray plateaus as aberration correctability.',
 'The mirror local proxy fits a bounded linear correction to image residuals and missing-ray penalties. It excludes the scalar evaluator\'s ordered-route progress term. It is NOT a full aberration-theory or wavefront predictor.',
 'Proxy construction costs 51 coarse traces per12-variable mirror parent, counted separately from960 refinement calls and warmup. Includes24 residual probes plus24 support probes, current measurements and measured proposed correction.', '']
 for label,data in [('Initial cohort',initial),('Fresh480-call-warmed cohort',working)]:
  lines.append(label)
  for key in ['initial_loss','residual_floor','physical_diagnostics']:
   r=data['scores'][key];lines.append(f"  {key}: held-out Spearman {r['spearman']:.3f}; pairwise ranking accuracy {r['pair_accuracy']:.3f} (16 starts; pairs not independent).")
 lines+=['TIR diagnostics:16 independent starts per corner family; choose best of8 initial measured proposals; coordinate probes ±.02, bounded correction ±.08; two independent160-call refinements. The response vector consists of constraint deficits and folded throughput, not imaging aberrations.']
 for k,r in tir_diag.items():lines.append(f"  {k} corners: initial merit rho={r['scores']['initial_merit']['spearman']:.3f}; local proxy rho={r['scores']['predicted_residual']['spearman']:.3f}; first-replicate60-call progress vs second-replicate final rho={r['scores']['early60']['spearman']:.3f}.")
 lines += ['', 'PAPER: WHAT FIGURE2 DOES AND DOES NOT SAY',
 'Bauer & Rolland, Roadmap for the unobscured three-mirror freeform design space, Optics Express29(17),26736–26753 (2021), DOI10.1364/OE.433643. Local PDF agent_context/paper_pdfs/oe-29-17-26736.pdf; Fig2 is PDF page6/printed26741.',
 'Fig2 plots minimum optimized package volume maintaining <=lambda/14 RMS WFE at550nm versus FOV, EPD and F-number, for several layout families. Based on200+ optimized designs, it is an empirical design roadmap under particular constraints. It is not a proof of the best attainable topology or a radius of optimizer convergence.',
 'Their envelope is the product of three Cartesian spans of reference-ray intersections across surfaces/fields. Stop placement and >10mm clearance assumptions matter. Failed diffraction-limited optimization is not proof of physical impossibility.',
 'Our existing 2D mirror/TIR models cannot reproduce this diffraction-limited-volume calculation. No diffraction-volume guarantee,90%-accurate optimum proxy, global route optimum, or paper reproduction is claimed.', '',
 'REPRODUCTION AND PROVENANCE',
 'All results, including reversals and physical failures, are retained. Per-run protocols contain seeds, budgets, sampling settings and source hashes; mirror searches save X/y populations in NPZ; TIR searches save every proposal in candidates.jsonl. Dense outcomes never feed restart/branch selection.',
 'Run Python through .venv/bin/python with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Entry points:',
 '  nslens.analysis.refinement_sampling_study --phase diagnostics|search --out NEW_DIR [--arms ... --seeds ...]',
 '  nslens.analysis.tir_sampling_study --out NEW_DIR --arms portfolio de_history --stage confirmation --seeds ... --corners 4 5 --search-calls 400',
 '  nslens.analysis.tir_racing_study --out NEW_DIR --seeds ...',
 'Use exact spaced CLI arguments in each saved protocol or module --help; the prose above abbreviates some values.',
 'Checks: 4 existing mirror-physics tests;10 existing polygon-discovery tests; synthetic4800-call ledger/bounds/audit-separation checks for five mirror arms; all completed TIR ledgers reconciled with physical-ray counts.',
 'Additional validation: all264 mirror search ledgers reconciled (1,267,200 coarse evaluations); all144 TIR search ledgers reconciled. Four selected TIR designs passed909-ray checks with identical results at200k and400k branch capacities. These examples are selected, not an extra unbiased comparison.',
 'Excluded from scientific results: initial spectrometer smoke and accidentally started/aborted spectrometer diagnostics; refined-start diagnostic retry after an unused feature-key error. Preserved logs identify these.', '',
 'NEXT RESEARCH TARGET',
 'Train a budget-conditioned continuation-value predictor from branch trajectories, evaluated by final physical success at equal total cost. Start with current loss plus short progress as the baseline. Do not train against an unknowable global optimum or imply that scalar current fitness measures topology potential.',
 'Before making any method default, incorporate the relevant optical/geometry acceptance constraints into search and test across more briefs. Distinguish proposal quality from allocation policy and from local-optimizer strength.', '']
 (ROOT/'RESULTS.txt').write_text('\n'.join(lines))
 save(ROOT/'key_results.json',dict(mirror_racing_confirmation=race['summary'],mirror_paired=race['comparisons'],tir_de_confirmation=tc['stats'],tir_racing=tr['stats'],tir_racing_paired=tr['paired']))
 print(ROOT/'RESULTS.txt')
if __name__=='__main__':main()
