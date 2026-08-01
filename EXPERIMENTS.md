# Experiment registry

Every `ns_probe/` script traced back to the experiment it produced. Built from the
experiment table in `agent_context/ns_rl_research/00_MASTER_CONTEXT.md` §7, the
discovery-control plan (doc 14), the descriptor ablation (doc 12), and the actual import
graph.

**Why this file exists.** `ns_probe/` looks like 46 redundant scripts. It is not. It is
~16 experiments and deliverables, each of which was given its own copy-pasted runner,
analyser and renderer. Deleting files here would delete *results*, not clutter. The fix
is to collapse the triplets onto shared machinery, and this table is the safety net that
makes that possible: each row becomes one config in phase 2.

Verdict column is copied from MASTER_CONTEXT, including the negative results — those are
part of the scientific record, not failures to clean up.

## Numbered experiments (MASTER_CONTEXT §6)

| # | experiment | runner | analysis / render | output | verdict |
|---|---|---|---|---|---|
| 1 | Phase-0 doublet | `run.py` (lib `probe.py`) | `analyze.py`, `render_designs.py` | `results/`, `results_seed1/` | ⚠️ negative (reproduced) |
| 2 | Phase-0 single | `run_single.py` (lib `probe_single.py`) | `analyze.py` | `results_single/` | ⚠️ negative |
| 3 | Reflectivity/TIR analysis (doc 06) | — (analysis only) | — | — | 💡 the pivot to TIR |
| 4 | Typed-interaction engine + demonstrator | `fold_demo.py` | `test_typed_interactions.py`, `tir_demo.py` | — | 🔧 + ✅ 86.8% via `T@A→TIR@B→TIR@A→T@B` |
| 5 | **Fold-QD (headline)** | `run_fold_qd.py` | `analyze_fold_qd.py` | `results_fold_qd/` | ✅ 109 cells, 2-TIR 96%, 4-TIR 91% / spot 0.66 |
| 6 | Orderly-QD | `run_orderly_qd.py` | `analyze_fold_qd.py` | `results_orderly_qd/` | ✅ coherent folds |
| 7 | Tilt ablation | `run_fold_qd.py --lock-tilt` | `analyze_fold_qd.py` | `results_fold_qd_notilt/` | ✅ tilt is causal |
| 8 | Multi-field QD | `run_multifield_qd.py` | `analyze_fold_qd.py` | `results_multifield_qd/` | ✅ 8 field-robust folds |
| 9 | Physics validation | — | `analyze_physics.py` | — | ✅ 100% of 697 TIR events past critical |
| 10 | Symmetric catadioptric (longer) | `run_single.py` | `analyze.py` | `results_single_long/` | ✅ overturned the "symmetric can't focus" cap |

## Discovery control — E1–E4 (docs 14, 15, 16)

The program's #1 question: does CMA-ME *search*, or merely enumerate?

| experiment | runner | analysis | output |
|---|---|---|---|
| E1–E4, collinear | `discovery_control.py` | `analyze_disc_e1.py`, `analyze_disc_e2a_e3_e4.py` | `results_disc/`, `logs_disc/` |
| E1–E4, prism / perpendicular (doc 16) | `discovery_control_prism.py` | `analyze_disc_prism.py` | `results_disc_prism/`, `logs_disc_prism/` |
| figures | — | `render_disc_figure.py`, `render_disc_top10.py`, `render_disc_random.py` | — |

Verdict: CMA-ME earns "search", robust across both collinear and forced-fold geometry.

## Descriptor ablation (doc 12)

| runner | analysis | output |
|---|---|---|
| `run_fold_qd_descriptors.py` | `compare_descriptors.py`, `descriptor_summary_figure.py` | `results_desc_*` (12 dirs: baseline / locus / combo / tilt / nsfrac / critmargin / robust, seeds 0–2) |

## AR/VR prism

| runner | geometry lib | analysis / render | output |
|---|---|---|---|
| `run_arvr_qd.py` | `arvr_prism.py` | `analyze_arvr_qd.py`, `render_arvr_gallery.py`, `arvr_prism_fig.py` | `results_arvr*` (6 dirs) |

## Prism fold (perpendicular)

| runner | geometry lib | output |
|---|---|---|
| `run_prism_fold_qd.py` | `prism_fold.py` (imports `arvr_prism`) | `results_fold_qd_*` |

## Refinement and thinning

| script | role | output |
|---|---|---|
| `refine_lm.py` | Levenberg–Marquardt refinement of archive champions | `results_refine_lm/` |
| `refine_sweep.py` | sweep driver over `refine_lm` | — |
| `render_refine_comparison.py` | before/after figure | — |
| `thin_design.py` | thickness continuation. **Must be run with `--min-margin 3`** or it cheats via knife-edge TIR | `results_thin/` |

## Deliverables and instrumentation

| script | role |
|---|---|
| `export_fold_for_codev.py` | CODE V sequential export (the freeform-prism handoff) |
| `export_fold_folded.py` | folded-axis variant; imports `_folded_trace.py` |
| `_folded_trace.py` | folded-axis tracer, validated against the engine |
| `export_arvr_for_autodiff.py` | hand-off from the AR/VR archive into `ns_autodiff` |
| `capture_cma_step.py` + `render_cma_step.py` | replays one CMA-ME generation (seed → population → winner). Visualization only — populations are not stored in `archive.json`, so this is the only way to see a step |
| `fold_gallery.py` | orderly-fold gallery; imported by the disc renderers |
| `render_fig_archive.py`, `render_fig_exemplars.py` | paper figures |

## What this means for the refactor

Nothing above is safe to delete. The redundancy is in the *shape*, not the content:

- **10 runners** that differ only in fitness, descriptor pair and source/detector placement
- **8 analysers** that re-implement the same `load / high_fold_families / metrics / ci / fmt` helpers
- **12 renderers** built on the same archive-grid and champion-panel primitives

Phase 2 collapses each of the tables above into one config row against a shared driver.
Target: ~12 modules + ~16 configs, replacing 46 scripts, with every experiment above
still reproducible by name.

`tir_demo.py` calls itself "legacy" and MASTER_CONTEXT agrees — but
`test_typed_interactions.py` (22 physics checks, all currently passing) imports it, so it
stays until those checks are ported.
