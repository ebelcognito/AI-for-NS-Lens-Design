# Confirmed numerical discovery improvement — 2026-09-11

The retained-candidate CMA policy (`portfolio`) is the working numerical discovery
baseline. On a frozen fresh-seed experiment it produced a dense-verified folded
routing design in **41/72 runs (56.9%)**, versus **12/72 (16.7%)** for naive CMA.
The paired improvement is **40.3 percentage points**, with a 95% seed-block bootstrap
interval **[26.4, 55.6] points** and one-sided exact seed-sign **p = 0.000488**.
It met the acceptance rule declared before execution.

This is an improvement in **verified discovery yield** on the declared tasks. It is
not an imaging-quality improvement or proof of better final compacted optics.

## Results

Each method received 400 physical search calls per task, followed by up to eight
reserved final audits. The experiment covered twelve fresh seeds (100–111), three
corner families (3/4/5), and two apertures (4/6 mm): 72 runs per method, 288 total.

| Method | Successful runs | Success rate | Summed within-run audited routes |
|---|---:|---:|---:|
| Naive CMA | 12/72 | 16.7% | 17 |
| Uniform random | 9/72 | 12.5% | 9 |
| Earlier guided CMA | 24/72 | 33.3% | 31 |
| **Retained-candidate CMA** | **41/72** | **56.9%** | **45** |

The selected policy improved on the earlier guided CMA by 23.6 percentage points
(95% seed-block interval [6.9, 38.9]; exploratory one-sided sign p=0.0195).
Secondary comparisons were not the primary acceptance test. Route totals sum separate
run archives; they do not represent globally unique optical topologies. The 226 passing
final candidates from the selected method are also not independent trials.

| Aperture | Corners | Naive successes /12 | Earlier guided /12 | Selected /12 |
|---|---:|---:|---:|---:|
| 4 mm | 3 | 1 | 3 | 5 |
| 4 mm | 4 | 6 | 5 | 8 |
| 4 mm | 5 | 1 | 5 | 7 |
| 6 mm | 3 | 0 | 2 | 3 |
| 6 mm | 4 | 3 | 3 | 11 |
| 6 mm | 5 | 1 | 6 | 7 |

Triangular discovery remains difficult. Even the improved method failed in 31/72 runs;
it should be used as a repeated numerical proposer, not a guaranteed design solver.

![Confirmation success rates](../../runs/discovery_improvement_confirm_20260911_v1/comparison.png)

## What changed in the proposer

The historical behavior flattened rejected candidates and restarted when a generation
added nothing to the archive. In the confirmation control, the median emitter lifetime
was **one generation**. The selected policy's median was **nine generations**.

The selected implementation combines:

1. Normalized geometry genes, dynamic chord-relative curvature bounds and reflected
   CMA proposals, shared with the new discovery environment.
2. Requirement-relative violation vectors. Infeasible candidates are ordered by the
   worst violation, then the next worst, etc.; measured routing progress breaks ties.
   Feasibility has priority over throughput. There is no weighted penalty sum that
   allows one optical requirement to compensate for another.
3. Forty scrambled Sobol startup proposals, included in the search allowance.
4. A pool of twelve measured candidates, including promising failures. Retain the best
   and geometrically separated alternatives within a quality shortlist; restart from
   the least-visited retained candidates. Every fourth restart bypasses the pool for
   the feasible-route archive, or a uniform start if that archive is empty.
5. Initial sigma **0.15 in [-1,1] coordinates** (7.5% of full range), population **10**,
   restart patience **five stagnant generations**. CMA adapts sigma and covariance.
6. A common final audit selector for all methods: cover routes first, then geometric
   alternatives, capped at eight candidates. Dense outcomes never feed back into search.

These settings were evaluated as a package. The experiment does not identify how much
of the gain comes from each component. Sigma, pool size and patience are empirical
settings, not theoretically optimal values.

The development sweep used seeds 10–12: 45 runs across five methods. Naive succeeded
1/9 times; the earlier guided, selected and smaller-step variants each succeeded 7/9.
The selected policy retained the most within-run audited routes (12 versus 8 and 9),
which broke the tie. The smaller-step variant was not promoted. Development results
were not combined with confirmation statistics. See
[`development comparison`](../../runs/discovery_improvement_dev_20260911_v1/comparison.json).

## Optical scope and fair comparison

All controls use exactly the same new polygon environment: 3/4/5 corners, circular
faces, fixed n=1.60, the documented geometry/curvature cage, and a broad receiver at
x=120 mm, y in [-30,30] mm. Each field illuminates the full finite aperture.

A pass requires at least two TIR events on delivered qualifying branches, at least
0.5 degree critical-angle margin at every TIR, at least 25% folded throughput and
60% distinct contributing source rays at every field, plus energy-accounting gates.
A root must deliver at least 10% of its input energy to count. Different source rays
may follow different qualifying routes. These conditions do not impose a minimum
separation between fold points or a subjective visual-interest score.

The naive control ports flat infeasible ranking, clipped bounds and immediate
no-archive-progress restarts into this environment. It is **not** an exact rerun of
historical four-corner tasks with different objectives and acceptance gates. The
comparison establishes the search-policy gain within the new environment; it does
not isolate the environment change itself.

Each run used all 400 search calls: 10,800 source rays. Final audits used 125 rays
(25 aperture samples × five angular fields) and 1,024 geometry samples. Eight audits
reserved 1,000 rays, yielding the same 11,800-ray maximum allowance for every run.
Unused audit allowance was not reinvested. Invalid proposals cost runtime/proposal
slots, but no fictional physical traces.

| Method | Search rays | Actual final-audit rays | Reserved total | Worker-seconds |
|---|---:|---:|---:|---:|
| Naive | 777,600 | 9,750 | 849,600 | 1,969 |
| Earlier guided | 777,600 | 18,875 | 849,600 | 2,108 |
| Selected | 777,600 | 28,500 | 849,600 | 2,364 |
| Random | 777,600 | 1,375 | 849,600 | 1,833 |

The selected method used about 20% more worker time than naive and 2.4% more actual
rays including audits, within the equal reservation. It is a yield improvement under
an equal allowed physical budget, not an equal-wall-time speed claim. The full cohort
took about 700 seconds with twelve CPU workers and one BLAS/OpenMP thread each; no GPU
or LLM was used.

## Replay and higher-density checks

All 288 saved candidate logs reconcile with their declared ray budgets. All **228**
stored audits for the selected method replayed **exactly**, including the two failed
audits. All 21 recorded implementation-source hashes matched at verification.

The best audited example in each corner/aperture combination was additionally checked
with **101 aperture positions × nine fields = 909 rays** and 4,096 geometry samples.
The initial check passed only 2/6: the tracer's default 20,000-branch **bundle-wide**
cap exhausted numerical capacity on larger bundles. Those initial failures are saved,
not erased. A subsequent diagnostic increased only that cap to 200,000 and 400,000.
All six passed at both capacities, the results matched exactly, and branch-limited
energy was zero. Optical acceptance thresholds and the frozen benchmark were unchanged.
Use the capacity diagnostic for these larger bundles; blindly increasing source-ray
count with a fixed total branch cap can produce capacity-induced rejection.

| Aperture | Corners | 909-ray worst-field folded throughput | Root support |
|---|---:|---:|---:|
| 4 mm | 3 | 68.22% | 76.24% |
| 4 mm | 4 | 92.41% | 100% |
| 4 mm | 5 | 89.59% | 100% |
| 6 mm | 3 | 89.30% | 100% |
| 6 mm | 4 | 89.83% | 100% |
| 6 mm | 5 | 89.63% | 100% |

Replay plus the first stress check consumed 33,954 additional source rays; the
capacity diagnostic consumed 10,908; rendering retraced 750. These post-run costs are
separate from benchmark budgets. The six examples were selected after the experiment
and are not an unbiased estimate of high-density population success. All checks use
the same analytic ray engine. Independent-engine agreement, fabrication tolerances,
3D behavior, imaging quality and compaction remain untested.

![Selected folded designs](../../runs/discovery_improvement_confirm_20260911_v1/folded_examples.png)

## Use and artifacts

Run the selected policy, with fresh output directory, without comparison arms:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.search \
  --out runs/discovery_NEW --seeds 200 201 202 --corners 3 4 5 \
  --briefs base wide --search-calls 400 --workers 6
```

This calls the same benchmark search/audit implementation. The original pilot CLI is
preserved for reproduction. [Package documentation](../../nslens/discovery/README.md)
contains full confirmation, replay and plotting commands.

The frozen experiment directory is
[`runs/discovery_improvement_confirm_20260911_v1`](../../runs/discovery_improvement_confirm_20260911_v1).
It contains the predeclared protocol, source snapshots, environment metadata, every
candidate trace, per-run archives, all audit outcomes and statistical summaries.
Useful artifacts:

- [`comparison.json`](../../runs/discovery_improvement_confirm_20260911_v1/comparison.json)
- [`replay_and_stress.json`](../../runs/discovery_improvement_confirm_20260911_v1/replay_and_stress.json)
- [`stress_capacity_check.json`](../../runs/discovery_improvement_confirm_20260911_v1/stress_capacity_check.json)
- [`example_verification.json`](../../runs/discovery_improvement_confirm_20260911_v1/example_verification.json): replayable genes, codec, brief, vertices and sags for the six selected starts.

Validation: seven geometry/bundle/reflection tests and nine portfolio/budget/statistical
tests pass. Next work can consume these verified starts for numerical and new-Qwen-guided
compaction/refinement. No model training is required to obtain this discovery gain.
