# Small-detector paired experiment — 2026-09-11

## Completed result

All 144 runs completed. Portfolio succeeded in 1/36 runs (2.8%); naive,
legacy_route and pilot each succeeded in 0/36. Gain against each control:
+2.8 percentage points, 95% seed-block interval [0,+8.3], one-sided seed sign
p=0.5. **No statistically established improvement on this environment.**

The sole passing design is `base_k3_seed305_portfolio`: three corners, 4 mm beam,
6 mm detector, 68.0% worst-field folded throughput in the 125-ray audit. It also
passed the 909-ray, nine-field stress check without increasing branch capacity.
All 144 budget logs matched; the sole audit replayed exactly; frozen sources matched.
18 targeted tests passed. Each method used exactly 388,800 search source rays.

Artifacts: [results](../../runs/small_detector_20260911_v1/RESULTS.md),
[environment](../../runs/small_detector_20260911_v1/environment.png),
[comparison](../../runs/small_detector_20260911_v1/comparison.png),
[folded example](../../runs/small_detector_20260911_v1/folded_examples.png),
[all-arm replay](../../runs/small_detector_20260911_v1/all_arm_replay.json).

## Frozen environment and protocol

The user approved detector full height = 1.5 × input beam width, with its centre
one-third of the fixed setup height upward from the bottom. There was no existing
physical setup-height parameter. This experiment explicitly defines the reference
envelope as y=[-55,50] mm, giving detector centre y=-20 mm. The source remains
at (55,50) mm and the detector plane remains x=120 mm.

The bottom is below the conservative allowed curved-body bound: minimum vertex
y=-40 mm, with sag bounded by 0.268*sqrt(30²+30²)=11.37 mm. The envelope is a
fixed reference, not recomputed from each candidate and not an additional geometry gate.

| Input beam width | Full detector height | Detector y interval |
|---|---|---|
| 4 mm | 6 mm | [-23,-17] mm |
| 6 mm | 9 mm | [-24.5,-15.5] mm |

Fixed material n=1.60; 3/4/5 circular-face corners. Search fields -0.5,0,+0.5 degrees,
9 rays/field. Every field must deliver ≥25% qualifying folded throughput and ≥60%
distinct root support, with ≥2 TIR events and ≥0.5° margin per TIR. Existing energy
accounting gates and geometry bounds remain in effect.

Six fresh seeds 300–305 × two beams × three corner families × four arms = 144 runs.
Each run allows 400 physical search calls (10,800 source rays), at most 1,600 proposals,
and eight reserved 125-ray audits (five fields ×25 rays), total allowance 11,800 rays.
Invalid geometry consumes proposals, not fabricated ray calls. Unused audit allowance
is not transferred into search. Dense audit outcomes never affect search.

## Baselines

- `naive`: existing confirmation control; flat failed ranks, clip bounds, restart
  after one generation without route-archive progress.
- `legacy_route`: adapted from `nslens/search/cma_me.py::_step_emitter`:
  new route > improved route (rank by improvement) > unchanged feasible > failure;
  clip bounds and immediate no-progress restart. Uses the common normalized polygon
  codec, route niches, one emitter, sigma 0.15 and population 10. This is **not an
  exact replay of the historical multi-emitter, two-axis grid driver**.
- `pilot`: earlier guided ranking, reflected bounds, five stagnant generations.
- `portfolio`: retained-candidate method, complete sorted violation ranking,
  40 Sobol proposals, pool of 12 and five stagnant generations.

All four share the same CMA core, environment, initial sigma, population, search/audit
budgets and final audit selection. This compares policies on the new environment;
it does not isolate individual changes or compare imaging quality.

Primary endpoint: fraction of runs with ≥1 dense-audited pass. Primary control:
`naive`, matching the existing frozen analysis. Comparisons against `legacy_route`
and `pilot` are explicitly exploratory. Confidence intervals resample six seed blocks,
averaging six environments within each block (20,000 bootstrap samples). Also report
seed sign tests, verified route counts, throughput and first coarse-success ray usage.
Six seeds give limited precision. Routes are within-run signatures, not global topologies.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.benchmark \
  --out runs/small_detector_NEW --stage confirmation --candidate portfolio \
  --arms naive legacy_route pilot portfolio --seeds 300 301 302 303 304 305 \
  --corners 3 4 5 --briefs base wide --search-calls 400 --workers 6 \
  --detector-beam-ratio 1.5 --setup-y-bounds -55 50 \
  --detector-height-fraction 0.3333333333333333
```

Results directory: `runs/small_detector_20260911_v1`. Protocol and source snapshot
are written before running jobs. Per-run candidate logs contain exact genes, vertices,
sags and measured routes. This experiment does not add CMA parent/state lineage logging.
