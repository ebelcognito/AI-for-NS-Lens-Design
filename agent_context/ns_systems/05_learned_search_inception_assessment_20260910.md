# Learned search methods for optical inception — 2026-09-10

Assessment based on the current workspace, stored experiment artifacts, and primary
literature. This is a recommendation, not a new search experiment or an implemented
algorithm. The existing EvoTF plan remains a record of its original experiment scope.

The most useful direction is persistent, failure-aware search with route-aware survivor
and parent selection. LQD best matches the inception objective; ConfigX motivates a
controller for proposal modes and search persistence; MetaDistill supplies a possible
training recipe after useful optical trajectories exist. EPOM and L2E are credible
later emitter comparisons. None establishes that a released checkpoint will improve
this optical verifier or discover strongly novel routes.

## What the repository actually shows

### EvoTF was mostly prevented from using its updated state

In `nslens/qd/driver.py`, every complete population is evaluated and passed to `tell`,
then `(not added) or em.restart_needed` replaces the emitter. This is a restart per
unsuccessful **population**, not per candidate. With population size ten, a 6,000-candidate
run contains 600 population evaluations.

The released-EvoTF archives in
`results/fold_evotf_pilot_a_ablation_6000/raw/released_evotf/` report:

| Seed | Generations | Restart events | Numerical failure events | Candidates clipped |
|---|---:|---:|---:|---:|
| 300 | 600 | 594 | 0 | 80.33% |
| 301 | 600 | 596 | 0 | 71.60% |
| 302 | 600 | 597 | 0 | 76.18% |

All recorded restart reasons are `no_archive_addition`. Thus these results do not
demonstrate numerical instability of EvoTF. The driver usually discards the updated
distribution before another population can use it. CMA's covariance and evolution paths
are also discarded under this rule. This is a mechanism diagnosis, not proof that
longer lifecycles will improve either optimizer.

The [Pilot A report](../../results/fold_evotf_pilot_a_ablation_6000/REPORT.md)
records over 99% invalid candidates, no improvement over the common seed archive's
best fitness, and median filled cells of 8, 8, 7, and 8 for CMA, released EvoTF,
trained numeric EvoTF, and Optical-EvoTF respectively. Budgets were equal in candidates;
engine-call counts differed because evaluations exit at different gates.

`results/fold_evotf_pilot_a_dataset/manifest.json` makes the supervision problem precise:

- 20,000 candidates, 2,000 complete generations, 1,971 emitter lifecycles;
- 19,871 invalid candidates;
- 123 usable hindsight targets, only 29 of which have `new` or `improved` status;
- no retained held-route test split after leakage removal, and only one environment.

There is also information loss before the update: `driver.py` gives all rejected
candidates the same `-1e9` sorting key. `stable_rank_fitness` in
`nslens/search/evotf_emitter.py` turns the resulting complete order into distinct
percentiles. An all-rejected population therefore acquires an arbitrary ranking despite
having no measured fitness distinctions. Merely allowing more generations would allow
more updates driven by this uninformative order.

The invalid label is broader than impossible optics. The corpus includes 2,923
`descriptor_out_of_bounds` rejections, alongside beam, exit, geometry, and no-light
failures. Audit these separately before choosing a repair policy. Archive eligibility,
optical feasibility, and numerical failure should remain distinguishable.

### CMA's small advantage depends on the benchmark

The newer fixed-environment results support concern about discovery breadth. In
`runs/fixed_baselines_20260903_v1/comparison_engine_calls_v1.json`, eight environments
with one seed each and 2,000 candidates per algorithm give mean occupied cells of
3.125 for CMA versus 3.5 for random. CMA has a higher valid fraction (33.99% versus
10.26%) and slightly higher final fitness. Those are different achievements.

The stricter `runs/fixed_call_budget_benchmark_20260903_v1/manifest.json` contains two
environments, three seeds each, and a 5,000-engine-call cap:

| Environment | CMA cells, seeds 1/2/3 | Random cells | Learned cells |
|---|---|---|---|
| fixed-fold-02 | 1 / 1 / 1 | 2 / 2 / 2 | 2 / 2 / 2 |
| fixed-fold-07 | 3 / 3 / 3 | 3 / 4 / 4 | 3 / 3 / 3 |

CMA beats random on best fitness in all three first-environment seeds, but only one
second-environment seed. The learned baseline is a global/pairwise proposal scorer,
not EvoTF. This small experiment does not establish a universal algorithm ordering.
Also, CMA starts around the specified initial geometry whereas random samples bounds;
separate warm-start and cold-start comparisons are needed to isolate adaptation.

The older discovery-control reports (`agent_context/ns_rl_research/15_*` and `16_*`)
report larger CMA-ME gains under their own geometry/objective protocols. Do not combine
those results with the hardened fixed-environment runs as if they were one benchmark.

### Current spherical inception is a separate, nonadaptive proposal loop

`ns_probe/run_novel_compaction.py::_proposal_batch` uses 8% uniform samples and otherwise
perturbs fixed audited anchors with a log-uniform scale from 0.012 to 0.18 of gene
ranges. For the 17-parameter two-element representation, first-body steps receive a
0.30 multiplier. `_run_discovery` records outcomes but does not use them to update the
anchor pool or proposal distribution during that run.

The [September 4 report](04_spherical_inception_compaction_20260904.md) records zero
strong-novelty promotions across 50,000 proposals in v6/v8. The weaker-threshold v7
design is a boundary demonstrator. Replacing an emitter in the older nine-gene driver
does not automatically change this inception sampler or solve its proposal limitation.

## Assessment of the five methods

### LQD: best match for deciding what to retain and reproduce

LQD learns competition scores from population fitness and behavioral descriptors, then
selects survivors from parents plus offspring. It changes population survival rather
than predicting EvoTF's Gaussian updates. It is related work with a shared author,
not a drop-in EvoTF successor checkpoint. Its BBOB-to-robot transfer supports testing
this mechanism, although the reported aggregate robot advantage over GA was not
statistically significant. [Paper](https://arxiv.org/html/2502.02190)

Optical adaptation: preserve candidates that lead to distinct viable routes or useful
descendants, even when they lose within a coarse packing/fold-count cell. Maintain an
active breeding population separately from the immutable novelty reference and verified
result archive. A learned survivor pool with CMA reproduction is an LQD-inspired hybrid;
it should not be labeled an exact reproduction of LQD.

Use canonical sequence distances or meaningful route descriptors, not numerical values
of sequence hashes. Establish route-balanced selection and dominated-novelty selection
as nonlearned controls. Selection cannot rescue proposals that never reach any useful
region; combine this work with informative feasibility feedback.

### ConfigX: useful controller architecture, not a ready-made CMA restart policy

ConfigX configures modular evolutionary algorithms through a Transformer policy trained
over algorithm/task combinations. Its module catalogue centers on DE, PSO, and GA-style
operations. The attached assessment's CMA learning-rate/population/restart diagram is
an extrapolation, not a demonstrated interface for this repository's CMA.
[Paper](https://arxiv.org/html/2412.07507),
[official code](https://github.com/MetaEvo/ConfigX)

Here a controller could select parent families, local versus broad proposals, group step
scales, and continue/shrink/reseed actions. It would need an optical state/action schema
and optical training or validation. Start with fixed rules or a small adaptive allocator
to establish whether these decisions have measurable value before training a Transformer.

### MetaDistill: useful training recipe once the teacher data are useful

The official release includes trajectory distillation, optional SSFT, and evaluations
for LES, LDE, LGA, and POM on BBOB and control tasks. It includes checkpoints and scripts
for collecting/selecting teacher trajectories. It is a training framework, not one
replacement optimizer. Its ICML 2026 listing is independently present in the conference
downloads. [Official repository](https://github.com/Roki-Xing/MetaDistill),
[conference listing](https://icml.cc/Downloads/2026)

For optics, collect diverse successful search segments from CMA, existing anchor
mutations, alternative reproduction operators, and repair/local-search policies. Keep
the negative siblings and record full lineage. Train on verified future discovery
utility, with teacher imitation as initialization. The current corpus's 29 improving
hindsight targets are insufficient evidence for scaling a sophisticated distillation
pipeline. Any test-time self-fine-tuning must count its simulator calls and wall time.
The full MetaDistill paper was inaccessible through OpenReview during this assessment;
detailed SSFT loss and performance claims were not independently audited.

### EPOM: relevant population model, later replacement benchmark

EPOM uses cross-attention tokenization, deformable attention for strategies, and
differentiable crossover. It reports BBOB evaluation at 30/100 dimensions and a
BipedalWalker transfer experiment. These support a population-optimizer comparison,
not a demonstrated solution to sparse optical validity or route novelty.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2fd37205c429a30cc1bb0afdbbf96857-Paper-Conference.pdf)

An optical model could attend from proposal queries to candidate, route, failure, and
archive features. That is a custom adaptation, not automatically the released EPOM.
First establish a useful input signal and persistent populations. I did not verify a
public official checkpoint in this review; paper availability alone is not deployment
readiness.

### L2E: useful bounded-update design, not a fix for driver-triggered restarts

L2E uses a structured Mamba/attention operator with bounded, averaged updates and
stability-oriented regularization. Algorithm 1 distinguishes proxy-gradient training
from evaluation-only inference; it would be incorrect to dismiss deployment simply
because this optical tracer is nondifferentiable. Its UAV/control transfer is relevant
evidence, but its stability statements are conditional, not guarantees for optical
route switches. [Paper, v2](https://arxiv.org/html/2512.11453v2)

Bounded corrections could help preserve feasible optical basins. However, the observed
EvoTF restart mechanism is an external archive policy, which L2E's update stability
does not remove. Differentiable meta-training would also require a suitable training
task or proxy; it is not supplied by the current hard verifier.

## Recommended inception architecture

```text
fixed optical environment + frozen canonical novelty reference
                         |
route-diverse active population + diagnostic history
                         |
parent / proposal-mode / scale / persistence controller
                         |
local CMA | bounded anchor edits | broad exploration | later learned proposals
                         |
geometry checks -> optical trace -> typed outcomes and measured margins
                         |
active survivor selection + strict verified archive promotion
                         |
qualified distinct viable seeds -> fixed-budget compaction
```

Use two distinct decisions: what is worth exploring again, and what qualifies as a
reported discovery. A bounded working pool can retain diagnostic near-misses without
admitting them to the verified archive. Untraceable/degenerate geometries must not acquire
fabricated optical margins. Missing diagnostics need explicit masks, and failure-stage
order alone is not a physical distance to feasibility.

Reward verified novel viable families and, later, their success under a fixed compactor
budget. Measure topology novelty against the frozen reference, and diversity among new
results separately. Preserve strong novelty thresholds and dense field/pupil validation.
Increasing old archive coverage or initial scalar fitness is only a secondary outcome.

## Smallest useful experiment sequence

1. **Diagnose before retraining.** Replay the current telemetry summaries; log lifecycle
   length, populations with no informative ranking, valid-but-not-added populations,
   failure classes, per-gene clipping, proposal scales, and actual engine calls. Audit
   descriptor exclusion separately from optical rejection. CMA needs clipping telemetry
   comparable to EvoTF; its pilot summary currently reports null.
2. **Test feedback and persistence independently.** Compare the current rule with short
   patience (for example 5 and 10 unsuccessful generations, selected on development
   tasks). Cross this with unchanged ranks versus measured feasibility-aware feedback.
   Preserve numerical-failure restarts. Treat all-equal failures explicitly: do not
   invent evidence from sample order. A repair/scale action or declared no-information
   update needs a defined interface that still completes ask/tell bookkeeping. Test
   smaller initial scales and reflection separately because clipping is also substantial.
3. **Measure adaptation against matched sampling.** Include uniform random and an
   archive/anchor-conditioned Gaussian sampler with the same initialization, scales,
   bounds, and restart-parent rules but no learned distribution update. This distinguishes
   the value of good anchors from the value of CMA or EvoTF adaptation. Report cold-start
   and warm-start tasks separately. Apply restart-policy comparisons to both optimizers.
4. **Add active population feedback to inception.** Retain the existing static-anchor
   sampler as a control. Compare route-balanced and dominated-novelty survivor selection,
   using identical reproduction and evaluation. Then compare learned LQD-style scores
   under the same active-population capacity and descriptor representation. Also test
   parent selection alone so benefits are not conflated with archive replacement.
5. **Train only after the signal improves.** Record complete multi-generation optical
   trajectories and productive descendant outcomes. Hold out environments, route families,
   and anchor lineages; reject splits with no usable validation/test targets. Test a
   MetaDistill-style initialization and optical outcome training against numeric-only
   controls. EPOM/L2E emitter replacements follow as independent comparisons.

Use one declared representation and verifier per comparison; the nine-gene EvoTF pilot,
fixed-environment search, and 17-parameter spherical inception are separate task suites.
Cap actual engine calls, report candidates and wall time, and account for verification,
repair, compaction, and online learning. Use paired task/seed runs and report the
distribution of outcomes. The existing two-environment, three-seed benchmark is a pilot,
not sufficient evidence of generalization.

The decision to proceed should depend on more qualified novel viable families per call,
or more reliable downstream compaction at the same total budget. Longer lifecycles,
lower offline loss, more valid samples, and attention visualizations are diagnostics,
not substitutes for that result.
