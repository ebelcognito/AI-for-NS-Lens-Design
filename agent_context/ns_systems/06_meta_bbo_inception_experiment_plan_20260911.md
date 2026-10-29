# Meta-BBO for inception after the CMA improvement

Assessment and proposed experiments, 2026-09-11. No new optimizer was implemented,
trained, or benchmarked for this assessment. The scope is polygon starting-design
discovery; spherical inception and downstream compaction require separate tests.

## Decision

Test whether learned retention and search configuration improve the **current
portfolio CMA**, starting with LQD and ConfigX mechanisms independently. Prepare
MetaDistill as a training comparison once trajectory provenance is adequate. EPOM
and L2E remain independent population-optimizer candidates, conditional on obtaining
and validating implementations. These are five research directions, not five
interchangeable CMA emitters, and their optical performance is presently unknown.

The new baseline already implements several ideas motivating the earlier review.
Adding retention, informative failure ranking, or patience again would not establish
a learned-method contribution. A useful result must beat those existing mechanisms.

## What the recent runs establish

| Task suite | Portfolio dense-audited success | Interpretation |
|---|---:|---|
| Broad detector, 12 seeds × six environments | 41/72 (56.9%) versus naive 12/72 | Confirmed search-policy gain under this protocol |
| Detector height 1.5× beam, six seeds × six environments | 1/36 (2.8%) versus controls 0/36 | Sparse success; no established gain |
| Detector height 2.5× beam, same six seeds/environments | 3/36 (8.3%) versus controls 0/36 | Sparse success; no established gain |

Sources: [broad-detector confirmation](discovery_improvement_20260911.md),
[1.5× experiment](small_detector_20260911.md),
[2.5× experiment](detector_2p5x_20260911.md).
The last two suites reuse seeds and are not independent cohorts. These are routing
results, not imaging quality, global topology novelty, or compaction results.

The broad-detector result shows that useful optical feedback can improve search.
The narrow-detector results show substantial remaining difficulty, but do not yet
identify whether retention, operator choice, budget, or reachable geometry is limiting.

Current code anchors:

- `nslens/discovery/benchmark.py::run_job`: common physical budgets, CMA lifecycle,
  startup sampling, coarse evaluation and final audit selection.
- `nslens/discovery/portfolio.py::RestartPool`: twelve retained measured candidates,
  quality-shortlisted geometric diversity, least-visited restart parents.
- `constraint_rank`: feasibility-first lexicographic violation ordering.

The pool currently influences **restart locations**, not every offspring. Thus an
LQD-style survivor rule has limited opportunity to help while an emitter persists.
Record the number of actual selection decisions and descendant outcomes; otherwise a
null retention result could reflect infrequent use rather than poor learned selection.

## Assessment of the five methods

### LQD — first retention experiment

LQD learns population competition from fitness and behavioral descriptors. Its official
repository supplies learning/evaluation code and specifies Python 3.13+ with JAX;
evaluation of a learned run takes a run path. A ready optical checkpoint has not been
verified. [Official implementation](https://github.com/maxencefaldor/learned-qd).

Hypothesis: route-aware learned survival preserves productive stepping stones that the
current geometric shortlist discards. First replace only restart-pool retention,
keeping capacity, parent visitation, CMA and final certification fixed. Compare with
route-balanced and dominated-novelty retention under identical reproduction. Use
physical sequence descriptors, never arithmetic distance between route hashes.

Call this **LQD-inspired retention with CMA**. A faithful whole-population LQD run
changes reproduction too and belongs in a separate comparison. If simple route-aware
retention matches the learned selector, retain the simpler method. If pool decisions
are too rare, test a separately labeled active-population variant rather than silently
changing CMA reproduction in the retention ablation.

### ConfigX — first configuration experiment

ConfigX provides a learned controller for modular evolutionary algorithms; its
repository includes policy/environment code and saved output assets. Its published
interface is not an established optical CMA controller.
[Official implementation](https://github.com/MetaEvo/ConfigX),
[paper](https://arxiv.org/html/2412.07507).

Hypothesis: configuration should vary with remaining budget and measured failures.
Define a small optical action set: continue CMA, restart from retained candidates,
restart broadly, or select a declared initial scale. Keep the current safety/numerical
restart behavior. Compare frozen settings, a simple adaptive controller, and a learned
controller with exactly the same actions and observations. Log action effects and
costs. Label the custom interface **ConfigX-inspired optical control** unless using
the actual published algorithm and documenting every necessary adaptation.

This could help narrow receivers if one policy wastes budget on unproductive basins.
It could fail if the actions cannot generate useful routes or coarse observations do
not distinguish good decisions. Do not train a Transformer before establishing that
the action set offers useful choices.

### MetaDistill — training comparison, not a fifth emitter

The official release provides LES/LDE/LGA/POM baseline and distilled checkpoints,
teacher-trajectory scripts and variants with/without SSFT; assets require Git LFS.
`md_j0` omits SSFT. [Repository](https://github.com/Roki-Xing/MetaDistill),
[variant definitions](https://github.com/Roki-Xing/MetaDistill/blob/main/docs/MODEL_VARIANTS.md).

Hypothesis: distilling diverse productive teachers improves a fixed student more
than one-teacher imitation. Compare the **same student architecture** trained from
portfolio CMA alone versus a matched amount of mixed-teacher data, then evaluate
without online updates first. Include an untrained/released checkpoint control where
compatible. SSFT is a separate later arm with every extra evaluation charged.

Improved CMA makes teacher data more promising, but candidate JSONL files are not
automatically complete optimizer trajectories. Add generation/lifecycle IDs, selected
parent IDs, retention decisions, full CMA state and state transitions, actual actions,
negative siblings and descendant outcomes before claiming distillation readiness.
Split specifications and ancestry before teacher selection. A null result may indicate
teacher/student mismatch or redundant teachers, not a general failure of distillation.

### EPOM — independent population-proposal comparison

EPOM learns population mutation and crossover using attention-based modeling. The
paper evaluates BBOB at 30/100 dimensions; training uses population 100 and up to
100 generations. An official runnable checkpoint was not established in this review.
[Primary paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2fd37205c429a30cc1bb0afdbbf96857-Paper-Conference.pdf).

Hypothesis: population interactions produce useful coordinated geometry changes.
Compare an authenticated implementation with portfolio CMA and an ordinary DE control,
sharing bounds, evaluator and budgets. Optical genes have 14/17/20 dimensions, and
400 physical calls allow few population-100 generations. Test compatibility and
budget sensitivity before interpreting transfer failure. Population-size adaptation
must be reported and tested, not described as unchanged pretrained deployment.

### L2E — independent bounded-update comparison

L2E here means Gao et al., arXiv:2512.11453v2. Its neural unrolling uses averaged
updates and proxy guidance during training; Algorithm 1 specifies evaluation-only
inference. Nondifferentiable tracing therefore does not by itself rule out deployment.
Its stability results do not guarantee feasibility through optical route changes.
An official runnable checkpoint was not established in this review.
[Primary paper](https://arxiv.org/html/2512.11453v2).

Hypothesis: learned bounded proposals preserve useful basins while improving geometry.
Compare with a nonlearned bounded-step control and portfolio CMA. Charge all objective
evaluations used by proposal selection/fusion. A small step can preserve a basin but
also prevent route discovery; report route changes and verified successes together.
Check code, weights, training proxy and inference equations before integration.

## Proposed experimental sequence

### 0. Diagnose existing logs and prepare interfaces

Use completed runs only as development evidence. Measure failure combinations,
time-to-first feasible design, pool-parent usage, lifecycle lengths and success
concentration by corner family/receiver. Report missing telemetry explicitly.
For causal selector comparisons, saved trajectories alone are insufficient: changed
parents cause changed future evaluations, requiring fresh online runs.

Add independent hooks for pool retention, restart configuration and population
proposal; preserve the evaluator and common final selector. Store source revisions,
checkpoint hashes, seed streams and complete action/state provenance. Include
numerical/ranking-tie tests and verify that a disabled hook reproduces portfolio CMA.
Use an isolated dependency environment for upstream models.

### 1. Small mechanism pilot before substantial optical training

Five arms: current portfolio, route-balanced retention, dominated-novelty retention,
a simple adaptive configuration controller, and matched nonadaptive parent-centered
Gaussian sampling. These are **nonlearned controls**, not implementations of the five
papers. Change one mechanism at a time; keep startup, final audits and optical gates
fixed. The sampler tests whether CMA adaptation adds value beyond parent selection.

Proposed development size: four fresh seeds × three corner families × two beam widths
× three receiver settings (broad, 1.5×, 2.5×) × five arms = **360 runs**.
Check seed IDs against all existing runs before assigning them. Reusing a seed across
arms/settings is intentional pairing, not independent replication.

Each run retains the existing ceiling of 400 physical search calls, 1,600 proposals,
and eight 125-ray final audits. At 27 rays/search call this is 11,800 reserved source
rays/run, or **4,248,000 reserved rays** for the pilot. Unused audit allowance cannot
be spent on search. Actual ray usage, proposals, worker time and model latency are
reported separately. Prior broad-suite runtime is only a rough planning reference;
first time a small batch on the harder receivers.

If all hard-task arms are near zero, run a declared budget diagnostic on development
tasks at 400 versus 1,600 calls. Do not loosen optical gates to manufacture training
success. Distinguish inadequate budget from limited geometric reachability; failure
of search alone does not prove infeasibility.

### 2. Learned comparisons

Prioritize LQD retention and ConfigX control according to the mechanism pilot, then
evaluate both independently where runnable. Compare each learned component with its
strongest matched nonlearned control and portfolio CMA. Combine retention and control
only after individual effects are measured, using a 2×2 ablation.

For MetaDistill, collect prospective trajectories from useful distinct teachers and
run the same-student training comparison above. For EPOM/L2E, require an upstream
smoke test and a documented optics adapter before spending a full optical budget.
If weights/code remain unavailable, record **not evaluated**; do not substitute an
invented heuristic and assign the paper's name to it.

### 3. Freeze and confirm

Select at most one challenger on development results. Freeze its code, weights,
inputs, hyperparameters and objective mapping before fresh-seed confirmation.
Use at least twelve fresh seed blocks as a planning floor, then size the cohort from
development discordance/variance and a predeclared practical effect. Twelve is not
a power guarantee, especially at the current low hard-task success rates.

Primary endpoint: dense-audited success per run, equally averaged over the two tight
receiver settings and six corner/beam combinations. Report each setting separately;
broad receiver performance is a secondary regression check. Use paired seed-block
intervals, preserving all environments within a seed block. A fresh-seed result on
known specifications is seed generalization only. Reserve unseen receiver/field
specifications for a separately declared transfer test with no training ancestry.

Provisional practical target: **at least +10 percentage points** in hard-task success
against portfolio at the same allowance, with a paired 95% interval excluding zero.
This is a proposed engineering threshold, not a literature claim. If the interval
includes zero, report inconclusive; if its upper bound excludes the practical target,
report no evidence of a gain large enough for this target. Do not equate p>0.05 with
proof of equivalence. Exploratory five-way results do not constitute five independent
confirmatory wins; a single frozen challenger avoids that selection error.

Secondary endpoints: within-run verified routes, rays to first coarse success,
throughput/support, inference overhead and training cost. Never sum route hashes
across runs and call them globally novel topologies. For a novelty claim, freeze an
external canonical route reference and predeclare the novelty criterion.

### 4. Validate the inception-to-compaction benefit

Only after discovery improves, give a deterministic capped set of starts from each
arm to the same frozen numerical compactor, with identical downstream budgets.
Predeclare package objective and feasible constraints. Report all-task attainment,
including discovery failures, so easy-to-find but unusable starts cannot look better
through conditioning on successful cases. Keep extra dense capacity checks and
compaction costs explicit. Discovery gain alone does not establish better final optics.

## Interpretation

The strongest near-term hypothesis is that learned decisions can improve **which
measured candidates receive more search**. The current CMA already supplies useful
continuous adaptation. LQD and ConfigX isolate the two relevant decisions; MetaDistill
can teach a student from those successful trajectories. EPOM and L2E ask a broader
replacement question and require more integration evidence. Any of these may fail to
beat the improved baseline; the proposed controls make that outcome informative.
