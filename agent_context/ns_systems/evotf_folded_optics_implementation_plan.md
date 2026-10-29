# EvoTF with optical embeddings for folded-optics search

## Status and corrected scope

This is an implementation plan, not an experiment report.

The goal is a small, direct modification of the repository's existing folded-optics
search:

> Put the actual Evolution Transformer into the current CMA-ME emitter position, then
> test whether adding folded-optics embeddings and training on related search histories
> improves archive discovery at the same ray-trace budget.

This is **not** a new compaction objective, a 12-gene freeform project, an LLM agent, or
a learned residual wrapped around full-covariance CMA. Compaction and proposer training
remain separate questions.

The initial target is the current normalized nine-gene folded-element environment:

```text
ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n
```

Its existing CMA-ME archive rule and optical verifier remain unchanged. The experiment
changes only the emitter/update policy.

## What “use EvoTF itself” means

There are two required EvoTF arms.

### Arm 1: released EvoTF, unchanged

Use the official `evosax.algorithms.EvoTF_ES` implementation and released checkpoint,
through a thin adapter implementing the repository's `ask()`/`tell()` interface. Do not
reimplement it in PyTorch or silently alter its attention blocks.

This arm answers:

> Is the released general-purpose EvoTF already a better emitter than the current CMA
> emitter in the folded-optics landscape?

It receives only the inputs supported by the released model: candidate solutions,
fitness/rank information, and its diagonal search-distribution history. It therefore
cannot receive new optical embeddings without changing its architecture.

### Arm 2: EvoTF plus optical context adapters

Use the same EvoTF architecture and update heads, with narrow additive input adapters
for optical metadata. Preserve the upstream implementation, source provenance, and
checkpoint-loading path. If the upstream public API cannot expose the required internal
features, vendor the minimum EvoTF model code with its upstream commit, license, and an
explicit patch rather than writing a vaguely similar transformer.

Call this model **Optical-EvoTF**. It is EvoTF with added conditioning, not an unrelated
“EvoTF-inspired” network.

Initialize every new context projection so its contribution is exactly zero. Before
training, Optical-EvoTF must reproduce the released checkpoint's update for identical
numeric inputs within floating-point tolerance. This separates the effect of embeddings
from accidental architectural drift.

This arm answers:

> Can parameter meaning, environment, route, and failure context help EvoTF reuse
> folded-optics search structure across runs?

## What remains unchanged

Keep the following identical across CMA and EvoTF arms:

- the nine-gene bounds and relative-sag convention;
- normalized search coordinates;
- population size and evaluation budget;
- initial means, archive seeds, and random-seed schedule;
- geometry validity and all hardened optical gates;
- field/pupil sampling;
- archive axes and bounds;
- exact candidate ranking rule: `new > improved > none > invalid`, with the existing
  within-tier value; and
- restart-from-archive behavior.

Every candidate is still traced. The learned model never replaces a geometry or optical
gate.

## EvoTF integration into CMA-ME

### Common emitter interface

Define a small protocol compatible with the existing driver:

```text
ask() -> population[N, D]
tell(order, observations) -> None
restart_needed -> bool
state_summary() -> serializable mapping
```

The existing `CMAES` satisfies the essential `ask()`/`tell(order)` behavior. Add:

```text
nslens/search/evotf_emitter.py
```

with an `EvoTFEmitter` that:

1. owns the official JAX/evosax strategy and state;
2. accepts the same initial mean, bounds, population size, and seed;
3. converts JAX arrays to NumPy only at the verifier boundary;
4. applies the same declared bound handling as the comparison arm;
5. converts the repository's candidate order to deterministic rank fitness for EvoTF;
6. calls the official EvoTF `tell()`; and
7. reports inference time, clipping, non-finite state, and fallback/restart events.

The ranking conversion must preserve exactly the order used by CMA. Do not feed archive
keys such as `2e12 + fitness` directly into the network; their magnitude is an internal
sorting device, not a physically meaningful fitness scale. Use rank percentiles with a
stable tie convention.

### Diagonal versus full covariance

The released EvoTF is a diagonal-Gaussian evolution strategy. The repository's CMA
emitter learns full covariance. This is a real algorithmic difference, not an
implementation detail.

For this experiment, EvoTF replaces a CMA emitter as a complete emitter. Do not retain
CMA's covariance behind it and then call the result unmodified EvoTF. Compare:

```text
CMA-ME with full-covariance CMA emitters
CMA-ME with released EvoTF emitters
CMA-ME with trained numeric-only EvoTF emitters
CMA-ME with Optical-EvoTF emitters
```

If diagonal search is the limiting factor, that will appear in the result. A later
full-covariance EvoTF extension would be a separate experiment.

### Bounds and restarts

The current CMA implementation clips candidates to bounds. Use the same behavior for the
first paired comparison and record a per-gene clipping mask. Later, reflection or bounded
distribution parameterizations can be tested as their own ablation.

Keep the driver's existing restart rule: restart an emitter from a random archive elite
when a population adds nothing or the emitter reports numerical failure. Optical-EvoTF
does not get a learned restart head initially. This keeps the only changing variable the
distribution update.

## Optical embeddings

Embeddings are part of the first trained Optical-EvoTF, not a deferred feature. They
augment raw numerical inputs rather than replacing them.

### 1. Parameter-role embeddings

Each EvoTF dimension token receives a learned parameter-role vector:

| Gene | Kind | Physical group |
|---|---|---|
| `ray_y` | source position | source |
| `ang` | source angle | source |
| `cx` | element position | pose/package |
| `thick` | element thickness | pose/package |
| `fsag` | relative face sag | surface shape |
| `bsag` | relative face sag | surface shape |
| `tilt_a` | face tilt | surface pose |
| `tilt_b` | face tilt | surface pose |
| `n` | refractive index | material |

The token combines:

```text
original EvoTF numerical dimension features
+ parameter-name embedding
+ parameter-kind embedding
+ physical-group embedding
+ units/normalization embedding
```

The raw normalized value and distribution statistics remain present. Invariance becomes
metadata-conditioned: if dimensions are permuted, their values, distribution state,
metadata, and outputs must move together.

Do not add a learned basin-ID embedding. It would encourage memorization and has no value
for a new basin.

### 2. Environment embeddings

Encode the fixed context that changes the meaning of a search:

- field angles and pupil sampling;
- sensor position and focus window;
- beam width and ray count;
- sag-ratio and clearance thresholds;
- minimum critical-angle margin;
- field aggregation rule;
- archive axes/bounds; and
- objective/verifier fingerprint.

Continuous values go through a small MLP after explicit scaling. Categorical items use
small lookup embeddings. Pool them into one environment vector and inject it into each
generation/dimension token through an additive projection or FiLM-style scale and bias.

For the initial nine-gene environment, `ray_y`, `ang`, and `n` are search dimensions.
When the later six-gene `FixedFoldProblem` is tested, those quantities move from gene
tokens into the environment encoder. The representation ID keeps the two conventions
distinct.

### 3. Candidate outcome and failure embeddings

After a population is evaluated, each candidate receives:

- archive status: `new`, `improved`, `none`, or `invalid`;
- typed invalidity/failure reason;
- clipping mask;
- within-population fitness rank;
- normalized archive-improvement value when defined;
- per-field throughput/quality summaries; and
- constraint margins.

Use a categorical embedding for status and failure reason, plus an MLP for continuous
metrics. These features are known before the next update and cost no additional ray
traces.

Failures must remain in the population. They are part of what the optical conditioning
is intended to learn.

### 4. Route embeddings

Encode the realized dominant interaction sequence for each valid candidate. One ordered
route event contains:

- event kind: transmission/refraction, TIR, Fresnel reflection, miss, or terminal;
- physical face: `A`, `B`, `T`, or `D`;
- repeat/contact index;
- field support or persistence;
- incidence/turning-angle features when available; and
- critical-angle and energy features.

Apply a small ordered route encoder and pool its output into the corresponding candidate
feature. Route order receives positional information; candidate order within a population
does not.

Begin with the dominant sequence plus `per_field`, route support, and worst critical
margin already available from the hardened evaluator. A full pupil-ray event encoder is
unnecessary for the first test.

### 5. Archive-context embeddings

The update may also see compact archive state:

- current coverage and filled-cell count;
- candidate's proposed behavior cell;
- whether the cell was empty;
- previous elite fitness in that cell; and
- parent niche/route family.

This matters because the emitter is optimized for CMA-ME archive improvement, not merely
single-objective fitness. Keep this context small and numerical; archive retrieval and
attention over all elites are outside the first implementation.

## Where embeddings enter EvoTF

Do not flatten all optical context into extra fake search dimensions. Search dimensions
must remain quantities the optimizer can change.

Add context at three existing semantic levels:

```text
dimension stream:
    original numeric feature projection + parameter-role embedding

solution/population stream:
    original solution/fitness projection + status/failure/route projection

generation stream:
    original distribution/history projection + environment/archive projection
```

The original attention blocks and mean/scale heads remain. New projections are zero
initialized for the checkpoint-equivalence test, then trained.

Required invariance tests:

- permuting population members does not change the update;
- permuting gene values and all corresponding metadata produces the same permuted update;
- permuting route events does change the route embedding;
- masking an absent field/event is identical to omitting it; and
- zero context adapters reproduce the official EvoTF result.

## Data required

### Existing traces

The current normalized nine-gene traces already retain same-parent candidate populations,
parent values, sigma, status, fitness, dominant signature, and several field metrics.
They can support:

- data-loader development;
- status/failure and route encoders;
- same-population ranking targets; and
- a first embedding overfit/sanity test.

They do not record the complete before/after CMA distribution, covariance/evolution
paths, exact environment/verifier version on every transition, or all structured failure
reasons. Therefore they are not sufficient for faithful CMA trajectory distillation.

### New transition logging

Add an opt-in `nslens.optimizer_transition.v1` record: one row per complete emitter
generation. It contains:

- task, representation, simulator, objective, and archive fingerprints;
- emitter lifecycle ID, generation, seed, and remaining budget;
- CMA state before `ask`: mean, sigma, covariance, evolution paths, and RNG/sample IDs;
- every candidate, clipping mask, rank, status, optical metrics, failure reason, and route;
- exact ranking order;
- CMA state after `tell`; and
- whether and why the emitter restarted.

This record supports two kinds of training:

1. **CMA distillation:** predict CMA's next mean and marginal diagonal scale. Because
   EvoTF is diagonal, it cannot reproduce CMA's cross-dimensional covariance.
2. **Optical hindsight/filtered training:** use later verified archive outcomes to favor
   updates from states that led to useful folded-optics discoveries, rather than merely
   reproducing every CMA step.

CMA distillation alone cannot establish an improvement over CMA. Its target is the
domain-independent CMA update, so parameter or route embeddings are logically unnecessary
for an exact imitation. It is useful only to initialize EvoTF safely and to provide a
numeric-only control. The improvement-bearing target must depend on verified optical
outcomes beyond the immediate teacher update.

Preserve full CMA state even though EvoTF uses only marginal targets. It keeps the source
record reusable for a later covariance-aware model and makes the teacher limitation
measurable.

### Data splits

Never split adjacent generations randomly. Split complete emitter lifecycles and prevent
exact parents/elites from crossing splits.

Report three levels:

1. held-out random seeds within known route families;
2. held-out complete realized route families/basins; and
3. held-out fixed folded environments when the six-gene environment suite is introduced.

The held-route and held-environment tests determine whether embeddings transfer rather
than memorize.

### Scaling schedule

Do not assume a huge dataset is required. Use a learning curve:

| Stage | Complete generations | Approx. candidates | Purpose |
|---|---:|---:|---|
| Smoke | 64 | 640 | schema, masks, tiny-batch overfit |
| Pilot A | 2,000 | 20,000 | detect embedding signal |
| Pilot B | 8,000 | 80,000 | first held-route online comparison |
| Scale | chosen from results | declared later | only if Pilot B improves search |

Count unique emitter states and route families, not only candidate rows. Ten siblings and
adjacent generations are highly correlated examples.

Generate new data under one hardened objective/verifier version. Historical traces with
different gates may be auxiliary pretraining data only, with an explicit source-domain
embedding and ablation.

## Training sequence

### Phase 0: official zero-shot folded-optics benchmark

Before training anything, run the unmodified released checkpoint in the actual folded
environment. Compare it with CMA-ME at equal calls. The existing synthetic toy result is
not an optical result.

This produces the real zero-shot baseline and validates the emitter adapter.

### Phase 1: numeric-only EvoTF distillation

Starting from the released checkpoint, train or fine-tune EvoTF on fresh folded-optics
CMA trajectories using only its original numerical inputs. Targets are CMA's next mean
and marginal scale plus the existing temporal context.

This controls for domain training. Without it, an embedding gain is confounded with the
fact that only the embedded model saw folded-optics data.

### Phase 2: Optical-EvoTF adapters

Load the same initial checkpoint and add zero-initialized context adapters. Train in two
steps:

1. freeze the EvoTF core and train only optical embeddings/projections;
2. optionally unfreeze the final EvoTF blocks and heads with a lower learning rate.

Use the same transition corpus, splits, steps, batch sizes, and model-selection metric as
the numeric-only arm.

First warm up the adapters with auxiliary optical predictions while preserving the
distilled update. Then train the update head with one of two simple verified targets:

1. **Hindsight update distillation:** from each state, look ahead a fixed, recorded number
   of generations and identify the feasible candidate with the strongest archive
   improvement. Train the next distribution to increase likelihood around its normalized
   displacement. Truncated episodes use an explicit mask.
2. **Filtered trajectory distillation:** for repeated searches from matched task/start
   distributions, retain or upweight emitter segments with stronger final verified
   archive contribution and distill their updates.

The first implementation should use hindsight update distillation because it can be
derived from complete CMA logs without an online policy-training system. Apply the same
hindsight target to numeric-only EvoTF and Optical-EvoTF; otherwise an apparent embedding
gain is confounded with different supervision.

Losses can include:

- next-mean distribution matching;
- next marginal log-scale matching;
- candidate rank prediction;
- validity/failure prediction;
- route-family/persistence prediction; and
- next-generation archive-improvement prediction.

Auxiliary losses shape the embeddings. Teacher matching is primary only during warm-up;
the hindsight/filtered objective is what can move the policy beyond imitation.

### Phase 3: online matched-budget evaluation

Evaluate frozen optimizers on held-out tasks:

```text
A. full-covariance CMA-ME
B. released EvoTF-ME, unchanged checkpoint
C. folded-optics-trained numeric-only EvoTF-ME
D. folded-optics-trained Optical-EvoTF-ME
```

Use paired initial archives, task instances, budgets, and seeds. JAX compilation is warmed
up before timed trials, but compilation time is also reported separately.

Primary metrics:

- archive coverage and QD score;
- best verified fitness;
- number of distinct route families/basins;
- deep-fold basin coverage;
- calls to first valid and first improving candidate;
- invalid and clipped-candidate rates;
- route persistence across fields;
- optimizer time and total wall time; and
- performance at early call checkpoints, not only final performance.

The unit of statistical comparison is a complete paired search task/seed, not an
individual candidate.

### Phase 4: embedding ablations

Run a cumulative ladder:

1. original numeric EvoTF only;
2. plus parameter-role embeddings;
3. plus environment embeddings;
4. plus status/failure embeddings;
5. plus route embeddings;
6. plus archive-context embeddings.

Compare learned categorical embeddings with one-hot features at a matched projection
size. This distinguishes useful semantics from a mere parameter-count increase.

Every arm in this ladder receives the same hindsight/filtered update supervision. A
comparison where only the embedded arm receives future optical labels does not isolate
the value of embeddings.

Only after this result should direct verified policy optimization, self-distillation, or
a covariance-aware EvoTF be considered.

## Repository implementation map

### Milestone 1: exact EvoTF emitter

Planned changes:

```text
pyproject.toml                              optional pinned EvoTF/JAX extra
nslens/search/evotf_emitter.py              official strategy adapter
nslens/qd/driver.py                         configurable emitter factory
nslens/analysis/run_fold_evotf_baseline.py  paired zero-shot experiment
tests/test_evotf_emitter.py                 ask/tell, ranks, bounds, determinism
```

The default emitter remains CMA. Importing the repository without the optional EvoTF
dependencies must continue to work.

Deliverable: the released checkpoint runs end-to-end in the folded environment and its
calls are counted identically to CMA.

### Milestone 2: trajectory contract and logging

Planned changes:

```text
nslens/rl/data/schema.py                         transition validator
nslens/rl/data/build_optimizer_trajectories.py  splits and manifests
nslens/search/cma_me.py                          lossless state snapshots
nslens/qd/driver.py                              full generation observer
tests/test_optimizer_trajectory.py               replay and leakage tests
```

Recorder-on and recorder-off CMA searches must produce identical candidate populations,
rankings, restarts, and archives.

Deliverable: smoke and Pilot A datasets with source hashes, route counts, state counts,
failure composition, and split audit.

### Milestone 3: Optical-EvoTF

Preserve the upstream JAX/Flax architecture in a focused package:

```text
nslens/learning/optical_evotf/
    upstream.py       pinned upstream model or imports
    metadata.py       nine-gene semantic registry
    embeddings.py     parameter/environment/status/route/archive adapters
    model.py          narrow composition with EvoTF
    data.py           padded batches and masks
    train.py          numeric-only and adapter training
    checkpoint.py     upstream and adapter provenance
```

Tests cover zero-adapter equivalence, population invariance, metadata-conditioned gene
equivariance, route order sensitivity, masks, finite gradients, tiny-batch overfit, and
checkpoint round trips.

Deliverable: numeric-only and Optical-EvoTF checkpoints selected without reading test
metrics.

### Milestone 4: matched online comparison

Planned changes:

```text
nslens/analysis/run_fold_evotf_ablation.py
nslens/analysis/summarize_fold_evotf_ablation.py
```

The result artifact includes raw paired runs, all configurations, environment and model
fingerprints, call accounting, wall time, and derived summaries.

Deliverable: CMA versus released EvoTF versus trained EvoTF versus Optical-EvoTF, including
the embedding ladder.

## Immediate coding slice

Implement only the zero-shot integration first:

1. pin and document the official EvoTF version/checkpoint;
2. implement `EvoTFEmitter` around the official strategy;
3. add an emitter-selection hook without changing the default CMA path;
4. translate archive ranking order into stable rank fitness;
5. add bounds, restart, determinism, and call-accounting tests; and
6. run a small paired folded-optics smoke benchmark.

Then implement trajectory logging and embeddings. This order proves that the actual
EvoTF works inside the environment before any architectural extension or data-generation
expense.

## Interpretation rules

- If released EvoTF loses to CMA, that is the zero-shot result, not evidence that
  embeddings cannot help.
- If trained numeric-only EvoTF improves but Optical-EvoTF does not, domain trajectories
  helped and the embeddings did not.
- If Optical-EvoTF beats numeric-only EvoTF on held routes at equal calls, the optical
  embeddings added transferable information.
- If both diagonal EvoTF variants fail on correlated basins, test covariance as a separate
  architectural limitation rather than hiding CMA covariance inside the EvoTF arm.
- Do not call archive interpolation new topology discovery. Novel routes must be verified
  and compared with the frozen reference set.
