# Hybrid proposer–compactor architecture for optical geometry search

## Status and scope

This is a planning document, not an experiment report. It describes a possible
learned system for **proposing and compacting** bounded optical geometries. It does
not claim that the system has been implemented, trained, or shown to improve over
CMA. No heavy experiments are implied by this plan.

The current emphasis is the proposer plus compaction loop. Freeform refinement is
an important later stage, but it requires a representation that gives the learner
control over freeform parameterisations, surface bases, departures, slopes, and
manufacturing constraints. It should not be silently folded into the first
proposer–compactor claim.

## Core proposal

The system should be a hybrid of a structured numerical learner and a verified
optical search loop:

```text
requirements and fixed environment
        -> route/geometry proposer
        -> non-sequential trace and validity checks
        -> compactor proposes bounded edits or search distributions
        -> trace and evaluate every candidate
        -> retain feasible compact candidates and their histories
        -> later: controlled freeform refinement
```

The deep-learning core is numerical. An optional language model may orchestrate
tools, select an operating mode, summarize diagnostics, or express a sequence of
typed edits, but it should not be expected to directly invent all continuous
geometry or replace the optical verifier. This is a no-free-lunch issue: proposal,
local numerical search, verification, and eventual freeform correction are
different capabilities.

The practical first representation is the current bounded circular-arc model. Its
12-gene geometry vector contains corner offsets and face-sag variables (with the
exact gene ordering and units treated as part of the interface). The environment,
surface connectivity, source/detector configuration, materials, and field/pupil
sampling remain explicit metadata rather than being hidden in an opaque vector.

## What does it mean to propose a geometry?

An interaction route does not uniquely determine a geometry. The same sequence of
surface labels can be realised by many positions, orientations, curvatures, and
clearances; conversely, a geometry can induce a route that differs from the one a
designer intended. Therefore a proposer should emit a **candidate geometry plus an
optional route hypothesis**, and the trace should determine the realised route.

A candidate record should contain at least:

```text
requirements and fixed environment
12-gene geometry in canonical units
optional intended route or route distribution
proposer mode, seed, parent IDs, and archive context
observed ray interaction records by field, wavelength, and pupil sample
validity, obstruction, clearance, and numerical-stability flags
package descriptors and optical metrics
```

The route head is optional at first. It may predict a discrete route family,
interaction graph, or sequence embedding, but it must not be treated as proof that
the geometry realises that route. If route discovery is a goal, geometry-first
proposals and route-conditioned proposals should both remain available.

### A concrete numerical proposer

The initial proposer can be a conditional density model rather than a language
model:

1. Encode requirements and fixed environmental quantities.
2. Encode route information when available, using a route head or an embedding of
   ordered surface interactions and field/pupil persistence.
3. Encode the current archive context or a selected parent design.
4. Combine these features with a noise vector.
5. Decode a **mixture distribution** over the 12-gene vector, initially a
   conditional Gaussian mixture or a mixture with diagonal scales.
6. Sample several candidates, apply exact geometry validity checks, trace them, and
   retain the verified outcomes.

An archive-retrieval baseline should be implemented conceptually alongside this:
retrieve geometrically and optically relevant designs, then apply a learned
bounded perturbation distribution. This may be stronger and easier to interpret
than asking a network to generate the entire design from scratch. The learned
perturbation can be conditioned on requirements, route features, archive niche,
and noise. Mixture components provide diversity across distinct geometry basins;
they are not a guarantee of topological novelty.

The decoder must respect the actual parameterisation:

- preserve gene type, physical units, bounds, and scaling;
- encode connectivity and surface identity explicitly;
- distinguish offsets, tilts, sags, and any derived quantities;
- reject self-intersection, degenerate arcs, impossible clearances, and other
  invalid geometry before expensive scoring;
- treat coordinate permutations as equivalent only when all associated metadata,
  surface labels, and interaction records move with them; and
- keep candidate-set order irrelevant when an input contains a population.

The model can first learn viability and route support. A later proposer objective
can ask whether a candidate is useful to a frozen compactor under a fixed budget.
This avoids defining “good proposal” only as the highest initial optical score.

## Three architecture options

### Option A — numerical-only learned search policy

Use a conditional proposer and a structured compactor, both operating on encoded
geometry, route, population, and trace metrics. CMA remains a baseline, teacher,
and optional deployment subroutine. This is the recommended first architecture:
it matches the continuous design space, has low inference overhead, and makes
parameterisation and invariance explicit.

### Option B — numerical core plus LLM orchestration

Keep Option A as the geometry engine. Add an LLM that selects among typed actions,
requests diagnostics, chooses a subset of variables, launches bounded CMA, and
records a human-readable search rationale. The LLM may emit actions such as
`perturb_sags`, `move_corners`, `select_route_family`, `run_bounded_cma`, or
`inspect_field_failure`, but the action schema must map to real operations and
bounded numerical arguments.

This gives AZR-like tool-feedback and curriculum opportunities without assuming
that token generation is the right representation for millimetre-scale geometry.
An adapter-tuned 3B or 7B coding model could be a later comparison, not a premise
of the project. Its value must be measured against the numerical core at equal
trace and wall-clock budgets.

### Option C — program synthesis or literal AZR induction

Have a model infer reusable proposal/compaction programs, then execute them on
hidden geometries. This is scientifically interesting when typed edit programs
have stable behaviour across many designs, but it is not required for a useful
proposer–compactor. A list of successful CMA mutations is not automatically a
dataset of reusable programs. Freeform parameterisations make this route more
ambitious still, so it belongs after the numerical loop is understood.

## Compactor design

The compactor should see a search state, not only a single geometry. Its input can
include:

- current and initial geometry, with normalised gene values and metadata;
- recent candidate populations and their parent/child lineage;
- realised route summaries by field, wavelength, and pupil region;
- optical metrics, packaging metrics, clearances, and constraint margins;
- recent accepted and rejected edits, including failure reasons;
- CMA mean, covariance or factorisation, step size, and evolution-path statistics;
- simulator-call and wall-clock budget remaining; and
- archive descriptors and nearby alternatives.

The compactor can output one of two compatible forms:

1. a structured edit distribution: selected gene groups, bounded deltas, mixture
   weights, and a stop/repair decision; or
2. a correction to a numerical search distribution: mean update, per-group scale
   update, and optional covariance/factor update.

In the first version, predict normalised mean and step-scale corrections while
retaining CMA correlations. A later low-rank factor head can learn additional
correlations if the data justify it. Directly predicting a full covariance is
unnecessarily fragile for a small initial geometry and risks losing positive
definiteness or physical interpretability.

The model should not assume that every proposed child improves the parent. Failed
proposals are useful evidence when they are labelled with the reason for failure;
the episode-level objective should preserve the best verified feasible candidate.
Temporary infeasibility can be explored later, but the first policy can keep
committed states feasible and use bounded repair actions.

## Relation to Evolution Transformer and CMA

EvoTF is a useful representation precedent, not an optics prescription. In §3 and
Fig. 2, it encodes candidate solutions, fitness, and distribution statistics with
causal attention and predicts mean and diagonal-scale updates, targeting
population-order invariance and dimension-order equivariance. §4 distils teacher
trajectories including Sep-CMA on synthetic BBOB tasks through distribution
matching. §6 reports meta-evolution overfitting concerns and §7 discusses
self-referential instability. See the [paper and linked sections](https://arxiv.org/html/2403.02985v1).

### Proposed optics adaptation (not from EvoTF)

The current CMA implementation may use a full covariance or other correlations,
whereas the cited EvoTF formulation predicts a diagonal scale. The first optics
adaptation should therefore retain the current CMA correlation structure and learn
mean/step-scale corrections, or distil a teacher distribution using a matching
parameterisation. It should not call every distribution update “RLVR” or assume
that a diagonal EvoTF head has reproduced full-covariance CMA behaviour.

Potential adaptations are:

- population encoders that are invariant to candidate ordering;
- gene encoders that preserve parameter type, units, bounds, and connectivity;
- route/field encoders that preserve ordered interactions while aggregating over
  pupil samples where appropriate;
- an archive-retrieval branch for diverse basins; and
- a low-rank covariance extension only after the simpler mean/scale policy is
  understood.

## CMA’s role

CMA can serve three roles without being treated as optical ground truth:

1. **Discovery:** generate diverse routes and geometry basins.
2. **Teacher/data source:** provide populations, search distributions, failures,
   and compaction trajectories for supervised warm-up or distribution matching.
3. **Deployment subroutine and baseline:** execute a bounded local search selected
   by the learned policy, while all internal trace calls are counted.

The learner should not imitate only the final elite or presume that every CMA child
is an improvement. Keep the full population, distribution state, candidate
lineage, and evaluation outcome. A legacy QD archive may have richer candidate
records than the active freeform relay trace, which currently appears to emphasise
aggregate progress; the latter needs transition-level logging before it can teach
individual compaction decisions.

## AZR-derived ideas without literal three-task induction

The useful AZR insight is adaptive task generation around a verified executor, not
necessarily the deduction/abduction/induction taxonomy. The optical proposer and
the task proposer are different roles:

- the **optical proposer** emits geometries and optional route hypotheses;
- the **task proposer** selects requirements, starting designs, perturbations,
  target packaging constraints, and budgets for which the current learner has
  informative but incomplete success.

An AZR-inspired task generator can use replay, solvability witnesses, occasional
successful compactions, and held-out tasks. It should avoid tasks that are never
solvable or always solved. It can also vary whether the episode begins from a
retrieved design, a proposer sample, or a CMA population.

The literal three-task split is optional. Deduction-like data (predict the result
of a known edit) may be a useful auxiliary task. Abduction-like data (find a
starting design for a specified transformation) may be tested later. Induction of
reusable edit programs should wait until the logs contain repeated, executable
program-like behaviour. The main compactor can learn directly from verified
state–action–outcome sequences.

See [the local RLVR/AZR notes](RLVR_AZR.md) and the [AZR paper, §§3.1–3.3](https://papers.nips.cc/paper_files/paper/2025/file/9837dc00ff67d176373268ed48042d49-Paper-Conference.pdf).

## Data requirements

No human-written rationales are required for the initial numerical system. It does
need substantial verified simulator experience, including negative outcomes.
Useful records are:

| Record | Minimum contents | Purpose |
|---|---|---|
| Starting design | requirements, geometry, route, metrics, constraints | proposer conditioning and viability |
| Population snapshot | all candidates, scores, validity, parent IDs, distribution state | distribution learning and diversity |
| Transition | state, typed action/distribution change, outcome, failure reason | compactor training |
| Episode | initial state, all decisions, best feasible state, budgets | policy updates and evaluation |
| Task | fixed environment, start-state source, target constraints, witness status | adaptive curriculum and held-out tests |

Preserve lineage, search group, random seed, environment identity, and archive
niche. Hold out complete geometry basins and environments, not random rows from the
same trajectory. Fold count is a useful coarse descriptor, but route order,
repeated contacts, pupil/field persistence, and interaction support should be
retained as richer descriptors. Archive adjacency is not parentage.

Within an episode, keep source, detector, material, field definition, pupil
sampling, and other environmental quantities fixed. Across episodes, vary them to
test generalisation. The prior sparse-verifier proxy pilot only tested a prediction
relationship; it did not validate this proposer–compactor architecture or prove
that compaction is learnable.

## Verification, reward, and accounting

Let $C(x)$ be the declared packaging quantity (for example, envelope width or a
specified combination of dimensions). A feasible episode can report:

$$
B_C = 1 - C(x_{\mathrm{best}})/C(x_0).
$$

Feasibility must include absolute optical floors, acceptable relative retention of
the starting design where appropriate, clearances, geometry validity, and stable
route behaviour across the required fields and pupil samples. Relative retention
alone is unsafe: preserving 98% of a poor starting score is not success.

The verifier should be treated as a fallible engineering component, not as exact
physics or an exploit-proof oracle. Record trace failures, grazing contacts,
sequence changes, numerical degeneracies, and suspicious boundary behaviour. The
reward can include a small evaluation-cost penalty, but every trace and every
internal CMA evaluation must be counted. Report both simulator calls and wall-clock
time.

For the proposer, use a staged objective: first verified viability and route
support; later expected compactability under a frozen compactor and fixed budget.
For the compactor, use the best feasible packaging improvement plus constraint and
cost accounting. Keep the baseline comparison explicit: CMA alone, archive
retrieval plus perturbation, learned numerical policy, and optional LLM
orchestration.

## Training and evaluation plan

This is a staged plan, not a request to run it now:

1. **Interface specification:** freeze the 12-gene encoding, units, connectivity,
   route record, validity checks, metric definitions, and candidate schema.
2. **Experience audit:** distinguish complete candidate/population/transition logs
   from aggregate progress traces; label failures and lineage.
3. **Numerical proposer warm-up:** train a viability/route-support model and
   compare conditional mixtures against archive retrieval plus learned perturbation.
4. **Compactor warm-up:** distil or imitate bounded CMA distribution changes using
   a parameterisation that matches the teacher; retain correlations initially.
5. **Verified policy updates:** apply stochastic policy updates from terminal/best
   feasible compaction rewards. PPO-style updates are natural for a numerical
   policy; token-level GRPO is optional only for an LLM action layer.
6. **Adaptive task generation:** introduce AZR-inspired replay and solvability
   targeting after fixed-task training is understood.
7. **Optional orchestration ablation:** compare numerical-only against an LLM that
   selects typed tools and diagnostics, at equal trace, CMA-call, and wall-clock
   budgets.
8. **Freeform stage later:** only after spherical/arc proposer–compactor behaviour
   is measured, add a declared freeform basis and a separate refinement study.

Important ablations include proposer-only versus proposer plus compactor, fixed
versus adaptive tasks, archive retrieval versus mixture generation, diagonal versus
retained/learned correlations, candidate-only versus full-population context, and
LLM orchestration versus no LLM. Evaluate on unseen requirements, environments,
and starting basins; vary inference budget independently of training experience.

The available two RTX 6000 Ada 48 GB GPUs are adequate for a small numerical model
and can support adapter experiments, but they should not be treated as one idle
96 GB pool. A 3B or 7B adapter model is an optional benchmark, not a dependency.
The numerical core is expected to be considerably cheaper and easier to inspect.

## Planned milestones and caveats

1. **Representation milestone:** a candidate and transition record can be replayed
   and independently re-traced with all metadata intact.
2. **Baseline milestone:** CMA, retrieval-plus-perturbation, and a simple numerical
   proposer/compactor share the same verifier and accounting.
3. **Generalisation milestone:** a learned policy transfers to held-out basins and
   fixed environments without collapsing route or archive diversity.
4. **Task-generation milestone:** adaptive tasks improve informative success or
   coverage over a fixed curriculum, rather than merely changing task difficulty.
5. **Architecture milestone:** any LLM orchestration or AZR-style program layer
   earns its additional cost under matched budgets.

No gain should be assumed in advance. The main scientific question is whether a
learned numerical policy reuses verified search structure across optical
requirements and geometry basins better than CMA alone at equal trace and
wall-clock budgets. Freeform refinement, literal program induction, and claims of
new interaction-topology discovery should remain separate hypotheses.

## Related local notes

- [EvoTF with optical embeddings for folded-optics search](evotf_folded_optics_implementation_plan.md)
- [Automated interaction-sequence discovery and later refinement](03_future_directions_interaction_sequence_discovery.md)
- [Spherical inception/compaction status](04_spherical_inception_compaction_20260904.md)
- [Dataset-generation audit](02_data_generation_methods_and_dataset_audit.md)
- [Related freeform-automation literature](01_related_literature_freeform_automation.md)
- [Ray-sequence training pipeline notes](ray_sequence_training_pipeline.md)
