# Numerical discovery with new-Qwen compaction and refinement

Assessment: 2026-09-11. This supersedes the smaller-Qwen training recommendation in
the initial AZR assessment. The new local Qwen is the only intended LLM for future
work. Historical 8B artifacts remain evidence, not a dependency of the proposed system.
No weights, historical experiments, or shared runtime code were deleted in this review.

## Recommendation

**User-confirmed direction, 2026-09-11:** strengthened numerical discovery plus the
new Qwen for compaction and refinement is the active frontier. The named learned
optimizers remain future options; implementing them is not an initial milestone.
Establish the numerical baseline and frozen new-Qwen tool integration first.

Pursue numerical starting-point generation followed by new-Qwen-guided compaction and
refinement. First repair the experimental formulation and establish a strong numerical
baseline. Then test learned population retention (LQD) or optimizer configuration
(ConfigX), depending on the measured bottleneck. MetaDistill becomes attractive after
collecting useful, diverse numerical trajectories. Do not begin by generating a huge
dataset from the current proposer or by training Qwen to output entire designs.

This is an engineering recommendation from the evidence below, not a measured ranking
of these methods on optics. None of the external methods was executed during this review.

Two meanings of proposer must remain separate: a geometry proposer finds starting
designs for a fixed optical specification; an AZR task proposer invents training tasks.
The immediate priority is the first. A future learning loop can use its trajectories,
but task generation should not distract from improving actual design discovery.

## Repository findings

1. `nslens/qd/driver.py` defaults to isotropic steps in mixed physical units. It already
   supports normalized coordinates. Historical comments identify severe boundary
   saturation in the index gene. This is not evidence that every later run used the
   legacy setting; each manifest must be checked.
2. The CMA implementation adapts both covariance and global step size. The concern is
   its surrounding parameterization, initialization, clipping, ranking, and restarts,
   rather than a claim that CMA itself uses a permanently fixed step.
3. Rejected evaluations receive the same `-1e9` ordering key. The search therefore
   cannot rank small versus large physical violations within that rejected population,
   even though diagnostics contain more information. One generation with no archive
   addition triggers a restart; this can repeatedly discard adaptation before it helps.
4. The `2e12`/`1e12` offsets implement archive status ordering, not physical reward
   weights. Replacing these numbers alone does not change rank-based CMA behavior.
   Whether new-cell preference serves downstream engineering is the real question.
5. The historical nine-gene problem changes source position, angle, and material with
   geometry. `fixed_environments.py` already supplies a six-geometry-gene alternative.
   Use externally fixed specifications for comparisons; allow discrete material choice
   only when it is genuinely part of the design brief.
6. The separate `ns_probe/run_novel_compaction.py` discovery stage uses anchor mutation,
   not CMA: 8% uniform draws, log-uniform mutation scale 0.012–0.18, and a 0.30 first-body
   multiplier. Its compactor uses `max(0.004, 0.035 * 0.82**iteration)` and ranks package
   size using 0.65 thinness + 0.35 width after baseline normalization. These are concrete
   hand-selected policies, distinct from CMA's adaptation.

The saved three-seed optical ablation reports median invalid/rejected rates of 99.38%
for CMA and 99.47% for released EvoTF. Median QD scores are 3.009 and 2.247 respectively;
the optical-trained arm also has median 2.247. All have identical median best fitness
0.8600 in this warm-started benchmark; separate inherited quality from new progress.
These metrics are from `results/fold_evotf_pilot_a_ablation_6000/summary.json`, not a new
replay. Rejection does not necessarily mean malformed geometry: it includes optical
gates. Audit the rejection stages before deciding which change will help.

The synthetic EvoTF experiment also lost to CMA in 84/100 paired instances. Its diagonal
distribution is one limitation, but losses on separable tasks show it is not the only
explanation. See `evotf_toy_experiment.md`.

The separate spherical discovery experiment promoted zero strongly novel viable designs
from 50,000 proposals. Its boundary compaction demonstration reduced envelope width
11.12% but thinness only 4.26%. That single case is useful feasibility evidence, not a
general improvement claim. See `04_spherical_inception_compaction_20260904.md`.

## How the named methods fit

| Method | What it learns | Proposed use and priority |
|---|---|---|
| LQD | Attention-based competition/retention among population members | Closest fit for preserving diverse useful stepping stones. First learned discovery candidate if archive selection is limiting. It is a different intervention from EvoTF's distribution updates, not a drop-in EvoTF checkpoint upgrade. |
| ConfigX | A modular evolutionary-algorithm configuration policy trained with multitask RL | Strong fit if choice of operators/settings/restarts is limiting. Start with a small numerical controller baseline; its published zero-shot results do not establish optical transfer. |
| MetaDistill | Distillation from numerical optimizer trajectories, plus optional SSFT variants | Good later path for exploiting mixed-teacher experience. Fix teacher quality first. This is training a numerical optimizer, not a requirement to train an old or smaller Qwen. |
| EPOM | Population-to-search-strategy mapping using cross-attention, deformable attention, and differentiable crossover | Worth an isolated pretrained comparison after the feasibility interface is repaired. Broader replacement than archive selection; training/integration and constrained-optics transfer need checking. |
| L2E | Structured neural unrolling with stability-biased updates and proxy numerical guidance | Longer-term option for coupled proposal/refinement. Validate its assumptions and proxy quality near route changes before applying its stability claims to optics. |

L2E here means Gao et al., *Learning to Evolve for Optimization via Stability-Inducing
Neural Unrolling*, arXiv:2512.11453v2. Several other papers share the name; the ranking
is conditional on this being the intended work.

Primary sources consulted:

- [LQD reference implementation](https://github.com/maxencefaldor/learned-qd) and
  [paper](https://arxiv.org/abs/2502.02190). The repository specifies Python 3.13+ and
  JAX; isolate it from the existing runtime. Released checkpoint availability and an
  optical ask/tell adapter still need verification.
- [ConfigX paper](https://ojs.aaai.org/index.php/AAAI/article/view/34904).
- [MetaDistill official implementation](https://github.com/Roki-Xing/MetaDistill), with
  pretrained assets and teacher-collection scripts. Its example uses population 200
  and 100 generations: optical evaluation cost must be assessed before transplanting
  those settings. The paper's OpenReview page was blocked during this review, so no
  detailed SSFT mechanism or paper-level numerical gain is asserted here.
- [EPOM paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2fd37205c429a30cc1bb0afdbbf96857-Paper-Conference.pdf).
  Its reported BBOB comparison uses 30 and 100 dimensions; this is not equivalent to
  our constrained, discontinuous optical objectives. A runnable official checkpoint
  was not established in this review.
- [L2E paper, current version](https://arxiv.org/html/2512.11453v2). Runtime and checkpoint
  readiness were not established in this review.

BBOB is valuable optimizer evidence, but ray-path switching, rejection plateaus,
manufacturing constraints, archive diversity, and downstream compactability add different
requirements. Learned methods still need specified objectives and meta-training budgets;
they do not automatically eliminate arbitrary engineering tradeoffs.

## Proposed pipeline

Fixed optical brief → numerically generated seed portfolio → dense certification →
Qwen chooses compaction/refinement operations → numerical execution → independent held-out
sampling audit. Keep topology exploration in discovery and preserve the chosen route
during an initial compaction experiment. Later topology-changing edits need a separate
experiment and explicit recertification.

For numerical discovery:

- Normalize continuous coordinates and parameterize obvious geometric constraints
  directly where possible. Compare principled boundary handling with clipping under
  identical budgets; count retries. Separate structural moves from local geometry moves.
- Separate simulation-invalid geometry from evaluable constraint violations. Never
  reward malformed geometry. For evaluable near-feasible candidates, retain a vector
  of violations normalized by specification tolerances; compare feasibility-dominant
  ordering or a separate infeasible population. Keep final physical gates unchanged.
- Benchmark restart patience and initial scales on development families, with an
  explicit tuning budget. Log covariance condition, clipping, lifespan, failure stage,
  and new valid designs. Do not select arbitrary replacement constants by intuition.
- Retain physically meaningful route and package diversity with canonical signatures.
  Measure how many seeds survive compaction; archive occupancy alone is insufficient.
  Evaluate a bounded subset downstream and charge that selection cost.

For compaction, use a declared engineering objective such as minimum package thickness
subject to width, throughput, output error, route support, TIR margin, and robustness
constraints. If there is no single package priority, report the feasible Pareto frontier
instead of selecting an unexplained weighted sum. A 2-D area is not a 3-D volume.

Let new Qwen inspect ray/field summaries and choose surface freedoms, variable blocks,
numerical solver, bounded budgets, rollback, and stopping. Numerical tools adapt steps
and execute edits. Test an adaptive trust-region/local-search compactor as the control
against the old fixed-decay schedule. Qwen's extra language compute must be reported.

## First decisive experiment

First run a small feasibility and runtime pilot on development specifications; use its
variance to size the held-out experiment. Freeze acceptance and compaction objectives
before evaluating held-out specification families and track shared seed ancestry.

1. Compare the historical proposer with normalized CMA under the same specification
   and acceptance rules. Add rejection-aware ranking and restart changes as separate
   ablations. Include a simple archive-perturbation or DE control to test whether CMA
   machinery is contributing at all.
2. Using the same resulting seed set, compare a strong numerical compactor with the
   new-Qwen-guided compactor. This isolates the LLM contribution.
3. Once stable, cross old/improved proposer with numerical/Qwen compaction in a 2×2
   comparison. Match total physical budgets across generation, selection, compaction,
   and verification; also report equal wall-time results if LLM latency is substantial.
4. Add one learned method based on diagnosis: LQD for retention/diversity, ConfigX for
   configuration, or MetaDistill when diverse good teachers are already available.

Default provisional target: 8–10% relative reduction in final package thickness at
fixed optical acceptance and total ray budget, compared with the strongest numerical
pipeline. Confirm which packaging dimension matters before implementation. Report
success rate jointly; do not hide failures by reporting only successful compactions.
Use all-task target attainment, rays-to-target, and family-level paired intervals.

Keep a compact trajectory corpus as a by-product: specification, genealogy, initial
population/state, all evaluated candidates including failures, physical metrics,
operator/settings, budgets, route signature, and downstream compaction outcome.
Split by specification/ancestry before selecting trajectories. A large archive of final
elites alone cannot teach which decision caused progress. This corpus can later support
MetaDistill or training the new Qwen, once either has a justified role.

The immediate work is therefore a proposer/compactor benchmark and interface cleanup.
Full AZR is deferred until the numerical supply and new-Qwen contribution are measured.
