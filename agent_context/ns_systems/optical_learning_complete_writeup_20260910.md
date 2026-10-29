# Optical environment learning: consolidated implementation and experiment write-up

**Status: 2026-09-10.** This is the consolidated handoff for the work from the initial
verified-data/repair experiments through the completed two-round Qwen pencil-beam
curriculum experiment. It records results, unsuccessful approaches, scope corrections,
code, and artifacts. Earlier experiment numbers are drawn from their saved reports;
this documentation pass did not rerun those experiments. Existing optical archives and
tracing infrastructure are dependencies, not newly discovered results of the Qwen loop.

## Current outcome

We have an executable, physically audited learning loop for a small **folded optical
repair-planning module**. It measures Qwen's current performance, selects training
environments, updates a LoRA adapter, and measures the updated model. A controlled
pilot compares uniform, fixed-schedule, and adaptive environment selection.

The latest experiment uses **a 1 mm source field and a ±0.25° angular fan**, actual
freeform boundaries, refraction and total internal reflection (TIR). Across four
held-out tasks and four decoding seeds, the final results are:

| Model / curriculum | Successful attempts |
| --- | ---: |
| Frozen Qwen3-8B | 1/16 |
| Uniform sampling | 5/16 |
| Fixed schedule | 14/16 |
| Adaptive sampling | 12/16 |

These are repeated attempts on **four tasks**, not 16 independent optical designs.
There is one training seed and one held-out starting prescription. The result supports
learning from selected verified experience in this pilot. It does **not** establish
adaptive selection as better than a fixed curriculum, broad optical-engineering
competence, or learned synthesis of new environments.

The adaptive sampler currently selects from a finite pool. The model selects a
numerical repair plan; the optimizer refines surface coefficients. New folded topology
discovery, a learned Qwen proposer, and reinforcement-learning self-play remain future
work. EvoTF is not a dependency of this learning loop.

## Objective and scope corrections

The intended objective is to discover useful training environments so that models learn
better freeform optical engineering. Several corrections shaped the implementation:

1. The initial transmitting-element experiments established data generation and learning
   mechanics, but covered too little of the requested folded-design domain.
2. Introducing Qwen established real language-model tool use, but successful tool calls
   were insufficient evidence of optical reasoning or a learning advantage.
3. The work shifted to actual folded freeforms, initially through repairs of a known
   TIR prism. This added physical folding but still lacked a measured learning cycle.
4. The user requested pencil beams and a source field no wider than 1 mm. The evaluator
   now separates field width from angular spread and supports that narrower problem.
5. The immediate goal was clarified as demonstrating a useful environment-learning loop.
   Broad architecture diversity is not a prerequisite for that first demonstration.

An earlier recommendation that tasks must defeat a universal numerical optimizer was
too strong for this objective. Such controls help assess what the model contributes,
but an optimizer solving a task does not invalidate the task as learning experience.
The relevant test is whether selected experience improves the learned module more than
appropriate nonadaptive training at matched cost.

## What the different “learned models” are

| Component | What is learned | What remains numerical |
| --- | --- | --- |
| Earlier response/Jacobian model | Local optical response to surface changes | A regularized solve computes a repair step |
| Full-tool Qwen baseline and LoRA | Tool and surface-selection behavior across an episode | Ray tracing, merit evaluation and coefficient optimization |
| Latest pencil-beam Qwen planner | Selection among eight bounded repair plans from the initial optical state | Each selected plan runs coordinate search and optical verification |
| Current adaptive environment sampler | No neural proposer is trained; sampling probabilities update from solver failures | Environment feasibility and outcomes are simulator-verified |

None of these is currently a language model that independently designs an arbitrary
folded architecture from scratch. Qwen3-8B is a real local pretrained language model,
not the small earlier response predictor.

There were also two different uses of “144”:

- The earlier response-model study reported **144 test tasks** across three training
  seeds and 36 test specification families.
- The original language-model workbench corpus contains **144 scripted episodes**:
  36 tasks, each attempted by four numerical plans. Its Qwen test split has 12 tasks.

These counts must not be interchanged.

## Experiment history

### 1. Verified transmitting-freeform response learning

The earlier experiment generated certified repair tasks, retained physical probes and
failures, learned a local response model, and compared uniform, fixed and adaptive
severity sampling. It edited four Fourier coefficients on a 2-D transmitting boundary.

The saved report gives these results over three training seeds:

| Method | Success after two candidate calls | After eight |
| --- | ---: | ---: |
| Initial response model | 82.6% | 100.0% |
| Continued uniform generation | 89.6% | 100.0% |
| Continued fixed curriculum | 87.5% | 100.0% |
| Continued adaptive curriculum | 88.9% | 100.0% |
| Finite-difference initialization plus Broyden | — | 98.6% |

Additional verified data improved short-budget repair. Adaptive sampling did not beat
uniform sampling. An earlier action-imitation attempt regressed; learning measured
responses proved more stable in the separate development experiments.

See [verified response-learning report](verified_freeform_learning_loop.md),
[implementation guide](../../nslens/learning/REPAIR_LOOP.md), and
[experiment artifacts](../../runs/repair_loop_verified_20260910_v3/).
The saved report records a historical relay-fixture failure; no claim is made that the
entire repository test suite is green.

### 2. Executable optical tools and the first transcript corpus

A transmitting-element workbench exposed inspect, enable, perturb, optimize,
restore-best, verify and finish operations. Every optimizer probe was charged;
requirements were immutable through the tools. Private witnesses stayed out of prompts.
State hashes and recorded evaluations made episodes replayable.

The corpus contained 12 specification families, 36 tasks and 144 numerical episodes.
The split was six training families, two validation families and four test families.
The training export contained 54 successful conversations and 18 plan preferences.
All 144 episodes replayed.

See [original workbench report](optical_workbench_pilot.md) and
[corpus](../../runs/optical_workbench_pilot_20260910_v1/).

### 3. Real Qwen inference and the unsuccessful mixed-teacher fine-tune

Qwen3-8B was installed locally and pinned to revision
`b968826d9c46dd6066d109eabc6255188de91218`. Native Qwen tool-call formatting, parsing,
raw-response recording, input-token reservation and exact physical replay were added.
Malformed generations are retained and are not partially executed.

| Original unfolded test setting | Verified submissions | Optical designs passing, including unsubmitted |
| --- | ---: | ---: |
| Frozen Qwen, 24k cumulative language ceiling | 0/12 | 12/12 |
| Frozen Qwen, 64k ceiling | 12/12 | 12/12 |
| Mixed-teacher LoRA, 64k ceiling | 10/12 | 10/12 |

The 24k limit could not accommodate even the complete successful teacher conversations
in the native template. A preliminary run also exposed a token-reservation bug, which
was fixed before the hard-cap comparison. Raising the ceiling enabled submission; it
was not optical learning. The two frozen runs produced identical final prescriptions.

The mixed-teacher adapter trained on 54 conversations and 378 assistant-call targets,
updating 3,833,856 LoRA parameters in approximately 822 seconds. It reduced repeated
verification but sometimes chose an unsuitable one-sided optimization plan. Mean audit
RMS worsened from 2.596 to 15.238 micrometers. Its promotion check rejected it.

A follow-up adapter using 18 conversations from a robust teacher completed training,
and a fresh numerical test corpus was built and audited. **Qwen evaluation of that
follow-up is parked.** It must not be described as a successful experiment.

See [Qwen results](qwen_optical_workbench_results.md),
[comparison artifacts](../../runs/qwen3_8b_workbench_comparison_20260910_v1/),
[parked adapter](../../runs/qwen3_8b_workbench_robust_lora_20260910_v1/), and
[fresh numerical corpus](../../runs/optical_workbench_fresh_test_20260910_v1/).

### 4. Actual folded freeform tools and Qwen repairs

The folded domain was derived from an existing archived AR/VR prism prescription.
Four face descriptions were converted to endpoint-preserving radial freeforms and a
single closed periodic C2 spline with 128 anchors. This **changes the archived
representation**; the resulting design was re-traced rather than accepting its old score.

Eight coefficients deform the four surface regions. The dominant useful route is
entry refraction, one TIR fold, then exit refraction toward the eyebox. Small multi-bounce
ghost branches do not make it a useful multi-fold architecture.

Two coefficient-corruption tasks from one starting prescription were run:

| Method | Submitted and audited successes |
| --- | ---: |
| Fixed numerical plan | 2/2 |
| Frozen Qwen, 64k language ceiling | 1/2 |
| Frozen Qwen, 128k language ceiling | 2/2 |

Both 64k optical outcomes passed, but one episode ran out of language budget before
submission. The 128k run produced the same prescriptions. Both were checked on a denser
273-ray grid, and the episodes replayed exactly. No folded fine-tuning occurred at this
stage, and no new architecture was discovered by Qwen.

See [folded report](folded_qwen_workbench.md) and
[completed Qwen run](../../runs/folded_workbench_qwen_128k_20260910_v1/).
The user's currently open `runs/folded_decisions_qwen_20260910_v1/episode001.json`
belongs to the subsequent constrained pilot below, not the newest pencil-learning run.

### 5. Constrained folded decision benchmark and archive screening

The tools gained enforced per-task editable coefficients, change limits and candidate
budgets. Illegal actions are rejected before tracing; numerical probes remain charged.
The best-prescription cache was also hardened against physically invalid candidates.

Twenty-four top archive candidates were screened under the then-current wide-field
profile. Only the previously used seed passed the complete screening pipeline. Six
candidates were outside the engine's supported refractive-index interval, and two
converted boundaries self-intersected. These are support/representation failures;
other candidates failed optical gates. Screening was not an impossibility proof.

Four feasible development repairs from the surviving seed were tested at 12 candidate
traces. Qwen passed 3/4, while the initial all-surface numerical control passed 2/4.
Qwen's different step size motivated an explicitly post-hoc control: the all-surface
plan with a fixed 0.5 mm step passed 4/4. This removed the apparent Qwen advantage over
the weaker control. All 24 episodes—16 initial controls, four Qwen, four step-size
controls—replayed exactly.

See [constrained benchmark report](folded_decision_benchmark.md),
[archive screen](../../runs/folded_screen_20260910_v1/),
[numerical cases](../../runs/folded_decisions_20260910_v1/), and
[Qwen cases](../../runs/folded_decisions_qwen_20260910_v1/).

### 6. Pencil-beam domain and the completed learning loop

The user's narrower field request was implemented as three source positions spanning
54.5–55.5 mm at y=50 mm, with a finite ±0.25° angular fan. Width and angular spread are
explicitly separate quantities. The old 12° FOV and image-ordering gates were removed
from this routing profile, without changing the older workbench defaults.

Four of the 24 screened prescriptions passed the single-fold pencil profile. None
passed when at least two useful TIR events were required. All four admitted witnesses
passed the coarse, 183-ray audit and 273-ray sampling checks. Their shared evolutionary
ancestry remains unresolved.

The latest task adds per-field output-angle targets with ±0.15° tolerance. It retains
at least 30% useful folded throughput per field, angular spread ≤1°, and geometry and
energy gates. The task generator created 16 initially failing, witness-backed tasks,
with a split fixed before Qwen rollouts:

- Eight training tasks from two starting prescriptions.
- Four validation tasks from a third prescription.
- Four test tasks from a fourth prescription.

Qwen receives the initial optical state and chooses one of eight plans: all surfaces,
A, B, or T, each with an initial step of 0.25 or 0.5 mm. Each plan uses 12 candidate
traces, restores its best coarse design and submits to the hidden audit. The teacher
selects a successful training plan by its audit merit. Training targets only the
assistant call; simulator text and witnesses are not prediction targets.

The curriculum arms run two collect/select → train → evaluate rounds:

| Arm | Selection rule |
| --- | --- |
| Uniform | Sample eligible training tasks uniformly, with replacement |
| Fixed | Seeded cyclic schedule, independent of learner outcomes |
| Adaptive | 75% probability mass on current failures; 25% uniform coverage |

The adaptive rule is programmed, not a trained proposer. It updates from training
feedback only. Its second round drew the same failing task three times, illustrating
that a probability floor does not guarantee diversity in a tiny batch.

Primary-seed results across rounds:

| Arm | Training: before → round 1 → round 2 | Validation | Test |
| --- | --- | --- | --- |
| Uniform | 6/8 → 5/8 → 4/8 | 2/4 → 2/4 → 2/4 | 0/4 → 0/4 → 0/4 |
| Fixed | 6/8 → 6/8 → 8/8 | 2/4 → 2/4 → 4/4 | 0/4 → 3/4 → 3/4 |
| Adaptive | 6/8 → 6/8 → 8/8 | 2/4 → 2/4 → 2/4 | 0/4 → 3/4 → 3/4 |

All training finished before opening test results. Three extra decoding seeds were
specified before the first test evaluation. Their combined final results appear at
the top of this document. Uniform training shows some improvement in those repeated
decodes even though its primary-seed result is zero. No later training used test feedback.

The experiment demonstrates a small planner-learning effect and sensitivity to the
chosen experience. It does not show continued test-success gains after round one or
an adaptive-over-fixed advantage. The fixed arm also has better validation performance
and lower primary test merit than adaptive.

See [detailed learning report](pencil_qwen_learning_results.md),
[experiment contract](pencil_environment_learning_loop.md),
[corpus](../../runs/pencil_learning_20260910_v1/), and
[models and results](../../runs/pencil_qwen_learning_20260910_v1/).

## Data flow and verification

```mermaid
flowchart LR
    A[Archived prescription] --> B[Verify narrow-field witness]
    B --> C[Generate corruptions and fixed targets]
    C --> D[Freeze prescription-grouped splits]
    D --> E[Execute and retain all eight numerical plans]
    E --> F[Qwen chooses a plan from the initial state]
    F --> G[Score against verified optical outcomes]
    G --> H[Select training environments]
    H --> I[Train Qwen LoRA]
    I --> F
    I --> J[Validation each round]
    I --> K[Final test after all training]
```

The numerical outcomes are cached because the environment and plans are deterministic.
Every one of the 128 task/plan combinations was physically executed and then replayed.
A model decision retrieves its selected plan's measured outcome; this is not represented
as a new live ray-tracing invocation. Its online-equivalent optical budget is recorded.

Latest verification completed:

- 128 exact physical episode replays.
- Six trained adapters and 112 primary model decisions audited.
- Another 48 decoding-check decisions audited, including raw native tool-call alignment.
- Train-only teacher selection and curriculum feedback checks; split IDs excluded from
  model observations; matching selection probabilities reproduced from saved feedback.
- 26 workbench/pencil-learning tests passed. This is a targeted result, not a full-repo claim.
- Code snapshots, task hashes, model manifests, raw outputs, numerical probes, failed
  plans, sampling probabilities, training logs and checkpoints retained.

Within each trained arm, the two rounds consumed 16 optimizer updates, 65,536 padded
training tokens, 352 supervised completion tokens, 40,398 development language tokens,
and 20,304 online-equivalent development rays. Actual LoRA training took about 32–34
seconds per arm on GPU 1; generation, inference, loading and audits take additional time.

The common corpus cost was 11,529 generation/certification rays plus 108,288 teacher-cache
rays. Replaying the cache cost another 108,288 rays as verification overhead. Shared costs
are charged consistently across arms; cached outcomes are not claimed as free training
data. Equal padded tokens and updates do not imply identical wall-clock runtime.

## Code and runtime entry points

| Area | Main files |
| --- | --- |
| Earlier response learning | [freeform_repair.py](../../nslens/learning/freeform_repair.py), [run_repair_curriculum.py](../../nslens/analysis/run_repair_curriculum.py) |
| Original tool corpus | [tools.py](../../nslens/workbench/tools.py), [episodes.py](../../nslens/workbench/episodes.py), [audit.py](../../nslens/workbench/audit.py) |
| Local Qwen transport and full-tool training | [qwen_baseline.py](../../nslens/workbench/qwen_baseline.py), [train_qwen.py](../../nslens/workbench/train_qwen.py), [compare_qwen.py](../../nslens/workbench/compare_qwen.py) |
| Folded physics and tools | [folded_domain.py](../../nslens/workbench/folded_domain.py), [folded_tools.py](../../nslens/workbench/folded_tools.py), [folded_baseline.py](../../nslens/workbench/folded_baseline.py) |
| Archive and constrained controls | [folded_screen.py](../../nslens/workbench/folded_screen.py), [folded_decisions.py](../../nslens/workbench/folded_decisions.py), [folded_step_control.py](../../nslens/workbench/folded_step_control.py) |
| Pencil environment and curriculum | [pencil_screen.py](../../nslens/workbench/pencil_screen.py), [pencil_loop.py](../../nslens/workbench/pencil_loop.py), [train_pencil_loop.py](../../nslens/workbench/train_pencil_loop.py) |
| Pencil audits and reporting | [audit_pencil_loop.py](../../nslens/workbench/audit_pencil_loop.py), [audit_pencil_training.py](../../nslens/workbench/audit_pencil_training.py), [pencil_sampling_check.py](../../nslens/workbench/pencil_sampling_check.py), [report_pencil_loop.py](../../nslens/workbench/report_pencil_loop.py) |

The NumPy/SciPy tracing scripts use the repository `.venv`. The Qwen runtime uses:

```bash
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/hisham26/optics_models/runtime:$PWD

/home/hisham26/Desktop/simu1/deeplens_env/bin/python \
  -m nslens.workbench.train_pencil_loop runs/pencil_learning_20260910_v1 \
  --weights /data/hisham26/optics_models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --out runs/pencil_qwen_learning_new
```

Use a fresh output directory. Qwen weights are BF16; native tool prompting uses
non-thinking mode with temperature 0.7, top-p 0.8 and top-k 20. LoRA targets q/v attention
projections with rank 8 and alpha 16. The latest loop uses AdamW at 1e-4, resetting the
optimizer each round while retaining the adapter. All arms start from the same base
and adapter initialization. GPU 0's existing workload was left untouched. The experiment
processes finished and GPU 1 was released. No model was deployed and no commit was made.

## Images

All solution images are Matplotlib renders from actual prescriptions and traced rays.
They are not generated illustrations. Selected examples are visual evidence, not an
unbiased sample of performance.

- [Latest learning curves](../../runs/pencil_qwen_learning_20260910_v1/learning_curves.png).
- [Latest pencil-beam repair and before/after angular errors](../../runs/pencil_qwen_learning_20260910_v1/learned_pencil_repair.png).
- [Earlier folded solution 0](../../runs/folded_workbench_qwen_128k_20260910_v1/folded_solution_0.png).
- [Earlier folded solution 1](../../runs/folded_workbench_qwen_128k_20260910_v1/folded_solution_1.png).
- [Constrained folded repair](../../runs/folded_decisions_qwen_20260910_v1/constrained_repair.png).
- [Unfolded baseline image collection](../../runs/qwen3_8b_workbench_comparison_20260910_v1/png_baseline/).
- [Unfolded LoRA image collection, including regressions](../../runs/qwen3_8b_workbench_comparison_20260910_v1/png_lora/).
- [Unfolded training-effect comparison](../../runs/qwen3_8b_workbench_comparison_20260910_v1/training_effect.png).

## What remains unimplemented or unproven

The latest loop does not yet generate novel environments in response to feedback. It
reweights a fixed set. There is no learned Qwen proposer, proposer reward training,
reinforcement-learning update, or full AZR self-play implementation. Current training
is simulator-verified supervised imitation of numerical-plan choices.

Only limited abduction—finding a compatible repair—is exercised. Independent prediction
of optical consequences and induction of an executable response law are not separately
scored for Qwen. A successful repair need not identify the original corruption.

Useful multiple-fold architecture generation, 3-D freeform imaging, polarization,
manufacturing constraints, tolerance robustness, and independent optical-engine agreement
have not been established by these experiments. Dense sampling uses the same tracer.
The narrow source bundle is not evidence of wide-field image quality.

## Recommended continuation

1. Preserve this pilot and its test set as completed evidence. Do not tune against its
   four exposed test tasks and call the result held-out improvement.
2. On newly reserved tasks, compare a sampler that enforces distinct environments and
   retains earlier successes with the current adaptive sampler and fixed schedule.
   Repeat across training seeds, reporting uncertainty by environment group.
3. Extend the controller from selecting tasks to proposing bounded changes in source
   bundle, target, surface access, corruption and trace budget. Keep feasibility witnesses
   private and keep verifier rules outside proposer control. Charge rejected proposals.
4. Judge proposed environments by subsequent learning benefit at matched total cost;
   current failure rate alone is a proxy that can overconcentrate the curriculum.
5. After a controlled benefit is established, add a learned proposer and broader folded
   architectures. Keep the current small domain as a retention and regression benchmark.

The deliverable at this point is an operational, auditable small learning loop with a
measured planner improvement and informative controls. The larger environment-discovery
framework remains the objective, not an accomplished claim.
