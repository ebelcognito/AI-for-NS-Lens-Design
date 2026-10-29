# Two-round Qwen pencil-beam curriculum experiment

This experiment trains an actual Qwen3-8B LoRA adapter and closes a small feedback loop:
measure the current planner, select training environments, train, then measure again.
It compares adaptive selection with uniform sampling and a fixed schedule. The learned
module chooses a numerical repair plan; it does not synthesize a new optical topology.

## Measured primary result

| Policy after two rounds | Training | Validation | Untouched test |
| --- | ---: | ---: | ---: |
| Frozen Qwen | 6/8 | 2/4 | 0/4 |
| Uniform curriculum | 4/8 | 2/4 | 0/4 |
| Fixed schedule | 8/8 | 4/4 | 3/4 |
| Adaptive curriculum | 8/8 | 2/4 | 3/4 |

On the primary decoding seed, the loop produces a measurable planner improvement in this pilot: the adaptive and
fixed arms improve test success from 0/4 to 3/4, while the equal-budget uniform arm
does not. This is evidence that the selected experience matters here. It does **not**
establish an advantage of adaptive selection over a fixed curriculum. The fixed arm
has better validation success and lower final test merit (1.319 versus 17.465).

Both successful arms reach 3/4 test success after round one and remain there after
round two. Therefore the experiment does not show continuing held-out success gains
across rounds. All rounds and the uniform regression are retained, rather than selecting
only a favorable checkpoint. No model is declared generally superior or deployed.

Adaptive round two selects `pencil-0-3` three times and `pencil-0-0` once. This is a
concentration problem worth testing next: a probabilistic coverage floor does not
ensure diversity in a four-task batch. A future sampler can enforce distinct environments
and retain earlier successful cases, but it must be compared on newly reserved test
variants; these observed test results must not guide tuning against the same test set.

There is one training seed and only four test tasks from one test prescription. The
primary result is a pilot observation, not a statistically established curriculum benefit
or evidence of general folded optical engineering.

## Physical task

The field is 1 mm wide, with sources at x=54.5, 55.0 and 55.5 mm, y=50 mm, and a
±0.25° angular fan. The body is a closed C2 freeform prism with actual TIR and Fresnel
branching. Each field must deliver at least 30% energy along TIR paths, retain angular
standard deviation ≤1°, and reach its prescribed output centroid within ±0.15°.
Energy closure and unresolved-ray gates remain active. The narrow-field routing task
has no inherited 12° FOV requirement.

Targets are generated from a verified private witness. The solver sees the desired
output angles, initial prescription and measured initial errors, not the witness or
corruption location. Every selected initial task fails both coarse and dense checks;
the witness passes the 183-ray audit and 273-ray sampling check. These checks share the
same physics engine and are not independent-simulator validation.

Four previously screened archive prescriptions supply eight training tasks, four
validation tasks, and four test tasks. Siblings stay together by starting prescription.
The archives do not establish independent evolutionary ancestry, so this is a small
within-design-space experiment, not unseen-architecture generalization. No family,
task or split IDs are included in the model observation.

## What Qwen learns

Qwen selects one of eight plans: all terms, A terms, B terms, or T terms, each with
an initial coordinate-search step of 0.25 or 0.5 mm. Every plan spends 12 candidate
traces, restores its best coarse prescription, and submits for the hidden 183-ray
audit. There is no additional model-controlled verification allowance. A plan costs
846 rays including its initial trace and audit. Invalid model calls count as failures.

Numerical refinement does the surface optimization. Training changes Qwen's choice of
plan given the initial optical state. This is a bounded engineering decision module,
not evidence that Qwen independently derives optical physics.

## Learning and environment selection

All arms start from the same pinned Qwen3-8B checkpoint and LoRA initialization. Each
has two rounds, four selected training tasks per round, and eight optimizer updates
per round. Selections can contain duplicates. Each update uses a sequence padded to
4096 tokens, with loss only on the assistant's tool-call completion. The teacher is
the successful plan with lowest final audit merit among the eight candidates. Teacher
selection reads training outcomes only. Simulator output is never a target token.

- Uniform: sample eligible training environments uniformly with replacement.
- Fixed: take four consecutive environments from a predetermined cyclic ordering,
  using a seeded offset per round, with no learner feedback.
- Adaptive: place 75% probability mass on current training failures with a successful
  teacher and 25% on uniform coverage; revert to uniform if there are no failures.

This is a transparent adaptive sampler over a finite task pool. It does not learn a
proposer or invent new environments. Mixed difficulty is a heuristic; the comparison
measures whether it helps learning. All failures, selections and checkpoints remain
in the artifacts. Checkpoints are not selected using test results.

The baseline and each round are measured on training and validation tasks using the
same decoding seeds. Test evaluation starts only after all arms finish training. Three
additional decoding seeds were declared before the first test evaluation, for a final
checkpoint sensitivity check. Repeated decodes are not independent optical families.

## Cost accounting and evidence

The shared teacher cache physically traces all 16 × 8 task/plan pairs once. Every pair
is subsequently re-executed by the replay audit. Model decisions are scored against
this deterministic cache; cached retrieval is not claimed as a new simulator run.
The online-equivalent 846-ray charge is reported for each valid planner decision.

All arms are charged the same common task-generation and teacher-cache cost, plus
matched development rollout counts and training-update budgets. Actual language token
counts and training durations are retained. Physical replay is verification overhead,
reported separately from the learner's planning allowance. This pilot does not establish
that adaptive selection reduces the cost of generating new optical experience.

- Corpus: `runs/pencil_learning_20260910_v1/`.
- Models, selections and evaluations: `runs/pencil_qwen_learning_20260910_v1/`.
- Physical domain and generator: `nslens/workbench/pencil_loop.py`.
- Qwen training loop: `nslens/workbench/train_pencil_loop.py`.
- Replay and training audits: `audit_pencil_loop.py`, `audit_pencil_training.py`.
- Results and learning curves: `report_pencil_loop.py`.

Reproduction uses a fresh output directory and the local Qwen runtime already documented
in `nslens/workbench/README.md`. The corpus command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.workbench.pencil_loop --out runs/pencil_learning_new --workers 6
```

Then run `nslens.workbench.train_pencil_loop <corpus> --weights <pinned-weights>
--out <new-model-run>` in that Qwen environment. GPU 1 is used; GPU 0 is left untouched.

## Verified execution

All 128 numerical episodes replay exactly. The training audit checks all six saved
adapters, 112 primary model decisions, train-only selections and teacher success, exact
selection-probability reproduction, and task-aligned cached physical outcomes.
The workbench and new pencil-learning tests pass (26 total).

Each arm consumes 16 optimizer updates, 65,536 padded training tokens, 352 supervised
completion tokens, 40,398 development language tokens, and 20,304 online-equivalent
development rays. The common corpus uses 11,529 generation/certification rays and
108,288 teacher-cache rays. Replaying that cache costs another 108,288 rays as shared
verification overhead. Training took approximately 32–34 seconds per arm on GPU 1.
All charged model decisions use physically executed cached outcomes; the cache cost
is not hidden or treated as free data generation.

Images: `runs/pencil_qwen_learning_20260910_v1/learning_curves.png` and
`learned_pencil_repair.png`. The repair illustration selects the first successful final
adaptive test decision; it is an illustration, not an unbiased performance sample.

## Additional decoding check

Across the primary seed and three additional seeds declared before test evaluation:

| Final policy | Successful attempts across four tasks × four decoding seeds |
| --- | ---: |
| Frozen Qwen | 1/16 |
| Uniform curriculum | 5/16 |
| Fixed schedule | 14/16 |
| Adaptive curriculum | 12/16 |

The adaptive arm succeeds on the same three tasks for every decoding seed. The fixed
schedule passes all four tasks on two seeds and three on two seeds. Uniform training
shows some improvement over frozen Qwen in this check, despite its zero primary-seed
success. Thus the supported conclusion is that fixed and adaptive selection outperform
uniform sampling in this small pilot, with fixed selection strongest. Adaptive selection
has not beaten the fixed control.

These are 16 attempts, not 16 independent optical tasks. Raw native calls and all 64
final-policy outcomes were checked against the physical cache. See `sampling_summary.json`.
No further training or checkpoint selection used these results.

## Next experiment

Keep this run as a completed pilot. On fresh reserved tasks, compare a sampler enforcing
distinct environments and retaining earlier successes against the current sampler and
fixed schedule. Then let that controller propose bounded changes to the source bundle,
repair constraints and corruption, accepting only verified feasible environments. This
would advance from selecting a fixed pool toward discovering new useful environments.
Use multiple training seeds before claiming a reliable curriculum advantage.
