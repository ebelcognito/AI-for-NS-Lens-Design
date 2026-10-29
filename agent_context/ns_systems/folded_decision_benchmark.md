# Folded decision benchmark — development gate, 2026-09-10

This is the first experiment toward the constrained abduction benchmark described in
the user's AZR discussion. It tests the task generator and decision sensitivity before
training a proposer or solver. It is not a self-play result.

## Archive screening

`nslens.workbench.folded_screen` screened the top 12 cells by original fitness from
each of the freeform and multifold prism archives. Every candidate was converted using
the existing endpoint-preserving closed spline representation. Coarse-pass candidates
were checked on the 183-ray audit and 273-ray grid. Only freeform cell (8,23), the
previously used seed, passed all stages. A coarse failure was not subsequently tested
at higher fidelity; these are screening outcomes, not exhaustive impossibility proofs.

Among 24 candidates, six failed the engine's supported refractive-index interval
[1.50,1.70], and two converted boundaries self-intersected. Those are representation
or engine-support failures. Other failures include inadequate delivered energy,
excess angular spread, or insufficient field of view. No requirement was relaxed to
admit an archive candidate. Archive fold labels alone do not certify useful multi-fold
energy delivery. Existing archive cells also lack sufficient ancestry evidence to
justify treating them as independent training/test families.

Full candidate prescriptions, archive hashes, verdicts and failed checks are in
`runs/folded_screen_20260910_v1/screen.json`.

## Constrained task generation

The surviving seed is perturbed by ±0.75 mm on A, B, or T second-order coefficients,
subject to the original ±2 mm bounds. A generated task must initially fail the audit
while remaining physically valid. A private original prescription must still pass
both audit grids and satisfy all access and change constraints. The witness establishes
feasibility; returning the original coefficients is not required for success.

Only the A corruptions survived this filter. Each has two surface-access variants:
A terms only, or all terms editable. This produces four development tasks from one
ancestor and only two distinct initial prescriptions. The task generator has therefore
**not yet achieved defect-location or architecture diversity**.

The tool now enforces per-task editable coefficients, maximum departure from the start,
and candidate budget. This pilot uses a 1 mm change bound and 12 candidate traces.
Illegal selections and out-of-bound perturbations are rejected before tracing;
coordinate trials are clamped to permitted bounds and charged even if rejected.
The best-prescription cache also excludes physically invalid candidates.

Public tasks are `runs/folded_decisions_20260910_v1/tasks.json`.
Private witnesses and generation checks are separate sibling files and are not passed
to the model. This separation prevents prompt leakage; it is not a security boundary
against a process with arbitrary filesystem access.

## Comparison protocol

Four numerical plans use the same 12 candidate traces at a fixed 0.25 mm initial step:
all permitted surfaces, A first, B first, or T first. A single-face plan falls back to
the first available face when its preferred face is locked. Thus the restricted tasks
intentionally produce identical numerical plans; they must not be counted as four
independent experiments.

Frozen Qwen3-8B receives the task and folded tools, with a 128k cumulative language
ceiling and the same candidate cap. It may use up to two 99-ray checks; numerical
plans use neither. Candidate ceilings are matched, but consumed total tracing cost
is not necessarily equal. Every episode has a 183-ray final audit and exact replay.
Numerical episode cost is 846 rays including the initial trace and final audit;
Qwen's extra verifications are reported separately.

After seeing Qwen choose a 0.5 mm step on its first repair, an additional fixed
all-permitted 0.5 mm control was added. This is explicitly a post-hoc development
control, not a preregistered test. It checks whether apparent model gains reduce to
optimizer step size. Neither this set nor its results should select a checkpoint and
then be reused as a held-out test.

## Completed results

| Strategy | Audited successes | Candidate traces per task | Total rays per task |
| --- | ---: | ---: | ---: |
| All permitted terms, fixed 0.25 mm step | 2/4 | 12 | 846 |
| A-first, fixed 0.25 mm step | 2/4 | 12 | 846 |
| B-first, fixed 0.25 mm step | 1/4 | 12 | 846 |
| T-first, fixed 0.25 mm step | 1/4 | 12 | 846 |
| Frozen Qwen3-8B | 3/4 | 12 | 1,044 |
| All permitted terms, fixed 0.5 mm step (post-hoc) | **4/4** | 12 | 846 |

Qwen used two extra verifications on every task. It made five invalid requests in
all: four locked-surface selections and one optimizer request after exhausting the
candidate budget. Its fourth submitted design failed the optical audit. Language
usage ranged from 28,091 to 52,504 cumulative tokens per task.

At 0.25 mm, two tasks have different outcomes depending on surface selection. However,
a single 0.5 mm all-permitted plan solves the entire development set. This defeats the
claim that these tasks require adaptive engineering decisions. Qwen has not shown an
advantage, even though it beats the weaker original all-permitted baseline.

All 24 newly executed episodes (16 original numerical controls, four Qwen, four
step-size controls) replay exactly. All 21 workbench tests pass, including new checks
for surface locks, change bounds, budget enforcement, malformed constraints, and
physically invalid best-candidate rejection. Both older 128k Qwen folded episodes also
replay under the updated tools. No model was trained or promoted.

**Decision: the benchmark is not ready for a learning claim.** Retain it as a regression
and tool-enforcement fixture, and address certified architecture diversity before
expanding the dataset or starting self-play. Archive-derived variants must not be
presented as independent families.

## Artifacts and reproduction

- `nslens/workbench/folded_screen.py`: archive screening.
- `nslens/workbench/folded_decisions.py`: witness-backed task generation and numerical controls.
- `nslens/workbench/folded_step_control.py`: post-hoc step-size control.
- `nslens/workbench/summarize_folded_decisions.py`: task-aligned model comparison.
- `runs/folded_decisions_20260910_v1/`: numerical transcripts and task generation evidence.
- `runs/folded_decisions_qwen_20260910_v1/`: Qwen responses, events, costs and audits.
- `runs/folded_decisions_step_control_20260910_v1/`: step-size control transcripts.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.workbench.folded_screen --out runs/folded_screen_new --limit 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.workbench.folded_decisions runs/folded_screen_new/screen.json \
  --out runs/folded_decisions_new
```

## Next gate

Do not fine-tune on this four-task set and call it engineering progress. First resolve
representation fidelity for promising archived multi-fold designs, obtain multiple
certified starting geometries, and generate failures on different editable surfaces.
Then freeze family groups and compare robust numerical plans across several step sizes
and allocation strategies. A learned proposer becomes useful only once this task space
contains diverse, feasible challenges that are sensitive to decisions.
