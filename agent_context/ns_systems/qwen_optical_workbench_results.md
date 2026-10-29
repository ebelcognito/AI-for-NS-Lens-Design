# Qwen optical workbench experiment — 2026-09-10

A real local Qwen3-8B baseline now operates the optical tools. The untrained model solves
12/12 held-out repairs when given a sufficient language-token allowance. This is an
8.2-billion-parameter language model choosing tools; the numerical optimizer behind the
`optimize` tool still performs the coefficient search. It is separate from the earlier
small learned-Jacobian model.

## Baseline evidence

| Frozen Qwen setting | Submitted and verified | Final optics pass, regardless of submission | Mean candidate traces | Mean tokens |
|---|---:|---:|---:|---:|
| 24k cumulative token ceiling | 0/12 | 12/12 | 21 | 18,529.5 |
| 64k cumulative token ceiling | 12/12 | 12/12 | 21 | 30,212.5 |

Every episode allows 24 candidate traces, 16 actions, and two dense verifications. The
separate hidden acceptance audit has 279 rays. Both baseline settings produce the same
final coefficients and mean audit RMS of 0.0025956946 mm. Qwen uses both sides and all
polynomial orders, optimizes, verifies twice, and then submits if tokens remain.

The 24k setting was a poor framework default: **all 54 successful teacher transcripts
need more than 24k cumulative tokens** in Qwen's native format (24,046–47,182 including
target completions). A validation smoke run exposed an additional implementation bug:
the original runner checked previous usage without reserving the next input. Local
transports now tokenize before dispatch and reserve that input cost. The 24k test run
uses this corrected hard cap. Its last request is rejected before generation, so actual
usage remains below the ceiling. The 64k control was added after inspecting this failure;
these are exploratory pilot measurements, not an untouched confirmatory benchmark.

There are **12 held-out tasks from four families**, not 144 independent test problems.
The original corpus has 36 tasks × four scripted plans = 144 episodes. No test-family
examples, hidden witnesses, or private corruption labels are provided for training.

## Model and runtime

- Model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0.
- Pinned revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- BF16 weights, native Qwen tool template, non-thinking mode, temperature 0.7,
  top-p 0.8, top-k 20, seed 20260910 plus task index.
- GPU 1 only; GPU 0's existing workload was left running.
- Weights: `/data/hisham26/optics_models/hub/`; separate added dependencies:
  `/data/hisham26/optics_models/runtime/`.
- Python: `/home/hisham26/Desktop/simu1/deeplens_env/bin/python`;
  torch 2.5.1+cu121, Transformers 4.57.1.

Native `<tool_call>` blocks become simulator calls. Malformed generations are retained
and never partially executed. Every evaluation stores raw outputs, actual tools,
measurements, token counts, and independent acceptance results. Replaying all 24 baseline
episodes reproduces all intermediate states and final results exactly. Replaying all
144 original teacher episodes also passes. The historical corpus source manifest differs
only in `model_baseline.py`, where the token reservation fix was made; every physics
source and historical snapshot remains unchanged.

## Training experiment

One LoRA pass is configured over all 54 verified successful training conversations:
378 assistant-call examples from six families. Training checks each conversation against
its actually executed teacher episode, masks every prompt/tool-result token from loss,
preserves the complete native prefix, and refuses silent truncation. No reasoning
narratives are fabricated. Configuration: rank 8, alpha 16, q/v attention projections,
learning rate 1e-4, microbatch 1, accumulation 4, one epoch, fixed seed. Final checkpoint
will be evaluated with the same 64k setting and optical budgets as the frozen control.

The mixed-teacher pass completed in 822 seconds, updating 3,833,856 LoRA parameters.
All 12 trained episodes replay exactly. Its acceptance check **rejects** the adapter:

| Original test families, same 64k ceiling | Frozen | Mixed-teacher LoRA |
|---|---:|---:|
| Verified submissions | 12/12 | 10/12 |
| Mean audit RMS | 2.596 µm | 15.238 µm |
| Mean candidate traces | 21.0 | 22.5 |
| Mean dense verifications | 2 | 1 |
| Mean total ray evaluations | 847 | 750.5 |
| Mean language tokens | 30,212.5 | 24,189.5 |

The adapter reduces redundant dense verification but chooses exit-only optimization on
several cases. Two cases in family 001 then fail at 0.11519 and 0.03976 mm RMS. These are
real planning regressions, not parser failures. Pooling successful demonstrations from
all plans did not reliably teach when each plan is appropriate.

A controlled follow-up selects a single teacher using **only all training-family
results**: maximize success rate, then minimize worst-case audit RMS, then mean RMS.
This selects `all_terms` (18/18 training successes; worst RMS 0.003810 mm), ahead of
`staged_order` (18/18; worst RMS 0.021927 mm) and both one-sided plans. The new pass starts
from the original Qwen base and uses 18 verified conversations, 108 assistant targets,
one epoch, and the same optimizer settings. It is a smaller-data teacher-selection
ablation, not a matched-update-count causal experiment. A new seed, 20260913, supplies
fresh test families for the paired frozen/adapter comparison. No results from these
new families select teachers or checkpoints.

**Scope correction:** this follow-up adapter finished training, but its evaluation
is parked. Neither frozen nor adapted Qwen has been evaluated on those fresh families.
The user's priority is folded freeform engineering; see
[the folded workbench results](folded_qwen_workbench.md). The unfolded experiments
above are interface/training diagnostics, not evidence of general optical engineering.

## What remains unproven

The all-terms numerical plan already solves every task. Successful tool use therefore
does not establish optical diagnosis, general 3-D freeform design, or an AZR curriculum
advantage. Useful next tasks must require meaningful choices—such as which surface can
be manufactured or changed—and must defeat a fixed universal plan at the same budget.
Training should then be accepted only for gains on isolated families, with failures and
all numerical probes retained.

## Artifacts

Paths below are relative to the repository root:

- `runs/qwen3_8b_workbench_test_20260910_v1/`: hard-24k baseline and corpus replay audit.
- `runs/qwen3_8b_workbench_test_64k_20260910_v1/`: 64k baseline and budget comparison.
- `runs/qwen3_8b_workbench_lora_20260910_v1/`: training configuration, log, and adapter.
- `runs/qwen3_8b_workbench_comparison_20260910_v1/`: model-file SHA256 manifest and comparison.
- `nslens/workbench/qwen_baseline.py`: local inference, parsing, token counting, physical replay.
- `nslens/workbench/train_qwen.py`: train-only, assistant-target LoRA pass.
- `nslens/workbench/compare_qwen.py`: verified outcome/cost summaries and comparability checks.

## PNG solutions

The PNGs are generated by Matplotlib from the actual prescriptions and ray tracer,
not an image-generation model. Each detailed plot re-traces both initial and final
prescriptions on the identical 279-ray audit grid, verifies that the final result
matches its recorded outcome, and labels the different before/after vertical scales.
Ray layouts show 27 primary rays; ghost branches are omitted only from the drawing.
The selection is explicitly for visual interest, not an unbiased performance sample.

See `runs/qwen3_8b_workbench_comparison_20260910_v1/png_baseline/` and `png_lora/` for
three detailed PNGs and a gallery for each model. `training_effect.png` shows every
original test case, including both regressions.
