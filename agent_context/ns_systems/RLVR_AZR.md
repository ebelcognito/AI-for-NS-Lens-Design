# Resuming AZR for optical design — feasibility assessment, 2026-09-11

**Superseded direction:** the user has ruled out further use of Qwen3-8B. The active
plan is numerical discovery with the new Qwen guiding compaction/refinement; see
[proposer assessment](proposer_assessment_20260911.md). Smaller-model recommendations
below are historical and are not part of the active plan.

## Decision

A bounded, simulator-verified learning loop is feasible with the existing project.
Full AZR with a learned proposer and solver is not implemented. An 8–10% gain is a
reasonable experimental target, not a prediction. Start with folded pencil-beam repair;
the current evidence does not extend to general 3-D lens or topology discovery.

## What is already available

- `start_qwen.sh` launches llama.cpp with the local `Qwen3.8-27B-Q8_0.gguf`,
  alias `qwen3.8-27b`, GPU 1, 32k context, and thinking enabled. These are local
  file/launcher identifiers; upstream provenance and exact architecture were not
  independently established in this assessment. The user described it as 28B.
- The chat UI shares a server with `/v1/chat/completions`; automation needs no browser.
  The installed llama.cpp server documentation describes tool calls and separate
  `reasoning_content`. Live model/tool compatibility remains to be tested.
- At inspection, port 8080 was not listening. Two RTX 6000 Ada GPUs each reported
  49,140 MiB; GPU 0 used 47,540 MiB and GPU 1 used 4 MiB. Treat GPU 1 as the
  available resource, not both cards. The model file occupies about 28 GiB on disk.
- `pencil_loop.py` provides bounded repairs, eight numerical plans, private witnesses,
  physical verification, family splits, and replayable measured outcomes.
- `train_pencil_loop.py` already implements actual Qwen3-8B LoRA updates with fixed,
  uniform, and adaptive finite-pool curricula. It is supervised plan selection, not RL.

The completed pilot improved primary test success from 0/4 to 3/4 for both fixed and
adaptive curricula. Across four decoding seeds, fixed achieved 14/16 attempts and
adaptive 12/16, versus frozen 1/16. Those remain four optical tasks from one test
prescription, not sixteen independent problems. Adaptive selection has not beaten
the fixed control. See [pilot evidence](pencil_qwen_learning_results.md).

## What AZR would add

The [official AZR implementation](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
alternates proposing executable tasks and solving them, with verifiable rewards and
updates to both roles. Its published application is code reasoning; transferring the
mechanism to optical simulation is a research proposal, not an established result.

The optical equivalent is:

1. Propose bounded changes to a training prescription, source bundle, repair constraints,
   or target. Keep simulator rules and acceptance thresholds under evaluator control.
2. Certify feasibility with a private witness and dense checks; reject invalid,
   trivial, duplicate, or uncertified tasks. Charge rejected proposals too.
3. Attempt repair using the current solver under a fixed trace and language budget.
4. Update the solver from verified experience. Initially use the working LoRA
   supervised-learning route; call this an AZR-inspired precursor.
5. Reward a future learned proposer for useful training tasks, using train/development
   feedback, coverage, and downstream learning progress. Intermediate solver success
   can be a difficulty heuristic, but is not proof of learnability.
6. Repeat, evaluating locked checkpoints on reserved families only after selection.

Deduction can mean predicting optical outputs from geometry; abduction can mean
recovering a feasible geometry from requested outputs; induction can mean predicting
responses to unseen perturbations from measured examples. Begin with abduction/repair,
where the executable environment exists. These are proposed optical analogues.

## Missing pieces and hardware implications

- API transport: existing runners use local Transformers or Responses. Add and test a
  Chat Completions adapter, including tools, malformed/truncated responses, token usage,
  timeouts, and thinking. Never interpret reasoning text as executable tool calls.
- Thinking budget: the pencil runner currently allows only 128 output tokens and uses
  `enable_thinking=False`. It cannot be reused unchanged for the new reasoning model.
  Set explicit output/context/cumulative limits and record reasoning cost and latency.
  A 32k context window is distinct from cumulative tokens across an episode.
- Weight updates: the GGUF server does inference; it is not a training pipeline.
  Existing Transformers/PEFT code cannot train this GGUF directly. Larger-model LoRA
  needs compatible training weights, tokenizer/template, architecture support, and a
  measured forward/backward memory test. An 8B adapter cannot attach to this model.
- At roughly 27B parameters, BF16 weights alone would require about 54 GB before
  activations or optimizer state, exceeding one available GPU. Quantized adapter
  training may be possible, but fitting inference does not establish training fit.
  Long thinking rollouts make online RL substantially more expensive.
- Practical route: use larger Qwen as a frozen proposer/teacher and train the existing
  smaller planner, alternating inference and training phases if necessary. This can
  improve the optical system; it does not update the larger Qwen's weights or reproduce
  same-model AZR. Test larger-model training separately before committing to full RL.
- Benchmark: fresh prescription families and broader feasible tasks are needed. The
  old tests are now development evidence. Shared simulator checks validate sampling
  consistency, not agreement with an independent physical engine.

## Staged experiment and acceptance rule

1. **Integration smoke test:** call the local API, then execute a real bounded repair
   on development data. Verify final tool output, physical replay, token accounting,
   and that thinking reaches a usable decision within budget.
2. **Headroom test:** evaluate frozen larger Qwen, the strongest fixed numerical plan
   selected on training/development data, and a simple learned plan selector. Evaluate
   all eight plans privately to estimate the restricted action space's oracle ceiling.
   Report that enumeration cost separately; the solver must not receive the cache.
   If the fixed strategy already saturates success, use harder certified development
   tasks or a predeclared quality/efficiency metric before locking the test set.
3. **Small learning experiment:** compare fixed, uniform, and generated/adaptive
   curricula with matched training updates and total data-generation/rollout budgets.
   Enforce distinct tasks and replay earlier successes. Use at least three training
   seeds for an initial robustness check. Compare the same learner before/after;
   larger-versus-smaller model differences are not a curriculum gain.
4. **Scale only after a gain:** add a learned proposer and on-policy verified-reward
   updates once solver learning, task headroom, and resource cost are demonstrated.

Default primary target: **8–10% relative increase in verified repair success**, with
equal per-task candidate traces and explicit language limits. For example, 50% to
55% is 10% relative improvement and 5 percentage points. Also report optical merit,
physical failures, rays, language tokens, wall time, and total training/generation cost.
If success is near its ceiling, predeclare an alternative such as 10% fewer traces
to reach the same acceptance threshold; do not switch metrics after seeing test results.

Start with 20–30 fresh development families to estimate headroom, variability, runtime,
and paired disagreements. Determine confirmatory test size from that pilot; this is
not a claim that 20–30 families can establish a small effect. Reserve test families
before generating siblings, track shared ancestry, and use paired family-level confidence
intervals. Repeated decodes do not increase the count of independent families. Report
uncertainty even if the point estimate reaches the target; a gain against frozen Qwen
alone is insufficient evidence that the approach improves on the numerical baseline.

This assessment changed documentation only. No new inference, training, or performance
experiment was executed, and no improvement is claimed for the new model yet.
