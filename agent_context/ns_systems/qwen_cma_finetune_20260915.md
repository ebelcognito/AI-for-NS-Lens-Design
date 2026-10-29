# First current-Qwen CMA fine-tuning pilot — 2026-09-15

User requested trying fine tuning after the frozen-controller runs. Prepared and
launched a small supervised QLoRA experiment with unattended validation scoring.
Do not claim completed training or improvement until the corresponding artifacts exist.

## Model and environment

- Current model remains Qwen3.8-27B; no substitution with the old Qwen3-8B.
- GGUF metadata identifies Qwen/Qwen3.8-27B as the upstream base and Unsloth as
  quantizer. Training uses `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, pinned revision
  `8aa5f05d26b7205477066e1449e0af13f762a299`, downloaded to
  `/playpen/hisham26/models/qwen3.8-27b-bnb4` (about 22.3 GB).
- This NF4 checkpoint differs in quantization from the historical Q8 GGUF.
  Frozen and trained policies in this experiment share the NF4 checkpoint and prompts.
- Dedicated environment `/playpen/hisham26/qwen_cma_train_env`; reads existing
  optifusion site-packages for torch 2.11.0+cu130 through a .pth file, without
  modifying that environment. Installed transformers 5.15.1, peft 0.21.0,
  bitsandbytes 0.50.2, accelerate 1.15.0, safetensors 0.8.0 in the new environment.
- GPU 1 only; GPU 0 was already occupied. BF16 autocast, BF16 embedding/output
  heads, gradient checkpointing, and expandable CUDA allocation segments.
- Initial smoke attempts exposed a tokenizer return-format mismatch, then FP32
  attention memory exhaustion. Both were fixed before any optimizer update.
- Longest training example completed a real finite forward/backward test with
  peak allocated memory 38,293,661,696 bytes (~35.7 GiB). See memory_smoke.json.

## Data and protocol

- Sources: first 96-case midpoint pilot and new 65-case failing-midpoint run.
  Original complete two-action trajectory preference pairs are excluded.
- One supervised action per state: highest measured reward among actual arms,
  preferring fixed continuation on ties. Single-seed label noise remains.
- 76 training states: 38 informative outcome-spread states and 38 all-pass
  unchanged-continuation anchors. No invented optical reasoning; rationale is empty.
- Training targets are all continuations (step multipliers). This dataset does
  not teach successful restart selection, although the action interface allows it.
- 60 validation states from eight task settings excluded from training. These
  are development-validation settings, not an untouched test set. Related tasks
  share ancestry; family-level grouping uses task-setting labels conservatively.
- Data checks passed: no shared train/validation setting or observation, valid
  targets, empty rationales, and allowlisted numerical observations.
- Rank 8, alpha 16, q_proj/v_proj adapters: 3,014,656 trainable parameters.
  Three epochs, learning rate 2e-5, accumulation 4, seed 1515; 57 intended updates.
- Only assistant action tokens contribute to loss. Full observations retained;
  no silent truncation, maximum recorded sequence 5,623 tokens.
- Adapter-disabled baseline is generated before training. Frozen and trained
  validation use identical prompts, disabled thinking, greedy decoding, 256-token cap.
- Final epoch is predeclared; checkpoints are not selected using validation outcomes.
- After training and model generation, the numerical evaluator compares frozen,
  trained, fixed, half-step, and double-step controls using identical midpoints,
  16 further candidates, dense audits, and doubled branch-capacity agreement.

## Live status and continuation

Root: `/playpen/hisham26/qwen_cma_finetune_20260915`.

- Trainer PID at handoff: 2820052. Detached supervisor PID: 2822454.
- `status.json`: detailed pipeline stage. At handoff it was collecting frozen actions.
- `supervisor_status.json`: unattended lifecycle and error report.
- `train.log`, `training_log.json`, `memory_smoke.json`, `training_protocol.json`.
- `adapter_step_1/`, epoch adapters, `adapter_final/` appear after actual updates.
- `adapter_update.json` certifies a nonzero parameter change after training.
- `model_evaluation/`, `evaluation_actions.json`, `simulator_evaluation/`.
- Final `summary.json` and `RESULTS.md` are written only after simulator evaluation.
- The supervisor waits for the existing trainer, resumes from the base if it was
  interrupted without a reported error, then automatically runs numerical validation.
  A reported trainer error stops the pipeline and is not silently ignored.
- Executable snapshot is under `code/nslens`, hashes in `source_hashes.json`.
  Do not start another trainer while this supervisor is active.

Implementations: `prepare_cma_finetune.py`, `train_cma_adapter.py`, and
`nslens/analysis/evaluate_cma_finetune.py`. This is a small action-supervision pilot,
not RL, a learned proposer, full AZR, or established optical improvement.

Model source: https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit
Upstream config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
