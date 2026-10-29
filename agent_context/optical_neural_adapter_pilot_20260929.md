# First architectural Qwen optical adapter — 2026-09-29

Implemented and trained a continuous optical-state encoder and residual action
head around the actual cached pretrained Qwen text backbone. This is an
architectural adapter, not JSON action fine-tuning or an external tool wrapper.

## Architecture and implementation

`nslens/optical_adapter/network.py` implements:

1. An MLP encodes each of nine field/wavelength measurement cells. A second MLP
   encodes 36 current normalized geometry parameters and a step radius.
2. Learned projections map these features into Qwen's 5,120-dimensional embedding
   space. They follow fixed task text embeddings and precede a learned action token.
3. The frozen Qwen transformer processes this sequence with gradients enabled
   through its operations. No backbone parameter enters the optimizer.
4. The final action-token hidden state feeds LayerNorm/MLP/tanh, predicting all
   36 bounded geometry corrections in one forward pass. No vocabulary logits,
   autoregressive number generation or token parsing are involved.
5. The spectrometer simulator evaluates the proposal. Closed-loop rollout retains
   it only when the existing coarse scalar loss improves. Dense audit follows
   selection; this loss guard does not establish physical feasibility.

The trained optical adapter has **1,022,500 trainable parameters**. The serving
alias/local checkpoint directory says Qwen3.8-27B; the checkpoint config identifies
`qwen3_5` / `Qwen3_5ForConditionalGeneration` and hybrid linear/full attention.
Training used `/playpen/hisham26/models/qwen3.8-27b-bnb4` (NF4), rather than the
HTTP server's Q8 GGUF. No model download was needed.

As authorized, the existing Qwen server PID 3578446 and paused optical runner
PID 4040577 were stopped gracefully to release GPU 1. GPU 0's workload was left
running. The chat server was not replaced with this adapter. The new adapter has
its own training and prediction commands.

## Executed pilot

Artifacts: `runs/optical_adapter_v1/`.

- Dataset: 48 states, 8 independent CMA seed families; 36 states from 6 families
  train, 12 from 2 families validate. Split made before label generation.
- Target: one fixed spectrometer setup, 18mm physical surface semi-apertures.
- Teacher: best of 12 bounded random block proposals, including no-op; weak local
  correction labels, not optimal geometry or a CMA-reproduction demonstration.
- Cost: 1,088 objective calls + 48 observation traces = **1,136 forward traces**.
- Numerical-only MLP and pretrained-Qwen adapter: 5 epochs, batch size 4, learning
  rate 0.001, seed 17, final epoch selected in advance. No validation updates.
- An earlier separate tiny random-Qwen architecture test used 30 epochs. Its
  measurements should not be pooled with the five-epoch pretrained comparison.

| Five-epoch arm | Improving raw actions | Mean raw loss change | Mean guarded loss change |
|---|---:|---:|---:|
| Numerical-only MLP | 6 / 12 | -0.075837 | -0.105042 |
| Pretrained Qwen + optical adapter | 9 / 12 | +0.048771 | -0.045703 |
| One random action per state | 1 / 12 | See per-state JSON | Not the teacher |

Negative loss change is better. Qwen improves more individual cases, but a few
regressions outweigh those improvements. The numerical-only baseline has the
better mean raw and guarded loss change. With only two validation families and
correlated states inside each, these are development diagnostics, not statistical
evidence of general superiority. The numerical model is also much smaller; this
is not a parameter-count controlled ablation.

The pretrained adapter's training loss fell from 0.10730 to 0.07712; measured
squared adapter weight change was 40.6081. First-backward squared gradient sums
were 0.13937 for the encoder and 0.74768 for the head, with **no backbone gradients**.
This establishes a real trainable optical input/output path through pretrained
Qwen. It does not prove that pretraining contributes useful optical knowledge.
Training/setup time was about 183 seconds, including checkpoint loading.

Four-step closed-loop checks ran from each validation family's first state. Dense
post-selection audits remain far from useful spectrometer performance: the two
starts have only about 49% and 6% ray survival and hundreds of waves of worst-cell
WFE. This dataset exercises early search/repair behavior, not high-quality local
refinement. Neither trained policy reproduces the paper.

## Validation and artifacts

- Four new adapter tests pass in the ML environment: real hybrid-Qwen encoder
  gradients, frozen weights, state sensitivity, bounded corrections, exact tiny
  checkpoint reload, provenance rejection, and ancestry leakage rejection.
- Existing 12 spectrometer physics tests pass. The ordinary `.venv` runs the two
  non-neural adapter tests and skips the two requiring optional torch/transformers.
- A fresh-process pretrained checkpoint reload and proposal is exercised by
  `nslens.optical_adapter.predict`; see `runs/optical_adapter_v1/reloaded_pretrained/`.
- Checkpoints: `runs/optical_adapter_v1/pretrained/pretrained_qwen.pt` and
  `runs/optical_adapter_v1/pretrained/numeric.pt`. Only adapter tensors are saved.
- Measurements: `pretrained/{RESULTS.md,summary.json,pretrained_qwen.json}`.
- The pilot began before schema manifests were added to checkpoint serialization.
  `checkpoint_packaging.json` records attaching final schema/simulator provenance
  to those files, preserving originals and verifying unchanged tensor hashes.
  Future runs write manifests and code snapshots directly.

Commands, limitations and feature definitions are documented in
`nslens/optical_adapter/README.md`. Model features and targets currently describe
one fixed objective; arbitrary natural-language goals are not supported yet.
The next research step is stronger optimizer-derived correction data across many
independent families, including useful high-throughput states, followed by a
predeclared comparison against the numerical-only policy. Selective LoRA and
multimodal action distributions remain unimplemented extensions.
