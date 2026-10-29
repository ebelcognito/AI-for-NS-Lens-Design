# Optical tokens and a continuous Qwen action head

This prototype changes the model computation. It never asks Qwen to write geometry
numbers as text. A trainable encoder turns numerical optical measurements into
continuous embeddings, inserts them after task text embeddings, and appends a
learned action token. Its final hidden state drives a trainable 36-dimensional
`tanh` residual head. Qwen's transformer stays frozen, but autograd traverses it
so the optical encoder can learn. There is no vocabulary decoding or LM loss.

```text
Fixed task text embeddings ──────────────────────────────┐
36 genes + step radius → global encoder → projection ────┼→ frozen Qwen → action-token state
9 field/wavelength cells → cell encoder → projection ────┤                    ↓
Learned action token ────────────────────────────────────┘             continuous head
                                                                             ↓
                                                         clip(genes + radius × action, -1, 1)
                                                                             ↓
                                                               actual ray-tracer evaluation
```

`network.py` supports the cached checkpoint's actual `qwen3_5` architecture,
including hybrid linear/full attention. The local directory and serving alias
are named `qwen3.8-27b`; provenance records the config architecture separately.
The pretrained loader uses the existing NF4 checkpoint, not the Q8 GGUF server.
It accesses `full.model.language_model` directly, bypassing vision and the LM
head. It needs a free GPU and the training environment, not the HTTP endpoint.
A small numerical-only MLP receives identical features and predicts identical
bounded actions. Its parameter count differs; this is a practical baseline,
not an architecture/parameter-count controlled scientific comparison.

## Scope of v1

- Fixed 500–1100nm spectrometer target, fixed 18mm physical semi-apertures.
- 3 × 3 field/wavelength cells, pupil grid 5, with survival, centroid, spot,
  WFE, obstruction and missing-data features. Exact ordering/scales: `state.py`.
- Observations are numerical tensors; text specifies a fixed task, so the pilot
  cannot establish natural-language controllability or multi-task generalization.
- Radius defaults to 0.05 in normalized gene units. Output is a local correction,
  not a complete prescription, a CMA controller, or a fabrication command.
- Supervised labels come from the best measured random block proposal, including
  no-op. This is a weak, stochastic teacher, not an optimal optical policy.
- Separate CMA seed families define training/validation before labels are made.
  Adjacent checkpoints stay together. Default pilot: 36 train / 12 validation
  examples from 6 / 2 families. These are development data, not a final test set.
- Selection uses the existing scalar geometric merit. WFE is observed but is
  not added to that merit. Distortion, curvature and manufacturing gates are
  not enforced by this pilot. Improving this loss is not paper reproduction.
- In closed-loop evaluation, the simulator retains only lower-loss candidates.
  Raw proposal results are also reported to expose failures hidden by that guard.
  Dense 5 × 5 × 7 audits happen after selection and do not select candidates.
- Tiny mode uses randomly initialized, reduced-size **actual Qwen hybrid layers**.
  It validates wiring and training, not pretrained Qwen performance.

## Run

The existing environment used here is
`/playpen/hisham26/qwen_cma_train_env/bin/python` (torch 2.11.0+cu130,
transformers 5.15.1). The normal `.venv` lacks torch, but can generate data.
The optional dependency group is `optical-adapter`; choose the CUDA torch build
appropriate to the machine before installing it. Nothing needs to be downloaded
when using the existing environment and local checkpoint.

```bash
.venv/bin/python -m nslens.optical_adapter.data \
  --out runs/my_adapter/data --families 8 --steps 6 --probes 12 --warmup 64

/playpen/hisham26/qwen_cma_train_env/bin/python -m nslens.optical_adapter.experiment \
  --data runs/my_adapter/data/dataset.json --out runs/my_adapter/tiny \
  --mode tiny --epochs 30

/playpen/hisham26/qwen_cma_train_env/bin/python -m nslens.optical_adapter.experiment \
  --data runs/my_adapter/data/dataset.json --out runs/my_adapter/pretrained \
  --mode pretrained --weights /playpen/hisham26/models/qwen3.8-27b-bnb4 \
  --device cuda:1 --epochs 5 --batch-size 4
```

Each experiment also trains the numerical-only baseline for the same epochs,
examples and batch size. Weights are saved at the final predeclared epoch, with
no validation selection. Outputs include adapter-only `.pt` checkpoints,
`protocol.json`, per-arm gradients/results, `summary.json` and `RESULTS.md`.
No pretrained backbone weights are duplicated. Checkpoints pin feature schema,
simulator source hash and backbone metadata; loading rejects mismatches.
The loader's GPU memory check is a conservative preflight estimate, not a guarantee
against OOM at larger sequence lengths/batches. It never stops existing workloads.

To load a trained adapter and evaluate one correction to a saved prescription:

```bash
/playpen/hisham26/qwen_cma_train_env/bin/python -m nslens.optical_adapter.predict \
  --adapter runs/my_adapter/pretrained/pretrained_qwen.pt \
  --source experiments/spectrometer_pilot_v1/seed_101/result.json \
  --out runs/my_adapter/proposal --device cuda:1
```

Use only prescriptions generated with this simulator and 18mm apertures; do not
feed the later aperture-corrected final design into this fixed-aperture prototype.
The command writes a new artifact; it never edits the source prescription.

```bash
/playpen/hisham26/qwen_cma_train_env/bin/python -m unittest discover \
  -s tests -p test_optical_adapter.py -v
.venv/bin/python -m unittest discover -s tests -p test_spectrometer_physics.py -v
```

## Research progression

First establish a real pretrained forward/backward/update and measure raw optical
outcomes against the numerical-only policy. Then enlarge independent training
families, use better local-optimizer teachers, vary requirements, and measure
held-out closed-loop quality per simulator call and total compute. The current
small fixed-target dataset cannot show that language pretraining helps optics.
Selective LoRA, flow-matching heads, surface/footprint tokens, and sensitivity
bases remain possible extensions; none is silently included in this first pilot.
