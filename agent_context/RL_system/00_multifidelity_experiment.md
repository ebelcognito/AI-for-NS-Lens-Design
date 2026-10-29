# Proxy-to-full-beam RL system: multifidelity experiment

**Status:** executable pilot completed; continue only after replication. The proposed
approach is simulator-grounded **multi-fidelity curriculum learning** with paired
low/high-fidelity residual (delta) learning, followed by verifier-guided expert iteration.
The transferable object is the shared ray-route, sensitivity, and basin structure—not
generic reasoning. Cheap labels are useful only if they predict the expensive verifier
under topology- and environment-held-out tests.

## Objective and decision rule

Test whether abundant cheap ray traces reduce the number of expensive full-beam labels
needed to rank or refine designs across FOV. At a fixed budget of 16, 32, or 64 full
labels, compare a high-fidelity-only model with a model trained on many proxy labels and
a small paired high-fidelity residual model. Do not scale to a large RL/LLM run unless
the proxy method wins on held-out full-beam ranking and per-trace-call cost.

## Fidelity ladder

All levels use the same genotype, glass, detector, focus window, and measurement code;
only pupil and field sampling change.

| Level | Rays/field | Fields | Purpose | Certification |
|---|---:|---:|---|---|
| `pencil_onaxis` | 1 | 1 (0°) | route/topology and coarse throughput | no |
| `beam_onaxis` | 17 | 1 (0°) | pupil expansion | no |
| `sparse_multifov` | 5 | 3 (-FOV, 0, +FOV) | primary cheap proxy | no |
| `full_multifov` | 17 | 9 | local authority/target | yes |

The current “full” level is the hardened local 2-D verifier, not CODE V, 3-D tolerancing,
MTF, or manufacturability. CODE V should be treated as a later, fifth fidelity.

## Dataset construction

Use certified local anchors from `runs/rl_tasks/certified_20k.json` as basin centers.
For each anchor sample normalized Gaussian perturbations at radii `0, .003, .01, .03,
.10`, plus 10% uniform designs. This supplies local validity boundaries, topology/routing
signals, field-edge failure modes, and gross invalidity without pretending synthetic
outputs are expert demonstrations.

Every row gets pencil, on-axis beam, and sparse multi-FOV labels. A frozen random subset
gets full labels; all held-out rows get full labels only for evaluation. Split by anchor
topology/cell, never by individual perturbation. The current seed-17 split has 10 training
topologies and 4 held-out topologies. There is no valid held-out environment test yet
because prism anchors are absent. Add a FOV holdout (train 4°/6°/8°, test 10°/12°) and
eventually a completely held-out prism environment.

## Experimental gates

### Gate A: proxy information

Generate 512 rows, with 64 full labels in training and full labels on held-out topologies.
Report per perturbation radius: Spearman proxy/full fitness, topology agreement and
topology-change confusion, false-safe rate, and full fitness/hard-pass precision among
the proxy top 20.

Pass if sparse multi-FOV Spearman is at least 0.5 and top-20 full fitness beats random by
at least 25%. Fail if correlation is below 0.2 or false-safe rate exceeds 50%.

### Gate B: fixed high-fidelity budget and scaling

Compare `hf_only`, `proxy_only`, `learned_proxy_delta`, and the oracle-like
`measured_proxy_delta` (cheap proxy evaluated at inference). Sweep proxy rows
`{64,128,256,409}` against full training rows `{16,32,64}` and at least five model
seeds. Report held-out-topology Spearman, MAE, top-20 realized full fitness, top-20
regret, hard-pass precision, and total engine calls.

At 32 full labels, pass transfer only if learned delta beats high-fidelity-only on both
Spearman and top-20 fitness with paired topology-bootstrap 95% intervals excluding zero.
Treat monotonic improvement from 64→128→256→409 proxy rows at fixed 32 full labels as a
scaling sign, not a scaling law. Compare against spending the same trace-call budget on
more full labels and against more best-of-N verifier calls.

### Gate C: refinement policy

Only after Gate B: pretrain a shared encoder on sparse fitness, topology, and failure-stage
labels; attach a genotype-delta action head; fine-tune on same-parent search groups and
the small full-label set. Every proposal is accepted only after full nine-field
verification. Compare verifier-call curves against CMA-ME and random local perturbation;
stop after two expert-iteration rounds without improvement.

### Gate D: CODE V bridge

Sample a stratified cohort over topology, FOV, proxy/full disagreement, and perturbation
radius. Fit `CODEV_merit = local_full_prediction + learned_delta`, then actively request
cases with high uncertainty times predicted utility. Keep local and CODE V labels separate;
never call local full-beam physical truth.

## Pilot evidence (v1)

The pilot generated 512 rows: 410 training and 102 held-out-topology rows. It produced
512 labels at each cheap fidelity and 166 full labels (64 train, 102 test), using 4,054
nominal engine calls in 66.7 seconds; 82 full hard passes were observed.

| Proxy | Held-out Spearman | Full fitness of top 20 | Random | Lift |
|---|---:|---:|---:|---:|
| Pencil/on-axis | 0.461 | 0.328 | 0.346 | -5% |
| Beam/on-axis | 0.772 | 0.655 | 0.346 | +89% |
| Sparse multi-FOV | **0.946** | **0.842** | 0.346 | **+143%** |

Sparse had 2 false-safe rows among 37 proxy-safe rows (5.4%). Gate A therefore passes
for sparse multi-FOV; pencil alone does not. The sparse correlation dipped at perturbation
radius .03, showing that proxy quality is basin/radius dependent.

The learned-delta baseline is weaker than the raw measured proxy but improves with proxy
data at fixed 32 full labels:

| Proxy rows | HF rows | HF-only Spearman/top20 | Learned-delta Spearman/top20 |
|---:|---:|---:|---:|
| 64 | 32 | 0.212 / 0.361 | 0.235 / 0.316 |
| 128 | 32 | 0.212 / 0.361 | 0.272 / 0.330 |
| 256 | 32 | 0.212 / 0.361 | 0.442 / 0.425 |
| 409 | 32 | 0.212 / 0.361 | **0.523 / 0.447** |

This is a promising scaling direction, not a formal Gate-B pass: the artifact has means
and standard deviations, but no paired bootstrap confidence intervals. The oracle
measured-proxy delta is about 0.88 Spearman and 0.83 top-20 fitness, an upper bound only.

## Caveats and interpretation

- The dataset is anchor perturbations plus 10% uniform samples, not 10,000 independent
  expert designs; the result tests local basin/routing transfer.
- Strong raw proxy correlation does not guarantee learned representation transfer.
- Pencil success with sparse failure means add edge-field tasks; sparse success with
  learned-delta failure means use the proxy explicitly at search time rather than claim
  pretraining transfer.
- Local success followed by CODE V failure indicates a missing-physics/fidelity problem,
  not necessarily a model-scale problem.
- A monotone verified best-of-N curve is stronger evidence for useful inference scaling
  than training loss alone.

## Exact next experiment

Run a preregistered Gate-B replication with **five independent data seeds and five model
seeds**. For each data seed, generate 512 rows and evaluate the complete grid
`{proxy=64,128,256,409} × {HF=16,32,64}`, preserving topology-held-out splits. Use paired
bootstrap over held-out topologies for learned-delta minus HF-only differences in
Spearman and top-20 full fitness. Require both 95% intervals to exclude zero at HF=32,
then compare total engine calls with buying additional full labels. If it passes, run
Gate C; if the learned replication fails while the measured-proxy upper bound remains
strong, make sparse evaluation an explicit search-time acquisition tool.

## Reproduction commands and pointers

```bash
python3 -m nslens.rl.data.build_multifidelity \
  --certified runs/rl_tasks/certified_20k.json \
  --n 512 --hf-train 64 --seed 17 \
  --out runs/multifidelity_pilot_v1

python3 -m nslens.learning.multifidelity_baseline \
  --data runs/multifidelity_pilot_v1/traces.jsonl \
  --proxy-sizes 64,128,256,409 --hf-sizes 16,32,64 \
  --model-seeds 5 --out runs/multifidelity_pilot_v1/baseline.json

python3 tests/test_rl_multifidelity.py
```

Implementation: `nslens/rl/multifidelity.py`,
`nslens/rl/data/build_multifidelity.py`, and
`nslens/learning/multifidelity_baseline.py`.

Results: `runs/multifidelity_pilot_v1/README.md`,
`runs/multifidelity_pilot_v1/baseline_cv.json`, and
`runs/multifidelity_pilot_v1/manifest.json`.
