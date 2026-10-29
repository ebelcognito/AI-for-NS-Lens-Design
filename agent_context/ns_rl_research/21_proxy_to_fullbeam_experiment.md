# Proxy-to-full-beam, multi-FOV experiment

**Status:** executable pilot scaffold implemented 2026-09-07.  This plan is a sequence of
falsifiable gates; it is not a claim that proxy data automatically transfers optical-design
expertise.

## 1. Question and decision

Can abundant, cheap ray-trace tasks teach a model enough route and basin structure that it
needs materially fewer expensive full-beam labels to rank or refine designs across FOV?

The first decision is narrower than “train the final model.”  At a fixed budget of 16, 32,
or 64 full local labels, does proxy supervision improve held-out full-beam ranking over an
otherwise identical high-fidelity-only baseline?  If not, do not scale an LLM/RL run.

The experiment tests the method usually called **multi-fidelity curriculum learning**.  The
specific construction is **paired low/high-fidelity residual (delta) learning**, followed—if
it works—by **verifier-guided expert iteration**.  It is not pure sci-fi: the transferable
object is not generic “reasoning,” but the ray-route, sensitivity, and basin structure shared
by the cheap and expensive versions of this simulator.

## 2. Fidelity ladder

All levels evaluate the same genotype, glass, detector, focus window, and measurement code.
Only pupil and field sampling change.

| label | rays/field | fields | role | may certify a design? |
|---|---:|---:|---|---|
| `pencil_onaxis` | 1 | 1 (0°) | route/topology and coarse throughput | no |
| `beam_onaxis` | 17 | 1 (0°) | pupil expansion | no |
| `sparse_multifov` | 5 | 3 (-FOV, 0, +FOV) | cheap cross-field proxy | no |
| `full_multifov` | 17 | 9 | local authority/target | yes |

This ladder exposes two possible failures separately: the chief ray may not predict the
full pupil, and on-axis quality may not predict field-edge collapse.  The sparse 3-field
view is the primary training proxy because it contains both signals at modest cost.

“Full” here means the repository's hardened 2D local verifier.  It does **not** mean CODE V,
3D tolerancing, MTF, manufacturability, or the final physical authority.  CODE V becomes a
fifth fidelity only after the local transfer gate passes.

## 3. Data without 10,000 expert designs

Use the 153 certified local anchors in `runs/rl_tasks/certified_20k.json` as basin centers.
They cover 14 fold topologies and FOV half-angles 4°, 6°, 8°, 10°, and 12°.  They do not yet
contain prism designs.

For each anchor, sample normalized Gaussian perturbations at radii
`0, .003, .01, .03, .10`, plus 10% uniform designs.  This creates supervision about:

1. invariants close to a valid design;
2. where topology changes;
3. where pupil or field-edge performance collapses;
4. gross invalidity away from known basins.

Every training sample receives the three cheap labels.  Only a random, frozen subset
receives the full nine-field label.  Every held-out row receives a full label strictly for
evaluation.  This is inverse procedural data generation around verified solutions, not an
attempt to manufacture new expert answers with an unverified model.

## 4. Leakage-safe evaluation

Never split individual perturbations randomly.  The anchor's topology owns the split, so
all descendants of one topology stay together.

Current executable split (seed 17):

- train: 10 topologies;
- `test_cell`: 4 entirely unseen topologies (`B2 T1`, `B2 D1 T1`, `A1 B3 D1`,
  `A1 B3 T1`);
- `test_env`: unavailable—there are zero certified prism specs.

Add two later audits:

- `test_fov`: train on 4°/6°/8°, test on 10°/12° while holding topology overlap fixed;
- `test_env`: certify prism anchors, then hold the entire prism environment out.

Do not report a cross-environment transfer claim until the second audit exists.

## 5. Gate A — does the proxy contain useful information?

Generate 512 samples, with 64 full labels in the training split and full labels on the held
out topology split.  Report, by perturbation radius:

- Spearman correlation between each proxy and full fitness;
- topology agreement and topology-change confusion;
- false-safe rate: proxy looks good while full fitness/gates fail;
- full fitness and hard-pass precision among the proxy's top 20 selections.

**Pass:** sparse-multifield Spearman ≥ 0.5 and top-20 realized full fitness exceeds random
selection by at least 25%.  **Fail:** correlation < 0.2 or false-safe rate > 50%; change the
proxy before training a large model.

## 6. Gate B — fixed-HF-budget transfer and scaling

The implemented baseline compares four estimators on identical inputs:

1. `hf_only`: geometry/spec → full fitness;
2. `proxy_only`: geometry/spec → sparse proxy fitness, evaluated against full;
3. `learned_proxy_delta`: many proxy examples + a small paired full residual model;
4. `measured_proxy_delta`: cheap proxy evaluated at inference + residual correction.  This
   is an upper bound, not a compute-matched learned-transfer result.

Sweep proxy labels `{64, 128, 256, 409}` against full training labels `{16, 32, 64}` over at
least five model seeds.  Main metrics are held-out-topology Spearman, MAE, top-20 realized
full fitness, top-20 regret, and full hard-pass precision.

**Transfer pass:** at 32 full labels, `learned_proxy_delta` improves both Spearman and
top-20 realized fitness over `hf_only`, with a bootstrap 95% interval excluding zero across
data/model seeds.

**Scaling sign:** at fixed 32 full labels, proxy sizes 64→128→256 must improve the median
top-20 result on at least two consecutive steps, and the 256-proxy result must exceed the
64-proxy result.  This is evidence of a useful scaling direction, not a scaling law.

**Compute-equivalence test:** compare (a) more proxy labels, (b) the same engine-call budget
spent on more full labels, and (c) more best-of-N verifier calls.  A proxy method only earns
its complexity if it wins per total trace call, not merely per nominal training example.

## 7. Gate C — refinement policy

Only after Gate B passes:

1. pretrain a shared encoder on proxy fitness, topology, and failure-stage prediction;
2. attach an action head that predicts a normalized genotype delta;
3. fine-tune on existing same-parent search groups and the small full-label set;
4. run best-of-N proposals through `full_multifov` and accept only verified improvements;
5. add accepted and informative failed moves to the next training round.

Compare against CMA-ME and random local perturbation at equal full-verifier calls.  Primary
outcomes are best full fitness, hard-pass yield, requested-topology retention, and area
under best-so-far versus verifier-call curve.  Stop if two expert-iteration rounds do not
improve the curve.

## 8. Gate D — CODE V delta transfer

Treat CODE V as high fidelity, not as another abundant dataset.  Start with a stratified
cohort over topology, FOV, proxy/full disagreement, and perturbation radius.  Fit

`CODEV_merit = local_full_prediction + learned_delta`.

Use active learning to request the next CODE V cases where uncertainty × predicted utility
is high.  Compare against random cohort selection at equal CODE V calls.  Keep the local
and CODE V labels separate; never relabel local full-beam as physical truth.

## 9. Commands and artifacts

Pilot data:

```bash
python3 -m nslens.rl.data.build_multifidelity \
  --certified runs/rl_tasks/certified_20k.json \
  --n 512 --hf-train 64 --seed 17 \
  --out runs/multifidelity_pilot_v1
```

Transfer baseline:

```bash
python3 -m nslens.learning.multifidelity_baseline \
  --data runs/multifidelity_pilot_v1/traces.jsonl \
  --proxy-sizes 64,128,256,409 --hf-sizes 16,32,64 \
  --model-seeds 5 \
  --out runs/multifidelity_pilot_v1/baseline.json
```

Contracts:

```bash
python3 tests/test_rl_multifidelity.py
```

The generator refuses to overwrite a non-empty output directory.  Its manifest records
the fidelity definitions, label counts, sampling mixture, nominal engine calls, elapsed
time, and number of full hard passes.

## 10. Immediate interpretation rules

- Pencil success + sparse failure means route knowledge transfers but field reasoning does
  not; add edge-field tasks rather than more pencil data.
- Sparse success + learned-delta failure means the proxy is useful only when explicitly
  called; use it as a search tool, not as pretraining.
- Learned-delta success on seen FOV but failure on held-out FOV means interpolate FOV during
  training and add an explicit field-angle representation.
- Local success + CODE V failure means the missing physics, not model scale, is the
  bottleneck.  Improve the fidelity bridge before buying more training compute.
- A monotone best-of-N curve with verified outputs is the cleanest early evidence that more
  inference compute buys quality; it is stronger than training loss alone.
