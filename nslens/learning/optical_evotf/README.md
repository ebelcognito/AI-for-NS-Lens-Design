# Optical-EvoTF workflow

This package keeps the released `evosax==0.3.1` EvoTransformer and update heads, then
adds zero-initialized optical-context projections. CMA remains the default search
emitter. All four online arms use the same verifier, archive rule, population, restart
rule, and candidate-call budget.

Install the pinned optional runtime:

```bash
python3 -m pip install -e '.[evotf]'
```

An NVIDIA driver alone is not sufficient for GPU execution. The pinned extra installs
CPU `jaxlib`; install the matching CUDA JAX wheel separately when an accelerator is
required.

## 1. Collect complete CMA generations

Transition logging is opt-in and is deliberately limited to the full-covariance CMA
teacher. Use at least three search seeds so the held-seed view has train, validation,
and test units. Evaluation counts should be divisible by the population size.

```bash
for seed in 0 1 2 3; do
  python3 -m nslens.qd fold24_clean_gated \
    --emitter cma --param-scaling normalized \
    --evals 160 --popsize 10 --seed "$seed" \
    --optimizer-transitions --trace \
    --out "runs/evotf_data/seed${seed}"
done
```

The 640 candidates above are the 64-generation schema smoke stage, not a corpus from
which to draw performance conclusions. Pilot A and Pilot B require the generation and
candidate counts declared in the implementation plan.

## 2. Build leakage-safe views and targets

```bash
python3 -m nslens.rl.data.build_optimizer_trajectories \
  --runs runs/evotf_data \
  --out runs/evotf_dataset \
  --hindsight 5
```

Inspect `runs/evotf_dataset/manifest.json` before training. It reports source hashes,
complete state and lifecycle counts, route/failure composition, split counts, and any
lifecycle removed because an exact parent or candidate crossed a split. A held-route or
held-environment test is absent when the corpus contains no separable held-out component;
the builder does not manufacture one by splitting adjacent generations.

## 3. Train matched numeric and optical arms

All commands below use the same `held_seed` train/validation files and never open its
test file. Population size, dimension, and context length become checkpoint contracts
and must match online evaluation.

Numeric-only warm-up and hindsight training:

```bash
python3 -m nslens.learning.optical_evotf.train \
  --dataset runs/evotf_dataset --view held_seed \
  --arm numeric --phase warmup --steps 1000 \
  --population-size 10 --dimensions 9 --context-len 16 \
  --output checkpoints/evotf_numeric_warm

python3 -m nslens.learning.optical_evotf.train \
  --dataset runs/evotf_dataset --view held_seed \
  --arm numeric --phase hindsight --steps 1000 \
  --population-size 10 --dimensions 9 --context-len 16 \
  --init-checkpoint checkpoints/evotf_numeric_warm \
  --output checkpoints/evotf_numeric_final
```

Optical adapter warm-up and the matched hindsight objective:

```bash
python3 -m nslens.learning.optical_evotf.train \
  --dataset runs/evotf_dataset --view held_seed \
  --arm optical --phase warmup --steps 1000 \
  --population-size 10 --dimensions 9 --context-len 16 \
  --output checkpoints/optical_evotf_warm

python3 -m nslens.learning.optical_evotf.train \
  --dataset runs/evotf_dataset --view held_seed \
  --arm optical --phase hindsight --steps 1000 \
  --population-size 10 --dimensions 9 --context-len 16 \
  --init-checkpoint checkpoints/optical_evotf_warm \
  --output checkpoints/optical_evotf_final
```

The optical core is frozen by default. Add `--unfreeze-core` and a smaller
`--core-learning-rate` for the optional second stage; only the final temporal block and
distribution-update heads are then trainable. Train every embedding-ladder rung with
the same data and hindsight settings, changing only `--enabled-adapters`. Use
`--categorical-mode one_hot` for the matched one-hot control.

Each checkpoint contains exact upstream provenance, adapter/core hashes, train and
validation source hashes, learning curves, validation metrics, and
`test_metrics_read: false`.

## 4. Run the paired online comparison

```bash
python3 -m nslens.analysis.run_fold_evotf_ablation \
  --experiment fold24_clean_gated \
  --seeds 20 21 22 23 24 \
  --evals 6000 --emitters 5 --popsize 10 --context-len 16 --jobs 1 \
  --numeric-checkpoint checkpoints/evotf_numeric_final \
  --optical-checkpoint checkpoints/optical_evotf_final \
  --ladder-checkpoint role=checkpoints/optical_role_final \
  --ladder-checkpoint role_environment=checkpoints/optical_role_env_final \
  --ladder-checkpoint role_environment_status=checkpoints/optical_status_final \
  --ladder-checkpoint role_environment_status_route=checkpoints/optical_route_final \
  --ladder-checkpoint one_hot_full=checkpoints/optical_one_hot_final \
  --output results/fold_evotf_ablation

python3 -m nslens.analysis.summarize_fold_evotf_ablation \
  results/fold_evotf_ablation/results.json
```

The runner refuses warm-up-only checkpoints, checkpoint/data mismatches, test-tuned
checkpoints, and mismatched population/context/adapter settings. Compilation is warmed
before timed searches and reported separately. `results.json` retains every paired
search unit and archive/trace path; `summary.json` and `summary.csv` aggregate complete
task/seed units.

`--jobs` defaults to one so timing runs are sequential. Increase it only when throughput
matters more than uncontended per-arm latency. Parallel workers use the multiprocessing
`spawn` context to avoid forking initialized JAX thread pools; each arm/seed still writes
an isolated archive and trace and retains its own optimizer and wall timers.

## Verification

```bash
python3 tests/test_optimizer_trajectory.py
python3 tests/test_optical_evotf.py
```

The second command runs dependency-free metadata checks everywhere and enables the
JAX/evosax equivalence, invariance, gradient, overfit, and checkpoint tests when the
optional runtime is installed.
