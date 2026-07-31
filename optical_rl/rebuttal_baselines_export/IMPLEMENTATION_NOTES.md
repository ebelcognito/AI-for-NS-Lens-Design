# Implementation notes

Deeper technical detail behind the [README](README.md). Covers the exact code
changes, the new descriptor mechanics, the orchestrator internals, and the
gotchas that cost time.

---

## 1. New param-space descriptor (`lens_param`) — P5

**Why:** R4.1 asks for the Antonov-style comparison (Seidel behavior descriptors
vs raw lens-*geometry* descriptors). The repo had `seidel`, `random`, and
`zernike` descriptor kinds but no way to index the archive by raw lens
parameters. `lens_param` selects K entries of the θ vector as the BD axes.

**Files:**
- `optical_rl/qd/lensmega/descriptors/lens_param.py` (NEW) — `LensParamDescriptor`.
  Mirrors `RandomProjectionDescriptor` exactly: it consumes the `theta_tensor`
  kwarg the problem evaluator already passes, and returns
  `theta_tensor.index_select(0, indices)` — a differentiable gather, so the
  CMA-MEGA gradient machinery works unchanged (∂measure_k/∂θ = e_k).
  `fixed_bounds` returns `None` → archive bounds auto-calibrate to the realized
  parameter distribution (gives the geometric archive its best shot; no strawman).
- `config/schema.py` (patch) — adds `"lens_param"` to the `DescriptorCfg.kind`
  Literal and a crosscheck branch (`k = len(params["indices"])`).
- `run.py` (patch) — `_build_descriptor` branch constructing
  `LensParamDescriptor(indices=..., names=..., theta_dim=len(theta_bounds))`.

**θ layout for the sixlet** (`sixlet_apsc_discovery_v2.yaml`, full_dof, D=37):
`[12 curvatures (idx 0–11), 12 gaps (12–23), flange (24), 12 materials (25–36)]`.
So `indices: [0, 11]` = curvature of element-1 front (c0) and element-6 back
(c12) — the "first & last element curvature" axes you approved.

---

## 2. JSON-pool refinement mode

**Why:** P1 (`m4`) and P2 (`m5`) emit a flat `{designs:[{theta, fitness}]}` JSON,
not an HL `archive.pkl` or an SA+GA elites file. To push those pools through the
*identical* refinement, `tools/refine_all_parallel.py` gained:
- `--pool-from json` choice + `--json-pool <file>` arg,
- `load_json_pool()` (reads `fitness` or `raw_fitness`, filters by threshold),
- `--pool-from json` requires `--problem-config` (no run-dir config.yaml to fall
  back on).

The QD arms (P3/P4/P5) refine via `--pool-from hl --hl-run <dir>` reading
`archive.pkl`. **Important refinement decision:** every arm is refined with
`--problem-config <sixlet_apsc_discovery_v2.yaml>` pinned, so the problem +
refine settings are byte-identical across all methods (keeps "identical
refinement" airtight, and avoids building the `lens_param` descriptor inside the
refine workers).

---

## 3. P2 driver (`m5_cmaes_restarts.py`)

Derivative-free CMA-ES on the raw θ box, IPOP-style restarts, until the 100k
forward-eval budget is spent.
- Works in normalized `[0,1]^D`; `denorm(x) = lo + x·(hi−lo)`; evaluates via
  `problem.evaluate(theta, with_grads=False).fitness` (same interface m3 uses).
- Objective = `−fitness` (minimized). Full population evaluated per `ask`/`tell`
  (no mid-population break → no ask/tell count mismatch); slight overshoot of the
  budget truncated at the end.
- On `es.stop()`, random-restart with `popsize *= 2` (capped) — IPOP.
- Logs **every** evaluation as the candidate pool; `gradient_evals: 0` recorded.
- Uses the `cma` package (Hansen's pycma; pure-Python, numpy-only). `pymoo`
  (which `m3_nsga2.py` imports) is **not installed** in the env — that's why P2
  is written directly against `cma` rather than reusing the m3 scaffold.

---

## 4. Orchestrator (`run_sixlet_baselines.sh`)

`bash run_sixlet_baselines.sh <ALLOC_JOBID> [budget=100000] [seeds="42 43"]`

- **Node-portable:** reads `NumCPUs` from `scontrol show job` and sizes srun
  steps so the concurrent serial baselines (P1/P2, ~2 CPUs each) and the
  sequential QD runs (P3/P4/P5) never oversubscribe the allocation
  (`QD_CPUS = NCPU − serial − 2`).
- **Phase 1a:** launches m4 + m5 for all seeds as background srun steps.
- **Phase 1b:** runs the QD searches (randbc, nograd, paramspace) **seed-major**
  and sequential, cloning each config per seed (`cma_seed`, `run_name`,
  `parallel_workers`). Seed-major so a complete seed-42 table lands before
  seed-43 starts.
- **Phase 2:** identical L-BFGS refinement of every new pool (top-500 ≥ −1.0).
- **Phase 3:** builds a manifest (new pools + the reused HL/SA+GA reference
  pools) and runs `analyze_rebuttal_sixlet.py`.
- Honors `HPY` from the environment (override to a node-local staged python).

**Robustness:** if the run is cut off, the refined pools already written are on
disk; rebuild a manifest and re-run `analyze_rebuttal_sixlet.py` on whatever
completed.

---

## 5. Analysis (`analyze_rebuttal_sixlet.py`)

Reads a manifest `{method: {seed: refined_pool.json}}`, and for each method×seed
computes the ε-net cardinality (greedy: sort by refined fitness desc, accept a
design iff its min normalized-L2/√D distance to all accepted > ε) at RMS
thresholds {<30 µm, <100 µm, <1 mm}, plus best/median/mean RMS. Prints
per-seed + aggregated (mean over seeds) tables with ratios vs the reference
method. Method-agnostic — identical formula for QD archives, multi-start pools,
and CMA-ES restart pools.

---

## 6. Gotchas that cost time (so future-you doesn't repeat them)

- **`activate_env.sh` is dead** (deleted venv → falls back to broken anaconda
  3.9). Use `$HPY` directly. Confirms the standing memory note.
- **Bare `python`** on login = anaconda 2021.11 (3.9), whose `import torch`
  crashes in sympy/mpmath. Always use `$HPY` or a staged local python.
- **Imports are NFS-bound (~5–9 min), not CPU-bound.** Don't chase a "less
  loaded" partition — both `hov` and `general` are `OverSubscribe=NO` and give
  dedicated cores. Stage the env node-local (`stage_env.sh`) instead.
- **`srun --overlap` needs the full flag set** or you get 1 CPU / no mem:
  `--overlap --exact --ntasks=1 --cpus-per-task=N --mem=M`.
- **`/tmp` is node-local** (per the repo's longleaf handoff doc) — good for
  staging, but temp configs/outputs that other srun steps must read go on NAS
  (the orchestrator writes everything under `results/`).
- **Backgrounded `&` inside a one-shot shell can die** when the shell exits;
  launch long jobs from a persistent tmux (or as tracked background tasks).

---

## 7. What was deliberately NOT done
- **All-5-Seidel descriptor arm (B4)** and **alternative Seidel pairs S3/S4 (B5)**
  — deprioritized by the user for this pass.
- **Curriculum ab-initio baseline** — argued out of scope in prose (combined
  explore+refine, gradient-only, expensive); not a search baseline.
- **Switching the worker pool from `spawn`→`fork`** — would let workers inherit
  torch (zero re-import, the biggest structural import win) but risks torch fork
  deadlocks; the repo chose `spawn` deliberately, so left untouched.
