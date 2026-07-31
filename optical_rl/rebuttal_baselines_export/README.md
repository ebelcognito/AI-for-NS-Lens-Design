# HyperLens rebuttal — additional baselines (sixlet)

Self-contained export of everything built to answer the reviewers' "the SA+GA
baseline is not enough" / "isolate what drives the gains" concerns. It adds
**five baselines** to the sixlet head-to-head, each of which strips exactly one
ingredient out of HyperLens so we can attribute the advantage precisely, all at
**matched 100k-evaluation compute**, all pushed through the **identical L-BFGS
refinement**, and compared on **ε-net diversity** + **refined-RMS**.

Nothing here has been run at full scale yet (the cluster was down at export
time). The code is written and **smoke-tested end-to-end** — see
[§ Validation status](#validation-status).

---

## What HyperLens is (the thing we compare against)

HyperLens = **CMA-MEGA** (gradient-guided quality-diversity) with an **archive
indexed by Seidel aberration coefficients** — specifically **S2 (coma)** and
**S3 (astigmatism)** as the two behavior-descriptor (BD) axes. One run yields a
whole archive of structurally distinct, refinable lens designs.

Three ingredients make it work: **(a)** a QD archive, **(b)** a gradient-guided
emitter (the "MEGA" part), **(c)** an *optically meaningful* descriptor (Seidel).
Each baseline below removes exactly one of these.

---

## The five baselines

| # | Name (code) | Removes vs HyperLens | Question it isolates | Reviewer(s) |
|---|---|---|---|---|
| **P1** | Multi-start + L-BFGS-B (`m4_multistart_lbfgs.py`) | the entire QD machinery — no archive, no behavior space; just many random starts each locally polished | Is QD needed at all, or is refining many random starts enough? | R1.3a, R2.4 |
| **P2** | CMA-ES + restarts (`m5_cmaes_restarts.py`) | QD **and** gradients **and** descriptors — plain derivative-free global search on raw lens params, restart to escape minima | Is QD needed vs a strong black-box optimizer? (the LDG-EA method class) | R1.3b |
| **P3** | Random-BC QD (`configs/p3_randbc.yaml`) | the descriptor's **meaning** — same CMA-MEGA + archive, but BD = a random linear projection of θ instead of Seidel | Does the *optical meaning* of the descriptor matter, or would any 2-D archive do? | R1.3d, R2.3 |
| **P4** | MAP-Elites, no-gradient (`configs/p4_nograd.yaml`) | the **gradient** — same Seidel archive, but filled by Gaussian mutation (`ablate_gradients: true`) instead of gradient-guided steps | Does the gradient ("MEGA") part matter, or does the archive alone suffice? | R1.3c |
| **P5** | Parameter-space QD (`configs/p5_paramspace.yaml`) | **optical → geometric** descriptor — same CMA-MEGA + archive, but BD axes = two raw element curvatures (first & last) instead of Seidel | Do *optical* descriptors beat *geometric* ones? (the Antonov et al. 2025 comparison) | R1.3e, R2.3, R4.1 |

Reference rows carried in the same table (already on disk, reused, **not** re-run):
- **HyperLens** — Seidel S2/S3 (the method itself)
- **SA+GA+Refine** — the paper's existing baseline

So the final table reads **HyperLens vs all seven**.

### Reviewer → experiment map
- **R1.3** (the explicit baseline list) — random-restart+LBFGS = P1; CMA-ES restarts = P2; MAP-Elites w/o gradients = P4; CMA-MEGA random descriptors = P3; parameter-space QD = P5.
- **R2.3** (descriptor ablation on sixlet: S2/S3 vs random-proj vs param-space) = P3 + P5 (+ HyperLens as S2/S3). *(All-5-Seidel arm was deprioritized by the user.)*
- **R2.4** (matched-compute multi-start) = P1.
- **R4.1** (Seidel-BD vs curvature-BD, Antonov 2025) = P5. *(R4.4 direct-refine-from-seed falls out of P1's data.)*

---

## The comparison metric

Every method runs at a **matched 100,000 forward evaluations**, every resulting
pool is pushed through the **identical L-BFGS refinement**
(`tools/refine_all_parallel.py`, top-500 designs above raw fitness −1.0), then
scored on two axes:

1. **ε-net diversity (ε = 0.20)** — the count of *structurally distinct* designs,
   where "distinct" = normalized parameter-space Euclidean distance / √D > ε.
   Reported at refined-RMS thresholds **< 30 µm, < 100 µm, < 1 mm**. This is the
   diversity metric (higher = a broader family of usable designs).
2. **Refined-RMS distribution** — best / median / mean, in µm. The quality metric.

Ratios are shown vs the HyperLens (Seidel S2/S3) reference row.

**Gradient budget** (reported separately, since 100k is matched on *forward*
evals): HyperLens / P3 / P4 / P5 ≈ 1 gradient per generation; P1 ≈ 1 gradient
per eval; **P2 = 0** (derivative-free — that's the point).

---

## How to run (end to end)

Prereqs: a SLURM allocation (`salloc -p general -N1 -n1 --cpus-per-task=32
--mem=64G -t 8:00:00`), and the `cma` pip package in the env
(`$HPY -m pip install cma` — already installed).

```bash
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
cd /nas/longleaf/home/hisham26/Desktop/hyperlens

# (optional but recommended) stage the env node-local for fast imports:
bash stage_env.sh <JOBID>          # copies env -> /tmp/hlenv_stage on the node
# then export HPY=/tmp/hlenv_stage/hyperlens/bin/python before launching

# run one seed's full suite (search -> identical refine -> eps-net table):
tmux new -s sixlet-baselines
bash run_sixlet_baselines.sh <JOBID> 100000 "42"      # then "43"
# C-b d to detach; results in results/rebuttal_sixlet_<TS>/
```

Two allocations → run both seeds in parallel (one per node). Output:
`results/rebuttal_sixlet_<TS>/table.json` and `logs/analysis.log`.

Re-run the analysis alone on existing pools:
```bash
$HPY analyze_rebuttal_sixlet.py --manifest <ROOT>/manifest.json --eps 0.20 --ref seidel_s2s3
```

---

## File inventory

**programs/**
- `run_sixlet_baselines.sh` — the orchestrator (search → identical refine → ε-net table). Node-portable (auto-sizes to allocation CPUs). Reuses the existing HyperLens + SA+GA refined pools as reference rows.
- `m5_cmaes_restarts.py` — **P2** derivative-free CMA-ES with IPOP restarts (uses `cma`). NEW.
- `m4_multistart_lbfgs.py` — **P1** random-restart + L-BFGS (pre-existing repo script; included for completeness).
- `lens_param_descriptor.py` — **P5** descriptor (copy of `optical_rl/qd/lensmega/descriptors/lens_param.py`). NEW.
- `refine_all_parallel.py` — refinement pool runner with the NEW `--pool-from json` mode (copy of `tools/refine_all_parallel.py`).
- `analyze_rebuttal_sixlet.py` — ε-net + refined-RMS comparison table from a manifest. NEW.
- `stage_env.sh` — copy the conda env to node-local `/tmp` for fast imports. NEW.
- `smoke_rebuttal.sh` — tiny end-to-end validation of every new code path. NEW.

**configs/** (each cloned from `sixlet_apsc_discovery_v2.yaml`; problem block byte-identical = the fairness anchor)
- `p3_randbc.yaml` — descriptor `random` (dim 2), archive bounds auto-calibrated.
- `p4_nograd.yaml` — Seidel S2/S3 archive + `ablate_gradients: true`.
- `p5_paramspace.yaml` — descriptor `lens_param`, axes = θ indices [0, 11] (first & last element curvature).
- `smoke_p5_paramspace.yaml` — tiny (300-iter) variant for smoke testing.

**patches/** (changes to existing tracked repo files)
- `schema.py.diff` — adds `"lens_param"` to the descriptor-kind enum + its crosscheck.
- `run.py.diff` — wires `lens_param` into `_build_descriptor`.
- `refine_all_parallel.py.diff` — note: this file is untracked in git; the full modified copy is in `programs/refine_all_parallel.py` (JSON-pool mode added).

---

## Environment & compute findings (operational knowledge)

- **Correct interpreter:** `$HPY = /nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python` (py 3.12, torch 2.11). `activate_env.sh` is **broken** — it points to a deleted venv (`playpen/hisham26/deeplens_env`) and silently falls back to system anaconda 3.9, whose torch crashes (`sympy/mpmath MPZ` assertion). All scripts here hardcode `$HPY`.
- **`cma`** (pure-Python, numpy-only) is installed in the env for P2.
- **Slow imports are NFS, not cores.** torch lives on NFS home (`/nas/longleaf`), and a *clean* import on idle dedicated cores still took 5:46+. Both `hov` and `general` are `OverSubscribe=NO` — you get dedicated cores; the high node load averages are other jobs on *other* cores of the big (~155-core) shared nodes, and do **not** steal yours. **A different partition does not fix import time.**
- **The fix is storage:** stage the env on node-local `/tmp` (local xfs, ~353 GB free) via `stage_env.sh` → imports read locally. `/work` is *also* NFS (different server), so it is not a reliable win.
- `general` currently has far more idle capacity than `hov` and schedules instantly; no `--exclusive` needed (that reserves the whole node and just queues longer).

---

## Validation status

Smoke test (`smoke_rebuttal.sh`, tiny budgets) passed on the compute node:
- P2 `m5_cmaes_restarts` ✓ (200 evals, wrote pool)
- P1 `m4_multistart_lbfgs` ✓
- JSON-refine mode ✓ (`success=3, errors=0`)
- P5 `lens_param` param-space QD ✓ (filled 398 archive cells, rendered elites, wrote a run dir)
- P3/P4 are trivial config variants of the already-working QD path.

**Not yet done:** the full 100k-eval runs (cluster was down). Everything is ready to launch via `run_sixlet_baselines.sh`.

---

## Reference pools reused (not re-run)
- HyperLens Seidel S2/S3: `results/lensmega/sixlet_apsc_discovery_v2_seed{42,43}_*/refined_top500_seed{42,43}_hl.json`
- SA+GA: `results/diversity_run_20260506_033447/sixlet/seed{42,43}/sa_ga/refined_above_-1.0_seed{42,43}_sa_ga.json`
