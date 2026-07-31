# Session export — 2026-07-02: local migration, CMA-MEGA fix, v2 anchor rebuild, 40k baseline head-to-head

Context handoff for continuing the sixlet rebuttal baseline work **locally** (no
Longleaf/SLURM). Covers: (1) moving to a fresh HyperLens clone, (2) fixing the
non-canonical CMA-MEGA loop, (3) rebuilding the missing v2 anchor lens from
recipe, (4) building a local no-SLURM baseline orchestrator, (5) running the 8-arm
40k head-to-head, and (6) the refinement + eps-net diversity analysis and what it
does/doesn't show. Ends with the open decision (100k run).

---

## 0. TL;DR

- Working copy is now **`~/Desktop/HyperLens`** (fresh clone of `github.com/ebelcognito/HyperLens`, HEAD `0dde12c`). The old `~/Desktop/RL` tree is untouched. Env = `/playpen/hisham26/deeplens_env/bin/python`.
- **Fixed the CMA-MEGA loop.** It was drawing a fresh archive parent *every generation* (no persistent walking θ*, no gradient step) — neither canonical CMA-MEGA nor OMG-MEGA. Fix is additive behind `algorithm.emitter_mode: canonical | hybrid` (default `hybrid` = unchanged/published).
- **Rebuilt the v2 anchor lens** `sixlet_50mm_f4_apsc_v2.json` from recipe (it was Longleaf-only). `r_sensor=17.0` is the control; validated at **0% seed rejection**.
- **Ran the full 8-arm 40k/seed-42 head-to-head locally**, refined, and analyzed eps-net diversity.
- **Result (robust across selections/bounds/thresholds):** HyperLens wins **productivity + quality** decisively (≥500 usable designs, best median 96µm; two baselines produce zero), but **loses eps-net θ-diversity** on the raw metric. **However** the diversity gap is largely a metric+budget artifact (the `<1mm` bar rewards scatter among *bad* designs; at `<100µm` nobody is diverse; sub-30µm is empty for everyone). 40k = 40% of the planned 100k budget, ~1/5 of the historical 197k run.
- **Open decision:** fire the **100k** run (planned budget, re-scoped Seidel bounds) — the first budget where the diversity question is measurable.

---

## 1. Environment / workspace (local)

| item | value |
|---|---|
| repo | `/home/hisham26/Desktop/HyperLens` (HEAD `0dde12c`; 3 commits ahead of the old `RL` tree, which is HEAD `fc2f150`) |
| python | `/playpen/hisham26/deeplens_env/bin/python` — py3.12, torch 2.5.1+cu121, numpy, **cma 4.4.4, pymoo 0.6.1.6, pydantic 2.11, pyarrow 24** |
| run cmd | `PYTHONPATH=/home/hisham26/Desktop/HyperLens <py> -m optical_rl.qd.lensmega.run --config <yaml>` from repo root |
| deeplens | env-installed editable → `RL/DeepLens` (imports anywhere) |
| DeepLens data | symlinked `HyperLens/DeepLens -> /home/hisham26/Desktop/RL/DeepLens` so `DeepLens/datasets/lenses/*.json` + `run.py`'s `parents[3]/DeepLens` resolve |

**Gotchas:**
- `optical_rl` is NOT env-installed → must set `PYTHONPATH` to the clone root (no shadowing).
- `activate_env.sh` only resolves its relative env paths when sourced from the `RL` *parent* dir; use the absolute python path.
- **Disk is tight:** `/` was 100% full (freed ~15G via `pip cache purge` — the only thing deleted this session, a re-downloadable cache); `/playpen` ~99%. `results/` is gitignored.
- Installed `pyarrow` (`pip install --no-cache-dir pyarrow`) — needed by `io_results.py` (Parquet pools).

---

## 2. CMA-MEGA fix (the "completely wrong" thing)

### The bug
`optical_rl/qd/lensmega/core/runner.py` (and the legacy `cmamega_triplet`/`cmamega_ext` runners) drew a **fresh random archive parent every generation** (`archive.sample_parent()`), recomputed gradients there, and applied the coefficient-CMA distribution to that ever-changing gradient basis — with **no persistent search point and no gradient-ascent step**. So the coefficient CMA adapted against an unrelated basis each gen. This is neither canonical CMA-MEGA (persistent walking θ*) nor OMG-MEGA (fixed isotropic coeffs). The emitter itself (`_cma.py`) was *already* canonicalized in prior work; only the runner was wrong. It was documented as a deliberately-kept divergence in `exported_context_method/canonical_cmamega_and_checkpointing.md` ("Things to be careful about").

### The fix (additive, default preserves everything)
`algorithm.emitter_mode: "hybrid" | "canonical"` (default `hybrid`).
- **canonical** = true CMA-MEGA: maintain `self._theta_star`; gradients taken at θ*; branches `θ* + η·Σ c·∇`; **after `tell()`, step θ\* by the recombination-weighted gradient** `θ* += η·(|mean₀|·∇f + Σ meanₖ·∇mₖ)` (pyribs' `θ ← θ + J @ cma_mean`); on restart `θ* ← archive.sample_parent()`.
- θ* persisted in `CMAMEGACheckpoint.extra["theta_star"]`; restored on `--resume`.
- Guard: `canonical` + `ablate_gradients` raises (P4 must stay hybrid).

**Files changed** (in `optical_rl/qd/lensmega/`): `config/schema.py` (field + crosscheck), `core/runner.py` (θ* init/walk/restart/ckpt), `run.py` (resume restore).

**Validated:** tiny sixlet smokes — canonical improves best-fitness each gen, σ self-regulates, θ* persists to checkpoint; **gen-1 byte-identical to hybrid** (proves default path untouched). Legacy triplet/ext runners still carry the bug (out of scope).

---

## 3. v2 anchor lens rebuild

`DeepLens/datasets/lenses/sixlet_50mm_f4_apsc_v2.json` was Longleaf-only (no local generator). Rebuilt from recipe at `RL/DeepLens/datasets/lenses/` (resolves via symlink):
- base `sixlet_50mm_f4.json` structure (13 surfaces, stop idx 6, n-sk16/f2).
- **`r_sensor = 17.0`** — THE control; DeepLens *derives* `sensor_size = 17·√2 = 24.04` → hfov `atan(12.02/50) = 13.52°` → 12mm max image height (matches recipe "13.5°, 34mm diagonal"). Writing `sensor_size` directly is ignored. **The fitness field points come from `sensor_size`, i.e. from `r_sensor`.**
- clear apertures capped at **22mm** (SAG `0.05·22²/2 = 12.1mm`, matches recipe); profile `[22,22,20,20,18,18, stop 6.25, 18,18,20,20,22,22]`.
- Validated: loads OK, **0% seed rejection** on the real config (recipe's success metric; v1 was 97%).
- Exact per-surface CAs differ from the lost Longleaf original (those came from v1 top-5 refined designs) but are recipe-faithful.

---

## 4. Local baseline orchestrator (no SLURM)

**`run_sixlet_baselines_local.py`** — replaces the SLURM `run_sixlet_baselines.sh`. Runs all arms on the v2 anchor at a matched budget, generating the reference rows (HyperLens + SA+GA) locally. Key flag: **`--no-refine`** (search-only: run every arm, report elites found, skip refine+analysis) — added per the directive **"search first, refine LATER."**

**Arms (seed 42):**

| key | method | reviewer |
|---|---|---|
| `seidel_s2s3` | HyperLens CMA-MEGA, Seidel S2/S3, **canonical** (ref) | — |
| `seidel_s2s3_hybrid` | HyperLens, Seidel S2/S3, **hybrid** (shows the fix) | — |
| `randbc` | P3 CMA-MEGA, random-projection BC | R1.3d, R2.3 |
| `nograd` | P4 MAP-Elites, Seidel BC, `ablate_gradients` | R1.3c |
| `paramspace` | P5 CMA-MEGA, raw-curvature BC (Antonov) | R1.3e, R4.1 |
| `multistart_lbfgs` | P1 random-restart + L-BFGS (m4) | R1.3a, R2.4 |
| `cmaes_restarts` | P2 derivative-free CMA-ES + restarts (m5) | R1.3b |
| `sa_ga` | SA+GA+Refine (paper's existing baseline) | — |

### 40k search results (seed 42, `--no-refine`, run root `results/rebuttal_sixlet_local_40k_s42/`)

| arm | elites | best raw fitness |
|---|---|---|
| `multistart_lbfgs` | 348 designs | −0.072 (best single) |
| `seidel_s2s3` (canonical) | 13,316 cells | −0.134 |
| `randbc` | 34,714 cells | −0.230 |
| `paramspace` | 28,100 cells | −0.324 |
| `seidel_s2s3_hybrid` | 15,043 cells | −0.392 |
| `nograd` | 23,943 cells | −1.667 |
| `cmaes_restarts` | 40,000 designs | −8.075 |
| `sa_ga` | elites json | — |

~19 min per QD arm at ~45 evals/s, 14 workers.

---

## 5. Refinement + eps-net diversity analysis

### Selection matters enormously (this was the big lesson)
- **top-100 by fitness** (initial, WRONG for QD): collapses each method to its highest-fitness cluster → kills QD's diversity. Don't use.
- **paper selection: `--threshold -1.0 --top-n 500`** (refine designs above a quality floor). This is what `run_sixlet_baselines.sh` did. `refine_top_elites.py --tag paperfloor`.
- **stricter floor: `--threshold=-0.5`** (user request). `refine_pf05.sh --tag pf05`.

**Designs above the −1.0 floor** (productivity — the archive-cell counts above are NOT comparable across descriptors; this is):

| method | # ≥ −1.0 |
|---|---|
| `seidel_s2s3` (canonical, wide bounds) | **119** |
| `sa_ga` | 32 |
| `multistart_lbfgs` | 24 |
| `seidel_s2s3_hybrid` | 21 |
| `randbc` | 17 |
| `paramspace` | 6 |
| `nograd` / `cmaes_restarts` | **0** |

### The Seidel-bounds finding
Bounds are **fixed** in config: **S2 (coma) [−0.008, +0.008], S3 (astig) [−0.012, +0.012], 512×512 bins.**
The 119 quality designs occupy only **1.6% × 7.3% of those bounds (~8 × 37 of 512² cells)**; S2 std among good designs is **6e-5** (barely varies). ⇒ ~99% of the archive is wasted on low-quality peripheral aberration states, and good designs have no room to spread.

**Re-scoped** to the quality manifold: **S2 [−0.0010, +0.0006], S3 [−0.0040, +0.0040]** (`rescope_seidel.sh`, run `results/seidel_rescope_40k_s42/`). Effect:
- **Productivity exploded:** designs ≥ −1.0 went 119 → **≥500** (hit top-500 cap); at the stricter ≥ −0.5 floor, **41 → ≥500** (~12×).
- **Median refined RMS: 292µm → 96µm** (3×).
- **eps-net diversity: unchanged (still 2)** — re-scoping did NOT fix diversity.

### Final eps-net tables (ε=0.20, D=37, seed 42)

Added a **`<40µm`** bucket to `analyze_rebuttal_sixlet.py` (line 44 `RMS_THRESHOLDS`). Note: eps thresholds are on `refined_fitness` (≈ −rms_mm), so `<30um` ↔ fitness ≥ −0.030 etc.

**Pool floor ≥ −0.5, combined (`table_pf05_combined.json`):**

| method | pool | <30µm | <40µm | <100µm | <1mm | best µm | median µm |
|---|---|---|---|---|---|---|---|
| `seidel_rescope` (canonical, re-scoped) | 500 | 0 | 0 | 2 | 2 | 50.0 | **96.3** |
| `seidel_s2s3` (canonical, wide) | 41 | 0 | 0 | 1 | 1 | 64.3 | 179.1 |
| `sa_ga` | 30 | 0 | 0 | 3 | 6 | 60.8 | 107.2 |
| `multistart_lbfgs` | 17 | 0 | **1** | **3** | **17** | 38.8 | 137.9 |
| `randbc` | 5 | 0 | 0 | 1 | 4 | 58.3 | 127.8 |
| `seidel_s2s3_hybrid` | 3 | 0 | 0 | 0 | 2 | 158.6 | 193.9 |
| `paramspace` | 3 | 0 | 0 | 0 | 3 | 210.2 | 217.4 |
| `nograd` / `cmaes_restarts` | 0 | 0 | 0 | 0 | 0 | — | — |

(At the ≥ −1.0 floor the pattern is the same, larger counts: multistart `<1mm`=24, randbc 8, sa_ga 6, hybrid/paramspace 4, seidel 2.)

---

## 6. Interpretation (where we landed)

**Two conclusions, invariant across all selections/bounds/thresholds tried:**

1. **HyperLens owns productivity + quality — decisively.** Re-scoped canonical HL yields **500 usable designs (≥12× any baseline) with the best median RMS (96µm)**; P2 (CMA-ES) and P4 (no-gradient) produce **zero**. Clean rebuttal story for "the gradient QD machinery produces far more refinable high-quality designs than any baseline."

2. **On raw eps-net θ-diversity, HyperLens loses** (2 distinct vs multistart's 17–24) — and re-scoping the bounds did NOT fix it (good sixlets are intrinsically ~1–2 θ-forms at 40k).

**BUT the diversity loss is largely a metric + budget artifact — do not over-conclude:**
- The **`<1mm` bar is meaningless** (1mm RMS = unusable blur). Multistart's "17 distinct" includes **14 designs in [100µm, 1mm]** — scatter among *bad* designs, which random restarts get for free. Not a real advantage.
- At the **`<100µm`** bar (usable designs), **nobody is diverse**: HL 2 (from 229 good designs), sa_ga 3 (from 9), multistart 3 (from 3), others 0–1. Multistart's "3" is just *having only 3 good designs, each isolated.*
- **`<30µm` / `<40µm` are empty for everyone** at 40k.
- **Budget:** 40k = 40% of the planned 100k matched budget, ~1/5 of the ~197k run that historically produced 25.9µm + a diversity story. The search barely reaches sub-100µm; diverse-*good*-designs is not yet testable.

**Bug to fix:** `analyze_rebuttal_sixlet.py` footer hard-codes *"all methods matched at 100k forward evals"* — it's a literal string, not the real budget (we ran 40k). Misleading; fix to report the actual budget.

---

## 7. Open decision / next steps

1. **Fire the 100k run** (planned matched budget, re-scoped Seidel bounds, canonical, all arms) — the first budget where sub-30µm designs appear and "diverse *good* designs" becomes measurable. This is the decisive experiment. **(recommended next)**
2. Fix the hard-coded `100k` footer in `analyze_rebuttal_sixlet.py`.
3. **Seed 43** for variance bands (everything so far is single-seed 42).
4. **Reconsider the diversity metric** — eps-net in raw normalized-θ may not match "structurally distinct design *forms*," and `<1mm` should be dropped (rewards bad-design scatter). Consider gating diversity on `<100µm` only.
5. **Descriptor question** — Seidel S2/S3 barely separates good sixlets (S2 std 6e-5); curvature/param-space BD showed marginally more θ-spread. If a structural-diversity claim is needed, the lever is the descriptor, not the bounds.

---

## 8. File inventory (all in `~/Desktop/HyperLens/`)

**New scripts this session (repo root):**
- `run_sixlet_baselines_local.py` — no-SLURM 8-arm orchestrator; `--no-refine` search-only, `--refine-threshold`, `--budget/--seed/--workers/--num-rays/--seed-count/--arms/--stamp`.
- `refine_top_elites.py` — refine top-N per method from a finished run; `--top-n --threshold --tag`; builds manifest + eps-net table.
- `overnight_refine.sh` — tmux wrapper: wait for search driver to exit, then refine (uses `--threshold=-1e9` **=form**; a bare `-1e9` breaks argparse).
- `rescope_seidel.sh` — re-scoped-bounds seidel experiment (search → refine → wide-vs-rescoped eps-net).
- `refine_pf05.sh` — re-refine all arms at ≥ −0.5 + re-scoped HL + combined table.

**Edited (tracked):** `optical_rl/qd/lensmega/config/schema.py`, `.../core/runner.py`, `.../run.py` (CMA-MEGA fix); `analyze_rebuttal_sixlet.py` (added `<40um` threshold).

**Results:**
- `results/rebuttal_sixlet_local_40k_s42/` — the main 40k run: `logs/master.log` (search summary), `pools/` (m4/m5/sa_ga), QD run dirs under `results/lensmega/sixlet_rebuttal_*_seed42_*`, `refined_paperfloor/` + `table_paperfloor.json` (≥−1.0), `refined_pf05/` + `table_pf05[_combined].json` (≥−0.5), `refined_top100/` + `table_top100.json` (deprecated selection).
- `results/seidel_rescope_40k_s42/` — re-scoped-bounds run + `table_rescope.json`.

**Reproduce/continue:**
```bash
cd ~/Desktop/HyperLens; export PYTHONPATH=$PWD
DL=/playpen/hisham26/deeplens_env/bin/python
# 100k next run (re-scope bounds first in the seidel config or via a cloned cfg):
$DL run_sixlet_baselines_local.py --budget 100000 --seed 42 --workers 14 --no-refine --stamp 100k_s42
# then refine at the paper floor and analyze:
$DL refine_top_elites.py --root results/rebuttal_sixlet_local_100k_s42 --top-n 500 --threshold=-1.0 --tag paperfloor
```

**Memory notes written** (`~/.claude/.../memory/`): `hyperlens-local-setup`, `cmamega-canonical-fix`, `hyperlens-baseline-data-gap`, `baselines-local-orchestrator`.
