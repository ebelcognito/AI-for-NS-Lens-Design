# Experiment Plan — Do we CONTROL the discovery mechanism? (CMA-ME vs multi-start, and is the geometry confining the good set?)

> **What this is.** The concrete, runnable plan for the number-one question in
> [13_new_direction_framing.md](13_new_direction_framing.md) §2. Written 2026-07-10 (Opus 4.8).
> Grounded in the real code: [`ns_probe/cma_me.py`](../../ns_probe/cma_me.py),
> [`ns_probe/run_fold_qd.py`](../../ns_probe/run_fold_qd.py),
> [`ns_probe/run_arvr_qd.py`](../../ns_probe/run_arvr_qd.py),
> [`ns_probe/compare_descriptors.py`](../../ns_probe/compare_descriptors.py).
>
> **The question, decomposed into falsifiable sub-questions:**
> - **Q1 — Is CMA-ME more than multi-start?** Does the *evolutionary* machinery (covariance
>   adaptation + archive-improvement ranking + persistence/restart-from-elite) surface more/better
>   NS designs than dumb sampling **at matched evaluation budget**?
> - **Q2 — What is discovery controlled by?** Rank the levers (geometry prior, tilt range, fitness,
>   descriptor, optimizer) by measured effect size.
> - **Q3 — Is the geometry confining the good set to a small manifold?** Measure the *size* of the
>   good-fold manifold directly (random-hit rate, basin count).
> - **Q4 — How does it find something "novel"?** Are CMA-ME's designs actually *unreachable* by
>   random sampling, or is every one within ε of a random draw?
>
> **One decision this plan produces:** which sentence goes in the paper —
> *"quality-diversity **search** illuminates…"* (CMA-ME earns it) vs
> *"geometry-conditioned **enumeration** structures…"* (multi-start matched it). See §7.

---

## 0. Prerequisite — kill the engine-drift confound FIRST (non-negotiable)

[12_descriptor_ablation.md](12_descriptor_ablation.md) §5 found the stored `results_fold_qd/archive.json`
headline is **stale vs the live engine** (the `dist-node`→`integrations` rebuild/cp sync gotcha).
**Every arm of every experiment below must run on the same freshly-built engine, in the same
session.** Do this once, up front, and never compare across it:

```bash
cd ~/Desktop/NS_lensdesign/ray-optics && npm run build-node && cp dist-node/rayOptics.js integrations/rayOptics.js
PY=/playpen/hisham26/deeplens_env/bin/python   # (RUN_NOTES.md env)
cd ~/Desktop/NS_lensdesign && PYTHONPATH=. $PY ns_smoke_test.py    # detector power == 1.0
```

All numbers in this plan are **relative within one session on one engine build**. Do not quote a
cross-session absolute.

---

## 1. The shared harness (write ONE script, swap only the sampler)

The cleanest apples-to-apples design: a single new script
**`ns_probe/discovery_control.py`** that imports `evaluate`, `FD.LO/HI`, `BC_BINS`, `BC_BOUNDS`
straight from `run_fold_qd` (and the AR analogues from `run_arvr_qd`), so **fitness, engine,
genotype bounds, archive grid, and insertion rule are byte-identical across arms** — only the
*sampler* differs. This isolates the evolutionary contribution and removes every "you changed
two things" objection.

```python
# ns_probe/discovery_control.py  (sketch — reuse, do not re-implement, evaluate/GridArchive)
from run_fold_qd import evaluate, BC_BINS, BC_BOUNDS   # identical fitness + BCs
import fold_demo as FD
from cma_me import CMAES, GridArchive
# --mode {cmame, random, restart_ls}; --evals; --seed; --out
# ALL modes: same archive = GridArchive(BC_BINS, BC_BOUNDS); same archive.add(...) call;
#            count invalids as evals (match run_fold_qd's n bookkeeping exactly);
#            log per-eval: (n_evals, coverage, qd_score, first-fill event per new cell).
```

Every arm logs, per 1000 evals: coverage, QD-score (`Σ fitness over filled cells`),
`best_focthru`, and — critically — **the eval index at which each cell was first filled**
(`first_fill[idx] = n_evals`), so we can plot *evals-to-discovery* per topology, not just endpoint
totals.

### The three arms (the ablation ladder — each removes one capability)

| Arm | Sampler | Removes | Isolates |
|---|---|---|---|
| **A — CMA-ME** | current `run_fold_qd` loop (5 emitters, restart-from-elite, archive-improvement ranking) | nothing | full stack |
| **B0 — pure random search** | `x ~ Uniform(FD.LO, FD.HI)` i.i.d., insert into the same archive | **all** adaptation + persistence (keeps niche protection) | does the archive alone explain it? |
| **B1 — random-restart local search** | uniform seed → short vanilla CMA-ES minimizing `−fitness` (**single-objective, NO archive-improvement ranking**), insert every evaluated point; restart from a *fresh uniform* seed (not an elite) on convergence | archive-improvement ranking + persistence (keeps covariance adaptation) | is it just "a good local optimizer," or is the QD-specific exploration load-bearing? |

**Reading the ladder (the whole point):**
- **B0 ≈ A** → the optimizer adds nothing; **geometry + archive niche-protection** do all the work.
  *Multi-start is as good as this.* Paper claim becomes "enumeration," not "search." (Q1 = no.)
- **B1 ≈ A > B0** → **local covariance adaptation** is the active ingredient; the QD-specific
  archive-improvement ranking is not. "Smart optimizer, not smart QD." (Partial credit.)
- **A > B1 > B0** → the full stack earns it; archive-improvement ranking + persistence specifically
  buy coverage of rare topologies. *This is the "intelligent discovery" result.* (Q1 = yes.)

---

## 2. Experiment E1 — the head-to-head (answers Q1)

**Run** all three arms on the **fold** problem (cheap, ~14 ms/eval) at matched budget, **≥5 seeds each**:

```bash
for seed in 0 1 2 3 4; do
  for mode in cmame random restart_ls; do
    PYTHONPATH=.:ns_probe $PY ns_probe/discovery_control.py --mode $mode --evals 14000 \
        --seed $seed --out ns_probe/results_disc/$mode/seed$seed
  done
done
```

**Metrics (all binning-independent or budget-matched; reuse `compare_descriptors.score()`):**
1. **# ε-distinct high-efficiency fold families** = distinct dominant interaction-signatures among
   elites with ≥2 TIR and focthru ≥ 0.9 (`fam_high` in `score()`). *Primary metric — this is
   "diverse good NS designs found."*
2. **Best fold spot** = min spot among high-eff NS elites (`champ.spot`).
3. **QD-score** and **coverage** at the budget.
4. **Evals-to-K families** (K = 3, 5): area-under-the-discovery-curve, from `first_fill` logs. A win
   here (A reaches K families in fewer evals) is the *sample-efficiency* claim.
5. **Deep-fold reach:** best focthru in cells with tir_count ≥ 4 (the rare, hard topologies where
   persistence should matter most).

**Report** mean ± 95% CI over seeds; a Mann–Whitney U on metric (1) between A and B0 is the headline
significance test. **Decision rule in §7.**

---

## 3. Experiment E2 — is the geometry confining the good set? (answers Q2/Q3)

Two measurements, both cheap, both aimed at "how big is the good-fold manifold, and who sets its size."

### E2a — Random-hit rate (the manifold's *measure*)
Draw **N = 50,000** uniform genotypes, evaluate, and tabulate: fraction valid, fraction landing in a
folded cell (tir≥2), fraction *good* folds (focthru≥0.5), fraction *high* (≥0.9). This is literally
B0's byproduct — no extra run needed.
- **If a large fraction of random draws are already good folds** → the good set is a *fat* target;
  the geometry (bounds + off-axis + tilt range) confines the space so tightly that folding is
  near-forced, and multi-start must win. *This would be the strongest evidence that "the geometry is
  already confining it to a small good solution space."*
- **If good folds are rare (<<1%)** → there is a real needle-in-haystack, and an efficient sampler
  can earn its keep. The random-hit rate is the single number that most directly answers Q3.

### E2b — Prior-sensitivity sweep (who controls discovery)
Vary **one lever at a time**, hold the rest at the fold/AR defaults, measure `fam_high` + best spot:

| Lever | Sweep | Expected (from prior results) |
|---|---|---|
| **geometry: on-axis vs off-axis** (AR) | `run_arvr_qd.py` collinear vs display-above | 12/482 → 488/493 folded — **already the dominant effect** ([02_arvr_prism_qd.md](../ns_differentiable_refinement/02_arvr_prism_qd.md)) |
| **tilt range** (AR) | `--tilt-range 25, 35, 45, 50` | ±25 starves folds (fitness 0.05); ±50 unlocks (0.63) — quantify the threshold |
| **thinness/compactness pressure** | archive BC2 bound wide vs tight; or drop it | without compactness pressure, good region → all-transmission ([03_design_discussion.md](03_design_discussion.md) §2.4) |
| **fitness** (fold-blind vs fold-naming) | control only: add a TIR-count bonus | sanity that emergence isn't smuggled |

**Output:** an effect-size ranking (Δ`fam_high` per lever) that *is the answer to Q2* — "discovery is
controlled by X ≫ Y ≫ Z." The hypothesis (from §13) is **geometry prior ≫ fitness ≫ descriptor ≫
optimizer**; E1+E2 confirm or overturn it with numbers.

---

## 4. Experiment E3 — persistence / basin structure (the "intelligent" test, Q3)

The specific mechanism CMA-ME claims over multi-start is **persistence reaching deep basins** (the
HyperLens 924 µm→2.7 µm story). Test it directly:

1. **Basin census.** Pool all high-eff NS elites from B0's 50k random draws; cluster their genotypes
   (normalized params, e.g. DBSCAN at ε=0.2). Count distinct basins per fold-count. This is the
   *ground-truth map* of the good manifold.
2. **Does A reach basins B0 misses at matched budget?** For each fold-count, compare the set of basins
   discovered by A@14k vs B0@14k (not B0@50k). If A consistently reaches a deep-fold (tir≥4) basin
   that B0@14k never enters, that is persistence *working* — the intelligent-discovery evidence.
3. **Persistence ablation inside A.** Re-run A with `_new_emitter` re-seeding from a **fresh uniform
   sample instead of `archive.random_elite`** (one-line change in the emitter factory). If deep-fold
   coverage collapses to B0 levels, persistence is the active ingredient; if unchanged, it isn't.

---

## 5. Experiment E4 — novelty reachability audit (answers Q4)

"How does it find something novel?" made falsifiable: **is each CMA-ME elite actually unreachable by
random draw?** For every A elite, compute normalized-parameter distance to its nearest B0 elite (the
50k pool). Report the distribution and the fraction with **no B0 neighbor within ε** (ε=0.2, the
HyperLens metric).
- **If ~every A elite has a close B0 neighbor** → nothing is "novel" about the search; A just polished
  points random sampling already lands near. (Consistent with Q1 = no.)
- **If a meaningful fraction of A's high-eff folds are ε-far from all 50k random draws** → the search
  reaches genuinely hard-to-sample regions. *This is the concrete meaning of "finds something novel."*

This is the direct analogue of Path A's memorization control ([01_extracted_research_context.md](01_extracted_research_context.md) §2.3)
— reused here to audit the *search*, not a generative model.

---

## 6. Supporting figure — objective evolution with optimization (the researcher's explicit ask)

For a few representative champions (fold + AR prism), log and plot **fitness vs eval index**,
annotated where the dominant fold count changes / TIR first appears. Purpose: *see* how the TIR-fold
topology influences the landscape (jumps at cell walls, plateaus within a basin) — visual support for
"the descriptor discretizes the discontinuous axis." One-off instrumentation on the emitter loop;
cheap; produces the slide-grade figure noted in [13](13_new_direction_framing.md) §7 item 2.

---

## 7. Decision matrix — what each outcome means for the paper

| E1 result | E2a random-hit | E3 deep-basin | Verdict | Paper wording |
|---|---|---|---|---|
| **B0 ≈ A** | high (good folds common) | A ≈ B0 | geometry confines; optimizer inert | *"geometry-conditioned **enumeration**; the good NS manifold is small and physics-determined; we **structure & refine** it"* — still a strong artifact paper (§13 §2.3) |
| **A > B1 > B0** | low (good folds rare) | A reaches deeper | full QD stack earns it | *"quality-diversity **search** illuminates a diverse family multi-start cannot reach"* — the H2 claim, defended |
| **B1 ≈ A > B0** | any | mixed | adaptation matters, QD-ranking doesn't | *"gradient-free **local search over a niche archive**"* — honest middle; drop "persistence" language |

**Crucial framing note:** *all three outcomes are publishable.* The experiment is not "prove CMA-ME
wins"; it is "**find out which true sentence to write.**" The failure mode to avoid is claiming
intelligent search when B0 matched it — that is the exact flank a reviewer probes, and E1 pre-empts it.

---

## 8. Effort, order, and gates

1. **Prereq (§0)** — engine rebuild + smoke test. ~10 min. **Blocks everything.**
2. **Write `discovery_control.py`** (§1) — reuses `evaluate`/`GridArchive`, adds `--mode` + `first_fill`
   logging + a multi-start and a restart-LS sampler. ~half a day. The only real code.
3. **E1 (§2)** — 3 modes × 5 seeds × 14k evals ≈ 15 runs × ~2 min = ~30 min compute. **The decisive run.**
4. **E2a (§3)** — free (B0 byproduct). **E2b** — a handful of AR runs (~9 min each), do overnight.
5. **E3/E4 (§4–5)** — pure analysis on E1/E2 archives (clustering + nearest-neighbor); ~half a day, no new sims.
6. **§6 figure** — ~2 h instrumentation.

**Total:** ~1–1.5 days, dominated by writing one harness; compute is minutes-to-overnight. **Gate the
rest of the program on E1's decision (§7)** — it fixes the single sentence in §13 §8 that the
number-one question changes, and tells Track 1/Track 2 (§13 §4) how to describe the search.

## 9. Confounds to control (so the answer is trustworthy)
- **Matched budget = matched `evaluate()` calls**, invalids included (both A and B0 count them; §1).
- **Identical archive grid & insertion** across arms (import, don't copy, `BC_BINS/BC_BOUNDS`).
- **Same engine build** (§0); re-run all arms if the engine is touched.
- **Multi-seed** (≥5) with CIs — single-seed fold-QD is known noisy ([12](12_descriptor_ablation.md): baseline found 0 high-eff folds at seed 1).
- **B1 restart from uniform, not elite** — otherwise B1 secretly re-imports persistence and the ladder collapses.
- **Report the null loudly** if B0 wins; do not bury it.
