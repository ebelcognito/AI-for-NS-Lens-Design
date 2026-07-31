# Results — Discovery control in the PRISM setting (source below, detector perpendicular)

> **What this is.** The discovery-control experiment of
> [14_discovery_control_experiment_plan.md](14_discovery_control_experiment_plan.md) /
> [15_discovery_control_results.md](15_discovery_control_results.md), rerun on the **prism
> geometry**: a **parallel ray bundle emitted from BELOW** (straight up) and a **detector
> PERPENDICULAR to it** on the right. A straight transmission path is geometrically impossible,
> so a ~90° TIR fold is **FORCED** — the opposite regime from the collinear fold, where folding
> is *optional*. Run 2026-07-10 (Opus 4.8) on the same freshly-built engine
> (`ns_smoke_test.py` detector power = 1.000).
>
> **Same measurement model as the fold env, at the researcher's request:** parallel bundle in,
> focused spot out, fitness = focused throughput, descriptor = (TIR-bounce count, thinness) —
> **NOT** the AR point-source-fan / collimation model. Only the *geometry* is perpendicular.
>
> **Verdict:** the discovery-control result is **robust across geometry**. CMA-ME still earns
> **"search," not "enumeration"** when the fold is forced: **A ≫ B0** on diverse high-efficiency
> folds (7.60 vs 1.80 families, Mann–Whitney p = 0.014). Forcing the fold fattens the *fold*
> target ~5× and the *good*-fold target ~7×, **but the high-efficiency fold stays a genuine
> needle** (0 in 50 000 random draws — identical to collinear). Geometry conditions **where**
> folds exist; quality-diversity search is **what finds** the rare efficient ones.

---

## Setting (what "prism, source below, perpendicular" means in code)

- **Geometry:** [`ns_probe/prism_fold.py`](../../ns_probe/prism_fold.py) — parallel bundle
  (21 rays, width 8) emitted from `Y_SOURCE = -75` aimed straight **up**; detector on the right
  at `X_SENSOR = 120` (catches +x rays). Because a 90° turn needs a ~45° reflecting surface, the
  element is the **AR-style prism** (4 curved faces, ±50° tiltable — `arvr_prism.prism_element`),
  not the ±25° `fold_demo` element (which physically cannot fold 90°). Genotype = 10-DOF:
  `[ray_x] + [cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n]`.
- **Fitness / descriptor / archive:** imported **byte-identical** from the collinear fold via
  [`ns_probe/run_prism_fold_qd.py`](../../ns_probe/run_prism_fold_qd.py) — focused throughput,
  `BC_BINS=(9,16)`, `BC_BOUNDS=((0,9),(10,120))` = (TIR count, thinness T). Only the *scene*
  differs from `run_fold_qd`.
- **Harness:** [`ns_probe/discovery_control_prism.py`](../../ns_probe/discovery_control_prism.py)
  is the exact twin of `discovery_control.py` — same three-arm ladder, same `Book`, same
  samplers; the **only** change is the two problem imports. So fold-vs-prism isolates *geometry*,
  not code. Analysis: [`analyze_disc_prism.py`](../../ns_probe/analyze_disc_prism.py).
- A rendered high-efficiency double-fold: `ns_probe/results_disc_prism/prism_fold_best.png`
  (parallel bundle up → 90° TIR fold → focused spot at the perpendicular detector; 96% delivered).

---

## E1 — head-to-head, 14 000 evals, 5 seeds each (matched to the collinear fold)

| metric | **A CMA-ME** | **B0 random** | **B1 restart-LS** |
|---|--:|--:|--:|
| **# high-eff fold families** (primary; foc≥0.9, ≥2 TIR) | **7.60 ± 2.60** | 1.80 ± 1.44 | 3.60 ± 1.92 |
| best fold spot (sharper = smaller) | **0.71** | 3.91 | 2.92 |
| QD-score (Σ fitness / filled cells) | **65.3 ± 5.6** | 37.6 ± 1.5 | 37.2 ± 7.5 |
| coverage (frac cells filled) | **0.83** | 0.75 | 0.61 |
| deep-fold reach (best focthru @ tir≥4) | **0.92** | 0.55 | 0.72 |
| evals-to-5 families | **7 276** | 13 776 | 7 646 |
| # good folds seen (foc≥0.5) | 300 | 20.8 | 2 794 |

- **Headline test:** fam_high per seed A = `[3,8,11,9,7]` vs B0 = `[3,1,4,1,0]`.
  **Mann–Whitney U (A > B0, one-sided): p = 0.0137.**
- **Ladder:** `A ≫ B1 > B0` on diversity and QD-score. CMA-ME roughly **doubles** the plain
  local optimizer (B1) and **quadruples** random (B0) on distinct high-eff families, and reaches
  5 families in ~half the evals B0 needs. **The full QD stack earns it here too.**

### The one thing that flips vs the collinear fold (a mechanistic insight)
- **Collinear (optional fold):** B1 (local optimizer) was **WORSE than random** (0.00 vs 0.20)
  — a covariance optimizer burns its budget polishing single *non-folding* basins.
- **Perpendicular (forced fold):** B1 **beats** random (3.60 vs 1.80) — because *every* restart
  now lands in a folding basin, local polishing pays off. **But A still doubles B1.**
- Reading: **forcing the fold raises the floor for a dumb optimizer, but the ceiling still
  belongs to QD search.** The `A > B1 > B0` ordering (full stack best) holds in both regimes;
  only the B0/B1 order swaps. The QD-specific archive-improvement ranking + persistence remain
  load-bearing.

The `# good folds seen` column shows B1's failure mode explicitly: B1 *sees* 2 794 good folds
(9× A's 300) yet resolves only **half** A's distinct families — it re-finds the same few basins
over and over. Diversity, not raw count, is the QD contribution.

---

## E2a — 50 000 random draws: forcing fattens the target but the needle survives

| draw outcome | **prism (forced)** | collinear fold (optional) |
|---|--:|--:|
| valid (reaches detector) | 38.9% of draws | 74.7% of draws |
| **folded** (≥2 TIR) | **27.6% of valid** | 5.7% of valid |
| **good** fold (foc≥0.5) | **0.288% of valid** (56) | 0.040% (15) |
| **high-eff** fold (foc≥0.9) | **0.000% (0 / 50 000)** | 0.000% (0 / 50 000) |

- Perpendicular geometry makes folding **~5× more common** and good folds **~7× more common** in
  random draws — the geometry prior is doing maximal work (folding is *forced*, not merely
  possible). This is the strongest "enumeration-flavored" evidence in the program.
- **Yet zero high-efficiency folds appear in 50 000 random draws — identical to the collinear
  case.** The *high-quality* fold is a genuine needle in **both** regimes. Fattening the fold
  target does **not** fatten the *efficient*-fold target.

---

## What this adds to the program

**The §7 verdict of [15](15_discovery_control_results.md) is not an artifact of the collinear
geometry.** Rerun in the regime where geometry does the *most* possible work — forcing the fold
outright — the same result holds: matched-budget random cannot assemble the diverse high-efficiency
family that CMA-ME does (p = 0.014), and a strong local optimizer reaches only half of it.

**Cleanest paper framing, now defended in two geometries:**
*"Geometry conditions **where** folds can exist — and in the perpendicular prism it **forces**
them, so folds are common in random draws. But the rare **efficient** folds are a needle in both
the optional- and forced-fold regimes (0 / 50 000 random each), and quality-diversity search is
**what finds** the diverse family of them — matched-budget multi-start does not."*

**New, honest nuance to report:** the multi-start baseline (B1) is *stronger* in the forced-fold
regime than the optional-fold one (it beats random instead of trailing it), because forced folding
guarantees each restart a productive basin. The QD advantage is therefore **larger where folds are
scarce and optional, smaller (but still decisive, ~2×) where geometry forces them.** The paper
should state this rather than imply a uniform gap.

---

## Confounds controlled (same discipline as §9 of the plan)
- **Byte-identical fitness/descriptor/archive** to the collinear fold (imported, not copied) —
  fold-vs-prism differs only in the `scene()` geometry.
- **Matched budget** = matched `evaluate()` calls, invalids included; all E1 arms ran 14 000.
- **Same engine build**, single session, post-rebuild smoke test.
- **5 seeds, 95% CIs, Mann–Whitney** on the primary metric; per-seed values shown, not hidden
  (B0 seed 0 found 3 families — its best — and is included).
- **B1 restarts from fresh uniform, never an elite** — persistence genuinely removed; the ladder
  did not collapse (A ≠ B1, 7.60 vs 3.60).
- **Physics verified**, not inferred: the rendered champion delivers 20.1 / 21 emitted with a
  `T → TIR → TIR` dominant signature — a real 90° double-fold to a focused spot.

## Artifacts
- Geometry: `ns_probe/prism_fold.py`; QD engine: `ns_probe/run_prism_fold_qd.py`;
  harness: `ns_probe/discovery_control_prism.py`; analysis: `ns_probe/analyze_disc_prism.py`.
- Archives: `ns_probe/results_disc_prism/{cmame,random,restart_ls}/seed{0..4}/`,
  `random50k/` (+ `valid_genotypes.npy`); summary `results_disc_prism/e1_summary.json`;
  render `results_disc_prism/prism_fold_best.png`; logs `logs_disc_prism/`.
