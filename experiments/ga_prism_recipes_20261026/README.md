# Our quad optimizer recipes versus the two-stage GA (Su & Romanova 2026), 2026-10-26

Same prism, tracer, 26 variables, bounds and in-spec rule as `../ga_prism_replication_20261026` (100% throughput, zero
wedge penalty, C-image separation ≥ 1 mm, |W − 12 mm| ≤ 1 mm). Same budget as the GA's two-stage arm: **19,000
evaluations**. 9 starts = the paper's own robustness variants (R_B ∈ {−56, −80, −90} × R_A ∈ {132, 150, 200}).
Recipe definitions are in `run.py`'s docstring (ported from `quad_imaging_env`, `quad_focus_recipe`, `quad_cma_control`).

| Method | Budget | In spec | Center RMS | **Worst-field RMS** | Mean RMS | s/run |
|---|---:|---:|---:|---:|---:|---:|
| **CMA σ = 0.1** | 19k | 9/9 | 30 µm | **64 µm** | 43 µm | 51 |
| CMA σ = 0.02 | 19k | 9/9 | 35 | 73 | 50 | 45 |
| CMA IPOP | 19k | 9/9 | 41 | 78 | 54 | 29 |
| CMA portfolio | 19k | 9/9 | 40 | 81 | 56 | 52 |
| STAGE (CMA geometry → TRF freeform) | 19k | 9/9 | 60 | 97 | 75 | 32 |
| CMA adaptive | 19k | 9/9 | 38 | 106 | 66 | 52 |
| GA two-stage, 300 gen (paper's full run) | 69k | 50/50 | 28 | 168 | 123 | 142 |
| GA Stage-2 only, 50 gen | 10k | 50/50 | 52 | 235 | 158 | 21 |
| **GA two-stage, 50 gen (same budget)** | 19k | 50/50 | 37 | **248** | 178 | 39 |
| TRF with restarts | 19k | 8/9 | 48 | 265 | 187 | 37 |
| TRF all genes | 19k | 7/9 | 119 | 347 | 250 | 2 |
| GA Stage 1 → TRF | 19k | 9/9 | 688 | 759 | 640 | 23 |
| TRF freeform + image only | 19k | 1/9 | 873 | 1227 | 906 | 3 |

Medians over in-spec runs. Figure: `recipes_vs_ga.png`. Raw: `runs/`, `RESULTS.json`.

**Fairness check** (`ga_best_worst.py`): the GA selects by center RMS, the recipes balance all fields. Taking the GA's
*best worst-field* in-spec member of its final population instead (18 reruns per arm): 209 µm median at 19k, 120 µm at
69k. CMA σ = 0.1 still wins 3.3× at equal budget and 1.9× against the GA with 3.6× the budget.

## Findings

1. **Plain CMA from the paper's seed beats the two-stage GA ~4× on the worst field at the same budget, with no loss on
   the center field** (30 vs 37 µm). Every CMA policy beats the paper's full 69k-evaluation run on the worst field.
2. **The GA's Stage 1 is unnecessary for CMA.** CMA starts from the Table 1 seed (all rays trace, wedge violated) and
   fixes feasibility itself. 54/54 CMA runs ended in spec.
3. **Local TRF alone is trapped.** From these seeds it converges in ~700 evaluations to 0.25–0.5 mm worst-field blur;
   restarts barely help. This prism, unlike our three-mirror relays, has a rugged landscape where TRF needs a global stage.
4. **Stage 1 → TRF is the worst hybrid** (688 µm): Stage 1 optimises mean RMS with no image-width term, so it hands TRF a
   ~24 mm-wide image it cannot bring back. Stage 1 is a bad warm start for anything but its own Stage 2.
5. **Freezing the body fails** (TRF_ff 1/9 in spec): the wedge violation lives in the geometry, not the freeform terms.
6. Same story as the quads (`quad_cma_control`, 2026-10-16): fixed CMA settings are as good as adaptive control;
   CMA adaptive is the weakest CMA policy here.

## Caveats

- One objective weighting for all recipes (r·r of field RMS + penalties); the GA optimises three objectives in a Pareto
  sense. The fairness check above covers the obvious version of that gap.
- Penalties aim 20 µm inside the wedge walls and 0.5 mm inside the width tolerance; without that, TRF parks on the boundary
  and is never in spec. The spec itself is unchanged.
- The checkpoint curves in `RESULTS.json` only log exact evaluation counts, so runs that stop early show `inf`.
- 9 starts per recipe (54 CMA runs) versus 50 GA runs.
