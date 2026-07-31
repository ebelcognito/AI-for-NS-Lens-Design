# Descriptor Ablation — which BC axis surfaces more good non-sequential designs?

> Run 2026-06-19 (Opus 4.8). Question (from the user): *"find a few better descriptors that
> result in more / better non-sequential designs."* Held **fitness fixed** (focused-throughput,
> fold-blind) and the genotype/engine/CMA-ME budget fixed; swapped **only the diversity axis (BC)**.
> Code: [run_fold_qd_descriptors.py](../../ns_probe/run_fold_qd_descriptors.py),
> [compare_descriptors.py](../../ns_probe/compare_descriptors.py),
> [descriptor_summary_figure.py](../../ns_probe/descriptor_summary_figure.py).
> Figure: `ns_probe/descriptor_summary.png`; table: `ns_probe/descriptor_comparison.md`.

## TL;DR (the 5 things not to get wrong)
1. **Descriptors set DIVERSITY, not quality.** In this archive, non-sequential (NS) designs survive
   *only* because they occupy their own niche — a single-objective optimizer would discard every fold
   for the transmission lens (focthru→1.0). So "a better descriptor → more good folds" works by giving
   good folds a **protected niche**, not by rewarding them.
2. **A good NS descriptor must be FOLD-DEFINING** — zero/undefined for non-folds and varying across
   fold *structure*. Otherwise transmission/shallow designs share the fold's cell, win on
   focused-throughput, and **evict** the fold. This is the central, mechanistic result.
3. **`locus` (fraction of TIR events on the flat edges T/D) is the best single new descriptor.** Over
   3 seeds it reliably surfaces **~2× the high-efficiency fold families** of the depth-axis baseline
   (mean 3.0 vs 1.7) and a larger good-fold archive (32 vs 25 cells), and finds a good fold at **every**
   seed — baseline found **zero** at one seed.
4. **The four "intuitive" NS-ness axes are WORSE** — continuous NS-fraction, tilt magnitude,
   critical-angle margin, tolerance/robustness all collapse NS-cell count from **80 → single digits**
   (4 / 0 / 5 / 9 at seed 0), because none is fold-defining. Use these as **objectives or post-filters**,
   not as the primary diversity axis.
5. **Reproducibility gotcha (separate but important):** the stored `ns_probe/results_fold_qd/archive.json`
   (the documented headline: 4-TIR spot 0.66) is **NOT reproducible on the current engine at seed 0** —
   a fresh `run_fold_qd.py --seed 0` now yields only 2-TIR folds (best spot 2.81). The refactor is
   faithful (fresh original == `results_desc_baseline` bit-for-bit); the **stored archive is stale**
   (engine drift, the `dist-node`→`integrations` rebuild/cp sync gotcha). See [05](05_probe_phase0_findings.md).

## Setup
- Genotype, engine (uncoated Fresnel+TIR), fitness (focused-throughput, [run_fold_qd.py](../../ns_probe/run_fold_qd.py))
  **all identical** to the headline fold-QD run. Only **BC1** (and optionally BC2) changes.
- BC1 axes tested (`--bc`): `baseline` (dominant-path TIR count), `nsfrac` (energy fraction that took
  ≥1 TIR, continuous), `tilt` (√(tilt_a²+tilt_b²)), `critmargin` (min incidence margin past critical on
  the fold path), `locus` (edge-TIR fraction), `robust` (focthru retained under 3× geometry jitter).
- BC2 = thinness throughout, **except** the combined run `--bc baseline --bc2 locus` (depth × locus).
- 14k evals, CMA-ME 5 emitters. Baseline confirmed bit-identical to `run_fold_qd.py` (seed 0).

## Results

**Panel A — seed-0 NS-cell count (does the axis preserve folds at all?):**

| BC1 axis | NS cells (≥2 TIR) | fold-defining? |
|---|---|---|
| baseline (TIR count) | **80** | yes (it *is* fold depth) |
| nsfrac (continuous NS-ness) | 4 | no |
| tilt magnitude | 0 | no |
| crit-angle margin | 5 | no |
| **locus (edge-TIR frac)** | **66** | yes |
| tolerance/robustness | 9 | no |

Diagnostic confirming the mechanism: in the `nsfrac` archive, of 103 elites with focthru ≥ 0.9, **zero**
have a folded dominant path; the only 4 folded elites sit at ns_frac 0.5–0.85. Fold *depth* (dominant-path
TIR count) and *energy fraction* are different quantities — a design can fold its dominant ray yet route
most bundle energy straight, so it lands in a low-nsfrac bin next to transmission and gets evicted.

**Panel B — quality of surfaced folds, 3-seed mean (focthru ≥ 0.9):**

| axes | NS cells | high-eff fold families | high-eff cells | good cells (≥0.5) | champ spot | champ found |
|---|--:|--:|--:|--:|--:|--:|
| depth × thinness (baseline) | 80 | 1.7 | 4.7 | 25.3 | 2.19 | 2/3 seeds |
| **locus × thinness** | 74 | **3.0** | **6.7** | **32.0** | 2.04 | **3/3** |
| depth × locus (combined) | 32 | 3.0 | 3.0 | 9.3 | **1.91** | **3/3** |

- **`locus × thinness` wins overall:** most high-eff fold families, most good-fold cells, a good fold at
  every seed. Baseline is erratic (seed 1: zero high-eff folds preserved).
- **`depth × locus` (drop thinness)** ties on high-eff families (3.0) and gives the most reliable single
  champion, but a **smaller** archive (32 cells) — it trades thinness diversity for fold-structure
  diversity. A **3-D archive (depth × locus × thinness)** is the natural way to keep both. *(not yet run)*
- Champion-spot differences (1.9–2.2) are within seed noise — locus' win is in **reliability and count of
  high-efficiency folds preserved**, not in finding a fundamentally sharper fold. (The one-seed "locus 1.10
  vs baseline 2.81" first read was noise.)

## Recommendation
- **Adopt `locus` as a primary diversity axis** (best single swap), and run a **3-D `depth × locus ×
  thinness`** archive to keep depth + structure + compactness simultaneously.
- **Do NOT** use non-fold-defining quantities (nsfrac, tilt, critmargin, robustness) as primary BC axes;
  fold them in as **objectives / robustness filters** instead. (Robustness is still the highest-value
  *objective* per [00_MASTER_CONTEXT.md](00_MASTER_CONTEXT.md) §10 — just not as a diversity axis here.)
- **Fix the engine-drift reproducibility issue** before trusting any cross-run absolute numbers: rebuild
  `ray-optics` and re-sync (`npm run build-node && cp dist-node/rayOptics.js integrations/rayOptics.js`),
  then regenerate `results_fold_qd` so the documented headline matches the live engine.

## Reproduce
```bash
PY=~/Desktop/simu1/deeplens_env/bin/python
for bc in baseline nsfrac tilt critmargin locus robust; do
  $PY ns_probe/run_fold_qd_descriptors.py --bc $bc --evals 14000 --seed 0 --out ns_probe/results_desc_$bc
done
$PY ns_probe/run_fold_qd_descriptors.py --bc baseline --bc2 locus --seed 0   # combined depth × locus
$PY ns_probe/compare_descriptors.py
$PY ns_probe/descriptor_summary_figure.py
```
