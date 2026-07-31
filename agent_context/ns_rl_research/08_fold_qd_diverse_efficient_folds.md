# Fold-QD — A Diverse Archive of Efficient, Focusing TIR Folds (the first positive result)

> Run 2026-06-12 (Opus 4.8) on Fable's typed-interaction + energy-ledger engine. This is the **payoff** of the
> whole arc: with the realistic mechanism (TIR, no coating) and the right geometry (prismatic tilt), the QD
> search discovers a *diverse archive of efficient folds that also focus* — unlike the earlier probes
> ([05](05_probe_phase0_findings.md)) whose folds were lossy back-reflection ghosts.
>
> Code: [ns_probe/run_fold_qd.py](../../ns_probe/run_fold_qd.py), [analyze_fold_qd.py](../../ns_probe/analyze_fold_qd.py).
> Archive: `ns_probe/results_fold_qd/archive.json`; figure `fold_qd_figure.png`; renders
> `results_fold_qd/renders/contact_folds.png`.

## Setup

- **Genotype:** the 9-DOF *tilt-enabled* single element from [fold_demo.py](../../ns_probe/fold_demo.py)
  (`ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n`). **Uncoated** — every surface event is honest
  Fresnel + TIR; the only loss-free reflection is geometric TIR.
- **Fitness = focused throughput:** energy (from the exact ledger's `emitted`) delivered to the sensor within
  a ±8-unit window of the energy centroid. Rewards routing a lot of energy AND concentrating it. **Names no
  folding/TIR/thinness** — any fold that appears emerged, it was not prescribed. (Fixed a double-count bug:
  each ray's energy is counted once, at its first sensor crossing; fitness is now correctly ≤ 1.)
- **Behaviour axes (descriptive, from the engine's TYPED events):** BC1 = number of TIR bounces on the
  dominant-energy interaction sequence (fold depth); BC2 = thinness `T = X_sensor − frontmost optic`.
- **Solver:** CMA-ME (5 improvement emitters), 14,000 evals, 109 cells (76% coverage), 105 s, ~7 ms/eval.

## Result — efficient TIR folds emerge AND focus

Champion (best focused-throughput) at each fold depth:

| TIR bounces | cells | focused-thru | efficiency | spot | T | dominant interaction sequence |
|---|---|---|---|---|---|---|
| 0 (transmission) | 15 | 1.000 | 1.00 | 2.42 | 120 | (straight) |
| 1 | 13 | 0.962 | 0.97 | 1.58 | 119 | TIR@D → T@B |
| **2** | 12 | **0.960** | 0.96 | 3.73 | 67 | T@A → TIR@D → TIR@B |
| 3 | 12 | 0.907 | 0.91 | 2.42 | 48 | T@A → TIR@D → TIR@B → TIR@B → T@B |
| **4** | 13 | **0.914** | 0.91 | **0.66** | 88 | T@A → TIR@D → TIR@B → TIR@A → TIR@T → T@B |
| 5 | 11 | 0.334 | 0.59 | 16.5 | 56 | … |
| 6 | 12 | 0.732 | 0.92 | 13.4 | 118 | (7-event TIR chain) |
| 7–11 | 31 | 0.04–0.42 | 0.10–0.42 | spread | — | (deep chains, degrade) |

- **81 folded cells (≥2 TIR); 21 with focused-throughput ≥ 0.5.** A genuinely diverse archive of efficient
  folds, not a single lucky design.
- **Folds focus.** The 4-TIR champion concentrates **91% of the input to spot 0.66**; the 2-TIR champion
  delivers **96% at spot 3.7**. Contrast [05](05_probe_phase0_findings.md): the earlier (coating-driven,
  on-axis) folds imaged at spot 6–11 with flux < 0.2. The difference is exactly the three fixes from
  [06](06_reflectivity_tir_analysis.md): **TIR (loss-free) instead of a lossy coating, prismatic tilt so the
  bundle can exceed the critical angle, and typed/ledgered measurement** instead of inference.
- **Efficiency stays > 0.9 up to ~6 bounces, then degrades** (more Fresnel entry/exit loss + harder to keep a
  deep chain focused). Fold depths 0–11 are all populated — the QD axis spans the diversity.
- The folds make heavy use of the element's **flat top/bottom edges** (`D`/`T` events) as additional TIR
  surfaces — an unprescribed structural discovery (the optimizer found edge-TIR light-guiding).

## What this establishes / caveats

- **First positive emergence result in the program:** the realistic high-energy fold (loss-free TIR) is *not
  only simulatable but QD-discoverable as a diverse, efficient, focusing family*, from an objective that never
  mentions folding. This is the multi-bounce generalization of the original single-element TIR result, now at
  bundle scale with honest energy accounting.
- **Caveats (next steps):** single element, **on-axis single field** — imaging quality across a FOV/eyebox is
  untested (a tilted/prismatic element will have field-dependent aberration). "Thinness" here is device depth,
  not a true FoV-normalized compactness. No refinement step (CMA-ME explore only). Multi-element and
  multi-field TIR folds, and the cheap local-refinement experiment from [05](05_probe_phase0_findings.md), are
  the natural follow-ups.
- **Polarization is still not involved and not needed** — see [09_polarization_pancake_assessment.md](09_polarization_pancake_assessment.md).

Reproduce: `python3 ns_probe/run_fold_qd.py --evals 14000 --seed 0` then `python3 ns_probe/analyze_fold_qd.py`.
