# Three Follow-up Experiments: Tilt Ablation, Multi-Field, Physics Validation

> Run 2026-06-12 (Opus 4.8), autonomously, to strengthen the [08](08_fold_qd_diverse_efficient_folds.md)
> result for a presentation. All non-overwriting (separate `results_*` dirs + sidecars). These (A) make the
> symmetry-breaking claim *causal*, (B) address the biggest caveat (single-field), and (C) answer the "is it
> really doing physics?" critique with evidence.

## A. Tilt ablation — prismatic tilt is *causal* for focusing folds

Re-ran the fold-QD identically but with the face tilts **locked to 0** (a rotationally-symmetric-ish element):
`python3 ns_probe/run_fold_qd.py --lock-tilt --out ns_probe/results_fold_qd_notilt`.

| | folded cells | efficient+sharp folds (eff≥0.7, spot≤2) | best fold efficiency | best fold spot |
|---|---|---|---|---|
| **WITH tilt** (asymmetric) | 81 | **3** | 0.97 | **0.08** |
| **NO tilt** (symmetric) | 72 | **0** | 0.91 | **5.13** |

**Conclusion:** without the prismatic DOF, folds still *form* and can be *efficient* (energy gets through —
72 folded cells, eff up to 0.91), but they **cannot focus** — the best fold spot collapses from **0.08 → 5.13**
and there are **zero** efficient+sharp folds. So the symmetry-breaking tilt is specifically what lets the
*whole bundle* TIR coherently and converge. This turns the slide-5-vs-slide-7 story into a controlled
ablation: remove one DOF, the focusing-fold family disappears.

## B. Multi-field — the folds are not single-field flukes

New objective: each design is scored on **mean focused throughput over input fields {−3°, 0°, +3°}** (a real
imaging property), BC = (TIR depth × thinness). 12k evals, 106 cells.
`ns_probe/run_multifield_qd.py`, archive `results_multifield_qd/`.

- **8 field-robust folds** (worst-field throughput ≥ 0.6). The genuinely sharp ones:
  - 2-TIR fold: per-field **[0.78, 0.88, 0.87]**, central spot **1.5** — `T@A → TIR@D → TIR@B`.
  - 3-TIR fold: per-field **[0.79, 0.79, 0.80]**, central spot **1.5** — `T@A → TIR@B → TIR@D → TIR@A → T@B`.
- **Finding:** when the objective demands several fields, the search finds folds that hold **~80% throughput
  and ~1.5-spot focus across ±3°** — i.e. folds that image a *range*, not a single-field trick.
- **Honest caveats:** the field range tested is small (±3°); throughput/sharpness are lower than the
  single-field champions (96% / spot 0.66), and spot grows off-axis. So: *folds are usable imaging elements
  over a modest field, with field-dependent aberration* — the expected behaviour of a single tilted element,
  and the motivation for multi-element / freeform follow-ups.
- Render of one fold imaging all three fields: `results_multifield_qd/renders/multifield_fold_3fields.png`.

## C. Physics validation — the discovered folds respect and exploit the TIR physics

For every fold in the archives, re-traced the dominant ray and, at each **engine-labelled TIR event**,
computed the geometric angle of incidence (from the ray's turn) and compared it to the material's critical
angle `θ_c = arcsin(1/n)`. `ns_probe/analyze_physics.py`, figure
`results_fold_qd/physics_validation.png` (3 panels).

- **143 folded designs, 697 TIR events. 100% of them sit past the critical angle** (`θ_i > θ_c`) — every
  reflection the engine calls "TIR" genuinely satisfies the TIR condition.
- **Margin past critical:** median **17.8°** (10th pct **2.4°**, 90th **41.1°**). The search drives geometry
  into the TIR regime and parks designs in a band from *just past* the boundary to *well past* it.
- **Efficiency vs fold depth** falls from ~0.75 (2–3 bounces) toward ~0.05 (9–10 bounces) — loss-free TIR keeps
  shallow folds efficient; deeper chains pay Fresnel entry/exit + aperture losses. A clean physical tradeoff.

**Why this matters for the talk:** it directly answers *"is it just searching, or does it respect the
physics?"* The optimizer doesn't *reason* about optics, but the discovered designs (a) all obey the
critical-angle law, (b) live in a meaningful, quantifiable band past it, and (c) show the expected
efficiency-vs-depth physics. The search reliably lands in — and exploits — the real TIR regime; the *insight*
(tilt enables it, A) is human, the *discovery and the physical map* are the machine's.

## Slide-ready takeaways
- **`physics_validation.png`** — the strongest single figure for the "does it understand physics?" question.
- **Ablation table (A)** — one-line causal claim: tilt 0 → best fold spot 0.08 → 5.13.
- **Multi-field (B)** — honest robustness: folds image ±3° at ~80% / spot ~1.5; field-limited as expected.
