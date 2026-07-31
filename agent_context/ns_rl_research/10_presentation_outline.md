# Presentation Outline — QD Control of Non-Sequential Optical Design

> A slide-by-slide outline of the ns_probe work (incl. Fable's engine contribution). Numbers are from the
> actual runs; visuals point at the generated figures. Honest about limitations. ~10 slides + backup.

## One-line contribution (the headline)

**We bring Quality-Diversity *illumination* to non-sequential (NS) optical design: by instrumenting an NS ray
tracer to expose *typed ray-surface-interaction sequences* and an *exact energy ledger*, we use QD (CMA-ME) to
discover a *diverse archive of designs organised by their interaction topology* — and we show that efficient,
loss-free *TIR-folded* optics emerge from an imaging objective that never mentions folding — all gradient-free
(no differentiable NS tracer required).**

Three non-trivial parts: (1) **diversity defined over interaction sequences**, (2) **emergent efficient TIR
folds**, (3) **QD substitutes for differentiability**.

---

## Slide 1 — Title + the contribution in one line
Use the headline above. Subtitle: *discovering folded optics by illuminating ray-surface-interaction space.*

## Slide 2 — Why non-sequential design is hard (motivation)
- **Sequential** tracing: surfaces visited in fixed order, transmission only. **Non-sequential**: a ray hits
  *any* surface in *any* order, and at each hit may transmit, partially reflect, or **totally internally
  reflect (TIR)** — paths can fold, retrace, branch.
- The design→performance map is **piecewise-smooth**: tiny parameter changes flip *discrete* decisions (which
  surface next, TIR-or-not, how many bounces). Gradient/local methods settle into **one topology basin** and
  polish — they can't hop topologies. That's why human designers still pick the starting fold.
- Goal: **discover** folded topologies, not prescribe them.

## Slide 3 — Method: QD over interaction sequences
- **QD mechanism = CMA-ME** (MAP-Elites + CMA-ES "improvement emitters"). **How it navigates:** an *archive* is
  indexed by behaviour descriptors; CMA-ES emitters propose designs and are ranked by **archive improvement**
  (fill an empty cell > beat an incumbent > nothing), adapting their search covariance and **restarting from a
  random elite when they stall**. → topology **discontinuities become archive cell walls, not optimizer
  traps**; many basins are illuminated *in parallel*.
- **How we define "diversity":** descriptors computed from the **ray-surface-interaction sequence** — the
  number of **TIR bounces on the dominant-energy path** (fold topology) and **thinness**. The NS analogue of
  classical aberration descriptors. *Diversity = covering fold topologies, each cell holding the best design
  of its kind.*
- **Gradient-free** — emitters need only *evaluations*, not derivatives.

## Slide 4 — What we modified in the NS framework (engineering contribution)
We instrumented a vendored 2D NS ray tracer (open-source "Ray Optics Simulation"), all **flag-gated**
(`recordInteractions`), default path **byte-identical**, **22 physics tests pass**:
- **Typed interaction events** — every surface hit is labelled `refract / TIR / partial-reflect / mirror /
  sensor`, with **energy-in/out per event** and a `fresnel|forced` mechanism tag.
- **Exact energy ledger** — `input = delivered + reflected-away + absorbed + truncated`, closes to **0** →
  honest efficiency, no hand-waving.
- **Interaction-sequence export + target-sequence matching** — you can request *"T@A → TIR@B → TIR@A → T@B"*
  and verify which energy actually takes it.
- **Tilt-enabled (prismatic) geometry** as a design DOF.
- (Earlier) a forced per-surface reflectivity — later recognised as an artificial lossy "coating" crutch; the
  faithful route is TIR.

## Slide 5 — Results I: cylindrically symmetric (on-axis) → folding does NOT pay
- Setup: rotationally-symmetric-ish elements, on-axis collimated beam; QD over (thinness × fold depth), ~18k
  evals.
- **Finding: folding never wins.** Transmission dominates the good-imaging region in **every** thinness niche
  (**0/24** niches won by a fold). Folds that appear are **lossy back-reflections** — spot 6–11, **<20%** of
  the light.
- **Why (the key physics):** the dominant rays are **paraxial → hit axis-symmetric surfaces near-normal →
  cannot TIR**. Only lossy partial reflection is available, so transmission is strictly better.
- Visual: `ns_probe/results/probe_figure.png` (archive heatmap) / `results_single/renders/contact_*.png`.

## Slide 6 — The pivot: TIR is the high-energy fold (and needs broken symmetry)
- **TIR = 100% loss-free reflection** above the critical angle — pure geometry, **no coating, no
  polarization**. It's how periscope/prism folds and light guides work; it carries the **dominant** energy.
- To TIR the **dominant bundle** you must **break on-axis symmetry** — a **prismatic tilt** so the whole beam
  exceeds the critical angle.
- **Verified:** a *single uncoated* tilted element delivering **~76–87%** of an input beam via
  `transmit@A → TIR@B → TIR@A → transmit@B`, energy-audited per event.
- Visual: `results_single/renders/TIR_demo_single_element.png` + the typed-sequence printout.

## Slide 7 — Results II: non-symmetric (tilted) → a diverse family of efficient folds
- Setup: tilt-enabled **uncoated** element; fitness = **focused throughput** (energy delivered into a tight
  window ÷ input — *never names folding*); BC = (TIR-bounce count × thinness). CMA-ME, **14k evals, 109 cells
  (76% coverage), ~7 ms/eval, 105 s**.
- **Finding: efficient, focusing TIR folds emerge — and a *diverse* archive of them.**
  - 2-TIR fold: **96%** throughput, spot 3.7  ·  4-TIR fold: **91%**, spot **0.66**.
  - **81 folded cells**; 21 with ≥50% focused throughput; efficiency stays **>90% up to ~6 bounces**.
  - Unprescribed discovery: folds use the element's **flat edges as TIR light-guides**.
- Contrast: vs the symmetric/coating probes (spot 6–11, <20%) — same QD method, **TIR + tilt + honest
  measurement** is what flipped it.
- Visuals: `results_fold_qd/fold_qd_figure.png` (heatmap + champion-vs-depth), `.../renders/contact_folds.png`
  (the diverse folds, 0→11 bounces).

## Slide 8 — Why it matters (the two benefits)
- **Benefit of TIR control:** *loss-free* folding → **compact** (long optical path in a thin element) **AND
  efficient** (~all energy stays in the dominant path) — with **no coatings and no polarization**. This is the
  realistic, high-efficiency thin-fold regime.
- **Benefit of QD control:** you get a **diversity of fold topologies** (an *archive*, not one point design);
  **emergence** (folding from a quality objective, not prescribed); **robustness** to the discontinuous NS
  landscape (topologies become cell walls); and it's **fast & gradient-free**.

## Slide 9 — Do we need a differentiable NS ray tracer? (the methodological point)
- **No — not for discovery/verification/illumination.** Gradient-free QD found *and verified* a diverse,
  efficient fold family in **minutes**. The QD signal (archive-improvement ranking + emitter adaptation) is
  enough to navigate the discontinuous space.
- A differentiable NS tracer would only help **local refinement** (gradient-polishing a candidate) — a
  *later, optional* step. Build it (as a DeepLens extension) **only when refinement is the bottleneck**, after
  the cheap black-box refinement experiment.
- *Takeaway: QD compensates for the lack of differentiability; differentiability is a refinement luxury, not a
  discovery requirement.*

## Slide 10 — Extensions & directions (incl. Fable's)
- **Can it be extended? Yes — because the framework is interaction-sequence-*general* and modular:**
  - More surfaces / **multi-element** NS systems; **multi-field / off-axis** imaging (real eyebox/FOV).
  - **Local refinement** of folded candidates (cheap, no new engine) — the immediate next step.
  - **Target arbitrary fold topologies** (sequence-as-input is already built).
  - **Polarization (Jones upgrade)** → the **pancake** family: QWP + reflective polarizer + 50/50 mirror.
    *Note:* a QWP/HWP is **not implementable today** — the engine is intensity-only (no phase); it needs a
    complex Jones-vector upgrade (Tier B), and only for the *lossy* pancake route, **not** the efficient TIR
    route.
  - Eventually a **differentiable NS tracer** for gradient refinement — gated on need.

## Backup / honesty slide (have it ready)
- 2D meridional (not full 3D); **single element**; **on-axis single field** — off-axis/FOV imaging untested.
- "Focused throughput" is an **imaging proxy**, not full multi-field MTF; "thinness" = device depth, not
  FoV-normalized.
- The symmetric "negative" also carried a coating confound — the honest statement is "symmetric geometry
  offers *only* lossy folding (no dominant TIR), so transmission wins."

---

## Key numbers (for slides)
- Engine: typed events + energy ledger (closes to 0), **22/22 physics tests pass**, default byte-identical.
- Symmetric probe: 18k evals, **0/24** niches won by folds, folds spot 6–11 at <20% flux; best transmission
  spot 0.08–0.20.
- TIR demonstrator: single uncoated tilted element, `T→TIR→TIR→T`, **~76–87%** energy, no coating/polarization.
- Fold-QD: **14k evals, 109 cells, 76% coverage, ~7 ms/eval**; 2-TIR **96%** (spot 3.7), 4-TIR **91%** (spot
  **0.66**); **81 folded cells**, efficiency >90% to ~6 bounces.
- QD = CMA-ME (MAP-Elites + CMA-ES improvement emitters), gradient-free.

## Which doc backs each slide
2→[02](02_state_and_directions.md)/[03](03_design_discussion.md) · 3→[03](03_design_discussion.md) ·
4→[06](06_reflectivity_tir_analysis.md)+Fable's `07_typed_interactions_energy_ledger.md` ·
5→[05](05_probe_phase0_findings.md) · 6→[06](06_reflectivity_tir_analysis.md)+[07](07_tir_engine_prompt_fable.md) ·
7→[08](08_fold_qd_diverse_efficient_folds.md) · 9→[03](03_design_discussion.md) · 10→[09](09_polarization_pancake_assessment.md)+[03](03_design_discussion.md).
