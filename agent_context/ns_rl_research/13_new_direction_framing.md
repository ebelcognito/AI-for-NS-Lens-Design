# New-Direction Framing — the paper's question, the discovery-control crux, and the three tracks

> **What this is.** A synthesis doc written 2026-07-10 (Opus 4.8) to convert the researcher's
> latest round of thinking into a defensible paper frame. It answers, in order: (1) the four
> framing questions (scientific question → hypothesis → math → what the paper accomplishes);
> (2) **THE number-one question** the researcher flagged — *do we have control over the discovery
> mechanism, or is multi-start as good as this?*; (3) the refinability question (*is every design
> polishable?*); (4) the three candidate tracks with a recommendation; (5) the prior-art tethers
> we are missing; (6) the Cheng et al. / OST-HMD tie-in. Grounded throughout in the actual
> experiments in [00_MASTER_CONTEXT.md](00_MASTER_CONTEXT.md), the refinement work
> ([../ns_differentiable_refinement/](../ns_differentiable_refinement/)), and the descriptor
> ablation ([12_descriptor_ablation.md](12_descriptor_ablation.md)).
>
> **Honesty stance.** Everything below is tagged **[tested]**, **[partially tested]**, or
> **[untested / open]**. The single most consequential item in this doc — CMA-ME vs multi-start —
> is **[untested]**, and the paper's "intelligent discovery" claim rests on it. Do not paper over
> that.

---

## 0. Objective → Approach → Outcome (the one-paragraph spine)

- **Objective.** Produce *good non-sequential (NS) optical designs* — folded, TIR, multi-bounce
  relays of the AR/VR-combiner class — **by discovery, not by prescribing the topology**, and then
  polish the promising ones to a user-defined spec.
- **Approach.** A two-stage pipeline. **Stage 1 (discover):** quality-diversity search (CMA-ME)
  over continuous surface parameters, on a physics-honest NS tracer instrumented to expose *typed
  ray-surface interaction sequences* + an *exact energy ledger*; the archive is indexed by the
  **interaction topology** (fold count, edge-TIR locus, thinness), so discrete topology jumps
  become *archive cell walls, not optimizer traps*, and a diverse family of designs is illuminated
  in parallel. **Stage 2 (refine):** freeze one candidate's topology and run guarded differentiable
  local refinement (FD/analytic-IFT Jacobian + Levenberg–Marquardt, freeform coefficients as the
  gradient DOF) to convert a coarse archive entry into a sharp, spec-satisfying design.
- **Outcome.** [tested] Efficient loss-free **TIR folds emerge from an imaging-only objective that
  never names folding** (2–4 TIR at 91–96 % throughput; AR prism collimates 3 fields to σ<0.1°);
  frozen-topology refinement then sharpens them (fold spot 2.33→**0.028 mm, 20×**; prism σ up to
  **+78 %**). The artifact is a *diverse, physics-verified, topology-structured archive of
  machine-discovered NS designs*, a class no prior lens dataset covers.

The four framing questions map onto this spine and are answered precisely in §1.

---

## 1. The four framing questions, answered

### 1.1 What is the scientific question?
**"How do we discover good non-sequential lens designs — folded/TIR relays — without a human
prescribing the fold topology, and then realize them to spec?"**

Sharpen it past the researcher's one-liner, because the sharp version is what makes it a *paper*
and not an engineering note: classical lens design takes the **topology as an input** (a human
picks the fold skeleton; the optimizer polishes geometry to a local optimum). The scientific
question is whether the **topology itself can be an output** of an automated search that only ever
sees an imaging-plus-compactness-plus-efficiency objective — and whether the resulting family is
*diverse, physical, and refinable* rather than a single lucky point.

### 1.2 What is the hypothesis?
Two claims, and it matters which one the paper actually stands on:

- **H1 (emergence — [tested, holds]).** Given an NS tracer and a fitness that rewards only
  *focused throughput* (imaging × efficiency) with compactness pressure on a diversity axis,
  **useful folded/TIR topologies emerge unprompted**, because under a thinness demand with broken
  symmetry, a loss-free fold (TIR) is the *physically optimal* way to satisfy the objective. This is
  the generalization of the single-element PPO-TIR existence proof to multi-surface NS systems, and
  it is **confirmed** (fold-QD, AR-prism off-axis emergence, tilt ablation showing symmetry-breaking
  is causal).
- **H2 (intelligent, controllable discovery — [untested], the load-bearing risk).** The
  *evolutionary* search (CMA-ME: persistence + covariance adaptation + archive-improvement ranking)
  **discovers this diverse family more efficiently and more completely than naive multi-start**, and
  the researcher has *control* over what it finds via the descriptor axes and geometric priors. **This
  is not yet demonstrated.** See §2 — it is the number-one open question and the decisive experiment.

The researcher's stated hypothesis ("sampling-based methods that adapt toward successful design
paths generate multiple solutions efficiently … coupled with a fast evaluator and differentiable
engine … rapidly characterise and refine") is **H1 + H2 fused**. Split them in the writing: H1 is
your bankable result; H2 is the claim a reviewer will attack and the one experiment in §2 must
defend.

### 1.3 How do we formulate it mathematically?

Three pieces, each already instantiated in code:

1. **Design representation.** A design is a continuous genotype `θ` of surface parameters. Faces are
   **conic + polynomial freeform arcs** (`c=−1/R`, conic `k`, coeffs `aᵢ`), positioned by tilts and
   thicknesses. For the fold: 9 base DOF `(ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n)` +
   4 aspheric. For the AR prism: 9 base (4 face sags + 2 tilts + index + geometry) + 8 aspheric.
   *(The researcher's "spline config" is already realized as the parametric-arc + radial-bump
   surface in [`geometry_prism_ff.py`](../../ns_autodiff/geometry_prism_ff.py); a callable-sag spline
   slot exists in `ray_freeform_roots` if we want true B-splines later.)*

2. **The forward map, made differentiable per-topology.** A ray's trace splits into **control flow**
   (which surface next; transmit / reflect / TIR — *discrete*) and **geometry** (intersection points,
   directions, energies given a fixed sequence — *smooth in `θ`*). Conditioned on a realized
   interaction sequence, spot position / OPL / energy are differentiable w.r.t. `θ`; the intersection
   gradient comes from the **implicit function theorem** on the ray-surface root
   (`dt/dp = −F_p/F_t`, one live Newton step after a detached bracket — validated to ~1e-8 vs finite
   differences). Discreteness bites only at measure-zero boundaries (first-hit surface switch, TIR
   critical-angle crossing). **This is the mathematical justification of the "prior/order" idea the
   researcher keeps returning to:** establishing the interaction order *freezes the control flow*, and
   backprop through the unrolled fixed graph is then exact and finite — the "gradients from the second
   interaction fighting the first" is real but is *ill-conditioning* (κ(J) ~1e8–1e11), not
   instability, and Levenberg–Marquardt damping `(JᵀJ+λI)` absorbs it (the same trick Code V/Zemax
   have used for 60 years).

3. **The search objective + diversity index (the QD formulation).**
   - **Fitness (quality)** `f(θ)` = *focused throughput* = fraction of emitted energy (from the
     ledger) delivered into a tight window about the energy centroid (fold), or `mean_s[T_s·exp(−σ_s/σ0)]`
     for the collimating prism. **It never references fold / TIR / bounce count** — that is what makes
     emergence non-circular.
   - **Descriptors (diversity)** `b(θ)` measured *from the typed sequence*: fold count on the
     dominant path, edge-TIR **locus**, thinness, FOV. These define the archive grid `A`.
   - **The QD problem:** fill each cell `k` of `A` with `argmax_{θ: b(θ)∈cell k} f(θ)`. CMA-ME
     approximates this with a pool of CMA-ES emitters ranked by **archive improvement** (new cell >
     beat incumbent > nothing), adapting `N(m, σ²C)` and restarting from a random elite on stall.

   The clean, non-obvious formal point: **the descriptor discretizes the discontinuous axis of the
   problem.** Topology jumps live *between* cells; within a cell the objective is smooth. QD owns the
   combinatorial structure; refinement owns the continuous geometry inside a cell. (This is why the
   two-stage split is principled, not just convenient — [00_differentiable_refinement.md](../ns_differentiable_refinement/00_differentiable_refinement.md).)

### 1.4 What does the paper accomplish?
- **Rediscovery (calibration).** Recover known AR/VR freeform-prism / TIR-relay skeletons from a
  fold-blind objective — the NS analogue of HyperLens recovering the Cooke triplet / Double Gauss,
  and the honest way to argue the method finds *real* designs, not artifacts. [partially tested — we
  produce off-axis TIR relays of the right class; a named-patent rediscovery match is not yet scored.]
- **Discovery.** Surface *novel* folded topologies (different bounce structure, edge-TIR
  light-guiding we never specified) that meet compactness + efficiency bars. [tested at proof-of-concept
  scale.]
- **The artifact.** A **diverse, physics-verified, topology-structured dataset of machine-discovered
  NS designs** — novelty daylight vs LensGAN / Côté (corpora of *human-designed sequential refractive*
  lenses). [tested — this is the strongest standalone claim.]
- **The realization.** Show 2–3 discovered designs refined to a stated FOV / eyebox / collimation
  spec, demonstrating the pipeline is *end-to-end useful*, not just a gallery. [refinement machinery
  tested; the spec-locked demonstration is the deliverable to produce.]

---

## 2. THE number-one question: do we have control over the discovery mechanism?

The researcher's sharpest question, verbatim intent: *is there anything truly intelligent here, or
is multi-start as good as this? Is the geometry of the problem already confining it to a small good
solution space?*

**Honest current answer: most of the "discovery" we can point to today is explained by physics +
human-set geometric priors, and the specific value of CMA-ME over multi-start is UNTESTED. This is
the paper's biggest exposed flank.** Break it into what *controls* discovery, in decreasing order of
evidenced importance:

### 2.1 What actually controls what gets found (ranked by evidence)

1. **Geometric preconditions set by the human — [tested, dominant].** The single biggest lever is
   not the optimizer. Folds emerge **only when the geometry makes a straight path impossible**:
   on-axis AR layout → folds do *not* emerge (**12/482** cells folded, best design is a
   TIR=0 transmission collimator); off-axis layout (display above, eye forward, ~90° turn required) →
   folds *dominate* (**488/493** cells, best design is a fold). And the fold only becomes representable
   once **tilt range is widened ±25°→±50°** so a ~45° TIR surface can exist (fitness 0.05→0.63, the
   search then lands `tilt_a≈45°` exactly). So a large fraction of "discovery" is the human encoding
   *off-axis + adequate tilt range*, after which **physics forces TIR** (100 % of 697 TIR events sit
   past the critical angle; the tilt ablation shows symmetry-breaking is *causal* for sharp+efficient
   folds). **Read plainly: the geometry IS confining the good solution space** — and for the *emergence*
   claim that is fine (physics doing the work is the point), but it means the optimizer is not where
   the intelligence lives *yet*.

2. **The fitness — [tested].** Focused-throughput + compactness pressure is what makes a
   non-transmission design *worth* anything; without thinness pressure the good region is all
   transmission and folding never competes. This is a *design choice*, controllable, and honest.

3. **The descriptor / BC axis — [tested, but narrower than it looks].** The descriptor controls
   **diversity, not quality**: a single-objective run would evict every fold for the transmission lens
   (focthru→1.0); folds survive *only because they occupy a protected niche*. And most "intuitive"
   NS-ness axes (continuous NS-fraction, tilt magnitude, crit-margin, robustness) **collapse** the fold
   archive (80 → single digits) because they are not *fold-defining*; only **fold count** and **edge-TIR
   locus** reliably surface fold families (locus ~2× the high-efficiency families of the depth
   baseline). So we *do* have a real, demonstrated control knob here — but it controls *which of the
   physically-available folds are preserved and separated*, not whether good folds exist.

4. **The optimizer (CMA-ME) itself — [UNTESTED vs the honest baseline].** What CMA-ME *plausibly* adds
   over naive multi-start: (a) **persistence** — a behaviorally-diverse but currently-poor design is
   kept because its cell is empty, and later yields the run's best (HyperLens's mechanism: a 924 µm
   ancestor → 2.7 µm champion); (b) **covariance adaptation** — efficient local hill-climbing once a
   basin is entered; (c) **parallel illumination** of many topology cells from one run. But **none of
   this has been measured against the control a reviewer will demand.**

### 2.2 The decisive experiment (do this before claiming "intelligent discovery")

**CMA-ME vs multi-start, matched evaluation budget, same archive.** Run, at identical eval count on
the identical fitness/genotype/engine:
- **(A) CMA-ME** (current: 5 improvement emitters, restarts).
- **(B) Multi-start + archive-insert:** draw `θ` i.i.d. from the prior, evaluate, insert into the same
  GridArchive by the same rule, no covariance adaptation, no persistence-driven restart. (This is the
  fairest "dumb" baseline — it still gets the niche-protection benefit, isolating the *evolutionary*
  contribution.)
- **(C, optional) Random-restart local search:** multi-start seeds + a few LM steps each (isolates
  "does adaptation matter beyond niching").

**Metrics:** evaluations-to-coverage, final QD-score, count of **ε-distinct high-efficiency fold
families** found, and best spot/σ per fold-count. **Decision rule:** if (B) matches (A) on ε-distinct
high-quality families at matched budget, then *for this problem the intelligence is in the
physics+prior+archive, not the evolutionary sampler* — and the honest paper claim becomes
"**geometry-conditioned QD illumination**," dropping any "smart search" language. If (A) beats (B)
meaningfully (esp. on deep/rare topologies via persistence), you have earned H2 and a real
"evolutionary discovery" claim. **Either outcome is publishable; only one of them is defensible to
claim, and right now we don't know which.** This is the highest-value single experiment in the whole
program — it is cheap (hours, existing code: `run_fold_qd.py` + a multi-start harness) and it directly
answers the researcher's number-one question.

### 2.3 Reframe if multi-start wins
If the geometry confines the good set to a small manifold that multi-start also finds, **that is not
a failure — it is a different, arguably stronger, paper**: *"the space of good NS AR-relay topologies
is small and physics-determined; here is a QD method that enumerates and structures it, and a
refinement stage that realizes any chosen member to spec."* The novelty then lives in **the artifact
(structured, verified, refinable topology atlas)** and **the discover→freeze→refine pipeline**, not in
"our search is smart." Keep both stories drafted; let §2.2 pick.

---

## 3. Is every design "polishable"? — the refinability question

The researcher's other central question. **Answer: no — and refinability is a *measurable* property,
which is the useful finding.** [tested]

- **The predictor is dominant-share.** Freeze a candidate to the rays on its dominant interaction
  signature; the fraction of energy there predicts refinement outcome. High-share champions refine
  **73–80 %** in RMS spot (FD+LM through the JS engine); the messy cell 2,8 champion (46 % dominant
  share) barely moves (−3 %) because the blur lives in the energy that *isn't* roughly focused. On the
  prism, degenerate all-ghost cells (bright eyebox share 0.000, ‖grad‖≈0) are simply unrefinable, while
  healthy cells (bright share 0.22) had fitness-improving steps available. **So "roughly focused with
  significant delivered energy" is not a nice-to-have; dominant-share is a quantitative refinability
  gate.**
- **What blocks the rest is *order*, not physics.** Under a strict order-preservation guard, multi-fold
  prisms yield 0 steps — but instrumenting shows this is *bimodal*: healthy multi-fold relays had good
  steps rejected *only* because they changed the interaction order; `--allow-order-change`
  (accept + re-freeze topology) unlocks them (σ 0.477→**0.002**, 0.729→0.252, 4.657→2.252). So the
  guard, not the physics, was the ceiling.
- **Fewer working surfaces → easier refine — and this matches the literature.** Our own fold-count
  study found **1 fold is the quality sweet spot; multi-fold collimate poorly** (each TIR adds spread
  the next surface only partly corrects). Cheng et al. say the same about their OST-HMD prism: *"the
  present main prism has only two freeform surfaces, S4 and S5, which reduces the difficulty of
  optimization."* This is a real tether (§6) and a design principle we can *state*, not just search.

**Consequence for the tracks.** Refinability is characterizable (dominant-share, Jacobian
conditioning, headroom, fold count). That makes the researcher's "identify refinable designs" idea
concrete — but see §4 for the honest caveat about whether a *learned predictor* earns its place.

---

## 4. The three tracks — and a recommendation

### Track 1 (simple, recommended first): the pipeline as a solid instrument
**Refine 2–3 novel discovered designs to a user-defined AR spec, as well as possible, and characterize
what made them refinable.** Emphasis on the search's *diversity* + the *discover→freeze→refine*
pipeline + refinability characterization (dominant-share). This is **the lowest-risk, highest-ready**
option: the refinement machinery already gives 20×/+78 % wins, refinability has a measurable predictor,
and it does *not* require winning the CMA-ME-vs-multi-start argument to be true (though you should still
run §2.2 to know how to *phrase* the search contribution). Sub-idea worth testing:
**combine sequential lens design with the discovered topology** — freeze the fold skeleton, hand the
refractive geometry to a classical sequential optimizer.

### Track 2 (advanced, defer): dataset + refinability predictor / generative proposer
Generate a large archive, run a **fast proxy refiner**, characterize a refinement-path behavior
(ease-of-optimization / trajectory features / topology), train a network to **classify priors by
refinability** or to **propose** designs from user specs. **Honest caveat the researcher already spotted:**
*if the proxy refiner is cheap and good, why need a predictor?* At current cost (fold ~14 ms/eval,
prism ~150 ms/eval; a full archive in minutes) the proxy refiner **is** cheap, so a predictor is
**premature** — it earns its place only when (a) evaluation gets expensive (3D, multi-wavelength,
tolerancing, 64-ray branching) or (b) you want **amortized inverse design** across many user specs
(the Path A generative proposer). The "bitter-lesson / use trajectories as features" instinct is
sound *in the expensive regime*; state it as motivation, build it only after Track 1 and after §2.2,
and only once the economics justify it. **Do not let Track 2 block Track 1.**

### Track 3 (the honesty/robustness axis — highest-value *addition* to Track 1)
The one axis a Google optical engineer (Ozan) cares about most and no lens dataset has: **tolerance /
manufacturability robustness.** Add it as an *objective or post-filter* (perturb surfaces/positions,
measure performance spread) — **not** as a primary diversity axis (the descriptor ablation showed
robustness collapses the fold archive as a BC). Open, publishable sub-question: *does a
robustness constraint change which topologies win?* Fold this into Track 1's characterization.

**Recommendation:** **Track 1 as the paper, with Track 3's robustness as its most valuable added axis,
gated on the §2.2 experiment to fix the discovery-claim wording. Track 2 is the clean second paper.**

---

## 5. Prior-art tethers we are missing (and the daylight for each)

The researcher correctly flags weak tethers. Concrete positioning:

| Prior work | What it is | **Our daylight** |
|---|---|---|
| **XLumina** (NeurIPS optics) | differentiable optical-setup discovery / auto-design of experiments | imaging *lens/relay topology* discovery with **non-sequential TIR folds + energy ledger**, not optical-table layout; our diversity is over *interaction topology* |
| **Differentiable caustic / luminance engineering** (e.g. the Delft caustic-image work) | *non-imaging*: shape one freeform surface to hit a target irradiance; radiance made differentiable by modeling each pixel as a **Gaussian** (the ray-pixel bin is a non-differentiable step function) | **imaging** (pupil/collimation/spot/MTF), **multi-surface**, **non-sequential branching + TIR topology discovery** — a fundamentally different objective class; their smoothing trick ≠ our per-topology IFT gradient |
| **Cheng et al. OST-HMD freeform prisms** (§6) | *hand-engineered + locally optimized* freeform AR prisms (Zemax-style) | we **discover** the fold topology (they prescribe it) and produce a *diverse archive*; their "fewer surfaces = easier optimize" remark corroborates our fold-count refinability result |
| **LensGAN / Côté** | generative starting points from **corpora of human-designed sequential refractive** lenses | ours is **non-sequential, machine-discovered, topology-structured, physics-verified**; diversity bounded by *physics*, not a human corpus |
| **OptiAgent** | LLM agent designs **sequential refractive** systems, physics reward, Zemax refine | NS regime OptiAgent does not touch; our "proposer" (if any) uses *verifiable archive-improvement*, not SFT on patents |
| **HyperLens** (researcher's own) | differentiable QD (CMA-MEGA) over **Seidel descriptors**, sequential | NS analogue: descriptors over **interaction sequences** (discrete → CMA-ME not CMA-MEGA); persistence mechanism reused |

**Action items to close the tethers (the researcher's "what do we lack" list):**
1. Read/cite XLumina and the caustic paper explicitly; state the one-sentence daylight above in the
   related-work section.
2. Find whether any paper **specifically walks the design *process*** for these AR prisms (vs just
   reporting a finished prescription). If none does, that *absence* is part of the novelty ("prior art
   reports prescriptions; we report a discovery process"). [to verify — literature search]
3. Ground the sampling framework: **run §2.2** (CMA-ME vs multi-start). This is the "is there a valid
   mechanism or is it chance" question, answered empirically.
4. Ground the refinement layer: report the **refinability map** (dominant-share vs headroom per cell),
   the optimizer (guarded LM), stability (LM descended on every high-share champion despite κ≈1e11),
   and the failure mode (all-ghost cells, order-guard on multi-fold).

---

## 6. Cheng et al. / OST-HMD tie-in

From the researcher's notes on the Cheng et al. OST-HMD paper (the design closest to ours):

- **Parametrization:** freeform surfaces (our conic+poly / parametric-arc faces are the same class).
- **Specs (their previous, very-similar design):** FOV **45.3°**, **F/1.8**, eye relief **18 mm**,
  exit pupil **12×8 mm**, see-through **0.6–0.4 %**, **plastic** substrate. → **Use these as our target
  spec envelope for the Track-1 realization demonstration** so the comparison is apples-to-apples.
- **The refinability remark:** *"the main prism has only two freeform surfaces S4, S5 … reduces the
  difficulty of optimization."* → direct external corroboration of our **fold-count refinability**
  finding (1-fold sweet spot; multi-fold images poorly). Cite it as independent support for
  "characterize designs by refinability."
- **Open terms to pin down from the paper before writing** (do not invent): the **JMRCC principle**
  they invoke and **how their differentiable/optimization engine is instantiated and realized in 3D**.
  Our current engine is **2D meridional** — flag this as the honest gap (their 45° FOV / F/1.8 is a 3D
  result; our σ<0.1° collimation is a 2D proxy). The 3D extension is a known next step, not a claim.

**The clean comparative sentence for the paper:** *Cheng et al. hand-design one freeform prism and
locally optimize it; we discover a diverse archive of TIR-relay topologies from a fold-blind objective,
select by a measurable refinability criterion, and realize a chosen member to a comparable spec
envelope.*

---

## 7. Concrete experiment queue (what to run, in order)

1. **[decisive] CMA-ME vs multi-start vs random-restart-LM**, matched budget, same archive — §2.2.
   Answers the number-one question and fixes the discovery-claim wording. *Cheap, do first.*
2. **Objective-evolution-with-optimization plots** — fitness vs eval, annotated where TIR folds appear
   / fold count changes, to *see* how TIR folding influences the landscape (the researcher's explicit ask).
3. **Archive-wide refinability map** — dominant-share + LM headroom per cell; turns "it refines" into
   "here is *which* topologies refine," and gives Track 1 its characterization figure.
4. **Track-1 realization** — take 2–3 novel high-share discovered relays, refine (freeform LM,
   `--allow-order-change` for multi-fold) to the Cheng spec envelope (§6); report before/after.
5. **Robustness axis (Track 3)** — tolerance objective/post-filter; does it change which topologies win?
6. **Desensitization / tolerancing literature** — read before (5) to borrow the standard perturbation
   model. [researcher's note]
7. **3D honesty** — scope the meridional→3D gap explicitly (needed to compare FOV/F-number to Cheng).

---

## 8. The one-paragraph paper claim (draft, pick wording after experiment 1)

> *Non-sequential optical design fixes the fold topology by hand and optimizes geometry to a local
> optimum. We instead make the topology an **output**: a quality-diversity search over a physics-honest
> NS tracer — instrumented with typed interaction sequences and an exact energy ledger — illuminates a
> diverse archive of designs indexed by their ray-path topology, from which efficient loss-free TIR-folded
> relays **emerge under an imaging-plus-compactness objective that never names folding**. We then freeze a
> chosen topology and refine its freeform geometry with guarded damped-least-squares to a stated AR spec,
> and we show that **refinability is a measurable property** (dominant energy share) that lets us select
> which discovered priors are worth realizing. The result is a machine-discovered, physics-verified,
> topology-structured atlas of non-sequential relays and a discover→freeze→refine pipeline that realizes
> any chosen member — a class of dataset and workflow no prior (human-corpus, sequential) lens-design work
> provides.*

Swap "quality-diversity search … illuminates" for "geometry-conditioned enumeration … structures" if
experiment 1 shows multi-start matches CMA-ME. That is the only sentence the number-one question changes.
