# Milestone Experiments — The NS Optical-Discovery Program

> Response to [04b_milestone_experiments_prompt_fable.md](04b_milestone_experiments_prompt_fable.md)
> (Claude Fable 5, 2026-06-12). Grounded in docs 01–03, [ray_optics_engine.md](../ray_optics_engine.md),
> [ns_extension_context.md](../../optical_rl/qd/lensmega/docs/ns_extension_context.md), and direct
> reading + execution of the code (`Simulator.js`, `BaseGlass.js`, `simulation_core.js`, the bridges,
> `optical_rl/envs/`, `optical_rl/qd/`, `DeepLens/deeplens/optics/geolens.py`, `ns_probe/`).
> File:line references verified against the working tree on 2026-06-12.

---

## Conclusion

Four milestones, ranked. Run **M1 and M2 first, in parallel** — they are independent, neither needs
the unbuilt tracer, and together they decide whether the science is alive and whether the ~4-week
DeepLens-NS build is worth paying for.

| # | Name | Tests | Depends on | Cost |
|---|------|-------|-----------|------|
| **M1** | **Flag-and-seek** — plant a hand-built pancake's descriptor coordinates as a hidden target, then ask whether CMA-ME on the instrumented JS engine rediscovers that region under an objective that never names folding | The core emergence hypothesis | Nothing unbuilt | ~1 week incl. runs |
| **M2** | **Basin probe** — conditioned-on-sequence differentiable evaluation of unrolled folds using DeepLens primitives; measure gradient fidelity and topology-basin width | The load-bearing assumption behind the DeepLens-NS build | Nothing unbuilt (~200–400 LoC against existing `ray_reaction`) | ~1 week |
| **M3** | **Persistence ladder** — archive vs. fairly-tuned scalarized multistart vs. random, matched budget, on the M1 domain; the no-compactness arm doubles as the emergence ablation | Whether QD/persistence is *necessary*, not just sufficient | M1 harness | ~4× M1 compute, little new code |
| **M4** | **Lift** — promote 2D-discovered folds to rotationally-symmetric 3D via M2's conditioned refinement; validate in pyOpTools | The discovery→design pipeline the canary needs; whether the 2D probe is trustworthy | M1 positive + M2 evaluator | ~1–2 weeks |

The decision tree they induce:

- **M1+ and M2+** → build DeepLens-NS with its riskiest assumptions retired; M3/M4 follow; then the canary run per the planning doc.
- **M1+ and M2−** → the program goes gradient-free (CMA-ME + per-basin derivative-free polish); the tracer is re-scoped to a forward-only validation engine; the "differentiable NS" claim leaves the paper but the discovery claim survives.
- **M1− because the hand-built flag itself fails in 2D** (found on day one, for free) → the 2D probe is uninformative; the science test moves to 3D and the tracer becomes the critical path, scoped by M2.
- **M1− with folds nowhere in the logs after descriptor fixes and a doubled budget** → the geometric-emergence bet itself is in trouble. Stop building. Rework the objective/template before spending another engineering dollar.

One material correction to the docs before anything else:

**The gatekeeper build is already done.** Doc 02 §C calls per-ray interaction-sequence instrumentation
"the gatekeeper" and prices the program's first weeks around it. It exists, today, in the tree, and it
works — I ran it:

- [Simulator.js:878-901](../../ray-optics/src/core/Simulator.js#L878-L901) records per-ray vertex
  sequences (`__seq`: x, y, surface index per strike) gated behind `scene.recordInteractions`;
  [Simulator.js:916-919](../../ray-optics/src/core/Simulator.js#L916-L919) propagates history to
  spawned children; detector arrivals are stored with landing point, direction, and brightness
  (capped at 200k records).
- [simulation_core.js:268-271](../../ray-optics/integrations/simulation_core.js#L268-L271) exports
  `result.raySequences`; it survives the NDJSON path (verified end-to-end through
  `PersistentRayOpticsRunner`).
- [BaseGlass.js:212-218](../../ray-optics/src/core/sceneObjs/BaseGlass.js#L212-L218) implements
  **forced continuous per-surface reflectivity** ("NS-probe extension" — designer-set r overriding
  Fresnel), i.e. the central new DOF already exists.
- [Simulator.js:858-876](../../ray-optics/src/core/Simulator.js#L858-L876) implements a per-ray
  `maxRayDepth` bounce budget (default Infinity, [Scene.js:220](../../ray-optics/src/core/Scene.js#L220)).
- `python3 ns_probe/_verify_instrumentation.py` (run 2026-06-12): a two-partial-mirror scene yields
  folded detector arrivals with **6 axial reversals**, full vertex chain
  `[0, 40.3, 49.8, 39.6, 51.0, 39.5, 49.8, 120.0]`, energy `b=0.0156`; an r=0.9 reflector collapses
  forward power from 3.000 to 0.1125 as expected.

So the question is no longer "how do we extend the env so sequences become signal" — it's
"which experiment do we point the already-instrumented engine at first." M1's schedule collapses
from weeks to days. The remaining build for M1 is pure Python: a 2–3 element scene template,
descriptor functions over `raySequences`, a fitness function, and runner glue.

---

## M1 — Flag-and-seek: the emergence probe on the instrumented JS engine (rank 1)

**Hypothesis.** Under a fitness that never references path topology, with compactness and optical
path length as *descriptive* archive axes only, archive illumination populates the pancake-like
region of behavior space with genuinely folded designs — i.e., thinness pressure expressed as
diversity pressure is sufficient for folding to emerge, exactly as the refocusing objective was
sufficient for TIR.

This is the experiment doc 03 §3 sketches as de-risk step 1, sharpened in three ways: a planted
flag that makes both positive and negative results interpretable, a two-arm reflectivity
initialization that separates *existence* from *reachability*, and full-sequence logging that makes
a negative result diagnosable instead of ambiguous.

### Design

**Template** (pure scene-builder work; the engine needs nothing): meridional 2D scene, axis along x.
2–3 elements, each a `Glass` with front/back arc sags, thickness, gap, refractive index
n ∈ [1.0, 1.9], and forced reflectivity r ∈ [0, 1]. θ ≈ 10–16 dims. Collimated single-wavelength
beam (the planning doc's on-axis canary choice), detector at fixed x. Two engine facts make this
template more expressive than it looks:

- A `Glass` with n ≈ 1 and forced r is a **curved beamsplitter in air** — the probe scene in
  `ns_probe/_verify_instrumentation.py` already exploits this (n=1.05, r=0.5/0.92 produced the
  folds). Standalone curved partial mirrors don't exist as scene objects
  (`ArcMirror` extends `BaseFilter`, no `transRatio`; `BeamSplitter` is flat-only), but this trick
  makes them unnecessary: n and r are both continuous knobs and the *same* element can be lens,
  beamsplitter, or mirror depending on where the optimizer pushes them. Topology from continuous
  parameters, as the planning doc demands.
- Forced r is per-*object*, not per-face. For the pancake skeleton that's fine (two elements = two
  working reflective interfaces); finer granularity, if ever needed, is a small JS patch, not now.

**Fitness — the reward-crux resolution.** Fitness = **encircled energy at a defined eyebox**: power
delivered through a narrow detector aperture at the image point. One scalar, purely outcome-based.
This deliberately *simplifies* doc 03's `imaging + w_e·efficiency − w_g·ghost` and resolves an
internal contradiction in the planning doc (§1 bans "fitness terms for efficiency"; §9 requires a
"flux-on-sensor penalty" — flux-on-sensor *is* an efficiency term). The argument is a three-way
ranking any candidate fitness must get right *within a thin cell*:

| Candidate within-cell ranking | well-focused fold @ ~25% throughput | unfocused thin single-pass | "perfect" image from 1% of light, 99% stray |
|---|---|---|---|
| energy-weighted RMS over all arrivals | **fails** — 75% unavoidable ghost energy destroys the score; kills exactly the designs we want | ok | ok |
| RMS of detected energy only | ok | ok | **fails** — gameable by imaging a sliver |
| **encircled energy at eyebox** | **wins cell** (~0.25) | **loses** (diffuse energy misses the slit) | **loses** (≤0.01 by construction) |

Encircled energy is also literally the metric that produced the TIR result (`reward =
detector.power` through a small detector, [quad_curve_glass_env.py](../../optical_rl/envs/quad_curve_glass_env.py)
docstring: "Reward: raw detector power") — the one objective in this program with an empirical
track record of driving multi-interaction emergence. Stray light needs no classifier and no
path labels: it is punished by *absence* (it misses the slit), which is exactly the
"valid = where it lands, never which path it took" boundary. For the eventual paper, RMS-of-in-focus,
throughput, and stray fraction are *reported* as evaluation metrics; the search never sees them
separately.

**Descriptors (archive axes, never in fitness).** Both computed from `raySequences`:
- BC1 = energy-weighted optical path length of detector arrivals ÷ physical track (first-surface
  vertex to detector). Continuous, physical, descriptive.
- BC2 = track ÷ EFL, with EFL estimated numerically from marginal-ray exit slope (the planning
  doc's chief-ray method, adapted to collimated input: EFL = h / tan u′). Designs that never focus
  get a sentinel band, not a crash.

Bounce counts and reversal counts appear *nowhere* in fitness, descriptors, or observations — they
are post-hoc analysis labels only. That keeps the thesis line exactly where doc 02 §D.1 drew it.

**Step 0 — plant the flag (also the canary calibration the docs keep deferring).** Hand-build the
3-pass pancake skeleton in the template (curved half-mirror + rear curved partial mirror, the
verified n≈1 trick). Verify: it images (encircled energy ≳ 0.15–0.25 of input — the geometric
25% bar minus losses), and record its (BC1, BC2) cell — **the flag region**. Hand-build the best
transmissive references at the same track (r=0, telephoto-style) to confirm the flag region is not
reachable at quality without folding. Then delete all of them; search never sees them. This fixes
doc 03 §4's open fork ("precise canary criteria must be fixed before a discovery run") with a
*measured* target instead of a guessed threshold, and it costs a day.

**Step 0.5 — noise floor.** Three engine landmines, all verified in code, all cheap to neutralize
but fatal if ignored:
- Default `rayModeDensity` = 0.1 ([Scene.js:202](../../ray-optics/src/core/Scene.js#L202)) gave the
  probe scene **3 rays**. Raise until repeated-eval std of fitness and both BCs < 5% of a cell width
  (expect 64–256 rays/fan; cost is linear).
- Child rays below brightness 0.01 are dropped or stochastically amplified
  ([BaseGlass.js:240-251](../../ray-optics/src/core/sceneObjs/BaseGlass.js#L240-L251)). With r below
  ~0.01 the engine *truncates exactly the branches folding needs* — a literal dead zone at the
  bottom of the reflectivity range. Either init r ≥ 0.02 or set a non-default `colorMode`, which
  drops the floor to 1e-6 (same line). This quietly redefines what "initialized near zero" can mean.
- `maxRayDepth` and `rayCountLimit` must be set explicitly (14 and a generous bound worked in the
  probe); budget-vs-deep-fold sensitivity is a sub-ablation here, not its own milestone.

**Main runs.** CMA-ME (pyribs, already a dependency of the sequential side) with the existing
generic [map_elites.py](../../optical_rl/qd/map_elites.py) as smoke-test fallback. ≥100k evals,
≥3 seeds, **two arms**:
- **Existence arm:** r ~ U[0,1] at init. Asks: do folded designs exist in this template and win
  their cells?
- **Reachability arm:** r init ≈ 0.02 (just above the floor). Asks: does search *grow* mirrors out
  of a transmissive world — the strictly stronger, cleaner emergence story, and the direct NS
  analogue of how TIR appeared. The prompt's default ("reflectivity initialized near zero") is this
  arm; running both upgrades a stylistic default into a falsifiable comparison. If the existence arm
  folds and the reachability arm doesn't, that is a *finding* about basin reachability and exactly
  the regime where HyperLens's persistence mechanism earns its keep.

Log full `raySequences` for every elite and a uniform subsample of all evaluations. This is what
makes a negative diagnosable (see below) and it feeds M4 directly.

### Decisive readout

**Success:** in ≥2 of 3 seeds of the existence arm, some cell within the flag neighborhood
(±20% per BC axis around the hand-built pancake's cell) holds an elite with encircled energy
≥ 50% of the hand-built reference, whose dominant arrival sequence shows ≥2 axial reversals on
post-hoc inspection. Reachability-arm success: the same, plus the elite's lineage shows r growing
from ≤0.05 — the "search grew mirrors" trace.

**Negative protocol — three distinguishable failure modes**, because sequences are logged:
1. *Folds exist in the logs but landed outside the flag cells* → descriptor problem, not a science
   problem. Re-cluster offline (the `recluster_families.py` /
   [benchmark_metrics.py](../../diversity_metric_export/code_excerpts/benchmark_metrics.py)
   tooling pattern), pick separating axes, re-run. This is the HyperLens BC-audit lesson executed
   for free.
2. *No folded design with non-trivial fitness was ever sampled, anywhere* → search/reachability
   problem; M3's arms say whether any optimizer family does better; the honest stop condition lives
   here: after descriptor fixes and one budget doubling, stop and rework objective/template before
   any tracer investment.
3. *The hand-built flag itself images poorly in 2D* (caught at step 0, before any search) → regime
   problem; the 2D probe is uninformative and the program pivots to 3D-native discovery (M2 then
   carries the whole tracer decision).

### What could mislead us
- **2D-meridional ≠ rotationally-symmetric 3D.** Energy spreads as 1/r not 1/r², focusing is
  cylindrical. Mitigation: the flag. If a hand-built 2D pancake images well in-engine, the regime
  supports folds and search results are about *search*; transfer is M4's job, not M1's.
- **Eyebox-width gaming**: a too-generous slit rewards unfocused throughput. Calibrate the width so
  the r=0 unfocused reference scores ≪ the hand-built pancake; report the ratio, not the raw power.
- The single-element warm-up (doc 03 §2.4) is *not* part of this milestone — see "Dominated" below.

**Dependencies:** none unbuilt. **Cost:** ~3–5 days of Python + roughly 0.3–1.2 core-hours per
100k-eval arm at 10–40 ms/eval (parallel across `PersistentRayOpticsRunner` processes), well within
the workstation.

---

## M2 — Basin probe: conditioned differentiability with DeepLens primitives (rank 2)

**Hypothesis.** Conditioned on a realized interaction sequence, the trace is differentiable in θ
with usable accuracy, and topology basins are wide enough that within-basin gradients are worth
having — the assumption that justifies CMA-MEGA-style hybrids and the entire DeepLens-NS build.

This tests the design discussion's central reframe (doc 03 §2.1, "non-sequential ≠
non-differentiable") *without building the NS tracer*. The trick: you don't need argmin routing to
test conditioned differentiability — you need an **explicit visit list**. DeepLens's per-surface
`ray_reaction` ([geolens.py:700-705](../../DeepLens/deeplens/optics/geolens.py#L700-L705) applies it
in a fixed `for i in surf_range` loop) already computes differentiable refraction; a ~200–400 LoC
evaluator that walks rays through an explicit ordered list of (surface, interaction-mode) events —
revisits allowed, reflection modes added, partial-mirror split as smooth energy weighting — is a
differentiable evaluator *conditioned on a fixed topology*. That is simultaneously:
(a) the de-risk for the tracer build, (b) the refinement engine for M1's discoveries (precision
stays with the optimizer, as required), and (c) the 3D half of M4's promotion pipeline.

### Design and readouts

Unroll three known topologies in rotationally-symmetric 3D: the canonical 3-pass pancake, a Mangin
mirror (the planning doc's own Stage-1 oracle), and a 5-pass variant.

1. **Gradient fidelity:** ∂(RMS spot)/∂θ vs. central finite differences across all θ components;
   pass at relative error < 1e-3. (DeepLens's autograd through `ray_reaction` should make this
   nearly free; reflection variants are the new code being checked.)
2. **Conditioned refinement:** perturb the hand pancake by 5% per parameter; L-BFGS conditioned on
   the fixed sequence must recover RMS to ≤1.5× the unperturbed value within 200 iterations, with a
   sequence-feasibility check (every conditioned intersection exists, is within aperture, with
   consistent propagation direction) holding throughout.
3. **Basin width — the decisive number:** along ≥100 random unit directions in θ from each unrolled
   design, the distance at which sequence feasibility breaks. **Decisive threshold: median basin
   half-width ≥ 1% of the parameter box** — commensurate with CMA-ME's final step sizes and
   refinement perturbations. Below that, gradients exist but you can't take a step without leaving
   the basin, and straight-through bias (the planning doc's accepted trade-off) stops being a
   corner case and becomes the typical case.

### What a negative kills
- (1) or (2) failing → the DQD/straight-through plan dies; the tracer re-scopes to a forward-only
  NS validator; refinement falls back to derivative-free polish on a frozen sequence. The
  "differentiable NS" methods claim leaves the paper; the discovery claim does not.
- (3) failing (razor-thin basins) → persistence becomes not just preferred but *the only*
  mechanism; CMA-ME-only program; tracer is a luxury. Also a genuinely interesting negative — it
  would say the NS landscape is qualitatively harsher than the sequential one HyperLens measured.

### What could mislead us
Basins around *canonical* designs may be atypically wide (they sit at engineered sweet spots) or
narrow (heavily co-optimized). Report the distribution across all three topologies, and re-run the
measurement on M1's actual discoveries once they exist — that, not the canonical trio, is the
population that matters.

**Dependencies:** none unbuilt; pure reuse of DeepLens surface primitives plus new glue.
**Cost:** ~1 week. Fully parallel with M1 (different codebase half, different machine load).

---

## M3 — Persistence ladder: is the archive load-bearing? (rank 3)

**Hypothesis.** Persistence of behaviorally-novel-but-poor candidates is *necessary* to reach
folded basins at realistic budgets — the HyperLens mechanism (924 µm ancestor → 2.7 µm result)
generalizes to NS. The conversation asserts this (doc 01 §3.2); nothing yet tests it in the NS
regime, and it is both the central methods claim of the eventual paper and a real fork: if it's
false, the program simplifies dramatically.

### Design
Same template, fitness, and budget (e.g. 200k evals each), four arms:
1. **CMA-ME archive** (M1's configuration).
2. **Multi-start CMA-ES on fitness alone** — no compactness anywhere. This arm doubles as the
   **emergence ablation**: with no thinness pressure, folding should *not* appear. If it does,
   something else is driving folds (stray-light gaming of the eyebox, a descriptor leak into the
   problem setup) and the audit happens now, cheaply, instead of in review.
3. **Multi-start CMA-ES on fitness + λ·thinness** — the prescriptive scalarization the thesis
   forbids for the headline method but demands as a baseline. Fairness is mandatory or the
   comparison is theater: λ grid of ≥5 values × multiple restarts, best-of reported. (HyperLens
   Table 19 — hard-constraint enforcement doubling best RMS and collapsing diversity 10→1 — says
   scalar pressure fails in the *sequential* regime; this is its NS test.)
4. **Random search + local polish** (the floor).

### Readouts
- **Evals-to-first-fold:** first sampled design with ≥2-reversal dominant sequence *and* encircled
  energy above the M1 success bar (post-hoc label, never signal).
- **ε-distinct folded designs at end** (the existing θ-space metric machinery,
  [benchmark_metrics.py](../../diversity_metric_export/code_excerpts/benchmark_metrics.py)).
- Flag-region best fitness per arm.

**Decision rules.** Archive ≥3× arm-3 on ε-distinct folds with comparable best fitness → the
persistence claim has direct NS evidence; build the paper's backbone on it. Arm 3 matches the
archive → QD framing is overhead here; the program pivots to "reflectivity-as-DOF + outcome-based
objective" as the story (weaker but honest, and M4 still proceeds). Arm 2 produces folds → red
alert, audit the objective before believing anything else.

**Dependencies:** M1 harness. If M1 came up empty, run a reduced M3 anyway — arm comparison is the
cheapest way to learn whether M1's emptiness was a search problem or an existence problem.

**What could mislead:** under-tuned baselines (hence the pre-registered λ grid); 2D noise floor
differences between arms (same step-0.5 settings everywhere).

---

## M4 — Lift: promote 2D discoveries to 3D and external validation (rank 4)

**Hypothesis.** Folds discovered in the 2D meridional probe survive promotion to rotationally
symmetric 3D and an external NS oracle — establishing the full discovery→design pipeline the
pancake canary will use, and answering doc 03 §4's open fork ("is a 2D probe informative enough to
trust?") empirically rather than by argument.

### Design
Top-k folded elites from M1 → map geometry to the rotationally-symmetric analogue (arc sags →
spherical curvatures; per-element n, r carried over) → extract each elite's dominant interaction
sequence from its logged `raySequences` → **conditioned refinement** with M2's evaluator on that
sequence → validate in **pyOpTools** (NS-native, the planning doc's designated oracle — its first
actual use) and cross-check the realized first-hit ordering.

### Decisive readout
≥1 of the top-5 M1 folds survives: pyOpTools reproduces the same dominant sequence, RMS spot within
2× of the conditioned-refined value, throughput within 20%. That single surviving design is the
program's first end-to-end artifact: *discovered by archive illumination under a non-prescriptive
objective, refined by gradients conditioned on its emergent topology, confirmed by an independent
tracer.* It is also the strongest possible justification slide for the DeepLens-NS build.

### What a negative kills
If no 2D fold survives, the 2D probe's evidence doesn't transfer; the canary requires 3D-native
discovery, the DeepLens-NS tracer becomes the critical path (now scoped with M2's numbers), and
M1's positive stands only as a 2D emergence result — still publishable, no longer the headline.

**Dependencies:** M1 positive, M2 evaluator. **Cost:** ~1–2 weeks including the pyOpTools harness.

---

## Where I disagree with the docs

1. **Doc 02 §C ("sequences are computed, then discarded") is stale.** Instrumentation is built,
   exported, bridged, and verified by execution (details in the Conclusion). The strategic
   conclusion survives — start on the JS engine — but the schedule and the "gatekeeper" framing
   don't. Whoever maintains doc 02 should update §C/§E.1 to past tense.
2. **Doc 03 §2.3 ("node.js is a dead end as the real engine") is half right.** It is not the
   research engine, but it is not merely a disposable probe either: it is the program's only
   *control-flow ground truth* until DeepLens-NS exists, the discovery half of the M4 pipeline,
   and (with pyOpTools) one of two independent validators of any claimed fold. Keep it maintained
   through the canary.
3. **The reward design in both docs is more complicated than the thesis needs.** Doc 03 carries a
   3-term scalar (`imaging + w_e·efficiency − w_g·ghost`); the planning doc bans efficiency terms
   in §1 and requires a flux-on-sensor penalty in §9 — an unacknowledged contradiction. The
   encircled-energy fitness resolves it: one outcome-based scalar that automatically punishes both
   ghosts (they miss the eyebox) and degenerate low-throughput imaging (little energy in the
   eyebox), needs no valid/invalid path classifier at all — eliminating the most dangerous
   prescription-smuggling vector — and is the very metric under which TIR emerged. Multi-term
   image-quality objectives return at canary time as *evaluation* metrics, not search signal.
4. **"Reflectivity initialized near zero" is under-specified in two ways.** Literally near-zero is
   inside the engine's brightness-truncation dead zone (children of r<0.01 surfaces are dropped at
   [BaseGlass.js:240](../../ray-optics/src/core/sceneObjs/BaseGlass.js#L240)) — so "near zero" must
   mean ≥0.02, or the floor must be lowered via `colorMode`. And the choice between near-zero init
   and full-range init is not a hygiene matter but two different hypotheses (reachability vs.
   existence); M1 runs both as arms rather than picking by taste.
5. **The literal framing "extend the simple NS RL environment" points the wrong way.** The right
   extension of the env is to *repurpose* it: same engine, same scene-builder lineage, but as an
   evaluation function for archive search, not a deeper multi-step MDP. The per-step
   geometry-nudging PPO heritage produced the existence proof, but it carries credit-assignment
   cost and no persistence, which is what the sparse NS landscape punishes (doc 01 §3.2, doc 02
   §D.3). I endorse the default regime's CMA-ME position; a PPO arm belongs in M3's ladder later
   only if a reviewer demands it. Otherwise pure RL on this problem is dominated.
6. **Endorsements, for the record:** compactness on the archive axis rather than the reward
   (doc 03 §2.6.2) — correct, and M1 operationalizes it; rotational symmetry as the target regime
   (§2.5) — correct, with the 2D meridional probe as a flag-controlled proxy; polarization strictly
   phase 2 (§2.7) — correct, nothing above touches it; single wavelength, on-axis first — correct.

## Dominated or not worth running now

- **Single-element CMA-ME TIR re-demonstration** (doc 03 §2.4's "warm-up"): dominated. The PPO
  result already proves emergence in that regime; the docs themselves prove a single element cannot
  fold; optimizer risk is low (generic MAP-Elites and the quad-glass demo already exist in
  [optical_rl/qd/](../../optical_rl/qd/)). Worth at most a half-day harness smoke test inside M1's
  step 0.5 — not a milestone. Its one unique question ("does CMA-ME drive interaction diversity at
  all?") is answered better, in the regime that matters, by M1's reachability arm.
- **Building the DeepLens-NS tracer now**: pay only after M1×M2. Every week of tracer work before
  those two read out is spent against an untested core hypothesis and an untested differentiability
  assumption.
- **Bounce-budget / brightness-threshold sweeps as standalone experiments**: sub-ablations inside
  M1 step 0.5.
- **All of doc 01 §2's generative paths (A–E)**: a different program. Nothing there is on this
  critical path, and warm-starting NS discovery with any learned prior would damage the discovery
  claim (doc 01's own Path-C caution).
- **Jones calculus / QWP / reflective polarizer work**: phase 2, gated on a phase-1 geometric fold,
  in either engine.

## What to run first

**M1 and M2, in parallel, starting now.** M1 is the cheapest decisive test of whether the science
is alive, and its step 0 (the flag) alone settles the canary-criteria fork and the
2D-informativeness fork for the price of a day. M2 is the cheapest decisive test of the build
decision the program keeps deferring. Neither blocks the other; together they collapse the largest
two uncertainties before any committed engineering. M3 follows on M1's harness; M4 closes the loop.

## Open questions for a literature check (with search terms)

1. **Is the QD-in-optics whitespace still open?** The planning doc's claim is ~2 months old.
   — "quality diversity optical design", "MAP-Elites lens design 2025 2026", "illumination
   algorithm photonic inverse design".
2. **van Gestel et al.'s gradient aggregation at ray splits** — still unread (flagged in the
   planning doc §10.5) and directly relevant to M2's partial-mirror handling and any future
   DeepLens-NS build. — "van Gestel algorithmic differentiable non-sequential ray tracing
   arXiv 2302.12031", "DiffRayFlow differentiable non-sequential Photonics 2025".
3. **Bias of gradients across discontinuities in differentiable simulators** — how M2's
   straight-through results should be interpreted; the rendering community solved the analogous
   visibility-discontinuity problem with edge sampling / reparameterization, which is the
   contingency if soft routing is ever needed. — "do differentiable simulators give better policy
   gradients", "reparameterizing discontinuous integrands differentiable rendering",
   "edge sampling visibility gradient path-space differentiable rendering".
4. **Basin structure of lens-design landscapes** — the closest existing theory to what M2(c)
   measures, and prior art to position against. — "saddle point construction lens design Bociort",
   "optical merit function landscape network structure", "design landscape topography lens
   optimization".
5. **Automated discovery of folded/catadioptric systems** — anything beyond Volatier's three-mirror
   work that could erode the whitespace. — "catadioptric folded optics topology search",
   "pancake lens automated design optimization", "folded optical system global optimization VR".
6. **Encircled/ensquared energy as a search objective** — precedent and known pathologies of the
   M1 fitness. — "encircled energy merit function lens optimization", "ensquared energy objective
   freeform optical design".
7. **Reflectivity as a continuous design variable** — has anyone treated beamsplitter ratio as a
   free optimization DOF in imaging design? — "beam splitter ratio design variable optimization
   imaging", "partial mirror reflectivity continuous optimization optical system".
