# Phase-0 Probe — Findings: Does Folding Emerge as a Good Imager?

> Built and run 2026-06-12 by Opus 4.8, executing the de-risking probe from
> [03_design_discussion.md](03_design_discussion.md) §3. This is the first *running experiment* in the
> NS program. It is a cheap, gradient-free go/no-go, not a publication result. Read
> [02](02_state_and_directions.md) and [03](03_design_discussion.md) first for context.
>
> Code: [ns_probe/](../../ns_probe/) · archive: `ns_probe/results/archive.json` · figure:
> `ns_probe/results/probe_figure.png` · engine instrumentation: see "What was built" below.

---

## TL;DR (headline verdict)

**For the cheapest, weakest configuration of the test, the answer is NEGATIVE: folding did not emerge as a
good imager. The archive's good-imaging region stayed transmission-dominated at every thinness level.**

- Folding is physically **reachable and was explored** — 123 of 212 filled archive cells are folded
  (≥2 mean axial reversals); the search did not ignore the folded regime.
- But in **every** thinness niche the best imager is a **transmissive** design. Across 23 populated
  thinness columns, folding beat transmission in **0** of them (folded best-fitness −8 to −17 vs transmission
  ~0 to +0.9). The single best design overall is transmissive: on-axis spot RMS **0.078** world units, full
  flux.
- Even the few folded designs that reach the **thin** region (device depth T≤45) image terribly
  (best spot ≈ **11** vs transmission's <1).
- **Crucially, this negative is dominated by two known, fixable factors** (no local refinement; deliberately
  impoverished phase-1 physics) — it is *not* evidence that the emergence thesis is wrong. It sharpens the
  go/no-go rather than settling it. See "Honest interpretation."

The precise diagnosis: **folded energy reaches the sensor (best fold: flux 0.14, 2.07 reversals) — what fails
is the focus quality of that folded energy, not the existence of folds.**

---

## What was built (reusable infrastructure — the durable deliverable)

1. **The JS NS tracer is now instrumented** (Doc 2 §E.1 realized), gated behind flags so default behaviour is
   unchanged, source-edited and rebuilt into the runner bundle:
   - Per-ray **interaction sequences** exported as `result.raySequences` (each forward sensor crossing carries
     its full ordered vertex list + energy + direction). Hook: [Simulator.js](../../ray-optics/src/core/Simulator.js)
     `processRays` (around the `onRayIncident` call), child rays inherit parent history.
   - **Forced per-surface reflectivity** as a continuous design DOF, overriding Fresnel when a surface carries
     `reflectivity`. Hook: [BaseGlass.js](../../ray-optics/src/core/sceneObjs/BaseGlass.js) `refract`.
   - Wiring + unknown-key sanitation in [simulation_core.js](../../ray-optics/integrations/simulation_core.js).
   - **Build note for future agents:** `npm run build-node` emits to `ray-optics/dist-node/rayOptics.js`; the
     runner loads `ray-optics/integrations/rayOptics.js`, which is a committed copy that **nothing
     auto-syncs** — after a rebuild you must `cp dist-node/rayOptics.js integrations/rayOptics.js`.
2. **The probe pipeline** ([ns_probe/probe.py](../../ns_probe/probe.py)): 2-element rotationally-symmetric-ish
   (2D meridional) scene builder with reflectivity DOF; descriptors (device depth T, energy-weighted mean
   axial reversals, optical-path ratio, folded fraction) and fitness, all computed from the interaction
   sequences.
3. **A self-contained CMA-ME** ([ns_probe/cma_me.py](../../ns_probe/cma_me.py)) — standard CMA-ES + improvement
   emitters + grid archive, no `pyribs`/`pycma` (neither is installable here). CMA-ES validated on a sphere
   (‖mean‖→1e-6).
4. **Runner + analysis** ([ns_probe/run.py](../../ns_probe/run.py), [ns_probe/analyze.py](../../ns_probe/analyze.py)).

This stack is the cheap, gradient-free testbed the program can keep using and extend.

---

## The experiment exactly as run

- **Scene (2D meridional proxy for a rotationally-symmetric system):** collimated on-axis input fan (≈51 rays),
  **2 generic curved elements**, each with continuous `{axial position, thickness, front sag, back sag,
  refractive index, reflectivity∈[0,1]}` (12-D genotype), detector at a fixed plane. Bounce budget 12.
- **Fitness = imaging quality only:** `−spot_rms + flux` once an honest flux floor (8% of input) is met;
  below the floor, climb flux. Spot RMS is energy-weighted over **all** forward sensor energy, so ghosts/stray
  light widen the spot and are penalised. **No term references reflection, bounce, fold, or thinness.**
- **Descriptive behaviour axes (measured, never targeted):** BC1 = device depth T (thin = small),
  BC2 = energy-weighted mean axial reversals (transmission ≈ 0, folded ≥ 2). The QD archive therefore grants
  folding its **own dedicated niche at every thinness level** and asks "what is the best imager in this niche?"
- **Solver:** CMA-ME, 5 improvement emitters, popsize 10, 18,000 evaluations. 212 cells filled (74% of a
  24×12 grid), 0 simulation errors, **157 s** wall (single Node process). Reproduced on seed 1 (see below).

Faithfulness check: this is the QD framing from [03](03_design_discussion.md) §2.6 — compactness on a BC axis,
not in the reward; imaging-only fitness; reflectivity as a geometry-like DOF. It is the **weakest** form of the
emergence test (folding is *allowed and measured*, never *demanded*).

---

## Results

| quantity | value |
|---|---|
| filled cells | 212 (transmission rev<1: 45; **folded rev≥2: 123**) |
| good-imaging cells (flux≥floor, spot≤3) | transmission **25**, folded **0** |
| good-imaging AND thin (T≤45) | transmission **5**, folded **0** |
| thinness columns where folded beats transmission | **0 / 23** |
| best design overall | transmissive, spot **0.078**, flux 1.00, T=95 (*not* thin) |
| best folded design (any fitness) | spot 4.41, flux 0.086, 2.07 reversals, folded-frac 0.62, T=109 |
| best folded design reaching thin region (T≤45) | spot ≈ 11, flux ≈ 0.14 |

Figure `ns_probe/results/probe_figure.png`:
- **Panel A** (archive heatmap, fitness over thinness×reversals): all high fitness sits in the low-reversal
  (transmission) band; the folded half of the space is uniformly poor.
- **Panel B** (best fitness vs thinness, transmission vs folded): the transmission frontier sits ~0.7–0.9 at
  every depth; the folded frontier sits −8 to −17 and never approaches it. The two never cross.
- **Panel C** (best folded design, re-traced): rays do fold (visible back-and-forth), but the folded bundle
  spreads widely and loses energy — exactly why its spot RMS is large.

Two predictions from the docs were **confirmed**: (i) good imaging and thinness are in tension — the very best
imager is *not* thin (T=95); (ii) with no mechanism, the good region is all-transmission.

**Rendered designs (setup + ray paths, native ray-optics renderer via node-canvas):**
`ns_probe/results/renders/contact_non_sequential.png` (top 5 folded) and `.../contact_sequential.png`
(top 3 transmission), plus per-design PNGs (`ns_top{1..5}.png`, `seq_top{1..3}.png`); regenerate with
[ns_probe/render_designs.py](../../ns_probe/render_designs.py). The two sheets *are* the verdict, visualized:
the transmission designs are clean bright cones converging to a sharp on-axis focus at full flux; the folded
designs reflect most light back off a strong front partial mirror (refl ≈ 0.8), leave only a dim forward
bundle, and never focus cleanly. The best two-mirror fold (NS #4, both surfaces reflective) reaches only
flux 0.19; the thinnest fold (NS #5, T=44) images at spot ≈ 11.

**Robustness (seed 1):** an independent 18k-eval run reproduced the verdict almost exactly — 216 cells, 124
folded, **26 good transmission / 0 good folded, 0/columns** where folding wins, best transmission spot 0.095.
The negative is not a single-seed artifact.

---

## Single-element variant (the faithful version of the existence proof)

The run above used **2 elements (a doublet)** — a scope error: the original existence proof and the program's
intent are a **single deformable element** (the TIR result). The single-element probe was therefore run
separately (the 2-element results above are **not** overwritten).

- **Model:** one glass element with **independent per-face reflectivity** (front/back — a Mangin /
  monolithic-pancake-style element, named in the planning doc), plus geometry DOFs. 7-D genotype. The engine was
  extended for per-face reflectivity (`reflectivityFront`/`reflectivityBack` keyed by `elemCx`), backward
  compatible with the uniform `reflectivity` used above. Code: [ns_probe/probe_single.py](../../ns_probe/probe_single.py),
  [ns_probe/run_single.py](../../ns_probe/run_single.py). Archive: `ns_probe/results_single/`.
- **Result (15k evals, 188 cells, 86 s, 0 errors): the same verdict.** Folding reachable/explored (92 folded
  cells), but **good-imaging: transmission 24, folded 0**, and folding beats transmission in **0/24** columns.
  Best transmissive single element images at spot **0.20** (full flux); best fold images at spot 6.3.
- **One informative difference:** single-element folds **do reach the thin regime** (best fold at T=17, even
  T=12, with 68% folded energy and high per-face reflectivity) — internal multi-bounce in one element *is*
  compact, unlike the 2-element folds which wanted depth. But the thin folds still **do not focus** (spot ≈ 6–7
  vs transmission's 0.2). So the single-element conclusion is sharper: folding becomes *compact* without being
  *prescribed*, yet remains a poor imager under focus-only fitness with no refinement.
- **Renders (10 distinct rays, per your request):** `ns_probe/results_single/renders/contact_non_sequential.png`
  and `.../contact_sequential.png`. The folded sheet shows individual rays bouncing inside the single element
  (visible crossing-line folds); the transmission sheet shows 10 rays through one lens converging to a sharp
  focus. Regenerate: `python3 ns_probe/render_designs.py --archive ns_probe/results_single/archive.json --rays 10`.

The interpretation and next steps below apply to **both** variants (the single-element result is the primary
one going forward).

## Honest interpretation — what the negative does and does not mean

This is a real, clean negative for the configuration tested. It is **not** a falsification of the program,
because the configuration omits the two things most likely to make a fold competitive:

1. **No local refinement (the strongest confound).** HyperLens's entire thesis is
   *persistence-then-improvement*: behaviourally-novel-but-poor designs are kept and **later polished**
   (L-BFGS) into the best results. This probe has CMA-ME **exploration only and no refinement step**. Folded
   designs land in the archive poor (spot 11, flux 0.14) and **stay** poor because nothing refines them. A
   folded design sitting near a good basin will look bad until it is polished — and polishing is precisely the
   value of the differentiable DeepLens-NS engine. So "folded designs are poor" here substantially means
   "folded designs were never refined," not "folded designs cannot image."
2. **Deliberately impoverished phase-1 physics.** 2 on-axis, single-wavelength, **geometric (no-polarization)**
   elements. The best folded design even drove its elements to n=1.0 (pure partial mirrors), which have little
   focusing flexibility. The canonical pancake earns its keep over a FoV/eyebox and is only efficient and
   ghost-free **with** the quarter-wave-plate + reflective-polarizer mechanism — phase-2 by design (see
   [01](01_extracted_research_context.md) §1, NS prompt fork 3). Geometric folds are intrinsically lossy and
   ghosty, and the honest spot metric correctly penalises the ghosts.

What the negative **does** license (these are solid):
- The **infrastructure works** end-to-end — interaction-sequence instrumentation, descriptors, CMA-ME — at
  ~9 ms/eval, 0 errors, reproducibly. The program can build on it.
- **Empirical answer to the "attribute for diversity while good" question** ([03](03_design_discussion.md)
  §2.4): in this model there is **no free attribute** that buys ray-surface-interaction diversity at good
  imaging. The QD archive gave folding a fair niche at every thinness and folding lost on imaging merit
  everywhere. Diversity-that-is-also-good does **not** come for free from focus+flux alone.
- **Do not expect folding from exploration-only QD on a focus objective.** That specific bet is now tested and
  negative.

What it does **not** license: any claim that folding cannot emerge, or that the pancake thesis is dead. The
test that would support that claim has not been run (it requires at least refinement, and realistically
polarization).

---

## Recommended next experiments (in order; cheap → expensive)

1. **Add a local-refinement step to the SAME probe (cheapest, highest information).** After CMA-ME, take the
   best folded archive candidates and polish each with a black-box local optimizer (Nelder–Mead / a short
   CMA-ES restart) on the *existing JS engine* — **no differentiable tracer needed**. This directly tests
   confound #1: can a folded design be polished into the good-imaging bar? If yes, the headline flips and the
   program is alive on the cheap engine. If no even after refinement, confound #1 is ruled out and the negative
   strengthens. **Do this before any tracer investment.**
2. **Enrich the model:** ≥3 surfaces, allow a genuine curved mirror (refl→1) plus refractive elements, and add
   2–3 field points (off-axis), since pancakes earn their advantage over a field, not on-axis. Still on the JS
   engine, still gradient-free.
3. **Only if 1–2 show life:** build **DeepLens→NS** for differentiable refinement (the proper
   persistence-then-improvement loop) — this is the engine investment [03](03_design_discussion.md) §2.3
   describes.
4. **Phase 2 (polarization)** remains the path to an *efficient, ghost-free* fold and to the headline
   "polarization-selective routing emerges." Gated on a successful geometric fold.

The probe's net contribution: it converts "should we build the NS tracer?" into a sharper question —
**"can refinement make a folded design image well on the cheap engine?"** — and hands over the tooling to
answer it in a few more minutes of compute.

---

## Reproduce

```bash
# engine already instrumented + bundle synced; if you re-edit src/core, rebuild then sync:
#   (cd ray-optics && npm run build-node && cp dist-node/rayOptics.js integrations/rayOptics.js)
python3 ns_probe/run.py --evals 18000 --seed 0 --out ns_probe/results   # ~2.6 min
python3 ns_probe/analyze.py                                             # verdict + figure
```

Tunables in [ns_probe/run.py](../../ns_probe/run.py) / [probe.py](../../ns_probe/probe.py): `--flux-floor`,
bounce budget `MAX_RAY_DEPTH`, element count `N_ELEM`, archive bins/bounds, fitness weights.
