# Prompt (Fable) — A more complete NS engine + a verified "transmit→TIR→TIR→transmit" demonstrator

> Context + a copy-pasteable prompt for **Claude Fable 5**. The goal: a more complete non-sequential
> ray-tracing engine in which we can *construct and verify* a setup where the **dominant energy** of a beam
> undergoes a specified ray-surface interaction sequence — concretely `transmit@A → TIR@B → TIR@A →
> transmit@B` in a **single element** — and, more generally, where interaction *types* are first-class and
> any target sequence is checkable. It does **not** need to focus well; the point is the *mechanism*.
>
> Read [06_reflectivity_tir_analysis.md](06_reflectivity_tir_analysis.md) first (the physics), then this.

---

## The disambiguation that motivates this (read before prompting Fable)

"transmit@A → reflect@B → reflect@A → transmit@B" in one element is **two physically different mechanisms**,
and which one you mean decides everything:

| | **(1) TIR route** (steep angles) | **(2) Partial-mirror route** (near-normal) |
|---|---|---|
| The reflections at A/B are… | **Total internal reflection** (angle beyond critical) | **Partial reflection** off a (coated) surface |
| Energy in the folded path | **~loss-free** (100% per bounce, both polarizations) | **lossy** (each near-normal reflection splits energy; uncoated Fresnel ≈ 4%) |
| Coating needed? | **No** — TIR is intrinsic to geometry+index | **Yes** — needs a reflective coating to reflect strongly |
| Polarization needed? | **No** | For efficiency + ghost suppression, **yes** (QWP + reflective polarizer) |
| Geometry | steep/curved/tilted/wedge; oblique entry (dominant bundle must exceed the critical angle) | can be on-axis, near-normal |
| Reflectivity model | **simpler** (delete the artificial knob; rely on physics) | **more complex** (real coating model; the forced-reflectivity knob is a crude stand-in) |
| Real-world analogue | periscope/prism fold, light guide, RXI | VR pancake, monolithic etalon |

So: **my earlier "make the reflectivity model *simpler*" was about route (1).** The researcher's single-element
A→B→A→B example is valid for *both* routes; if it is meant near-normal (a monolithic-pancake-style double
bounce), it is route (2) and is **not** simpler — it needs the coating model and, for efficiency, polarization.
The high-energy version the researcher wants (the dominant rays doing it) is **route (1), TIR** — loss-free,
coating-free, polarization-free.

**Already proven possible.** The current instrumented engine *already simulates route (1)*. With a single
high-index element, **no reflectivity coating, partialReflect off**, the only reflection the engine can produce
is TIR (`BaseGlass.refract` `sq1<0` branch, 100%, geometry-driven). A from-scratch search found a single-element
geometry where the dominant detector-reaching ray does
`transmit@A → TIR@B → TIR@A → TIR@B → TIR@A → transmit@B` carrying ~76% of its energy (the ~24% is Fresnel at
the steep uncoated entry/exit faces, not a fold loss). Demonstrator + geometry:
[ns_probe/tir_demo.py](../../ns_probe/tir_demo.py) (`python3 ns_probe/tir_demo.py`), render at
`ns_probe/results_single/renders/TIR_demo_single_element.png`.

So this is **not** a "can we?" — it is "make the engine *complete enough* that such interaction sequences are
first-class, explicitly typed and energy-accounted, constructible from scratch, and verifiable" — and decide
how much (if any) coating/polarization machinery is worth adding for the lossy route.

---

## The prompt

```
You are a senior research engineer in non-sequential ray tracing, computational optics, and differentiable
simulation. I want your judgment and your implementation, not a survey.

GOAL. Produce a more complete non-sequential (NS) ray-tracing capability in this repo such that we can
construct and *verify* a setup where the DOMINANT energy of an input beam undergoes a specified ray-surface
interaction sequence. The concrete target: a SINGLE optical element in which the high-energy rays do
  transmit at surface A  ->  reflect at surface B  ->  reflect at surface A  ->  transmit at surface B
i.e. a loss-free internal double (or multi-) bounce that exits toward the sensor. It does NOT need to focus
well — the deliverable is the mechanism, demonstrated and explicitly verified, not image quality.

READ FIRST (ground yourself; do not improvise architecture):
  agent_context/ns_rl_research/06_reflectivity_tir_analysis.md   (how reflectivity/TIR are handled; the two routes)
  agent_context/ns_rl_research/05_probe_phase0_findings.md        (the probe + the forced-reflectivity mistake)
  agent_context/ns_rl_research/02_state_and_directions.md, 03_design_discussion.md  (engine + program context)
  agent_context/ray_optics_engine.md                             (the JS engine + the Python bridge)
Then read the code that matters: ray-optics/src/core/sceneObjs/BaseGlass.js (refract: the sq1<0 TIR branch and
the refraction/partial-reflection branch), ray-optics/src/core/Simulator.js (processRays: the interaction loop +
the recorder I added), ray-optics/integrations/simulation_core.js (the bridge wiring), and the WORKING
DEMONSTRATOR ns_probe/tir_demo.py (a single-element, coating-free, TIR-only fold that already exhibits the
target sequence carrying the dominant energy). Run it: `python3 ns_probe/tir_demo.py`.

WHAT IS ALREADY TRUE (do not redo, build on it):
- The engine already does real, geometry-driven, loss-free TIR (sq1<0 reflects 100%, both polarizations,
  untouched brightness). The target sequence is ALREADY SIMULATABLE via the TIR route — tir_demo.py proves it.
- There are two mechanisms for "transmit-reflect-reflect-transmit" (see doc 06): the TIR route (steep angles,
  loss-free, no coating, no polarization) and the partial-mirror route (near-normal, lossy, needs a coating
  and — for efficiency/ghost-suppression — polarization). The high-energy version we want is the TIR route.
- The recorder currently logs ray VERTICES but NOT interaction TYPE, so the engine cannot yet *label* a
  sequence as transmit/TIR/partial — tir_demo.py infers TIR only because the scene is coating-free.

WHAT "MORE COMPLETE" SHOULD MEAN (decide the design; these are the gaps, not a recipe):
- First-class interaction TYPES. Every recorded event should carry its type (refract / TIR / partial-reflect /
  mirror / sensor / absorbed) and its energy, so ANY target sequence (e.g. transmit@A,TIR@B,TIR@A,transmit@B)
  is explicitly checkable and the per-event energy is auditable — not inferred from a coating-free trick.
  The refract method already knows the type at decision time; surface it.
- Honest energy accounting end to end: input energy = delivered + reflected-away + absorbed + truncated, with
  the split visible, so "the dominant energy did the fold" is a measured statement.
- Both routes supported and clearly distinguished: the loss-free TIR route (no coating), and — if you judge it
  worth it — a physical partial-reflection/coating route (the current forced-reflectivity knob is a crude,
  angle/wavelength-independent stand-in; a real model is angle/polarization-dependent). State plainly what each
  route needs and which we should prefer for a high-energy fold.
- A robust, reproducible "construct from scratch" path: generalize tir_demo.py from a lucky single ray to a
  setup where a whole DOMINANT bundle does the target sequence, and/or a small gradient-free search that
  reliably finds a geometry realizing a *requested* interaction sequence. Make the target sequence an input,
  not a hardcoded outcome.
- Decide explicitly: extend the JS engine, and/or scope a cleaner from-scratch NS tracer (the differentiable
  DeepLens-NS path discussed in doc 03). Recommend, justify, and implement at least the extension that makes
  the demonstrator first-class and verified; scope the from-scratch tracer if you judge it the better long-term
  home for typed interactions + energy accounting + differentiability.

BOUNDARIES:
- Do NOT implement Jones-vector polarization or multilayer coatings to hit the TIR target — they are not needed
  for it (TIR is 100% for both polarizations). Only consider them for the separate, lossy partial-mirror /
  pancake route, and if you do, keep it clearly partitioned and optional (Tier B in doc 06).
- Do NOT chase imaging quality / focus — the goal is a verified interaction-sequence demonstrator. A barely- or
  non-focusing setup that provably carries the dominant energy through transmit→TIR→TIR→transmit is a success.
- Preserve the existing engine's default behaviour (the instrumentation is flag-gated; keep it so). Note the
  build/sync gotcha: `npm run build-node` writes ray-optics/dist-node/rayOptics.js, but the runner uses
  ray-optics/integrations/rayOptics.js — copy it across after a rebuild.
- Keep the emergence thesis in view (docs 01/06): an artificial reflectivity coating that lets folds happen
  "for free" is a soft prescription; the faithful, high-energy fold is TIR emerging from geometry.

DELIVER: (1) the engine completeness improvements implemented (typed interactions + energy accounting at
minimum); (2) a reproducible demonstrator where the DOMINANT energy provably does the target single-element
sequence, with the per-event types and energies printed and a render; (3) a short written account of the design
decisions, what each route costs, and whether/when the from-scratch differentiable NS tracer is worth building.
Lead with what you built and what it proves; show the verified sequence early.
```

---

## How to use
- Run on **Claude Fable 5** at `high`/`xhigh` effort; expect a multi-minute, multi-file turn. If it has no repo
  access, paste docs 06, 05, and `ns_probe/tir_demo.py` ahead of the prompt.
- Sanity check on its result: the demonstrator must **print the per-event interaction types** (not infer them
  from a coating-free trick) and show the dominant ray's energy through the fold. If it reaches for polarization
  to hit the TIR target, push back — that's the tell it conflated the two routes.
