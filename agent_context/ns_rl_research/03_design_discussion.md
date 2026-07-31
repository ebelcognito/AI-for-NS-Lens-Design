# Design Discussion — Differentiability, Engine Choice, Descriptors, and Scoping the First NS Experiment

> Third doc in the thread. Companion to [01_extracted_research_context.md](01_extracted_research_context.md)
> (verbatim prior conversation) and [02_state_and_directions.md](02_state_and_directions.md) (code-grounded
> state + build list). This file captures an **exploratory design discussion** between the researcher and the
> agent (2026-06-12). It is decision-shaping, not yet decided; nothing here is implemented. Where it refines
> doc 02, that is flagged.
>
> Engine background: [../ray_optics_engine.md](../ray_optics_engine.md). DeepLens (vendored, differentiable,
> rotationally-symmetric, **sequential**) lives at [../../DeepLens/](../../DeepLens/); its sequential trace is
> [geolens.py:700-705](../../DeepLens/deeplens/optics/geolens.py#L700-L705)
> (`for i in surf_range: ray = self.surfaces[i].ray_reaction(ray, n1, n2)`).

---

## 1. The researcher's exploration prompts (the questions on the table)

Preserved close to verbatim so the reasoning that follows has its anchor.

1. **Plain HyperLens/CMA methods on a single-element NS env** (not necessarily RL — CMA/QD may be better).
   - (a) Does it need a lot of rays / is it hard to trace quickly?
   - (b) Does aberration lead to a mostly transmission-dominated regime?
   - What attribute should we look for to get **greater diversity in ray-surface interactions while still being
     good**?
   - How hard is it to implement a strong env with the features required for such interactions?
2. **A cylindrically/rotationally-symmetric NS lens system** — an optical axis exists; basically the DeepLens
   env extended with NS ray-surface interaction capability. Is this interesting?
3. **Is the 4-term reward too complex?** (`imaging_quality + w_c·compactness + w_e·valid_path_efficiency −
   w_g·ghost_energy`, no term naming reflection/bounce/fold). Why would it work with few rays?
4. **DeepLens vs node.js for the engine.** DeepLens has no public NS. Extend DeepLens (but GPU batched tracing
   seems to break because surface order differs per ray → not a single matmul)? Or extend the node.js env to be
   rotationally symmetric?
5. **What does the descriptor framework rely on?** Sequences of ray-surface interactions per ray + adjacency
   relationships (to enable focusing), or the spots formed? Since node.js NS is non-differentiable, descriptors
   on interactions might be the move. Could we build a "descriptor gradient"? In HyperLens gradients came from
   differentiating w.r.t. params; ray-surface interactions are discrete — how to define gradients on that?
   Alternative-to-backprop vs implementing it properly?
6. **The "proxy" idea.** Is there a proxy for the surface — an intermediate way to go from ray-sequence →
   surface params, aware of intersection points (which DeepLens/dO is good at)? Is it possible to use **dO in
   the NS framework**, or easier to modify DeepLens? DeepLens has dO + surface libraries + backprop but lacks
   NS; node.js has NS but lacks dO + differentiability + rotational symmetry.
7. **Do we need to simulate polarization for anything?**

---

## 2. The agent's analysis and resolutions

### 2.1 The central reframe: non-sequential ≠ non-differentiable
A trace splits into two parts:
- **Control flow (discrete):** which surface a ray hits next, and transmit/reflect/TIR. This is the *index
  set* of the interaction sequence.
- **Geometry (continuous, smooth in θ):** given a fixed sequence, every intersection point, exit direction,
  segment length, and energy is a smooth differentiable function of surface params (Snell + sag).
  DeepLens's `ray_reaction` already computes this differentiably, per surface.

**Therefore: conditioned on a ray's realized interaction sequence, its spot position / optical path length /
energy are differentiable w.r.t. params.** NS changes only the control flow, not the per-surface math.
Discreteness bites only at measure-zero boundaries in θ (a ray switches first-hit surface, or crosses the TIR
critical angle). This is exactly what the NS planning doc already decided: **hard argmin for next-surface +
straight-through gradient**, **soft/energy-continuous branching at partial mirrors** (the reflectivity split is
smooth). Within a topology basin the whole trace backprops; straight-through is biased only near basin walls,
and the doc accepts that, leaning on emitter diversity.

**Implication for the "proxy" idea (Q5/Q6):** the intermediate differentiable quantity the researcher is
reaching for already exists — the ordered list of intersection points/directions/energies, smooth in θ given
the sequence, produced by dO/`ray_reaction`. **No learned surrogate is needed to get gradients.** A learned
surrogate (SAIL-style surrogate-assisted illumination — real prior art, see [01](01_extracted_research_context.md)
§2 Path B) is only an accelerator and, as the researcher suspected, would be flimsy/overfit as a load-bearing
component. Do not make it a prerequisite. "Alternative to backprop" is unnecessary; standard backprop works
within a basin.

### 2.2 "How to differentiate discrete descriptors" — you mostly don't; pick the QD variant by measure type
The **differentiability of the measure** decides the QD algorithm:
- **Discrete interaction-sequence descriptors** (dominant bounce count, # distinct high-energy sequences): not
  differentiable → **do not use CMA-MEGA/DQD for these. Use CMA-ME / MAP-Elites**, which only *evaluate* the
  measure, never differentiate it. Keep a differentiable *objective* (spot RMS, efficiency) for gradient
  emitters and L-BFGS refinement.
- **Continuous proxy descriptors** (energy fraction in dominant path, path-length ratio, thinness): *are*
  differentiable within a topology → could stay in DQD if desired.

In HyperLens, Seidel S2/S3 were differentiable (paraxial traces), which is why CMA-MEGA applied. For NS the
natural descriptors are partly discrete, so the honest stance is **CMA-ME with a differentiable objective**,
not full DQD. Smaller claim, sound, and it sidesteps inventing an alternative to backprop.

### 2.3 DeepLens vs node.js — different questions; use both, in order *(refines doc 02 §E.7)*
| | node.js (instrument existing) | DeepLens → NS (extend) |
|---|---|---|
| NS control flow | already there | **must build** (the hard part) |
| Differentiable? | No | **Yes**, within a basin, free from `ray_reaction` |
| Geometry | 2D planar (meridional only) | 3D **rotationally symmetric** = the pancake regime |
| Surface libs / dO | would have to rebuild | already there |
| Best for | cheap gradient-free CMA-ME probe to check the science | the differentiable research engine for the paper |

- The "can't matmul because surface order differs" worry is **addressable** and does not break GPU batching.
  You don't need a shared surface order; you need a fixed number of **bounce iterations**. Each iteration: a
  `[N_rays × N_surfaces]` intersection test → per-ray `argmin` → **gather** the chosen surface's params → apply
  `ray_reaction`. Per-ray different surfaces = gather/scatter, not a broken matmul. Branching grows the ray set
  by `concat`. This is "wavefront path tracing", GPU-friendly and differentiable.
- DeepLens already exposes a `record` flag on `trace(... record=False)` — a hook to extend for interaction-
  sequence recording.
- **Net recommendation:** node.js is good **only** as a quick non-differentiable probe; it's a dead end as the
  real engine (you'd reimplement dO). Extending DeepLens is the smaller net-new surface area for the capable
  engine. The NS control-flow build is **medium**, not "easy" — well-understood but a real effort, with the
  straight-through gradient approximate at basin walls.
- This also corrects doc 02: **"extend DeepLens" beats "write a fresh PyTorch NS tracer"** (same dO, less
  rebuild), and the gap between gradient-free probe and differentiable engine is **smaller** than doc 02
  implied, because you'd be extending already-differentiable code.

### 2.4 Single-element NS env is too weak — diagnosis (answers Q1a/Q1b)
- **(b) Yes, well-imaged ⇒ transmission-dominated.** A well-corrected lens has near-normal incidence and low
  Fresnel reflectivity → almost all energy transmits. Aberration does **not** drive reflection; **high
  incidence angle** (→ TIR / large Fresnel) and **explicit reflectivity DOF** do. So *good imaging* and *rich
  interaction diversity* are in direct tension for a simple element.
- **The "diversity while good" attribute is not free:** **compactness (thinness) pressure is what makes
  non-transmission worth it** — the exact analogue of how the refocusing objective made TIR worth it in the PPO
  result. Without a thinness demand, the good region of the archive is all transmission and folding never
  competes.
- **(a) Ray count:** a single element with a small bounce budget has few distinct sequences → few rays needed
  to estimate the interaction distribution. The cost is **branching** (each partial mirror doubles the set;
  budget 6 → up to 64×), mitigated by energy-weighted pruning of dim branches.
- **The real limit:** the canonical fold/pancake needs **≥2 working surfaces** (a partial mirror + a curved
  mirror to retrace). A single element gives TIR but **cannot** produce the pancake skeleton. So single-element
  NS is a **cheap warm-up to re-show TIR emergence under CMA-ME** (de-risking optimizer + descriptors), **not a
  pancake demonstrator.** Plan the first *real* experiment as a 2–3 surface rotationally-symmetric NS scene.

### 2.5 Rotationally-symmetric NS env — endorsed (Q2)
Not a toy restriction; it is the **actual regime of the target** (the canonical pancake is rotationally
symmetric). Restricting to an optical axis with `z(r)` surfaces: (a) makes EFL / paraxial quantities / the
thinness descriptor well-defined; (b) collapses ray counts to a **radial fan**; (c) reuses DeepLens's surface
machinery unchanged; (d) keeps comparability to classical pancake patents for the canary. You lose freeform
skew-ray folds, which are not the phase-1 target. Compatible with the planning doc's fixed-N homogeneous slots.

### 2.6 The reward is not too complex, but two terms hide landmines (Q3)
1. **`valid_path_efficiency` / `ghost_energy` can smuggle prescription.** Defining "valid" as "the folded path"
   prescribes folding and breaks the thesis. **Define by outcome, not route:** valid = energy landing in the
   in-focus imaging region of the sensor/eyebox; ghost = energy reaching the sensor out of that region / out of
   focus. Measurable from spot + energy, names no topology. Getting this wrong kills the emergence claim.
2. **Scalarization weights are finicky and philosophically wrong here.** Compactness is *descriptive*, so it
   belongs on a **QD archive axis, not the scalar reward** — which the planning doc already does (BC2 =
   thickness/EFL). Move it there; reward simplifies to `imaging + w_e·efficiency − w_g·ghost`, and "does folding
   emerge under thinness pressure" becomes "do the high-thinness cells get populated by good designs" — the
   pancake canary as illumination, not a weighted penalty. Cleaner, more faithful, one fewer knob.

**Why few rays works here:** rotational symmetry → on-axis is a single **radial fan** (1D pupil sampling), not
a 2D/3D grid; first canary is on-axis + single wavelength (planning-doc decision); few surfaces + small budget
→ few sequences → cheap energy-fraction estimates; a **differentiable objective gives far more signal per ray**
than black-box. Branching is the only real multiplier, and it's prunable.

### 2.7 Polarization — phase it (Q7)
- **Not needed for phase 1** (the interesting science): emergent geometric folding, TIR re-demonstration, and
  recovering the fold *skeleton* are pure geometry/energy phenomena (half-mirror + curved mirror folds at ~25%
  with ghosts, no polarization).
- **Needed only for the phase-2 headline:** the canonical pancake's **ghost suppression + efficiency recovery
  (25% → 70–90%)** *is* a polarization mechanism (QWP + reflective polarizer + handedness flip on reflection).
  Required iff the claim includes "ghost suppression / efficiency recovery emerges."
- **Cost is real in either engine:** JS tracks s/p *intensities* + Fresnel split but has **no Jones state**
  (complex amplitude + phase) and no QWP/reflective-polarizer types; DeepLens has no polarization. Jones
  tracking + polarization-dependent operators + new surface types is a genuine phase-2 build regardless. On-axis
  rotationally-symmetric phase 1 lets you defer it cleanly.

---

## 3. Resolved-for-now positions and the de-risking order

**Leaning positions (subject to the phase-1 probe):**
- Optimizer: **CMA-ME / MAP-Elites with a differentiable objective**, not pure RL and not full DQD. Persistence
  is the mechanism for the sparse, discontinuous NS landscape.
- Engine: **node.js (instrumented) for the gradient-free probe; DeepLens→NS for the differentiable research
  engine.**
- Descriptors: interaction-sequence based (discrete) → evaluated, not differentiated; continuous proxies
  optional for DQD.
- Reward: **imaging + efficiency − ghost**, with **compactness on a BC axis**; valid/ghost defined by outcome.
- Scope: **rotationally symmetric, on-axis, single wavelength, ≥2 surfaces** for the first real experiment.
- Polarization: **phase 2 only.**

**De-risking order (no building committed yet):**
1. Instrument the JS engine for per-ray interaction sequences (doc 02 §E.1); run a **CMA-ME probe** on a
   2-surface rotationally-symmetric-ish meridional NS scene with **reflectivity as a free DOF** and **thinness
   on a BC axis**. Decisive question: does any high-thinness, good-imaging cell get populated by a folded
   design, or does the good region stay all-transmission? Cheap, gradient-free, tests whether the science is
   alive before any tracer investment. Also empirically answers the "diversity while good" attribute question.
2. **Only if (1) shows life:** build **DeepLens → NS** (reuse `ray_reaction`; add argmin routing + branching +
   reflect/TIR variants; keep batched) for differentiable objective + L-BFGS refinement + CMA-ME exploration.
3. Polarization / full Jones engine = phase 2, gated on a successful phase-1 geometric fold.

**The one belief to hold onto:** don't expect a single element to show folding. It's an optimizer/descriptor
warm-up. The pancake needs ≥2 working surfaces, and good imaging fights interaction diversity until compactness
pressure is present.

---

## 4. Open forks still unresolved (for downstream agents)
- Exact descriptor axes that **separate** fold topologies without collapsing them (HyperLens BC-audit lesson).
- Multi-step geometry-polishing MDP vs single-step emit-and-score, if any RL is used at all.
- Whether the JS probe should be 2D-meridional-as-a-proxy or whether even the probe should wait for DeepLens-NS
  (i.e., is a non-rotationally-symmetric 2D probe informative enough to trust?).
- Precise canary criteria (FOV / eyebox / efficiency bar) — must be fixed before a discovery run.
- Whether DeepLens-NS straight-through gradient bias at basin walls is tolerable in practice, or whether
  soft-argmin (with temperature) is needed from the start (planning doc currently says hard + straight-through).
