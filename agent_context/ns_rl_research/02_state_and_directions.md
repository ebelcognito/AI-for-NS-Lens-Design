# State of the NS RL Functionality, and What the Interaction-Signal Direction Requires

> Companion to [01_extracted_research_context.md](01_extracted_research_context.md) (the verbatim research
> conversation). This doc is **my reflection, grounded in the actual code** as of 2026-06-12: what exists,
> what is already achieved, what is missing, and what must be built for the stated direction —
> **using each ray's tracked ray-surface interaction sequence as signal to guide the RL process toward
> emergent folding** — with specific attention to the reward and the step updates. Written for advanced
> agents who will build on top of it. File:line references are from a snapshot; verify before relying.
>
> Engine/infra background is in [../ray_optics_engine.md](../ray_optics_engine.md). Read that first if you
> have not.

---

## TL;DR (the five things that decide this direction)

1. **The "NS ray tracer" the RL env uses today is the vendored JS engine** ([../../ray-optics/](../../ray-optics/)),
   driven from Python over stdin/stdout JSON. It **is** genuinely non-sequential (nearest-intersection,
   partial mirrors, TIR, in-place ray recursion). The differentiable PyTorch NS tracer is a *separate, unbuilt*
   plan ([ns_extension_context.md](../../optical_rl/qd/lensmega/docs/ns_extension_context.md)). **For
   model-free RL (PPO) you do not need the differentiable one** — you can pursue the interaction-signal
   direction on the JS engine *now*, once it is instrumented. This is the single most important strategic fact.
2. **The exact signal the direction wants — per-ray interaction sequences — is computed inside the engine but
   thrown away.** Only a scalar (`detector.power`) plus a couple of aggregate counters leave the engine. So
   the direction is *blocked on instrumentation*, not on a new tracer. The hooks are identified below.
3. **Today's reward is essentially `reward = detector power`** (minus small boundary/movement penalties). There
   is no compactness term, no efficiency-through-valid-paths term, no stray-light/ghost accounting. The
   research direction's reward crux needs all of those, and several of them can only be computed *from*
   interaction sequences — which is why #2 is the gatekeeper.
4. **There is a real tension between the direction ("use interactions as signal") and the emergence thesis
   ("never prescribe reflection/folding").** Resolving it is a design decision, not a detail: interaction data
   is safe as a *descriptor* and safe inside *physically-grounded objective terms* (optical path length →
   thinness, valid-path efficiency, ghost energy); it is thesis-violating if used to *reward bounces directly*.
   §D works this out.
5. **There is no QD archive on the RL side.** HyperLens's QD/persistence machinery lives on the separate
   differentiable *sequential* side. The conversation argues the sparse NS reward regime wants QD/hybrid, not
   pure RL. Bridging the JS-NS env to an archive indexed by interaction-sequence descriptors is a major,
   high-value build item.

---

## A. What exists today (code-grounded map)

### A.1 The working NS tracer (JS) and how RL reaches it
- Engine core: [../../ray-optics/src/core/Simulator.js](../../ray-optics/src/core/Simulator.js). The
  per-ray loop is `processRays()` (~lines 520–936): for each pending ray it finds the **nearest** surface via
  `opticalObjs[i].checkRayIntersects(ray)` (non-sequential), then calls
  `s_obj.onRayIncident(ray, j, s_point, surfaceMergingObjs)` (~line 878), which returns
  `{isAbsorbed, newRays, truncation}`. Rays are mutated **in place** (`ray.p1`, `ray.p2` reassigned) and set
  to `null` once terminal. A `depth` counter bounds recursion; partial mirrors push children to `newRays`.
- Interaction physics is real and already present:
  - **TIR vs refract vs partial-reflect** decided in
    [../../ray-optics/src/core/sceneObjs/BaseGlass.js](../../ray-optics/src/core/sceneObjs/BaseGlass.js)
    `refract()` (~lines 197–207): `sq1 = 1 - n1*n1*(1 - cos1*cos1)`; `sq1 < 0` ⇒ TIR, else refract with a
    Fresnel-weighted secondary (partial-reflection) ray spawned if bright enough.
  - **Mirror** reflection in `mirror/Mirror.js`; **BeamSplitter** in `mirror/BeamSplitter.js`; **absorb** in
    `blocker/Blocker.js`; **detector** accumulation in `other/Detector.js`.
- Integration boundary: [../../ray-optics/integrations/simulation_core.js](../../ray-optics/integrations/simulation_core.js)
  `runSingleSimulation(sceneJson)` extracts **only** detector readings (`power`, `normal`, `shear`,
  `irradianceMap`, `binPositions`) plus aggregate stats (`totalTruncation`, `processedRayCount`,
  `brightnessScale`). The persistent NDJSON loop is `runner_persistent.js`.
- Python bridge: [../../optical_rl/utils/ray_optics_bridge.py](../../optical_rl/utils/ray_optics_bridge.py)
  (one-shot) and [../../optical_rl/utils/ray_optics_bridge_persistent.py](../../optical_rl/utils/ray_optics_bridge_persistent.py)
  (`PersistentRayOpticsRunner`, the workhorse for training loops).

### A.2 The RL envs (the "simple NS RL env")
All in [../../optical_rl/envs/](../../optical_rl/envs/). They are gym-style, PPO via stable-baselines3
(training entry points `optical_rl/main_stage1*.py`). Representative set:

| Env | File | Rays | Action | Deformable DOF |
|-----|------|------|--------|----------------|
| SimpleSteeringEnv | `simple_steering_env.py` | 1 | `Box(-1,1)` | quad position X |
| SingleSurfaceCurvatureEnvV1 | `single_surface_curvature_env_v1.py` | 1 | `Box(-1,1)` | left-surface sag |
| DualSurfaceCurvatureEnvV2 | `dual_surface_curvature_env_v2.py` | 1 | `Box(-1,1,2)` | both surfaces |
| **QuadCurveGlassEnv** (+v2/v3) | `quad_curve_glass_env.py` | **2–4** | `Box(-1,1,12)` | 8-point deformable glass (4 corners + 4 arc control pts) |

The **TIR-emergence existence proof** corresponds to the QuadCurveGlass family (a 4-ray deformable-glass
refocusing task; config `configs/stage1_quad_curve_glass.yaml` places rays at several Y heights). Episodes are
multi-step: `step()` applies a bounded geometry delta → rebuilds the scene
([../../optical_rl/utils/scene_builder.py](../../optical_rl/utils/scene_builder.py)) → calls the tracer →
reads back power.

### A.3 The reward today (verbatim from code)
- `simple_steering_env.py`: `reward = self.current_power`
- `single_surface_curvature_env_v1.py:238`:
  `reward = self.current_power - boundary_penalty*was_clipped_sag - movement_penalty_coef*delta_sag_mm`
- `quad_curve_glass_env.py:178`: `reward = float(self.current_power)` (v2/v3 add a scalar `reward_scale`)

`current_power` is `result["detectors"][0]["power"]`. **No env references reflection, TIR, bounce count, path
length, or folding.** That is exactly why the TIR result is a clean emergence result — and exactly the
discipline the new direction must preserve.

---

## B. What is already achieved (mapped to the research directions)

1. **Non-sequential tracing is real and battle-tested.** Nearest-intersection, revisits, partial mirrors, TIR,
   energy-weighted recursion with a depth budget already work in the JS engine. The NS prompt's "the tracer
   does not impose a surface order" requirement is already satisfied.
2. **Emergence of a multi-interaction strategy is already demonstrated** (TIR in QuadCurveGlass under a
   power-only reward). The whole NS direction is a generalization of a result you already have, not a leap.
3. **A fast Python↔engine loop exists** (`PersistentRayOpticsRunner`, with timeout + auto-restart), suitable
   for RL rollouts without per-call Node startup cost.
4. **A scene-construction layer exists** ([scene_builder.py](../../optical_rl/utils/scene_builder.py)) that
   already parameterizes deformable elements as point/arc paths — the natural place to add multi-surface,
   reflectivity, and mirror DOF.
5. **Polarization is *partially* present.** The engine tracks `brightness_s` / `brightness_p` (s- and
   p-intensities) and does Fresnel splitting, and `BeamSplitter` exists. So phase-1 energy-split folding is
   buildable on current primitives. (Caveat in §E.6.)
6. **The QD/persistence philosophy and tooling exist** in the codebase (HyperLens side: archive,
   epsilon-distinct metric, behavior-hull coverage) — not wired to the RL env, but available to port.

---

## C. The core gap: interaction sequences are computed, then discarded

This is the gatekeeper for the stated direction. The information the direction needs **exists at the decision
points** in the engine but is **never recorded or exported**:

- In `Simulator.processRays()` the loop knows, at every strike: the **ray index** (`j`), the **surface object
  and index** (`s_obj`, `s_obj_index`), the **incident point** (`s_point`), and the **energy**
  (`ray.brightness_s + ray.brightness_p`). It does not know the *type* of interaction because
  `onRayIncident` does not return it.
- In `BaseGlass.refract()` the `sq1 < 0` branch *is* the TIR-vs-refract decision, and a spawned secondary ray
  *is* the partial-reflection event — but neither is labeled or surfaced upward.
- Rays are mutated in place and nulled on termination; **no per-ray path history is accumulated**, and
  `processedRayCount` / `totalTruncation` are scalar counters, not paths.
- `simulation_core.js` exports only detector aggregates; **there is no channel by which per-ray data leaves the
  engine.**

**Consequence:** "use the tracked ray-surface interactions seen by each ray as signal" is currently
*impossible* — not because the physics is missing, but because the trace is not retained. The build is
**instrumentation**, and it is structurally feasible (all data is live at the hooks). Concrete hook locations
are in §E.1.

---

## D. The reward / step-update analysis (and the emergence-thesis tension)

This section is the heart of the reflection because the researcher explicitly flagged reward and step updates.

### D.1 The tension, stated plainly
The direction says "use tracked interactions as signal." The thesis says "never prescribe reflection/folding."
These collide if interaction data enters the reward as *counts of bounces/reflections*. They do **not** collide
if interaction data is used in either of two thesis-safe ways:

- **(Safe-1) As descriptors / archive axes / diagnostics**, never in the reward. This is exactly the NS
  prompt's "behavior descriptors from interaction sequences" and the HyperLens analogue. The agent is neither
  rewarded nor observed on them; they only characterize and diversify what emerged. **Fully thesis-clean.**
- **(Safe-2) Inside physically-grounded objective terms that do not *name* folding** but can only be *computed*
  from the interaction sequence. Two are load-bearing and both are sanctioned by the NS prompt's reward crux:
  - **Compactness / thinness ratio** = effective focal length ÷ physical axial track length. The *optical path
    length* needed to relate these correctly in a folded system can only be reconstructed from each ray's
    interaction sequence (sum of segment lengths between strikes). You are rewarding thinness, not bounces; the
    bounces are how thinness is achieved, exactly as TIR was how the single-element objective was achieved.
  - **Efficiency through valid paths + stray-light/ghost accounting.** "Energy delivered to the sensor via a
    valid interaction sequence" vs "energy that reaches the sensor by an unintended path (ghost)" is precisely
    a function of the per-ray interaction sequence. Without it you cannot stop the agent gaming the imaging
    metric with stray light. You are rewarding honest throughput, not reflection count.

**The line (state it in any paper):** interaction sequences may *define how an objective term is measured*
(path length, valid-path energy, ghost energy) and may *index diversity*; they may **not** appear as a
target/penalty on reflection count, bounce count, or "foldedness." `reward += k * num_reflections` is the
thesis violation to avoid; `reward += compactness(opl, track_len)` is not.

### D.2 What the reward must become (vs. the scalar today)
Today: `reward ≈ detector.power`. The NS-direction reward (phase 1, non-polarized) is, conceptually:

```
reward =  imaging_quality(spot/wavefront at eyebox)              # RMS spot or WFE; later MTF
        + w_c * compactness(EFL / physical_track_length)          # the thinness pressure (Safe-2)
        + w_e * valid_path_efficiency(energy via valid seqs)      # honest throughput (Safe-2)
        - w_g * ghost_energy(energy via invalid/stray seqs)       # anti-gaming (Safe-2)
# NO term references reflection / bounce / fold.
```

`imaging_quality` needs more than `detector.power`: it needs the **spot distribution** at the sensor (the
engine already returns `irradianceMap`/`binPositions`, a usable starting point for an RMS-spot proxy) and,
for valid/ghost separation, the **per-ray interaction sequence** to label which contributions are valid.
Compactness needs **optical path length per ray** (segment-length sum from the sequence). So three of the four
terms are blocked on §C instrumentation.

### D.3 Step-update considerations (reward timing / credit assignment / episode shape)
- **Episode shape.** Current envs are multi-step geometry-nudging MDPs with a per-step power reward. For the
  NS direction, decide deliberately between (a) keeping the multi-step "polish the geometry" MDP (dense-ish
  reward, the heritage that produced TIR) and (b) a single-step "emit a full design, get one verifiable
  reward" bandit (closer to OptiAgent/Sim2Reason framing, cleaner for an archive-improvement reward). The
  conversation's sparse-reward warning (NS prompt fork 2) bites hardest in (b).
- **Reward timing and shaping from interaction data — caution.** It is tempting to hand out *dense per-strike
  shaping* using the live interaction stream (e.g. reward as a ray progresses). Be careful: per-strike shaping
  that rewards reaching/using a surface is a backdoor into prescribing topology and can violate the thesis.
  Prefer terminal, energy-weighted objective terms computed *after* the full trace. If shaping is needed for
  sample efficiency, shape on **imaging progress and valid-path efficiency**, never on interaction structure.
- **Sparse reward ⇒ persistence.** The conversation argues (and HyperLens demonstrates) that under a hard,
  discontinuous NS landscape, *persistence of behaviorally-novel-but-poor candidates* is what reaches deep
  basins. Pure PPO will likely settle into the nearest single-pass basin. This is the strongest argument for a
  **QD/hybrid** outer loop (archive indexed by interaction-sequence descriptors) wrapping the
  RL/gradient local moves — see §E.5. Treat "pure RL vs QD-hybrid" as the central optimization fork to resolve
  before scaling.
- **Branching cost in the loop.** Partial mirrors make the path tree exponential in the bounce budget. The
  reward/step cost per design grows with the budget; budget, energy-weighted pruning, and the
  evaluations-to-coverage economics all interact. This is the regime where amortization/QD pays off (NS prompt;
  §2.6 of the context doc).

---

## E. What must be built (concrete, phased)

Ordered so the earliest items unblock the most. None of E.1–E.4 require the differentiable PyTorch tracer.

### E.1 Instrument the JS engine to emit per-ray interaction sequences *(unblocks everything)*
- Add an `interactionType` to the `SimulationReturn` of `onRayIncident` across object types:
  `BaseGlass.refract()` returns `'TIR'` on the `sq1<0` branch, `'partial-reflect'` when a secondary ray is
  spawned, else `'refract'`; `Mirror`/`BeamSplitter` return `'reflect'`; `Blocker` returns `'absorb'`;
  `Detector` returns `'sensor'`.
- In `Simulator.processRays()`, maintain a per-ray accumulator (e.g. `this.rayInteractionSequences`,
  `rayId -> [{surfaceId, type, brightness_s, brightness_p, point, depth}]`), appended right after the
  `onRayIncident` call (~line 878), and propagate the parent sequence to spawned `newRays` so branches inherit
  history. Handle ray-id assignment for spawned children explicitly.
- Gate it behind a scene/sim flag (e.g. `recordInteractions: true`) so default/perf paths are unaffected
  (per-ray logging is not free).
- Export it in `simulation_core.js` `result.raySequences` and confirm it survives the NDJSON path in
  `runner_persistent.js`.
- **Validate** against the existing TIR scene: a known TIR ray must show a `'TIR'` event at the expected
  surface. Self-intersection/epsilon-offset robustness on revisits must be checked (NS prompt: re-intersection).

### E.2 Extend the Python bridge + add interaction-sequence descriptors
- Parse `result.raySequences` in both bridges; expose a typed structure to the env.
- Implement the NS-analogue descriptors from the interaction-sequence distribution (NS prompt):
  **dominant-path bounce count**, **fraction of energy in the dominant sequence**, **path-length ratio**,
  **thinness ratio**. Keep these as Python utilities so they serve both reward terms (Safe-2) and archive axes
  (Safe-1).

### E.3 Redesign the reward (phase-1, non-polarized) per §D.2
- Replace scalar power with the four-term objective. Build an RMS-spot (or WFE) proxy from the sensor
  distribution; build `valid_path_efficiency` and `ghost_energy` from labeled interaction sequences; build
  `compactness` from per-ray optical path length. Keep weights configurable. Encode the thesis line from §D.1
  in code comments and config (no reflection/fold term).

### E.4 Expand the action/scene space for multi-surface NS designs
- The NS prompt's faithful fork-1 choice: the agent sets **geometry + continuous per-surface reflectivity
  only**; interaction sequences emerge from physics. So extend `scene_builder.py` and the env action space to
  fixed-N homogeneous surface slots with continuous curvature/position/tilt/**reflectivity**, plus the option
  of fully reflective surfaces (mirrors). Reflectivity initialized at/near zero so folding is not seeded
  (matches the NS planning doc's discipline and Path C's caution).
- Add an eyebox/exit-pupil sampling of rays (the canary is defined at a stated FOV and eyebox).

### E.5 Wrap an archive (QD/hybrid) indexed by interaction-sequence descriptors *(the persistence mechanism)*
- Port the HyperLens-side archive + epsilon-distinct + behavior-hull tooling to index designs by
  (e.g.) `(path_length_ratio, thinness_ratio)` or `(dominant_bounce_count, energy_fraction)` — **descriptive,
  not prescriptive** axes. Use RL/gradient for local moves and the archive for persistence. This is the
  conversation's recommended primary for the sparse NS regime; it is the largest build but the one that most
  directly carries the "discovery" claim.

### E.6 Polarization (phase 2 only) — needs new engine capability
- Current engine carries **s/p intensities** and Fresnel splitting, but **not a full Jones state**
  (amplitude + relative phase), and has **no quarter-wave-plate or reflective-polarizer surface types** (grep
  of `sceneObjs/` finds no waveplate/retarder/Jones; only `brightness_s`/`brightness_p` intensities). The
  canonical pancake's ghost suppression is a *phase* phenomenon (QWP + reflective polarizer). So phase 2
  requires: a per-ray Jones vector (complex amplitudes, tracked phase), polarization-dependent surface
  operators, and QWP/reflective-polarizer object types. This is a substantial engine extension — keep it
  strictly out of phase 1. (Whether to do this in the JS engine or finally in the planned PyTorch tracer is an
  open fork — see §F.)

### E.7 The differentiable PyTorch tracer — when (and whether) you actually need it
- Per the NS planning doc, this is for *gradient-based* QD (CMA-MEGA needs `∂(objective)/∂θ`). **Model-free RL
  does not.** Build it only if/when you commit to the gradient-based or CMA-MEGA-hybrid optimization fork, or
  need autograd through the trace for the differentiable-rendering alternative the NS prompt weighs. For an
  RL-first interaction-signal program, E.1–E.5 on the JS engine come first and de-risk the science before the
  ~4-week tracer build.

---

## F. Open forks for the advanced agents to resolve (don't leave implicit)

1. **Optimization paradigm** (NS prompt fork 2): pure PPO vs gradient/differentiable vs **QD-hybrid**. The
   conversation leans QD-hybrid for the sparse NS regime; confirm with a small sparse-reward pilot on the JS
   engine before committing tracer effort.
2. **Where interaction data is allowed to enter** (§D.1): descriptors only (Safe-1), or also physically-grounded
   objective terms (Safe-2). Recommend Safe-1 + Safe-2 with the explicit line documented; forbid bounce/reflection
   penalties.
3. **Episode shape** (§D.3): multi-step geometry-polishing MDP (TIR heritage) vs single-step verifiable-reward
   bandit (OptiAgent/Sim2Reason framing, archive-improvement reward). Affects credit assignment and the archive
   design.
4. **Which engine carries polarization** (E.6): extend the JS engine with Jones state, or make phase 2 the
   first real use of the planned PyTorch tracer. Trade-off: JS reuse vs differentiability + cleaner physics.
5. **Descriptor axes for the archive** (E.5): which 2 (or more) interaction-sequence descriptors actually
   *separate* fold topologies without collapsing them (the HyperLens "BC-space audit" lesson — wrong axes
   re-introduce discontinuities intra-cell).
6. **Canary definition** (NS prompt deliverable 4): the precise pancake-skeleton rediscovery criteria at a
   stated FOV/eyebox/efficiency bar — must be fixed *before* a discovery run so the result is falsifiable.

---

## G. Bottom line

The direction is viable and closer than the "unbuilt NS tracer" framing suggests. The science can start on the
**existing JS non-sequential engine** because PPO does not need gradients. The one hard prerequisite is
**instrumenting the engine to retain and export per-ray interaction sequences** (§C/§E.1) — that single change
turns "use tracked ray-surface interactions as signal" from impossible into a tractable reward-design and
archive-design problem. The reward must grow from a scalar power into imaging + compactness +
valid-path-efficiency − ghost (§D.2), with interaction data entering only as descriptors and as the
*measurement basis* of physically-grounded terms (§D.1), never as a bounce/reflection target. Persistence
(a QD/hybrid archive over interaction-sequence descriptors) is the mechanism most likely to reach the deep
folded basins under the sparse, discontinuous NS landscape. Polarization and the differentiable tracer are
genuinely later phases and should not block phase 1.
