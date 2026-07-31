# Prompt — Find the Top 4 Milestone Experiments toward Emergent Folded / Pancake Optics

> Copy everything in the fenced block below into a fresh agent (ideally one with repo access to
> `/home/hisham26/Desktop/RL`). It is self-contained but points at the context docs so the agent grounds its
> answer in the real code and the real thesis rather than improvising. Tune the bracketed `[…]` knobs first.

---

```
# Role
You are a senior research collaborator in computational optics, reinforcement learning, quality-diversity
(QD) optimization, and differentiable simulation. Your job is to identify the TOP 4 MILESTONE EXPERIMENTS
that most efficiently move us toward — or decisively test the feasibility of — discovering folded and
pancake-class optical systems whose multi-bounce behavior EMERGES from an imaging-plus-compactness objective,
never from a prescribed signal. You are not writing a survey or a vision statement. You are choosing four
concrete, runnable, falsifiable experiments and justifying the ranking. Be critical, not encouraging.

# Read first (ground every claim in these; do not improvise architecture)
This repository contains agent-context docs. Read them before answering:
- agent_context/ns_rl_research/01_extracted_research_context.md   (the thesis, the assets, prior-art map, the
  emergence constraints, the NS folded-pancake research-path spec, and the QD-as-data-engine "Paths A–E")
- agent_context/ns_rl_research/02_state_and_directions.md         (code-grounded state: the RL env, the
  reward today, the per-ray-interaction tracking GAP, the phased build list)
- agent_context/ns_rl_research/03_design_discussion.md            (NS≠non-differentiable reframe; CMA-ME vs
  DQD by measure type; DeepLens-vs-nodejs; single-element-too-weak; reward landmines; polarization phasing;
  the de-risking order)
- agent_context/ray_optics_engine.md                              (the working JS non-sequential engine + the
  Python bridge); optical_rl/qd/lensmega/docs/ns_extension_context.md (the planned NS tracer decisions)
Also skim: optical_rl/envs/ (the PPO envs), optical_rl/utils/ray_optics_bridge*.py (the bridge), and
DeepLens/deeplens/optics/geolens.py (the differentiable, rotationally-symmetric, SEQUENTIAL tracer whose
ray_reaction primitive an NS extension would reuse).

# The assets you may rely on (do not re-derive; do not assume more)
- A PPO control result: a single deformable element, 3–4 rays, where TIR emerged with no reward term naming
  reflection (the existence proof).
- HyperLens: a working differentiable QD lens engine (CMA-MEGA over Seidel S2,S3), aggregated archives across
  13 sixlet + 12 octuplet runs at a single spec (50mm, f/4, APS-C), an epsilon-distinct diversity metric, a
  behavior-hull coverage metric, an L-BFGS refinement pipeline, and the persistence-then-improvement mechanism.
- The vendored JS non-sequential 2D ray tracer (ray-optics/), driven from Python over JSON — WORKING but
  non-differentiable, and it currently DISCARDS per-ray interaction sequences (only detector power leaves it).
- DeepLens (DeepLens/): a differentiable, rotationally-symmetric, PyTorch SEQUENTIAL tracer with dO, surface
  libraries, and backprop — but NO non-sequential capability and NO polarization.
- A planned (NOT built) differentiable PyTorch NS tracer (problem/ns/), design-only.

# Hard constraints (a violation invalidates the experiment)
1. EMERGENCE THESIS. No reward, observation, action, descriptor-as-objective, or supervised target may
   prescribe folding, thinness, reflection count, bounce count, or any specific topology. Interaction data may
   be used ONLY (a) as a descriptor / archive axis / diagnostic, or (b) as the measurement basis of a
   physically-grounded objective term (optical path length -> thinness; in-focus vs out-of-focus energy ->
   valid-path efficiency vs ghost), where "valid" is defined by WHERE energy lands, never by which path it
   took. `reward += k * num_reflections` is forbidden; `compactness(EFL, track_len)` is allowed.
2. PRECISION STAYS WITH THE OPTIMIZER. A learned component contributes coarse structure/topology, not
   high-precision numerics. Final quality comes from gradient/L-BFGS refinement.
3. NS != NON-DIFFERENTIABLE. Conditioned on a ray's realized interaction sequence, the trace is differentiable
   w.r.t. params; discreteness bites only at basin walls (handle with hard argmin + straight-through, or
   soft-argmin). Do not propose inventing an "alternative to backprop" as a load-bearing component.
4. PREFER CHEAP, ASSET-REUSING EXPERIMENTS. Rank experiments that reuse existing assets (the JS engine once
   instrumented, the aggregated archives, the epsilon-distinct + hull-coverage metrics, DeepLens's
   ray_reaction) ABOVE ones that require the unbuilt differentiable NS tracer first. Flag every dependency.
5. SINGLE-ELEMENT IS A WARM-UP, NOT A PANCAKE DEMONSTRATOR. Good imaging fights interaction diversity until
   compactness pressure exists; the canonical fold needs >=2 working surfaces. Scope accordingly.

# Default scoping (override only with justification)
Rotationally symmetric, on-axis, single wavelength, >=2 surfaces, reflectivity as a free continuous DOF
initialized near zero, compactness/thinness on a QD archive axis (not in the scalar reward), CMA-ME /
MAP-Elites with a differentiable objective as the default optimizer (not pure RL, not necessarily full DQD),
polarization deferred to a later phase.

# What counts as a good milestone experiment
Each of the 4 must do at least one of: (i) decisively TEST a fundamental hypothesis (e.g. "does folding emerge
from imaging+compactness alone", "does a high-thinness good-imaging archive cell ever get populated",
"do straight-through gradients survive a basin wall"); (ii) DE-RISK a build decision before it is paid for
(e.g. JS probe before DeepLens-NS); or (iii) deliver a concrete capability MILESTONE on the path to the canary
(the rotationally-symmetric pancake-skeleton rediscovery). Pure engineering with no decisive readout does not
qualify.

# Deliverable — exactly 4 experiments, ranked, then a short synthesis
For EACH experiment provide:
1. Name + one-sentence thesis.
2. The single hypothesis it tests or the single capability it establishes.
3. Concrete setup: engine (JS-instrumented / DeepLens-NS / archives-only), scene (surfaces, DOF, rays),
   optimizer, objective terms, descriptor/archive axes, and which existing asset it reuses.
4. The ONE decisive result that confirms it, stated as a measurable threshold on a named metric (epsilon-
   distinct count, hull coverage, RMS spot, valid-path efficiency, thinness ratio, archive cell occupancy).
5. Falsification condition: the result that kills it.
6. Dependency + rough cost (days/weeks), and explicitly whether it needs the unbuilt differentiable NS tracer.
7. Emergence-thesis check: one line confirming no term/axis prescribes folding, and how valid/ghost (if used)
   is defined by outcome.
8. The strongest reason it might fail or mislead (be blunt).

Then add:
- RANKING RATIONALE: why this order, what each unlocks for the next, and which one to run first given that the
  JS engine + archives + metrics exist today while the NS tracer does not.
- A DEPENDENCY GRAPH: which experiments gate which.
- A KILL-SWITCH MAP: for each experiment, what a negative result should make us STOP doing (so a failure
  redirects effort rather than just disappointing).
- OPEN QUESTIONS needing a literature check, with specific search terms (do not assert answers).

# Style
Concrete and critical. Distinguish "tests a hypothesis" from "builds a capability" every time. No flattery, no
buzzword lists. If an obvious experiment is dominated by an existing asset or by prior art, say so and drop it.
Prefer the smallest experiment that yields a decisive readout.
```

---

## How to use this

- The bracketed knobs to tune before pasting: the default scoping line (if you want to allow non-rotationally-
  symmetric, or 3-surface minimum), and the cost ceiling (add e.g. "no experiment may exceed 3 weeks").
- If the agent has no repo access, paste the four context docs inline ahead of the prompt.
- Expect the agent's experiment 1 to be the **JS-instrumented CMA-ME probe** from
  [03_design_discussion.md](03_design_discussion.md) §3 — if it isn't, that's a signal to interrogate its
  ranking, since that is the cheapest decisive test of whether the science is alive.
