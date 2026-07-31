# Prompt (Fable-tuned) — Direct the NS Optical-Discovery Research

> A de-prescribed variant of [04_milestone_experiments_prompt.md](04_milestone_experiments_prompt.md) for
> **Claude Fable 5**. Fable produces worse output when over-prescribed — it wants the goal, the real
> constraints, and the pointers, then room to reason and structure the answer itself. So this version states
> *what matters* and *what's off-limits*, names where to read, and hands it the wheel. Run it at high or xhigh
> effort. Give it room — a good answer here may take several minutes.
>
> Copy the fenced block into a fresh Fable 5 agent with repo access to `/home/hisham26/Desktop/RL`.

---

```
You are a senior research collaborator in computational optics, reinforcement learning, quality-diversity
optimization, and differentiable simulation. I want your judgment, not a recap.

There is a research program in this repo whose context is already written down. Read it before forming any
view — it carries the thesis, the assets, the prior art, the physics, and a design discussion that has
already resolved several questions and surfaced others:

  agent_context/ns_rl_research/01_extracted_research_context.md   (thesis, assets, prior art, the NS spec)
  agent_context/ns_rl_research/02_state_and_directions.md         (what the code actually does today)
  agent_context/ns_rl_research/03_design_discussion.md            (decisions made, forks still open)
  agent_context/ray_optics_engine.md                              (the working engine + the bridge)
  optical_rl/qd/lensmega/docs/ns_extension_context.md             (the planned NS tracer's decisions)

Then read enough of the actual code to ground yourself — at least optical_rl/envs/, the ray-optics bridge,
the vendored JS tracer (ray-optics/src/core/Simulator.js), and DeepLens/deeplens/optics/geolens.py. The docs
are a map, not a substitute; check the territory where it matters to your argument.

The goal of the program: discover folded and pancake-class optical systems whose multi-bounce behavior
*emerges* from an imaging-and-compactness objective — never prescribed. The scientific claim is emergence,
generalizing a prior result where total internal reflection emerged in a single deformable element under a
reward that never mentioned reflection. The near-term engineering question is how to extend the simple
non-sequential RL environment so that each ray's tracked ray-surface interaction sequence can become usable
signal.

What I want from you: decide the handful of milestone experiments — I'm thinking around four, but trust your
own read — that most efficiently move this program forward or most decisively test whether it can work at all.
Some should test a fundamental hypothesis, some should de-risk a build decision before it's paid for, some
should establish a concrete capability on the path to the canary (rediscovering the pancake skeleton). Rank
them, defend the order, and be honest about which ones could mislead us or are dominated by something we
already have.

These are real boundaries, not suggestions — an experiment that crosses them isn't interesting to me:

- Emergence is the whole point. No reward, observation, action, or descriptor-as-objective may prescribe
  folding, thinness, reflection count, or any topology. Interaction data is fair game as a descriptor, and as
  the *measurement basis* of a physically-grounded objective term (optical path length, in-focus vs stray
  energy) — but "valid" energy must be defined by where it lands, never by which path it took. Rewarding bounce
  count is the line you don't cross.
- Precision stays with the optimizer; a learned component contributes coarse structure, not numerics.
- Non-sequential does not mean non-differentiable — conditioned on a ray's realized interaction sequence the
  trace is differentiable in the parameters; the discreteness only bites at basin walls. Don't propose an
  exotic alternative to backprop as a load-bearing idea.
- Prefer experiments that reuse what exists (the JS engine once instrumented, the aggregated archives, the
  epsilon-distinct and hull-coverage metrics, DeepLens's differentiable surface primitives) over ones that
  need the unbuilt differentiable NS tracer first. Say plainly what each one depends on.
- A single deformable element is a warm-up, not a pancake demonstrator — good imaging fights interaction
  diversity until compactness pressure exists, and the canonical fold needs at least two working surfaces.

The default regime the program is leaning toward, unless you can argue otherwise: rotationally symmetric,
on-axis, single wavelength, two or more surfaces, continuous per-surface reflectivity initialized near zero,
compactness on a quality-diversity archive axis rather than in the scalar reward, CMA-ME / MAP-Elites with a
differentiable objective rather than pure RL, polarization deferred.

Where you disagree with the docs or with that default, say so and why — the design discussion is a position,
not a verdict, and I'd rather you correct it than echo it. For each experiment make the decisive readout
concrete (a threshold on a named metric), state what a negative result should make us stop doing, and flag any
dependency on the unbuilt tracer. Close with the one or two you'd actually run first given that the engine,
the archives, and the metrics exist today while the tracer does not — and with the open questions you'd want a
literature check on, with the search terms you'd use.

Lead with your conclusion. I want the experiments and the reasoning, not a tour of what you read.
```

---

## Notes

- **Why this differs from 04:** no numbered output schema, no per-field template, no enumerated deliverables.
  Fable degrades under that scaffolding; it does better told the goal, the boundaries, and "use your judgment."
  The constraints are kept (they're real physics/thesis boundaries) but framed as lines, not a checklist.
- **Effort:** run at `high` or `xhigh`. Expect a multi-minute turn; that's normal for Fable on a demanding,
  open-ended question.
- **If you'd rather have the rigid, fielded output** (ranking rationale, dependency graph, kill-switch map),
  use [04_milestone_experiments_prompt.md](04_milestone_experiments_prompt.md) instead and run it on Opus 4.8 —
  that prompt is tuned for the explicit style Opus wants.
- **Sanity check on the result:** if Fable's first-ranked experiment isn't some cheap, gradient-free probe on
  the existing engine, interrogate the ranking — that's the cheapest decisive test of whether the science is
  alive (see [03_design_discussion.md](03_design_discussion.md) §3).
