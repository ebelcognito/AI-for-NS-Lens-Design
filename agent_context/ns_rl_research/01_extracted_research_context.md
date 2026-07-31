# Extracted Research Context — NS Optical Design, QD-as-Data-Engine, and Emergent Folding

> **Provenance.** This file is an extraction of a prior strategy conversation between the
> researcher (h4742m) and an advanced agent. It is preserved **verbatim / minimally reworded**
> on purpose. Do not compress it; later agents are meant to read the original reasoning, not a summary.
>
> **Source PDFs referenced throughout** (explained inline where they are used, so you do not need the PDFs):
> 1. **HyperLens** — the researcher's own paper. Differentiable QD lens design with CMA-MEGA over Seidel (S2, S3) descriptors; persistence-then-improvement mechanism.
> 2. **OptiAgent** — agentic LLM design of sequential refractive systems, physics-driven lexicographic reward, Zemax/optimizer refinement.
> 3. **Sim2Reason** — RL on physics simulators as a scalable verifiable signal, sim-to-real transfer.
> 4. **Collaborator's proposal** — an LLM proposing single-element designs with internal ray-path topology (TIR, partial reflection, folded paths).
> 5. **Context files** — the researcher's project notes and the NS extension planning doc (see [../../optical_rl/qd/lensmega/docs/ns_extension_context.md](../../optical_rl/qd/lensmega/docs/ns_extension_context.md)).
>
> **How to use this file.** Section 1 (the NS / folded-pancake research-path prompt) is the **most
> relevant** to the current goal of extending the simple NS RL environment. Section 2 (the
> QD-as-data-engine "Paths A–E" analysis) is attached for relevance-mining — it is mostly about a
> *different* direction (generative proposers trained on QD archives) but contains transferable
> framing about emergence, descriptor lossiness, RL-vs-SFT, and prior-art positioning.
> Section 3 collects the shared asset/thesis facts both prompts rely on.
>
> See the companion reflection doc: [02_state_and_directions.md](02_state_and_directions.md).

---

## 0. Shared assets and thesis (carried by both exchanges)

**The PPO control result (the existence proof).** A single deformable optical element, 3 to 4 rays,
where total internal reflection emerged as a learned strategy with **no reward term mentioning it**.
Validated as compelling by a lead optical engineer at Google. This is the existence proof for the whole
direction: a multi-interaction strategy emerged because it was the way to satisfy an imaging objective.

**HyperLens (attached).** A working differentiable QD lens-design engine: CMA-MEGA over (S2, S3); a
2048×2048 archive that is sparse by design (~1.0 to 1.35 percent cell occupancy); refined pools capped at
top-500; up to 99 vs 16 epsilon-distinct sixlet forms against an SA+GA baseline; the
**persistence-then-improvement** mechanism (a 924 µm ancestor stored only because its cell was empty
eventually yields the run's best result at 2.7 µm). Aggregated archives across 13 sixlet and 12 octuplet
runs already exist. The paper already defines an **epsilon-distinct diversity metric** and **behavior-hull
coverage** as evaluation tools. Demonstrated lesson: persistence of behaviorally diverse but poor
candidates is what reaches deep basins under a hard landscape.

**NS extension (planned, not yet built).** An own differentiable non-sequential PyTorch ray tracer,
fixed-N homogeneous surface slots, continuous per-surface reflectivity, descriptive behavior coordinates
(path-length-ratio, thickness/EFL), a bounce budget, soft branching at partial mirrors, and a
pancake-discovery canary. **The emergence thesis forbids any signal that prescribes folding, thinness, or
reflection count.**

**The thesis to preserve (stated for the NS task).** Folding, multi-bounce behavior, thinness, and
reflection count must never be prescribed by the reward, the action space, or any supervised target. They
must emerge from an objective that rewards only imaging quality, compactness, and efficiency. The
scientific claim is emergence, exactly generalizing the single-element TIR result to multi-surface
non-sequential systems.

---

## 1. MOST RELEVANT — The NS / folded-pancake emergence research-path prompt (verbatim)

> This is the second exchange. The researcher flagged it as the main relevant discussion ("things
> specifically about the NS framework"). Reproduced verbatim. **NOTE: the source transcript was truncated
> at the end of this prompt (cut off at "Defend the whitesp…"), and the agent's *response* to this NS
> prompt was NOT included in the provided transcript.** So below is the detailed *specification* the
> researcher posed, not an answer to it.

### Role
You are a senior research collaborator in computational optics, reinforcement learning, and differentiable
simulation. Your task is to produce one rigorous, concrete, end-to-end research path (phased, with
alternatives weighed) for discovering folded and pancake-class optical systems by learning over
non-sequential ray-surface interaction sequences. Do not produce a vague survey or a list of buzzwords.
Every design choice must be justified against the physics, the prior art, and the researcher's stated
thesis. Assume the obvious framing is already taken and find the version that is defensible and runnable.

(PDFs to read first: HyperLens, OptiAgent, Sim2Reason, the collaborator's proposal, and the context files;
treat the NS planning document as the most relevant context.)

### The problem, stated precisely
In a sequential ray tracer, surfaces are visited in a fixed order, once each, with transmission only. In a
non-sequential (NS) tracer, a ray may strike any surface in any order, may revisit a surface, and at each
strike may transmit, reflect (partially or fully), undergo TIR, scatter, or be absorbed, according to
geometry, incidence angle, and surface properties. Define a ray's **interaction sequence** as the ordered
list of (surface, interaction-type) events it undergoes from source to sensor. The system's behavior is the
energy-weighted distribution over interaction sequences that its geometry induces. A pancake or folded
system is one whose dominant interaction sequence is multi-bounce, producing an optical path length far
larger than the physical track length.

The goal is a method that, given an imaging-plus-compactness objective and no instruction to fold,
discovers geometries whose emergent interaction sequences fold the path; that can rediscover the canonical
pancake as a calibration canary; and that can discover novel folded topologies beyond the patent literature.

### Central design forks you must resolve and justify
Do not leave these open. Pick a position for the primary path, state the tradeoff, and name the alternative
you rejected.

1. **Emergent physics vs prescribed program.** Either (a) the agent sets only geometry plus continuous
   per-surface reflectivity and the tracer lets interaction sequences emerge from physics, or (b) the agent
   emits an intended interaction sequence and geometry to realize it. State which is faithful to the
   emergence thesis and why. Argue from the TIR result.
2. **Optimization paradigm.** Weigh three options for the non-differentiable, branchy NS environment:
   model-free RL (PPO, the researcher's heritage, native to discrete branching and non-differentiable
   dynamics); differentiable NS rendering with soft or relaxed branching (gradient-based, the HyperLens
   lineage, but the branch and next-surface decisions are discrete and need relaxation that may bias); and a
   QD or hybrid scheme that uses RL or gradients for local moves while keeping an archive for persistence.
   Note explicitly that imaging reward on random NS systems is extremely sparse, that this is the regime
   where HyperLens's persistence mechanism matters, and that this is a strong argument for a QD or hybrid
   formulation rather than pure RL. Recommend one for the primary path and justify it on sample efficiency
   and sparse-reward grounds.
3. **Polarization: phase it.** The canonical pancake routes light with a quarter-wave plate and a reflective
   polarizer so the desired multi-bounce path exits while single-pass stray light is suppressed; without
   this, a half-mirror plus curved mirror still folds but suffers ~25 percent efficiency and severe
   ghosting. Propose a phase 1 that is non-polarized (partial mirrors and curved reflectors, energy-split) to
   test whether folding emerges and to recover the geometric fold skeleton, and a phase 2 that carries a
   polarization state per ray (Jones calculus) with polarization-dependent surface operators, so the method
   can discover ghost suppression and efficiency recovery as a second-order emergent phenomenon. Treat "does
   polarization-selective routing emerge when the agent is given the degrees of freedom" as a headline result
   if it holds.

### The reward-design crux
This is the heart of the path and you must get it right. The objective must force folding without ever
naming it. Concretely, reward image quality at a defined exit pupil or eyebox (RMS spot or wavefront error,
and later MTF), jointly with a compactness term defined as a thinness ratio of effective focal length to
physical axial track length, and an efficiency or throughput term that counts only energy delivered through
valid interaction sequences to the sensor. Folding must be the emergent consequence of demanding a large
thinness ratio at acceptable image quality, in the same way TIR was the emergent consequence of the
single-element objective. State exactly why no term in this objective references reflection, bouncing, or
folding. Then specify how stray light and ghosts are accounted for, so the agent cannot game the imaging
metric by ignoring the energy that does not follow the intended path. The efficiency term and the stray-light
accounting are what make the result honest.

### Non-sequential mechanics you must specify, not hand-wave
- **Branching and tractability.** A ray hitting a partial mirror spawns reflected and transmitted children,
  so the path tree is exponential in the bounce budget. Specify the bounce budget, energy-weighted pruning of
  low-energy branches, and how the budget interacts with discovering deep folds.
- **Re-intersection.** Rays revisit surfaces, so specify epsilon offsetting and robust self-intersection
  handling, and confirm the tracer does not impose a surface order.
- **Differentiability strategy.** If gradient-based, specify exactly how the discrete next-surface selection
  and the transmit-versus-reflect decision are relaxed (for example energy-continuous splitting with a softmin
  over candidate intersections), and state the bias this introduces. If RL, specify the MDP: state (the
  partial or current design, or a ray-level state), action space (continuous curvatures, positions, tilts,
  reflectivities, and any discrete surface-type or add-remove actions), episode structure, and reward timing.
  If hybrid, specify the division of labor.
- **Behavior descriptors from interaction sequences.** Even in a pure-RL formulation, define descriptors that
  summarize the interaction-sequence distribution: dominant-path bounce count, fraction of energy in the
  dominant sequence, path-length ratio, and thinness ratio. These serve to characterize, cluster, and
  diversify discoveries, and they are the natural NS analogue of HyperLens's Seidel descriptors. Discuss
  whether indexing a QD archive by interaction-sequence descriptors is the cleanest way to get emergent
  diversity of folds.

### Deliverable requested (the phased path)
Produce one primary research path, phased, with the following fully worked out (present close competing
formulations as alternatives and say why the primary wins):

1. **Thesis and contribution claim.** One sentence of thesis, then a precise statement of what is new, and
   whether it is a new mechanism or a new instantiation.
2. **The method.** The resolved positions on all three forks, the reward, and the NS mechanics, specified
   concretely enough to implement.
3. **Phase plan.** Phase 0: build and validate the NS differentiable tracer on a known folded system
   (forward-only sanity check). Phase 1: non-polarized emergence — can folding appear from imaging plus
   compactness alone. Phase 2: polarization degrees of freedom — can ghost suppression and efficiency recovery
   emerge. Each phase states its hypothesis, its single decisive result, and its falsification condition.
4. **The canary.** A precise specification of the rediscovery target: recover the canonical polarization
   pancake skeleton from an imaging-plus-compactness-plus-efficiency objective with no fold prescription,
   analogous to HyperLens recovering the Cooke triplet and Double Gauss. State what counts as a successful
   rediscovery.
5. **The discovery target.** What a novel result looks like: folded topologies with different bounce
   structure, alternative ghost-suppression mechanisms, or freeform-mirror folds that are not the canonical
   pancake yet meet the compactness and efficiency bars. State how you would argue these are genuinely novel
   and not relabelings of known designs.
6. **Evaluation and ablations.** Metrics (thinness ratio, image quality at a stated FOV and eyebox,
   efficiency, ghost energy, diversity of discovered folds). Ablations that test the emergence claim directly:
   remove the compactness term and show folding disappears; remove polarization and show ghost suppression
   disappears; vary the bounce budget; compare RL versus gradient versus QD-hybrid. Include a baseline a
   reviewer will demand: local NS optimization from a hand-seeded pancake in a commercial-style NS optimizer,
   to show the learned method finds folds that seeded local optimization cannot, mirroring the HyperLens
   versus SA+GA argument.
7. **Prior-art positioning.** Place the work against pancake and polarization-folded optics design (patent and
   commercial NS-optimizer practice, which is hand-engineered and locally optimized), differentiable rendering
   (graphics inverse rendering, not lens-topology discovery), differentiable sequential lens design (Tseng,
   Wang, Yang, Teh), and RL for optical design (Fu, Yang freeform, OptiAgent which is sequential). State the
   specific daylight for each. Defend the whitesp…

> **[SOURCE TRUNCATED HERE.]** The prompt was cut off mid-word ("whitesp…" = "whitespace"). No response to
> this NS prompt was captured. Everything above is the specification the researcher wants a research path to
> satisfy. The companion doc [02_state_and_directions.md](02_state_and_directions.md) is the first concrete
> step toward answering it, grounded in the actual code.

---

## 2. The QD-as-data-engine analysis ("Paths A–E") — attached for relevance-mining (verbatim)

> This is the **first** exchange. It concerns a *different* direction: using a QD archive (HyperLens now, the
> NS-QD engine later) as a diversity-structured data engine to train a generative design proposer, then test
> transference (templates: Sim2Reason and OptiAgent). It is included because the researcher wanted to mine it
> for context relevant to the NS-env problem. The transferable parts are: the emergence-thesis constraints,
> the RL-beats-SFT evidence, the descriptor-lossiness argument, and the prior-art map. Reproduced verbatim.

### 2.1 The hard constraints the first prompt imposed (verbatim)
1. **Preserve the emergence thesis.** No training signal, reward, or supervised target may prescribe folding,
   thinness, reflection count, or any specific topology. Supervised fine-tuning on patent-derived or textbook
   topologies is suspect, because it launders the discovery story out of the work and biases toward existing
   forms. Treat heavy design-prior SFT as a likely mistake; a thin format or validity scaffold may be
   acceptable.
2. **RL tends to beat SFT for this task class.** OptiAgent (its training-paradigm table) and Sim2Reason (its
   RL-vs-SFT table) independently show heavy SFT underperforms or actively degrades out-of-distribution
   performance.
3. **Precision stays with the optimizer.** The learned model contributes topology and coarse structure, not
   high-precision numerics, since a 1 percent radius error is catastrophic (OptiAgent).
4. **The Seidel descriptor is a lossy, many-to-one label for geometry.** HyperLens reports a parameter-space
   vs behavior-space distance Pearson correlation of roughly 0.15 to 0.20, and shows epsilon-distinct designs
   that share nearly the same (S2, S3) cell yet differ substantially in geometry and off-axis PSF. Consequences:
   a deterministic regressor from (spec, cell) to design is wrong because the target is multimodal; a
   generative/multimodal proposer is the right object; and any claim that "topology transfers" must be
   evidenced at the geometry level, not the descriptor level.

### 2.2 Prior art the first prompt told the agent to position against (verbatim)
- **Archive distillation and descriptor-conditioned generation:** Faldor et al. 2023 (DCG-MAP-Elites,
  descriptor-conditioned actor); Hegde et al. 2023 (latent diffusion over policy parameters, "Generating
  Behaviorally Diverse Policies"); Mace et al. 2023 (archive into a decision transformer). These compress or
  condition a finished archive; none do inverse optical design or cross-regime transfer.
- **QD data synthesis plus language models:** "Generative Design through Quality-Diversity Data Synthesis and
  Language Models" (GECCO 2024). Closest conceptual neighbor.
- **MAP-Elites to co-evolve LLM training data:** the DejaQ line.
- **LLM-for-lens and learned starting points:** OptiAgent (sequential refractive, Zemax refine); Côté et al.
  (deep nets for starting points, bounded by the training distribution); Yang et al. (curriculum); Teh et al.
  (transdimensional MCMC).
- Whitespace to defend: the optical / inverse-design instantiation, the cross-fidelity or cross-topology
  transfer claim, and the NS discovery loop.

### 2.3 The agent's response — ranked paths (verbatim)

**Framing note.** The existing aggregated archives (13 sixlet runs, 12 octuplet runs) are all at a single
specification: 50 mm, f/4, APS-C. "Amortization across specs" cannot be tested on existing data alone; it
needs a spec sweep, which is cheap (under an hour per run on the CPU workstation per HyperLens Sec. 4.1) but
is new compute. What the existing data can test immediately is the multimodality question and the
constraint-carrying question. Both Path A and Path B exploit that.

#### Path A: Constraint- and spec-conditioned multimodal proposer trained on QD archives
**Thesis.** A conditional generative model trained on aggregated HyperLens archives can (i) carry hard
constraints by conditioning that the QD search itself cannot hold via penalties, and (ii) reproduce
geometry-level design diversity at unseen specifications, faster than rerunning QD.

**Contribution claim.** New instantiation of a known mechanism (QD-as-data-engine for a conditional
generative model), plus two measurable claims not shown anywhere: first, that the conditional target is
multimodal and a deterministic regressor fails at the geometry level, grounded in HyperLens's own evidence
(Pearson r of 0.15 to 0.20 between behavior and parameter distance, Supplementary S4.2; epsilon-distinct
designs sharing nearly the same (S2, S3) cell yet differing in geometry and off-axis PSF, Figs. 22 and 23);
second, that conditioning on a realized constraint value beats penalizing toward it, directly reversing the
Table 19 finding where hard f-number enforcement roughly doubles best RMS (2.1 to 10.2 or 16.0 µm on the
sixlet) and collapses epsilon-distinct sharp designs from 10 to 1. The mechanism is borrowed; the phenomena
are the contribution.

**Novelty positioning.** TileGPT (Gaier et al., GECCO 2024) is the closest neighbor: QD generates the
dataset, a language model is fine-tuned to produce designs from attribute descriptions, and a
constraint-satisfaction stage refines them. Daylight: TileGPT is a discrete tile domain with SFT and no
multimodality analysis, no constraint-vs-penalty comparison, and no held-out-spec transfer evaluation; the
optical version is continuous, physics-verified, and the diversity claim is evidenced at geometry level with
an existing epsilon-distinct metric. DCG-MAP-Elites (Faldor et al. 2023) conditions an actor on descriptors
inside the QD loop for robot control; daylight: it is policy distillation for closed-loop control, not
one-shot inverse design, and no cross-spec generalization. Côté et al. 2019/2021 and LensGAN (SPIE 2024)
generate lens starting points from databases of human designs; daylight: their diversity is bounded by the
training corpus of human designs (HyperLens Sec. 2.2 makes this exact criticism), whereas here the training
corpus is machine-discovered and diversity-structured, and the evaluation is diversity transference rather
than single-design quality. One caution: Parametric-Task MAP-Elites (Anne and Mouret, GECCO 2024) appears to
amortize MAP-Elites solutions across a continuous task family with a distilled model; it may be the closest
mechanism for the cross-spec claim and must be read before writing. If PT-ME already distills multi-task
archives into a generalizing model, Path A's amortization claim becomes "PT-ME instantiated in optics with
multimodal generation," still defensible given the constraint and multimodality results, but the framing must
change.

**Minimal first experiment.** Two stages, both reusing existing assets, no NS tracer.
- *Experiment zero (multimodality test):* train a small conditional generative model p(theta | S2, S3) on the
  13-run aggregated sixlet archive (raw elites plus the refined top-500 pools). Model class: conditional
  diffusion or normalizing flow over the 37-D normalized parameter vector; a mixture density network is an
  acceptable cheaper start. Baselines: (a) a deterministic MLP regressor theta-hat(S2, S3), (b) nearest-elite
  retrieval from the archive. Protocol: sample k designs per query cell, refine all with the existing L-BFGS-B
  pipeline (the refine_unrefined_elites.py machinery exists), then measure: fraction refinable under 100 µm and
  under 30 µm, epsilon-distinct count at eps = 0.20 among refined samples (existing metric), and a memorization
  control: distribution of normalized distance from each sample to its nearest training elite. If samples sit
  within eps of training elites, the model is a lookup table and retrieval wins by definition.
- *Experiment A1 (the constraint wedge):* each archived elite has a realized f-number (the unconstrained pools
  span roughly f/3.4 to f/8). Condition the same model additionally on realized f/#: p(theta | S2, S3, f/#) or
  simply p(theta | f/#). Sample at f/4, refine, and score against Table 19's rows on its own metrics: percent
  in band [3.6, 4.4], median and best RMS of the top-100, n_sharp under 20 µm, and epsilon-distinct sharp count
  at eps = 0.30. The interesting regime is exactly where training data is thin (only 7 percent of the
  unconstrained sixlet top-100 is in band), so success requires the conditional structure to interpolate along
  the constraint axis rather than memorize.
- *Experiment A2 (spec transfer, new compute):* run the existing sequential pipeline at 50 to 100
  specifications varying EFL (say 35 to 85 mm) and FOV, hold out a subset, train p(theta | spec, [cell]), and
  at held-out specs compare proposer-plus-refinement against a fresh full QD run on: epsilon-distinct count
  under threshold, behavior-hull coverage of refined samples, and wall-clock.

**Falsification condition.** If the deterministic regressor matches the generative model on epsilon-distinct
refined output in experiment zero, multimodality does not matter and the central framing collapses. If in A1
the proposer's in-band sharp epsilon-distinct count is no better than the enforced-QD value of 1, conditioning
does not beat penalization. If in A2 the proposer at held-out specs produces fewer than ~a quarter of
fresh-QD epsilon-distinct designs, amortization fails.

**Most likely failure mode.** Memorization with thin generalization: 2048×2048 archives at ~1.1 percent
occupancy give ~46k raw elites per run, dense in a few basins and sparse elsewhere; the model may interpolate
within dense basins (where retrieval already works) and hallucinate unrefinable garbage in sparse regions. The
memorization control and the refinability rate are there to catch this honestly.

**Effort and dependency.** Experiment zero plus A1: ~two to three weeks, no new tracer, no new physics code.
A2: another three to four weeks including the spec-sweep compute. No NS dependency.

**Wider-audience note.** The constraint result (conditioning beats penalization, with a published ablation as
the foil) and the multimodality result travel to amortized inverse design and generative engineering broadly,
beyond optics. Best ratio of reach to cost.

#### Path B: Proposer-in-the-loop, and whether amortization cannibalizes discovery
**Thesis.** Using the Path A proposer to seed CMA-MEGA (replacing the 5,000 uniform seeds) accelerates archive
illumination at new specs, and the open scientific question is whether warm starts suppress the
persistence-driven deep-basin discovery that random seeding enabled.

**Contribution claim.** Mostly a new instantiation of model-assisted QD; the genuinely new element is the
discovery-cannibalization study. HyperLens's best sixlet result came from a 924 µm ancestor stored only
because its cell was empty (Fig. 7), a product of broad, undirected exploration. A proposer seeds near known
basins by construction. Whether warm-started runs still find basins absent from the training distribution is a
real, unanswered question about learned priors in exploration, with an audience well beyond optics.

**Novelty positioning.** SAIL (Gaier et al. 2018, surrogate-assisted illumination) and DCG-MAP-Elites both
put learned models inside the QD loop; daylight is that neither studies cross-task warm-starting nor the
exploration-suppression question. PT-ME again looms; if it warm-starts across tasks, the daylight narrows to
the cannibalization analysis and the optical domain. TileGPT does not close the loop at all.

**Minimal first experiment.** Requires Path A's A2 sweep. At a held-out spec, run CMA-MEGA three ways at
matched 100k evaluations: uniform seeds, proposer seeds, and a 50/50 mix. Metrics: evaluations-to-X
epsilon-distinct designs, final archive QD-score, and, for the cannibalization question, count of refined
elites whose nearest training-set neighbor exceeds eps (genuinely new basins) per condition.

**Falsification condition.** If proposer seeding gives no evaluations-to-coverage gain over uniform seeding,
the loop is dead. Separately, if proposer seeding wins on coverage but the new-basin count drops to near zero,
the result becomes a cautionary one (still publishable, a different paper).

**Most likely failure mode.** A reviewer correctly observes that a single sequential QD run already costs
under an hour, so amortizing it saves little; the honest answer is that the value case is expensive-evaluation
regimes (NS branching at up to 64 rays per input, multi-wavelength, tolerancing), which makes B's full payoff
contingent on the NS engine.

**Effort and dependency.** Two to three weeks on top of A2. The seeding hook is a config change. No NS
dependency for the sequential demonstration; the compelling version eventually wants NS.

**Wider-audience note.** "Do learned priors cannibalize discovery in open-ended search" is a question QD,
open-endedness, and AI-for-discovery people care about independent of lenses. Strongest hook for the broader
community.

#### Path C: Cross-regime warm-starting (element count now, NS later, never topology)
**Thesis.** A proposer trained on cheap-regime archives transfers coarse refractive structure to more
expensive regimes (higher element count; later, the refractive backbone of NS search), without ever claiming
topology transfer.

**Contribution claim.** New instantiation of the Sim2Reason pattern (cheap verifiable data transfers
zero-shot to a harder target) in design space rather than QA space. Be precise about what transfers: power
distribution, glass choices, and gap structure, evidenced at geometry level. The sequential-to-NS topology
version is incoherent: folds are absent from sequential training data, so the proposer cannot propose them,
and per the emergence thesis it must not be taught them. Restrict the NS version to warm-starting the
refractive backbone while reflectivities initialize at zero.

**Novelty positioning.** Sim2Reason is the conceptual template, not a competitor. Côté et al. 2021's dynamic
architecture already handles variable lens structures, so the cross-element-count claim must beat it as a
baseline. OptiAgent's prescription-completion task (Sec. 3.4) is a related knowledge-injection trick but
within one regime.

**Minimal first experiment.** Sixlet-to-octuplet, using existing aggregated archives from both: train the
proposer on sixlet data only (forces a sequence or set representation over surfaces, the one place an
autoregressive model earns its keep), generate octuplet warm starts, run matched-budget QD or direct
refinement, compare against random initialization and against octuplet-trained samples on epsilon-distinct
count and best RMS. The octuplet is the harder space (1.35 percent occupancy, compressed diversity ratios), so
headroom exists.

**Falsification condition.** Sixlet-trained octuplet proposals refine no better than random initializations at
matched refinement budget.

**Most likely failure mode.** The transfer is real but small; a hostile reviewer says "you showed transfer
between two nearly identical regimes." Also a strategic conflict: warm-starting the NS pancake run with any
learned prior weakens the discovery paper's "random seeding, from scratch" commitment. Keep the pancake paper
random-seeded; C's NS version is a separate later study.

**Effort and dependency.** Cross-element-count version: three to four weeks, no tracer. NS version: blocked on
the four-week tracer build plus a successful discovery run.

**Wider-audience note.** Moderate; cross-fidelity amortized design is a recognized theme. The NS version would
be the headline, but it is the furthest away.

#### Path D: Archive-improvement as a verifiable RL reward for a generative designer (the bridge to the collaborator)
**Thesis.** Train a proposer (LLM or policy) with RLVR where the reward is archive improvement (fill an empty
cell or beat the incumbent) rather than per-design fitness, turning OptiAgent's lexicographic reward idea into
an explorer's reward and replacing the collaborator's subjective novelty term with a verifiable one.

**Contribution claim.** Borderline. The reward concept (rank by archive improvement) is CMA-MEGA's own ranking
rule (HyperLens Sec. 3.3) and exists in QD-RL (PPGA, DCG-MAP-Elites). Applying it as an RLVR reward for
one-shot design emission, in the OptiAgent and Sim2Reason sense, is plausibly a new instantiation; the
collaborator's doc explicitly flags its novelty/diversity reward as subjective, and the archive fixes exactly
that.

**Novelty positioning.** OptiAgent: same proposer-plus-optimizer split and the same RL-over-SFT evidence
(Table 4: direct RL at 90.1 percent SR vs SFT at 36.1), but its reward targets a single spec-compliant
design; daylight is the diversity objective. PPGA and DCG-MAP-Elites: same reward concept, different object
(closed-loop policies, not design emission). DejaQ: co-evolves LLM training data with MAP-Elites; mechanism
unsure, needs a read.

**Minimal first experiment.** Do not start with an LLM. Take the Path A generative model, fine-tune it with a
policy-gradient step where reward is computed by inserting samples into a live archive. Compare archive
coverage growth against the frozen Path A model used as a pure sampler. Only if the RL signal clearly helps
does an LLM version (with OptiAgent-style format scaffolding) become worth the cost.

**Falsification condition.** RL fine-tuning against archive improvement does not increase coverage-per-sample
over the frozen generative sampler.

**Most likely failure mode.** Sparse, nonstationary reward (the archive saturates) makes training unstable,
and the result reduces to "we reinvented a QD emitter with extra steps." Blunt version: CMA-MEGA may already be
the better optimizer of this exact objective.

**Effort and dependency.** Four to eight weeks after Path A. No tracer dependency for the sequential version.

**Wider-audience note.** High if it works (verifiable diversity rewards for generative models is an
RLVR-community topic); high risk of being mechanism-thin.

#### Path E: Archive-derived diversity benchmark for generative optical design
**Thesis.** Release the aggregated archives as a benchmark where the score is coverage of the verified design
space, not single-design quality, complementing OptiDesignQA.

**Contribution claim.** A dataset-and-benchmark artifact, not a mechanism. Daylight from OptiDesignQA (711
training, 80 test tasks, single-design success rate and EFFL error as metrics): theirs tests "can you produce
one valid design per spec"; this tests "how much of the known viable design space can you reach," using
epsilon-distinct and hull coverage as the scoring. Sim2Reason Sec. 3.5 (simulator as benchmark, Spearman 0.79
correlation with IPhO) is the template for arguing benchmark validity.

**Minimal first experiment.** Package the sixlet aggregate plus the epsilon-distinct scorer; evaluate two or
three frontier LLMs and the Path A model on coverage at the fixed spec. Two weeks.

**Falsification, failure, effort.** Hard to falsify because it is an artifact; the failure mode is being
ignored. Pursue only as a rider on Path A, not standalone.

### 2.4 Critique pass (verbatim)
- **Path A.** Hostile sentence: "This is TileGPT with lenses; conditioning a generative model on QD data is
  known, and optics already has LensNet and LensGAN for starting points." Answerable: concede the mechanism,
  then point at the three results no prior work has — multimodality necessity (regressor-failure ablation
  grounded in r = 0.15 to 0.20 lossiness), constraint-carrying vs the published Table 19 penalty failure, and
  geometry-level diversity transference at held-out specs. If reviewers demand a new mechanism anyway, target
  SIGGRAPH, Optica, or TEVC rather than NeurIPS.
- **Path B.** Hostile sentence: "Warm-starting QD with a learned model is SAIL plus DCG-MAP-Elites, and a
  single run already takes under an hour." Half answerable: concede the wall-clock objection for sequential;
  the cannibalization study and the expensive-regime argument are the actual contributions.
- **Path C.** Hostile sentence: "The target regime's distinguishing structures are absent from training data,
  so what exactly transferred?" Answerable only for the restricted claims (power distribution, glass, gaps,
  measured at geometry level); not answerable for topology, so never claim it.
- **Path D.** Hostile sentence: "You replaced a working QD emitter with a worse, learned one." Possibly not
  answerable; this is why D runs only after A and only as the cheap policy-gradient version first.
- **Path E.** Hostile sentence: "A single-spec benchmark from a single group's optimizer measures coverage of
  that optimizer's biases, not of the design space." Partially answerable (pyOpTools-style external
  validation, multiple-run aggregation), but a real weakness.

### 2.5 Drop list (verbatim)
- Pure archive distillation without a transfer or constraint claim ("compress the archive into a model"):
  dominated by DCG-MAP-Elites, Hegde et al., and Macé et al.
- SFT-heavy LLM proposer trained on archives to emit prescriptions: dominated by TileGPT mechanically, by
  LensGAN and Côté in optics, and contradicted by both training-paradigm tables (OptiAgent Table 4; Sim2Reason
  Table 3, where SFT costs 3.9 points on IPhO while RL gains 5.4).
- Sequential-to-NS topology transfer as a headline: incoherent (folds absent from sequential training data),
  and it would damage the pancake paper's discovery claim.
- Standalone benchmark paper: keep the artifact, drop it as a path.

### 2.6 Recommendation (verbatim)
Start with Path A, experiment zero and A1, in that order; both run entirely on existing aggregated archives
and existing refinement and epsilon-distinct tooling, and A1 converts a published limitation (Table 19) into
the headline result if it works. Run the A2 spec sweep in the background (embarrassingly parallel config
work). Path B follows naturally from A2 and carries the reach-beyond-optics question (does amortization
cannibalize discovery). Do not touch the NS-dependent versions of anything until the tracer exists and the
pancake run has been attempted with random seeds, because the discovery paper's claim is worth more
unpolluted.

### 2.7 Decision forks (verbatim)
- **(a) LLM vs small conditional generative model.** Small model first, decisively. The design is a
  fixed-length continuous vector (33 to 45 D), the conditioning is a handful of scalars, and the dataset is
  order 10^4 to 10^5 examples; a conditional diffusion model or flow is the natural object. OptiAgent's entire
  reward apparatus exists to fight format and numeric-precision failures that a continuous-output model never
  has. The LLM earns its place only with variable-length cross-element-count generation (Path C), a
  natural-language spec interface (product, not science), or RLVR positioning (Path D). The determinant is the
  representation, not the fashion.
- **(b) Unit of transfer.** Three levels, measured separately, claim made only at the level the evidence
  supports. Behavior level: hull coverage of refined samples in (S2, S3); necessary but weak alone (0.15 to
  0.20 correlation). Geometry level: epsilon-distinct count among refined samples plus the
  distance-to-training-set memorization control; this is the level at which "diversity transfers" must be
  claimed. Function level: off-axis PSF and RMS-vs-field ribbons for a handful of exemplars, as the
  qualitative seal. Any paper claiming transfer from descriptor-space coverage alone should be rejected.
- **(c) Does the proposer beat rerunning QD?** Three metrics, answer differs per metric. Wall-clock per single
  new spec: proposer wins trivially but the absolute saving is small in the sequential regime; concede.
  Evaluations-to-coverage when warm-starting (Path B): the metric where a win is meaningful, because it
  predicts behavior in expensive regimes. OOD specs outside the training hull: where the proposer most
  plausibly loses to fresh QD; reporting that loss honestly is part of the paper. The forward-looking argument
  is that NS evaluation (branching to 64 rays, bounce budgets) shifts the economics toward amortization; say so
  as motivation, not evidence.
- **(d) F-number as motivation.** Yes, the single best wedge available. Table 19 shows the search-time penalty
  mechanism cannot hold the constraint without destroying what the method is for (best RMS roughly doubles,
  epsilon-distinct sharp designs collapse from 10 to 1). Training on realized f-numbers across the
  unconstrained pool, then sampling at the target, moves constraint satisfaction from the search objective into
  the conditioning variable, structurally the right place. Risk: data thinness near the target band (7 percent
  in band); if interpolation along the constraint axis fails, an honest fallback is importance-weighting
  near-band elites during training, which still avoids touching the search objective.

### 2.8 Open questions needing a literature check (verbatim, with search terms)
- Parametric-Task MAP-Elites and successors (biggest threat to the amortization claim): "Parametric-Task
  MAP-Elites Anne Mouret", "multi-task MAP-Elites archive distillation", "MAP-Elites task generalization neural
  network".
- QD-sourced data to train generative samplers with multimodality analysis: "Quality-Diversity Generative
  Sampling Chang Fontaine", "archive-conditioned diffusion model design", "quality diversity dataset generative
  model engineering".
- What LensGAN trains on and evaluates: "LensGAN generative model lens design starting points SPIE 13100-14".
- Conditioning-vs-penalty for constraint satisfaction in generative engineering: "classifier-free guidance
  constraint satisfaction inverse design", "conditional generation versus penalty constrained design
  optimization".
- DejaQ specifics: "DejaQ MAP-Elites LLM training data co-evolution".
- Warm-start cannibalization precedents: "learned initialization exploration suppression evolutionary search",
  "prior-guided quality diversity novelty loss".
- Amortized inverse design baselines in adjacent physics: "amortized inverse design photonics diffusion",
  "OptoLlama masked diffusion multilayer thin films".

### 2.9 Assessment of the collaborator's single-element proposal (verbatim)
It is wedged, as written. Against OptiAgent: the architecture is the same template (LLM proposes coarse
structure, physics-verified reward, downstream optimizer for precision), with the domain swapped to
single-element NS topology in a 2D simulator. A domain swap plus a weaker simulator is not a contribution on
its own. Against the PPO result: the PPO agent already demonstrated topology emergence (TIR) with no
prescriptive signal; the proposal's design-prior SFT stage on patent-derived topologies injects exactly what
the PPO result proved could emerge, which both launders the discovery story and is empirically contraindicated
by OptiAgent Table 4 and Sim2Reason Table 3.

There is a defensible kernel: single-element internal ray-path topology is genuinely the regime OptiAgent does
not touch (OptiAgent is sequential refractive only), and a proposer that operates at the topology level is the
right division of labor. The salvage is concrete: cut the design-prior SFT entirely; keep a thin format
scaffold whose training content is sampled from random valid scenes (syntax only, content distribution
matching the random seeding QD would use); train the design content with RL only; and replace the doc's
self-admitted subjective novelty reward with archive improvement against an NS-QD archive, which makes novelty
verifiable and merges the proposal into Path D as the LLM instantiation. Patent data survives only as a
held-out evaluation set: if the RL-trained proposer rediscovers patented single-element topologies it never
saw, that is the result, and a much better one than imitating them.

---

## 3. The transferable lessons (which of the above actually bears on extending the NS RL env)

These are the load-bearing points from both exchanges for the current goal (using tracked ray-surface
interactions to guide RL toward emergent folding). They are restated compactly here only as a pointer; the
reasoning lives verbatim above.

1. **Emergence thesis is a hard constraint on reward and signal design.** No reward/observation/target may name
   folding, thinness, reflection count, or any topology. (§0, §2.1) This directly governs how interaction-
   sequence data may be used — see [02_state_and_directions.md](02_state_and_directions.md) §D.
2. **For the sparse NS reward regime, the conversation argues QD/hybrid over pure RL** (NS prompt fork 2; §2.6).
   Persistence is what reaches deep basins under a hard landscape.
3. **Interaction-sequence descriptors are the NS analogue of HyperLens's Seidel descriptors** (NS prompt,
   "Behavior descriptors from interaction sequences"): dominant-path bounce count, fraction of energy in the
   dominant sequence, path-length ratio, thinness ratio. These are for characterizing/diversifying, and they
   are the cleanest emergent-diversity index.
4. **The reward must be imaging quality + compactness (thinness ratio) + efficiency-through-valid-paths, with
   explicit stray-light/ghost accounting** so the agent cannot game imaging by ignoring stray energy (NS
   prompt, "reward-design crux").
5. **Polarization is phase 2, not phase 1.** Phase 1 = energy-split partial mirrors + curved reflectors. Phase
   2 = per-ray Jones state + polarization-dependent surface operators; "does polarization-selective routing
   emerge" is a headline if it holds (NS prompt fork 3).
6. **Descriptor lossiness lesson transfers:** any claim that a topology/fold "is" something must be evidenced at
   the geometry/ray-path level, not at a descriptor cell (§2.1.4, §2.7b).
