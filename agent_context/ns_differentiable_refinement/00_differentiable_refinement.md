# Differentiable refinement of QD-discovered non-sequential designs — the discussion

> **What this is.** A captured design discussion (2026-06-19, Opus 4.8) about whether/how to
> add a *differentiable* refinement stage downstream of the gradient-free QD discovery in
> `ns_probe/`. Each bullet heading below is the user's **exact words**; underneath is the
> explanation + position we reached, now grounded by a real gate experiment.
>
> **The gate experiment that backs this up:** `ns_probe/refine_lm.py` — finite-difference
> Jacobian + Levenberg–Marquardt (damped least squares) local refinement, run *through the
> existing JS engine* (no differentiable tracer). Results + figures:
> `ns_probe/results_refine_lm/` (see `FINDINGS.md` there). Before/after ray renders:
> `ns_probe/results_refine_lm/compare_cell*.png`.
>
> **One-line conclusion.** Yes, add a refinement stage; do it AFTER QD, on a frozen topology,
> with a densified bundle and physics guards — but you do **not** need a differentiable tracer
> (nor DeepLens/dO) yet: FD+LM on the existing engine already gives 70–80% spot improvement.

---

## • "are there any ways to make this a more rigorous 'differentiable' framework? ... so as to make these found solutions [more] rigorous and of any meaningful use?"

Yes — and the empirical answer is now in hand. We took QD champions and ran damped least
squares on a **densified** ray bundle, minimizing the energy-weighted spot. Clean-topology
champions improved **73–80%** in RMS spot (e.g. cell 3,5: 2.42 → 0.67; cell 4,11: 0.66 → 0.13)
while **preserving throughput, the TIR critical-angle margin, and the interaction topology**.

So the QD designs were *not* locally optimal — QD finds the right *basin/topology* but leaves
the *geometry* coarse. A refinement stage is exactly what makes them "rigorous and of
meaningful use": it converts a coarse archive entry into a polished, sensor-focused design
with a real, dense spot metric instead of QD's coarse 21-ray throughput proxy.

Important subtlety on the word **"differentiable"**: we did this *without* analytic gradients.
Reverse-mode autodiff is one way to get the Jacobian `∂(sensor hit)/∂(params)`; **finite
differences are another**, and at the current 7-DOF scale FD is fast (~10 s/champion) and
sufficient. "Differentiable framework" should be read as "gradient-style local refinement,"
not "must build an autodiff tracer." The latter is an optimization of the former, justified
only when DOF count explodes (see the migration bullet).

## • "I'd imagine we'd need a p significant amount of 'rays' that end up at the sensor that's 'somewhat' roughly focused"

Correct, and this is the **precondition that makes refinement well-posed** — confirmed two ways:

1. **The champions already satisfy it.** They deliver 90%+ of input energy to the sensor,
   roughly focused (spot 0.66–3.7). There is real structure to refine; we're polishing a
   working imager, not trying to coax focus out of noise.
2. **"Roughly focused" is literally the gate's tracking mechanism.** We freeze the topology by
   keeping only the rays that reach the sensor via the champion's *dominant* interaction
   signature. When that set is large (high dominant-share), refinement works great; when it is
   small (the messy cell 2,8 champion, only 46% of energy on the dominant path), refinement
   barely moves (−3%) because the blur lives in the energy that *isn't* roughly focused. So
   "a significant amount of rays, roughly focused" isn't just a nice-to-have — **dominant-share
   is a quantitative predictor of whether a design is refinable at all.**

## • "We can make this beam however dense to make the spots more rigorous, but it gives us effective control for some ray bundle/fov"

Exactly right, and it's a genuine value-add of the second stage that we exploited directly.
QD traces *thousands* of candidates, so it must keep the bundle coarse (21 rays). Refinement
traces only a *handful* of designs, so we can afford **10×+ rays** (we used 121) and define a
much sharper, denser objective. The two stages differ not just in gradient/no-gradient but in
**resolution**: QD is coarse-and-broad (illuminate the topology archive); refine is
dense-and-local (sharpen one design's spot for a chosen bundle/field). The frozen input beam
(`ray_y`, `ang`) *is* the FoV/field you take "effective control" of — we refine the lens to
focus that specific field, which is why those two DOF are held fixed by default.

## • "whatever differentiable ray tracing framework we have, is in deeplens, more specifically dO ... the issue with this is that it is a) sequential and b) cylindrically symmetric, meaning all the components are on an axis (the optical axis)"

This is the decisive constraint, and it rules DeepLens/dO **out** for this job — not as a
limitation to work around, but as a fundamental data-model mismatch:

- **Sequential.** dO traces surfaces in a *fixed order*, transmission-style. Our designs
  revisit surfaces (`T@A → TIR@D → TIR@B → TIR@B → T@B`): surface B is hit twice, the order is
  data-dependent. dO has no representation for "ray returns to a surface it already used."
- **Cylindrically symmetric.** dO surfaces are sag `z(r)` about the optical axis. Our
  sharp+efficient folds **require broken symmetry (prismatic tilt)** — that's the headline 2×2
  trade in the QD work. A tilted, off-axis element is not expressible in dO's surface model.

The liberating consequence: **because the whole system is 2-D meridional, none of DeepLens's
value (3-D asphere library, rotationally-symmetric image pipeline, sensor model) transfers.**
A differentiable 2-D NS tracer is a few hundred lines (ray–conic intersection + Snell +
Fresnel + the critical-angle test), and reimplementing it in JAX/PyTorch is *less* work than
bending dO — with the bonus that the existing JS engine becomes a forward-truth oracle to
validate the twin design-by-design. **Don't migrate into dO; build a small twin if/when you
need analytic gradients.**

## • "I mentioned a differentiable network for something we use AFTER the designs have been found and after the beams have somewhat 'significant' convergence ... because there's some idea of order in this case, just for those rays concerned"

This instinct is the single most important architectural insight, and the gate confirms it is
**principled, not just convenient.** The reason is *where the discontinuities live*:

- A non-sequential objective is **piecewise-smooth with jumps**. The jumps come from discrete
  decisions: TIR-vs-refract at the critical angle, and which-surface-is-hit-next (visibility).
- **Within a fixed topology** (branch decisions and surface-visit order frozen), the objective
  is perfectly smooth and differentiable. The discontinuities live *between* topologies.
- The QD archive **bins designs by interaction topology** — i.e. it is literally the
  discretization of the discontinuous axis.

So the "idea of order ... just for those rays concerned" is exactly a *fixed topology cell*.
Doing differentiable refinement only AFTER discovery means: **QD owns the
discontinuous/combinatorial structure (which topology), refinement owns the continuous geometry
within a cell.** Empirically, every guarded refinement *stayed in its archive cell* (dominant
signature unchanged, zero ray churn). You were right to want this stage downstream and gated on
"significant convergence" — that convergence is what guarantees a fixed, smooth topology to
descend on.

## • "the gradients supplied by the second set of interactions would fight with the gradients supplied by the first interaction ... what sort of instabilities this would arise ... I am not sure how you'd handle this"

This was the deepest worry, and the gate lets us answer it precisely. **It is a real
phenomenon, but the wrong consequence.**

- **Why it's not instability.** Surface A being hit twice (transmit early, TIR-revisit late)
  contributes two terms to `∂loss/∂(A's params)` that *sum* — exactly like every sequential
  lens, where one surface gets gradients from the marginal ray, chief ray, and every field, all
  "fighting" and summing. Within a fixed topology the gradient is finite and well-defined; the
  revisit just creates a longer chain-rule path. The honest residue of the worry is
  **ill-conditioning, not instability**: the revisit makes the Jacobian's sensitivity coupling
  cyclic, so contributions can near-cancel.
- **We measured exactly this.** Per-parameter sign-cancellation `1 − |Σ Jr| / Σ|Jr|` hit
  **~1.0 for `cx`, `bsag`, `thick`** (their rays pull almost perfectly opposite — the literal
  "fighting"), and the Jacobian condition number κ(J) ran **1e8–1e11** with a singular spectrum
  spanning ~7 orders of magnitude. So your intuition about cancellation is *correct and visible
  in the data*.
- **How you handle it.** Not vanilla gradient descent — **Levenberg–Marquardt / damped least
  squares**, the exact method Code V and Zemax have used for 60 years precisely because
  ray-aberration Jacobians are ill-conditioned. The damping term `(JᵀJ + λI)` regularizes the
  near-null directions; LM **descended cleanly on every champion** despite the 1e11 condition
  number. The "fighting gradients" are a conditioning problem the damping absorbs, full stop.

## • "Defining these gradients for non cylindrically symmetric frameworks, as opposed to deeplens ... it'll be interesting to see how this would be constructed"

Two layers to the answer:

1. **You don't need to construct analytic gradients to start.** The gate gets the full residual
   Jacobian by **central finite differences** (2·n_opt forward traces per step) through the
   existing tilted/asymmetric JS engine — symmetry is irrelevant to FD. That's how we got every
   number above without writing a single derivative.
2. **When you do want analytic gradients** (for speed at high DOF), a non-symmetric 2-D NS
   tracer is straightforward because nothing depends on rotational symmetry: each interaction is
   a *local* operation — ray–line/ray–conic intersection (closed form), Snell at the actual
   surface normal, Fresnel split, critical-angle test — all differentiable and all indifferent
   to whether the element sits on an axis. The cylindrical symmetry in dO is a *modeling
   convenience* (parameterize surfaces by `z(r)`), not a requirement for differentiability. A
   freeform/tilted 2-D twin in JAX gets gradients by autodiff over those same local ops; the
   "construction" is just: write the forward trace as differentiable arithmetic over an
   **unrolled, fixed (champion-recorded) ray graph**, with no `if`-branches inside the autodiff
   graph (the topology is frozen, so the branches are already decided).

## • "would it be better to have the first part ... QD based exploration ... in the current node js physics env then the setup info is used and migrated to a newer one where this setup can be constructed in this new framework based in deeplens/dO and 'refined'?"

**Yes to the two-phase split; no to dO as the second framework.** Concretely:

- **Phase 1 (discover): keep it in the JS NS engine.** Gradient-free QD is the right tool for
  the discontinuous/combinatorial axis — you can't descend across a topology wall.
- **Hand-off:** the "setup info" you migrate is the champion genotype **plus the recorded ray
  graph** — and you already export it as `result["raySequences"]` (typed, ordered
  `(surface, type, energy)` per ray). The topology hand-off is essentially already built.
- **Phase 2 (refine): NOT dO.** dO can't represent NS+tilted designs (previous two bullets).
  Today, refine in-place with **FD+LM on the JS engine** — it works (70–80% improvement) and
  needs no migration at all. Build a JAX/PyTorch 2-D NS *twin* only when DOF count explodes
  (multi-element / freeform / many fields), validated forward-pass against the JS oracle.
- **Mandatory guardrails (learned from the gate):** refinement must be **regularized**, or it
  *cheats* — unguarded LM drove TIR events to the critical angle (margin → 0.04°) to sharpen the
  tracked sub-bundle while the honest 21-ray spot *didn't improve at all*. The fixes: a
  **no-erosion critical-angle margin floor** (stay at least as robust as the champion), a
  **throughput-floor penalty** (don't sharpen by dumping energy), and an **independent
  apples-to-apples metric** (re-score at QD's 21-ray density) to detect the cheat.

---

## Artifacts
- `ns_probe/refine_lm.py` — the gate (FD-Jacobian + LM, topology-frozen, guarded).
- `ns_probe/render_refine_comparison.py` — before/after ray-diagram renderer.
- `ns_probe/results_refine_lm/FINDINGS.md` — the 4-champion results table + 5 findings.
- `ns_probe/results_refine_lm/refine_cell*.json|png` — per-run reports + diagnostics figures.
- `ns_probe/results_refine_lm/compare_cell*.png` — before/after setup renders.
- Upstream context: `agent_context/ns_rl_research/00_MASTER_CONTEXT.md` (§10.6 = "gate this first").

## Open next steps (not yet done)
1. **Archive-wide refinability map** — run the gate over every cell; record headroom +
   dominant-share per cell. Turns "it refines" into "here's which topologies refine."
2. **Dominant-share as a QD descriptor/objective** — so the archive fills with
   refinement-ready (high-share) designs.
3. **The JAX 2-D NS twin** — only when scaling to multi-element/freeform makes FD's
   2·n_opt traces/step the bottleneck; validate forward-pass against the JS oracle first.
