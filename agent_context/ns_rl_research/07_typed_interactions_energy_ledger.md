# Typed Interactions + Energy Ledger — the Verified Dominant-Energy Fold

> Written 2026-06-12 (Fable 5), implementing the requirements of
> [06_reflectivity_tir_analysis.md](06_reflectivity_tir_analysis.md) (Tier A1 + honest energy
> accounting) and delivering the verified single-element fold demonstrator. Read doc 06 first
> for the physics analysis; this doc records what is now BUILT, what it measures, and the
> design decisions.

---

## 0. Headline result (the verified sequence)

A single uncoated glass element, full Fresnel physics, 21-ray input beam:

```
requested sequence : T@A -> TIR@B -> TIR@A -> T@B          (input to the search, not hardcoded)
input (emitted)    : 21.000000
delivered to sensor: 18.225201   -- 100% of it via the target sequence (no competing class)
reflected away     :  2.692014   (traced Fresnel entry/exit ghosts)
absorbed           :  0.000000
truncated          :  0.082785   (2nd-order ghosts below the brightness threshold)
ledger closure     :  0.00e+00   (exact)
target share of input: 86.8%  ->  DOMINANT-ENERGY SEQUENCE VERDICT: PASS
```

Each event is *typed by the engine at decision time* and carries its energy; e.g. the
strongest ray: `1.000000 → refract@A → 0.931651 → tir@B (loss-free, bIn==bOut bitwise)
→ tir@A → refract@B → 0.868020 → sensor`. Per-ray throughput 0.868 = (1−R_Fresnel)², exactly
the doc-06 prediction for n≈1.71. Reproduce:

```bash
python3 ns_probe/fold_demo.py              # baked-in design, verify + render
python3 ns_probe/fold_demo.py --search     # re-discover from scratch (seeded, deterministic)
python3 ns_probe/test_typed_interactions.py  # 22 physics checks (all pass)
```

Render: `ns_probe/results_single/renders/fold_demo.png` (+ geometry sidecar `.json`).
The discovered element is a **tilted-face wedge** (tilt_B ≈ −20°) — exactly doc 06 §A3's
prediction that prismatic tilt, not curvature, is what lets a whole bundle exceed the
critical angle and then drop below it for the exit.

For contrast, the same demonstrator run on the lossy partial-mirror route
(`--coating 0.5 --sequence "T@A PR@B PR@A T@B"`) converges to the closed-form optimum
(1−r)²r² = **6.25%** of input via the target, with the straight-through leak (25%) dominating
→ verdict FAIL. The two routes of doc 06 §2 are now *measured*, not argued.

---

## 1. What was added to the engine (all flag-gated behind `recordInteractions`)

### 1.1 First-class interaction types
`BaseGlass.refract` now labels its outcome at decision time and the simulator records it:

- `sq1 < 0` branch returns `{interactionType: 'tir'}` (previously returned `undefined`;
  the simulator ignores unknown fields, so default behaviour is unchanged).
- The refraction branch returns `interactionType: 'refract'` and tags the spawned reflected
  child with `__birthType: 'partial_reflect'` and `__birthMech: 'forced' | 'fresnel'` —
  so any use of the artificial coating knob is *auditable* in the data.
- `Simulator.processRays` types every recorded vertex after `onRayIncident` returns:
  `source | refract | tir | partial_reflect | mirror | beam_split | sensor | absorbed | other`
  (mirror/beam-splitter/detector via constructor-type fallback since their handlers don't
  return types). Each vertex now carries `{x, y, i, t, bIn, bOut, mech?}` — energy in,
  energy continuing on that branch.
- Child rays inherit the parent's history with the shared event *re-typed* on the child's
  copy (vertex replaced, not mutated — the histories share vertex objects).

### 1.2 Honest energy ledger
`result.energyLedger = {emitted, delivered, escaped, absorbed, truncated, unfinished}`:

- `emitted` accumulated at source emission; `delivered` = terminal energy of rays that
  crossed a detector; `escaped` = left the scene without one; `absorbed` = blockers/edge
  guards (glass's below-threshold absorption is inside `truncated`, marked by a new
  `absorbedInTruncation` flag so nothing is double-counted); `truncated` = sub-threshold
  ghosts + depth-budget kills + source-side truncation; `unfinished` = rays alive if the
  simulator stops early (ray-count limit).
- Closure `emitted == Σ rest` is **exact** (residual 0.0 in all tests) because under the
  flag the "amp resurrection" of sub-threshold ghosts (a statistical rendering nicety) is
  disabled — measurement mode trades that visual aid for exact books.

### 1.3 Bug fixes found while instrumenting
- **Lost child history:** children spawned by an *absorbed* parent (glass transmitted ray
  below threshold, reflected child alive) lost their `__seq` because the parent was nulled
  before inheritance. Fixed by capturing the parent reference before `onRayIncident`.
- **NaN ray:** the amp-resurrection path divided by zero when R was exactly 0
  (`Math.floor(0.01/0)+1 = Infinity`), occasionally pushing a NaN-brightness ray. Now
  guarded with `(brightness > 0)` in all modes.
- **Recorder accumulation across passes:** `__raySeqRecords` and the ledger are reset at
  the start of every light pass, so CropBox-render + detector runs in one request no
  longer duplicate records.

Sync note (unchanged): after editing `ray-optics/src/core/`, run
`cd ray-optics && npm run build-node && cp dist-node/rayOptics.js integrations/rayOptics.js`.
This was done; the bundle is current.

### 1.4 A correction to docs 05/06 and tir_demo
`Glass` **defaults to `partialReflect: true`** ([Glass.js:43](../../ray-optics/src/core/sceneObjs/glass/Glass.js#L43)).
Every probe/demo scene has therefore always run under full Fresnel physics — there was
never a "perfect AR" mode in play unless explicitly set. `tir_demo.py`'s original argument
("coating-free + partialReflect off ⇒ every bounce must be TIR") was wrong about the flag;
its conclusion survived only because a 0.76-energy ray is impossible as a chain of Fresnel
back-reflections. With typed events the claim no longer rests on inference at all.

---

## 2. Python toolkit

- [ns_probe/sequence_tools.py](../../ns_probe/sequence_tools.py) — parse a target spec
  (`"T@A TIR@B TIR@A T@B"`, faces may be `*`), classify each sensor-arriving record into a
  typed signature, aggregate energy per signature, per-ray energy audits (TIR must be
  bitwise loss-free; no energy jumps in free space), ledger closure, and
  `dominance_report(...)`: PASS iff target-sequence energy > 50% of input AND > every
  competing class.
- [ns_probe/test_typed_interactions.py](../../ns_probe/test_typed_interactions.py) — 22
  assertions: TIR typing + exact loss-freeness; Fresnel R = ((n−1)/(n+1))² at the event,
  the child, and the through-beam; forced-coating tagging (`mech='forced'`) and splits;
  ledger closure on multi-bounce scenes; depth-budget booking; flag-off leaves zero
  instrumentation and identical detector readings.
- [ns_probe/fold_demo.py](../../ns_probe/fold_demo.py) — the demonstrator. Target sequence
  is an **input**; geometry is a 9-DOF single element (position, thickness, two sags, two
  **face tilts**, index, beam y, beam angle); search = seeded random stage + CMA-ES polish
  (reusing [cma_me.py](../../ns_probe/cma_me.py)) on the objective *fraction of input
  energy delivered via exactly the target sequence*. Renders never overwrite (numbered
  suffixes) and every render gets a geometry sidecar JSON.

---

## 3. Design decisions and route costs

1. **Extend the JS engine, don't start the from-scratch tracer for this.** The deliverable
   (typed, energy-audited sequence verification) needed ~150 lines in two engine files plus
   Python tooling. The engine's physics was already right; what was missing was *retention
   and labeling* of decisions it already makes. Cost: one day; default path untouched.
2. **Type at decision time, not by inference.** The refract method knows TIR vs refract vs
   partial-reflect; surfacing it kills the entire class of "inferred from a coating-free
   trick" arguments (which, per §1.4, was already broken).
3. **Ledger semantics: classify at termination.** Energy is booked exactly once, when a ray
   dies (escape/absorb/truncate), split by whether it ever crossed a sensor. Splitting
   conserves by construction (R + (1−R) = 1; TIR untouched), so closure is exact rather
   than approximate.
4. **TIR route vs partial-mirror route.** The high-energy fold is the TIR route: no coating,
   no polarization, no new physics, 86.8% of input measured through the target sequence.
   The partial-mirror route stays available as the now-*typed* forced-reflectivity knob
   (`mech='forced'`), but it is angle/wavelength-independent and capped at (1−r)²r² ≤ 6.25%
   for the uncompensated double bounce — measured by the same demonstrator. A *physical*
   coating model (R(θ, λ, pol) from a layer stack) and Jones polarization remain Tier B
   (doc 06): build them only if/when the lossy pancake family with QWP ghost suppression
   becomes the experiment. They are not needed for any TIR-based science.
5. **Tilt DOFs were the unlock.** With only on-axis curvature the bundle cannot share one
   TIR sequence (doc 06 §2's argument); two face-tilt parameters made the whole-bundle fold
   findable by a 600-sample random stage. Any future NS probe should include tilt/decenter
   in the genotype from the start.

## 4. When is the differentiable NS tracer worth building?

Recommendation unchanged from doc 03 §2.3 but sharpened by this work:

- **Not for verification or discovery-of-mechanism** — this work shows the instrumented JS
  engine + seeded gradient-free search finds and *proves* a requested interaction sequence
  in minutes. Typed sequences and the ledger also transfer directly as descriptors/terms
  for the QD probes (docs 02 §E.2–E.4).
- **Yes, when the program needs persistence-then-improvement at imaging quality** — i.e.
  polishing folded candidates with gradients (the strongest confound in the phase-0
  negative, doc 05). Then **extend DeepLens to NS** (argmin routing + gather + branch;
  straight-through at basin walls) rather than writing a tracer from scratch: the
  differentiable per-surface math (`ray_reaction`) already exists, and within a fixed
  interaction sequence the trace is smooth — *conditioned on the typed sequence this
  engine can now report*, which is exactly the data needed to seed and validate such a
  tracer. Gate the build on doc 05's "refinement experiment" showing folded designs can be
  polished into competitiveness on the cheap engine first.

---

## 5. Interface quick-reference

Scene flags (top-level JSON): `recordInteractions: true`, `maxRayDepth: N`.
Result fields: `raySequences` (one record per detector crossing:
`{x, y, dirx, diry, b, seq:[{x,y,i,t,bIn,bOut,mech?}]}`), `energyLedger` (§1.2).
Per-object: `reflectivity` / `reflectivityFront`+`reflectivityBack`+`elemCx` (forced
coating, typed `mech:'forced'`), `partialReflect` (Fresnel; **default true for Glass**).
