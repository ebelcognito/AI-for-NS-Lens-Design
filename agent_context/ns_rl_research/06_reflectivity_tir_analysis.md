# Reflectivity, TIR, and What a Realistic High-Energy Fold Requires

> Analysis written 2026-06-12 (Opus 4.8) in response to a sharp question:
> *"Can we realistically have a lens with the interaction sequence transmission → TIR → TIR → transmission,
> where the **high-energy** part of the rays does this? Until now all the folded designs we found were weak
> back-reflections being focused. What would it take — polarization? a more complex reflectivity model?"*
>
> This doc is the analysis + an implementation-requirements list to hand to Claude Fable 5. It supersedes the
> probe's reflectivity assumptions (see [05_probe_phase0_findings.md](05_probe_phase0_findings.md)); read that
> for the probe context. Code refs are to the instrumented engine ([ray-optics/](../../ray-optics/)).
>
> **UPDATE 2026-06-12: Tier A1 + energy accounting are IMPLEMENTED and the dominant-energy
> T→TIR→TIR→T fold is demonstrated and verified (86.8% of an input beam, exact energy ledger).
> See [07_typed_interactions_energy_ledger.md](07_typed_interactions_energy_ledger.md). One
> correction to §1's table: `Glass` defaults `partialReflect: true`, so probe scenes always ran
> with full Fresnel unless explicitly disabled.**

---

## 0. The short answer

- **Yes, that fold is realistic and is the *high-efficiency* way to fold an imaging path.** Transmission →
  TIR → TIR → transmission is exactly how a periscope / prism fold works (and how smartphone periscope camera
  modules fold a long focal length into a thin body). TIR is **loss-free** (100% reflection, both
  polarizations), so the **dominant energy** travels the fold — unlike the lossy beamsplitter route.
- **The engine already does real, loss-free TIR.** No new reflection physics is needed for it.
- **It does NOT require polarization.** Polarization only matters for the *other* thin-fold family — the
  lossy beamsplitter "pancake," whose ghost suppression uses a quarter-wave plate + reflective polarizer.
- **The reason the probe never found it is a flaw I introduced:** the **forced-reflectivity DOF** gave the
  optimizer a cheap, artificial "coating" to make folds by **splitting energy** (lossy back-reflections), so it
  never had to discover the harder, loss-free TIR geometry. The fix is to **remove that crutch**, **open the
  geometry to admit TIR** (tilted/prismatic surfaces), **measure interaction *types*** (which the probe
  currently does not), and **reward throughput** so the *high-energy* path is what gets optimized.

So the realistic move is mostly to make the model **simpler and more physical**, not more complex. The genuinely
complex additions (Jones polarization, multilayer coatings) are a separate, deferrable track for the
*beamsplitter* family.

---

## 1. How ray-surface interactions are handled today (grounded in code)

Every glass-surface event runs through `BaseGlass.refract`
([ray-optics/src/core/sceneObjs/BaseGlass.js](../../ray-optics/src/core/sceneObjs/BaseGlass.js)). The decision
tree:

```
cos1 = -(normal · ray_dir)                       # cosine of incidence angle
sq1  = 1 - n1^2 * (1 - cos1^2)   (= 1 - n1^2 sin^2 θ)   [line ~197]
                                  # n1 = index ratio; for inside→outside, n1 = n_glass > 1

if sq1 < 0:                       # angle beyond the critical angle  [line ~200]
    ── TOTAL INTERNAL REFLECTION ──
    ray reflects; brightness_s / brightness_p are LEFT UNTOUCHED  →  100% energy, loss-free
                                  # geometry-driven; no coating, no polarization dependence

else:                             # refraction branch  [line ~207]
    choose (R_s, R_p):
       if reflectivity[Front/Back] set  →  R_s = R_p = r        # MY FORCED COATING (artificial)
       elif partialReflect              →  Fresnel R_s, R_p     # physically correct, uncoated
       else                             →  R_s = R_p = 0        # perfect AR (no reflection)
    spawn reflected child  : brightness * R          [lines ~240-261]
    continue transmitted   : brightness * (1 - R)    [lines ~274-277]
```

Established facts (verified):

| Aspect | Current behaviour |
|---|---|
| **TIR** | `sq1 < 0` branch. Reflects **100%** of energy, both polarizations, **loss-free**, purely from geometry (Snell + critical angle). Already correct and physical. **My forced reflectivity does NOT touch this branch.** |
| **Partial reflection (refraction branch)** | Energy is **split**: reflected child gets `R`, transmitted gets `1−R`. `R` is either **Fresnel** (`R_s≠R_p`, physical, ~4% near normal up to 100% near grazing) or my **forced coating** `R∈[0,1]`. |
| **Forced reflectivity (my NS-probe addition)** | An **artificial, angle/wavelength-independent coating** `R_s=R_p=r`. It is the *only* way the probe could make a strong reflector in the refraction regime. It is **not a real optical element** — real surfaces reflect by Fresnel (uncoated) or by a designed thin-film stack. |
| **Polarization** | `brightness_s`, `brightness_p` are two **intensity scalars** (each `0.5·brightness`, i.e. unpolarized in). Fresnel splits them differently. **No phase, no Jones vector, no waveplates/polarizers, no circular polarization.** Cannot represent polarization rotation or interference. |
| **Absorption** | **None** (no Beer–Lambert bulk absorption; the only "absorb" cases are degenerate edge-hit guards). Energy is conserved except for (a) reflected ghosts that miss the sensor and (b) truncation of sub-threshold dim rays. |

**Key consequence:** the model already has the loss-free mechanism (TIR) the question asks for. What it *also*
has — and shouldn't, for a faithful emergence test — is a cheap artificial coating that lets the optimizer fold
by *throwing energy away*.

---

## 2. Two families of thin/folded optics — and which one the probe drifted into

| | **TIR / prism fold** (what you described) | **Beamsplitter "pancake"** (what the probe found) |
|---|---|---|
| Mechanism | Transmit in → **TIR** off internal surface(s) → transmit out. Geometry past the critical angle. | Partial mirror **splits** energy; the folded component is a fraction; rest is lost/ghost. |
| Energy in dominant path | **High** (~92%: only two ~4% Fresnel entry/exit losses; TIRs are loss-free) | **Low** (geometric max ~25%; the probe's were far worse, ~8–19%) |
| Needs polarization? | **No** | For *efficiency/ghost-suppression*, **yes** (QWP + reflective polarizer, ~25%→70–90%) |
| Needs coatings? | No (TIR is intrinsic) | Yes for a real 50/50 or polarization beamsplitter |
| Geometry | **Tilted / prismatic / decentered** surfaces (to exceed critical angle on the dominant bundle) | Can be on-axis rotationally symmetric (curved partial mirror + mirror) |
| Real-world example | Smartphone **periscope** camera; Porro/roof prisms; light guides; RXI concentrators | VR **pancake** lenses |

The probe gave the optimizer the **forced-reflectivity coating** and an **on-axis rotationally-symmetric**
element. Both choices steer *away* from TIR and *toward* the lossy beamsplitter:

1. **The coating made lossy folding cheap.** Reflection was available for free at any surface via `r`, so the
   optimizer never needed to discover the precise geometry that produces TIR. It took the easy (lossy) path.
2. **On-axis rotational symmetry forbids dominant-bundle TIR.** The high-energy rays are paraxial — they hit
   the optical surfaces **near-normal** (small θ), which is the *opposite* of the steep angle TIR needs.
   Only steep *marginal* rays can approach the critical angle. So in this geometry, any TIR is a minority,
   off-axis effect — never the dominant path. **TIR folds of the dominant bundle require tilted/prismatic
   surfaces** (a 45°-ish face is the canonical periscope element).

That is precisely why "all the designs were back-reflections being focused": the optimizer was *built* to make
lossy splitters in a geometry that can't TIR the main beam.

---

## 3. Is the forced-reflectivity DOF even faithful to the emergence thesis?

**Arguably not, and this is the deeper issue.** The original existence proof had TIR emerge with **no reflection
term anywhere** — purely from geometry satisfying an imaging objective. By adding a continuous `reflectivity`
knob, I handed the optimizer a *prescribed* "put a mirror here" mechanism. That:
- lets it fold the lossy way (defeating the point), and
- is itself a soft prescription of the folding mechanism (a coating), which the thesis says to avoid.

The faithful generalization of the TIR result is: **give the agent only geometry (shapes, positions, tilts,
index), let reflection happen *only* as physics dictates (Fresnel + TIR), and demand efficiency.** Then a
loss-free fold, if it emerges, is a genuine discovery — the multi-surface analogue of the single-element TIR
result. The forced coating should be removed, or at minimum made to *cost* energy so it cannot win against TIR.

---

## 4. What it would take — implementation requirements (for Fable)

Grouped by necessity. The headline: **Tier A needs no polarization and no more-complex reflection model — it
needs the model made simpler/physical, the geometry opened, and the interaction *types* measured.**

### Tier A — get the loss-free TIR fold to emerge (do this first)

**A1. Instrument interaction *type* (small engine change; currently missing).**
The probe records ray *vertices* but not *what happened* at each. It therefore **cannot distinguish a TIR fold
from a lossy back-reflection** — the exact thing the question is about. `refract` already knows the type at
decision time (`sq1<0` ⇒ TIR; a spawned `ray2` ⇒ partial reflection; otherwise pure refraction). Surface a
per-event tag `{type ∈ refract|tir|partial_reflect|mirror|sensor, energy}` into `raySequences` (extend the
recorder in [Simulator.js](../../ray-optics/src/core/Simulator.js) `processRays` + return the type from
`refract`/mirror handlers). Then "does the **high-energy** path do T-TIR-TIR-T" becomes directly measurable:
the **energy-weighted dominant interaction sequence**.

**A2. Remove the lossy-coating crutch (or make it cost energy).**
Faithful option: drop forced `reflectivity`/`reflectivityFront/Back` from the design space — surfaces are
**uncoated** (Fresnel + TIR only). Then the *only* way to fold strongly is TIR (or near-grazing Fresnel).
Softer option: keep a reflectivity DOF but add an **efficiency/throughput term to fitness** (energy delivered
to the sensor via valid sequences ÷ input energy), so lossy splitter-folds are dominated by loss-free TIR
folds. Either way, the objective must **reward the high-energy path** — this is the "honest efficiency" term
from the reward crux ([01](01_extracted_research_context.md) §1), and it is what makes "the *high-energy* part
does the fold" an explicit pressure rather than an accident.

**A3. Open the geometry to admit TIR of the dominant bundle.**
This is the crux and the biggest change. The current arc element is on-axis and rotationally symmetric, so
paraxial (dominant) rays hit near-normal and cannot TIR. Add **surface tilt / decenter**, or a **prism/wedge
element** (flat faces at chosen angles), and allow **thick, high-index** bodies (more room and steeper internal
angles; higher index ⇒ smaller critical angle ⇒ TIR easier — critical angle is 41.8° at n=1.5, 33.7° at n=1.8).
A 2D meridional cross-section is fine; the JS tracer already supports arbitrary tilted lines/arcs, so **no
engine limit** here — only the probe's scene parameterization needs to expose tilt/decenter (and likely a
periscope-style off-axis input or sensor placement, since a straight-through on-axis path resists folding).

**A4. Interaction-type-aware descriptors.**
Add descriptors over the energy-weighted dominant sequence: **# of TIR events on the dominant path**, **fraction
of delivered energy that underwent ≥1 TIR**, **fraction lost to splitting**. Keep these *descriptive* (archive
axes / diagnostics), not fitness targets, per the thesis. This lets the archive separate "true TIR fold" cells
from "lossy ghost" cells — the distinction the current probe is blind to.

### NOT required for Tier A

**Polarization.** A TIR fold is 100% for both s and p above the critical angle, so polarization adds nothing to
the mechanism the question describes. Do **not** implement Jones calculus to get T-TIR-TIR-T.

### Tier B — only for the *beamsplitter* family (defer; separate track)

**B1. Jones-vector polarization** (complex amplitude + phase per ray; QWP, linear/reflective polarizer surface
types). This is the **only** way to model the real pancake's ghost suppression and efficiency recovery
(~25%→70–90%). It is a substantial engine rewrite (the current intensity-only `brightness_s/p` cannot carry
phase). Needed *only* if pursuing the lossy beamsplitter pancake — which is the less efficient family and not
what the question is about.

**B2. Realistic multilayer thin-film coatings** (`R(θ, λ, polarization)` from a layer stack). Replaces the
crude forced `r` with a *physical* designed reflector — relevant if you want engineered partial mirrors or
AR/HR coatings with honest angle/wavelength behaviour. Tier B; not needed for TIR folds.

---

## 5. What does **not** need to change

- **The engine's TIR core** — already correct and loss-free ([BaseGlass.js:200-206](../../ray-optics/src/core/sceneObjs/BaseGlass.js#L200)).
- **The Snell/refraction core, the persistent bridge, the CMA-ME solver** — all reusable as-is.
- The interaction-*sequence* recording exists; it just needs **type tags** added (A1).

---

## 6. Recommended path (the decision for Fable)

Pursue the **TIR/prism fold first** — it is the high-efficiency, polarization-free, coating-free fold the
question targets, and it is the faithful multi-surface generalization of the original single-element TIR result:

1. **A1** add interaction-type tags (so the T-TIR-TIR-T sequence is measurable, energy-weighted).
2. **A2** remove the forced-reflectivity crutch and add a throughput/efficiency objective (demand the *dominant*
   energy take the fold).
3. **A3** open the geometry to tilt/decenter/prism + thick high-index (so the dominant bundle *can* exceed the
   critical angle), likely with a periscope-style off-axis layout.
4. **A4** add TIR-aware descriptors and re-run the QD probe; inspect whether a **loss-free, dominant-energy TIR
   fold** emerges and images well — and at what efficiency vs the transmission baseline.

Defer **all polarization and coating work (Tier B)** to a later, separate experiment on the lossy beamsplitter
family, where it is actually required.

**One-line framing for the prompt to Fable:** *the realistic high-energy fold is TIR, the engine already does
TIR losslessly, and the blockers are (i) an artificial reflectivity coating I added that lets folding cheat
lossily, (ii) an on-axis geometry that can't TIR the dominant bundle, and (iii) the probe not recording
interaction types — fix those three (no polarization needed) before considering the lossy pancake route.*
