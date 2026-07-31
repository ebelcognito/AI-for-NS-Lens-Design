# How the geometry is built, and why

Background for anyone modifying the `.seq` or porting another design. If you only need to
*run* the file, `AGENT_RUNBOOK.md` is enough.

---

## 1. A `.seq` is a program, not a data file

It is CODE V's macro language — the same commands you would type into the Command Window.
Running it **builds** a lens database imperatively. Writing one is really writing the exact
keystroke sequence a designer would use.

```
LEN NEW                         ! wipe and start a new lens
DIM M / WL / REF / WTW          ! units, wavelengths, reference wavelength, weights
XAN / YAN / EPD                 ! fields and pupil
PRV / PWL / 'NAME' n1 n2 n3 / END   ! private catalog of model glasses
SO  0  1e18                     ! object surface (infinity)
S <rdy> <thi> [glass] [REFL|TIRO glass]   ! append a surface -- repeat
<modifiers addressed by index: ADE S3, YDE S3, DAR S3, CIR S3, PIK, STO>
SI  0 0                         ! image surface
PIM                             ! paraxial image solve
GO                              ! execute
```

Two phases matter: `S` lines build the list **positionally**, then modifiers address
surfaces by the index they landed at. Insert or drop an `S` line and every subsequent
`ADE Sn` silently retargets.

---

## 2. The coordinate system — the whole difficulty

CODE V is an **axis-following sequential code**. There is a running local frame, and it is
the source of every hard problem in this project.

**The default chain.** Each surface has a local frame: origin at the vertex, local **+Z**
along the axis of propagation, **+Y** meridional, **+X** out of the page. After surface *k*
you translate along local +Z by `THI(k)` to reach surface *k+1*'s origin. No rotation. A
centred lens is therefore a pure 1-D chain of numbers.

**Perturbations.** `XDE/YDE/ZDE` shift a surface's vertex and `ADE/BDE/CDE` rotate it —
both relative to the *inherited* frame. The mode decides what happens afterwards:

| mode | effect | use |
|---|---|---|
| basic decenter | perturbation **propagates** to everything downstream | folding an axis; the relay chain |
| `DAR` | applies to this surface only, then **undoes itself** | placing surfaces absolutely |
| `GLB` | place by **global coordinates** relative to a reference surface | the native answer — syntax unresolved on this build |

**Authoring in global coordinates.** The prescription gives absolute vertices, so the frame
must not accumulate. `DAR` + `THI 0` achieves that: each decenter is undone, the base frame
never moves, and every `YDE/ZDE/ADE` is measured from the same origin. That is the
construction in `benchmark_tierA_v2c.seq`.

**The asymmetry.** The prism faces use `DAR`; the first relay surface does **not**, because
the relay needs its tilt to propagate down the following surfaces. Getting this backwards
either scatters the prism or detaches the relay from the exit axis.

**Reflected space.** After a reflection CODE V works in reflected coordinates and thickness
signs flip, once per reflection. This is the trap that scattered the earlier fold export.
The present file avoids it completely: absolute placement with `THI 0` means no
reflected-space sign is ever computed, and the relay sits downstream of an *even* number of
reflections, so its thicknesses are positive.

**For the 2-D source designs** (Objective B) the mapping is:

```
CODE V Z = engine x   (propagation)      CODE V X = out of plane
CODE V Y = engine y   (meridional)
```

Because those designs are strictly meridional, a cylinder along X and a sphere of the same
radius trace **identically**, so the exporters emit `RDY` only.

---

## 3. Surface instantiation

A CODE V surface bundles four independent things:

| component | set by | in this design |
|---|---|---|
| local frame | `ADE`/`YDE`/`ZDE` (+ `DAR`) | face tilt (negated), absolute vertex |
| sag function | `RDY` (+ conic / asphere / XY-poly for freeform) | sphere radius, or 0 = flat |
| aperture | `CIR` / `REX`,`REY` (+ `ADY`) | decentred rectangles on the prism faces |
| medium after | glass name, blank = air, or `REFL`/`TIRO` | toggles at the two refractions |

**Reflection flavour is a physics decision, not a syntax choice.** `REFL` is a coated
mirror and works at any incidence. `TIRO` is TIR-only — the ray *must* exceed the critical
angle or the trace fails. In this design, n = 1.53 gives a critical angle of 40.813°, and:

- face B sits at **21.374°** — below critical, so it physically *cannot* TIR → coated mirror;
- face A's second hit sits at **58.704°** — +17.89° over critical → genuine TIR.

Interchanging them produces a file that traces but is physically fictional. It is scored
separately in the rubric for exactly that reason.

---

## 4. Ray–surface interaction

**CODE V is strictly sequential: the surface list *is* the interaction order.** It does not
search for what the ray hits next; it visits surfaces in order and each one must be hit.
Three consequences drive everything:

**(a) A face hit three times appears three times.** Surfaces 2 and 4 here are *the same
physical plane*. This is the defining feature of a freeform prism, not a workaround. CODE V
now holds two independent copies, and an optimiser will happily pull them into two
different faces — so they are coupled with pickups:

```
PIK RDY S4 S2      PIK YDE S4 S2      PIK ZDE S4 S2      PIK ADE S4 S2
```

Explicit values first, pickups second, so the geometry stays correct even if a `PIK` line
is refused.

**(b) Non-sequential → sequential is just the dominant-ray event list.** The source engine
emits a signature like `T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B`. That string *is*
the CODE V surface list: six events → six surfaces.
`target_fold_design/_folded_trace.py` replays the chief ray analytically along the
signature, recording hit point, directions, normal and incidence at each event, and
`export_fold_folded.py` turns each event into one surface record. That is the entire
bridge.

**(c) Nothing works until the reference ray traces.** Before CODE V computes a spot, an MTF
or even a layout it traces the chief ray for every field. Fail, and every downstream
analysis is unavailable. **But see the diagnostic reflex in `CODEV_SYNTAX_FINDINGS.md`:
with the stop first and in air, that error is nearly always a symptom of misplaced
surfaces, not a chief-ray problem.**

---

## 5. Three constructions, three lessons

**Gen 1 — global `DAR`, zero thickness.** Every surface at its absolute vertex, `THI 0`,
decenter-and-return. Geometry provably correct — vertices landed exactly on the engine's
own hit points. But `THI 0` everywhere meant no glass *volume* (faces drew as lines), and
an object at 1e18 with a decentered stop at the source gave the reference-ray solver no
coherent axis. Imported, drew, would not trace.

**Gen 2 — folded axis.** Make the axis *follow the beam*: `THI` = actual segment length
between hit points, refractive tilt = signed incidence, mirror tilt = half the deflection,
thickness sign flipping per reflection. Real glass volume, real axis. Underlying geometry
exact to <0.001 mm. **Still scattered**, because every tilt and reflected-space sign was
hand-computed — roughly ten independent sign decisions with no self-correction. A further
subtlety: curved faces are hit far off their vertices, so fold tilts must come from the
*hit-point normal*, not the vertex tilt.

**Gen 3 — the current file.** Real stop in air ahead of all optics; absolute placement so
no sign can scatter the layout; the repeated face coupled by pickups; reflection type
chosen by incidence versus critical angle; `PIM` for the image solve. This is the shape a
real freeform-prism prescription takes, and it is the first version whose remaining
unknowns are all *CODE V behaviours* rather than *our arithmetic*.

---

## 6. Where the numbers come from

`verify_v2.py` is the source of truth. It takes **8 geometry parameters** (eye relief,
three face tilts, two path lengths, three curvatures) and derives every vertex, normal,
radius, centre of curvature and clear aperture in the prescription — then traces
5 fields × 13 pupil points × 3 wavelengths through `face A → face B → face A → face C` and
proves the path closes, with TIR margins, at every one.

`check_seq_v2.py` closes the loop from the other end: it parses the `.seq` back,
reconstructs each surface's global frame from its `DAR`/`YDE`/`ZDE`/`ADE` records, and
compares vertex, tilt, radius **and centre of curvature** against `verify_v2.build()`. It
also checks reflection types, pickup coupling, that apertures actually contain every traced
ray, and — added after it bit us — that no `S` line carries extra data CODE V would
silently discard.

Consequence: **the `.seq` and its proof cannot drift apart.** Change geometry in
`verify_v2.py`, regenerate, and re-run the checker. Never hand-edit numbers into the `.seq`.

---

## 7. What this design does *not* yet have

The faces are **spheres**. Real freeform prisms in this class (Cheng & Wang, Beijing
Institute of Technology) use **XY-polynomial or Zernike faces of 5th–6th order**, which is
where the aberration correction and the compactness live. `history/BENCHMARK_PROMPT.md` §4
defines that as the Tier B task and it is unaffected by any of the repairs here.

The documented design methodology for getting there:

1. Start from a rough folded layout with the bounce count and order fixed up front. *(Done
   — CMA-ME discovered it.)*
2. Define feature rays across the full field and pupil.
3. Make the faces freeform; set coefficients plus tilts/decenters variable.
4. Optimise RMS wavefront/spot under constraints: no obscuration, no surface interference,
   **incidence > critical angle on every TIR face**, hold the overall geometry. Start loose,
   then raise polynomial order and ray sampling in later stages.
