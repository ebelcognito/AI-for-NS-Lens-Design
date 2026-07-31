# CODE V incremental build — context / handoff

Status as of 2026-07-24. Goal: get the discovered fold design tracing in CODE V as a
**freeform prism**, then render it solid, land rays on a sized sensor, and optimize the
focus. This doc is specifically about the **incremental build** now in progress.

## 0. The big reframe (READ FIRST)

The design we're exporting IS a **freeform prism** (off-axis freeform reflective/
refractive prism — the HMD/near-eye class). Confirmed from a paper figure the user
showed: these are built **sequentially in CODE V**, routinely. Consequences:

- The **folded-axis approach is correct.** Do NOT go non-sequential (earlier detour —
  wrong). CODE V is the right tool; the freeform-prism community does exactly this.
- "**Same physical face refracts AND TIRs**" is the *defining feature* of a freeform
  prism (enter S1 → TIR off S1' → exit S3), not an obstacle. Listing a face 2–3× in the
  surface list is standard.

Reference designs to mirror: **Beijing Institute of Technology group, Dewen Cheng &
Yongtian Wang** — freeform prism HMDs with exactly this RGB-ray / solid-block look.

## 1. The 4 ingredients that make a freeform prism TRACE in CODE V

1. **Freeform surface type** — faces are `XY Polynomial`, `Extended Fringe Zernike`, or
   `Q2D Freeform Asphere` (not plain sphere/cylinder). Design freedom + aberration
   correction live in these coefficients. (We currently emit sphere radius only — fine
   for first trace, but upgrade to freeform for the real design.)
2. **Folded tilt/decenter chain**, real thicknesses, reflected space after each fold.
3. **Same face listed 2–3×** (enter/TIR/exit).
4. **Real pupil + SOLVED fold tilts + GLOBAL coordinate reference** — the chief ray
   traces because there's a real stop and the fold tilts are set by an **auto-solve
   against the reference ray** (analogous to OpticStudio's chief-ray solve on a coordinate
   break), NOT hand-typed. Surfaces placed via a **global coordinate reference**, not a
   fragile local negative-thickness chain.

## 2. WHY our current .seq scatters (the actual bug)

`fold_cell_4_11.folded.seq` imports but the 7 surfaces fly apart into a "star" and the
chief ray still fails ("Unable to find chief ray: F1,W1,Z1"). Root cause: we **hand-
computed every fold tilt** (mirror ADE = ½ deflection, refract ADE = signed incidence)
plus reflected-space negative thicknesses. Any sign error scatters the surfaces, and then
there's no coherent path for the reference ray to thread. This is ingredient #4 done the
fragile way. **The pros don't hand-type tilts — they solve them.**

## 3. The robust rebuild plan (do this)

Replace the hand-typed local chain with:
- a **real stop** at a defined location (reconsider the collimated + decentered-stop-at-
  infinity input — it's unusual; may itself block the chief ray; see step in §5),
- **fold surfaces as REFL/TIRO with tilt solves** so CODE V aligns the axis to the
  reference ray at each fold (stop guessing signs),
- **global coordinate reference** for placing surfaces robustly.

## 4. Incremental build PROTOCOL (the method we agreed on)

Neither side can trace CODE V here, so build up ONE surface at a time:
1. Start from the simplest system that traces.
2. Run + **screenshot View Lens AND the Command Window** (need the error text).
3. Assistant reads geometry, fixes the one sign/param, confirms it traces.
4. Only THEN add the next surface. Never add a surface while the current one is broken.

**Immediate next step (was pending when context exported):** user runs `diag_slab.seq`
(straight-through tilted slab, NO fold) to isolate the input/chief-ray setup:
- traces cleanly → input is fine; the fold sign convention was the whole problem →
  proceed to add fold #1 (`@D`) the robust way (§3).
- still "Unable to find chief ray" → the object-at-infinity + decentered-stop is the gate;
  fix that in isolation first (turn on **Ray Aiming**: System Data → General; and/or move
  the beam offset off the stop / use a real finite pupil) before any fold.

## 5. The design test case (cell 4_11 — validated geometry)

Genotype: `ray_y=9.4592, ang=23.8848, cx=58.3601, thick=28.6574, fsag=-3.9085,
bsag=2.5657, tilt_a=25.0, tilt_b=15.6382, n=1.5`
dom_signature: `T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B` (4 TIR bounces).

Folded-axis geometry (from `_folded_trace.py`, validated vs engine to <0.001 mm).
n=1.5 → critical angle **41.810°**. ALL four TIR incidences clear it comfortably:

```
 seg (thi, mm)        | hit face event   incidence  TIR?
 stop->A  37.2073     |  A    A   refract  10.9°     n/a
 A->D     11.8446     |  D    D   tir      62.5°     yes
 D->B     28.4868     |  B    B   tir      48.1°     yes
 B->A     39.5413     |  A    A   tir      52.1°     yes
 A->T     17.1786     |  T    T   tir      54.4°     yes
 T->B     13.1297     |  B    B   refract  26.8°     n/a
 B->sensor 63.9221    |  sensor hit engine (x,y)=(120.000, 27.545)
```
Face radii (cylinder/sphere, meridionally exact): A `RDY=-156.0548`, B `RDY=-197.5431`,
T/D flat. Sensor: `CIR SI <half>` sizes the detector; solid render is automatic once THI
is real. UNVERIFIED (needs keyboard): ADE handedness, RDY sign, reflected-space THI —
which is exactly what the §3 solve-based rebuild replaces.

## 6. Design methodology (how the paper GOT the shape — for step (c), later)

1. Start from a rough planar/folded layout (flat faces ~ where freeform faces go); the
   **folding geometry (bounce count/order) is fixed up front** by constraints. We already
   have this — CMA-ME discovered it.
2. Define feature rays across full field + pupil.
3. Make faces **freeform** (XY-poly 5th–6th order); set coefficients + tilts/decenters VAR.
4. Optimize RMS wavefront/spot with constraints: no obscuration, no surface interference,
   **incidence > critical angle on TIR faces**, hold geometry. Loose → tighten; raise
   polynomial order + ray sampling in later stages.

## 7. Files

- `ns_probe/_folded_trace.py` — analytic chief-ray tracer (validated). Self-contained.
- `ns_probe/export_fold_folded.py` — folded-axis emitter (`--tir N`, `--cell I0 I1`).
- `ns_probe/results_fold_qd/codev_export/`:
  - `diag_slab.seq` — straight-through slab diagnostic (STEP 0, pending).
  - `fold_cell_4_11.folded.seq` / `.folded.spec.txt` — hand-typed folded (scatters; being
    replaced by the §3 solve-based build).
  - `fold_cell_1_15.folded.*` — a 1-bounce case for sign validation.
- Prior handoff: `CONTEXT.md` (global-DAR attempt, syntax findings). This doc supersedes
  it for the tracing/build strategy.

## 8. Sources
- Keysight/Synopsys "Top 5 Tools for Designing Freeform Surfaces in CODE V".
- "Automated design of freeform imaging systems" (PMC6061905) — methodology.
- BIT freeform prism HMD papers (Cheng, Wang) — reference designs.
