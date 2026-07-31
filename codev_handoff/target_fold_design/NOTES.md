# Objective B — the real design

**Do not start here.** Get `../benchmark_tierA_v2c.seq` tracing first; every convention it
settles applies directly to this. This folder is the actual goal, preserved with its exact
geometry so it is ready when the conventions are known.

---

## What it is

A **4-TIR-bounce folded prism**, discovered by CMA-ME search in this repo — not
hand-designed. One glass element with four faces (A and B curved, T and D flat edges),
constant index n = 1.5, monochromatic, 2-D meridional.

```
genotype       ray_y=9.4592  ang=23.8848  cx=58.3601  thick=28.6574
               fsag=-3.9085  bsag=2.5657  tilt_a=25.0  tilt_b=15.6382  n=1.5
dom_signature  T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B
```

That signature *is* the CODE V surface list: six events → six surfaces, with face A
appearing twice and face B twice. Critical angle 41.810°; **all four TIR incidences
(62.5°, 48.1°, 52.1°, 54.4°) clear it comfortably.**

Geometry is validated against the source ray-tracing engine to **<0.001 mm**.

---

## Status

`fold_cell_4_11.folded.seq` **imports but scatters into a star**, and the chief ray fails.

Root cause, established before the Tier A work: every fold tilt and every reflected-space
thickness sign was **hand-computed** — roughly ten independent sign decisions with no
self-correction, so a single error destroys the layout with no diagnostic. A further
subtlety specific to this design: the curved faces are hit **far off their vertices**, so
each fold tilt must be derived from the **hit-point normal**, not the vertex tilt.

---

## Why the Tier A work should unblock this

The current file uses the fragile construction (real thicknesses threading the axis, signs
flipping per reflection). The Tier A file demonstrates the robust one:

- absolute placement with `THI 0`, so **no reflected-space sign is ever computed**;
- a real stop in air ahead of all optics;
- repeated faces coupled by `PIK` rather than duplicated independently;
- reflection type chosen by incidence versus critical angle.

Porting this design to that construction is mostly mechanical — the hit points are already
known exactly. What is *not* yet built is a `verify_*.py` equivalent for this design that
proves closure over the full beam rather than the single dominant ray. **Build that first**;
`../verify_v2.py` is the template, and `history/FAILURE_LOG.md` run 0 is what happens
without it.

---

## Files

| file | what |
|---|---|
| `fold_cell_4_11.folded.spec.txt` | **the geometry, source of truth** — one row per surface: face, event, radius, thickness, incidence, deflection, tilt, medium |
| `fold_cell_4_11.folded.seq` | the folded-axis macro that scatters |
| `fold_cell_4_11.spec.json` / `.spec.txt` | the earlier global-DAR export of the same design |
| `_folded_trace.py` | self-contained analytic chief-ray tracer through the signature; validated. Run it directly to reproduce the hit points |
| `export_fold_folded.py` | folded-axis emitter (`--tir N`, `--cell I0 I1`) |
| `export_fold_for_codev.py` | global-DAR emitter; also holds `faces_of()`, the face geometry |
| `diag_slab.seq` | straight-through tilted slab, **no fold** — isolates the input/chief-ray setup |
| `diag_fold1.seq` | single-fold diagnostic |
| `INCREMENTAL_BUILD.md` | the build protocol and the freeform-prism reframe |
| `CONTEXT.md` | earlier handoff: coordinate mapping, first syntax findings |

`_folded_trace.py` and the exporters import from each other and expect the repo layout
(`ns_probe/`); they are copied here for reference. The `.spec.txt` files are self-contained
and are what you actually need to build by hand.

---

## Validated chief-ray path (cell 4_11)

Engine (x, y) = CODE V (Z, Y). Regenerate with `python3 _folded_trace.py` in the repo.

```
P0 source        ( 0.000,  9.459)
P1 refract @A    (34.021, 24.524)   seg 37.207  dir   23.885  (air -> glass)
P2 tir     @D    (44.524, 30.000)   seg 11.845  dir   27.535
P3 tir     @B    (69.784, 16.831)   seg 28.487  dir  -27.535
P4 tir     @A    (55.407,-20.004)   seg 39.541  dir -111.321
P5 tir     @T    (69.378,-30.000)   seg 17.178  dir  -35.584
P6 refract @B    (80.056,-22.360)   seg 13.130  dir   35.584  (glass -> air)
P7 sensor        (120.000, 27.545)  seg 63.922  dir   51.326
```

Face radii: A `RDY = -156.0548`, B `RDY = -197.5431`, T and D flat.
D and T are horizontal-mirror reflections (flip the y-component); A and B are the curved
reflections.

---

## The longer arc

Once it traces: upgrade the faces from spheres/cylinders to **XY-polynomial freeform**
(5th–6th order) and optimise RMS wavefront/spot under constraints — no obscuration, no
surface interference, incidence > critical on every TIR face, hold the overall geometry.
That is where a freeform prism's real performance comes from. See `../BUILD_METHOD.md` §7.
