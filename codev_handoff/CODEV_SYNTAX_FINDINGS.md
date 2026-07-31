# CODE V syntax — empirically confirmed

Every line here was established by a **live run in CODE V 2026** (PC version) and cost a
diagnostic cycle. Nothing in this file is inferred from documentation. Treat the CONFIRMED
list as settled and the KNOWN-BAD list as closed questions.

---

## CONFIRMED WORKING

| construct | notes |
|---|---|
| `LEN NEW` | starts a new lens. **`NEW LENS` is invalid** ("Error: Invalid command NEW"). |
| `S <rdy> <thi>` | appends a surface. **Takes NO surface index** — it is positional. |
| `S <rdy> <thi> 'GLASS'` | one glass token only. |
| `S <rdy> <thi> REFL` | reflective surface. LDM Refract Mode → **"Reflect"**. |
| `S <rdy> <thi> TIRO 'GLASS'` | TIR-only. LDM Refract Mode → **"Only TIR"**. |
| `SO <rdy> <thi>` | object surface. Infinity = `SO 0 1e18`. |
| `SI <rdy> <thi>` | image surface. |
| `PRV` / `PWL <wl...>` / `'NAME' n1 n2 n3` / `END` | private catalog of model glasses, one index per wavelength in `PWL` order. **This is the reliable way to specify dispersion.** |
| `STO Sn` | aperture stop. |
| `XDE/YDE/ZDE Sn <v>` | surface decenter. |
| `ADE Sn <v>` | surface tilt about X. |
| `DAR Sn` | decenter-and-return. |
| `CIR Sn <r>` , `REX Sn <x>` , `REY Sn <y>` | apertures. |
| `ADY Sn <v>` | aperture decenter in y. |
| `PIK <item> Sn Sm` | pickup; LDM flags the picked value with **`P`**. Confirmed for `RDY`, `YDE`, `ZDE`, `ADE`. |
| `PIM` | paraxial image solve; LDM flags the solved thickness with **`S`**. |
| `WL` `REF` `WTW` `XAN` `YAN` `EPD` `DIM M` `TIT` `GO` | system data. |

---

## KNOWN BAD — do not reintroduce

### 1. `n_d`/`V_d` pair on an `S` line — the silent-drop trap

```
S -46.0130 0.0 1.53 56.0 REFL
   -> Warning: Extra data '56.0 REFL' ignored
```

The `S` line takes **one** glass token. Given a pair, CODE V reads the first number as a
zero-dispersion fictitious glass and **discards the remainder of the line — including
`REFL` / `REFLTIRO`**. The result: a lens with *no reflective surfaces at all* (LDM
Refract Mode read "Refract" on every surface) and three wavelengths tracing identically.

**This is emitted as a WARNING, not an error.** The file loads. Nothing looks broken until
you inspect the Refract Mode column. This is the single most dangerous behaviour found.

*Detector:* the Glass column shows `100000.530000` — CODE V's fictitious-glass encoding,
`100000 + (n_d − 1)`. If you see `100000.*` where you expected a named glass, this happened.

*Fix:* declare glasses in a private catalog (`PRV`/`PWL`/`END`) and reference them by name.

### 2. `REFL` followed by a glass token

```
S -46.0130 0.0 REFL 'PRISM'
   -> Warning: Extra data 'PRISM' ignored
```

A reflection does not change the medium, so `REFL` carries no glass; CODE V inherits it
correctly anyway. Note the **asymmetry**: `TIRO 'PRISM'` *is* accepted without complaint.

### 3. `GLB` — global reference surface. Syntax not established.

Two forms tried, both rejected:

```
GLB S2 S1  -> Error: Too many surface qualifiers
GLB S2 1   -> Error: A global surface qualifier is expected
```

The second message implies `GLB` wants exactly one argument, of a "global surface
qualifier" type whose notation we never determined. **Stop guessing** — if you need `GLB`,
get the signature from CODE V's own help (`Help → Command Reference`) rather than
attempting a third form.

**Consequence when it fails:** every `GLB` erroring left the LDM in **"Basic Decenter"**,
i.e. each surface's decenter measured from the *previous surface's already-tilted frame*,
while the values were authored as **global** coordinates. Surface 2 still landed correctly
(its base frame *is* surface 1's frame, at the origin) and everything after it did not —
producing a scattered layout and rays that refract cleanly through surface 2 then die.

*Workaround in use:* `DAR` + `THI 0` (see below). Also available: set "Global Reference
Surface = 1" per-surface in the Surface Properties dialog, which needs no value changes.

---

## The placement problem, and the DAR workaround

CODE V chains surfaces: after surface *k* you translate along its local **+Z** by `THI(k)`
to reach surface *k+1*'s frame. A tilt/decenter perturbs a surface relative to that
*inherited* frame. So authoring **global** coordinates requires stopping the frame from
accumulating.

`DAR` (decenter-and-return) applies a surface's decenter and then **undoes it**. With
`THI 0` between them, the base frame never moves, so every `YDE/ZDE/ADE` is measured from
the same frame — which is exactly the global behaviour `GLB` was meant to provide.
[`CONTEXT.md`](target_fold_design/CONTEXT.md) §3 records an earlier build where this
construction placed surfaces *exactly* on the source engine's ray-hit points.

**The asymmetry that matters:** `DAR` on the prism faces (each placed absolutely), but
**no `DAR` on the first relay surface** — the relay chain after the prism needs that tilt
to *propagate* down through the following surfaces.

---

## Sign conventions

**Tilt.** +X is out of the page, so a right-handed rotation about +X sends
ẑ → (0, −sin A, cos A). A face whose outward normal lies at **+β** from +Z toward +Y
therefore needs **`ADE = −β`**. Right magnitudes with wrong signs is the classic failure —
it is what scattered `target_fold_design/fold_cell_4_11.folded.seq` into a star.

**Radius.** `RDY` is positive when the centre of curvature lies on the **+local-Z** side of
the vertex. Because `ADE` puts local +Z along the outward normal, a radius prescribed as
`centre = vertex + R × normal` carries through **unchanged**.

*Always verify by centre of curvature, never by inspecting the sign.* For the current
design they must come out at:

```
face A -> (0,  -19.2579,  -66.3804)
face B -> (0,  +14.1213,  -16.3513)
face C -> (0, -267.2870, -204.5363)
```

**Reflected space.** After a reflection CODE V works in reflected coordinates and
thicknesses conventionally flip sign, once per reflection. The current file **sidesteps
this entirely** — every prism surface is placed absolutely with `THI 0`, so no
reflected-space sign arises anywhere. The relay has exactly two reflections upstream, an
even number, so its thicknesses are positive.

---

## Benign messages — expect these, do not chase them

```
Warning: Apertures defined on the image surface (SI) are ignored...
Warning: Decentered aperture(s) ignored for this operation      (paraxial GO only)
Warning: Default overage ratios (DOR) and Radius to Flat (RTF) ... not supported in VIE
Warning: Aperture value is larger than radius on surface 6/7. Reduced.
```

The last one is real but is a **defect in the prescription's relay group**, not in the
file — see README §4.

---

## Diagnostic reflex

`Unable to find chief ray: F1,W1,Z1. Use CRA command.` is **usually a symptom, not a
cause.** When the stop is the first surface in air and the object is at infinity, the chief
ray is simply the field-angle ray through the centre of the stop — there is nothing to
solve. CODE V is failing to *complete* it through misplaced surfaces. **Check the layout
before reaching for `CRA` or Ray Aiming.**

If the layout is provably correct and it still fails, then turn on Reference Ray Aiming
(System Data → Advanced) — that is what first produced real rays in the earliest builds.
