# Single-shot CODE V `.seq` benchmark — off-axis freeform-prism eyepiece (V2)

Replaces the Tier A prompt in `BENCHMARK_PROMPT.md`, whose geometry was **not
satisfiable**: 3 of its 5 prescribed fields never closed (+11° and +22° left S1
travelling away from S3; −22° fell 3.4° under the critical angle) and all three faces
overran their stated apertures by 2–3×. A perfect answer could not have scored full
marks. See §4.

The geometry below is **verified to close for all 5 fields, over the whole 8 mm pupil,
at all 3 wavelengths** (`verify_v2.py`). Everything the rubric asks for is achievable.

- **§1** — the prompt. Copy the block verbatim.
- **§2** — answer key.
- **§3** — rubric.
- **§4** — what changed from V1 and why.

---

## §1. THE PROMPT

Give the model this and nothing else. It gets **one response, no tools, no follow-up**.
Attaching the figure is optional — the prompt is self-contained without it.

---

> You are given the complete prescription of an off-axis, plane-symmetric prism
> eyepiece for a near-eye display. Write a single **CODE V sequence file (`.seq`)**
> that builds and traces it.
>
> **This is a one-shot task.** You have no optical software to test against, no
> ability to iterate, and no opportunity to ask questions. Output **only the contents
> of the file**. Get it right in one pass.
>
> **Coordinate frame (used to specify the design; how you realise it is up to you).**
> Right-handed global XYZ, origin at the centre of the exit pupil, **+Z** toward the
> prism, **+Y** up, **+X** out of the page. The system is symmetric about the YZ plane.
>
> **System conditions**
> - Units mm. Object at infinity.
> - The **aperture stop is the exit pupil (the eyebox)**: in air, at the global origin,
>   **ahead of every optical surface**. Entrance pupil diameter **8.0 mm**.
> - Wavelengths **656.27 / 587.56 / 486.13 nm**, reference **587.56 nm**.
> - Five field points in the YZ meridian: Y field angles **0, +11, −11, +22, −22
>   degrees**. Zero X field.
>
> **Prism** — one solid block of model glass **n_d = 1.5300, V_d = 56.0**. It has three
> physical faces. Each is a **sphere**, specified here by its vertex, its outward unit
> normal at that vertex, and its radius. The centre of curvature lies at
> `vertex + radius × normal` (so a negative radius puts the centre on the −normal side).
>
> | face | vertex (x, y, z) | outward normal (nx, ny, nz) | radius | clear aperture (x × y) |
> |------|------------------|------------------------------|--------|------------------------|
> | S1 | (0, 0.000000, 18.000000) | (0, +0.222505, +0.974931) | −86.5501 | 10 × 38 mm, centred at local y = +5.52 |
> | S2 | (0, 0.760776, 27.679225) | (0, −0.290365, +0.956916) | −46.0130 | 10 × 32 mm, centred at local y = +0.26 |
> | S3 | (0, 27.495340, 17.878452) | (0, +0.798271, +0.602299) | −369.2762 | 10 × 28 mm, centred at local y = −2.08 |
>
> ("local y" is measured in the face's own plane from its vertex, +y in the direction of
> increasing global Y.)
>
> **Required ray path.** The chief ray of the central field leaves the pupil along +Z and
> must visit the faces in exactly this order, with exactly these events:
>
> | # | face | event | medium after |
> |---|------|-------|--------------|
> | 1 | S1 | refraction, air → glass | glass |
> | 2 | S2 | reflection off a **coated mirror** | glass |
> | 3 | S1 | **total internal reflection** — the *same physical face* as step 1 | glass |
> | 4 | S3 | refraction, glass → air | air |
>
> Step 3 is the same physical surface as step 1: same sphere, same vertex, same tilt,
> same aperture. Step 2 sits **below** the critical angle and therefore cannot be TIR —
> it is a reflective coating. Step 3 sits **above** it and is true TIR. Your file must
> encode that distinction.
>
> **Relay group.** After S3 the central chief ray travels in air along the unit direction
> **(0, +0.995498, −0.094779)**. On that axis, in order:
> - **4.0000 mm** of air from the S3 vertex, then a **cemented doublet** — vertices on
>   that axis, surfaces normal to it:
>   R1 = **+6.4000**, t = **2.5000**, glass n_d 1.51680 / V_d 64.17;
>   R2 = **−4.8000**, t = **1.0000**, glass n_d 1.67270 / V_d 32.25;
>   R3 = **−20.0000**. Clear aperture **Ø 26 mm**.
> - then air, then a **plane cover plate 0.5000 mm** thick, n_d 1.51680 / V_d 64.17,
>   normal to the same axis.
> - then the **microdisplay** (the image surface), normal to the same axis, active area
>   **12.0 × 10.0 mm**.
>
> Set the airspace between the doublet and the cover plate with a **paraxial image
> solve**, so the display lands at the paraxial focus rather than at a number you guess.
>
> **The file must**
> 1. Build the system so that it traces — all 5 fields, all 3 wavelengths, real rays.
> 2. Place every surface so its true global position matches the table above.
> 3. Keep the two S1 entries locked to one physical surface, so that editing S1 edits
>    both.
> 4. Carry the stated clear aperture on every surface.
> 5. Be complete and runnable: loadable as-is, ending ready to view.
>
> Begin the file with a short comment block giving the ray path, the critical angle, the
> chief-ray incidence angle you compute at each of the four events, and any assumption
> you made.

---

## §2. ANSWER KEY

Prism n_d 1.5300 → **critical angle 40.813°** (d line); 40.908° at C, 40.601° at F.
The C line is the worst case for TIR.

### Central chief ray (exact)

| # | face | event | segment (mm) | hit point (y, z) | incidence | note |
|---|------|-------|--------------|------------------|-----------|------|
| 1 | pupil | stop | — | (0.0000, 0.0000) | — | EPD 8.0, object at ∞ |
| 2 | S1 | refract | 18.0000 | (0.0000, 18.0000) | 12.856° | air → glass |
| 3 | S2 | mirror | 9.7091 | (0.7608, 27.6792) | 21.374° | **below critical → must be a coating** |
| 4 | S1 | TIR | 16.5093 | (10.9824, 14.7149) | 58.704° | **+17.89° over critical** |
| 5 | S3 | refract | 16.8132 | (27.4953, 17.8785) | 26.189° | glass → air |

Exit direction **(0, +0.995498, −0.094779)** = **95.4386°** from +Z.
Prism vertex envelope 12.96 mm (Z) × 27.50 mm (Y). Pupil → S3 path 79.03 mm.
Doublet front vertex at global (y, z) = **(31.4773, 17.4993)**.
Doublet EFL′ = 12.3924 mm, BFD = 10.3534 mm; display centre ≈ (45.27, 16.19).

### Centres of curvature (for checking the radius sign)

S1 → (0, −19.2579, −66.3804)  S2 → (0, +14.1213, −16.3513)  S3 → (0, −267.2870, −204.5363)

### Worst case over the full 8 mm pupil × 3 wavelengths

| field | min S1 TIR incidence | TIR margin | max S2 incidence | max S3 incidence | status |
|-------|----------------------|-----------|------------------|------------------|--------|
| 0° | 49.900° | **+8.99°** | 26.14° | 32.90° | closes |
| +11° | 48.146° | **+7.24°** | 27.43° | 37.12° | closes |
| −11° | 51.179° | **+10.27°** | 24.53° | 26.89° | closes |
| +22° | 45.575° | **+4.67°** | 28.20° | 38.58° | closes |
| −22° | 52.311° | **+11.40°** | 22.83° | 20.26° | closes |

Every S2 incidence stays ≥ 12.6° **below** critical (so a model that makes S2 a TIR
surface is wrong on physics, not just on syntax). Every S3 exit incidence stays below
critical, so nothing is trapped. Every segment length is positive at every field.

### The sign trap — the single most discriminating item

Faces are given as **normal vectors, not tilt angles**, so the model must convert. With
+X out of the page, a right-handed rotation about +X carries +Z toward −Y, so a face
whose normal lies at +β from +Z toward +Y needs a **negative** X-rotation of magnitude β.

| surface | normal angle from +Z | expected CODE V tilt |
|---------|---------------------|----------------------|
| S1 (both entries) | +12.8562° | **ADE −12.8562** |
| S2 | −16.8798° | **ADE +16.8798** |
| S3 | +52.9653° | **ADE −52.9653** |
| relay axis | +95.4386° | **ADE −95.4386** |

Right magnitudes with wrong signs is the classic failure — it is exactly what scattered
`fold_cell_4_11.folded.seq` into a star. Score it separately from the magnitudes.

---

## §3. RUBRIC — 10 points

| # | Check | Pts | How to verify |
|---|-------|-----|---------------|
| 1 | Loads without a syntax error | 1 | open in CODE V |
| 2 | Built **sequentially** — no non-sequential construct attempted | 1 | inspect; see note |
| 3 | Surface count and order match the 4-step path (S1, S2, S1, S3) | 1 | surface list |
| 4 | S2 is a **mirror**, S1's second entry is **TIR** — not interchanged | 1 | surface properties |
| 5 | Global coordinates of all surfaces match §2 to <0.05 mm / <0.05° | 2 | global-coordinate listing |
| 6 | Real chief ray traces at **all 5 fields × 3 wavelengths** | 2 | ray trace |
| 7 | The two S1 entries are genuinely coupled (pickup / global reference), not two independent copies | 1 | edit S1's radius, confirm both move |
| 8 | Apertures, radii, model glasses and the paraxial image solve all present | 1 | lens data |

**Note on check 2.** CODE V models this class *sequentially* — the surface list simply
names a physical face more than once. A model that reaches for a non-sequential / NSC
construct has misread the tool and loses the point even if the file loads. This is a
known attractor; watch for it specifically.

**Partial-credit signals, logged separately** — they discriminate better than the total:
(a) tilt magnitudes right, signs wrong; (b) local negative-thickness chain instead of a
global coordinate reference (fragile, not strictly wrong); (c) hand-typed fold tilts
rather than solves; (d) silently drops the second S1 entry and treats the prism as three
separate faces; (e) gets the radius **sign** wrong (check against the centres of
curvature in §2); (f) self-reported incidence angles in the header disagree with §2 —
tells you whether the model actually traced the thing in its head or guessed.

Checks 1–5, 7, 8 and the self-reported header table are gradable **without running CODE
V**. Only check 6 needs the software.

---

## §4. What changed from V1, and why

V1's prism was a plane-faced block at 18 mm eye relief with tilts 14° / −16° / 80°.
Re-tracing it (`trace3d.py`, `closure.py`) showed:

- chief rays close only over roughly **−11° to +2°** of field — not the ±22° prescribed;
- at +11° and +22° the post-TIR ray hits the S3 *plane* behind itself (negative segment);
- at −22° the S1 incidence is 37.4°, **under** the 40.8° critical angle, so no TIR;
- required apertures were ±28.6 / ±14.5 / ±18.0 mm against stated ±10 / ±8 / ±7;
- with the full 8 mm pupil enforced, **no** field closed.

The repair needed three things: spherical rather than plane faces (a plane-faced block
in this family cannot hold ±22° — the search returns no feasible point at any tilt),
larger honest apertures, and a re-solved fold. V2 keeps the 18 mm eye relief, the 8 mm
eyebox, the ±22° field and the mirror/TIR split; it gives up V1's fictional 12.5 × 16 mm
block for a real 13 × 27.5 mm one.

**The prism is bigger than the figure's and the relay is fast.** That is honest: with
spherical faces this is what ±22° over an 8 mm eyebox costs. Getting the figure's
compactness needs 6th-order XY-polynomial freeform faces — which is the Tier B task in
`BENCHMARK_PROMPT.md` §4, still valid and unaffected by this repair. The relay here is a
stand-in whose only job is to give the trace a real image surface; it is not a
manufacturable f/#, and the rubric does not grade image quality.

## §5. Files

- `verify_v2.py` — regenerates every number above from the 8 geometry parameters.
- `trace3d.py`, `closure.py`, `fixed_key.py` — the V1 checkers that found the defect.
- `fold_cell_4_11.folded.seq` — the scattering failure this benchmark detects.
