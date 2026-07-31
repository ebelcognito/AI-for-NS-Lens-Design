# Single-shot CODE V `.seq` benchmark — off-axis freeform-prism eyepiece

Purpose: measure whether a model can produce a **working CODE V sequence file** for a
folded, non-rotationally-symmetric prism system in one shot, with no tool access and no
iteration. Built 2026-07-28 from the paper figure the user supplied.

Three parts:
- **§1** — the figure, read back as an optical spec.
- **§2** — Tier A prompt (fully prescribed geometry; objectively gradable). Copy-paste.
- **§3** — answer key + rubric for Tier A.
- **§4** — Tier B prompt (open design from the functional spec; trace-graded).

---

## §1. The figure, in optical-spec language

**Class.** Off-axis, plane-symmetric ("freeform") wedge-prism eyepiece with a tilted relay
group feeding a microdisplay — the near-eye / OST-HMD architecture. Symmetric about the
YZ plane (the plane of the page); no rotational symmetry anywhere. This is the
Beijing Institute of Technology (Cheng & Wang) / Hua freeform-prism family.

**Conjugates and stop.** The stop is the **exit pupil (eyebox)**, standing free in air
*ahead of all optics* — the two tick marks at left, Ø ≈ 8 mm at the 8 mm scale bar. The
system is therefore laid out and traced **from the pupil backwards to the display**.
Object space (left) is at infinity: each of the five ray bundles is collimated, so the
user sees a virtual image at infinity.

**Field.** The five colours are **five field points, not five wavelengths** — each bundle
converges to its own spot on the display. They span roughly ±22–25° in the drawn
meridian, i.e. **≈ 45–50° full field**, plane-symmetric about the page.

**Eye relief.** ≈ 16–18 mm from pupil to the first prism face.

**Prism.** A single moulded block, ≈ 12.5 (Z) × 16 (Y) mm in section, low-index polymer
(n_d ≈ 1.49–1.53, PMMA / ZEONEX class). Three optical faces, all freeform
(XY-polynomial or Zernike over a tilted base conic). The canonical ray path uses **one
face twice**:

```
pupil → S1 refract (air→glass) → S2 reflect (coated mirror) → S1 again, TIR → S3 refract (glass→air)
```

That double use of S1 — refracting on the way in, total-internal-reflecting on the way
back — is the defining feature of the class, and it is why the surface list contains more
entries than the block has physical faces.

**See-through channel.** The blue-shaded wedge cemented on the right of the block is a
**freeform compensation lens**. Its outer face is figured so that prism + compensator is
afocal and distortion-free for the real world; the rays continuing past the block to the
vertical plane at far right are that second channel. In CODE V this is a **second zoom
position / configuration**, not a second lens file.

**Relay group.** Four powered elements (at least one cemented doublet) above the prism,
axis tilted ≈ 30–35° from the prism exit-face normal, ≈ 15 mm long.

**Microdisplay.** Behind a plane cover plate at the top, active area ≈ 8–10 mm in the
drawn meridian, tilted with the relay axis (possibly a Scheimpflug tilt to straighten the
image surface).

**Waveband.** Visible, 486.13 / 587.56 / 656.27 nm, display-weighted.

*(Dimensions marked ≈ are scaled off the figure against its 8 mm bar; treat them as
±15%.)*

---

## §2. TIER A PROMPT — copy everything between the rules

The geometry below is fully prescribed and verified to close (see §3). Nothing about
CODE V syntax is given: naming the right commands is part of what is being measured.

---

> You are given the complete prescription of an off-axis, plane-symmetric prism eyepiece.
> Write a single **CODE V sequence file (`.seq`)** that builds and traces it. Output only
> the file contents. You have no software to test against — get it right in one pass.
>
> **Coordinate frame (for specifying the design only).** Right-handed global XYZ: origin
> at the centre of the exit pupil, **+Z** toward the prism, **+Y** up, **+X** out of the
> page. The system is symmetric about the YZ plane.
>
> **System conditions**
> - Object at infinity. The **aperture stop is the exit pupil**, in air, ahead of every
>   optical surface, at the global origin, entrance pupil diameter **8.0 mm**.
> - Wavelengths 656.27 / 587.56 / 486.13 nm, reference 587.56 nm.
> - Five field points in the YZ meridian: Y field angles **0, ±11°, ±22°**. Zero X field.
> - Units: mm.
>
> **Prism** — one solid block of model glass **n_d = 1.5300, V_d = 56.0**. Three physical
> faces, each a plane in this tier, each specified by a point on the face and its outward
> normal (unit vector, in global XYZ):
>
> | face | point on face (x, y, z) | normal (nx, ny, nz) | clear aperture (x × y) |
> |------|-------------------------|---------------------|------------------------|
> | S1 | (0, 0, 18.0000) | (0, +0.241922, +0.970296) | 24 × 20 mm |
> | S2 | (0, 0.6409, 25.4726) | (0, −0.275637, +0.961262) | 24 × 16 mm |
> | S3 | (0, 11.3144, 17.7788) | (0, +0.984808, +0.173648) | 24 × 14 mm |
>
> **Required ray path.** The reference (chief) ray of the central field leaves the pupil
> along +Z and must visit the faces in exactly this order, with exactly these events:
>
> | # | face | event | medium after |
> |---|------|-------|--------------|
> | 1 | S1 | refraction, air → glass | glass |
> | 2 | S2 | reflection off a **coated mirror** | glass |
> | 3 | S1 | **total internal reflection** (same physical face as step 1) | glass |
> | 4 | S3 | refraction, glass → air | air |
>
> Step 3 must be the *same physical surface* as step 1 — same plane, same tilt, same
> aperture. Step 2 is below the critical angle and therefore cannot be TIR; it is a
> reflective coating. Your file must encode this distinction.
>
> **Relay group.** After S3 the central chief ray travels in air along the direction
> (0, +0.834030, +0.551718). On that axis, in order:
> - 4.0000 mm air, then a **cemented doublet**, vertices on that axis, surfaces normal to it:
>   R1 = +6.4000, t = 2.5000, glass n_d 1.51680 / V_d 64.17;
>   R2 = −4.8000, t = 1.0000, glass n_d 1.67270 / V_d 32.25;
>   R3 = −20.0000. Clear aperture Ø 11 mm.
> - then air, then a **plane cover plate** 0.5000 mm thick, n_d 1.51680 / V_d 64.17,
>   normal to the same axis.
> - then the **microdisplay** (image surface), normal to the same axis, active area
>   10.0 × 5.6 mm.
>
> Set the airspace between the doublet and the cover plate with a **paraxial image
> solve**, so the display lands at the paraxial focus rather than at a number you guess.
>
> **The file must**
> 1. Build the system so it traces — all 5 fields, all 3 wavelengths, real rays.
> 2. Place every surface such that its true global position matches the table above.
> 3. Keep the two S1 entries locked to one physical surface, so that editing S1 edits
>    both.
> 4. Size every surface with the stated clear aperture.
> 5. Be a complete, runnable file: it should be loadable as-is and end ready to view.
>
> Add a short comment block at the top stating the ray path and any assumption you made.

---

## §3. Tier A — answer key and rubric

### Verified chief-ray solution (central field, exact)

Prism n_d 1.5300 → **critical angle 40.813°**.

| # | face | event | segment (mm) | hit point (y, z) | incidence | note |
|---|------|-------|--------------|------------------|-----------|------|
| 1 | pupil | stop | — | (0.0000, 0.0000) | — | EPD 8.0, object at ∞ |
| 2 | S1 | refract | 18.0000 | (0.0000, 18.0000) | 14.000° | → 4.902° inside glass |
| 3 | S2 | reflect | 7.5000 | (0.6409, 25.4726) | 20.902° | **below critical → must be a mirror** |
| 4 | S1 | TIR | 11.7430 | (7.6920, 16.0822) | **50.902°** | +10.09° margin over critical |
| 5 | S3 | refract | 4.0000 | (11.3144, 17.7788) | 15.098° | exits at 56.515° from +Z |

Prism envelope traversed: 9.39 mm in Z, 11.31 mm in Y. Relay axis 56.515° from +Z
(33.485° from vertical). Doublet EFL′ = 12.3924 mm, BFD = 10.3534 mm; half-image height
at 22° = 5.007 mm, so the 10 mm display is correctly filled. Relay L1 front vertex is at
global (y, z) = (14.6505, 19.9857); display centre at (26.2046, 27.6290). Total pupil →
display path 59.096 mm.

**Sign trap (the main thing being measured).** Face normals are given as vectors, not as
tilt angles, so the model must convert. With +X out of the page, a right-handed rotation
about +X carries +Z toward −Y — so a face whose normal is at +β from +Z toward +Y needs a
**negative** X-rotation of magnitude β. Correct magnitudes are 14°, 16°, 80°; the
expected tilt entries are therefore −14°, +16°, −80° in CODE V's convention. Getting the
magnitudes right and the signs wrong is the single most common failure and is worth
scoring separately — it is exactly what scattered `fold_cell_4_11.folded.seq` into a star.

### Scoring — 10 points

| # | Check | Pts | How to verify |
|---|-------|-----|---------------|
| 1 | File loads without a syntax error | 1 | open it in CODE V |
| 2 | Built **sequentially** — no non-sequential construct attempted | 1 | inspect; see note below |
| 3 | Surface count and order match the 4-step path (S1, S2, S1, S3) | 1 | surface list |
| 4 | S2 is a **mirror**, S1's second entry is **TIR** — not interchanged | 1 | surface properties |
| 5 | Global coordinates of all surfaces match the §3 table to <0.05 mm / <0.05° | 2 | global-coordinate listing |
| 6 | Real chief ray traces at **all 5 fields, all 3 wavelengths** | 2 | ray trace; "unable to find chief ray" = 0 |
| 7 | Two S1 entries are genuinely coupled (pickup / global reference), not two independent copies | 1 | edit S1 curvature, confirm both move |
| 8 | Apertures, model glasses, and the paraxial image solve all present | 1 | lens data |

**Note on check 2.** CODE V models this class *sequentially* — the surface list simply
names a physical face more than once. A model that reaches for a non-sequential /
NSC-style construct has misread the tool and should lose the point even if the file
happens to load. Watch for this specifically; it is a known attractor.

**Partial-credit signals worth logging separately** (they discriminate better than the
total): (a) magnitudes right / signs wrong on the tilts; (b) uses a local
negative-thickness chain instead of a global coordinate reference — fragile but not
strictly wrong; (c) hand-computes the fold tilts rather than letting a solve set them;
(d) silently drops the second S1 entry and treats the prism as three separate surfaces.

---

## §4. TIER B PROMPT — open design (harder, trace-graded)

Same architecture, no prescription. Use when Tier A saturates.

---

> Design and write a **CODE V sequence file** for an off-axis freeform-prism near-eye
> display eyepiece, from this functional spec alone:
>
> - Exit pupil (eyebox) **Ø 8 mm** in air, **18 mm** eye relief, stop ahead of all optics.
> - Full field **44° × 24°**, plane-symmetric about the YZ plane. Virtual image at infinity.
> - Wavelengths 656.27 / 587.56 / 486.13 nm, reference d.
> - A **single polymer prism**, n_d ≈ 1.53, envelope no larger than 13 (Z) × 20 (Y) mm,
>   with three optical faces and the ray path
>   `pupil → S1 refract → S2 mirror → S1 TIR → S3 refract`.
>   Every TIR incidence must exceed the critical angle by ≥ 5°.
> - A **tilted relay group** of up to four elements between the prism and a
>   **10.0 × 5.6 mm microdisplay** behind a 0.5 mm cover plate.
> - The three prism faces must be **XY-polynomial freeform** surfaces (through 6th order),
>   symmetric in X, on tilted/decentred base conics, with the coefficients and the
>   tilts set up as optimisation variables.
> - Add a **second configuration** for the see-through channel: a cemented freeform
>   compensator on the far side of the prism, such that the assembly is afocal and
>   distortion-free for a real-world object at infinity.
>
> Output the complete file, plus a short table of the chief-ray incidence angle at every
> surface for the central field, so the TIR condition can be checked.

**Grading Tier B** on: does it trace at all (5 fields × 3 λ); do the reported incidence
angles actually clear critical; is the prism envelope respected; are the freeform faces
real freeform surface types with variables declared; is the see-through channel a second
configuration rather than a second file; is S1 still one physical surface.

---

## §5. Files

- `answer_key.py` (scratchpad) — regenerates every number in §3 from the five geometry
  parameters `B1=14°, B2=−16°, B3=80°, L2=7.5, L4=4.0`, eye relief 18 mm.
- Related context: `INCREMENTAL_BUILD.md` (why solved tilts beat hand-typed ones),
  `fold_cell_4_11.folded.seq` (the scattering failure this benchmark is designed to detect).
