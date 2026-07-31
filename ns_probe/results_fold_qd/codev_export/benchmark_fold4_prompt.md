# SINGLE-SHOT BENCHMARK PROMPT — copy everything below this line

You are given the complete physical description of a folded solid-glass optical element
and of the beam that traces through it. Write a single **CODE V sequence file (`.seq`)**
that builds this system and traces it.

**Rules — read first.**
- This is one shot. Do not ask questions, do not propose alternatives, do not iterate.
  You have no ray-tracing software to check against; get it right in one pass.
- No CODE V syntax is given below. Choosing the right commands, surface types, solves
  and sign conventions is exactly what is being measured.
- Your entire reply must be the contents of the `.seq` file and nothing else — no prose
  before or after it, no markdown fences.

---

## Coordinate frame (for specifying the design only)

Right-handed, units **mm**: **z** horizontal, the nominal propagation direction, to the
right; **y** vertical, up; **x** out of the page. Points are written **(z, y)**.

The whole design lives in the plane x = 0 and is **invariant along x**: it is a strictly
2-D meridional design. Its curved faces are therefore **cylinders with axes parallel to
x**, not spheres — a sphere would not reproduce this design's physics.

## The element

One solid block of glass, refractive index **n = 1.500000**, constant (monochromatic
model, λ = 587.56 nm; no dispersion). Its cross-section is a closed contour of four
faces — two straight, two circular arcs:

| face | type | endpoint 1 (z, y) | endpoint 2 (z, y) | arc passes through (z, y) |
|------|------|-------------------|-------------------|---------------------------|
| **A** (left)   | circular arc | (30.042190, +30.0) | (58.020649, −30.0) | (47.939893, 0.0) |
| **D** (top)    | straight     | (30.042190, +30.0) | (64.291029, +30.0) | — |
| **B** (right)  | circular arc | (64.291029, +30.0) | (81.086511, −30.0) | (75.254482, 0.0) |
| **T** (bottom) | straight     | (58.020649, −30.0) | (81.086511, −30.0) | — |

Derived, for convenience (both are shallow arcs bulging toward +z):

| face | radius (mm) | centre of curvature (z, y) |
|------|-------------|----------------------------|
| A | 156.054833 | (−94.183972, −64.450895) |
| B | 197.543133 | (−115.161434, −52.583912) |

Everything outside the contour is air. There are no coatings anywhere: every face is a
bare glass/air interface, so a reflection at a face is a reflection only if it is a true
total internal reflection.

## The input beam

Collimated, in the meridional plane, propagating up and to the right at
**23.884815°** above +z. At the plane z = 0 the beam fills **y ∈ [+5.459225, +13.459225]**
— i.e. it is centred on y = +9.459225 and is 8.000 mm wide *measured vertically at that
plane*. (Perpendicular to the beam that width is 8·cos 23.884815° = 7.3157 mm; either
convention is acceptable provided your file states which one it uses.)

## The image plane

A flat detector perpendicular to z at **z = 120.0**, at least ±20 mm in y.

## Required ray path

The beam enters the block, folds four times by total internal reflection, and exits.
The events, in order, are exactly:

| # | face | event | medium after |
|---|------|-------|--------------|
| 1 | **A** | refraction, air → glass | glass |
| 2 | **D** | total internal reflection | glass |
| 3 | **B** | total internal reflection | glass |
| 4 | **A** | total internal reflection | glass |
| 5 | **T** | total internal reflection | glass |
| 6 | **B** | refraction, glass → air | air |
| 7 | detector | — | — |

Note that **face A is used twice** (event 1 refracting, event 4 reflecting) and **face B
is used twice** (event 3 reflecting, event 6 refracting). Both times it is the *same
physical face* — same arc, same position, same radius, same tilt. Your file must encode
that identity, not two independent copies.

## Verified reference trace

The following is the exact traced path of the central ray, computed analytically and
validated against the ray-tracing engine that produced this design to better than
0.001 mm. It is correct; you may rely on it rather than re-deriving it. The surface
normal given is the unit normal at the hit point, oriented against the incoming ray.

| # | face | event | path length to hit (mm) | hit point (z, y) | normal (nz, ny) | incidence (°) | direction after (° from +z) |
|---|------|-------|------|------------------|-----------------|-----------|-------------|
| 1 | A | refract | 37.2073 | (34.0209, +24.5244) | (−0.821538, −0.570154) | 10.876 | +27.535 |
| 2 | D | TIR | 11.8446 | (44.5239, +30.0000) | (0.000000, −1.000000) | 62.465 | −27.535 |
| 3 | B | TIR | 28.4868 | (69.7841, +16.8310) | (−0.936229, −0.351391) | 48.107 | −111.321 |
| 4 | A | TIR | 39.5413 | (55.4074, −20.0041) | (+0.958582, +0.284815) | 52.132 | −35.584 |
| 5 | T | TIR | 17.1783 | (69.3779, −30.0000) | (0.000000, +1.000000) | 54.416 | +35.584 |
| 6 | B | refract | 13.1297 | (80.0559, −22.3599) | (−0.988226, −0.152999) | 26.783 | +51.326 |
| 7 | detector | — | 63.9225 | (120.0000, **+27.5455**) | — | — | — |

Critical angle at n = 1.5 is 41.8103°, so all four reflections clear it (margins 20.66,
6.30, 10.32, 12.61°). Total geometric path, source plane to detector: 211.3105 mm.

## The file must

1. Build the system so that it loads and traces as written.
2. Reproduce all seven steps above in order, with the right event at each: the four TIR
   faces must behave as reflections that keep the ray in glass, and the two refracting
   events must cross the glass/air boundary in the right direction.
3. Place every surface at its true position and orientation, matching the geometry above
   to better than 0.01 mm and 0.01°.
4. Use cylindrical surfaces for A and B, with the correct radius sign for each of their
   appearances in the path.
5. Keep the repeated uses of one physical face genuinely coupled — editing face A's
   radius once must move both of its appearances.
6. Give every surface a clear aperture large enough to pass the whole 8 mm beam and no
   larger than the physical face it sits on.
7. Reproduce the collimated, off-axis, laterally offset input beam and put the aperture
   stop where the beam is defined (z = 0).
8. End with a comment block containing a table of: surface number, physical face, event,
   the surface's global (z, y) vertex and its tilt, and the incidence angle of the
   central ray on it — plus the predicted (z, y) where the central ray lands on the
   detector.

Put any assumption you had to make in a comment block at the top of the file.
