# SINGLE-SHOT BENCHMARK PROMPT — copy everything below the line into the agent under test

---

## Task

Design a **folded focusing prism**: one solid block of glass in which collimated light
from infinity is turned by at least one **total internal reflection** and brought to a
focus on a detector. This is the workhorse element of AR/VR and folded-camera optics —
the fold buys you optical path in a thin package.

You are designing inside an existing 2-D non-sequential ray-tracing environment. Your
entire deliverable is **nine numbers**: the parameters of the element and of the incoming
beam. A grader traces them through the real engine and checks them against the spec below.

**This is a commit-then-verify benchmark. Read the Protocol section before you start —
it caps how many times you may run anything, and the caps are part of the test.**

---

## The environment

### The element

One glass block, uncoated, refractive index `n`, cross-section a closed 6-vertex contour
in the plane. Two faces are flat and horizontal; two are circular arcs. Given your
parameters, let

```
lx = cx - thick/2          rx = cx + thick/2
dA = tan(tilt_a) * 30      dB = tan(tilt_b) * 30        (tilts in degrees)
```

the contour is, in order:

| vertex | position | role |
|--------|----------|------|
| P1 | `(lx + dA, -30)` | bottom end of face A |
| P2 | `(rx + dB, -30)` | bottom end of face B |
| P3 | `(rx + bsag, 0)` | **arc control point** of face B |
| P4 | `(rx - dB, +30)` | top end of face B |
| P5 | `(lx - dA, +30)` | top end of face A |
| P6 | `(lx - fsag, 0)` | **arc control point** of face A |

giving four faces:

| face | label in the event log | shape |
|------|----------------------|-------|
| left optical face | **A** | circular arc through P5, P6, P1 |
| right optical face | **B** | circular arc through P2, P3, P4 |
| top flat edge, y = +30 | **D** | straight, P4 → P5 |
| bottom flat edge, y = −30 | **T** | straight, P1 → P2 |

Note the labelling: **D is the top edge and T is the bottom edge.** In the event log a
vertex is labelled `D` if y ≥ +30, `T` if y ≤ −30, otherwise `A` if x < cx and `B` if
x ≥ cx.

`fsag > 0` pushes face A's arc control point to the left, i.e. face A bulges **outward**
(convex to the incoming light); `bsag > 0` makes face B bulge outward to the right. The
arc through the three points is a true circular arc — for a face whose two endpoints are
60 mm apart the sag `s` and the radius are related by `R = (h² + s²) / 2s` with `h` the
half-chord, so small sags mean nearly flat faces and large sags mean strong curvature.
The tilts shear the two optical faces: positive `tilt_a` moves face A's bottom end in +x
and its top end in −x.

### The beam — light from infinity

**21 parallel rays**, hence a single field point at infinity. They start on the plane
**x = 0** at

```
y = ray_y + linspace(-4.0, +4.0, 21)        (an 8.000 mm wide beam)
```

each travelling in direction `(cos(ang), sin(ang))`. Every ray is emitted at brightness
1.0 (total input energy = 21, and all reported shares are normalised by it).

### The detector

A flat detector on the plane **x = 120**, spanning y ∈ [−120, +120].

### The physics

Fully non-sequential and honest. Surfaces are **uncoated**: every glass interface splits
energy by the Fresnel equations, so the only loss-free reflection available to you is
**true geometric total internal reflection**. There are no mirrors and no coatings, and
you cannot add any. Critical angle is `asin(1/n)` — 41.81° at n = 1.5. Ray depth budget
is 16 interactions.

### Your nine parameters and their hard bounds

| parameter | meaning | min | max |
|-----------|---------|-----|-----|
| `ray_y`  | y-centre of the incoming beam at x = 0 | −20 | +20 |
| `ang`    | beam direction, degrees from +x | −55 | +55 |
| `cx`     | x-centre of the element | 35 | 85 |
| `thick`  | axial thickness (rx − lx) | 6 | 45 |
| `fsag`   | sag of face A (+ = bulges left/outward) | −18 | +18 |
| `bsag`   | sag of face B (+ = bulges right/outward) | −18 | +18 |
| `tilt_a` | shear of face A, degrees | −25 | +25 |
| `tilt_b` | shear of face B, degrees | −25 | +25 |
| `n`      | refractive index of the glass | 1.50 | 2.00 |

---

## Required behaviour — the spec

The graded quantities are all measured on the **dominant path**: the interaction sequence
that carries the most energy to the detector.

1. **Fold.** The dominant path contains **at least one genuine TIR**.
2. **Entry and exit.** The dominant path **begins** with a refraction into the glass and
   **ends** with a refraction back out into air.
3. **No cheating the reflection.** No `partial_reflect`, `mirror` or `beam_split` event
   anywhere on the dominant path — the fold must be real TIR.
4. **The fold must do work.** The beam's total direction change from input to output must
   be **≥ 20°**. A grazing TIR that barely bends the beam does not count as a fold.
5. **The whole beam behaves as one.** At least **20 of the 21 rays** must follow the same
   dominant path.
6. **Throughput.** Energy delivered by the dominant path must be **≥ 0.80** of emitted.
   (Uncoated Fresnel losses at two glass/air crossings put the ceiling near 0.92 at
   n = 1.5 and lower at higher index, so this is tight but reachable — a lossy fold is not.)
7. **Focus — the primary requirement.** The RMS spot radius of the beam on the detector
   at x = 120 must be **≤ 1.0 mm**, down from the 8 mm input beam.
8. **The focus must land on the detector.** The plane of best focus must be within
   **10 mm** of x = 120.
9. **Source in air.** The glass must not reach the source plane: the element's leftmost
   point must be at **x ≥ 3.0**. A design whose glass encloses the source at x = 0 is
   rejected — the light has to come from outside the prism.
10. **Detector in air**, outside the glass, and the element must be non-degenerate (its
    faces must not cross). All nine parameters must lie inside the bounds above.

Ranking among passing designs is by RMS spot first, then throughput.

---

## Protocol — read this carefully, it is enforced

**Phase 1 — commit (no engine access).**
Work out the nine numbers by reasoning about the geometry and the physics. In this phase
you **may** write and run your own pure-geometry code — your own ray tracer, a solve for
where the focus lands, anything — provided it does **not** import or invoke the project's
ray-tracing engine. Keep it to a single scratch file and a handful of executions.

You may **not**, in this phase or any other:
- import `PersistentRayOpticsRunner`, `optical_rl.*`, or run anything under `ray-optics/`;
- run the verification script;
- read `ns_probe/results_fold_qd/`, `ns_probe/results_arvr*/`, or any `archive.json` —
  those contain solved designs and reading them voids the result;
- read anything in `ns_probe/benchmarks/fold1_env/` other than this prompt. You will
  **run** the verifier from that directory in Phase 2, which is required — but do not open
  it or any file beside it, as they contain the grading key;
- write a search or optimiser over the environment, or sweep parameters through it.

You do not need to read any project source file; this prompt is complete. If you do open
`ns_probe/fold_demo.py` for the element definition, **ignore its `DEMO` constant** — that
is a different, non-focusing design and submitting it fails the spec.

End Phase 1 by writing `design.json`:

```json
{"ray_y": 0.0, "ang": 0.0, "cx": 0.0, "thick": 0.0, "fsag": 0.0,
 "bsag": 0.0, "tilt_a": 0.0, "tilt_b": 0.0, "n": 0.0}
```

and stating your **predictions** — commit to these numbers before you verify:
the dominant path in `T@A -> TIR@D -> T@B` notation; the incidence angle at each TIR and
the critical angle; the y-centroid of the spot on the detector; the RMS spot radius; and
the throughput share.

**Phase 2 — verify (exactly one run).**

```
python3 ns_probe/benchmarks/fold1_env/verify_fold1.py design.json --render spot.png
```

Run it **once**. It prints the traced path and a PASS/FAIL line for every requirement.

**Phase 3 — one correction, then stop.**
If the verdict is FAIL you may revise `design.json` **once** and re-run the verifier
**once**. Then stop, whatever the outcome. Maximum two verifier runs for the whole task.

Do not ask clarifying questions, do not propose alternatives, and do not iterate beyond
the single correction allowed above. Running out of allowed runs with a failing design is
a legitimate outcome to report — quietly continuing to iterate is not.

---

## Deliverable

Leave `design.json` on disk with your final nine numbers, and reply with:

1. **Reasoning** — how you placed the fold and set the focus, in ≤ 300 words.
2. **Predicted vs measured** — a table of your Phase 1 predictions against the verifier's
   numbers, with the discrepancy on each. This is graded: a design that passes while your
   predictions were wrong tells us less than one that passes as predicted.
3. **Verdict** — the verifier's final PASS/FAIL, the RMS spot, the throughput, and which
   checks (if any) failed.
