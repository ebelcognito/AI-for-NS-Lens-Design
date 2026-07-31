# Answer key — single-fold focusing prism benchmark (fold_demo env)

Companion to `benchmark_fold1_prompt.md`. **Do not paste this into the model under test.**
Built and verified 2026-07-29 against the live engine.

## The environment being tested

`ns_probe/fold_demo.py` — 9-DOF single glass element, 21-ray collimated beam from x = 0,
detector at x = 120, uncoated Fresnel + TIR physics on the Node ray-optics engine.
The task asks for the **honest single-fold focusing** family, which the original fold-QD
run never optimised for (it maximised focused *throughput* in a ±8 mm window, not spot
size), so the archive contains no ready answer to this spec.

## Reference solution

Found by random seeding + 3-stage CMA-ES over the 9 parameters, minimising RMS spot
subject to throughput ≥ 0.85 (~10k engine evaluations, ~4 min). **Not unique** — this is a
broad family, not a needle.

```json
{"ray_y": -18.65635660138779, "ang": 37.483939982719576,
 "cx": 63.993227745392275, "thick": 41.69415522629459,
 "fsag": 9.34758117693124, "bsag": 0.7439142577369253,
 "tilt_a": 20.4858703051664, "tilt_b": -14.223046072754633,
 "n": 1.6571399294573437}
```

Verified: **13/13 checks PASS.**

| quantity | value |
|---|---|
| dominant path | `T@A -> TIR@D -> T@B` (1 TIR, faces A, D, B) |
| RMS spot at x = 120 | **0.0008 mm** (full span 0.0033 mm, from an 8 mm input beam) |
| best focus | x = 120.00 → defocus **0.00 mm** |
| throughput share | **0.8800** of emitted |
| beam deviation | 37.484° in → −37.201° out = **74.69°** |
| glass spans | x = 31.94 … 92.44 (source plane clear by 31.9 mm) |
| rays on path | 21 / 21 |

Traced path of the central ray:

| event | face | x | y | E in | E out |
|---|---|---|---|---|---|
| refract | A | 32.444 | +5.024 | 1.000000 | 0.938064 |
| tir | D | 80.721 | +30.000 | 0.938064 | 0.938064 |
| refract | B | 91.301 | +24.526 | 0.938064 | 0.880043 |
| sensor | — | 120.000 | +2.105 | 0.880043 | 0.880043 |

The beam enters low on the left face travelling up at 37.5°, crosses the block, folds once
off the flat top edge, and exits the right face heading down to a point focus exactly on
the detector. Throughput 0.880 is the two-surface Fresnel ceiling at n = 1.657 — the fold
itself is loss-free, as TIR must be.

Regenerate the render with:
`python3 ns_probe/benchmarks/fold1_env/verify_fold1.py design.json --render spot.png`

## How hard is the spec? (measured, not guessed)

Single-parameter drift from the reference that still keeps **all 13** checks passing:

| param | value | may drift + | may drift − |
|---|---|---|---|
| `ray_y` | −18.656 | +20.56 | to the bound |
| `ang` | 37.484 | +7.14° | −4.01° |
| `cx` | 63.993 | +11.00 | −9.52 |
| `thick` | 41.694 | to the bound | −16.24 |
| `fsag` | 9.348 | +2.38 | −1.65 |
| `bsag` | 0.744 | +2.50 | −2.39 |
| `tilt_a` | 20.486 | to the bound | −9.30 |
| `tilt_b` | −14.223 | +7.22 | to the bound |
| `n` | 1.657 | +0.24 | −0.13 |

So the pass region is **forgiving in every direction** — roughly ±20% on the strongest
curvature (`fsag`), several degrees on the input angle, ±10 mm on element placement. A
submission that has the physics broadly right should pass; a submission that fails did not
get the physics broadly right. That is the intended discrimination.

**Check 12 (defocus ≤ 10 mm) is the sharpest constraint, not a redundant one.** Measured
counter-examples that pass the RMS bar but fail defocus: `fsag` = 5.2 gives RMS 0.98 mm
with true focus 35 mm past the detector; `fsag` = 13.5 gives RMS 0.89 mm with 15.5 mm of
defocus; `cx` = 80 gives RMS 0.59 mm with 14.3 mm. These are barely-converging beams that
merely *look* small at x = 120. Grade check 12 seriously.

## What this benchmark measures, ranked by discriminating power

1. **Does it realise the fold must be TIR, and reason about the critical angle?** At a
   flat horizontal edge, TIR needs the internal ray within 48.2° of horizontal (n = 1.5).
   A model that tries to bend the beam 90° off a flat edge, or that reaches for a mirror
   or a coating, has missed the physics — coatings are not available.
2. **Does it put the focus on the detector rather than merely somewhere?** This requires
   combining the two curved faces' power with the *folded* path length, and it is where a
   textbook thin-lens calculation goes wrong: the beam hits face A at ~37° incidence, so
   the effective meridional power is far from the paraxial lensmaker value. Expect
   paraxial-only designs to land tens of mm out of focus.
3. **Sag-to-radius conversion.** Faces are specified by a sag, not a radius:
   `R = (h² + s²)/2s` with h the half-chord (~30 mm). A model that treats `fsag` as a
   radius will be off by an order of magnitude.
4. **Does it keep the source outside the glass?** The bounds permit `cx` = 35, `thick` = 45,
   `fsag` = 18 → glass reaching x = −5.5, which swallows the source plane at x = 0. Every
   `TIR@D -> T@B` cell in the fold-QD archive is exactly this degenerate case (it scores
   0.96 throughput by starting *inside* the glass and skipping the entry surface). Check 3
   and check 6 exist to kill it.
5. **Prediction calibration.** The Phase 1 commit is graded against the measurement. A
   model that passes but predicted the wrong path, the wrong centroid, or a wildly wrong
   spot has fitted numbers rather than understood the system.
6. **Protocol compliance.** Two verifier runs maximum. Writing a search loop over the
   engine, or quietly iterating past the cap, is a fail regardless of the optics — that
   behaviour is the whole reason this variant exists.

## Known-bad submissions and what they score

Both were run through the verifier as calibration:

| submission | result | fails on |
|---|---|---|
| `fold_demo.DEMO` (the baked-in constant, reused lazily) | **10/13 FAIL** | 11 (RMS 2.02 mm), 12 (defocus 23.5 mm), 13 (deviation only 4.78°) |
| fold-QD archive cell (1,14) (embedded-source cheat) | **10/13 FAIL** | 3 (glass_left = −0.83), 6 (path starts with TIR, never enters), 12 (defocus 10.25 mm) |

`DEMO` is the instructive one: it is a genuine 2-TIR fold with 0.868 throughput that a
model could plausibly copy from `fold_demo.py`, and it fails precisely on **focus** — it
was never designed to focus. Any submission scoring ~10/13 with those three checks red is
almost certainly copied rather than designed.

## Quality tiers for passing designs

| tier | RMS spot |
|---|---|
| excellent | ≤ 0.05 mm |
| good | ≤ 0.25 mm |
| pass | ≤ 1.00 mm |

Rank by RMS first, then throughput (ceiling ≈ 0.92 at n = 1.5, falling with index — a
high-index design pays for its stronger TIR with more Fresnel loss, which is a real
trade-off worth noting if a model reasons about it explicitly).

## Harder variants, once this saturates

- **Zero engine access**: delete Phase 2/3, grade the committed numbers directly.
- **Two folds**: require `tir_count ≥ 2` and `thick ≤ 20` (a genuinely thin folded
  package). The fold-QD archive shows 2-TIR designs reaching 0.96 focused throughput, so
  it is satisfiable, but the focus requirement on top of it is unexplored.
- **Tighten** `MAX_DEFOCUS` to 2 mm and `MAX_RMS` to 0.1 mm — the reference clears both by
  three orders of magnitude, so headroom exists.
