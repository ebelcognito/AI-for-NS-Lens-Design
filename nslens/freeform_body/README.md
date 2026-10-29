# Stage 1: continuous-boundary freeform optical discovery

This package is **not a waveguide-design model**. It represents one homogeneous
freeform optical element in air. Rays may refract, Fresnel-reflect, revisit the
surface, or undergo TIR, but no guiding path or fold is prescribed.

The package deliberately does not reuse the legacy four-face `Element` genotype.
Its boundary is one periodic C2 spline with no corners and no optical face labels.
Interactions are recorded by normalized boundary arclength.

## Stage-1 scope

Included:

- continuous closed freeform geometry;
- a multiresolution Fourier-to-spline genotype;
- hard material-index enforcement, `1.50 <= n <= 1.70`;
- nearest-hit non-sequential tracing;
- full uncoated Fresnel branching and TIR;
- shape and interaction descriptors for diversity;
- a joint route/shape archive with route-uniform parent selection;
- a discovery-search driver interface (defined but not run here);
- the fixed-eyebox three-field AR/VR relay objective (`relay.py`);
- deterministic geometry and physics validation tests.

Explicitly deferred:

- compaction;
- a learned compaction policy or dataset;
- optimization/search runs;
- 3-D lifting and CODE V translation.

`FreeformBodyCodec.n` is part of the problem configuration, not the genotype. A
search therefore cannot improve a difficult candidate merely by raising its index.

## The AR/VR relay objective

`FreeformRelayProblem` states the existing three-field AR/VR relay in the **reverse**
direction. Ray paths are reciprocal, so a body that takes three microdisplay points to
three collimated eyebox beams is the same body that takes three collimated eyebox
fields to three focused display-plane spots. Stating it in reverse makes the eyebox the
*entrance* aperture, so every field fills the fixed box by construction rather than
scoring off one ray through a thin slice of it, and makes image quality a spot size
again.

The geometry is fixed and matches `nslens/qd/arvr.py`: eyebox at x = 120 with half
height 9, microdisplay plane at y = 50, three targets at x = 49/55/61, three input
fields at -6/0/+6 degrees. Fixing the fields means **field of view is a specification
here, not a discovered descriptor** -- the axis it occupied in the forward prism
archive is free for another shape or route descriptor.

Two properties are worth keeping in mind when editing it:

- **Reflection parity is not prescribed.** Whether field angle maps to increasing or
  decreasing display x depends on the number of reflections. Both pairings are scored
  and the better is kept, so pinning one would silently exclude every design of the
  opposite parity.
- **Ordering needs no gate.** The target windows are required to be disjoint, so a
  design cannot satisfy every field and still scramble the image.

## The feasibility ladder

A three-field product behind a throughput floor is zero for almost every random body,
which leaves the emitter nothing to climb. `relay_field_report` grades a field by how
far its light got: never touched the body, entered but never escaped, escaped but never
reached the image plane, reached it but missed the window, landed in the window. Only
the top band produces an optical score, and the archive admits only top-band designs --
the ladder shapes the *search*, never what counts as a discovered design.

The banding follows the two-stage freeform-prism GA, but that method keys its bands to
named surfaces (the second hit on A, mirror B, surface C), which would reintroduce a
prescribed path. These bands are keyed to outcomes instead, so they name no face.

`FreeformDiscoveryProblem` supplies the outcome-only optical fitness.
`DiverseDiscoveryArchive` keeps one elite per joint route/shape niche and samples
route families uniformly. `FreeformDiscoverySearch` connects these contracts to the
existing CMA emitter; importing or constructing it does not start a run. Its
`run(max_evaluations=...)` allowance is a hard evaluation cap, including when the
remaining allowance is smaller than one CMA population.
