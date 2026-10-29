> **Obstruction audit correction:** `trace_rays` currently ignores back-side detector crossings. Full ray support does not certify an unobstructed layout. All five designs in `experiments/three_mirror_tra_refinement_20261013` fail the sampled opaque-detector-footprint check; see that directory’s `OBSTRUCTION_AUDIT.md`. Do not use these as valid unobstructed benchmark designs.

# Three-mirror adapter for the existing 3D optical engine

`model3d.py` converts saved three-mirror records into 3D layouts and supplies
camera fields, pupils, nearest-event routing, and spot statistics. It reuses
`nslens.spectrometer.model.sag`, `reflect`, and the shared batched all-root
backend `nslens.spectrometer.intersections.intersect`. The original scalar
`physics_audit.exact_intersect` is retained as an independent implementation
for regression checks. These saved layouts require nearest-hit routing, not
just the first Newton root of a prescribed surface sequence.

The shared engine now accepts optional explicit polynomial `terms` on surfaces.
The original nine-term spectrometer path and its default arithmetic remain
unchanged. The adapter uses all 25 monomials of total degree two through six,
including mixed terms and odd powers of either coordinate. Coefficients in
the adapter multiply `(u/30)^i (v/30)^j` and have units mm; `optical_surface`
converts them to physical-coordinate coefficients for the shared engine.

## Initializing saved 2D layouts

`lift_sample(record, sagittal="equal_curvature")` preserves vertices, local
in-plane axes, apertures, the detector, all quadratic u curvature, and existing
final-mirror u³–u⁶ terms. The added v² curvature equals the u² curvature.
`sagittal="cylindrical"` instead sets it to zero. These are explicit starting
assumptions; a 2D record does not determine a 3D prescription. Neither lift is
an optimized three-mirror anastigmat.

Mirrors have circular projected clear apertures; pupils sample a disk. Layouts
may be independently translated and rotated in 3D with `pose_delta`. No
symmetry constraint is imposed on subsequent coefficients or poses. The initial
lift itself has a planar layout and is symmetric under v reflection.

## Numerical baseline

The reproducible 2D population audit is in
`experiments/three_mirror_spot_baseline_20261013`. It measures 5,260 distinct
saved records, excluding the separate bulk NPZ populations. Within each model
and specification it retains the best ceil(5%) of full-support, non-crossing
records by worst-field centroid-referenced RMS. Up to five representatives are
selected by farthest-point distance in physical positions, surface normals,
detector frame, and aperture sizes, starting with the best spot. This is a
geometric spread heuristic, not a proof of distinct topology families.

`experiments/three_mirror_3d_baseline_20261013` evaluates those representatives
without optimization. It uses 25 field points and 197 pupil rays per field.
The provisional vertical half field is 0.75 times the original horizontal
half field. These are 2D-selected candidates evaluated in 3D, not the top 5%
of a 3D search. Each record also verifies its original 15×101 meridional slice.

## Metrics and boundaries

Radial RMS is the square root of the sum of x and y variances around each field
centroid. Mapping RMS is reported separately. Each launched ray contributes to
the support denominator; spot statistics use received rays and must be read
with support. Good rays encounter each of the three mirrors once and then the
forward-facing infinite detector. The actual route is returned for future
fixed-route Jacobian checks.

Zero-thickness surfaces, no diffraction, no radiometry, no mechanical clearance
certification. A new 3D surface-overlap audit is still needed. There is no new
LM/GTRA refiner or verified parameter Jacobian in this adapter yet.

Validation: planar agreement for legacy/current encodings and freeform terms;
arbitrary 3D rigid-motion invariance; general polynomial intersections against
a scalar polynomial oracle; analytic normals against numerical tangents;
finite aperture, nearest-root, wrong-route rejection, and ray-count checks.
The existing spectrometer physics/fold tests also pass.

## Fast intersection backend

The default three-mirror backend now uses analytic linear/quadratic solves and
batched companion-matrix eigenvalues for higher-order surfaces. Rays are grouped
by actual polynomial degree, and only mathematically zero leading coefficients
are removed. Candidate roots are screened for real/forward/finite-aperture hits
before sag substitution; the nearest accepted root is selected. Sag coordinate
powers are cached for the explicit polynomial basis. The original nine-term
spectrometer sag path is unchanged.

Root solving is chunked at 4,096 rays by default. This is NumPy CPU float64,
not a GPU implementation, automatic differentiation backend, or a claim of
state-of-the-art performance against external tracers. The shared Newton solver
remains the default for the older spectrometer application; this change enables
the fast all-root backend for the new three-mirror work.

To compare the scalar reference through exactly the same scene traversal:

```python
from nslens.spectrometer.physics_audit import exact_intersect
reference = trace_rays(layout, origins, directions, intersect_fn=exact_intersect)
fast = trace_rays(layout, origins, directions)
```

Matched timings, individual repeats, code hashes, and comparisons against all
41 saved 3D baselines are in `experiments/three_mirror_tracer_speed_20261013`.
Timing scripts use one BLAS thread, fixed inputs, a warm-up for each backend,
and three interleaved timing repeats. Freeform case complexity is reported
separately; quadratic timings are not advertised as general freeform speed.

## Finite opaque detector support

When `layout["detector"]["bounds_mm"]` is supplied as `[[umin,vmin],[umax,vmax]]`, the finite rectangle absorbs from both sides; only forward-side hits can record image coordinates. Without bounds the legacy infinite forward-only detector behavior remains for diagnostic compatibility and must not certify an unobstructed layout. The corrected three-design run is in `experiments/three_mirror_unobstructed_refinement_20261013/RESULTS.md`.
