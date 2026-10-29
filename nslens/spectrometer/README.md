# Architecture-conditioned spectrometer discovery prototype

This domain adds a NumPy 3D optical forward model and reuses the repository's
`nslens.search.cma_me.CMAES` implementation. The existing JavaScript ray-optics
engine remains 2D; this is a new experimental domain, not evidence that that
engine already models a 3D spectrometer.

The supplied Reimers paper motivates a concave primary → convex reflection
grating → concave tertiary architecture. No published radii, spacings, tilts,
or freeform coefficients initialise the search. Chief-ray scaffolding ensures
a central route, while CMA discovers three meridional fold angles, four lengths,
six vertex curvatures and two detector tilts. A second stage unlocks seven
additional polynomial terms on each surface, for 36 parameters in total.

The selected baseline uses a 10 mm slit, 500–1100 nm, nominal F/3.8 input cone,
300 lines/mm grating and a common planar detector. Desired inverse dispersion
is 100 nm/mm. Unit slit magnification is an explicit experimental requirement.
Fixed circular surface semi-apertures are 18 mm. This is a baseline discovery
experiment, not a claim to reproduce the compact paper prescription.

## Model

Distances are millimetres; wavelengths passed into the model are millimetres.
Local sag terms are x², y², x³, xy², x⁴, x²y², y⁴, x⁵, x³y². Slit reflection
symmetry is enforced, while the dispersion direction can be asymmetric.
The grating phase in cycles is `density * local_x`; its surface-projected
phase gradient enters the vector grating law without renormalisation.
Thus groove density is uniform in the projected local coordinate, not in
surface arc length. Manufacturing/efficiency of this grating is unvalidated.

Every launched ray remains in the throughput denominator. The forward tracer
checks unintended surfaces on each segment. Its Newton solver is local and
can miss alternative roots; `physics_audit.py` provides a separate polynomial
root check. Neither method establishes mechanical substrate clearance.

Spot sizes are geometric radial RMS about each field/wavelength centroid.
Wavefront error is geometric path plus grating phase-length, followed by a
reference sphere to that centroid; only piston is removed. No diffraction PSF,
coating, groove efficiency, scattering, polarisation, tolerancing, substrate,
finite slit width or detector pixel model is included. Throughput is geometric
ray survival only. Sag/slope/curvature and volume diagnostics sample the fixed
apertures; sample gaps are not certified clearance. Vertex curvature signs are
bounded, but higher-order freeforms may change principal-curvature signs away
from the vertex and must be audited separately.

## Reproduce

```
.venv/bin/python -m unittest discover -s tests -p test_spectrometer_physics.py -v
.venv/bin/python -m nslens.spectrometer.search --out experiments/my_spectrometer --seeds 101 102 103
.venv/bin/python -m nslens.spectrometer.plot experiments/my_spectrometer/seed_102/result.json
```

`search.py` performs scalar CMA with retained-candidate restarts, not a CMA-ME
comparison. Denser evaluation uses the same model, and must not be described
as an independent optical package validation. Consult `BENCHMARK.md` and the
experiment report for exact budgets, acceptance gates, stages and outcomes.

The first completed runs are saved under `experiments/spectrometer_pilot_v1`,
`experiments/spectrometer_hybrid_v1` and `experiments/spectrometer_quality_v1`.
The second and third stages use bounded SciPy least-squares after CMA; they
are not additional CMA results. The quality stage downweights exact centroid
mapping while retaining a strong spectral-span residual. Its saved source
snapshots and configuration identify the precise merit used. The initial quality plot
is `experiments/spectrometer_quality_v1/seed_102/candidate.png`; that variant
fails the later curvature gate. The corrected constrained experiment is
`experiments/spectrometer_constrained_v1`, with the accepted result and plot
identified in the final experiment report.

The best initial candidate improves substantially but is not diffraction
limited or an established practical instrument. Read the experiment report
and independent root/pupil audits before interpreting spot or volume scores.

## Physical-aperture curvature gate

Vertex signs alone do not enforce the architecture once freeform terms are
released. The first unconstrained quality variant fails convexity on the
illuminated grating and is retained only as an invalid diagnostic. The
corrected run fixes physical aperture radii to the preceding valid hybrid
candidate's independently audited footprint plus a 1 mm radial margin; these
are physical masks, not a post-hoc throughput exclusion. Every launched ray
must still survive the complete route.

`curvature_violations` evaluates signed sag-Hessian eigenvalues on nine radial
rings and 48 azimuths including each aperture rim. Hessian definiteness gives
the same curvature signs as the graph's principal curvatures. The merit uses
a positive 2e-5/mm margin; acceptance additionally requires a separate denser
principal-curvature audit over the actual physical disks. Soft penalties
alone do not establish acceptance. Mechanical and fabrication validation
remain separate work.

The final artifact uses an aperture-only correction after the constrained
optimisation: physical radii 8.90, 8.43 and 12.11 mm, with the optical gene
vector unchanged. It is saved separately in `experiments/spectrometer_final_v1`.
This correction is independently checked for pupil support and curvature;
it does not constitute a new optimisation result. The smallest audited
ray-to-aperture margin is approximately 0.05 mm at the grating, so assembly
and fabrication tolerances remain a significant unresolved qualification.
