# Flat automotive thermal-camera reconstruction

Current workflow: **20-variable base → 41-variable spectrometer-style XY freeform**;
see the final section below. Earlier radial and degree-by-degree experiments are
retained for reproducibility.

This domain runs a computational reconstruction pilot for Fraunhofer IOF's
2024 2 x 4 channel thermal camera. See [BRIEF.md](BRIEF.md) for public sources
and the distinction between announced F/1.1 and the later paper's effective
F/1.4. No numerical Fraunhofer prescription was available.

The implementation reuses the repository's full-covariance
`nslens.search.cma_me.CMAES`, following the spectrometer's staged search pattern.
It adds an independent NumPy 3D optical domain; the existing JavaScript 2D
engine is not used for these camera traces.

## Model

The original baseline channel is an axisymmetric, two-mirror Schmidt-Cassegrain family:

1. A planar entrance and radial-aspheric exit form a finite refractive corrector.
2. Rays that miss the front secondary reach an annular concave primary.
3. The primary reflects back to the convex secondary.
4. The secondary sends the beam through the primary's physical central hole.
5. A planar image sensor receives the image.

The base stage has eight normalized variables: primary and secondary spacing,
independent mirror curvature perturbations, secondary radius, primary hole,
corrector thickness, and back focus. Six more variables release primary and
secondary conics, three corrector radial terms (r^4, r^6, r^8), and a secondary
r^4 term. This baseline is aspheric, not a general XY freeform model; the extension below adds actual XY freedom.
The mirror curvatures start from a paraxial focal-length scaffold with random
perturbations. Neither the topology nor its analytical construction is claimed
as a discovery by CMA.

An assumed 12 micrometre pixel and 16 pixels/degree set a target focal length
of 11.00079 mm and an F/1.1 entrance diameter of 10.00072 mm. Silicon is modeled
with constant index 3.42. Wavelength labels do not change geometric rays;
8–14 micrometre chromatic and radiometric performance remain unvalidated.

Equal-area pupil quadrature retains all launched rays in the denominator.
Every field traces both transverse dimensions. Loss considers geometric spot,
image scale, lost pupil support, aperture overflow and sampled surface gaps.
Local Newton intersections are checked separately by polynomial roots on a
sample of the frozen candidate's surviving paths. The root audit is independent
of the intersection algorithm, not an independent optical design package.

## Eight-channel audit

`array_audit.py` constructs actual finite planar steering wedges. Vector Snell
refraction through both prism interfaces steers the central viewing direction
of each channel into a common +z optical axis. Rays are aimed onto the corrector
stop and clipped by the finite prism, corrector, mirrors, hole and assumed
160 x 120 pixel sensor. Channel centers tile 34 x 13 degrees. The prism model
is our reconstruction choice; it is not a recovered Fraunhofer prescription.

The audit reports spots both before and after sensor clipping, pupil survival,
angular sampling, affine mapping residual and optical depth including the prism.
Pupil survival is geometrical, not coating/absorption-weighted efficiency.
The area-equivalent f-number is a nominal diagnostic and does not establish
agreement with Fraunhofer's unspecified effective-f-number convention.

Pilot gates require sampled fields to receive rays, at least 50% pupil support
per field, geometric radial RMS no larger than one assumed pixel before sensor
clipping, optical depth <=10 mm, and fitted angular sampling within 10% of
16 pixels/degree. These are explicit pilot gates, not a claim about the original
camera's MTF requirements. Diffraction, central-obscuration MTF, coatings,
absorption, thermal defocus, tolerances, finite-distance parallax, crosstalk,
wafer process constraints, mechanical supports and packaged thickness remain
outside this pilot. Do not infer a stitched 530 x 210 resolved image from the
field coverage alone.

## Run

```bash
.venv/bin/python -m unittest tests.test_thermal_camera_physics -v
.venv/bin/python -m nslens.thermal_camera.search \
  --out experiments/my_thermal_search --seeds 101 102 103 \
  --base-evals 1200 --freeform-evals 3600 --fields 3 --waves 1 --pupils 5 \
  --audit-fields 7 --audit-waves 3 --audit-pupils 13 --workers 3
.venv/bin/python -m nslens.thermal_camera.array_audit experiments/my_thermal_search/seed_103/result.json
.venv/bin/python -m nslens.thermal_camera.root_audit experiments/my_thermal_search/seed_103/result.json
.venv/bin/python -m nslens.thermal_camera.plot experiments/my_thermal_search/seed_103/result.json
```

Select a seed by its training merit, not by inspecting the denser audit. In the
first completed run, seed 103 has the lowest training loss. Output directories
must be new. Checkpoints retain genes, CMA state, last population and history;
result files include decoded surfaces and dense audits. Snapshots preserve the
model/search source used in each run. Later edits to logging do not rewrite
previous experiment artifacts.

An optional second experiment uses the first candidate as a warm start and
increases the penalties for lost pupil support and worst-field spot:

```bash
.venv/bin/python -m nslens.thermal_camera.refine \
  --source experiments/my_thermal_search/seed_103/result.json \
  --out experiments/my_thermal_refine --budget 4800 --seed 201
```

This refinement is also CMA-only. It is a follow-up experiment with a changed
merit and denser training pupil, not an independent discovery replication.
The input evaluation and dense audits are separate from the stated CMA budget.

First artifacts: `experiments/thermal_camera_pilot_v1/` and
`experiments/thermal_camera_refine_v1/`. Their measured results and remaining
failures are recorded in `experiments/thermal_camera_pilot_v1/REPORT.md`.

## Actual XY freeforms: coarse to fine

`freeform_model.py` adds independent normalized Cartesian monomials to the
primary, secondary and corrector exit surfaces. It is not restricted to radial
symmetry. The existing 14 radial/geometry genes remain active throughout:

| Stage | Newly released terms per surface | Total active genes | Training fields | Pupil rays/field |
| --- | --- | ---: | ---: | ---: |
| Quadratic | x², xy, y² | 23 | 3×3 | 64 |
| Cubic | x³, x²y, xy², y³ | 35 | 3×3 | 144 |
| Quartic | x⁴, x³y, x²y², xy³, y⁴ | 50 | 5×5 | 144 |

New coefficients start at zero, preserving inherited surfaces exactly. Each
stage restarts CMA with a new covariance dimension and reevaluates the source
and inherited incumbent at the new sampling. This is staged polynomial degree
refinement, not a local adaptive mesh or a topology search. Analytic XY slopes
feed 3D reflection/refraction. Sampled gaps and slope penalties screen geometry;
Hessian values are diagnostic, not an imposed convexity restriction. Geometric
sampling does not certify the entire continuous aperture.

```bash
.venv/bin/python -m unittest tests.test_thermal_camera_physics tests.test_thermal_camera_freeform -v
.venv/bin/python -m nslens.thermal_camera.freeform_search \
  --source experiments/thermal_camera_pilot_v1/seed_103/result.json \
  --out experiments/my_xy_freeform --seed 301 --stage-evals 1200
.venv/bin/python -m nslens.thermal_camera.freeform_search \
  --source experiments/thermal_camera_pilot_v1/seed_103/result.json \
  --out experiments/my_radial_control --seed 301 --stage-evals 1200 --radial-control
.venv/bin/python -m nslens.thermal_camera.freeform_plot \
  experiments/my_xy_freeform/result.json --control experiments/my_radial_control/result.json
```

The control uses the same merit, source, stage sampling and evaluation budget,
but only 14 active genes. Source and frozen-stage candidates are compared on a
fixed 7×7-field, 676-ray-per-field audit. These audits do not select candidates.
A single seed and small budget cannot establish which representation is better.
The camera topology, material and detector assumptions remain the same as the
baseline: this is a computational reconstruction experiment, not recovery of a
published proprietary surface prescription.


## Spectrometer-style two-stage workflow (current)

The requested workflow is implemented in `spectrometer_style_model.py` and
`two_stage_search.py`. The degree-by-degree search above remains available only
for reproducing the earlier experiments. The current basis matches
`nslens.spectrometer.model.TERMS` exactly:

`x², y², x³, xy², x⁴, x²y², y⁴, x⁵, x³y²`.

The camera retains its 14-variable radial/geometry carrier. Six independent
quadratic x/y coefficients (two on each mirror and corrector) give a **20-variable
base stage**. All seven higher terms per surface are then released together,
giving a **41-variable full-freeform stage**. The base variables stay active.
New coefficients initialize at zero, preserving the base geometry and ray trace.
This deliberately uses the spectrometer’s even-in-y polynomial symmetry, not
every monomial of degree five. The camera topology and coaxial surface frames
remain supplied; unlike the spectrometer, mirror tilts are not search variables.

Both stages use identical 5×5 field sampling and 196 equal-area pupil rays per
field. Source and both frozen stages receive a common 7×7-field / 676-ray audit.
The initial camera radial candidate is a warm start; this is not discovery from
an unoptimized random prescription. The separate optional least-squares stage below can follow this CMA run.

```bash
.venv/bin/python -m nslens.thermal_camera.two_stage_search \
  --source experiments/thermal_camera_pilot_v1/seed_103/result.json \
  --out experiments/my_two_stage_camera --seed 401 \
  --base-evals 1200 --freeform-evals 2400 --sigma .006
.venv/bin/python -m nslens.thermal_camera.array_audit experiments/my_two_stage_camera/result.json
.venv/bin/python -m nslens.thermal_camera.root_audit experiments/my_two_stage_camera/result.json
.venv/bin/python -m nslens.thermal_camera.freeform_plot experiments/my_two_stage_camera/result.json
```

Add `--radial-control` with a different output directory for an equal-budget,
identical-merit control retaining only the original 14 variables. Saved results
identify the basis in `representation`; never infer the polynomial ordering from
parameter count alone. The initial full experiment is in
`experiments/thermal_camera_two_stage_v1/`.



## Optional post-CMA least-squares polishing

`polish.py` runs bounded SciPy trust-region least squares on all 41 variables.
It preserves the CMA source and logs a separate hybrid merit and all residual
calls, including numerical Jacobian probes (which SciPy nfev does not count).
Only the returned solver iterate or the source can be accepted, using training
residual alone. Hard clipping remains nonsmooth; solver termination is not a
guarantee of global or full-image-quality convergence.

Image residuals retain every initially transmitted training ray even if its
route later becomes invalid. Central-obstruction ghost rays have no image
quality target; all launched rays still enter ray-loss/support and aperture
penalties. New transmitted rays also receive image errors. The fixed residual
length permits finite-difference least squares without rewarding a shrinking
reference pupil. Dense audits remain separate and can expose support changes.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.thermal_camera.polish \
  --source experiments/thermal_camera_two_stage_v1/result.json \
  --out experiments/my_thermal_polish --max-nfev 40
.venv/bin/python -m nslens.thermal_camera.array_audit experiments/my_thermal_polish/result.json
.venv/bin/python -m nslens.thermal_camera.root_audit experiments/my_thermal_polish/result.json
```

The first run and measured tradeoffs are recorded in
`experiments/thermal_camera_polish_v1/REPORT.md`. It improves aggregate spot RMS
slightly but worsens worst-field spot; do not treat the polish as an unqualified
improvement or as additional CMA progress.

