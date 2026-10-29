# Three-mirror curiosity pilot

This asks whether the existing CMA optimizer finds useful mirror layouts. It is
not a benchmark against the authors or a reproduction of their algorithm.

Design brief: Zhang, Jin & Zhu (2021), Example 1:
https://www.nature.com/articles/s41377-021-00510-z
The published example specifies 50 mm focal length, 27.78 mm entrance pupil,
f/1.8, and an 8 x 6 degree full field. Here only the 8 degree cross-section is
used. Their full 3D wavefront acceptance criterion is not evaluated.

## Model and search assumptions

Three finite, ideal, two-sided mirrors in air; each is a local parabola
`vertex + u*tangent + curvature*u^2*normal`. This first pilot has no cubic or
higher freeform coefficients, coatings, diffraction, sagittal rays, mechanical
substrates, tolerances, or surface thickness. Dimensions are millimetres.

The twelve normalized variables describe three chief-ray deflection angles,
three propagation lengths (including the last-mirror-to-image distance), three
curvatures, and three mirror half-apertures. These bounds are our exploratory
choices, not a transcription of the paper's search space. A chief-ray scaffold
provides a construction prior. The initial layouts and mirror powers are random;
no successful published prescription is supplied. Actual intersections always
use the nearest of all three mirrors, including repeated hits and obstructions.
The image plane is an infinite absorbing plane reached in its forward direction.
All three mirrors must be encountered exactly once for a ray to count as signal.

The existing `nslens.search.cma_me.CMAES` supplies full-covariance CMA. This is
scalar-loss CMA with restarts and an external diversity archive, not the CMA-ME
improvement-emitter algorithm. Restarts sometimes draw from the archive. Seed
streams share that archive and are not independent replication trials.

The archive groups deflection sign patterns relative to the coaxial scaffold,
curvature signs, and a spacing-ratio bin. Occupied bins include failed designs:
cell count is not a count of usable systems or genuinely distinct topologies.
Selected layouts are the best twelve different archive cells by search loss.

Search uses three fields x nine pupil rays. Loss combines squared error relative
to the prescribed 50 mm image mapping (one shared image parity), missing-ray
penalties, and an ordered-route progress penalty. It counts every entrance ray;
lost rays are not silently dropped from the objective. Per-field spot RMS is
computed only for received rays and must be read together with support.

Final selected layouts are rescored on eleven fields x 101 pupil rays. This is
same-engine sampling validation, not independent optical validation. Crossings
of unused mirror portions and mechanical clearance need a separate audit.

## Run

```bash
.venv/bin/python -m unittest discover -s tests -p test_three_mirror_pilot.py
.venv/bin/python -m nslens.three_mirror.pilot \
  --out experiments/three_mirror_pilot_v1 --budget 12000 --seeds 3
```

The output directory must be new. Budgets round down to complete populations of
24. Outputs: `summary.json` (dense audits, progress and settings), `archive.json`
(normalized prescriptions), and `gallery.png`. `decode(genes)` supplies physical
vertices, axes, curvature coefficients, apertures and detector pose.

A discovery worth following up should keep the full pupil across the field,
show reasonable imaging at the specified scale, and survive a mirror-clearance
check. Full freeform refinement and eventual 3D validation are later steps.

Post-search screening and physical export:

```bash
.venv/bin/python -m nslens.three_mirror.audit experiments/three_mirror_pilot_v1
```

This writes `geometry_audit.json`: 21 fields x 201 rays and pairwise intersections
of mirror polylines sampled at 513 points. Tangencies and mechanical clearance
are not certified. Rerendering after this audit labels detected crossings.
See `experiments/three_mirror_pilot_v1/FINDINGS.md` for the first run's findings.
