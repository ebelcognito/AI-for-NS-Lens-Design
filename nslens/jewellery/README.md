# Jewellery appearance pilot

An isolated 3D convex dielectric evaluator plus a reproducible appearance-search
experiment using the existing `nslens.search.cma_me.CMAES` optimizer.

```bash
.venv/bin/python -m unittest discover -s tests -p test_jewellery.py -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.jewellery.pilot \
  --out experiments/jewellery_NEW --calls 96 --seeds 101 102
```

Output directories must be new. Open `index.html` to compare synchronized rocking
animations. `report.json` records genes, all coarse evaluations, refinement inputs,
dense training evaluation, held-out lighting/orthogonal motion, numerical checks,
volume, smallest edge/facet and descriptor archive.

## Physical scope

- 3D half-space intersections on a closed convex, multi-tier faceted solid.
- Nearest forward boundary, vector Snell/reflection, Fresnel transmission and TIR.
- Every camera path includes surface reflection and internally transmitted exits.
  For a convex solid only the internally reflected path continues.
- Scalar unpolarized Fresnel at every interaction **does not preserve polarization
  memory across changing incidence planes**. No claim of full polarized accuracy.
- Monochromatic synthetic n=1.60 material. No real gemstone material calibration,
  dispersion/fire, absorption, roughness, coatings, setting or finite camera pupil.
- Radiance paths start/end in air. Reported closure is camera-path weight closure,
  not a forward illuminated-power efficiency measurement.
- Orthographic camera, fixed finite smooth environment sources, head-shadow model,
  fixed display gamma/exposure, no bloom or inserted sparkle.
- Fixed maximum 10 mm girdle diameter; volume and depth vary and are reported.
- Reference is a simple 8-fold tier family, **not** a standard brilliant cut.

## Search scope

Six continuous genes: mean crown and pavilion angles, table control, two bounded
asymmetry amplitudes and pavilion azimuth offset. Geometry remains convex by
construction. Individual nominal planes may become inactive; active facets and
minimum feature sizes are reported on final candidates. Geometry is not certified
for cutting, polishing or durability.

The pilot compares random search and three scalar CMA objectives (brightness,
broad-flash proxy, fine-sparkle proxy), with identical coarse rendering and search
calls per seed when calls is divisible by 8. CMA uses `8 * floor(calls/8)` calls.
Selected CMA candidates receive 24 guarded coordinate-refinement evaluations each.
The descriptor archive collects candidates from **all** arms. It does not drive
search or establish a quality-diversity advantage. A matched archive-driven search,
conventional-cut baseline, real-material model and human evaluation remain later
steps in the research protocol.

Metrics are experimental proxies: fixed linear threshold .3 for bright connected
regions, brightness, lower-tail brightness, dark fraction, flash-region size/count,
and image variation per degree. Motion activity can include silhouette effects;
common masks exclude background but are not material-coordinate registration.
Pixel convergence matters especially for tiny regions. The fine objective can
exploit aliasing: inspect the dense and resolution checks before interpreting it.

A residual-weight penalty rejects candidates above .001 mean unresolved weight
at any sampled pose. The renderer separately reports maximum residual per pixel.
Increasing bounce depth and sample density remains mandatory for selected designs.

## Files

- `model.py`: geometry, physics, environment, rendering and appearance metrics.
- `pilot.py`: CMA/random search, local refinement, audits, GIF/HTML report.
- `tests/test_jewellery.py`: analytic and invariance checks.

See `agent_context/jewellery_demo_research_20260929.md` for prior work and the fuller
proposed experiment. This module implements the first capability/control pilot,
not all validation gates or the complete proposed study.

## Archive-driven comparison

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.jewellery.discovery \
  --out experiments/jewellery_archive_NEW --calls 192 --seeds 201 202 203
```

This experiment makes the archive active: CMA ranks each batch against the
pre-generation archive (new feasible cells first, then brightness improvement),
and restarts from uniformly sampled archive elites. Restarts occur every four
batches or after three stale batches. This is an archive-guided CMA variant,
not a claim to reproduce a published CMA-ME implementation exactly.

Compare it with a multistart CMA portfolio split equally across brightness,
broad and fine objectives, and uniform random search. Each arm uses the same
six-parameter geometry, evaluator, bounds and candidate budget. Each baseline
also retains its own best-per-cell archive. Seeds are paired; initial CMA centres
are zero, subsequent portfolio restart centres are uniform. No refinement occurs
in any arm for this comparison.

The fixed 12x15 archive uses bright-region area (bin width .01) and motion
activity (.002 per degree). Out-of-range descriptors are excluded, not clamped.
Admission requires brightness >=.045, lower-tail brightness >=.025, and mean
unresolved camera-path weight <=.001 at every evaluated pose/environment.
Within cells, brightness is quality. These are pilot thresholds, not beauty grades.

Every coarse elite is re-evaluated with 48 pixels, 15 frames and 96 bounces in
training and held-out scenes. Rebuild archives to count migrated/colliding cells.
Search call budgets are equal; audit costs vary with elite count and are reported
separately. This audit is not proof of numerical convergence. Galleries select
up to six diverse candidates per arm using the same greedy spread rule on dense
training metrics, pooled across seeds; held-out metrics never select winners.

Results and checkpoints are saved in the new output directory; index.html has
training/held-out GIFs for each arm. tests/test_jewellery_discovery.py checks
admission, ranking, exact budgets, replay determinism and actual elite restarts.

The renderer records internal edge/grazing intersection failures as unresolved
camera-path weight. Such paths contribute no radiance. The same residual gate
therefore rejects significant numerical loss instead of rewarding it.

Interrupted archive experiments can use `--resume` with the same output path,
call count and seed list. Completed search checkpoints are reused after checking
config.json. Use a new directory after changing search/evaluator semantics.
After completion, generate the comparison plot and embed it in the HTML with:

```bash
.venv/bin/python -m nslens.jewellery.plot_discovery experiments/jewellery_archive_NEW
```
