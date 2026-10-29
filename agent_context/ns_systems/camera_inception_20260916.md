# Multi-body camera inception and Qwen geometry construction — 2026-09-16

User requested three ways to construct an Apple-inspired thin camera: existing
inception/search framework; Qwen directly constructing geometry in our engine;
and a fair frontier-model interface, with frontier inference deferred. Qwen must
not merely control CMA in this experiment. ADHD output skill remains active.

## Implementation

See `nslens/camera/README.md` for contracts, commands and limitations. New package:
`scene.py`, `problem.py`, `run.py`, `compare.py`, `render.py`.

The existing `nslens/freeform_body/trace.py` now accepts hit-local material index,
ideal mirror metadata and body/face identities supplied by `OpticalScene`. All
single-body defaults are preserved. The scene selects the nearest hit among all
bodies at each interaction, including air-gap propagation and revisits. The existing
Fresnel/TIR energy ledger is reused. Only disjoint bodies in air are supported.
Ideal mirrors are unit-reflectance coatings, explicitly distinct from TIR.

Default 4/3/3-face bodies use existing `PolygonCodec`; pose, shape, curvature and
coatings are variables. No named lens/prism roles or surface sequence is imposed.
The component count is a declared prior, not discovered. Optional endpoint-clamped
polynomial freeform departures use existing `PolynomialFaceBody`; default pilot is
circular-face inception. Geometry uses conservative nonoverlapping bounding boxes,
which can exclude otherwise valid Apple-like arrangements. Do not describe this as
a completed reproduction or general 3D camera engine.

Patent Fig. 3A supplies EFL 12.24 mm, f/2, half-FOV 11.6 degrees and semi-diagonal
2.52 mm. Our development choices are 24 x 8 mm envelope, n=1.6, 50% throughput,
80% root support, <=0.10 mm RMS and <=0.10 mm centroid error. Fixed sensor at x=24,
entrance centered x=4 facing downward. No IR filter, dispersion, MTF or 3D analysis.
Three coarse fields x five pupil rays; final 7 x 25 audit repeated with doubled
branch capacity. Audits do not feed search. One global image parity; no target fitting.

Inception reuses `CMAES` and `RestartPool`, population 4, sigma .2, reflected bounds,
five-generation restarts. Qwen gets complete physical geometry JSON, its observations,
last three proposals and best candidate. It has no optimizer action. Every invalid
proposal consumes a candidate. Source rays and audits are accounted separately.
Model endpoint is local llama.cpp; response/model metadata, usage and timing saved.
The explicit `--reasoning-tokens` flag is a llama.cpp extension, not generic API.
Frontier access remains untested/deferred; compatible endpoints can use the interface.

## Runs and provenance

- `/playpen/hisham26/camera_inception_20260916_v2`: matched 8-candidate numerical
  smoke pilot, seed 16; 105 search rays, no successful camera.
- `/playpen/hisham26/camera_qwen_20260916_v2`: local frozen `qwen3.8-27b`, same
  8-candidate/120-ray ceiling, seed 16; 1024 reasoning tokens and 4096 total output
  tokens per response. Completed: eight valid proposals, 120 search rays, 350 audit
  rays, zero sensor-delivering proposals, no camera pass. 61,593 reported local
  tokens, 469 seconds; budget-exhaustion selected candidate 0. No transport errors.
- `/playpen/hisham26/camera_inception_20260916_development`: separate 400-candidate
  exploratory run, seed 17, 3525 search rays. Its candidate 375 sends some light from
  every coarse field to the sensor but fails throughput/support/image position.
  `geometry.png` shows dominant paths, not all scored Fresnel branches.
- Earlier `_smoke` outputs are obsolete workflow checks. The unrestricted-thinking
  Qwen `_smoke` was interrupted before any returned candidate; `interrupted.json`
  records why. Exclude these from the final comparison.

Output directories are immutable/new-only. Manifests record source hashes; this run
runner does not save a full source snapshot. All repeated seeds are development,
not held-out families. No training, frontier calls or long unattended camera runs
were authorized/launched as part of this implementation. The local Qwen server was
started through `start_qwen.sh` on GPU 1; GPU 0 workloads were untouched.

## Validation

12 new camera unit checks plus 13 existing polygon-discovery/polynomial checks pass.
18 standalone freeform-body and 16 standalone arc-polygon checks also pass.
`test_freeform_body.py` and `test_arc_polygon.py` use standalone test functions;
loading them through unittest silently runs none. Use their script entry points.
25 other standalone relay checks pass. Four legacy relay checks fail:
`test_a_dead_field_cannot_hide_behind_two_good_ones`,
`test_reflection_parity_is_discovered_rather_than_prescribed`,
`test_the_ladder_grades_designs_the_optical_fitness_cannot_separate`, and
`test_the_throughput_floor_excludes_designs_that_leak_almost_all_the_light`.
All four assertions also fail with the exact original `trace_bundle` function
restored in memory. These failures predate the camera extension.

Completed comparison: `/playpen/hisham26/camera_comparison_20260916/REPORT.md`
and `comparison.json`. Qwen and exploratory inception PNGs are saved beside their
run summaries. The locally started Qwen server remains available on port 8080.

## Next research step

Inspect paired completed reports before allocating more search. The eight-call
pilot checks construction/protocol, not model superiority. Sparse initial feedback
when no light reaches the sensor is a known search weakness; richer per-body hit,
escape and missed-sensor diagnostics can be added under a fresh shared protocol.
Before claiming a camera solution, establish a certified solvable development case,
increase pupil/field sampling, validate freeform intersections and validate in 3D.

## Follow-up: graded CMA and workspace artifacts

The user could not access `/playpen` and requested estimated bounds plus CMA with
prior surface-to-surface sequential grading. The complete user-facing report and
all selected ray plots are now inside the workspace:
[graded camera report](../../results/camera_inception_20260916/REPORT.md).
`nslens/camera/sequential_grade.py` grades body 0 → body 1 → body 2 contacts and
forward ray distance to the next body. The final imaging gate is unchanged and
remains route agnostic. `CameraCodec.placement_bounds` and `--placement-bounds`
allow explicit per-body centre intervals. `--seed-design` consumes candidate zero;
`--sigma` controls normalized CMA step size. `RestartPool` accepts a rank callback,
retaining its original default for other tasks.

Four runs completed: unseeded CMA, 2,000 proposals each, at centre bounds
(3–6,10–13,18–21) and (2–5,6–10,17–21); a 2,000-proposal seeded run at sigma=.2;
and a 1,000-proposal seeded run at sigma=.05. No run passed the dense camera audit.
The second unseeded layout delivered ~16–18% across three coarse fields but failed
one dense field. The seeded runs raised worst coarse throughput from 16.0% to
17.8–17.9%, while worst dense throughput stayed 3.6–4.0% and worst dense pupil
root support stayed 4% (one of 25 rays). Full measurements, PNGs, source hashes
and exact source snapshots are in the run directories linked from the report.
The seeded design is a reordered geometry from the earlier 400-proposal run, not
an independent from-scratch discovery or a patent prescription. Prior Qwen results
are not matched against this revised task. The frontier model was not run.

`python -m unittest tests.test_camera tests.test_polygon_discovery
 tests.test_polynomial_faces -q` passed 27 checks after the grading/bounds edits.
