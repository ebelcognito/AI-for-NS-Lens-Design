# Executed polygon discovery prototype — 2026-09-11

The user's clarified priority is better geometry/curvature bounds, including three-
and five-corner bodies, finite-width illumination, and interesting folded paths.
`nslens/discovery/` now implements that scope, using the existing analytic circular-face
body and Fresnel/TIR tracer. Qwen and compaction are not involved in this experiment.

## What changed

- Search three, four, or five ordered corners with variable pose, aspect, radial shape,
  angular offsets, and one independently curved circular face per edge.
- Replace absolute sag boxes with sag/half-chord limits and an independent minimum
  radius; reject short edges and intersecting surfaces. Exact arc extrema check plane
  clearances. The old four-corner genotype and historical scripts remain available.
- Use a 4 mm source aperture at three angular fields, rather than an isolated ray.
  Require meaningful delivered energy from a majority of source rays via paths with
  at least two TIR events. The surface sequence is not prescribed.
- Separate initial folded routing from imaging and compaction requirements. A broad
  60 mm receiver is a discovery screen, not an eyebox or focused image.
- Compare uniform sampling, flat-rejection CMA, and constraint-feedback CMA. All CMA
  runs use the same normalized representation, reflection, initial sigma, and restart
  patience; feedback is the varied policy. These settings remain experimental tunables.
- Save all candidates and failure metrics, hard candidate/ray budgets, full settings,
  source hashes, dense audits, and actual ray-layout illustrations.

See [package documentation](../../nslens/discovery/README.md) for exact ranges and gates.

## Completed pilot

Run: `runs/polygon_discovery_pilot_20260911_v2/`.
Three corner counts × three search policies × three seeds × 300 candidates = 8,100
evaluated candidates on **one optical brief**. Each policy therefore uses 2,700
candidate evaluations. This is a development pilot, not a held-out family benchmark.

| Policy | Geometrically traceable candidates | Coarse qualifying candidates | Retained representatives passing dense audit | Search source rays | Audit source rays |
|---|---:|---:|---:|---:|---:|
| Uniform random | 2,655 | 5 | 5 | 71,685 | 625 |
| Flat-rejection CMA | 2,693 | 0 | 0 | 72,711 | 0 |
| Constraint-feedback CMA | 2,698 | 26 | 6 | 72,846 | 750 |

These retained counts sum **per-run route representatives**. They are not counts of
globally novel or mutually independent optical topologies. Coarse passing candidates
include related prescriptions along one search trajectory. Actual ray budgets differ
because invalid geometry is rejected before tracing; the comparison is matched on
candidate allowances, not exactly matched on rays or elapsed time.

| Corner count | Random dense passes | Flat-CMA dense passes | Feedback-CMA dense passes |
|---|---:|---:|---:|
| 3 | 0 | 0 | 1 |
| 4 | 4 | 0 | 4 |
| 5 | 1 | 0 | 1 |

All 11 retained representatives passed the 125-ray, five-angular-field audit. No audit
result fed back into search. All 11 audit records were independently re-executed and
matched exactly; that replay costs an additional 1,375 source rays. The source hashes
in the run manifest also matched the current implementation.

The new representation can produce verified folded starting designs in every requested
corner-count family. Feedback improves the number of passing search candidates over
flat ranking here. Its retained portfolio is only slightly larger than random search's,
so the evidence does not yet establish a dependable optimizer advantage or an 8–10%
end-to-end engineering improvement. There is no comparison to the historical optical
acceptance objective: this experiment deliberately changes both geometry and task scope.

## Visual examples and limits

`discovery_examples.png` shows one selected design per corner count. Re-traced dense
worst-field folded throughput / contributing-root fraction are:

- three corners: 82.5% / 92%;
- four corners: 87.0% / 100%;
- five corners: 78.9% / 88%.

These are selected illustrations, not typical performance. Only central angular-field
paths are drawn; qualification checks all five angular fields. The receiver is broad
and output beams are not yet focused or collimated. Different rays may follow different
qualifying folded routes. The geometry audit uses sampled self-intersection checks,
whereas ray/circular-face intersections are analytic. Dense checks share the same
physics kernel, so independent-engine validation remains future work.

## Verification and next decision

Passed: seven new geometry/aperture/budget/replay tests, sixteen existing arc-polygon
tests, three optimizer-trajectory checks, and a 100-evaluation legacy QD equivalence
run with 13 identical cells. The only shared optimizer extension is opt-in reflected
boundary handling with snapshot/replay support; its default remains clipping.

The next useful experiment is a development sweep of curvature/radius ranges and
shape envelopes, evaluated on folded-energy yield, root support, route diversity,
and tolerance to perturbations. Keep each range and finite-aperture requirement fixed
within a run. Preserve diverse geometries within a route before adding compaction;
the current one-representative-per-route archive can discard alternative useful basins.
Do not turn this into a huge training corpus until those representation choices and
the downstream usefulness of these seeds have been measured.
