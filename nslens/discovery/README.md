# Polygon discovery

**Working baseline: retained-candidate CMA (`portfolio`).** Fresh-seed confirmation
produced verified designs in 41/72 runs versus naive CMA's 12/72, with equal search and
reserved audit budgets. [Results and limitations](../../agent_context/ns_systems/discovery_improvement_20260911.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.search \
  --out runs/discovery_NEW --seeds 200 201 202 --corners 3 4 5 \
  --briefs base wide --search-calls 400 --workers 6
```

The original pilot CLI is retained below for replay. The later equal-budget section
documents the strengthened policy and its controls.

Active geometry prototype: **3-, 4-, and 5-corner bodies**, with an analytic circular
arc per edge and a finite-width input bundle. It reuses `ArcPolygonBody` and the
tested Fresnel/TIR tracer. It does not use the legacy four-corner tilt genotype.

## Shape and curvature

Each family has `5 + 3*K` normalized genes: center x/y, envelope half-sizes, rotation,
K angular offsets, K radial factors, and K signed curvature controls. Ordered angular
sectors replace fixed corner boxes. Asymmetric shapes are allowed. The radial cage is
still a shape prior; it does not span every simple polygon or smooth freeform body.

Curvature uses `q = sag / half-chord`. For half-chord `a`, the face radius is
`R = a*(1+q*q)/(2*abs(q))`. Both a maximum absolute q and a minimum radius are enforced
by construction. Zero curvature is a plane; concave and convex faces are allowed.
Short edges and crossing faces are rejected. Source/detector clearance uses exact
circular-arc extrema, so bulges cannot hide between sampled points.

| Development setting | Initial range |
|---|---|
| Center x / y | 40–70 mm / −10–15 mm |
| Corner-cage half-width / half-height | 12–30 mm / 12–30 mm |
| Radial factor | 0.65–1.0 |
| Angular offset | ±20% of nominal inter-corner angle |
| Absolute sag / half-chord | ≤0.268 |
| Minimum radius / edge length | 8 mm / 4 mm |
| Fixed material index | 1.60 |

These are explicit pilot assumptions, not optimized ranges or manufacturing limits.
The cage dimensions do not equal the curved body's exact outer extent. Material stays
within the existing tracer's 1.50–1.70 contract and is never an optimization gene.

## Finite-bundle folded routing

Initial layout: downward source centered at (55, 50) mm, **4 mm aperture**, receiver
at x=120 mm over y∈[−30, 30] mm. Every angular field (−0.5°, 0°, +0.5°) illuminates
the full aperture with 9 independent rays. This is a broad routing receiver, not an
eyebox or imaging requirement. General receiver orientation is not implemented.

At every field, a provisional design must deliver ≥25% input energy through branches
with **at least two TIR events**, each with ≥0.5° critical-angle margin. At least 60%
of distinct source rays must contribute qualifying energy, with ≥10% of that ray's
input energy per contributor. Fresnel descendants cannot inflate source-ray support.
The code refuses apertures below 1 mm and fewer than five rays per field.

Closure, unresolved-energy, and pruned-energy limits remain explicit requirements.
Trapped light cannot qualify. There is no bonus for indefinitely increasing fold count.
No shrinking-envelope, clean_pack, collimation, package-size objective, or prescribed
surface sequence is imposed at this discovery stage. Those would prematurely require
compaction or imaging refinement. Acceptance permits different qualifying folded paths
across the bundle; it does not assert one common path for every ray.

## Controls and audits

- `random`: uniform normalized sampling.
- `flat`: CMA with flat rejected-candidate ranks.
- `cma`: CMA first minimizes worst requirement-relative violation, then breaks ties
  using measured routing progress. This is an explicit policy, not a universal reward.

Full-covariance CMA supports reflection at normalized bounds; historical CMA still
defaults to clipping. Sigma and restart patience remain declared tunables. Each corner
count has a separate population and equal candidate allowance. The route archive
retains the highest-throughput representative per typed edge/revisit pattern, with
surface labels canonicalized by first occurrence, and samples routes uniformly on
restart. This is a small route portfolio, not a geometry Pareto archive. Earlier
candidates remain in the trace. Diversity is within-run, **not global novelty**.

Every archived representative is subsequently checked with 25 rays across the full
aperture at five angular fields (125 rays) and a denser geometry audit. Audits do not
feed search. Ray/arc intersections are analytic; self-intersection checking is sampled.
Dense verification shares the same engine and is not independent-engine validation.

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery \
  --out runs/polygon_discovery_NEW --corners 3 4 5 \
  --arms random flat cma --seeds 0 1 2 --evals 300 --workers 3

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.render runs/polygon_discovery_NEW

.venv/bin/python -m unittest discover -s tests -p test_polygon_discovery.py
```

Budgets apply **per corner/arm/seed**. Output directories must be new. Invalid geometry
consumes a candidate slot but no source rays. Actual launched source rays are reported;
Fresnel descendants are not additional source rays. `--ray-budget` adds a hard search
ray cap, reserving a complete bundle before dispatch. Dense audit cost is separate.
Partial final populations never update CMA using unevaluated candidates. Equal
candidate budgets need not imply equal ray counts or wall times; report all three.

`--config settings.json` accepts `codec` and `brief` overrides, for example:

```json
{
  "codec": {"max_sag_ratio": 0.4, "min_radius_mm": 12.0},
  "brief": {"beam_width_mm": 6.0}
}
```

Such overrides change the search space/task and must be labeled. `manifest.json`
records settings and source hashes. Each run writes all candidates, an archive with
every dense audit, the best candidate, and counters. The renderer retraces selected
examples, labels failures, and writes PNG plus verification JSON. Illustrations are
selected, not an unbiased sample. No Qwen or GPU is needed for this stage.

## Equal-budget discovery improvement experiment

`benchmark.py` is the newer comparison harness. It gives every method the same
number of **physical bundle evaluations** and the same reserved final audit budget.
The older pilot runner above is preserved for replay; its equal-candidate results
are not interchangeable with this experiment.

The `portfolio` policy adds:

- Lexicographic minimax ranking of **all** requirement-relative violations: reduce
  the largest deficit, then the next largest, before routing-progress tie breakers.
  Physical feasibility always outranks an infeasible candidate. No weighted sum
  permits good throughput to compensate for a failed optical requirement.
- Forty scrambled Sobol proposals at startup, charged to the search budget.
- A pool of twelve measured candidates, including near-feasible failures. Retain
  the best, then geometrically separated alternatives within a quality shortlist.
  Distance is in normalized gene coordinates, not an optical novelty certificate.
- Restart from least-visited retained candidates. Every fourth restart bypasses the
  pool and draws from the feasible route archive, or uniformly if it is empty.
- Five stagnant generations before restart; initial CMA sigma 0.15 in [-1,1]
  coordinates (7.5% of the full coordinate range), population 10, reflected bounds.
  CMA adapts its covariance and sigma thereafter.

These are explicit, empirically compared settings, not globally optimal constants.
The development alternative `portfolio_small` uses sigma 0.08 and patience 8.
The common final selector visits distinct routes first, then alternative geometries,
with at most eight candidates audited per run. The same selector is used by controls.

Controls use the **same new geometry and optical requirements**:

- `naive`: flat infeasible ranks, clipped bounds, restart after a generation without
  route-archive progress. This ports the historical search behavior to the new task;
  it does not recreate the historical optical environment.
- `pilot`: earlier worst-violation/progress ranking, reflected bounds, patience five,
  feasible-route restarts, no retained-failure pool or Sobol phase.
- `random`: uniform independent normalized proposals.

At 400 physical calls, search consumes 10,800 source rays per run. Eight final audits
reserve 1,000 more rays (25 aperture positions × five fields per audit). Unused audit
allowance is not reinvested. Invalid geometries consume proposal slots and runtime,
not fictitious traces; a four-times-call proposal guard prevents endless rejection.
Actual ray counts and wall time remain separate from the equal reserved allowance.
Dense audits never feed search or restart selection.

Reproduce the frozen confirmation in a **new** directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.benchmark \
  --out runs/discovery_confirmation_NEW --stage confirmation --candidate portfolio \
  --seeds 100 101 102 103 104 105 106 107 108 109 110 111 \
  --arms naive pilot portfolio random --briefs base wide --search-calls 400 --workers 12

.venv/bin/python -m nslens.discovery.summarize_benchmark \
  runs/discovery_confirmation_NEW --candidate portfolio
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.audit_benchmark \
  runs/discovery_confirmation_NEW --workers 8
.venv/bin/python -m nslens.discovery.report_benchmark runs/discovery_confirmation_NEW
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.stress_capacity_check \
  runs/discovery_confirmation_NEW
```

`protocol.json` is written before execution and records the selected candidate,
all tasks, settings, acceptance rule, and source hashes. Source copies are preserved
under `source_snapshot/`. The primary metric is runs with at least one dense-audited
passing design. Confirmation requires a positive lower 95% paired seed-block bootstrap
bound and one-sided exact seed-sign p < 0.05 versus naive. All declared jobs must finish.
Each seed block averages all declared corner/brief tasks. Secondary comparisons are
exploratory; summed within-run route counts are not globally unique designs.

The higher-density post-run check uses 101 aperture positions × nine angular fields
and 4,096 geometry samples on the best audited example per brief/corner. It tests
selected examples, not an unbiased population. It and exact audit replay consume
additional verification rays recorded separately. Both use the same analytic tracer;
neither establishes independent-engine agreement, imaging quality, or manufacturability.

The tracer's default 20,000-branch cap applies to the **whole bundle**. The initial
909-ray stress check exhausted it on four of six selected confirmation examples.
`stress_capacity_check` preserves that first check and repeats only the selected
examples with 200,000 and 400,000 branches. All six passed and agreed exactly between
those capacities, with zero branch-limited energy. This diagnostic changes numerical
capacity only; it does not change optical acceptance or frozen search results.

## Long exploratory inception run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m nslens.discovery.long_search \
  --out runs/inception_long_20260911_v1 --hours 3.25 --workers 6 \
  --calls-per-job 4000 --videos 3 --video-frames 150 \
  --render-timeout-minutes 10
```

Use a fresh output path. Search runs for 3 hours 15 minutes; final in-flight
populations/audits add time, followed by up to ten minutes per video. This targets
roughly 3–4 hours overall, not a hard process-kill deadline. CPU only; no learned
models are trained. The long run has not been launched as part of setup.

A seeded shuffled grid covers 216 task settings: 3/4/5 corners, 4/6 mm beams,
broad/right-1.5×/right-2.5×/bottom-2.5× receivers, ±0.5°/±2°/±5° angular fields,
and standard/small/thin geometry ranges. After one grid traversal it repeats with
fresh seeds. Each independent portfolio CMA job permits 4,000 physical calls and
8 final dense audits. Time can expire before every setting is visited; completion
reports actual coverage. This is broad exploration within a fixed 2D circular-face,
n=1.60 representation, not exhaustive search or an imaging-FOV/compaction claim.

Every job saves candidate history, its configuration, verified archive and summary.
The root saves the declared grid, source snapshot/hashes, submitted jobs, incremental
summaries and completion counts. These are replay records, not resumable CMA checkpoints.
Do not modify source code during execution. Training lineage/state instrumentation
for the proposed meta-BBO comparison remains separate work.

After search, up to three successful/strongest runs from distinct task settings are
rendered to `video_1/evolution.mp4`, etc. Each movie shows chronologically sampled
**generation winners**, not monotonic best-so-far evolution or a genealogical lineage.
It uses the earlier dark-blue/gold-ray style with no labels, title or other text.
A green tick requires matching passing dense checks at 200k/400k branch capacity
and a new verified route or ≥0.05 quality improvement. Ticks are not fabricated when
no candidate qualifies. Videos are selected examples, not evidence of success rate.
`videos.json` records rendering failures/timeouts; histories survive for rerendering:

```bash
.venv/bin/python -m nslens.discovery.animate_search PATH_TO_INDIVIDUAL_RUN \
  --out runs/inception_rerender_NEW --max-frames 150 --no-text
```

Video verification and rendering are post-search work and their evaluations are not
included in job search/audit budgets. Per-frame dense outcomes are saved in each
video's manifest. Search stops between populations and still saves/audits partial
jobs; abrupt machine shutdown is not an orderly stop.
