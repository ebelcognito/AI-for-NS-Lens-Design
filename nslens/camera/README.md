# Multi-body camera inception and direct geometry construction

This development environment reuses the project's `PolygonCodec`, analytic arc and
polynomial-face geometry, Fresnel/TIR `trace_bundle`, full-covariance `CMAES`, and
retained-candidate `RestartPool`. It is a 2D monochromatic imaging cross-section,
not a reconstruction or a qualified 3D smartphone camera.

## Three approaches

1. `inception`: numerical geometry proposals with the existing CMA and restart pool.
2. `model`: Qwen directly proposes complete physical geometry, receives measurements,
   and revises. **There is no CMA/optimizer action in the model interface.**
3. The same `model` runner accepts a frontier Chat Completions-compatible endpoint
   later. No frontier run is required or automatically launched. `task` exports the
   brief and interface without inference. Endpoints with a different API need an
   adapter; no particular frontier service has been tested.

This is a separate camera task, not a change to the existing overnight routing runs.
The fixed default scaffold is three bodies with 4/3/3 faces. It does not prescribe
lens/prism roles, mirror faces, fold counts, or interaction sequences. Body count
and corner counts are protocol choices (`--corners`), not discovered by this pilot.
There are no patent surface coefficients or successful starting designs in the input.
The schema example overlaps and is explicitly only a format example.

## Specification and limits

Patent reference: https://patents.google.com/patent/US20210096338A1/en, Fig. 3A.
EFL 12.24 mm, f/2, half FOV 11.6 degrees and sensor semi-diagonal 2.52 mm come
from that example. The **24 mm length, 8 mm thickness, n=1.60, 50% throughput,
80% pupil-root support, 0.10 mm RMS and 0.10 mm centroid-error limits are our
explicit development assumptions**, not Apple's claimed prescription/performance.
The 2D aperture is EFL/f-number = 6.12 mm; it does not establish a 3D f-number.
No IR filter, dispersion, diffraction/MTF, sagittal field, tolerancing, or real coating
losses are modeled. Ideal mirrors are deliberately labeled `mirror`, never `tir`.

Bodies are disjoint with air gaps. The nearest surface wins across *all* bodies at
every interaction; an escaped ray can hit a different body or return to an earlier
one. Material index is taken from the actual hit body. Energy remains in the existing
ledger, including pruning, depth and branch truncation. Cemented/nested bodies are
not supported. Bounding-box separation is conservative: interleaving boxes are
rejected even if actual surfaces would be disjoint. Polynomial extents use a safe
coefficient-based overbound. These restrictions can exclude Apple-like arrangements.

Default inception has circular faces. `--freeform-bound 0.03` exposes three
endpoint-clamped polynomial departures per face, bounded to +/-0.03 mm. Both arms
must use the same setting; freeform tracing is slower and requires denser intersection
checks. The model controls positions, sizes, rotation, corner locations, curvature,
polynomial departures when enabled, and binary ideal mirror coatings.

The objective is imaging, not just routing: each angular field fills the entire
entrance aperture. All optical branches reaching the finite sensor contribute to
throughput, centroid and RMS. Direct bypass rays are excluded and reported. One
common image parity is allowed; the focal length and targets cannot be fitted away.
Unique input roots, not Fresnel descendants, determine support. Fold counts are
reported but never rewarded. Missing fields fail. Sensor clipping is reflected in
throughput/support; RMS describes received signal, not lost rays.

Coarse trace: 3 fields x 5 pupil rays. Final audit: 7 x 25, repeated at doubled branch
capacity, with denser geometry/intersection sampling. Final audits never feed search.
This is same-engine verification; independent-engine and 3D validation remain future
work. The small coarse grid may miss narrow defects, hence only final audited passes
count. Dense sampling alone is not a proof of manufacturability or global validity.

## Run

Use fresh output directories; existing runs are never overwritten or resumed.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.camera.run inception \
  --out /playpen/hisham26/camera_inception_NEW --budget 100 --seed 16

# Start the existing local Qwen service separately if it is not already running.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.camera.run model \
  --out /playpen/hisham26/camera_qwen_NEW --budget 100 --max-calls 100 \
  --max-output-tokens 4096 --reasoning-tokens 1024 --seed 16

.venv/bin/python -m nslens.camera.render /playpen/hisham26/camera_qwen_NEW
.venv/bin/python -m nslens.camera.compare \
  /playpen/hisham26/camera_inception_NEW /playpen/hisham26/camera_qwen_NEW

.venv/bin/python -m unittest tests.test_camera -v
```

The displayed 100-call Qwen command is a runnable example, **not an executed run**.
Model endpoint defaults to localhost:8080/v1, alias `qwen3.8-27b`. Explicit
`--endpoint` and `--model` select a later frontier model; optional `MODEL_API_KEY`
is used only as an Authorization header and never saved. No inference runs for
`python -m nslens.camera.run task --out NEW`.

## Budget and benchmark contract

Every proposal, including malformed/invalid geometry, consumes a candidate slot.
A complete coarse bundle is reserved before tracing. Invalid geometry costs zero
source rays; actual ray counts are reported, so equal candidate budgets are ceilings,
not a claim of equal realized work. Audits and diagnostic rendering are separate.
Partial CMA populations never cause fabricated updates. Numerical search uses
population size 4, default sigma .2 (`--sigma` overrides it), reflected bounds, and restarts every five generations
or when CMA requests it, taking retained measured candidates when available.

Models receive the same brief/bounds/evaluator, recent proposals and their own best
candidate, with no optimizer, shell, witnesses, other-arm results or audit results.
`finish` can submit a previously measured candidate. At budget exhaustion, both
methods automatically submit their best measured candidate under the same ranking.
Transport errors are recorded and do not count as successful model runs. Full raw
requests, response model metadata, reported token usage and timing are saved.
Request and per-response output limits bound inference; there is no tokenizer-based
hard cumulative token ceiling. Compact history contains the previous three actions
plus the best geometry. Report these inference constraints separately from optical
budgets, and hold them fixed for Qwen/frontier comparisons.

`compare` rejects task, physical-budget or physics-source mismatches. Use multiple
matched seeds and preserve failures; a smoke test is not a benchmark result. Token
budgets are not physically equivalent across tokenizers. Numerical evaluations,
actual rays, final audit rays, wall time, and model usage are separate costs.

Artifacts: `protocol.json` (brief/budgets/source hashes), `candidates.jsonl`,
`model_calls.jsonl` for model runs, `selected.json`, `design.json`, `summary.json`,
and optional `geometry.png`. Candidate arrays are normalized internal coordinates;
`design.json` is the physical prescription Qwen uses.

For a bounded local smoke test, add `--reasoning-tokens 1024 --max-output-tokens
4096 --budget 8 --max-calls 8`. `reasoning_budget_tokens` is a llama.cpp extension
verified against the installed server source; omit it for other endpoints. The
server's default unrestricted thinking can otherwise consume most of the output
allowance before a geometry is returned. A different reasoning cap is a different
inference protocol, not an interchangeable result.

## Executed development pilot

See [the session handoff](../../agent_context/ns_systems/camera_inception_20260916.md)
and [completed report](/playpen/hisham26/camera_comparison_20260916/REPORT.md).
The 8-candidate matched runs and separate 400-candidate inception run completed;
none passed imaging requirements. Qwen constructed eight valid geometries directly.
No frontier model was run.

## Graded surface progression and estimated bounds

See the [workspace report](../../results/camera_inception_20260916/REPORT.md).
The camera search now grades surviving energy field by field as it encounters bodies
0, 1 and 2 in order. Within each stage, forward-ray distance to the next body's
boundary breaks ties. An already sensor-delivering layout gets priority for further
improvement. This is **search feedback only**: the final imaging audit accepts any
physical route, including a route that visits bodies in a different order. The
ordered stages are a deliberate construction prior for an Apple-like layout.

`--placement-bounds '2,5;6,10;17,21'` sets per-body x-centre intervals; default
centres span x=1..22 mm. `--corners 3 3 4` changes the body-family prior. Size
halves now span 0.5..3 mm. A previous design may be supplied with `--seed-design`;
it is evaluated and charged as candidate zero. Seeded refinement should be reported
separately from a from-scratch search. The four executed CMA runs and their source
snapshots are in the workspace report. None passed the dense camera audit.
