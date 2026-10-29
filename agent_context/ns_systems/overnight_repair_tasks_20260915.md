# New development repair tasks — 2026-09-15

User requested repair tasks from the new overnight design dataset. Completed
generation and fixed-CMA profiling; no Qwen inference or training was performed.

## Results

- 218 accepted tasks from 38 task settings, using one highest worst-field-throughput
  certified witness per setting. 228 recipes attempted; 10 exhausted/rejected.
- Shapes: 78 triangle, 88 quadrilateral, 52 pentagon tasks.
- Pose, curvature, and shape perturbations at normalized scales 0.04 and 0.12.
- Witnesses freshly reverified using 5x25 / 7x51 sampling and doubled branch capacity.
- Degraded starts fail all three grids, with throughput in [0.20, 0.775), and
  stress evaluations agree at doubled branch capacity. All attempts are retained.
- One-seed fixed-CMA profiles: 147/218 verified after 16 candidates; 170/218
  after 32. 65 coarse-observation midpoint failures; 71 dense-audit midpoint failures.
- The 65-case list is an available diagnostic cohort, not a held-out benchmark.
  Baseline outcomes did not filter the full 218-task dataset.
- All tasks are DEVELOPMENT data. Related starts share a witness/setting; no
  independent-family or held-out generalization claim. Keep ancestry grouped.
- Model inputs are separately saved allowlisted observations; witnesses, recipe,
  split, family identifiers, and generation provenance remain outside those inputs.

## Artifacts

Root: `/playpen/hisham26/overnight_repair_tasks_20260915`.

- `RESULTS.md`, `summary.json`, `task_index.csv`, `tasks.json`
- `observations/` — model-facing initial numerical observations
- `private_witnesses.json`, `private_generation/` — provenance and failed attempts
- `profiles/` — saved first/second batches and midpoint/final dense audits
- `midpoint_coarse_failing_ids.json`, `midpoint_audit_failing_ids.json`
- `validation.json`, `protocol.json`, `source_snapshot/`, `run.log`

Costs: 114 witness-recheck evaluations / 31,882 rays; 2,207 generation evaluations /
233,167 rays; 8,284 profile evaluations / 554,156 rays. Original discovery and
certification computation is reused and excluded from these new-work totals.

Nine repair/CMA automated checks passed. All 218 tasks additionally passed explicit
provenance, failure-grid, capacity-agreement, uniqueness, and observation-allowlist checks.

Implementation: `nslens/analysis/build_overnight_repair_tasks.py`. It is resumable
with protocol/source hash checks. The source directory includes a full nslens/tests
archive. Run with single-threaded BLAS and eight worker processes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m nslens.analysis.build_overnight_repair_tasks \
  --out /playpen/hisham26/overnight_repair_tasks_20260915
```
