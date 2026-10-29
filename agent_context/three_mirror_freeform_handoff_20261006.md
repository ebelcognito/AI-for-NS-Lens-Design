# Optical discovery and freeform refinement handoff

Date: 2026-10-06 (America/New_York)
Workspace: `/home/hisham26/Desktop/NS_lensdesign`

## Current conclusion

Final-mirror freeform refinement substantially increases the number of saved candidate pools yielding a usable design. We have **not demonstrated a superior sampling mechanism**: the latest fresh searches tied across continued CMA, optical-residual proposals, and random-direction proposals.

The latest completed user request was to collect all stored three-mirror samples into a large directory. That collection is complete. No experiment is intentionally left running. This document is a continuation guide, not a new experimental result.

## Research intent and user preferences

The ultimate objective is one excellent optical design. Diversity is useful insofar as it helps discover better interaction sequences and basins; it is not an end in itself. The user wants an optically grounded mechanism that proposes useful new directions from previous samples, rather than relying on random restarts or text-based LLM proposals.

Treat CMA as a local optimizer for now. Do not assume the current process uses an archive-cell framework. Distinguish the local optimizer, proposal mechanism, and allocation of refinement budget.

The user challenged initial spot size as a measure of a layout's potential: a promising layout may become good after its final surface is made freeform. Judge candidates after a standardized refinement opportunity. The current operational threshold is 1 mm **worst-field mapping RMS**, not spot size alone. This is an experimental acceptance criterion, not an optimality guarantee.

Experiments expanded from three mirrors to 3–6 mirrors, and from one specification to changes in field, aperture, and target focal mapping. Prefer these mirror environments and eventually TIR penta/quad cases over spectrometers. The latest collection request specifically concerns exactly three mirrors.

Communicate concisely. The user has explicitly requested cheap Luna workers for bounded delegated work. Relevant skill: `/home/hisham26/.codex/skills/cheap-luna-workers/SKILL.md`. Do not infer authorization to publish or send results externally. This work does not establish what was shared with the OPDO team.

## Physical models: do not mix the encodings

### Current variable-count model

`nslens/analysis/variable_mirror.py`: 2D, finite parabolic mirrors, 3–6 mirrors. There are `3*n` genes grouped as turns, leg lengths, and curvatures:

- Turns: `180*g` degrees.
- Leg lengths: `65 + 40*g` mm.
- Curvature: `0.025*g`.
- Fixed mirror half-width: 30 mm.

Detector location and orientation follow the final nominal ray and leg. Rays intersect the nearest actual surface. Routing and sampled surface crossings matter; a low error alone does not establish feasibility.

### Legacy three-mirror model

`nslens/three_mirror/pilot.py`: 12 genes, still exactly three mirrors. Turns are offsets around `[-150,180,-180]` degrees with scales `[28,58,58]`; the last three genes vary mirror half-widths as `30 + 16*g` mm.

**A 12-gene vector means four mirrors in the current model, but three mirrors in the legacy model.** Use recorded model identity, not vector length alone across datasets.

### Freeform extension

`nslens/analysis/final_freeform.py` adds cubic through sixth-order terms to the final mirror:

```text
sag(u) = c*u² + a3*(u/30)³ + a4*(u/30)⁴ + a5*(u/30)⁵ + a6*(u/30)⁶
```

Coefficients are in physical mm and bounded individually to ±2 mm during refinement. The tracer uses polynomial intersections and analytic reflection slopes. This is a 2D profile, not a full 3D freeform optical system.

## Shared refinement implementation

`nslens/analysis/optical_refinement.py` exposes:

```python
refine_design(genes, n, task, seed, refiner="last_freeform", budget=261)
```

Supported controls:

- `last_parabolic`: final curvature only.
- `joint_parabolic`: all original parabolic design genes.
- `last_freeform`: final curvature plus four freeform coefficients; other geometry and detector stay fixed during this stage.

Default budget: five initialization trace calls followed by 256 CMA evaluations. Initialization includes a frozen-hit slope observation and correction candidates at scales 1, 0.5, and 0.25. CMA uses population 8 and sigma 0.03. A failed slope fit falls back to zero correction with the same evaluation ledger. Retain the coarse incumbent; dense audits never feed selection.

Training refinement grid: 5 fields × 21 pupil rays. Original discovery grid: 3 × 9. Primary final audit: 15 × 101, with 513 surface-crossing samples. Additional robustness check: 31 × 201, with 1025 crossing samples.

Passing criterion: worst-field mapping RMS ≤1 mm, worst-field support ≥99%, and no sampled crossing. Optimization merit also includes routing/lost-ray penalties and a 30000 crossing penalty. It is not identical to the final worst-field threshold.

The user-highlighted design was:
`experiments/design_pngs_20261006/latest_confirmation/field8/optical_residual/field8__optical_residual__1__r1.png`.
Its original mapping RMS was about 1.306 mm. Final-curvature refinement reached about 0.870 mm; final-mirror freeform refinement reached about 0.441 mm. See `experiments/final_freeform_20261006` for the original demonstration. Do not generalize from that one illustration alone.

## Completed controlled comparison: cached candidate pools

Directory: `experiments/shared_freeform_comparison_20261006`
Driver: `nslens/analysis/compare_shared_refinement.py`
Report: `nslens/analysis/report_shared_freeform.py`
Verification: `nslens/analysis/verify_shared_refinement.py`

144 pre-existing discovery pools × 3 refiners = 432 jobs. Each pool contains candidates for all four mirror counts. Refine all four candidates before choosing the count; do not prefilter on initial quality. Each result gets 1044 new refinement calls (4 × 261), plus 2056 inherited discovery calls = 3100 calls. The additional refinement budgets and ray grids are matched between controls.

| Task / proposal | Final curvature | Joint parabolic | Final freeform |
|---|---:|---:|---:|
| field8 / reuse_only | 5/24 | 6/24 | 11/24 |
| field8 / optical_residual | 8/24 | 6/24 | 12/24 |
| field8 / direction_random | 2/24 | 4/24 | 9/24 |
| aperture42 / reuse_only | 5/24 | 6/24 | 13/24 |
| aperture42 / optical_residual | 3/24 | 7/24 | 10/24 |
| aperture42 / direction_random | 4/24 | 6/24 | 12/24 |
| Total | **27/144** | **35/144** | **67/144** |

All 129 passing outputs across the three refiners survived the additional dense audit. Source hashes, call counts, frozen-layout invariants, coefficient bounds, and count selection were verified. Eight donor families underlie the repeats; repeats are not independent new donor families.

This isolates refinement capability on cached pools. It does not establish an improved discovery policy. With freeform refinement, proposal success totals were 24/48 reuse, 22/48 optical residual, and 21/48 random direction.

Useful artifacts: `RESULTS.txt`, `summary.json`, `paired_comparisons.json`, `comparison.png`, and `design_pngs/overview.png`. The six illustrated designs use a fixed donor/repeat, not best-case selection.

## Completed fresh searches: freeform refinement inside the loop

Directory: `experiments/online_freeform_comparison_20261006`
Driver: `nslens/analysis/search_with_freeform.py`
Report and verification: `nslens/analysis/report_online_freeform.py`

48 runs: two specifications × three proposal methods × eight donor families. One fresh optimizer repeat, indexed 4. This is not four repeats.

Each of eight rounds runs 257 parabolic discovery calls and then 261 shared freeform refinement calls **before incumbent replacement**. Counts cycle 3,4,5,6 twice. Total per run: 4144 calls, split into 2056 discovery and 2088 refinement calls. Final selection uses refined coarse merit, followed by a final dense audit.

| Proposal method | field8 | aperture42 | Total |
|---|---:|---:|---:|
| Continued CMA (`reuse_only`) | 4/8 | 5/8 | **9/16** |
| Optical residual | 5/8 | 4/8 | **9/16** |
| Random direction | 5/8 | 4/8 | **9/16** |

All 27 primary passes survived the extra dense audit. All 48 final designs passed the reported routing screen. Verification reports 198912 coarse calls, 48 primary dense audits, and 27 extra dense audits.

Important distinctions:

- `reuse_only` continues CMA from developed donors. It is **not fresh random-restart CMA**.
- `direction_random` is a matched bounded random-direction proposal, not a random design restart.
- Future proposals still use the parabolic gene representation. Higher-order terms are reset during discovery and refit afterward. The sampler is **not freeform-aware**.
- The cached experiment used 3100 calls/result; this fresh experiment uses 4144. Do not attribute differences between those experiments solely to refinement placement.
- Corrected smoke output is `online_freeform_smoke_v2_20261006`. The earlier `online_freeform_smoke_20261006` is incomplete; do not use it as evidence. A crossing-penalty issue in the early draft was fixed before the valid smoke and final batch.

## Full three-mirror sample collection

Directory: `experiments/all_three_mirror_samples_20261006` (about 1.2 GB).
Open `index.html` for a paginated local gallery.

Contents:

- `samples/`: 5260 distinct model + genes + freeform + specification records, with all source JSON pointers.
- `pngs/`: one rendered PNG per distinct JSON record, including failures and intermediate designs.
- `source_records/`: copies of contributing original JSON files.
- `index.csv`: searchable design index.
- `populations/`: 264 NPZ files containing **1267200 saved optimizer evaluations** from legacy three-mirror experiments. Arrays are `X` (N × 12 genes) and `y` (saved loss).
- `population_index.csv` and `.json`: paths, counts, model identity, and checksums.
- `collection_report.json` and `provenance_verification.json`: collection and integrity checks.

Composition: 3714 current-model three-mirror records and 1546 legacy-model records; 254 have nonzero freeform terms. Eight specifications. Exact record deduplication preserved 24707 source occurrences. The bulk populations include duplicates and overlap gallery records; **do not add these counts and call the result unique designs**. Bulk population evaluations are not individually rendered.

All 24707 source gene occurrences were checked against originals, and all 264 copied population checksums matched. No JSON read failures or PNG rendering failures were reported.

Collector: `nslens/analysis/collect_three_mirror_samples.py`. It covers the relevant experiment-family JSON files; separate `collect_bulk_mirrors.py` in the collection copies NPZ populations. The latter was an additional export step, and the final README/index navigation was augmented afterward. Running only the main collector does not recreate those additions. Main collector requires a new output directory.

Scope is stored experiment artifacts, not unsaved optimizer evaluations. Unrelated spectrometer, camera, and TIR encodings are excluded. Sparse PNG display rays are not new quality audits.

## Validation and reproducibility

Fourteen tests previously passed across:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  tests.test_optical_refinement tests.test_final_freeform \
  tests.test_hard_optical_environments tests.test_mirror_residual_moves \
  tests.test_variable_mirror_specs
```

Freeform tests include zero-coefficient equivalence and an independent polynomial-in-ray-distance intersection oracle. Shared-refiner tests cover call budgets, layout invariants, coefficient bounds, incumbent retention, and fit fallback.

To rerun fresh search, use a **new** output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.search_with_freeform \
  --out experiments/online_freeform_NEW --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.analysis.report_online_freeform \
  --out experiments/online_freeform_NEW --workers 8
```

Inspect each saved `protocol.json` for full seeds, hashes, task definitions, and budget accounting before changing a comparison. Existing outputs should remain immutable evidence.

Repository has substantial unrelated modified/untracked work. The new analysis modules are largely untracked. Do not reset, clean, or claim changes are committed. During this session sandbox execution failed with `bwrap`; local shell calls used reviewed `require_escalated` execution. Use `.venv/bin/python`, since bare `python` was unavailable.

## What remains open

The evidence supports a better standardized refinement stage, not a learned potential model, global optimum, or better sampler. Models remain ideal 2D optics with an infinite detector plane, sampled crossing checks, and no manufacturing, tolerance, coating, or diffraction guarantee.

Reasonable next research direction, **not yet executed**: make proposals preserve and use the refined freeform state, then test whether optically informed proposals outperform matched random directions at equal full costs on held-out donor families/specifications. Avoid changing representation, refiner, budget, and sampler simultaneously. If claiming a benefit over random-restart CMA, include that actual baseline; continued CMA is not a substitute.

The original question about predicting attainable quality for an interaction sequence remains open. Finite-budget achieved refinement is an operational signal of potential, not the best possible loss for that sequence. Any future learned potential should model refinement budget and uncertainty and be tested on unseen starting layouts.

Referenced paper for earlier discussion of refinement extent/diffraction-limited volume, Fig. 2:
`agent_context/paper_pdfs/oe-29-17-26736.pdf`. Do not claim this handoff re-reviewed or implemented its method.
