# Nine-hour overnight inception — 2026-09-15

The user explicitly requested a nine-hour discovery/inception run before going to
bed. It was launched as detached work on the current server, alongside the ongoing
frozen-Qwen midpoint experiment. No model training or direct model geometry edits.

## Launch and scope

- Supervisor PID at launch: 2431321, session leader with parent PID 1.
- Root: `/playpen/hisham26/overnight_inception_20260915`.
- Search: nine hours, eight CPU workers, 2,000 physical calls per job, portfolio CMA.
- Fixed refractive index: 1.50 to align with current repair tasks.
- Shuffled 216-setting grid: 3/4/5 corners, 4/6 mm beams, four receiver layouts,
  three angular ranges, and standard/small/thin size priors. New seed 2026091500.
- Search deadline: **2026-09-15 10:07:58 EDT**. Final in-flight audits and repair
  certification add time. Automatic videos disabled to prioritize designs.
- Code runs from an isolated snapshot under `code/nslens`; workspace edits do not
  change the active discovery worker imports. Source hashes are in discovery/protocol.json.
- Each job saves candidate histories, its config, audited archive, and summary.
- After search, `harvest_inception` inventories exact-distinct audited candidates
  and certifies ≥80% worst-field folded throughput and root support, retaining
  stricter original constraints, using 5x25 / 7x51 and doubled-capacity checks.
- Inventory count is not independent-family count. No train/test split assigned;
  preserve ancestry and group related geometry/task settings before future learning.

## Where to check

- [Supervisor status](/playpen/hisham26/overnight_inception_20260915/status.json)
- [Search log](/playpen/hisham26/overnight_inception_20260915/search.log)
- [Incremental search summaries](/playpen/hisham26/overnight_inception_20260915/discovery/summary.json)
- After completion: `discovery/completion.json`, `discovery/inventory.json`,
  `discovery/repair_ready_designs.json`, `discovery/harvest_summary.json`,
  `discovery/RESULTS.md`, and `harvest.log`.

`status.json.phase=complete` means search AND harvesting finished. `failed` records
an exception; consult supervisor.log. Do not relaunch long_search into the same
output directory: this runner requires a fresh path and its candidate records are
replay histories, not resumable CMA checkpoints.

## Validation

Eight discovery/portfolio tests passed. A short real search completed three jobs
and saved its completion artifact. Empty-inventory harvesting completed correctly.
The certification function also rechecked a known passing midpoint-control geometry
and agreed with its previously audited reward. Smoke artifacts are separate from
the nine-hour discovery output and must not be included in its totals.

This is exploratory 2D circular-face folded-body discovery, not imaging qualification,
compaction, AZR training, or independent-engine verification. More raw designs do
not guarantee more strict-quality witnesses or better CMA decision tasks.
