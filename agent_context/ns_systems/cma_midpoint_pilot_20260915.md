# Frozen Qwen CMA midpoint pilot — 2026-09-15

The user chose CMA-only control and authorized the next matched midpoint experiment.
This run tests frozen `qwen3.8-27b`, not model training or direct geometry edits.

## Protocol

- All 32 existing development tasks from the interactive pilot, three predeclared
  numerical seeds (2026091501, 2026091502, 2026091503): 96 cases across six families.
- Shared first batch: all parameters, scale 0.025, radius 0.5, 16 candidates.
- From each exact saved midpoint: unchanged continuation, half steps, double steps,
  and one Qwen-selected continuation/restart. Each valid action gets 16 candidates.
- Qwen chooses before alternative arms run and sees only the allowlisted optical
  observation, history, optimizer summary, and budget. No private witnesses or sibling outcomes.
- Continuations preserve saved RNG; a restart uses the recorded seed plus one.
- Dense audit, stress audit, and doubled-capacity stress agreement determine success.
- Primary endpoint: paired audited success, Qwen versus unchanged continuation.
  Half/double controls are separate comparisons. No best-of selection counts as policy performance.
- Model-response exhaustion retains/audits the midpoint; no silent fallback action.
- Three seeds per task are repeated numerical runs, not three independent families.
  Family bootstrap intervals are exploratory given only six development families.
- The experiment does not isolate use of optical context or validate written rationales.

## Status and artifacts

The run was launched; consult `summary.json.complete` for authoritative completion.
At overnight handoff, 55/96 cases were complete: fixed 47, half 45, double 48,
Qwen 46. These are partial results, not the final outcome.
A detached supervisor (PID 2431320 at launch) monitors completion, resumes the
runner if it exits early, generates the final chart/diagnostics, and stops the
experiment-owned Qwen server afterward. See `supervisor_status.json` and
`supervisor.log` in the output directory. Exact numerical replay passed.

The user subsequently requested a nine-hour overnight inception run; see
[overnight handoff](overnight_inception_20260915.md). The CMA experiment remains
active and must not be mistaken for a completed result while the user sleeps.

- [Output directory](/playpen/hisham26/cma_midpoint_pilot_20260915)
- [Summary](/playpen/hisham26/cma_midpoint_pilot_20260915/summary.json)
- [Report](/playpen/hisham26/cma_midpoint_pilot_20260915/RESULTS.md)
- [Single-action preferences](/playpen/hisham26/cma_midpoint_pilot_20260915/decision_preferences.json)
- [Run log](/playpen/hisham26/cma_midpoint_pilot_20260915/run.log)

Source hashes and copies are frozen in the output directory. A full nslens/tests
archive was also captured. Never silently mix changed source into this run.
Output lives on `/playpen` because the root filesystem has very little free space.
The server was launched by this experiment on GPU 1; GPU 0 was already occupied.

## Commands

```bash
.venv/bin/python -m unittest tests.test_cma_midpoint_pilot tests.test_interactive_cma_pilot tests.test_azr_repair_pilot
.venv/bin/python -m nslens.workbench.cma_midpoint_pilot run --out /playpen/hisham26/cma_midpoint_pilot_20260915
.venv/bin/python -m nslens.workbench.cma_midpoint_pilot replay --out /playpen/hisham26/cma_midpoint_pilot_20260915
.venv/bin/python -m nslens.analysis.report_cma_midpoint_pilot /playpen/hisham26/cma_midpoint_pilot_20260915
```

Nine automated tests passed before the run. Replay/diagnostic outcomes should be
read from their output artifacts. The next milestone depends on measured headroom;
do not claim learning or held-out improvement from this frozen development run.
