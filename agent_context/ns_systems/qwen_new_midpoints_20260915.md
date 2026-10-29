# Frozen Qwen on new failing midpoints — remote run

User authorized: “make qwen work on some … remote.” Launched detached on this
server under supervisor PID 2803860. Check live status; no completion claim yet.

- Root: `/playpen/hisham26/qwen_new_midpoints_20260915`.
- Cohort: all 65 coarse-failing midpoint cases from the 218 new development tasks,
  spanning 29 task settings. Selection is based on fixed-CMA midpoint failure.
- Exact first-batch states are reused from saved profiles; their hashes are checked.
- One existing numerical seed per task. Qwen gets one choice of continue/restart,
  then 16 candidates. Unchanged, half-step, and double-step controls get the same budget.
- Frozen local `qwen3.8-27b`; GPU 1. No direct edits or model training.
- Same numerical optical observations and strict action validation as the prior pilot.
  No witness, recipe, or other-arm outcomes enter Qwen's observation.
- Dense final audits and doubled branch-capacity agreement determine success.
- No best-of selection, independent-family, or held-out performance claim.
- The initial 16-candidate batches and generation costs are reused. Arm artifacts
  retain full candidate/ray and audit accounting; avoid counting the shared first
  batch repeatedly when totaling cost.
- Isolated executable snapshot under `code/nslens`; hashes in `source_hashes.json`.
- Supervisor can retry/resume once, writes status and a report, and terminates only
  the Qwen server it started. Nine existing repair/CMA tests passed.

Inspect `status.json`, `run.log`, `summary.json`, `RESULTS.md`, `model_calls/`,
`episodes/`, `cases.json`, `decision_preferences.json`, and `supervisor.log`.
`status.json.phase=complete` is the unattended completion signal; `failed` includes
the exception. No automatic model training or subsequent experiment is scheduled.

Implementation: `nslens/workbench/qwen_saved_midpoints.py`, reusing the existing
matched-arm executor. Do not launch a second runner while the supervisor is active.
