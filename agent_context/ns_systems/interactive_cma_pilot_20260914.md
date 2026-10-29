# Interactive Qwen/CMA pilot — 2026-09-14

The user authorized the 32-development-task, four-continuation data-collection pilot
after discussing EvoTune and GLEET. This run uses the local **Qwen3.8-27B Q8_0**
model served by `start_qwen.sh`. Do not substitute the old Qwen3-8B pilot model.
The server reports 27,320,697,856 parameters; the local file and alias identify the
checkpoint used, not independently verified upstream model provenance.

## Live status and artifacts

- [Machine-readable status](/playpen/hisham26/azr_interactive_pilot_20260914/summary.json)
- [Report](/playpen/hisham26/azr_interactive_pilot_20260914/RESULTS.md)
- [Measured outcome chart](/playpen/hisham26/azr_interactive_pilot_20260914/outcomes.png)
- [Raw trajectory preferences](/playpen/hisham26/azr_interactive_pilot_20260914/trajectory_preferences.json)
- [Exact replay](/playpen/hisham26/azr_interactive_pilot_20260914/replay.json)

`summary.json.complete` is authoritative. A partial report must not be described
as a completed run. Results are stored on `/playpen` because the workspace disk
was nearly full. Model weights are frozen: no LoRA or other training is performed
in this data-collection milestone.

## Final result

The run completed all 32 tasks. At the same 32-candidate numerical budget, Qwen's
first-listed continuation repaired 20/32 and fixed all-parameter CMA repaired 27/32.
There were zero Qwen-only wins and seven fixed-only wins in this paired comparison.
The diagnostic best of four Qwen continuations repaired 28/32, but spent four times
the numerical-search budget and is not a fair policy comparison.

Across 128 Qwen branches, 81 repaired successfully. Seventeen tasks had mixed
pass/fail branch outcomes and 21 had verified-reward spread of at least 0.05.
After excluding one branch that exhausted three malformed-response attempts, the
corpus contains 67 complete two-action trajectory preference pairs. These establish
useful control-dependent variation, not trained-policy improvement.

Costs: 5,104 candidate evaluations and 480 final audits across Qwen and baseline;
272,048 solver source rays. Fresh degradation generation used another 373 evaluations
and 27,057 rays. Model serving used 177 requests, 744,970 input tokens and 152,451
output tokens, including reasoning; 18 responses were rejected. One full branch
replayed exactly. Two provisional search passes failed dense final audits.

## Experiment

32 newly degraded starts come from six previously admitted train/validation
families; old test families are excluded. Related starts do not count as
independent design families. Hidden witnesses establish feasibility and never
reach the model. Task generation retains failures and all trace costs.

For each task, Qwen proposes four first CMA actions together. Each action specifies
individual normalized geometry parameter IDs, initial per-parameter search scales,
and a bounded trust-region radius. Each branch executes 16 candidate evaluations.
The model then sees only that branch's numerical results and search history and
chooses a second 16-candidate batch. It can preserve CMA state and change sigma,
or restart on a different parameter subset around the best retained geometry.

The model receives physical corners/sags, gene descriptions, per-field throughput,
root support, detector centroid/spot RMS, energy violations, and search statistics.
It receives no PNGs, exact surface loss maps or per-surface TIR margins. The
canonical route IDs are explicitly distinguished from physical face IDs.
Model diagnoses are hypotheses, not independently verified explanations.

Every branch has two decision points, 32 candidates and three final dense checks:
5x25 audit, 7x51 stress, and 7x51 stress at doubled branch capacity. Brief, material,
and quality requirements are immutable. This version does not include direct
geometry edits, separate probes, or compaction objectives.

The fixed baseline uses all parameters, initial scale 0.025, normalized radius
0.5, and continuous CMA state for the same 32-candidate budget. Qwen's first-listed
branch is compared at equal candidate budget. The best-of-four result is only a
diagnostic: it spends four times as much as one branch.

## Interpretation

Preference pairs require a >=0.05 difference in final verified reward. They compare
whole two-action continuations and are not ready-made single-action DPO labels.
Intervening observations must be retained by a sequential trainer. A single search
seed per strategy cannot separate strategy quality from numerical-search luck.
No trained-policy improvement or held-out generalization claim follows from this pilot.

## Commands

```bash
.venv/bin/python -m nslens.workbench.interactive_cma_pilot run \
  --out /playpen/hisham26/azr_interactive_pilot_20260914
.venv/bin/python -m nslens.analysis.report_interactive_cma_pilot \
  /playpen/hisham26/azr_interactive_pilot_20260914 --replay
.venv/bin/python -m unittest tests.test_interactive_cma_pilot tests.test_azr_repair_pilot
```

The runner resumes saved branches and checks cached observation hashes. Starting
the server is a separate prerequisite. Record source changes before any new run;
do not silently mix configurations within this output directory.
