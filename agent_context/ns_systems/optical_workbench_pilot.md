# Optical workbench and first episode corpus

The next step toward a language model that uses optical engineering tools is implemented.
This original milestone built an executable workbench and verified tool-use supervision.
A subsequent local Qwen3-8B experiment now provides actual model results; see
[Qwen workbench results](qwen_optical_workbench_results.md).

**Executable tools:** inspect field/pupil errors and energy/route support; enable selected
entrance/exit polynomial terms; perturb a coefficient; run bounded numerical optimization;
restore a prior best design; verify densely; and submit. All internal optimizer probes are
charged. Requirements cannot be changed through the tools. The task exposes no witness or
exact corruption. The final higher-density audit is hidden until the episode ends.

This domain has six independently editable coefficients, quadratic/cubic/quartic on each
side of a 2-D transmitting spline element. It is a new workbench domain, not the previous
four-global-coefficient learned Jacobian experiment. Folded routes and 3-D surfaces are
still outside the implemented pilot.

**Completed corpus:** 12 certified families; 36 repair tasks; four scripted plan alternatives
per task; 144 episodes. Split sizes are six training families / two validation / four test.
The training exports contain 54 successful conversations and 18 same-start complete-plan
preferences. All 35 failed episodes and their measurements are retained in the full corpus.
The supervision is explicitly labeled `scripted_numerical_teacher`; no LLM reasoning traces
or performance results are fabricated.

| Scripted plan | Held-out success | Mean candidate calls |
|---|---:|---:|
| Enable all six terms | 12/12 | 21.00 |
| Entrance side only | 7/12 | 23.42 |
| Exit side only | 8/12 | 23.50 |
| Quadratic first, then higher terms | 11/12 | 20.00 |

Every plan has a 24-candidate allowance, plus an initial measurement and separately counted
dense/final audits. Actual use differs because an optimizer does not spend unused calls on
fictitious evaluations. Final RMS distinguishes plans even when both satisfy the threshold.

**Interpretation:** the blanket all-terms strategy solves every pilot task. The corpus is
useful for testing tool execution and numerical-outcome supervision, but binary success
does not yet require sophisticated diagnosis. A stronger follow-up should impose physically
justified surface-access, fabrication, or packaging constraints and re-certify its tasks.
Prefer these changes over claiming that the present success metric measures general
engineering judgment.

**Verification:** all 144 full episodes replayed exactly, including intermediate tool
outputs and all physical evaluations. The audit also checked source snapshots, split
isolation, exclusion of hidden answers, training-only successful exports, equal candidate
allowances, and common initial states for preferences. Eight new tests passed, covering
independent surface controls, locked parameters, budget enforcement, verification limits,
hidden-answer rejection, invalid calls, the Responses tool-call protocol, and transport
failure handling.

**Historical dependency (now resolved with local Qwen):** no API-key or model-endpoint environment configuration was present.
The frozen-model runner requires an explicit model and a Responses-compatible endpoint.
It is ready to run against the held-out tasks once configured; offline protocol tests do
not count as a language-model baseline. Do not fine-tune on this corpus before establishing
that baseline and recording which failures are model decisions versus tool/environment limits.

- [Pilot report and example transcript](../../runs/optical_workbench_pilot_20260910_v1/REPORT.md)
- [Full audit](../../runs/optical_workbench_pilot_20260910_v1/audit.json)
- [Model-baseline status](../../runs/optical_workbench_pilot_20260910_v1/model_baseline_status.json)
- [Workbench usage and model configuration](../../nslens/workbench/README.md)
- [Typed tools](../../nslens/workbench/tools.py)
- [Episode generator](../../nslens/workbench/episodes.py)
- [Frozen-model runner](../../nslens/workbench/model_baseline.py)
