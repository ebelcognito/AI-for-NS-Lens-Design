# Key findings

24 development states, 19 shared parent settings; frozen Qwen3.8-27B Q8_0. Four methods, two attempts each, 16 candidates per attempt. 3072 candidates and 192 selected outcomes, each independently audited on two denser grids. All capacity checks agreed; nine search-grid apparent passes failed final verification.

## Evidence worth inspecting

- Detector misses: Qwen repaired all six states through at least one of its two interfaces, but fixed CMA alone repaired all six. No demonstrated advantage here.
- Too few TIR reflections: case_06 was repaired by fixed CMA in both attempts, but neither Qwen interface repaired it. Qwen improved throughput from 26.9% to 71.4% at best, versus fixed CMA 92.2%.
- TIR-margin losses: both Qwen interfaces repaired all six states at least once, as did random edits. Case_12 direct editing raised worst-field throughput 30.5% → 92.1%; field-mean margin-rejected energy dropped 28.0% → 0.36%.
- C15100 at 4° (case_20): Qwen direct edits achieved RMS 0.9376 → 0.3745 mm at 91.19% throughput; Qwen CMA achieved 0.3922 mm at 90.93%. Each arm passed one of two attempts. Fixed CMA achieved 0.4886 mm but exceeded the body dimension cap by 0.974%.
- C15300 at 4° (case_21): Qwen CMA achieved RMS 0.6876 → 0.3640 mm at 90.58% throughput in one of two attempts. Fixed CMA's 0.4819 mm stress result failed the separate audit grid at 0.6231 mm.
- C15200 at 6° (case_23): fixed CMA achieved 0.2550 mm at 91.07% throughput, but exceeded the dimension cap by 0.912%. Qwen CMA achieved 0.4194 mm at 90.51%, but exceeded it by 1.253%. These are size-constraint near misses, not evidence that sharp 6° imaging is impossible.
- Original A/B and C15100 at 6° did not obtain full repairs. Some severe-blur searches sacrificed throughput because the ranking minimizes the maximum normalized constraint violation. Treat this as an objective-design issue shared across methods.

## Dataset interpretation

Qwen's two arms together repaired 19 states; controls together repaired 18. Seventeen were shared, two Qwen-only, one controls-only, four unresolved. This union gives each side four attempts per state; it is not a single-call success rate. No statistical/generalization claim follows from this small, correlated development cohort.

Only selected outcome labels are densely verified. The 3072 raw candidate labels are provisional. Model rationales are hypotheses, not verified causal explanations. The known_feasible_witness field in frozen tasks refers to source feasibility and must not be used as certification of added imaging targets, especially C15300.

## Next experiment suggested by these results

Keep the current results frozen. For imaging, separate a strict-size benchmark from a predeclared small-size-relaxation benchmark, and prevent search from improving blur by discarding most light. Test feedback-driven direct edits against equal-budget numerical controls on fresh parent designs. No additional run has been launched.

## Browse

- index.html: filterable capability table with all 24 paired setup PNGs.
- RESULTS.md: complete method table and limitations.
- PHYSICAL_EFFECTS.md and case_metrics.csv: before/after physical metrics.
- action_outcomes.json: 192 selected actions/outcomes with provenance.

## Implementation and state

Run root: /home/hisham26/Desktop/NS_lensdesign/runs/optical_capability_map_20260916

Runner: nslens/workbench/optical_capability_map.py; immutable execution snapshot in source_snapshot/. Reports: nslens/analysis/report_optical_capability_map.py, analyze_capability_effects.py, render_capability_pairs.py. Five capability tests and three polynomial-face tests passed. All 192 outcomes complete. Experiment-owned model server PID 3131716 stopped after all 96 decisions. GPU0 workload untouched.
