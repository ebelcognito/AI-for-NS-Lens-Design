# Single-fold focusing prism benchmark (environment-native)

Benchmarks a model on **designing** a folded focusing prism in our own `fold_demo`
environment — not on emitting a CODE V `.seq` file. Built 2026-07-29.

| file | role |
|---|---|
| `benchmark_fold1_prompt.md` | the prompt. Paste everything below the rule into the model under test. |
| `verify_fold1.py` | the grader. `python3 ns_probe/benchmarks/fold1_env/verify_fold1.py design.json [--render spot.png]` — 13 checks, exits 0 only on a full pass. Reveals no reference solution, so the model under test may run it. |
| `benchmark_fold1_key.md` | reference solution, measured tolerances, rubric, known-bad calibration. **Never show the model under test.** |

The task: nine numbers (element + beam) such that a collimated 8 mm beam from infinity is
folded by ≥ 1 genuine TIR and brought to a focus on the detector at x = 120, at ≥ 0.80
throughput, with the source outside the glass. Single field.

**Protocol is commit-then-verify:** the model reasons analytically (its own tracer is
allowed, the engine is not), commits nine numbers plus predictions, then gets one
verification run and at most one correction. This caps token burn — the failure mode this
variant was built to prevent is open-ended iteration against the engine.

Satisfiability was verified before the spec was written (the mistake that broke the Tier A
CODE V prism benchmark): the reference solution passes 13/13 with RMS spot 0.0008 mm at
0.880 throughput, and the pass region tolerates ±20% on the strongest curvature parameter.
Re-check with `verify_fold1.py` after any threshold edit.

Related: `../../results_fold_qd/codev_export/` holds the older CODE V `.seq` benchmarks
(`benchmark_fold4_prompt.md` for one-shot `.seq` grading, `BENCHMARK_PROMPT_V2.md` for the
5-field Tier A prism).
