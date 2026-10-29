# Verified freeform learning loop — 2026-09-10

There is now an executed, audited learning loop. Its demonstrated capability is local
repair of a 2-D transmitting freeform element. Generated physical intervention data
improves held-out repair efficiency. Adaptive task selection has **not** demonstrated
an advantage over uniform task selection, and folded/3-D engineering remains untested.

The implementation and experiments do not depend on EvoTF.

**Measured result.** Three independent training seeds, each with 24 training, eight
validation, and 12 test specification families. Four perturbation severities produce
144 held-out test tasks across 36 families. Every family was certified with a hidden
solvability witness; certification and all subsequent attempts are logged.

| Method | Success after 2 candidate calls | Success after 8 candidate calls |
|---|---:|---:|
| Initial response model, 16 training tasks | 82.6% | 100.0% |
| Continued uniform generation | 89.6% | 100.0% |
| Continued fixed curriculum | 87.5% | 100.0% |
| Continued adaptive curriculum | 88.9% | 100.0% |
| Random local search | 0.7% | 2.8% |
| Reset all freeform coefficients to zero | 0.7% | 0.7% |
| Finite-difference teacher, one guarded step | — | 62.5% |
| Finite-difference initialization + Broyden updates | — | 98.6% |

All methods additionally receive the same initial measurement. Dense verification is
evaluation overhead, not feedback to the search. Training and certification costs are
separate; the results do not establish a net total-compute saving for a few designs.

The adaptive model's gain over its bootstrap is **6.25 percentage points**, with a
family-clustered bootstrap interval of **+2.08 to +11.11 points**. Per-seed gains are
+8.33, +8.33, and +2.08 points. The interval is conditional on the fitted policies;
the separate seed results should not be replaced by a claim of 144 independent trials.

The adaptive-versus-uniform difference is **−0.69 points**. The experiment supports
learning from additional verified data, not superiority of the adaptive sampler.
The eight-call results saturate this domain, so the demonstrated learning benefit is
primarily lower inference call requirements, not a new reachable optical-quality limit.

**What runs end to end.**

1. Sample physical requirements and nominal lens geometry; keep source, pupil,
   detector, material, and tolerances fixed within each task.
2. Optimize and densely certify a freeform witness. Failed certification is recorded
   as unsolved, not impossible.
3. Generate perturbed repair starts without revealing the witness or corruption vector.
4. Measure the current learner's response, finite-difference optical responses, and
   guarded teacher repairs. Retain rejected edits and failed outcomes.
5. Fit a small model of how surface edits change ray errors. It predicts a Jacobian;
   a regularized solve turns current measured error into a corrective edit.
6. Select subsequent severity levels using learner success rates, with replay through
   the accumulated corpus and a nonzero sampling floor.
7. Evaluate candidate checkpoints on whole validation families. Promote only an
   improvement; retain the incumbent otherwise.
8. Freeze every selection before opening the test results. Re-trace final candidates
   on a denser, shifted ray grid with lower branch-pruning thresholds.

The selected adaptive checkpoints were rounds 1, 2, and 2 respectively. Each arm
generated 16 bootstrap tasks plus 192 subsequent tasks. Additional rounds did not
reliably improve the validation score; the promotion mechanism prevented automatic
replacement by a worse checkpoint.

**Verification.** Training uses 15 rays per design; admission uses 119; final test
audits use 279. Success requires overall image-intercept RMS ≤ 0.03 mm, worst-field
RMS ≤ 0.05 mm, at least 75% energy in every primary ray, the declared two-transmission
route, valid geometry, and negligible unresolved/closure energy. Rays outside the
target are scored rather than dropped. Ghost energy is recorded separately.

The main run contains **47,880 evaluator calls**, including certification, training,
validation, and test evaluation. The additional Broyden benchmark contains 1,584 calls.
The artifact audit checked source snapshots, runtime hashes, split isolation, feature
reconstruction without witnesses, label consistency, all attempted-call budgets,
checkpoint hashes, metric recomputation, and nine deterministic physical replays.
All checks passed, including the Broyden budget/metric checks.

Ten new learning-loop tests and 18 existing continuous-boundary physics checks passed.
The separately invoked existing relay suite failed at
`tests/test_freeform_relay.py:424` (`test_a_dead_field_cannot_hide_behind_two_good_ones`):
its historical fixture was not feasible under the current relay configuration. This
work did not modify that test or the relay implementation. The experiment uses its own
explicit imaging specification and the directly tested boundary/tracing kernel; it
does not claim that the full repository suite is green.

**Development finding worth keeping.** A first pilot that imitated accepted optimizer
actions regressed when more examples were added. Learning measured local optical
responses was substantially more stable in a separate development experiment. Both
development runs remain on disk. The reported three-seed experiment used previously
unused seeds, a fixed response learner, fixed budgets and thresholds, and validation-only
checkpoint selection. This is simulator-supervised learning, not literal zero-data AZR.

**Where to inspect and rerun.**

- [Experiment report and figure](../../runs/repair_loop_verified_20260910_v3/REPORT.md)
- [Machine-readable summary](../../runs/repair_loop_verified_20260910_v3/summary.json)
- [Audit results](../../runs/repair_loop_verified_20260910_v3/audit.json)
- [Stronger numerical baseline](../../runs/repair_loop_verified_20260910_v3/broyden_baseline/summary.json)
- [Usage, contracts, and extension guide](../../nslens/learning/REPAIR_LOOP.md)
- [Physics environment and learners](../../nslens/learning/freeform_repair.py)
- [Curriculum experiment driver](../../nslens/analysis/run_repair_curriculum.py)
- [Record and replay auditor](../../nslens/analysis/audit_repair_curriculum.py)
- [Learning-loop tests](../../tests/test_freeform_repair_loop.py)

**What this still needs to become the larger optical-engineering framework.** The
current domain edits four Fourier coefficients on one 2-D transmitting spline. The
train/test families differ in physical requirements and lens envelopes, but share that
architecture. Start by adding more surface modes and held-out requirement ranges, then
adapt the task/verifier contract to route-preserving folded witnesses. Add packaging
and manufacturing constraints explicitly, and validate a stratified subset against an
independent optical engine before making external-fidelity claims. The task sampler
should stay simple until it outperforms uniform generation at equal total data cost.
