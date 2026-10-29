# Verified freeform repair curriculum

This experiment closes a small learning loop: certify solvable optical tasks, collect
measured interventions, fit a response model, use current learner performance to select
new tasks, and promote checkpoints on held-out validation families. Final test families
are evaluated only after every curriculum arm has finished training and selection.

It is a 2-D, monochromatic transmitting-element study using the existing continuous
periodic-spline boundary and non-sequential Fresnel tracer. Four radial Fourier
coefficients are editable. It does not establish folded-route discovery, 3-D freeform
imaging, manufacturability, or agreement with CODE V.

## Run and inspect

From the repository root, using the environment with NumPy, SciPy, and scikit-learn:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.analysis.run_repair_curriculum \
  --out runs/repair_loop_new \
  --seeds 17 29 43 --workers 12 \
  --train-families 24 --validation-families 8 --test-families 12 \
  --bootstrap 16 --batch 64 --rounds 3

.venv/bin/python -m nslens.analysis.audit_repair_curriculum runs/repair_loop_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m nslens.analysis.benchmark_repair_broyden runs/repair_loop_new --workers 12
.venv/bin/python -m nslens.analysis.summarize_repair_curriculum runs/repair_loop_new

.venv/bin/python -m unittest discover -s tests -p test_freeform_repair_loop.py
```

Use a new output directory for each run. A run records configuration, source hashes,
executable source snapshots, environment versions, all attempted evaluations, training
examples, curriculum decisions, checkpoint hashes, and final metrics. The source snapshot
matters because this workspace contains pre-existing uncommitted development.

## What is learned

`ResponsePolicy` predicts the local Jacobian from surface coefficients to the 15 coarse
ray-intercept errors. Its inputs are the current coefficients and physical specification;
the measured current ray errors then enter a regularized local solve for the next edit.
Finite-difference response labels come from four measured surface perturbations. Inference
predicts those responses instead of tracing four new probes at each step.

This differs from imitating an optimizer's accepted step. The original `RepairPolicy`
implements that action-imitation baseline and is retained so development checkpoints remain
readable. In the first development experiment, adding action demonstrations degraded
generalization. The response formulation was selected using separate development runs,
before the three-seed verification experiment.

Each generated training task consumes exactly ten coarse evaluations: its initial state,
one current-learner probe, four finite-difference probes, and four teacher line-search
candidates. Physically valid response probes are useful even when the teacher does not
find a better edit. All failed and rejected attempts remain in `evaluations.jsonl`.

## Task and verifier contracts

- A family fixes material, nominal lens envelope, pupil, field angles, image-plane
  position, and acceptance thresholds. These cannot change during repair.
- A certified witness is created by four bounded local-optimizer steps and checked on
  a denser grid. Certification failures remain recorded and are not called impossible.
- Repair starts perturb a witness at one of four severity levels. The witness and exact
  corruption vector are excluded from solver inputs. Family IDs and severity are also
  excluded from policy features.
- The evaluator measures image error for every launched pupil ray. Missing rays cannot
  disappear from an RMS score. Every primary ray must deliver at least 75% energy and
  realize the declared two-transmission route; Fresnel ghosts and unresolved energy are
  accounted separately.
- Acceptance requires overall RMS at most 0.03 mm and worst-field RMS at most 0.05 mm,
  plus geometry, route, energy, and numerical-closure gates.
- Optimization uses 3 fields × 5 pupil rays. Dense checks use 7 × 17; final audits use
  9 × 31, shifted pupil interiors, and lower branch-pruning thresholds. Neither check
  supplies feedback to the search. This is additional sampling of the same physics,
  not an independent optical engine.

## Fair comparison and promotion

Uniform, fixed-stage, and adaptive curricula share the bootstrap, candidate learner,
teacher, family pool, budgets, and held-out task sets. Adaptive selection uses a smoothed
estimate of one-step learner success per severity, preferring mixed outcomes while
retaining a sampling floor. It is a learnability heuristic, not a causal estimate of
training benefit and not a learned AZR task proposer.

Every arm generates the same number of training tasks. Only a checkpoint with better
dense validation success, or equal success and lower mean error, replaces the incumbent.
Unpromoted checkpoints and their data are retained. All selection decisions precede the
test runs.

The primary comparison is after **two candidate calls**, plus the common initial
measurement. The secondary comparison allows eight candidate calls. Saved prefix states
are evaluated after the episode; dense checks never choose the next action. The numerical
teacher requires four probes before it can form a correction, so its two-call prefix is
not a useful strong baseline. Compare the learned method to its completed eight-call step
and the stronger finite-difference/Broyden baseline, as well as to equal-budget random
search and coefficient reset. Broyden spends four calls initializing its Jacobian and
reuses it with secant updates for four subsequent corrections. Training and certification
costs are separate from inference savings and must be amortized over future tasks.

Uncertainty intervals resample entire test families, keeping the four severity siblings
together. They are conditional on the fitted policies; the three separate training seeds
must also be examined. Exact split disjointness does not mean the test distribution is a
new optical architecture: all families belong to the same declared transmitting domain.

## Extending the experiment

The first extension should make verification harder before increasing model size:

1. Increase the number of freeform modes, then add a held-out field/pupil range. Keep
   the same logging, split, promotion, and call-accounting contracts.
2. Add a route-preserving folded-domain adapter using certified folded witnesses. Replace
   this domain's explicit two-transmission admission rule and observation grid; do not
   pool its labels silently with transmission-only examples.
3. Introduce packaging, clearance, and fabrication constraints as separate labels and
   hard gates. Preserve all outcomes for response/failure prediction.
4. Pair a stratified subset with an independent tracer or CODE V, including failures
   and high-disagreement cases. Version the verifier and re-certify witnesses on changes.
5. Compare adaptive generation to the strongest fixed sampler at matched total generation
   cost. Only increase sampler complexity if it improves held-out engineering outcomes.

Do not describe the current result as zero-data AZR: it uses simulator-generated
finite-difference supervision and deliberately constructed solvable tasks. Its contribution
is an executable, auditable learning loop that can test whether new experience helps.
