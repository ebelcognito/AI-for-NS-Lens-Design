# Pencil-beam environment learning loop

## Objective corrected by the user

Demonstrate that an iterative loop discovers better training environments and that
experience from those environments improves the optical-engineering modules. Broad
architecture discovery is an eventual extension, not a prerequisite for this first
experiment. Do not substitute more archive screening for the learning experiment.

## Small physical domain

Use a full source-field width no greater than 1 mm, initially three source locations
spanning 54.5–55.5 mm at y=50 mm. Field width and angular spread are separate: the first
screen uses a finite ±0.25 degree angular fan. Keep the continuous closed freeform
boundary, actual TIR events, Fresnel energy accounting, and hidden denser ray checks.
This is a 2-D routing experiment, not a wide-field imaging certification. Consequently
there is no inherited 12-degree FOV requirement or image-field ordering requirement.

Start with an available verified folded seed. Two-or-more-fold proposals can join when
verified; failure to find them must not hold up the learning-loop experiment. Do not
count negligible ghost energy as a useful extra fold. Do not silently change material
indices or relax geometry validity to make a seed pass.

## What constitutes an environment

An environment specifies a fixed source bundle, output target, editable coefficients,
shape-change limits, initial shape corruption, and trace budget. These remain immutable
during solving. The proposer operates within declared parameter bounds and does not
control acceptance thresholds, reward code, hidden audits, or evaluation task selection.
A private feasible witness may be used initially, with every solver-visible field
explicitly separated from proposal-only information.

## Minimum experiment

1. Freeze small training, validation and test task pools before policy selection. Group
   siblings by their underlying environment. If they all descend from one optical seed,
   report within-architecture generalization; do not claim unseen-architecture transfer.
2. Measure frozen Qwen on the training pool. Numerical controls remain useful, but a
   numerical optimizer solving these tasks does not disqualify a curriculum experiment:
   the question is whether selected experience teaches Qwen more effectively.
3. Run uniform, fixed-curriculum, and adaptive-environment arms from the same Qwen base.
   Use the same teacher, update rule, retained-history fraction, seeds, and training-token
   and simulator-call budgets. Count task feasibility checks, proposer scoring rollouts,
   and rejected proposals in generation cost. Equal episode counts alone are insufficient.
4. For the first adaptive arm, use a transparent sampler that explores task parameters
   and updates its sampling distribution from current solver outcomes. Mixed success is
   a difficulty heuristic, not measured learning gain. Include a nonzero coverage floor
   and retain earlier environments to check forgetting. Label this as an adaptive sampler,
   not a learned AZR proposer. Upgrade to a Qwen proposer only after this control works.
5. Execute at least two collect → train → evaluate rounds. Train only on verified training
   trajectories, mask simulator responses from prediction targets, and retain failures.
   The model must actually change; successful optimization in an episode is not learning.
6. Select checkpoints using validation only. Once all arms and rounds are fixed, evaluate
   the final untouched tasks. Log optical success, quality, constraint violations, rays,
   tokens, and retention. Show learning curves and uncertainty across seeds/environment
   groups. A small pilot can be inconclusive; report that rather than declaring improvement.

## Claim boundary

The desired positive result is: at equal total learning cost, the adaptive arm produces
better held-out optical outcomes than both its frozen starting point and nonadaptive
training controls. Improvement over frozen Qwen alone demonstrates learning, not that
environment discovery caused the improvement. Increasing language or simulation budgets
cannot count as training progress.

A learned task proposer, reinforcement learning, all three AZR reasoning modes, broad
folded topology diversity and production optical engineering are later extensions.
This contract does not claim that any folded Qwen learning gain has been measured yet.

## Initial screen result

The 1 mm, ±0.25° profile admits four of 24 screened archive-derived designs when
at least one useful TIR is required. All four pass coarse, 183-ray audit, and 273-ray
sampling checks. None of the same 24 candidates passes with at least two useful TIRs
required. These are distinct prescriptions with unresolved shared archive ancestry.
Results are in `runs/pencil_singlefold_screen_20260910_v1/` and
`runs/pencil_folded_screen_20260910_v1/`. The evaluator accepts environment-owned
layout and requirement profiles without mutating the older wide-field defaults.
The legacy solver still accepts only its original fixed profile; the pencil screen
is not yet a Qwen training loop.

## Executed follow-up

The [two-round Qwen pilot](pencil_qwen_learning_results.md) now implements this first
finite-pool selection loop and trains six adapters. Adaptive and fixed curricula improve
the primary test result from 0/4 to 3/4; uniform does not. Adaptive selection does not
beat the fixed schedule, and useful new-environment synthesis remains unimplemented.
