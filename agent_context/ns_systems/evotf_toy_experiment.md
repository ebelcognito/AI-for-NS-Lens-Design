# EvoTF toy experiment

## Question

Can the released Evolution Transformer improve the evaluation efficiency of the
current search, and what part of the proposer–compactor design would plausibly
provide that gain?

## Method

The experiment uses the official `evosax.algorithms.EvoTF_ES` class from evosax
0.3.1 and its bundled `2024_03_SNES_small.pkl` checkpoint.  This is the released
21,010-parameter model distilled from SNES; no local approximation of EvoTF was
substituted.

It compares:

1. the repository's full-covariance `nslens.search.cma_me.CMAES`;
2. evosax SNES, the diagonal teacher-family baseline; and
3. the released pretrained EvoTF-ES.

Each paired run uses an 8-dimensional problem, population size 10, 50 generations,
the same initial mean, and 501 objective calls (the initial mean plus 500 population
members).  There are 20 deterministic instances of each of five function families,
for 100 runs per optimizer.  The suite includes both separable and randomly rotated
objectives so that a diagonal distribution is not rewarded accidentally.

These are independently generated synthetic instances.  They should not be called
held-out function families because the public checkpoint was trained on a subset of
BBOB and may have seen related landscapes.  No optical trace is used, so this tests
the optimizer representation and released checkpoint, not optical performance.

Run it with:

```bash
python -m pip install evosax==0.3.1
python -m nslens.analysis.run_evotf_toy \
  --seeds 20 --generations 50 --output results/evotf_toy.json
```

## Results

Lower objective values are better.

| Function family | CMA median | SNES median | EvoTF median | EvoTF wins vs CMA | EvoTF wins vs SNES |
|---|---:|---:|---:|---:|---:|
| Sphere | 0.000790 | 0.00475 | 0.0748 | 0/20 | 0/20 |
| Separable ellipsoid | 6.77 | 0.324 | 6.51 | 11/20 | 0/20 |
| Rotated ellipsoid | 4.10 | 12.15 | 28.14 | 1/20 | 2/20 |
| Rotated Rosenbrock | 7.25 | 7.22 | 14.03 | 1/20 | 0/20 |
| Rotated Rastrigin | 27.79 | 32.57 | 36.91 | 3/20 | 5/20 |
| **All paired runs** | — | — | — | **16/100** | **7/100** |

Across all tasks, median log10 relative regret was -2.388 for CMA, -2.812 for
SNES, and -2.069 for EvoTF.  The median paired final objective ratio was 2.25 for
EvoTF/CMA and 2.32 for EvoTF/SNES.  At generation 10 (101 calls), EvoTF beat CMA
on 39/100 runs and SNES on 26/100; its gap widened with additional evaluations.

Median optimizer time per 50-generation run on CPU, after JIT compilation was
amortized, was about 0.024 seconds for CMA, 0.046 seconds for SNES, and 0.303
seconds for EvoTF.  For a costly ray tracer this overhead would usually be small
relative to evaluation time, but an equal wall-clock comparison should still report
it.

## Interpretation

The public pretrained EvoTF is not a plug-in performance improvement for this
setting.  It loses to full CMA on 84% of paired runs and also loses to its SNES
teacher baseline on 93%.  The rotated ellipsoid result exposes the expected cost of
its diagonal Gaussian, while the losses on sphere and separable objectives show
that covariance alone does not explain the result.  Checkpoint/task mismatch and
off-policy rollout error also matter: a distilled model predicts teacher updates on
teacher histories during training, then consumes its own histories at deployment.

EvoTF can still help the proposed optical system through amortization.  A model
trained on many verified optical searches can use population history, failure
patterns, requirements, route persistence, and archive context to choose a better
mean or step scale earlier than a fresh CMA emitter can infer them from one run.
That gain would appear as fewer expensive trace calls to reach the same feasible
fitness or package size.  The generic BBOB checkpoint has never learned those
optical regularities, so this experiment gives it no such advantage.

The next optical implementation should therefore use EvoTF as a correction head:

- retain CMA's full covariance and positive-definite update;
- predict bounded mean and step-scale corrections in normalized gene units;
- add requirement, gene-type, route, validity, and constraint-margin features;
- train first by distilling complete CMA population transitions, including failed
  candidates, then optionally optimize terminal verified compaction reward; and
- compare against CMA, SNES, archive retrieval plus perturbation, and proposer-only
  warm starts at equal trace budgets on held-out lens programs or geometry basins.

The synthetic result is negative for immediate deployment and positive only as a
justification for testing an optics-trained hybrid.  It does not support replacing
the current CMA covariance update with the released diagonal EvoTF update.

