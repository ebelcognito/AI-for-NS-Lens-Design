# 20 — RL milestones 1–2: the reward service and the task supply

**Date:** 2026-08-14. Implements milestones 1 and 2 of
[18_rl_engine_reward_plan.md](18_rl_engine_reward_plan.md) §7, using the topology axis
adopted in [19_topology_axis.md](19_topology_axis.md) as the conditioning label.

Code: `nslens/rl/` (`env_config`, `reward`, `parse`, `server`, `tasks/{generate,certify,splits}`),
tests `tests/test_rl_env_hardened.py` and `tests/test_rl_reward.py`, both wired into
`tests/run_all.py`.

---

## 0. What doc 18 got wrong, and it matters

Four of the plan's measured facts did not survive being implemented. Three are corrections
to numbers; one is a defect in a metric this repo has been quoting for months.

### 0.1 The verifier is ~3.3× more expensive than claimed

Doc 18 §0.1 reports **291 nine-field designs/s at 20 workers** and builds every subsequent
decision on it ("a GRPO step with 256 rollouts costs ≈0.9 s of reward"). Reproducing that
benchmark through the actual reward path:

| population | 1 worker | 4 | 10 | 20 | reached the engine |
|---|---:|---:|---:|---:|---:|
| uniform random genotypes | — | — | — | **290/s** | 30% |
| perturbed archive elites | 6.9/s | 31.3/s | 68.7/s | **88.2/s** | 91% |

The 291 was measured on uniform genotypes, and **70% of those fail a geometry test before a
ray is traced**. It is a benchmark of the rejection path. On designs that actually trace —
which is what a trained policy proposes — the reward service sustains **88 designs/s at 20
workers**.

Where the time goes, profiled on 40 real designs (`cProfile`, cumulative):

| | share |
|---|---:|
| engine round-trip (`select.select` blocking on Node) | 52% |
| `Observation.convergence` (beam legs, crossings, exit) | 39% |
| everything else (descriptors, measure, margins) | 9% |

So doc 18's *engine* number was right — 165 engine calls/s single-worker against its 197 —
and its inference was wrong. The claim "evaluate every design at all 9 fields, run every
physical gate, all of it is noise against generation cost" priced only the tracing. The
per-field beam diagnostics cost about as much as the tracing does.

**The conclusion survives; the margin does not.** A 256-rollout GRPO step now costs ≈2.9 s
of reward against 30–60 s of generation — 5–9% of the step rather than 2%. Large group
sizes are still nearly free and there is still nothing to amortise with a surrogate. But
"the verifier is effectively free" should be quoted as "the verifier is under a tenth of the
step", and any plan that assumed a 10× headroom for extra reward computation does not have it.

One small win banked while measuring: `convergence()` walked the beam legs and then called
`focus_crossings()`, which walked them again with the same arguments. `focus_crossings` now
accepts the caller's legs, and `beam_legs` memoises. `tests/run_all.py` is 9/9 after the
change, so no number moved.

### 0.2 `crit_margin` measures the wrong ray

The registered `crit_margin` descriptor takes the **single brightest ray** and reports how
far its TIR events clear the critical angle. On the catalogue's 0.742 four-fold design that
ray carries 0.79 of the energy and goes **straight through the element** — `T@A -> T@B`, no
TIR at all — while the 11-ray bundle that defines the signature and sets the fitness folds
four times. The descriptor returns its no-TIR sentinel (−5.0) for a design whose entire
point is four folds.

This is not exotic. Fitness is the *worst field* over a bundle; the signature, the archive
cell and every beam diagnostic are properties of the *dominant bundle*. `crit_margin` is the
only one that is not, and it is an `info_descriptor` on every fold24 run, so the column is in
the archives.

`reward.dominant_crit_margin` measures the minimum over every TIR event of every ray in the
dominant bundle. It is much stricter, and the number it produces changes the story below.

### 0.3 The 3° margin floor does not transfer, and would have rejected the best design

Doc 18's exploit table prescribes "margin floor ≥ 3 mm", from `thin_design.py --min-margin 3`
(the units are degrees). Applied to the **bundle** statistic and priced on the 95 ab40k
elites:

| floor | elites kept | top ten kept | best design kept |
|---:|---:|---:|---:|
| 0.5° | 59/85 | **10/10** | 0.878 |
| 1.0° | 53/85 | 7/10 | 0.878 |
| 2.0° | 42/85 | 3/10 | 0.858 |
| 3.0° | 34/85 | **1/10** | 0.843 |

At the prescribed 3° the gate discards **nine of the ten best designs in the program**,
including the 0.878 that is the best fitness ever reached here. The 3 came from a statistic
computed on one ray at one field; the bundle statistic across nine fields is a different
quantity and needed its own pricing. This is exactly the etendue-growth mistake the driver's
own comments warn about — caught before shipping this time rather than after.

Resolution: floor at **0.5°** (costs nothing at the top), and grade the reward on a ramp from
0.5° to 3° so robustness keeps being paid for above the floor instead of falling off a cliff.

**The finding underneath deserves its own line.** The fold24 family folds on **0.6–2° of
margin** at its best. By the standard the single-ray descriptor implies, every one of these
designs is knife-edge, and a tolerance study is owed before any of them is called buildable.

A related trap, found while measuring: −5.0 is a *sentinel*, and `min()`-ing over it treats
"this bundle does not TIR here" as a margin of −5°. Because 36 of 40 fold24_clean elites
change dominant signature across the field sweep, an edge field that stops folding would drag
the worst-field margin to the sentinel and the gate would read a topology change as a knife
edge. Skipping sentinels moves the median margin over the ab40k elites from −5.0 to 0.66.

### 0.4 Certification is billed per variant, not per spec

Doc 18 §3.1 costs certification at one 2,000-eval probe *per spec* (~7 s each, ~10⁴ specs per
20-core day). That is not how the driver works: **one probe fills every cell it can reach**,
so a single probe certifies a dozen or more specs at once. The bill is set by the number of
*variants* (25 per environment), not by the number of specs. Certification is therefore far
cheaper than planned — minutes, not a day.

The catch is in the other direction, and it is real: **the probes must warm-start**. A ~15%
acceptance gate stack cannot cold-start (measured previously: 0–3 of 48 cells in 4k evals,
best fitness 0.03), which is why every archive in this program resumes from a seed. So for a
cell the seed already occupies, `achievable` may be the seed's own score rather than anything
the probe found. Every certified spec carries `from_seed` so a curriculum can tell the
difference.

---

## 1. What was built

### 1.1 `env_config` — the hardened environment, and the record of why

Twelve knobs differ from the frozen `ExperimentConfig` defaults. Each frozen default exists
so `tests/test_qd_equivalence.py` can reproduce committed archives cell-for-cell, and each is
also an exploit search already found. `DIVERGENCES` is doc 18's exploit table in executable
form; `tests/test_rl_env_hardened.py` asserts every row still differs, and **fails naming the
exploit** if somebody tidies the file back toward the defaults.

Two gates have no `ExperimentConfig` counterpart and live on `RLEnv` instead, so the frozen
search path is untouched: the crit-margin floor (§0.3) and a source-clearance test for the
embedded-source cheat.

Two metrics are recorded as **explicitly not gates**, with the reason attached, because both
were measured to reject the best designs in the program: `etendue_growth` (a collimated
source makes any focusing read as 6–10× growth) and the ratio-flavoured exit tests.

The first behaviour axis is `tir_faces`, not `tir_bounces`. Under fold counts a spec can name
three attainable values, which is not a conditioning label; the topology axis resolves 79
observed classes and its cells *are* ray paths. This is the doc 19 result doing work.

**The spec space needed widening.** The descriptor grid is 121 topology bins × 4 thinness
bins = 484 cells, of which ~79 topologies are ever realised — the same order as the 695
(experiment, cell) specs already on disk, and therefore subject to the same objection doc 18
§0.3 raises against the archive. Half-FOV and detector-window width become spec axes
(5 × 5 variants), which puts the space at O(10⁴) as §3.1 requires, with every variant a
genuinely different optical requirement rather than a relabelling.

**One cost recorded rather than removed**: the genome box is copied verbatim from
`fold24_topo_gated` and reaches sag ratios about twice the 0.268 cap, so measured on 4,000
uniform genotypes only 28% of the fold box is buildable (58% bulb faces) and only 6% of the
prism box is. Retuning the box would raise throughput and would also make this a different
search space from the one every archive was built in, invalidating the equal-engine-call
comparison of doc 18 §6 item 3. Comparability is worth more.

### 1.2 `reward` — gates score, they do not reject

`driver._evaluate` returns `None` the moment a gate fails. That is right for CMA-ME and wrong
for a policy gradient, which needs to know that *this* failure was a bulb face and *that* one
was a caustic — and needs partial credit in between, or every rollout in a group scores the
same and the step is wasted (§4.3 advantage collapse).

Stages, worst to best: `degenerate`, `bulb_face`, `detector_intersect`, `embedded_source`,
`engine_error` → `no_light` → `off_class` → `scored`. Reward follows doc 18 §2.2:
0.15 geometry + 0.35·f_norm + 0.20·niche + 0.15·gate, with `f_norm` normalised by the spec's
certified achievable score.

Three deliberate departures, each because the plan's version is a cliff:

- **`niche_match` is graded** on multiset symmetric difference, not vocabulary index
  distance. The enumeration is grouped by fold depth, so neighbouring indices are unrelated
  routes; what a designer calls "close" is how many bounces you must add or remove.
- **`gate_score` is graded** over per-field fractions rather than the all-fields verdict.
- **the crit-margin floor is not a stage.** As a cliff it sits below 26 of 85 known-good
  elites. It is enforced where rejection *is* right — `tasks.certify.harden` refuses to
  certify a spec whose only known answer is a knife edge, so no spec ever asks for one.

`off_class` — buildable, images, wrong fold class — earns geometry credit and nothing else.
Paying it fitness credit would reward the doc 19 §5.1 failure directly: straight transmission
scores ~0.92 *because* it pays no fold losses, so any task-blind fitness term makes "do not
fold" dominant.

`tests/test_rl_reward.py` asserts fitness and behaviour coordinates are **bit-identical to
`driver._evaluate`** on every elite the driver accepts (32/40 checked). Without that the RL
result cannot be compared to the CMA-ME baseline at all.

### 1.3 `parse` — strict, with a taxonomy

A lenient parser that repairs the model's JSON hides the format-failure rate, and doc 18
§4.2 makes that rate a go/no-go gate on the SFT warm start. Failures are labelled
(`no_design_block`, `bad_json`, `not_an_object`, `missing_keys`, `unknown_keys`,
`non_numeric`, `out_of_bounds`) so the training log can separate "does not know the format"
from "knows the format, proposes absurd numbers" — different fixes, identical in a scalar.

One level added to doc 18's ladder: out-of-bounds scores 0.02, between malformed (0.00) and
in-bounds (0.05). Collapsing them throws away the only signal that says the model learned
the format.

### 1.4 `server` — the pool

Long-lived pool of engine workers behind a batch call. The cache stores **verdicts, not
rewards**: a verdict depends only on (variant, genotype), a reward additionally on the spec
that asked, so caching the verdict makes a design free across every spec mentioning it.
Cache keys are the full variant triple — the same genotype at a different FOV is a different
measurement, and sharing an entry would hand the training loop a score for a spec it never
evaluated. Timeouts come back as `engine_error` and unblock the step rather than stalling it.

### 1.5 `tasks` — generate, certify, split

`certify` runs CMA-ME probes per variant, harvests every cell they filled, and lets the score
reached *be* the spec's `achievable`. Cells no probe reached do not become specs. This is the
rule the Tier A V1 benchmark broke — it shipped a prism spec where only 2 of 5 fields could
ever close — and it is why nothing here ships uncertified.

`splits` holds out by **topology, never randomly**: the thinness axis of one topology is the
same optic at a different thickness, and two ray paths account for 60% of the fold-count
archive, so a random split measures memorisation and reports generalisation. `test_cell` is
held-out topologies inside a trained env (interpolation); `test_env` is the whole `prism`
environment (transfer).

`curriculum_weights` samples into the 0.2–0.8 pass band. Before a policy exists there is no
pass rate, so the probe's own hit rate across seeds stands in.

---

## 2. Certification results — and the one that matters

`nslens/rl/tasks/certify.py`, 50 variants × 3 seeds × 2,000 evals = 150 probes, 313 s on 20
workers. Output `runs/rl_tasks/certified.json`.

| | |
|---|---:|
| cells filled across all probes | 288 |
| unique (variant, cell) | 123 |
| certified (achievable ≥ 0.05) | 99 |
| survive `harden` (certifying design not a knife edge) | **86** |
| carried over from the warm-start seed | 34 / 86 |
| **distinct topologies across the whole task set** | **6** |
| variants that certified anything | 24 / 50 |
| `prism` specs certified | **0** |

Achievable spread is healthy — 5th/50th/95th percentile 0.298 / 0.765 / 0.862 — and the
difficulty signal is real: 60 specs hit by all three probe seeds, 15 by two, 11 by one. FOV
falls off exactly as physics predicts (26 specs at ±4°, 6 at ±12°).

Everything else is bad news, and it is the useful part.

### 2.1 Procedural generation produced FEWER specs than the archive it replaces

Doc 18 §0.3 rejects the archive as a task source because 533 (now 695) unique specs is a test
set, not a distribution, and prescribes procedural generation to fix it. Measured: generation
plus certification yields **86**.

That is not a failure of the implementation, it is the certification step doing its job. The
695 archive specs were never certified. The honest count of *specs backed by a design a probe
can reach under gates that reject the known exploits* is 86. The other ~600 are cells that
were filled once, under permissive gates, by designs that are bulbs, intersect the detector,
or never form an image.

**So doc 18's diagnosis was right and its remedy does not fix it.** Task supply is the
binding constraint on this whole arm, and it is worse than the plan thought.

### 2.2 The task set is six ray paths, and one of them is 45% of it

```
A1 B1 D1 T1   39      A1 B1 D1   20      A1 B1 T1   16
B2 T1          8      B2 D1 T1    2      A2 D1 T1    1
```

Six topologies. The whole point of conditioning on `tir_faces` is that it resolves 79 observed
classes; a task set spanning six of them cannot teach a policy to hit a route on request, and
`test_cell` — held out by topology, as it must be — ends up with **2 specs and one topology**.
There is no interpolation split worth reporting.

The likely cause is probe budget: 2,000 evals under a ~15% acceptance gate is ~300 valid
designs across a 484-cell grid, and the ab40k runs needed 40,000 evals per seed to retain 17
distinct signatures. Doc 18's 2,000 was calibrated when it believed one probe certified one
spec; since one probe harvests every cell it reaches (§0.4), probes can be an order of
magnitude bigger for the same total bill. §2.4 tests that.

### 2.3 `prism` certified nothing at all, so there is no transfer split

75 probes, 150,000 evaluations, **zero cells filled**. The held-out environment cannot
cold-start: only 6% of its genome box is buildable under the hardened geometry (55% bulb
faces, 39% degenerate), and with no seed archive the CMA-ME emitters never get an elite to
sample around. This is the same 15%-gate cold-start problem that forced every fold24 run to
warm-start, but worse.

`test_env` is therefore empty, and the transfer claim of doc 18 §3.2 currently has nothing
behind it. Two ways out, and they are not equivalent:

1. **Build a prism seed the way fold24's lineage was built** — a permissive-gate run to
   populate the archive, then filter its elites through the hardened config, then warm-start
   the probes from what survives. This is the honest fix and it costs one extra run.
2. **Hold out a FOV band inside `fold` instead** (train on ±4–8°, test on ±10–12°). Available
   today, but a much weaker claim: it is the same geometry at a wider field, not a different
   optical problem, and doc 16's whole point was that the perpendicular geometry behaves
   differently.

Option 1 is the one worth doing. Option 2 should not be quietly substituted and called
transfer.

### 2.4 Probe budget explains much of the shortfall, but not all of it

Two variants (`fold` at ±8°/±8 mm and ±12°/±4 mm), 3 seeds each, same code, only the eval
budget changed:

| evals/probe | specs | distinct topologies | carried from seed | best achievable |
|---:|---:|---:|---:|---:|
| 2,000 | 5 | 3 | 2 / 5 | 0.860 |
| **20,000** | **11** | **7** | **1 / 11** | **0.878** |

A 10× budget buys 2.2× the specs and 2.3× the topologies, and — the more telling column —
**drops seed carry-over from 40% to 9%**. At 2k the probe is mostly re-certifying what it was
handed; at 20k it is finding its own cells, and it reaches the program's best fitness (0.878)
rather than stopping at the seed's 0.860.

This is consistent with the ab40k runs, which needed 40,000 evals per seed to retain 17
distinct signatures. Doc 18's 2,000 was costed on the mistaken belief that one probe certifies
one spec (§0.4); since one probe harvests every cell it reaches, the same total bill buys
probes an order of magnitude larger. The full sweep at 20k evals costs ~50 minutes against
313 seconds.

**But the scaling does not close the gap.** Extrapolated to 50 variants the corrected sweep
should land near 190 specs over 12–17 topologies — a real improvement on 86 over 6, and still
two orders of magnitude below doc 18 §3.1's O(10⁴). Probe budget was a genuine defect in the
plan and fixing it does not make the task supply adequate. §2.1 stands.

### 2.5 The full sweep at the corrected budget

50 variants × 3 seeds × 20,000 evals = 150 probes, 2,384 s on 20 workers.
Output `runs/rl_tasks/certified_20k.json`.

| | 2,000 evals | 20,000 evals |
|---|---:|---:|
| cells filled | 288 | 322 |
| unique (variant, cell) | 123 | 293 |
| certified (achievable ≥ 0.05) | 99 | 224 |
| **survive `harden`** | **86** | **153** |
| distinct topologies: filled / ≥0.05 / hardened | — / — / 6 | 22 / 16 / **14** |
| variants that certified anything | 24 / 50 | 25 / 50 |
| carried over from the seed | 34 (40%) | **11 (7%)** |
| median achievable | 0.765 | 0.618 |
| specs outside "every seed hit it" | 26 (30%) | **66 (43%)** |
| `prism` specs | 0 | **0** |
| `test_cell` split | 2 specs, 1 topology | **30 specs (20%), 6 topologies** |

**1.8× the specs, 2.3× the topologies, and a task set that is now its own rather than the
seed's** — carry-over falls from 40% to 7%. The median achievable *drops* (0.765 → 0.618),
which is the right direction: the extra specs are harder ones, and the fraction sitting in a
non-trivial difficulty band rises from 30% to 43%. `test_cell` becomes a real interpolation
split instead of a rounding error.

Two things to read carefully.

**Where the attrition actually happens.** The funnel is 322 unique (variant, cell) filled →
224 clear `achievable ≥ 0.05` → 153 survive hardening, and by topology 22 → 16 → **14**. So
the margin floor costs 71 specs but only **two** ray paths (`A2 D1`, `A2 B2 D1`) — those two
are reachable and have no known certifying design clearing 0.5° of bundle margin, so no spec
asks for them. The larger topology drop, 22 → 16, is the near-zero-fitness threshold removing
routes that were touched once and never worked. Both filters are doing what they should; it is
worth keeping them separate, because "hardening deletes a third of the specs" and "hardening
deletes eight ray paths" are very different claims and only the first is true.

**The sweep's best design is 0.912, and it is NOT a program record.** It sits in the
`fov4_w10` variant — ±4° field against a ±10 mm window — which is a materially easier task
than the ±8°/±8 mm at which the 0.886 and 0.878 were measured. Fitness is not comparable
across variants and should never be quoted that way; within the canonical variant the sweep
reproduces the familiar numbers.

**One fix the real task set forced.** `splits` originally held out 20% of *topologies*, which
on this set held out 37% of the specs — the classes run from 1 to 43 members, so drawing three
of fourteen can take the dominant route and a third of the data with it. The cut is still made
at topology granularity (never randomly, for the leakage reason above), but the budget is now
counted in specs: 20% held out over 6 topologies, 123 training specs, stable across seeds.

**Where this leaves task supply**: 153 certified specs over 14 topologies, against doc 18
§3.1's O(10⁴). Probe budget was a real defect in the plan and correcting it moved the number
by 1.8×, not by two orders of magnitude. §2.1 stands, and §5 is the decision it forces.

---

## 3. Milestone gates

| # | Milestone | Gate | Verdict |
|---|---|---|---|
| 1 | Reward service + hardened config + parser | 20-worker pool sustains > 250 designs/s through `reward.py` | **not met as written — 88/s.** The 250 was derived from a benchmark that never traced 70% of its designs (§0.1). Restated against generation cost, a 256-rollout step costs 2.9 s of reward against 30–60 s of generation, which is what the gate was proxying for. |
| 1 | " | hardened config rejects ≥ 3 known-exploit designs from existing archives | **met** — refuses 10 of 95 ab40k elites on geometry alone (8 bulb faces, 2 embedded sources), before tracing. A further 26 of the 85 that trace fall below the crit-margin floor. |
| 2 | Task generator + certification, ~2k specs | ≥ 60% of sampled specs certify | **met on its own terms, and the terms are wrong** — at 20k evals, 224 of 293 filled cells certify (76%), 153 survive hardening (52%). But this measures the harvest, not a sample of the space: 25 of 50 variants and 14 of 79 topologies produced anything. |
| 2 | " | difficulty spread covers the 0.2–0.8 band | **met** — 66 of 153 specs sit outside "every seed hit it" (43%), and median achievable falls to 0.618 as harder specs enter. |
| 2 | " | ~2,000 specs | **not met — 153.** See §2.1 and §2.5. |

Milestone 1 is functionally complete: the service, the hardened config, the parser and the
tests exist and the reward provably agrees with the driver. Milestone 2's machinery is
complete and its *output* is far too thin to train on.

---

## 4. What is not done

- No `prompts.py`, no SFT dataset builder, no `train_grpo.py`. Milestones 3–6 are untouched;
  this is the reward and the task supply only.
- Nothing is committed. The working tree carries `nslens/rl/`, the two tests, the
  `run_all.py` wiring, and the `observation.py` memoisation.
- The reward has never been driven by a model. Every number here comes from archive elites
  and probe output.
- `arvr` is not an RL env. `ArvrProblem` carries its own `evaluate`, so it needs a separate
  path through `reward.evaluate`.
- The knife-edge finding in §0.3 is a measurement, not a tolerance study. It says these
  designs fold on 0.6–2° of margin; it does not say what manufacturing tolerance that implies.

---

## 5. Is there signal to learn from? Two measurements

Doc 18 §4.3 names advantage collapse as the most likely way this arm quietly fails, and sets
the milestone-3 gate at "within-group reward std > 0.05". That is measurable now, without a
model, because CMA-ME populations are GRPO groups in the shape that matters: 10 proposals
drawn around one mean.

### 5.1 Within-group reward variance: 0.10 median, 3% dead groups

31 complete 10-member populations reconstructed from `ab40k_topo/seed0/trace.jsonl` by keying
on `(emitter, parent)` — the emitter mean *is* the population identity — and scored through
`reward.reward_of` against a fixed spec:

| | |
|---|---:|
| within-group reward std, mean / median | 0.094 / **0.103** |
| groups with std < 0.01 (a dead GRPO step) | 1 / 31 = **3%** |
| groups with std < 0.05 (below the milestone gate) | 5 / 31 = 16% |
| stage mix over 310 proposals | 108 scored, 114 off_class, 41 no_light, 47 unbuildable |
| reward percentiles (50 / 75 / 90 / 99) | 0.15 / 0.31 / 0.37 / 0.46 |

The staged ladder is used across its whole range rather than saturating at either end, and the
median group carries twice the variance the gate asks for.

**Read the caveat with the number.** These are populations from a warm-started CMA-ME run, so
they come from a proposer that is already competent — an optimistic proxy for a policy's early
rollouts and a fair one for the post-SFT regime where GRPO actually starts. It says the reward
*shape* is not the thing that will kill the run; it does not say a base model will produce
groups like these.

### 5.2 The topology basin is ~1–2% wide, and the performance basin is half that

Does the conditioning label even depend smoothly on the genotype? 153 certified anchors, each
perturbed 8× at five noise levels, each at its own variant — 6,120 traced designs:

| σ (% of each parameter's range) | same topology | still images | median fitness / achievable |
|---:|---:|---:|---:|
| 0.2% | 98% | 99% | **1.00** |
| 0.5% | 92% | 98% | 0.94 |
| 1.0% | 82% | 97% | 0.75 |
| 2.0% | 64% | 96% | 0.17 |
| 5.0% | 30% | 88% | 0.00 |

Two decay scales, and the gap between them is the finding. **Topology survives to ~1–2% of
range; fitness does not survive past ~0.5%.** Getting the *route* right is roughly twice as
forgiving as getting the *performance* right — at 1% noise the design still folds the same way
82% of the time but has already lost a quarter of its achievable score, and at 2% it keeps the
route two thirds of the time while scoring essentially nothing.

Consequences:

- **The inverse problem is not chaotic.** Each route sits in a basin of finite width, so
  spec → genotype is learnable in principle. This was not obvious: 36 of 40 fold24_clean
  elites change dominant signature across the *field* sweep, which had suggested the map might
  be knife-edge in parameter space too. It is not.
- **The precision bar is ~0.5% of range** to score well: 0.25 mm on `cx`, 0.0025 on `n`,
  0.125° on the tilts. Within the reach of 4-decimal JSON, and demanding for a model that has
  to *choose* those digits.
- **Single-turn (stage A) should be expected to plateau at mediocre fitness even with perfect
  topology control**, because the fitness basin is narrower than any one-shot proposal will
  reliably hit. This is a positive argument for the stage B multi-turn setting that doc 18
  §1 already prefers, and it predicts what gate 5 will measure.
- Note the contrast with §0.3: these designs are *robust in genotype space* and *fragile in
  physical margin*. Those are different fragilities and should not be conflated.

Both measurements are local — around known-good anchors and competent proposers. Neither says
anything about finding a basin from scratch, which is the thing a policy would have to do.

---

## 6. What to do next, in order

1. **Decide whether the task supply can be made adequate at all.** §2.1 is the result that
   should drive the decision, not §0.1. Three levers, in descending confidence:
   *widen the genome* (the third curved face already bought 1.8× distinct ray paths), *widen
   the variant grid* beyond FOV × window (detector distance, aperture, index range), and
   *accept a smaller task set* and train for topology-conditioned generation rather than broad
   coverage. The third is honest and changes the claim.
2. **Build a `prism` seed** (permissive run → filter through the hardened config → warm-start),
   or the transfer split of doc 18 §3.2 does not exist.
3. **Only then** milestone 3 (SFT warm start). Its go/no-go gate — ≥70% of samples parse and
   are geometrically valid — is now measurable: `parse.failure_histogram` plus
   `reward.geometry_verdict`.

The alternative worth weighing before spending on step 3 is the signature-conditioned inverse
model of doc 19 §5.5: supervised on 58,980 scored rows, no RL stack, no advantage collapse,
and it does not need a large certified task set because the catalogue's signatures *are* the
conditioning labels. If task supply cannot be widened, that is the arm the data supports.
