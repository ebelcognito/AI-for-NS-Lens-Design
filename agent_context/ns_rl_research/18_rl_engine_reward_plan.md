# 18 — RL with the engine as reward: detailed plan

**Date:** 2026-08-13. Follows [17_related_work_search.md](17_related_work_search.md) §5 (OptiAgent's GRPO
is the paper this arm argues with) and reuses the discovery-control apparatus from
[14](14_discovery_control_experiment_plan.md)/[15](15_discovery_control_results.md)/[16](16_prism_discovery_control_results.md)
as the baseline.

**Goal.** Fine-tune an open-weights model (Qwen) to design non-sequential TIR-folded optical
elements, using the ray-optics engine as a verifiable reward. No human labels.

---

## 0. Measured facts this plan is built on

All measured on this machine on 2026-08-13, not estimated.

### 0.1 The engine is not the bottleneck

`scratchpad/bench_engine.py`, `FoldProblem`, uniform genotypes, 9 fields per design:

| workers | engine calls/s | full 9-field designs/s | scaling |
|---:|---:|---:|---:|
| 1 | 197 | 21.9 | 1.0× |
| 4 | 725 | 80.6 | 3.7× |
| 10 | 1,694 | 188.2 | 8.6× |
| 20 | 2,624 | 291.5 | 13.3× |

Near-linear to 20 workers (20 cores). Note this is *far* above the 13–66 evals/s the archives'
`wall_seconds` imply — those runs were single-process and paying gate overhead.

**Consequence, and it drives every other decision here:** a GRPO step with 256 rollouts costs
**≈0.9 s of reward** against **≈30–60 s of LLM generation** on one RTX 6000. The verifier is
effectively free; the *policy* is the bottleneck. This inverts the usual RL-for-science budget,
where the simulator dominates. Practical implications:

- Do not build a reward surrogate. There is nothing to amortize.
- Spend the free budget on reward *quality*: evaluate every design at all 9 fields, run every
  physical gate, and re-evaluate the winner under perturbation for robustness. All of it is noise
  against generation cost.
- Large GRPO group sizes (G = 16–32) are nearly free on the reward side. Use them — they are the
  main defence against the advantage collapse described in §4.3.

### 0.2 Hardware

- 2 × NVIDIA RTX 6000 Ada, 48 GB each; 20 cores; 125 GB RAM.
- torch 2.11.0+cu130 at `/playpen/hisham26/optifusion_env/bin/python3` (use this one, not
  `deeplens_env`'s 2.5.1 — TRL/vLLM versions we need want the newer stack).
- Fits: Qwen3-8B LoRA training on GPU1 + vLLM generation on GPU0. A 14B is possible with LoRA +
  bf16 but leaves no headroom for large G; start at 7–8B.

### 0.3 Data and task supply on disk

| Artifact | Count |
|---|---|
| trace rows (45 runs) | 539,483 |
| physically scored rows | 58,980 |
| `status: invalid` (geometry rejections) | 480,503 |
| `improved` / `new` (archive-advancing moves) | 2,081 |
| within-parent orderable pairs | 770,757 |
| archives | 145 |
| elite entries | 1,397 |
| **unique (experiment, cell) specs** | **533** |
| elites with fitness > 0.3 / 0.5 / 0.7 | 143 / 54 / 24 |

**The one bad number is 533.** The archive-as-task-generator idea from the strategy discussion does
not survive contact with the data: 533 specs, of which only ~143 are backed by a design good enough
to prove the spec is worth asking for, is not a task distribution — it is a test set. A model
trained on 143 goal specs will memorize them. §3 replaces this with procedural generation, which
the throughput in §0.1 makes affordable.

---

## 1. What the model actually emits

Two stages. Do not start at stage B.

### Stage A — single-turn prescription

```
system: You are an optical designer working in a 2D non-sequential ray tracer. ...
user:   <spec: env, target fold count, max thickness T, min worst-field throughput,
         parameter names + bounds>
model:  <reasoning> ... </reasoning>
        <design>{"cx": 52.1, "thick": 18.4, "fsag": -6.2, ..., "n": 1.72}</design>
```

Reward computed on the `<design>` block. Purpose is not the final capability — it is to get the
reward plumbing, the parser, the gate stack and the curriculum calibrated **in a setting with no
credit-assignment ambiguity**. Every failure at stage A is a reward failure, which is exactly what
you want to debug first.

### Stage B — multi-turn closed-loop design (the real target)

```
turn 1  model proposes a design
        harness traces it, returns an observation block:
          dominant signature   T@A -> TIR@D -> TIR@B -> PR@A -> TIR@T -> T@B
          tir_count 3   spot 38.5   worst-field throughput 0.0068
          per-field [0.104, 0.003, ..., 0.494]
          gate verdicts: sag_ok=True, det_clear=True, clean_pack=False (caustic at leg 2)
turn 2  model revises
        ... up to K turns (K = 6)
reward  best design produced across the episode, minus a small per-turn cost
```

This is the setting that matches ALE's format (agent works on a real machine, artifacts are graded)
and the setting where the 539K-row trace corpus is actually about the same task. It is also where
the interesting research claim lives: does the model learn a *search strategy*, or only a prior?

**Reward on best-of-episode, not last-turn.** Last-turn rewards punish exploration — a model that
tries a bold restructure on turn 5 and lands slightly worse gets penalized for the information
gathering that made turns 1–4 useful. Add a small per-turn penalty (≈0.01) so it does not stall.

---

## 2. Reward design

### 2.1 The exploit list — read this before writing any reward

This repo already documents, in its own source and in memory, the holes that search found in this
exact fitness function. An LLM under GRPO will find them faster than CMA-ME did, because it can
reason *about the scoring rule* rather than only sample against it. Every one of these is a live
reward hack:

| Exploit | Fix | Note |
|---|---|---|
| Body passes **through** the detector plane; `thinness_T` then measured against a plane the glass intersects | `det_clearance` set (≥ 12 mm) | `Problem.det_clearance` defaults to **None = no check**. Measured: 1–6 elites per fold24 run already sit past `x_sensor`. |
| "Faces" that are bulbs (arc > 180°) | `max_sag_ratio = 0.268` | legacy default permits q up to 5.0 |
| Absolute sag box means something different at every thickness — a 6 mm element carries a 15 mm bulb | `relative_sag = True` | |
| Knife-edge TIR: continuation cheats via grazing incidence at a corner | margin floor ≥ 3 mm | this is the `thin_design.py --min-margin 3` finding |
| On-axis excellence, blind off-axis | `field_aggregate = "worst"` | `"mean"` is the legacy default and it hides this completely |
| Beam-quality gates evaluated at field 0 only — 40/40 elites "clean" on axis, 16/40 clean at every field | `gate_all_fields = True` | |
| Caustics passing the width-only packing test (26 of 42 elites in the run that exists to reject caustics) | `convergence_gate = "clean_pack"` | |
| Unfolded design occupying a folded niche via BC clamping | `reject_out_of_bounds = True` | |
| Embedded-source cheat (source geometry inside the glass) | source placement validated per env | taints every TIR@D archive cell in the older runs |
| Etendue growth as a quality proxy | **do not use it** | collimated source ⇒ any focusing reads as 6–10× growth; it rejects the two best designs in fold24_wide. Use `emittance`. |

**Therefore: the RL environment config is NOT the frozen-archive config.** The archive defaults exist
to keep `tests/test_qd_equivalence.py` comparing cell-for-cell against committed results. That is the
right call for reproducing published archives and the wrong call for training. Define
`nslens/rl/env_config.py` with every gate switched **on**, and add a test asserting it diverges from
the frozen defaults deliberately (so nobody "fixes" it back).

### 2.2 Reward shape

A 0/1 "did it meet spec" reward gives GRPO nothing on hard specs — if all G rollouts fail, the group
advantage is identically zero and the step is wasted. Use a staged reward with partial credit, gates
as multiplicative zeros only where the design is *physically meaningless*:

```
r = 0.00                       malformed / unparseable output
  = 0.05                       parses, in-bounds parameter vector
  = 0.15                       geometrically valid element (is_valid + sag + clearance + margin)
  = 0.15 + 0.35 * f_norm       rays reach the detector; f_norm = worst-field focused throughput
                               normalized by the spec's certified achievable score (§3), clipped [0,1]
  + 0.20 * niche_match         descriptor coordinates land in the requested cell
                               (partial credit on |Δcell| — hitting an adjacent fold count is
                                not the same as producing an unfolded slab)
  + 0.15 * gate_bonus          all beam-quality gates pass at all fields
  - penalties                  per-turn cost (stage B), format violations
```

`f_norm` normalized **per spec** by the certified achievable score matters: without it, easy specs
dominate the gradient purely because they score higher in absolute terms, and the model learns to
ignore the hard half of the curriculum.

### 2.3 Anti-hack instrumentation, from day 1

- Log the full `Observation` for the top-5 reward rollouts every N steps and **render them**. Reward
  hacks in this domain are visually obvious and numerically invisible — this is the single highest
  value-per-hour monitoring you can build, and `ns_probe/render_best_designs.py` already does most of it.
- Track the *distribution* of dominant signatures over training. A collapse to one signature is
  either mode collapse or a discovered exploit; either way you want to know the step it happened.
- Hold out a "gate-hardened" eval config strictly stricter than the training one. Score drop from
  train-config to eval-config is the reward-hacking metric.

---

## 3. The task distribution

533 archive specs is not enough (§0.3). Generate specs procedurally and **certify** them.

### 3.1 Generator

1. Sample a spec: env (`FoldProblem` / `PrismFoldProblem` / `ArvrProblem`), target descriptor cell
   (fold count × thinness, or FOV × depth), field set, throughput threshold.
2. **Certify it** with a short CMA-ME probe (2,000 evals) using the existing driver. Record the best
   fitness found → that is the spec's `achievable` score and its difficulty label.
3. Keep specs whose probe reaches a usable score; discard the unsatisfiable ones.

At 291 designs/s a 2,000-eval probe is ~7 s of wall clock, so **one 20-core day certifies on the
order of 10⁴ specs**. That is a real task distribution, and the certification step is precisely the
thing that prevents the failure this program already hit once — the Tier A V1 benchmark spec that was
unsatisfiable, where only 2 of 5 fields could close. Never ship an uncertified spec.

Certification is embarrassingly parallel and reusable: it is also, verbatim, the ALE task-generation
pipeline from the strategy discussion. Build it once, get the benchmark and the RL environment from
the same artifact.

### 3.2 Splits

Split by **cell and by environment**, never randomly. Elites in adjacent cells of the same run are
near-duplicates; a random split leaks. Reserve:

- `test_cell`: held-out descriptor cells within trained environments → interpolation.
- `test_env`: an entire environment never trained on (hold out `PrismFoldProblem` — the
  perpendicular-fold geometry, which doc 16 already shows behaves differently) → transfer.

### 3.3 Curriculum

Maintain a running per-spec pass rate. Sample training specs preferentially from the **0.2–0.8 pass
band**. Specs the model solves every time and specs it never solves both produce zero advantage. This
is not an optimization — it is the difference between the run learning and the run flatlining (§4.3).

---

## 4. Algorithm

### 4.1 GRPO, and why

No value model (the reward is a black-box simulator; a critic would have to learn the physics), group-relative
advantage, well-supported in TRL, and it is the same family OptiAgent used — which makes the comparison
legible to reviewers. G = 16, β (KL) small but nonzero (0.02–0.04); the KL term is the main thing keeping
the model from degenerating into a fixed parameter vector once it finds a locally good one.

### 4.2 Cold start is mandatory

A base Qwen zero-shot on this prompt will emit malformed JSON, out-of-bounds indices, and physically
absurd geometry. Under a staged reward that means nearly every rollout scores 0.00–0.05, groups have
near-zero variance, and GRPO does nothing. Warm-start with a short LoRA SFT pass:

- format + valid-geometry behaviour from the 1,397 elites and a sample of valid trace rows;
- the 2,081 `improved`/`new` moves, rendered as (design + observation → revised design) for stage B format;
- optionally a slice of the 480,503 `invalid` rows as negative examples of buildability.

A few hundred steps. The bar is not quality — it is "produces parseable, in-bounds, geometrically
valid designs most of the time," i.e. reward ≥ 0.15 with meaningful variance. Measure that explicitly
before launching GRPO.

### 4.3 The failure mode to instrument first

**Advantage collapse.** If within-group reward variance goes to ~0 — all rollouts fail, or all succeed,
or the model has collapsed to one output — GRPO's gradient vanishes and the run looks "stable" while
learning nothing. This is the most likely way this project quietly wastes a week.

Log per step: mean reward, **within-group reward std**, fraction of groups with std < 0.01, unique-design
count per group, format-failure rate. Put a hard alert on the third one. The curriculum in §3.3 is the
primary mitigation; raising G is the secondary one and it is cheap here (§0.1).

---

## 5. Infrastructure

```
nslens/rl/
  env_config.py      the hardened gate config (§2.1) + a test asserting it differs from frozen defaults
  reward.py          design dict -> (reward, Observation, gate verdicts, descriptor coords)
  server.py          persistent pool of N Node workers; batch endpoint; genotype->reward cache
  tasks/
    generate.py      procedural spec sampling
    certify.py       2k-eval CMA-ME probe -> achievable score + difficulty
    splits.py        cell/env-held-out splits
  prompts.py         spec -> prompt; Observation -> observation block (stage B)
  parse.py           model output -> design dict; strict, with a logged failure taxonomy
  sft/               warm-start dataset builders from runs/*/trace.jsonl
  train_grpo.py      TRL GRPO; vLLM on GPU0, LoRA training on GPU1
  eval.py            held-out scoring + the baselines in §6
```

Notes that will cost time if missed:

- **The reward server must be a long-lived service, not spawned per step.** `PersistentRayOpticsRunner`
  exists specifically to avoid Node startup cost, and it already auto-restarts on timeout. Spawning a
  pool per GRPO step throws away the entire §0.1 result.
- **Cache genotype → reward** on a hash of parameters rounded to ~6 decimals. Rollouts repeat heavily
  once the policy sharpens; this is free throughput.
- **Timeouts are rewards too.** A design that hangs the tracer must score 0 quickly, not stall the step.
  Cap at 20 s (the existing default) and treat a timeout as an invalid-geometry outcome.
- Determinism: fix the field set and ray counts per spec so a cached reward stays valid.

---

## 6. Evaluation — and the actual research claim

Reward is not evaluation. On the held-out splits report:

1. **Pass@1 / pass@8** against certified spec thresholds, on `test_cell` and `test_env`.
2. **Score under the gate-hardened config** (§2.3) — the reward-hacking check.
3. **Equal-engine-call comparison against CMA-ME.** Give the search the same number of engine calls
   the model consumed and compare best-achieved fitness per spec. This is the honest baseline and this
   program already knows how to run it — it is the discovery-control apparatus from docs 14–16.
4. **Novelty against the training archive.** Does the model produce viable designs in descriptor cells
   the CMA-ME runs never filled? This is the claim worth writing a paper about: *the policy trained on
   a search's byproducts reaches designs the search itself did not.* It is also the honest way to
   position against LDG-EA and OptiAgent — neither has an emergent-topology descriptor to measure
   novelty in.

Expect (3) to be *unfavourable* per-spec at first and to be the wrong frame anyway: CMA-ME re-searches
from scratch for every spec, the model amortizes. Report both the per-spec comparison and the
amortized one (total engine calls including training) and let the reader see the crossover.

---

## 7. Milestones with go/no-go gates

| # | Milestone | Gate to proceed |
|---|---|---|
| 1 | Reward service + hardened config + parser | 20-worker pool sustains > 250 designs/s through `reward.py`; hardened config rejects ≥ 3 known-exploit designs pulled from existing archives |
| 2 | Task generator + certification, ~2k specs | ≥ 60% of sampled specs certify; difficulty spread covers the 0.2–0.8 band |
| 3 | SFT warm start | ≥ 70% of samples parse and are geometrically valid; reward std within a group > 0.05 |
| 4 | Stage A GRPO | held-out pass@8 beats the SFT checkpoint by a clear margin; no signature collapse |
| 5 | Stage B multi-turn | model's turn-2 design beats its turn-1 design more often than chance — i.e. it is *using* the observation |
| 6 | Evaluation + writeup | §6 items 1–4 complete on both held-out splits |

Gate 5 is the real one. If turn-2 does not beat turn-1, the model has learned a prior, not a search
strategy, and the multi-turn framing should be dropped rather than defended.

---

## 8. Risks, honestly

| Risk | Severity | Mitigation |
|---|---|---|
| Advantage collapse (§4.3) | **High** — most likely killer | curriculum band, large G, instrument from step 0 |
| Reward hacking (§2.1) | **High** — the exploit list is empirical, not hypothetical | hardened config, rendered top-5, hardened-eval delta |
| Task supply too thin (§0.3) | Medium — the 533 finding | procedural generation + certification (§3) |
| Model learns a prior, not a strategy | Medium | gate 5; fall back to stage A as the deliverable |
| OptiAgent got there first | Medium | our differentiator is non-sequential TIR topology, not sequential refractive prescriptions — keep the novelty framed as doc 17's conjunction |
| 2D tracer limits the claim | Low but real | state it plainly; the engine is 2D non-sequential and that is the scope |

## 9. What this plan deliberately does not do

- No reward surrogate — §0.1 says there is nothing to amortize.
- No differentiable NS tracer. That is a separate (paused) track; this plan uses the existing
  non-differentiable Node engine, which is exactly why RL rather than gradients.
- No LLM-as-judge anywhere. The whole point is that we own a deterministic verifier.
- No SFT-on-CMA-ME-rationales. CMA-ME's step has no reasoning; back-narrating one with a stronger
  model would be teaching fiction. Let RL find its own reasoning.
