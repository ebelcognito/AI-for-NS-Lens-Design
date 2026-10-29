# AZR-style optical repair: attempt, results, and next direction

**Status: 2026-09-14. Pilot completed; richer interactive Qwen control is proposed, not implemented.**

**Main finding:** We generated physically verified, solvable repair tasks from existing n = 1.50 designs. Training Qwen's limited proposer/solver policies did **not** improve held-out repair success: frozen Qwen, trained Qwen, and a fixed CMA plan all solved **9/12** test tasks. Every model chose the same plan. This tests six-preset selection, not broad optical-engineering ability.

## 1. Why this direction

The intended downstream objective is **compaction along the fixed incoming beam direction**, preserving the optical brief. Body thickness is the extent of the complete curved body projected onto that axis. Transverse width and source/receiver package depth should be recorded separately. Internal ray directions change at folds and should not redefine the thickness axis during optimization.

We lack a high-quality compact-design dataset. The proposed AZR-style approach makes data generation part of learning:

**Discover designs → characterize → generate challenges → attempt edits → verify → train → generate further challenges.**

The simulator can verify improvement without knowing the global optimum. Original good designs can serve as hidden feasibility witnesses when generating degraded tasks. A failed search alone cannot establish that an open compaction target is impossible.

The inspiration is [Absolute Zero Reasoner](https://arxiv.org/abs/2505.03335): proposer/solver learning grounded in executable verification. Our pilot is a **finite-action adaptation**, not a reproduction of AZR's original algorithm or open-ended task generation.

## 2. Brief history of the larger n = 1.50 searches

These CPU portfolio-CMA searches explore three-, four-, and five-corner circular-face bodies, 4/6 mm input beams, multiple receiver layouts, angular half-ranges of 0.5°/2°/5°, and standard/small/thin geometry bounds. The declared grid has 216 settings. Each run uses six workers and up to 4,000 search calls per job; deadline-truncated jobs may use fewer.

| Run | Search allowance | Completed / successful jobs | Search calls | Distinct passing geometries within run |
|---|---:|---:|---:|---:|
| [2026-09-12, n=1.50](../../runs/discovery_ri1.50_30min_20260912_183819/) | 30 min | 51 / 8 | 191,482 | **47**: 11 triangles, 4 quadrilaterals, 32 pentagons |
| [2026-09-14, n=1.50](../../runs/discovery_ri1.50_40min_20260914_143838/) | 40 min | 72 / 16 | 276,302 | **111**: 44 triangles, 45 quadrilaterals, 22 pentagons |
| [Later small-detector/spot run](../../runs/discovery_ri1.50_small_detector_spot_40min_20260914_155038/) | 40 min | 72 / 1 | 276,578 | **2 quadrilaterals** |

The first two runs have 20 and 23 passing designs respectively with recorded worst-field throughput ≥80%; that filter alone is not the stricter pilot admission test. The third run uses throughput penalized by RMS spot size and a detector/beam-ratio constraint, so its `quality` values are not directly comparable. **It supplied no pilot witnesses.**

The main 40-minute run finished in **49.6 minutes including audits and rendering**. All three videos completed: [video 1](../../runs/discovery_ri1.50_40min_20260914_143838/video_1/evolution.mp4), [video 2](../../runs/discovery_ri1.50_40min_20260914_143838/video_2/evolution.mp4), [video 3](../../runs/discovery_ri1.50_40min_20260914_143838/video_3/evolution.mp4). Its earlier reported 72 and 103 designs were live progress counts; **111 is final**. These are routing discoveries, not certified compact imaging designs or globally unique optical topologies.

## 3. What characterization is for

Characterization should support retrieval, curriculum selection, and editing decisions. Numerical facts come from geometry and tracing; Qwen can summarize them and propose testable hypotheses.

| Record | Engineering use |
|---|---|
| Fixed optical brief, index, thickness, width, clearances | Define the task and preserve requirements |
| Routes, TIR counts, per-field throughput and root support | Describe how the design transports light |
| TIR margins, ray footprints, receiver/corner clearance | Identify likely failure points |
| Curvature, surface usage, small-edit response measurements | Identify available editing freedom |
| Qwen diagnosis and proposed edit, followed by measured outcome | Build grounded decision-training examples |

Descriptions alone do not teach successful compaction. The valuable record is **state → diagnosis → action → measured consequence**. Characterizing the whole archive with Qwen has not been performed in this attempt.

## 4. Executed degradation-and-repair pilot

The user proposed defining minimum quality, making existing designs worse, and testing whether an AZR loop could learn repair. We implemented **quality restoration first**, without adding a thickness-reduction requirement.

- **Quality floor:** ≥80% worst-field folded throughput and ≥80% contributing-root support. Material remains n = 1.50; original illumination, receiver, clearance, TIR and energy-accounting requirements remain, including stricter original thresholds.
- **Witness admission:** originals pass 3 fields × 9 rays, 5×25 audit, and 7×51 stress sampling. Stress results agree at 200,000 and 400,000 branch capacity. All checks use the same physics kernel.
- **Degradation:** perturb pose, curvature, or shape at normalized Gaussian scales 0.04 or 0.12. Accept traceable starts with throughput between 20% and 77.5% that fail the quality floor on all three grids. Preserve failed generation attempts too.
- **Corpus:** 10 brief/range families, drawing seven witnesses from the main 40-minute run and three from the earlier 30-minute run. Four train families produce 23 tasks; two validation families produce six; four test families produce twelve. All **41 tasks** are repairable by at least one tested plan.
- **Leakage boundary:** solver input excludes original witness genes, corruption recipe, task/family/split identifiers, and counterfactual outcomes. Related tasks stay in the same split. Shared geometry priors and similar topologies remain possible; ten families are not ten proven independent topologies.

## 5. How much control Qwen actually had

**Initial measurements → Qwen chooses one preset → CMA executes 40 candidates → final audits.**

| Editable parameter group | Available initial CMA sigma |
|---|---|
| All parameters | 0.025 or 0.10 |
| Curvature only | 0.025 or 0.10 |
| Shape and pose only | 0.025 or 0.10 |

That is six actions. CMA chooses the numerical changes to body position/orientation, dimensions/corner positions, and circular-face sag, within codec bounds. It runs five populations of eight. Qwen **already selected the parameter group**, but selected it **once**. It could not inspect intermediate outcomes, change groups, edit an individual surface directly, restart, revert, or allocate more evaluations.

Qwen3-8B used separate rank-8 proposer/solver LoRA adapters on one frozen backbone. The proposer selected one of six **cached** perturbation recipes. The solver selected a CMA plan through a constrained single-letter action. Neither generated unrestricted geometry edits or an explanatory reasoning trajectory.

Two rounds and two sampling seeds compared uniform task selection with a learned proposer. Each arm received 16 solver gradient steps per seed; the learned-proposer arm also received 16 proposer steps. Solver updates used exact finite-action expected-reward gradients over simulator-derived returns, with a KL penalty—not supervised witness prescriptions or AZR's original PPO/TRR++ method. Proposer reward was `4p(1-p)`, where `p` is current solver probability of a verified successful plan: a learnability proxy, not measured learning progress.

All six numerical outcomes per task were computed privately: **246 repair episodes**, reused as reward tables. Generation was part of corpus construction, but new tasks were **not** generated every training round. Learned-proposer overhead was additional; only solver update counts were matched.

## 6. Results and what they mean

All checkpoints were evaluated after training finished; none was selected using test results.

| Method | Test repairs |
|---|---:|
| Fixed CMA plan selected on training tasks | **9/12** |
| Frozen Qwen | **9/12** |
| Uniform-curriculum Qwen, both rounds/seeds | **9/12** |
| Learned-proposer Qwen, both rounds/seeds | **9/12** |
| Best of six plans per task, diagnostic upper bound | **12/12** |

Every model chose **plan A: all parameters, sigma 0.025**, for every test task. Adapter parameters changed, but selected actions did not. Round-two probability-weighted success was 74.8–74.9% for uniform training, 74.2–74.6% for learned-proposer training, and 74.0% frozen. These small probability shifts do not demonstrate better engineering decisions. Validation was 6/6 for every method.

**Established:** verified repair tasks can be generated without an existing refined-design dataset; the numerical action menu contains solutions the chosen policy misses. **Not established:** improved held-out decision-making, a learned-curriculum advantage, compaction ability, or broad optical reasoning. The result does not rule out a richer interactive controller.

Fourteen checks passed, and one full numerical trajectory plus audits replayed exactly. Recorded computation: 8,660 admission rays, 31,823 generation rays, 471,696 counterfactual repair rays, and 1,919 replay rays. The cache is real computation, not free supervision. Only four test families were used; the zero paired difference and degenerate bootstrap interval do not establish population equivalence.

## 7. Proposed next direction: richer interactive control

The user's correction is that the six-preset interface is too limited to assess meaningful engineering control. **Do not describe the next change as merely allowing parameter-group choice: that already existed.** Add repeated decisions and finer control:

| Control | Example |
|---|---|
| Individual parameters/surfaces | Freeze everything except face B's curvature |
| Direct geometry edits | Move a selected corner inward by 0.5 mm |
| Diagnostic probes | Measure both signs of a small curvature change |
| Optimizer settings | Choose bounds and parameter-specific search scales |
| State and budget management | Restore a previous state or allocate another short batch |

“Feedback between short CMA batches” means: Qwen selects an action; CMA evaluates a bounded batch; Qwen receives measured optical/geometry changes and failure diagnostics; Qwen decides the next action. Continuing CMA with the same variables and restarting it with a changed parameter set must have explicit, reproducible semantics.

The next work should follow this order:

1. Build an **inspect → edit/probe/optimize → verify** interface with individual-parameter control and immutable optical requirements.
2. Develop tasks requiring different actions and recovery strategies, using training/fresh development families. Include successful and failed trajectories.
3. Compare fixed numerical control, simple rules, frozen Qwen, uniform-training Qwen, and learned-proposer Qwen at matched total simulator budgets; charge internal CMA calls and diagnostics.
4. Evaluate on newly frozen families. The present test set has now informed interpretation and must not be reused as an untouched final benchmark after further tuning.
5. Extend to thickness reduction once repair decisions improve; generate fresh tasks within the loop and test whether learned task generation improves transfer.

Success means **better verified outcomes on unfamiliar designs at the same budget**, not lower training loss, plausible descriptions, or a successful video. Independent-engine validation, imaging/dispersion, tolerances and 3D performance remain later requirements for broader engineering claims.

## 8. Artifact map

| Artifact | Location |
|---|---|
| Pilot guide, report, logs and adapters | [runs/azr_repair_pilot_20260914_v1](../../runs/azr_repair_pilot_20260914_v1/) |
| Complete results and limitations | [RESULTS.md](../../runs/azr_repair_pilot_20260914_v1/RESULTS.md) |
| Corpus, admission, generator attempts, private reward tables | [corpus/](../../runs/azr_repair_pilot_20260914_v1/corpus/) |
| Generation/repair and Qwen training implementations | [azr_repair_pilot.py](../../nslens/analysis/azr_repair_pilot.py), [train_azr_repair_pilot.py](../../nslens/workbench/train_azr_repair_pilot.py) |
| Reporting and tests | [report_azr_repair_pilot.py](../../nslens/analysis/report_azr_repair_pilot.py), [test_azr_repair_pilot.py](../../tests/test_azr_repair_pilot.py) |

The run link points to `/playpen/hisham26/azr_repair_pilot_20260914` because the workspace filesystem was nearly full. Source snapshots, hashes, environment versions, prompts' input boundaries, and exact evaluation results are retained there. Richer interactive controls are **not yet implemented**.
