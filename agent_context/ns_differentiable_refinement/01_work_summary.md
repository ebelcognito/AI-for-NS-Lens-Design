# Refinement work — session summary (what was built, run, and found)

> **What this is.** A summary of the differentiable-refinement work done in the 2026-06-19
> session (Opus 4.8). Companion to `00_differentiable_refinement.md` (the design discussion,
> with the user's exact questions answered). This file is the **"what we actually did"** record.
>
> **TL;DR.** We built the cheap **gate experiment** that Master-Context §10.6 said to run before
> investing in a differentiable NS tracer: finite-difference Jacobian + Levenberg–Marquardt local
> refinement, run *through the existing JS engine* (no differentiable tracer, no autodiff, no GPU).
> It works: **shallow folds (2–4 TIR) refine 50–99 % tighter** with throughput + topology
> preserved. But the archive-wide sweep also found the **honest limits**: deep folds (5+ TIR) and
> low-dominant-share designs are *not* reliably refinable — 53/109 cells got worse on the honest
> metric and 27/109 broke topology. **Conclusion: build the refiner (with a missing honest-metric
> guard, below); do NOT build a differentiable tracer yet.**

---

## 1. The question this answered
Earlier work (QD over non-sequential fold topologies, `ns_probe/`) produces a diverse archive of
designs but each is only coarsely optimized (21-ray throughput proxy). The user asked whether a
**differentiable framework** could refine these into rigorous, sharply-focused designs, and raised
specific worries: (a) DeepLens/dO is sequential + cylindrically symmetric (our designs are neither),
(b) in NS systems a surface is revisited, so "gradients from the second interaction fight the
first," (c) is it better to discover in the JS env and migrate to refine? See
`00_differentiable_refinement.md` for the full point-by-point answers. This file records the
experiment that turned those answers from argument into measurement.

## 2. What was built (code)
| File | Purpose |
|---|---|
| `ns_probe/refine_lm.py` | The gate. FD-Jacobian + Levenberg–Marquardt (damped least squares) refinement of one champion. Freezes topology, densifies the bundle, guards throughput + TIR margin. Emits a JSON report (both geometries + full history + diagnostics) and a 6-panel diagnostics figure. |
| `ns_probe/refine_sweep.py` | Runs the gate over **every** archive cell (one persistent Node process), emits `refinability_map.json` + a 2-panel refinability map figure. |
| `ns_probe/render_refine_comparison.py` | Renders a champion **before vs after** as a side-by-side ray diagram (reads a refine report, traces both geometries with crop-render on, stitches + labels). |

**All never-overwrite** (unique filenames + JSON sidecars), per the standing render rule.

## 3. The method (refine_lm.py)
- **Forward oracle:** the JS `ray-optics` engine (CPU, single Node process over stdin/stdout). No
  differentiable tracer; the Jacobian is **central finite differences** in bound-normalized params.
- **Topology freeze:** track only the input rays reaching the sensor via the champion's *dominant*
  typed interaction signature, keyed by beam offset `dy` (invariant to the lens DOF). Within a fixed
  topology the objective is smooth → ordinary nonlinear least squares.
- **Densified bundle:** 121 rays (QD used 21) — refinement traces few designs, so it can afford a
  sharper objective. Objective = energy-weighted RMS spot of the tracked rays (frozen per-ray
  weights, floating best-focus centroid).
- **Optimizer:** damped least squares — `(JᵀJ + λI)δ = −Jᵀr`, standard λ up/down acceptance, a
  trust-region cap on the step, and bound projection. (This is the Code V/Zemax method.)
- **Two guards (both essential):** a **throughput-floor penalty** (can't sharpen by dumping energy)
  and a **no-erosion TIR critical-angle margin floor** (can't sharpen by driving folds to critical /
  out of the archive cell). Default margin floor = the champion's own start margin.
- **Diagnostics:** spot/throughput/margin per iter; κ(J)=σmax/σmin; per-DOF gradient
  sign-cancellation `1−|ΣJr|/Σ|Jr|`; topology-event counts; and an **independent re-score** at QD's
  21-ray density (the honest, apples-to-apples spot used for all headline numbers).

## 4. Results — the 4-champion gate
| champion | dom-share | 21-ray spot start → refined | throughput | TIR margin | topology |
|---|---|---|---|---|---|
| **3-TIR** (cell 3,5) | 0.91 | **2.42 → 0.67 (−73 %)** | 0.907 → 0.906 | 2.51 → 2.26° | preserved |
| **4-TIR** (cell 4,11) | 0.91 | **0.66 → 0.13 (−80 %)** | 0.914 → 0.916 | 5.03 → 4.78° | preserved |
| 3-TIR *unguarded* | 0.91 | **2.42 → 2.42 (no real gain)** | 0.907 → 0.884 | 2.51 → **0.04°** | churns |
| 2-TIR messy (cell 2,8) | 0.46 | 3.73 → 3.71 (−3 %) | preserved | preserved | preserved |

Five findings:
1. **Headroom is real and large** — clean champions improve 73–80 %. QD finds the basin; geometry is
   left coarse.
2. **The QD↔refine split is empirically principled** — every guarded refinement stayed in its
   archive cell (dominant signature unchanged).
3. **"Gradients fight" = ill-conditioning, not instability** — κ(J) ran **1e8–1e11**, per-DOF
   cancellation hit **~1.0 for `cx`/`bsag`/`thick`** (rays pull oppositely — the literal fighting),
   yet LM/damped least squares descended cleanly every time. The worry names a real phenomenon but
   the wrong consequence.
4. **Regularization is mandatory** — unguarded LM *cheats*: drives TIR to the critical angle
   (margin → 0.04°) to sharpen the tracked sub-bundle while the honest 21-ray spot doesn't improve.
   The no-erosion guard + independent metric catch it.
5. **Refinement is not universally good** — the messy (low-share) champion barely moved.

## 5. Results — tightness ↔ robustness Pareto (cell 3,5)
The 0.67 was the *conservative* (no-erosion) number; a finer FD step + more iters changed nothing
(optimizer is not the limiter — it's **physics**: focusing a fold tighter needs rays closer to
critical). Relaxing the margin floor:

| margin floor | 21-ray spot | TIR margin | throughput |
|---|---|---|---|
| no-erosion | 0.67 | 2.26° | 0.906 |
| **1.0° (knee)** | **0.51** | 1.00° | **0.912** ↑ |
| 0.5° | 0.50 | 0.57° | 0.891 |
| 0° (unguarded) | 2.42 ✗ cheat | 0.04° | 0.884 |

**Knee ≈ 1.0°:** spot 0.51, throughput *up*, topology intact, 1° from critical. Past ~0.5° you pay
throughput for nothing.

## 6. Results — archive-wide sweep (the honest, nuanced picture)
`refine_sweep.py` over all **109 cells**, no-erosion guard, ~23 min on CPU:

- **Median headroom 0 %**, **28/109 improved >50 %**, **82/109 topology preserved**, **53/109 got
  *worse*** on the honest metric, **27/109 broke topology**. Refinability is **bimodal**, not uniform.
- **By fold depth — the key structure:**
  - tir=0 (transmission): median +0 % (most already sharp; some go to a near-perfect point, e.g.
    3.82 → 0.03).
  - **tir=2,3,4 (shallow folds): +53 %, +31 %, +54 %** — the sweet spot; these refine well.
  - **tir=5+ (deep folds): mostly negative** (−5 %, −23 %, −6 %, −14 %, −65 %) and **frequently break
    topology** (tir=6: 4/12 preserved; tir=10–11: 0 preserved). Deep folds stack many near-critical
    TIR events with low dominant-share → fragile, chaotic, unrefinable as-is.
- **By dominant share:** [0.5,0.8): **+49 %**; [0.0,0.5): −4 %; [0.8,1.0): +0 % (dominated by
  already-sharp transmission). So **dominant-share alone is *not* the predictor** (an earlier
  oversimplification): the real predictor is **high starting spot AND mid-to-high share AND shallow
  fold depth**.

Artifacts: `ns_probe/results_refine_lm/refinability_map.json`, `refinability_map.png`.

## 7. The dramatic before/after renders
Four high-quality focus comparisons (`ns_probe/results_refine_lm/compare_*.png`):
| image | design | spot before → after |
|---|---|---|
| `compare_cell3_10.png` | 3-TIR fold ⭐ | **9.31 → 0.06** |
| `compare_cell3_7_2.png` | 3-TIR (4-bounce) fold | 7.57 → 1.87 |
| `compare_cell3_5_1.png` | 3-TIR fold | 2.42 → 0.51 |
| `compare_cell0_1.png` | transmission lens (contrast) | 3.82 → 0.03 |
Folds render best (rays converge *at* the sensor line in-frame); pure-transmission focuses land at a
steep angle that clips the crop edge.

## 8. Open issue discovered → the next code change
The sweep exposed a real flaw: LM minimizes the *dense dominant-topology* spot, but for low-share /
deep-fold designs that **diverges from the honest 21-ray spot**, so "improving" can make the real
image worse (53 cells) or leave the cell (27 cells). **Fix to add to `refine_lm.py`:** an
**honest-metric acceptance guard** — re-score the 21-ray all-forward spot at each accepted step (or
as a final gate) and reject refinements that worsen it; and/or **gate refinement to shallow folds
with adequate dominant-share**. This converts the gate from "trusts a proxy" to "optimizes the thing
we actually care about."

## 9. The decision this informs (differentiable tracer / DeepLens)
**No differentiable NS tracer needed yet, and not DeepLens/dO ever for this.** FD+LM on the existing
JS engine extracts everything above at 7 DOF in ~10 s/champion. A differentiable JAX/PyTorch **2-D NS
twin** (not dO — dO can't represent NS+tilted designs) pays off only when DOF count explodes
(multi-element / freeform / many fields), where the FD Jacobian's 2·n_opt serial traces become the
bottleneck. Build it then, validated forward-pass against the JS oracle. (This is also where a GPU
would matter — batching all rays + all FD perturbations — see the GPU note: nothing here uses a GPU;
it's CPU + a single-threaded Node tracer, latency-bound on round-trips.)

## 10. How to reproduce / environment
**Python:** `/home/hisham26/Desktop/simu1/deeplens_env/bin/python` (numpy 2.1.2, matplotlib 3.10.6).
⚠️ The repo's bare `python3` is the **bluedot venv** and lacks numpy — use the deeplens_env path.
```bash
PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
$PY ns_probe/refine_lm.py                         # default 3-TIR champion, no-erosion guard
$PY ns_probe/refine_lm.py --cell 3,10 --iters 80  # the razor-sharp showcase (9.31 -> 0.06)
$PY ns_probe/refine_lm.py --cell 3,5 --margin-floor 1.0   # tightness knee (0.51)
$PY ns_probe/refine_lm.py --margin-floor 0         # unguarded (demonstrates the cheat)
$PY ns_probe/refine_sweep.py --rays 81 --iters 40  # the whole-archive refinability map (~23 min)
$PY ns_probe/render_refine_comparison.py --report ns_probe/results_refine_lm/refine_cell3_10.json
```

## 11. Next steps (ranked)
1. **Add the honest-metric acceptance guard** to `refine_lm.py` (§8) — highest value, small change.
2. **Gate / classify refinability** — only auto-refine shallow folds (tir ≤ 4) with share ≥ ~0.5;
   flag deep folds as "needs joint multi-topology refinement," not single-topology LM.
3. **Promote the refined champions** back into the archive (or a parallel "refined archive").
4. **Differentiable JAX 2-D NS twin** — only when scaling to multi-element/freeform makes FD the
   bottleneck (see `00_differentiable_refinement.md` §migration). Multi-element extension was scoped
   in this session but deliberately deferred to focus on refinement.

## 12. Artifact map
- Code: `ns_probe/refine_lm.py`, `ns_probe/refine_sweep.py`, `ns_probe/render_refine_comparison.py`
- Results dir: `ns_probe/results_refine_lm/`
  - `FINDINGS.md` — the 4-champion table + 5 findings + tightness Pareto
  - `refine_cell*.json` / `.png` — per-run reports + 6-panel diagnostics figures
  - `refinability_map.json` / `.png` — the archive-wide sweep
  - `compare_cell*.png` — before/after ray renders
- Discussion: `agent_context/ns_differentiable_refinement/00_differentiable_refinement.md`
- Upstream: `agent_context/ns_rl_research/00_MASTER_CONTEXT.md` (§10.6 = "gate this first")
