# MASTER CONTEXT — QD Discovery of Non-Sequential Optical Designs

> **What this is.** A single self-contained handoff of the whole `ns_probe` effort: the goal, the engine, the
> method, every experiment run (with status + numbers), the main findings, where all the code/data live, the
> honest limitations, and the open directions + paper framing. Written 2026-06-12 (Opus 4.8). Detailed
> per-topic docs are `01`–`11` in this folder; this is the index + synthesis.
>
> **How to use this (two intended consumers):**
> - **Build-on-top agent:** read §3 (engine), §4 (method), §7 (artifacts map), §9 (reproduce), §8 (open
>   directions). The code is in `ns_probe/`; the engine mods are in `ray-optics/`.
> - **Ideation/discussion agent:** read §1–2 (thesis), §5 (findings), §6 (the 2×2 trade), §10 (paper framing +
>   open questions). The whole point is *quality-diversity discovery of folded light-path topologies*.

---

## 1. One-paragraph summary

We brought **quality-diversity (QD) illumination** to **non-sequential (NS) optical design**. We instrumented a
2D NS ray tracer to expose **typed per-ray interaction sequences** (refract / TIR / partial-reflect / …) and an
**exact energy ledger**, then used **CMA-ME** (MAP-Elites + CMA-ES emitters, *gradient-free*) to fill an
**archive indexed by the ray-surface interaction topology** (e.g. number of TIR bounces) with the best-imaging
design in each cell. The headline result: **efficient, focusing total-internal-reflection (TIR) folds emerge
from an imaging-plus-throughput objective that never mentions folding**, organized into a diverse archive — and
this requires **symmetry-breaking (prismatic tilt)**, which an ablation shows is causal. No differentiable
tracer and no neural net were needed for discovery.

## 2. Thesis / research question

Classical lens design is **sequential** (fixed surface order, transmission only) and optimizes a *human-chosen*
topology to a local optimum — the topology is an *input*. We ask: **can we discover novel non-sequential
light-path topologies** (folds, TIR, multi-bounce) by **illuminating** the design space, with the *interaction
sequence itself* as the diversity axis, and have useful topologies **emerge** rather than be prescribed? This
generalizes a prior result (TIR emerged unprompted in an RL refocusing task) to multi-surface NS systems, at
bundle scale, with honest energy accounting.

## 3. The engine (what exists + what we modified)

**Base:** the vendored open-source **ray-optics** JS simulator (`ray-optics/`) — a 2D NS geometric tracer
(nearest-surface intersection, refraction, partial reflection, TIR), driven from Python over stdin/stdout JSON
via `PersistentRayOpticsRunner` (`optical_rl/utils/ray_optics_bridge_persistent.py`). Not differentiable.
(Full map: `agent_context/ray_optics_engine.md`.)

**Our modifications** (all flag-gated behind `recordInteractions`; **default path byte-identical**; **22
physics checks** by Fable — TIR loss-free, Fresnel R exact, ledger closure, flag-off identity):
1. **Typed interaction events** — every surface hit is labelled `refract | tir | partial_reflect | mirror |
   sensor | absorbed`, with per-event energy-in/out and a `mech: fresnel|forced` tag. (`BaseGlass.js`,
   `Simulator.js`.)
2. **Exact energy ledger** — `emitted = delivered + escaped + absorbed + truncated + unfinished`, closes to 0.
3. **Per-ray interaction-sequence export** — `result.raySequences` (vertices + types + energies).
4. **Forced reflectivity DOF** — `reflectivity` (uniform per element) and `reflectivityFront`/`reflectivityBack`
   + `elemCx` (per-face, Mangin-style), overriding Fresnel. *(Later understood as a lossy "coating" crutch —
   see §5.)*
5. Files: `ray-optics/src/core/sceneObjs/BaseGlass.js`, `ray-optics/src/core/Simulator.js`,
   `ray-optics/integrations/simulation_core.js`.

**Reflection physics (the decision at each surface):** if incidence > critical angle → **TIR (100%,
loss-free, both polarizations)**; else **refract + split** energy `R` reflected / `1−R` transmitted, where `R`
is forced reflectivity (if set), else physical Fresnel, else 0. **Polarization is intensity-only** (`b_s`,
`b_p` scalars — no phase/Jones; QWP/HWP are *not* representable → see §8 Tier-B). No bulk absorption.

> ⚠️ **Build/sync gotcha:** editing `ray-optics/src/core/*` requires `cd ray-optics && npm run build-node`,
> which writes `dist-node/rayOptics.js`; the runner loads `integrations/rayOptics.js`, a committed copy that
> **nothing auto-syncs** — you must `cp dist-node/rayOptics.js integrations/rayOptics.js` after a rebuild.
> (`simulation_core.js` is *not* bundled — edits take effect immediately.)

## 4. The method (CMA-ME, gradient-free)

- **Genotype:** continuous design parameters (e.g. the 9-DOF tilt element: `ray_y, ang, cx, thick, fsag, bsag,
  tilt_a, tilt_b, n`). The optimizer searches these; reflectivity (when used) is also a DOF.
- **Fitness (quality):** **focused throughput** = fraction of input energy (from the ledger) delivered into a
  tight window around the energy centroid. Rewards efficient *and* concentrated delivery; **never names
  folding/TIR/thinness**.
- **Descriptors / behaviour axes (DESCRIPTIVE, measured from the typed sequence):** (BC1) **# of TIR bounces on
  the dominant-energy path** = fold topology; (BC2) **thinness** (device depth). These define the archive grid.
- **Solver:** CMA-ME (`ns_probe/cma_me.py`) — a pool of CMA-ES "improvement emitters." Each samples from a
  Gaussian `N(mean, σ²C)`, evaluates, inserts into the grid archive, then **ranks by archive improvement**
  (*new cell* > *beat incumbent* > *nothing*) and adapts mean/covariance; **restarts from a random elite** when
  a batch adds nothing. → topology **discontinuities become archive cell-walls, not optimizer traps**.
- **Why no differentiable tracer:** discovery/verification only needs *evaluations*, not gradients. A
  differentiable NS tracer would only help *local refinement* (a later, optional step).

## 5. Experiment ledger (chronological; ✅ success / ⚠️ informative-negative / 🔧 infra)

| # | experiment (dir) | setup | run | result | status |
|---|---|---|---|---|---|
| 1 | **Phase-0 doublet** (`results/`) | 2 on-axis elements, **forced reflectivity (coating)**, BC=(thinness, reversals) | 18k / 157s, 212 cells | folds = **lossy back-reflections** (best ~spot 4.4 @ **9%** flux); transmission wins **0/23** thinness niches; best transmission spot **0.078** | ⚠️ negative (reproduced, `results_seed1`) |
| 2 | **Phase-0 single** (`results_single/`) | 1 element, per-face reflectivity, on-axis | 15k / 86s, 188 cells | folds reach thin (T=12–17) but spot **~6** @ ~10% flux; transmission still dominates | ⚠️ negative |
| 3 | **Reflectivity/TIR analysis** (doc 06) | — | — | **the pivot:** the high-energy fold is **TIR** (loss-free, no coating, no polarization); needs **prismatic tilt**; the forced-reflectivity DOF was a lossy crutch | 💡 insight |
| 4 | **Typed-interaction engine + `fold_demo`** (Fable, doc 07) | typed events + ledger + sequence-as-input | — | verified a **single tilted element delivers 86.8%** via `T@A→TIR@B→TIR@A→T@B` | 🔧 + ✅ |
| 5 | **Fold-QD** (`results_fold_qd/`) | tilt element, **uncoated** (Fresnel+TIR), fitness=focused throughput | 14k / 105s, 109 cells (76% cov) | **diverse efficient focusing TIR folds:** 2-TIR **96% / spot 3.7**, **4-TIR 91% / spot 0.66**; 81 folded cells; *edge-TIR light-guiding discovered* | ✅ **headline** |
| 6 | **Orderly-QD** (`results_orderly_qd/`) | coherent fitness (dominant-signature focus) | 14k / 114s, 105 cells | archive of *coherent* (uniform-bundle) folds | ✅ |
| 7 | **Tilt ablation** (`results_fold_qd_notilt/`) | same as #5 but **tilt locked to 0** (symmetric) | 14k / 109s, 99 cells | folds form & stay efficient but **can't focus**: best fold spot **5.13** (vs 0.08), **0** efficient+sharp folds | ✅ (tilt is **causal**) |
| 8 | **Multi-field QD** (`results_multifield_qd/`) | fitness = mean focused-throughput over **±3°** fields | 12k / 247s, 106 cells | **8 field-robust folds** (~80% throughput across ±3°, central spot **~1.5**) — folds aren't single-field flukes | ✅ |
| 9 | **Physics validation** (`analyze_physics.py`) | re-trace every fold, check TIR incidence vs critical angle | 143 folds, 697 TIR events | **100% of TIR events past critical angle**, median margin **17.8°**; efficiency falls with fold depth | ✅ (designs respect physics) |
| 10 | **Symmetric catadioptric (longer)** (`results_single_long/`) | 1 element, per-face reflectivity, **on-axis**, 26k evals | 26k / 153s, 254 cells | **CORRECTION:** sharp symmetric folds DO exist with more search — **spot 0.44 @ 31% flux** (Mangin: front AR=0, back mirror=1.0). Symmetric folds can be sharp **but lossy (~30%)** | ✅ (overturned the "symmetric can't focus" cap) |

**Galleries (rendered PNGs + per-design geometry sidecars):** `results_gallery_focus_all/` (41, best-focusing),
`results_gallery_focus2/` (18, high-eff sharp), `results_gallery_focus/` (25), `results_gallery/` (19,
orderly+search), `results_gallery_symmetric/` (10), `results_gallery_symmetric_sharp/` (5, Mangin folds).

## 6. Main findings

1. **QD-over-interaction-topology works and is the right framing.** Defining diversity over the *ray-surface
   interaction sequence* lets CMA-ME illuminate a diverse archive of NS designs by topology — something a
   single-objective optimizer cannot do (it would converge to one transmission lens and hide the fold family).
2. **Efficient TIR folds emerge** from a fold-blind objective: 2–4 TIR bounces deliver **91–96%** of the input,
   focused to spot **0.66–3.7**. The optimizer even found *edge-TIR light-guiding* we never specified.
3. **The 2×2 trade (the cleanest single result):** you can pick any **two** of {symmetric, folded, sharp+efficient}.

   | | folded (non-sequential) | transmission (no fold) |
   |---|---|---|
   | **cylindrically symmetric (on-axis)** | sharp **but lossy** (Mangin, spot 0.44 @ ~30%) *or* efficient but unsharp | **sharp** (spot ~0.1) |
   | **tilted (asymmetric)** | **sharp AND efficient** (spot 0.08 @ 91%, loss-free TIR) | sharp |

   The deep reason: on-axis, the dominant bundle hits curved surfaces near-normal, so a *loss-free* fold (TIR)
   needs **broken symmetry (tilt)**; symmetric folds must use **coated partial mirrors** (lossy) to fold.
4. **Physics is respected, not just searched:** 100% of TIR events sit past the critical angle (median margin
   17.8°); the tilt ablation shows symmetry-breaking is *causal*; folds hold up across ±3° fields.
5. **No deep learning / no differentiable tracer needed** for discovery — the gradient-free QD signal suffices.
   Polarization (Jones/QWP) is *not* in the engine and is *not* needed for the TIR route (Tier-B, for the
   lossy pancake family only).

## 7. Code & data artifacts map (`ns_probe/`)

**Core libs:** `cma_me.py` (CMA-ES + GridArchive + CMA_ME loop), `sequence_tools.py` (typed-signature parsing,
dominance report, energy-ledger audit), `probe.py` (doublet), `probe_single.py` (single, per-face refl),
`fold_demo.py` (9-DOF tilt element; sequence-as-input demonstrator/search).
**Runners:** `run.py` (#1), `run_single.py` (#2, #10), `run_fold_qd.py` (#5, #7 via `--lock-tilt`),
`run_orderly_qd.py` (#6), `run_multifield_qd.py` (#8).
**Analysis/render:** `analyze_fold_qd.py`, `analyze_physics.py` (#9), `analyze.py`, `render_designs.py`,
`fold_gallery.py`, `tir_demo.py` (legacy single-ray demo).
**Tests:** `test_typed_interactions.py` (Fable's 22 physics checks; needs `pyyaml` to import in a fresh shell).
**Engine mods:** `ray-optics/src/core/sceneObjs/BaseGlass.js`, `ray-optics/src/core/Simulator.js`,
`ray-optics/integrations/simulation_core.js`.
**Detailed docs (this folder):** `01`(thesis/prior-art), `02`(state), `03`(design decisions),
`05`(phase-0 findings), `06`(reflectivity/TIR), `07`(typed-interactions engine + the Fable prompt),
`08`(fold-QD result), `09`(polarization/pancake), `10`(presentation outline), `11`(ablation/multifield/physics).

## 8. Honest limitations
- **2D meridional**, mostly **single element**, **on-axis or small field (±3°)**. Not a validated 3D lens.
- **"Focused throughput" is an imaging proxy**, not full MTF; "thinness" = device depth, not FoV-normalized.
- **No tolerance/manufacturability robustness** yet (the axis an optical engineer cares about most — see §10).
- The optimizer **does not "understand" optics** — it's gradient-free search over a verifiable simulator; the
  key physics insight (tilt→TIR) was human analysis, *confirmed/mapped* by the search.
- **Scale is proof-of-concept** (hundreds of designs per archive).

## 9. Reproduce (key commands)
```bash
# (after editing ray-optics/src/core/*: cd ray-optics && npm run build-node && cp dist-node/rayOptics.js integrations/rayOptics.js)
python3 ns_probe/fold_demo.py                                   # verify the single-element TIR fold (86.8%)
python3 ns_probe/run_fold_qd.py --evals 14000 --seed 0          # the headline fold-QD (~2 min)
python3 ns_probe/analyze_fold_qd.py                             # archive heatmap + champion renders
python3 ns_probe/run_fold_qd.py --lock-tilt --out ns_probe/results_fold_qd_notilt   # the tilt ablation
python3 ns_probe/run_multifield_qd.py                           # field-robustness
python3 ns_probe/analyze_physics.py                             # critical-angle validation figure
python3 ns_probe/fold_gallery.py --focus --n 40 --from-archive ns_probe/results_fold_qd/archive.json,ns_probe/results_orderly_qd/archive.json --outdir ns_probe/results_gallery_focus_all
```

## 10. Open directions + paper framing (for the ideation agent)

**Proposed first-paper framing (the considered direction):** *the process + the artifact* — **a diverse,
tolerance-robust, physics-verified dataset of non-sequential designs, organized by ray-path topology, generated
by quality-diversity search.** It stands on its own without a learned model (physics is in a verifiable tracer;
every design is energy-audited). Define terms precisely: **diverse = topology coverage** (the interaction-
sequence axis); **robust = physics-validated + field-tested + tolerance-stable**. Context: a Google optical
engineer (Ozan) and the advisor (Praneeth) reacted positively; the open questions they raised were "what's the
first-paper story" and "does it even need AI" — answered above (it needs *quality-diversity*, not deep learning,
and that's a feature). Novelty daylight: prior lens datasets (LensGAN, Côté) are corpora of *human-designed,
sequential refractive* lenses; ours is **non-sequential, machine-discovered, diversity-structured by topology,
physics-verified**.

**Ranked next experiments / extensions:**
1. **Tolerance/perturbation robustness** — add a robustness descriptor *or* objective (perturb surfaces/positions,
   measure performance spread). This is the axis Ozan cares about most, is industrially valuable, novel, and a
   natural QD fit. **Highest-value next step.**
2. **Scale the dataset** — more surfaces, multi-element NS, eventually 3D; more topologies; designs across specs.
3. **Generative / surrogate model trained on the archive** (Path A in doc 01) — amortized inverse design; this
   is where heavier "AI" earns its place (and a clean second paper).
4. **Stronger imaging** — real multi-field MTF, larger FoV/eyebox.
5. **Polarization (Jones upgrade)** — the QWP + reflective-polarizer **pancake** family (doc 09); Tier-B,
   substantial engine work, only for the *lossy* fold family. *Not* needed for the TIR route.
6. **Differentiable NS tracer** (DeepLens-NS) — only when *local refinement* of folded candidates becomes the
   bottleneck; gate on a cheap black-box refinement experiment first.

**Open questions worth ideating:** Does a tolerance-robustness axis change *which* topologies win? Is there a
clean scaling law (efficiency / aberration vs fold depth, beyond doc 11's hint)? Do discovered tilt angles
cluster at a physically-meaningful relation to the critical angle? Can the archive be distilled into stated
design rules (turning "it searches" into "it found laws")? What is the right *unit of novelty* to claim — the
method, the dataset, or specific discovered topologies?
