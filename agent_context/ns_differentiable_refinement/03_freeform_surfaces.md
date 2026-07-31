# Freeform surfaces: differentiable refinement + QD search, fold & AR/VR prism

> **What this is.** A self-contained handoff for the freeform-surface extension to the
> differentiable NS twin (`ns_autodiff/`). Built 2026-06-25/26 (Opus 4.8). It adds conic+polynomial
> (and parametric-arc) **freeform faces** to the 2-D non-sequential tracer, then carries them
> through the full pipeline: a validated differentiable intersection primitive → frozen-topology
> **refinement** → topology-**discovering QD search** → a faithful **branching tracer** for the
> AR/VR prism → a **3-edge triangular prism** with fold-count diversity.
>
> Read [00_differentiable_refinement.md](00_differentiable_refinement.md) and the
> [`ns_autodiff/README.md`](../../ns_autodiff/README.md) first — this doc assumes the frozen-topology
> twin and the QD-discovers-topology / refinement-owns-geometry split.

---

## TL;DR — the 5 things not to get wrong

1. **Two tracers, never confused.** SEARCH is topology-**discovering** (no frozen step count, loops
   until each ray exits). REFINE is **frozen-topology** (fixed step count, differentiable). The
   pipeline is **discover (free) → freeze → refine (gradient)**, NEVER freeze-then-search.
2. **In QD search, circular arcs ≥ freeform — for BOTH fold and prism.** Arcs are enough to
   *discover* good designs; freeform's extra DOF just enlarge the space and cost convergence/coverage
   at equal budget (the search objective doesn't reward aspheric). Freeform's payoff is **refinement**
   (fold cell 3,5: **20× tighter spot** on a frozen topology). "Arcs discover, freeform perfects."
3. **Refinement guard = order preservation, NOT robustness.** Per user: don't guard the TIR margin;
   the only thing that must not change is the **interaction order** (the differentiator's frozen
   graph). A TIR margin crossing 0 *is* an order change (TIR→refract). See
   [04_freeform_refine_guard](#) note below.
4. **The prism single-path tracer is WRONG; use the branching one.** A single dominant path is exact
   for the fold (dominant ~90% energy) but the prism's collimation metric is sensitive to Fresnel
   **ghost** energy reaching the eyebox. `prism_branch.py` is the full Fresnel ray-tree + parametric
   aspheric (any arc span) — validated against the JS oracle.
5. **Diversity needs STRUCTURAL descriptors.** FOV×depth (performance axes) fill the archive with one
   shape. Making **fold count** an archive axis forces structurally-different relays into separate
   niches. The 3-edge triangle + fold-count axis gives 0–6+ fold relays.

---

## File map (all under `ns_autodiff/`, isolated — never edits `ray-optics/`, `ns_probe/`, or the
oracle-validated `geometry_torch.py`/`forward.py`)

### Stage 1 — differentiable freeform primitive (the one new piece of math)
| file | role |
|------|------|
| [`geometry_freeform.py`](../../ns_autodiff/geometry_freeform.py) | `ray_freeform_roots`: conic+poly sag in a local tilted frame; intersection = enumerate→nearest-forward (aperture scan, **forward∩aperture slice in h-space** so grazing rays bracket), safeguarded bisection-Newton (detached), then **one live Newton step → exact IFT gradient** (`dt/dp=-F_p/F_t`). Near-tangent guard. |
| [`element_freeform.py`](../../ns_autodiff/element_freeform.py) | `arc_to_freeform`: upgrade a discovered arc → freeform surface initialized to reproduce it exactly (`c=-1/R, k=0, coeffs=0`). |
| [`validate_freeform.py`](../../ns_autodiff/validate_freeform.py) | **Gate.** reduce-to-arc **3e-9** (125/125 rays, both faces); IFT grad vs FD: `c` **1.2e-10**, `a4` **3.5e-7**. |

### Stage 2 — freeform forward trace + frozen-topology refine (fold)
| file | role |
|------|------|
| [`forward_freeform.py`](../../ns_autodiff/forward_freeform.py) | `trace_ff` (frozen-topology, freeform faces A/B + line T/D) + twin-side `face_label`/`dominant_signature_twin` (the oracle can't trace freeform → drift detection lives in the twin). |
| [`validate_freeform_forward.py`](../../ns_autodiff/validate_freeform_forward.py) | **Gate.** `trace_ff(coeffs=0)` == `forward.trace` (~5e-8 mm) == oracle (~6e-7 mm) on cells 3,5 / 4,11 / 0,1 (incl. the meniscus where labels lie); twin signature matches frozen topology. |
| [`refine_freeform.py`](../../ns_autodiff/refine_freeform.py) | Guarded LM on poly coeffs (Tikhonov + throughput + **order-preservation** guard + trust region), twin-native dense honest re-score. Coeffs SCALED `u=a·H_CHAR^pw` for conditioning; report stores PHYSICAL `a`. |

### Stage 3 — topology-discovering QD search (fold)
| file | role |
|------|------|
| [`search_freeform_geom.py`](../../ns_autodiff/search_freeform_geom.py) | shared **numpy** freeform geometry (vectorized over rays), powers per surface. |
| [`search_freeform.py`](../../ns_autodiff/search_freeform.py) | `trace_discover` (no frozen step count) + CMA-ME over `[9 base + 4 aspheric u]`. Same fitness/BC as `ns_probe/run_fold_qd.py` (focused throughput; BC = TIR-count × thinness). `--validate` matches oracle to **3 decimals** (3,5/4,11/0,1). `--lock-aspheric` = arc baseline. |
| [`refine_discovered.py`](../../ns_autodiff/refine_discovered.py) | **The bridge:** freeze a search-discovered design's topology straight from `trace_discover` (`build_frozen_topology`, no oracle) → polish base+aspheric. Closes discover→freeze→refine. |

### Stage 4 — AR/VR prism (4-face): the branching tracer
| file | role |
|------|------|
| [`search_freeform_prism.py`](../../ns_autodiff/search_freeform_prism.py) | **DEPRECATED / single-path** — drops Fresnel branches → wrong collimation on messy designs. Kept as the cautionary baseline; do not run for results. |
| [`prism_branch.py`](../../ns_autodiff/prism_branch.py) | **The faithful tracer.** Full Fresnel ray-tree (`trace_faces`, spawns transmitted+reflected at each refraction), parametric aspheric (radial bump on normalized arc position — **any span**, fixes the >90° leak), `build_arcs_ff` (4-face), `build_triangle` (3-edge), `chief_ray_folds`. Validated: matches oracle **exactly** on clean champions (0.569/0.569), conservative elsewhere; reduces-to-circle **1e-15**. `max_seg=6000` cap stops trapping-cavity tree explosion (returns None→reject). |
| [`search_prism_freeform.py`](../../ns_autodiff/search_prism_freeform.py) | CMA-ME over `[9 base + 8 aspheric]`, branching tracer, BC = FOV×depth. `--lock-aspheric` = arc baseline. |

### Stage 5 — 3-edge triangular prism, fold-count diversity
| file | role |
|------|------|
| [`search_prism_tri.py`](../../ns_autodiff/search_prism_tri.py) | Triangle (8 DOF), circular faces, archive BC = **(fold count, FOV)** — fold count as axis 1 = structural diversity. `--smoke`. |

### Stage 6 — DIFFERENTIABLE PRISM REFINEMENT (closes open step #1; built 2026-06-26, Opus 4.8)
The freeform-aspheric refiner for the 4-face prism — same engine class as the fold's 20× refiner
(analytic IFT gradients + LM + guard family + honest re-score), but for the prism's angular
collimation metric. **Bright/dominant path** (per user: dominant-path tracing is fine for a
converged design; branching was only ever a search-scoring concern). Base geometry + per-source
topology FROZEN; the **8 aspheric coeffs are the grad DOF** (freeform IS the refine win — "arc ≥
freeform" was a *search* statement, never a refinement one).
| file | role |
|------|------|
| [`geometry_prism_ff.py`](../../ns_autodiff/geometry_prism_ff.py) | **The one new piece of math.** Differentiable torch port of `prism_branch.hit_ff`: the **parametric-arc** face (full circle + radial bump `Δ(s)=Σaᵢsⁱ` on normalized arc position `s∈[-1,1]`, valid for the prism's >90° arcs — NOT `geometry_freeform`'s sag(h), which caps at 90°). Same IFT recipe as `geometry_freeform`: detached Newton from the two circle roots → one live Newton step → exact `dt/dp=-G_p/G_t`, `G=(|P-ce|-r)-Δ(s)`. |
| [`validate_prism_ff.py`](../../ns_autodiff/validate_prism_ff.py) | **Gate (PASS).** forward-match `prism_branch.hit_ff` to **~1e-14** (reduce-to-circle AND with real coeffs, all 4 faces); IFT-grad vs FD on a2/a4 **~1e-8**. |
| [`forward_prism_ff.py`](../../ns_autodiff/forward_prism_ff.py) | `forward_arvr`'s structure (per-ray refract/TIR + medium parity, free-flight to pupil, **n_events grouping** for heterogeneous per-source fold counts) with the parametric-arc faces. `freeze_topology` (per-ray bright-path event count), `collimation` (differentiable per-source σ/θ/T, matches `prism_branch.metrics`). Reproduces the branching metric **exactly** on clean champions (cell 9,18: 0.4895/0.2239 to 4dp). |
| [`refine_prism_ff.py`](../../ns_autodiff/refine_prism_ff.py) | Analytic LM on the 8 aspheric coeffs (autograd Jacobian → `(JᵀJ+λI)`). **Guards** = throughput-floor no-erosion + TIR-margin no-erosion + ordering + **per-ray bright (face,kind) signature unchanged** (the order-preservation guard — the only thing doc 03's guard must protect). Every accepted step re-scored through the faithful branching tracer (`prism_branch`). Never-overwrites → `results_refine_prism_ff/`. `--cell i,j` / `--top K --max-folds N`. |

**Refinement results (results_refine_prism_ff/, branching-honest):** single-fold sharpens hard —
cell **9,25 σ 0.718→0.157 (+78%, fit 0.35→0.63)**, 11,17 +54%, 7,21 (σ1.46)→0.72 +50%, 6,18 +37%;
mean **+25.5% σ** over the loose 1-fold batch. **Multi-fold** under strict order-preservation = 0
steps, but the cause is BIMODAL (instrumented): healthy cells (6,12: bright share 0.22) had
fitness-improving steps rejected *only* for order change, while degenerate ones (11,26: bright
eyebox share 0.000, ‖grad‖≈0) are all-ghost / unrefinable. **`--allow-order-change`** (accept
order-changing steps + re-freeze topology) unlocks the healthy ones: **19,15 σ 0.477→0.002**, 18,8
σ 0.729→0.252 (f5→3, +65%), 18,26 4.657→2.252 (+52%), 18,4 +24%, 6,12 +11%. So the order guard, not
physics, blocked the good multi-fold relays. Open: refine *base* geometry (for the zero-dominant-share
cells); triangle (3-face) needs its own `build_faces`.

**Galleries (old dark ray-optics theme, never-overwrite):** `render_prism_refine_compare.py`
(`--min/--max-folds`, `--allow`... via the JSON dir) → `prism_refine_1fold_LATEST.png` /
`results_refine_prism_ff/multifold/prism_refine_multifold_LATEST.png`; `render_fold_refine_gallery.py`
→ `results_search_freeform/fold_refine_gallery_LATEST.png` (6 diverse folds 0–7 TIR, before/after).

### Renderers (never-overwrite: unique PNG + JSON sidecar)
`render_search_gallery.py`, `render_top9_fold.py`, `render_top9_fold_refined.py`,
`render_fold_compare.py` (fold); `render_prism_gallery.py`, `render_prism_compare.py`,
`render_prism_tri_gallery.py` (prism). Results in `ns_autodiff/results_refine_freeform/`,
`results_search_freeform/`, `results_search_prism_freeform/`, `results_search_prism_tri/`.

---

## Headline results

**Fold refinement (frozen topology, freeform > arc):**
- cell **3,5**: champion 2.33 → arc-refine 0.466 → **freeform 0.028 mm (+94% / 20×)**, order preserved.
  Density-stable (0.029 at 401 rays = no ripple).
- cell **4,11**: arc 0.077; freeform **+0%** even with `--allow-order-change` (genuinely no headroom).
- Discover→freeze→refine on search designs: 0-TIR 0.858→**0.001**, 2-TIR 1.70→**0.205**, 3-TIR 0.82→**0.351**.

**Fold search (30k evals, arc ≥ freeform):** freeform 109 niches / arc 118; per-niche freeform tighter
59/109 (median +5.5%, noisy; +20% in 2–3 TIR). Arc covers more (extra DOF cost coverage).

**Prism search (24k evals, branching tracer, arc > freeform):** freeform best 0.489 / 495 niches;
arc best 0.539 / 504. Per-niche tied (freeform higher 51%, median delta 0.000) but **arc wins the
high-fitness sweet spot**. All top relays **single-fold**.

**Triangle + fold-count axis (4k evals, diversity achieved):** 106 cells, fold counts **0–6 (+8,13,15)**.
Best per fold: 0→0.283, **1→0.486 (sweet spot)**, 2→0.114, 3→0.008, … — collimation degrades fast with
fold count (each TIR adds spread). 1 fold is best; multi-fold exist but image poorly.

---

## Gotchas (cost more than gaps)

- **Single-path vs branching:** fine for the fold (dominant path ~90%, exact to oracle), WRONG for the
  prism collimation metric (ghost-sensitive). Always use `prism_branch.trace_faces` for the prism.
- **sag(h) can't represent >90° arcs** (curves back, multivalued). Prism faces hit 112–137°. Use the
  full-circle chord-side `on_arc` (any span ≤180°) for the base + **parametric aspheric on normalized
  arc position** `s∈[-1,1]` (`prism_branch`), NOT the sag-cap (`search_freeform_geom`, fine only <90°).
- **Branching ray-tree explodes** in a trapping cavity (rays refract in/out, bright branch never prunes,
  ~2^depth). `max_seg` cap → reject. The fold never trips this (rays exit fast).
- **Triangle `sag=0` → 3 collinear points → degenerate circumcircle** (inf center). Clamp `|sag|≥0.3`.
- **Render stray-ray view blow-up:** one near-parallel exit ray sends a polyline to y~1e5. Clamp the
  view (`ylim`), don't autoscale.
- **Coeff units:** the fold refine opt vector uses SCALED `u=a·H_CHAR^pw`; the report stores PHYSICAL
  `a`. Don't double-scale when re-tracing (a real bug we hit).
- **Env:** `/home/hisham26/Desktop/simu1/deeplens_env/bin/python` (torch+numpy). Branching prism
  ~150 ms/eval on multi-bounce relays (much slower than the fold's ~14 ms) — budget accordingly.

---

## Entry points (reproduce)

```bash
PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
# primitives / gates
$PY -m ns_autodiff.validate_freeform
$PY -m ns_autodiff.validate_freeform_forward --cells 3,5 4,11 0,1
# fold refine (the 20×)
$PY -m ns_autodiff.refine_freeform --cell 3,5 --powers 2 4
# fold search + arc baseline + galleries/compare
$PY -m ns_autodiff.search_freeform --evals 30000 --seed 0           # (--lock-aspheric = arc)
$PY -m ns_autodiff.render_top9_fold_refined --tir-set 2 3
$PY -m ns_autodiff.render_fold_compare
# prism: validate branching tracer, then search (4-face) + galleries/compare
$PY -m ns_autodiff.prism_branch                                      # validate vs oracle
$PY -m ns_autodiff.search_prism_freeform --evals 24000 --seed 0      # (--lock-aspheric = arc)
$PY -m ns_autodiff.render_prism_gallery --archive .../archive_prism_freeform.json --tag _freeform
# triangle (3-edge) fold-count diversity
$PY -m ns_autodiff.search_prism_tri --evals 4000 --seed 0
$PY -m ns_autodiff.render_prism_tri_gallery
```

---

## Open next steps

1. ~~**Prism refinement** (the missing freeform win)~~ **DONE 2026-06-26** — see Stage 6 above
   (`geometry_prism_ff.py`/`forward_prism_ff.py`/`refine_prism_ff.py`). Single-fold sharpens up to
   +78% σ; multi-fold has no dominant-path headroom. Remaining: refine **base** geometry (not just
   the 8 aspheric) to give multi-fold relays something to descend on.
2. **Better multi-fold prism niches:** the triangle 4k run is coarse; longer run / refine the 2–3 fold
   relays. (1-fold is the quality sweet spot; multi-fold collimate poorly — physical.)
3. **Spline surface** via the callable sag in `ray_freeform_roots` (interface already accepts one).
4. **Differentiable prism refine** through `prism_branch` (currently numpy/non-grad) — port to torch for
   analytic collimation gradients if refinement needs it at scale.
