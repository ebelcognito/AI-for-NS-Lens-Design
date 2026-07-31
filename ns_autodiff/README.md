# ns_autodiff — a differentiable PyTorch twin of the non-sequential tracer

**What this is.** The "full scoped autodiff" stage that the design discussion in
[`agent_context/ns_differentiable_refinement/00_differentiable_refinement.md`](../agent_context/ns_differentiable_refinement/00_differentiable_refinement.md)
scoped but deferred: a **differentiable** non-sequential 2-D ray tracer (PyTorch) that
refines a QD-discovered fold using the **exact reverse-mode Jacobian** instead of the
finite-difference Jacobian used by [`ns_probe/refine_lm.py`](../ns_probe/refine_lm.py).

It is the analytic-gradient counterpart to `refine_lm.py`. Same problem, same objective,
same guards, same outputs — but `∂(spot)/∂(params)` comes from autograd, not from
`2·n_opt` central-difference traces through the JS engine.

> **Isolation rule.** This package never edits anything under `ray-optics/` or `ns_probe/`.
> It imports the JS engine **only as a read-only forward-truth oracle** (via
> `PersistentRayOpticsRunner`) and copies — does not mutate — the element geometry from
> `fold_demo.py`. Outputs land under `ns_autodiff/results_refine_autodiff/` and never
> overwrite (unique filenames + JSON sidecars), per the standing render rule.

---

## Why a differentiable tracer at all

`refine_lm.py` already shows local refinement works (70–80 % spot improvement) **without**
autodiff: at 7 DOF, a finite-difference Jacobian (`2·n_opt` traces/step ≈ 10 s/champion)
is cheap and sufficient. The design doc is explicit that you only *need* the differentiable
twin when **DOF count explodes** (multi-element / freeform / many fields), where the FD
Jacobian's serial traces become the bottleneck and a batched autograd pass on a GPU wins.

This package builds that twin **now**, at the current 7-DOF scale, for three reasons:
1. **Validation.** Prove the differentiable forward reproduces the honest JS engine
   bit-for-bit (it does — to ~1e-6 mm / ~1e-10 energy), so the twin can be trusted as the
   forward model when the problem grows.
2. **Answer the open question.** "It'll be interesting to see how the gradients would be
   constructed for a non-cylindrically-symmetric framework" — they are constructed here,
   and the analytic Jacobian agrees with `refine_lm`'s finite differences (cosine ≈ 1.0).
3. **A drop-in faster refiner** that already matches `refine_lm`'s published numbers.

---

## The key fidelity insight (read this before editing)

The face labels `A/B/T/D` in a recorded signature are a **positional classification of the
hit point** (`x<cx → A`, `x>cx → B`, `y` at the edges → `T/D`), **not** stable identifiers
of which physical surface was struck. A strongly meniscus element (e.g. cell `0,1`, where
`fsag=-18` pushes the *left* arc's apex to `x=97.5 > cx=85`) transmits through the geometric
"B" side **twice** (`T@B → T@B`) even though the ray physically enters the right arc and
exits the left arc.

So the twin does **not** trust labels for geometry. It does what the NS engine does:

> at each step, intersect the ray with **all four** surfaces (both circle roots per arc,
> kept only if on the real arc via a **chord-side test**; one hit per edge, kept only if
> within the segment), take the **nearest forward** hit, and decide refract-vs-TIR from
> **physics** (`sq1 = 1 − n1²(1−cos1²) < 0 ⇒ TIR`).

The champion only freezes (a) *how many* interaction steps occur and (b) the expected
type/label per step (used for validation). The nearest-surface `argmin` and the TIR test
are locally constant within a fixed topology, so the trace is differentiable — the gradient
flows through the selected surface's geometry and the smooth Snell/Fresnel arithmetic, with
**no `inf` allowed into any differentiable tensor** (an early `0·inf` from `where(..., inf)`
masking produced NaN gradients; the fix keeps masked-out candidates finite and puts `inf`
only in a *detached* selection cost consumed by `argmin`).

---

## File map

| File | Role |
|------|------|
| [`geometry_torch.py`](geometry_torch.py) | Differentiable primitives: circumcenter (arc from 3 path points), ray–circle (both roots), ray–line, chord-side `on_arc` test, reflect, vector-form Snell (`snell_refract`), Fresnel `R_s/R_p`, critical angle. Each is a faithful port of `ray-optics/src/core` (line refs in the docstrings). |
| [`element_torch.py`](element_torch.py) | Builds the 4 element surfaces (2 arcs A/B + 2 edges T/D) and the input ray from the 9-param genotype — a torch port of `fold_demo.element()/scene()`. |
| [`topology.py`](topology.py) | Runs the JS oracle once, extracts the dominant typed signature and the tracked-ray set (by beam offset `dy`), records each ray's oracle sensor `y`/energy + frozen weight. Mirrors `refine_lm.dom_records_by_dy`. |
| [`forward.py`](forward.py) | The batched differentiable forward trace over the unrolled frozen graph, carrying **s/p energy channels separately** (input is unpolarized `0.5/0.5`, matching `SingleRay.js`). Returns sensor `y`, delivered energy, and TIR incidence for the margin guard. |
| [`loss.py`](loss.py) | Energy-weighted RMS spot (frozen weights, floating centroid) + throughput-floor penalty — the same objective `refine_lm.Objective` minimizes, as a torch graph. |
| [`validate.py`](validate.py) | **The success gate.** (1) forward-match twin vs oracle; (2) analytic-vs-FD Jacobian cross-check. |
| [`refine_autodiff.py`](refine_autodiff.py) | The refiner: LM (analytic J) or Adam, with the `refine_lm` guards (throughput floor, no-erosion TIR margin floor, trust region), then **honest re-validation through the JS oracle** at 21-ray density (topology preserved? honest spot? TIR past critical?). Writes a never-overwrite JSON report + 6-panel diagnostics figure. |

---

## Validation results (reproduce with the commands below)

**Forward match — twin vs JS oracle at the champion geometry** (max abs error over 121 rays):

| cell | topology | sensor-y err | energy err |
|------|----------|-------------:|-----------:|
| `0,1` | `T@B → T@B` (meniscus) | 4.0e-7 mm | 1.3e-11 |
| `3,5` | `T@A → TIR@D → TIR@B → TIR@B → T@B` | 5.9e-7 mm | 4.6e-10 |
| `4,11` | `T@A → TIR@D → TIR@B → TIR@A → TIR@T → T@B` | 2.2e-7 mm | 3.0e-10 |
| `3,10` | `T@A → TIR@D → TIR@B → TIR@A → T@B` | 1.0e-6 mm | 1.6e-10 |

→ the twin is a **faithful reimplementation**, not an approximation (sub-micron geometry,
machine-precision energy, including a revisited surface and a flat-edge TIR).

**Jacobian match — analytic autograd vs `refine_lm`-style finite difference** (cell 3,5):
cosine(analytic, FD) = **0.999999**. The only sizeable per-DOF gap is `bsag` (the stiffest
direction): that is FD *truncation error* on the high-curvature DOF — the analytic gradient
is the exact one. This is the documented ill-conditioning (κ(J) ≈ 1e9) made visible.

**Refinement — autodiff LM vs `refine_lm`'s published numbers** (oracle 21-ray honest spot):

| cell | this package (autodiff LM) | `refine_lm` (FD+LM) | topology |
|------|---------------------------:|--------------------:|----------|
| `3,5` (no-erosion) | 2.42 → **0.66** (−72.6 %) | 2.42 → 0.67 (−73 %) | preserved |
| `3,5` (margin-floor 1.0° knee) | 2.42 → **0.51** (−78.8 %), thru ↑ | 2.42 → 0.51 | preserved |
| `4,11` | 0.66 → **0.13** (−79.8 %) | 0.66 → 0.13 (−80 %) | preserved |

Adam (first-order) on cell 3,5 reaches only −57.9 % vs LM's −72.6 %, confirming the doc's
point that **damped least squares is the right tool** for the κ≈1e9 aberration Jacobian.

**Honest caveat (faithful to `refine_lm`).** Cell `3,10` improves the dense twin spot
4.88 → 0.15 (−96.8 %) but the oracle 21-ray honest spot only halves (9.31 → 4.60) while the
TIR margin collapses toward critical. This is the *same* proxy-vs-honest divergence the work
summary flags (§8: "add an honest-metric acceptance guard"); the twin reproduces it rather
than hiding it. The fix — re-scoring the honest 21-ray spot at each accepted step — is the
top-ranked next step, and applies identically here.

---

## Run it

```bash
PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python      # numpy+torch; repo python3 lacks numpy

# 1) validate the twin against the oracle (forward + Jacobian)
$PY -m ns_autodiff.validate --cell 3,5 --jac
$PY -m ns_autodiff.validate --cell 0,1                       # the meniscus edge case

# 2) refine (autodiff Levenberg-Marquardt, analytic Jacobian)
$PY -m ns_autodiff.refine_autodiff --cell 3,5
$PY -m ns_autodiff.refine_autodiff --cell 4,11
$PY -m ns_autodiff.refine_autodiff --cell 3,5 --margin-floor 1.0     # tightness knee

# 3) first-order variant
$PY -m ns_autodiff.refine_autodiff --cell 3,5 --optimizer adam --iters 400 --lr 8e-3
```

Requires Node.js on PATH (the oracle) and the `deeplens_env` python (`torch 2.5.1+cu121`,
numpy, matplotlib). Everything here runs on CPU float64 for exact oracle matching; the
forward is batched over rays, so it moves to GPU unchanged when the problem grows.

---

## When to use this vs `refine_lm.py`

* **`refine_lm.py`** — the default for the current single-element, 7-DOF folds. No torch, no
  twin to keep in sync; FD+LM is enough.
* **`ns_autodiff/`** — when DOF count explodes (multi-element / freeform / many fields), so
  the FD Jacobian's serial traces dominate; or when you want exact gradients (the FD
  Jacobian is noisy on stiff DOF like `bsag`). The forward is already validated against the
  oracle, so it can replace the JS engine as the differentiable forward model at scale.

## AR/VR display-prism extension (`*_arvr.py`)

A second problem the twin now handles: the off-axis near-eye **display-prism relay** exported
to `ns_probe/results_arvr_qd/autodiff_export/`. Three differences from the spot fold, all
isolated in `*_arvr.py` (the geometry primitives in `geometry_torch.py` are shared unchanged):

| spot fold | AR/VR prism |
|---|---|
| 2 arcs + 2 flat edges | **4 arc faces** (`element_arvr.build_surfaces_arvr`) |
| one collimated beam | **3 display point-source fans** (explicit per-ray origins) |
| spatial spot at a sensor | **angular collimation**: per-source exit-direction spread `sigma_s` |

**Imaging condition = collimated, not focused.** A near-eye relay forms a virtual image at
infinity: each display pixel must leave the pupil as a *parallel* bundle (the eye focuses it
onto the retina). So the objective minimises the per-source angular spread `sigma_s`, and
different pixels map to different mean exit angles `theta_s` — their spread is the FOV. (This
is the opposite of the spot fold, where rays are *focused* to a point.)

* `forward_arvr.py` — per-ray trace (per-ray medium/refract-vs-TIR/energy), exit **direction**
  at the pupil plane. Physics mode follows the bright image bundle; a type-driven mode
  reproduces the exact recorded path (incl. Fresnel ghosts) for validation.
* `validate_arvr.py` — forward-matches all 12 exported designs to **~5e-6° / ~5e-9 energy**.
* `refine_arvr.py` — autodiff Levenberg-Marquardt on `sum_s sigma_s²` with an **honest-metric
  acceptance guard** (drives with the twin gradient, returns the best-by-oracle iterate). The
  collimation Jacobian is near-rank-deficient (κ ≈ 1e20) and the QD designs start already
  well-collimated, so the guard is essential — without it the twin proxy overfits and the
  true fitness regresses.

Refining all 12 (`--all`): **10/12 improved, 0 regressed** (guard guarantees no regression).
Mean oracle fitness 0.552 → 0.573, mean collimation 0.151° → 0.138°; best gains cell 8,19
**+17%**, 6,16 +7%, 5,6 +7%. Reports under `autodiff_export/refined/`.

> A latent bug surfaced and was fixed here: a ray that *misses* a circle (`disc<0`) was being
> clamped to a spurious closest-approach "hit" that could win the nearest-surface argmin.
> `ray_circle_roots` now marks `disc<0` candidates invalid — this fixed 7/12 AR/VR designs and
> hardens the spot-fold tracer too.

## Next steps

1. **Honest-metric acceptance guard** (shared with `refine_lm` §8): reject any step that
   worsens the oracle 21-ray spot — fixes the cell `3,10` proxy divergence. (Already
   implemented for the AR/VR collimation refiner in `refine_arvr.py`.)
2. **GPU batch** all rays + (for higher DOF) multi-champion refinement.
3. **Multi-element extension**: the per-interaction ops are already local and
   symmetry-free, so additional elements are more surfaces in `build_surfaces` + a longer
   frozen event list — no change to the autograd machinery.
