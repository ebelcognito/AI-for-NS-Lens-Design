# AR/VR display-prism QD discovery — full context

> **What this is.** The complete record of the AR/VR near-eye display-prism effort built in the
> 2026-06-19 session (Opus 4.8), a sibling to the spot-fold refinement work in `00`/`01`. We
> repurposed the non-sequential QD engine to **discover freeform TIR prisms that relay a
> microdisplay to the eye**, and got the headline result: **TIR folds emerge across the whole
> archive once (and only once) the geometry is off-axis so a straight path is impossible.**
> Then we exported the designs for the differentiable twin (`ns_autodiff`) to refine.
>
> **Environment:** all Python runs use `/home/hisham26/Desktop/simu1/deeplens_env/bin/python`
> (numpy/matplotlib). The repo's bare `python3` is the bluedot venv and lacks numpy.

---

## 1. The problem & the locked decisions

Classical AR freeform prisms relay an off-axis microdisplay into the eye's forward line of
sight via internal (TIR) folds — exactly the tilted multi-bounce family this NS-QD framework
discovers. We pointed the discovery engine at it. Decisions locked with the user:

- **Imaging = collimated** (virtual image at infinity): each display point must exit the pupil
  as a **parallel** bundle → the metric is **angular**, not spatial.
- **Scope = relay path only** (display → eye); no see-through/world path yet.
- **Surfaces = TIR-only** (uncoated, loss-free); no coatings.
- **Archive axes = FOV × form-factor** (field of view × prism depth).

## 2. Forward model & the angular metric (`ns_probe/arvr_prism.py`)

- **Microdisplay → 3 point sources**, each emitting a diverging fan (a pixel radiates). A
  tracked ray's identity is `(source, fan_angle)`.
- **Searched TIR prism** (the genotype, §4) folds the light.
- **Eye-pupil plane** at `PUPIL_X=120` with an **eyebox aperture** (`|y-EYEBOX_Y| ≤ EYEBOX_HALF`)
  — the aperture stop; only light through it counts.
- Per source *s* (eyebox rays): `T_s` = throughput, `θ_s` = mean exit **direction** (field
  angle), `σ_s` = angular spread (collimation error, deg).
- **Fitness** = `mean_s [ T_s · exp(−σ_s/σ0) ] × ordering_ok` (σ0=0.7°; ordering_ok requires a
  strictly monotonic field map → no scrambled image). Never names prism/TIR/fold.
- **Descriptors**: `BC1 = FOV` (|θ_+edge − θ_−edge|), `BC2 = prism depth` (x-extent of glass the
  light uses; compact = small).

The engine (`raySequences`) already records per-ray `dirx,diry`+energy at any plane, so the
angular metric needed **no engine change**.

## 3. The geometry story — three findings, in order

**(a) On-axis = negative.** First build put display and eye collinear (both on the x-axis).
QD filled a 482-cell FOV×depth archive of real collimating relays (best **0.43**), but
**folds did NOT emerge: only 12/482 cells folded; the best designs were TIR=0 transmission
collimators.** Correct, and informative: a straight relay needs no fold.

**(b) Off-axis = the fix and the emergence.** Mounted the display **above, firing straight
down**; eye looks **forward** — a ~90° turn is now geometrically required. This is the real
AR-combiner layout. (NB: the engine renders y-down, so the display appears at the *bottom* of
renders — cosmetic flip only.)

**(c) The ±50° tilt unlock.** Off-axis initially gave only **1 ray/field** through the eyebox
(fitness 0.05). Root cause: **a 90° fold needs a ~45° TIR surface, but face tilt was capped at
±25°** — the geometry literally couldn't represent the diagonal fold surface. Widening tilt to
**±50°** fixed it instantly (fitness 0.05 → **0.63**); the search found `tilt_a ≈ 45°` exactly
as predicted.

## 4. The element & genotype (4 curved faces — user's idea)

`prism_element(p)` builds a **4-curved-face freeform prism** (all four edges are arcs:
top/right/bottom/left), each a circle through corner–apex–corner. **9-DOF genotype:**
`[cx, thick, fsag, bsag, tsag, dsag, tilt_a, tilt_b, n]` (left/right/top/bottom sags + 2 face
tilts + index). Ranges: `SAG=±15` per face, `TILT=±50°`, `n∈[1.5,2.0]`, `cx∈[35,85]`,
`thick∈[6,45]`. Why 4 curved faces matter *off-axis* (and didn't on-axis): the **top edge is
now the entry surface and the bottom/side a TIR-fold surface**, so curving them adds power on
the surfaces the imaging light actually uses — the real freeform-prism recipe (3–4 working
surfaces).

## 5. Headline result — off-axis single-fold archive

`run_arvr_qd.py --evals 40000` (≈9.4 min, 20×28 archive). **Fold emergence is now positive
and dominant:**

| | on-axis | **off-axis** |
|---|---|---|
| folded cells (≥1 TIR) | 12 / 482 | **488 / 493** |
| median TIR (all depth bands) | 0 | **1** (up to 7) |
| best design | TIR=0 transmission | **TIR=1 fold** |
| best fitness | 0.43 | **0.65** |

- **Champion** (`results_arvr_qd/`, best=compact): fitness 0.651, FOV 11.7°, depth 22.6, TIR=1,
  3 fields collimated to **σ < 0.1°** at **65–73% throughput**.
- **Emergent structure→function law:** multi-bounce designs (3–7 TIR) cluster at **wider FOV +
  thinner prism** — folding deeper packs more field into less depth.
- Off-axis folded relays collimate *better* than on-axis transmission (0.65 > 0.43).

## 6. Multi-fold variant (forcing ≥2 TIR)

User asked to "tilt the bottom source left to induce multi-fold." **Tested honestly: source
tilt alone does NOT induce it** — with ±50° tilt a single ~45° surface always does the 90° turn
in one bounce (swept 0–25° left; all stayed TIR=1, just lower fitness). **The lever is the
opposite: NARROW the fold-surface tilt** so no single surface can turn 90° → the light
**zig-zags through 2+ TIRs**. Run: `run_arvr_qd.py --evals 18000 --tilt-range 28 --source-tilt
12 --min-tir 2 --out results_arvr_qd_multifold` (kept the 12° source tilt; hard ≥2-TIR filter).

Result: **342/342 cells ≥2 TIR** (median 2, up to **8 bounces**), best fitness **0.243**
(FOV 14.7°, TIR=2). Multi-fold **trades both collimation quality and compactness** — best
multi-fold is deeper (depth 69 vs 22) and looser (σ 0.24–0.74° vs <0.1°), because each TIR adds
aberration the next surface only partly corrects. Topology = `refract→tir→tir→refract`.

## 7. Export for the differentiable twin (`ns_autodiff`)

`export_arvr_for_autodiff.py` writes each design as an **AR-extended `FrozenTopology`** (the
twin's existing structure), with the 3 AR differences made explicit and validatable:

| twin (spot fold) | AR prism (export) |
|---|---|
| 2 arcs + 2 flat edges | **4 arcs** (`design.surfaces`, center/r/end1/end2/apex) |
| one beam, id=`dy` | **3 fans**, id=`(source,fan_angle)`; explicit `origin`+`input_dir` |
| sensor `y` (spatial) | **pupil exit direction** (`oracle_exit_dir/angle`, angular metric) |

Each design carries the frozen `type_sequence` (e.g. `refract→tir`, `refract→tir→refract`, or
`refract→tir→tir→refract`) + per-ray **oracle validation targets** so the twin can prove forward
agreement (~1e-6) before refining. **The exporter reads `source_aim` from the archive meta** so
it traces with the right source (critical: the multifold export uses −102°, not −90°).
Three engine extension points (build 4-arc surfaces, emit 3 fans, angular loss at pupil) and the
validation recipe are in the export READMEs.

- Single-fold export: `results_arvr_qd/autodiff_export/` (12 designs + README)
- Multi-fold export: `results_arvr_qd_multifold/autodiff_export/` (12 designs + README) —
  **the better refine targets** (looser σ = more headroom); start with `cell_7_21/8_21/11_22`.

## 8. Code & artifact map (`ns_probe/`)

**Code:** `arvr_prism.py` (scene + angular metric + Phase-A search + render),
`run_arvr_qd.py` (CMA-ME archive; `--tilt-range/--source-tilt/--min-tir`),
`analyze_arvr_qd.py` (heatmap + fold-emergence test + champion renders; `--dir`),
`render_arvr_gallery.py` (3×4 gallery), `export_arvr_for_autodiff.py` (twin export).
**Results:** `results_arvr/` (Phase-A renders, `arvr_offaxis_fold.png`),
`results_arvr_qd/` (single-fold archive 493 cells, `arvr_qd_archive.png` heatmap,
`arvr_{best,compact,widest_fov,best_folded}_2.png` big-crop renders, `arvr_gallery_top12.png`),
`results_arvr_qd_multifold/` (342-cell archive, renders, `arvr_gallery_top12_1.png` in
`results_arvr_qd/`). Render crop auto-fits the off-axis layout (`width=1800`).

## 9. Honest limitations & open directions

- **2-D meridional**, single prism, relay-only (no see-through), **modest FOV** (best ~12–25°;
  not chasing étendue per user). Angular σ is a collimation proxy, not full MTF.
- Multi-fold output points **~28–43° off forward axis** (constrained-tilt artifact) — re-centring
  is a search/constraint change, not a refinement one.
- **Next:** (1) refine champions with the `ns_autodiff` angular Jacobian (multi-fold = most
  headroom); (2) re-center / widen FOV; (3) add the see-through world path (true combiner);
  (4) dial fold depth (`--tilt-range/--min-tir`). Memory: [[arvr_prism_qd]].

## 10. Reproduce
```bash
PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
$PY ns_probe/arvr_prism.py --evals 8000                       # Phase-A: find one off-axis fold + render
$PY ns_probe/run_arvr_qd.py --evals 40000                     # single-fold FOV×form-factor archive (~9 min)
$PY ns_probe/run_arvr_qd.py --evals 18000 --tilt-range 28 --source-tilt 12 --min-tir 2 \
     --out ns_probe/results_arvr_qd_multifold                 # multi-fold archive (~5 min)
$PY ns_probe/analyze_arvr_qd.py --dir ns_probe/results_arvr_qd        # heatmap + fold test + champions
$PY ns_probe/render_arvr_gallery.py --archive ns_probe/results_arvr_qd/archive.json  # 3×4 gallery
$PY ns_probe/export_arvr_for_autodiff.py --archive ns_probe/results_arvr_qd/archive.json \
     --out ns_probe/results_arvr_qd/autodiff_export           # export for the differentiable twin
```
