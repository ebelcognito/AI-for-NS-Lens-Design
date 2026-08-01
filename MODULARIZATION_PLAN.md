# Modularization plan

Status as of 2026-08-01. Phases **0, 0b and 1 are done**; phases 2-4 remain.

Gate for every commit: `python tests/run_all.py` (currently **5/5**).

## The finding that drives everything

The codebase has 15 different `PARAM_NAMES` lists, 4 element implementations, and ~250
scripts. They are not 15 different problems. Every genotype in the project decomposes
into exactly three groups:

| group | params seen across all scenarios |
|---|---|
| **body** | `cx`, `thick`, `n` (+ fixed `ELEM_H`) |
| **per-face** | tilt (`tilt_a`, `tilt_b`) + shape: `fsag/bsag/tsag/dsag` (arc) or `a2_X/a4_X` (freeform) |
| **source** | `ray_y, ang` (collinear) or `ray_x` (perpendicular) |

`fold_demo.element` and `arvr_prism.prism_element` are the same function — a closed
vertex path with four faces — differing only in whether T/D carry `arc: True`.
`prism_fold` has no element of its own: it imports arvr's and moves the source and
detector. `ns_autodiff/element_torch.build_surfaces` re-derives that identical geometry
in torch, with a comment cross-referencing fold_demo's vertex indices *by number*.

So the general parametric lens object is **derived from the existing code, not invented**:

```
Element(body  = Body(cx, thick, half_h, n),
        faces = {A: Face(tilt, Arc(sag)),  B: Face(tilt, Freeform(a2, a4)),
                 T: Face(0,    Flat()),    D: Face(0,    Flat())})
```

- `fold_demo`   = arcs on A/B, flat T/D
- `arvr_prism`  = arcs on all four
- `search_prism_freeform` = freeform on all four

Same object, three configurations.

### What actually varies between "scenarios"

Only five things. These are the module seams (Parnas: modularize on what changes
independently), and they are *not* the current file boundaries:

1. source placement
2. detector placement
3. which faces are curved
4. the fitness expression (~6 lines)
5. the descriptor pair

Everything else in ~250 scripts is scaffolding around those five axes. Measured
evidence: `run_fold_qd.py` vs `run_orderly_qd.py` are 189 and 149 lines; the only real
difference is the fitness. `analyze_disc_e1.py` vs `analyze_disc_prism.py` differ by 57
of 281 lines and share five identical helpers verbatim.

## Method

- **Beck, _Tidy First?_ (2023)** — structural and behavioural changes never share a
  commit. Non-negotiable here because the results are published.
- **Feathers, _Working Effectively with Legacy Code_** — characterization tests before
  touching anything. `ns_probe`/`ns_autodiff` had zero tests. Now pinned (phase 0).
- **Metz, "duplication is far cheaper than the wrong abstraction"** — the real risk.
  Mitigated by deriving `Element` from the four existing implementations bottom-up.
- **Ousterhout, _APoSD_** — 130 `__main__` files is textbook shallow-module sprawl.
  Target: few deep modules behind a thin CLI.
- **Fowler, strangler fig** — `nslens/` grows beside the old tree; old scripts become
  shims; shims die last. No big-bang rewrite.
- **Mikado Method** — ordering: attempt, observe breakage, revert, record prerequisites.

## Phase 0 — containment (DONE, 2026-08-01)

- Full archive: `~/Desktop/NS_lensdesign_ARCHIVE_2026-08-01.tar.gz` (239 MB, 2423
  entries; excludes only `ray-optics/node_modules`, regenerable via `npm ci`).
- Deleted `optical_rl/qd/` (incl. `lensmega`, `cmamega_*`, `render_theta_deeplens.py`)
  and the 6 `main_qd_*.py` that imported it. **265 -> 176 Python files.**
  Verified safe first: `ns_probe`/`ns_autodiff`/`codev_handoff` import exactly one thing
  from `optical_rl` — `utils.ray_optics_bridge_persistent`. Nothing else was reachable.
- Moved 46 `results_*` dirs (125 MB) to `runs/`, leaving compat symlinks because 70
  scripts and 28 docs hardcode the old paths. Symlinks come out in phase 3.
- Added `.gitignore` (ignores `runs/`, `node_modules`; keeps vendored `ray-optics/src`).
- Added `tests/characterization/pin_geometry.py` + `golden_geometry.json`.

### Regression evidence

| check | result |
|---|---|
| `codev_handoff/verify_v2.py` | ALL FIELDS CLOSE (5/5) |
| `benchmarks/fold1_env/verify_fold1.py` on the key design | 13/13 PASS, RMS 0.0008 mm, throughput 0.8800, defocus 0.00 |
| `pin_geometry.py` | PASS, 11 probes / 3 geometries |
| mutation test (`ELEM_H` 30.0 -> 30.001) | correctly FAILS, exit 1; cascades into `prism_fold` |

The pin independently reproduces both documented ground truths: the DEMO wedge's
`T@A -> TIR@B -> TIR@A -> T@B`, and the fold1 key's 1-TIR `T@A -> TIR@D -> T@B`.

> **deeplens correction.** There is no deeplens package in this repo. The string is a
> *virtualenv path* (`~/Desktop/simu1/deeplens_env/bin/python`) appearing only in
> docstrings — and that venv is the only Python on this machine with torch
> (2.5.1+cu121), which all 24 torch files in `ns_autodiff` need. Deleting it would break
> the autodiff half. The real fix is phase 1's `pyproject.toml`, which retires the name.

## Phase 0b — legacy `optical_rl` (DONE)

Deleted the stage-1 RL tree (`envs/`, `tests/`, `experiments/`, `configs/`,
`rebuttal_baselines_export/`, 10 `main_stage1_*.py`, `utils/scene_builder.py`).
`optical_rl` is now only the two ray-optics bridges plus the three helpers
`utils/__init__` imports. **265 -> 120 Python files.**

## Phase 1 — the core (`nslens/`) — DONE

Built and proven equivalent:

```
nslens/
  geometry/  shapes.py (Flat | Arc | Freeform), element.py (Body/Face/Element)
  scene/     sources.py (ParallelBundle | PointFan), layout.py (Detector/CropBox/Layout)
  engine/    runner.py (seam for the ray-optics process), signatures.py (moved here)
```

Equivalence is proven with **exact float equality**, not tolerance:

| check | result |
|---|---|
| `fold_demo.element` | 2000 genotypes exact (+ DEMO champion with coating) |
| `arvr_prism.prism_element` | 2000 genotypes exact |
| `geometry_valid` | 2000 agree (233 rejected by both) |
| `element_x_extent` | 2000 agree |
| `fold_demo.scene` / `prism_fold.scene` / `scene_arvr` | 500 scenes each, exact |

**Face-label trap, now fixed in one place.** The authoritative classifier maps
`y <= -half_h -> 'T'` and `y >= +half_h -> 'D'`, so **`T` is the BOTTOM edge and `D` is
the TOP**. `arvr_prism`'s docstring says the opposite and is wrong; `element_torch`'s
comment is right. The legacy sag names are crossed too: `dsag` lands on the bottom
(face `T`), `tsag` on the top (face `D`). `nslens` uses `left/right/bottom/top`
internally and pins the mapping in `LEGACY_FACE_LABEL`. Signatures still read
`A/B/T/D` so existing archives stay comparable. (Note the overload: in `T@A -> TIR@D`
the token before `@` is the *event*, after it the *face* — `T` means transmit on the
left and bottom-edge on the right.)

Remaining phase-1 tail, not yet done: `to_torch()` on `Element`, so `ns_autodiff` stops
re-deriving the same geometry. That is the prerequisite for folding `element_torch`,
`element_arvr` and `element_freeform` into the one object.

## Phase 1 (original sketch, for reference)

```
nslens/
  engine/      bridge (from optical_rl/utils), ledger, sequence signatures
  geometry/    Element, Body, Face, shapes: Flat | Arc | Freeform
               -> to_scene_dict()  (ray-optics adapter)
               -> to_torch()       (autodiff adapter, same object)
  scene/       Source (parallel bundle | point fan), Detector, Layout
  search/      cma_es, grid_archive, cma_me   (from ns_probe/cma_me.py)
  fitness/     focused_throughput, coherent, multifield   -> registry
  descriptors/ tir_bounces, thinness, tilt, locus, ...    -> registry
  refine/      lm, autodiff
  export/      codev, folded_trace   (ONE copy; 8 files are byte-identical today)
  viz/         archive_grid, gallery, champions
  cli/         qd, analyze, render, refine   (4 entrypoints, not 130)
configs/       one YAML per experiment
pyproject.toml
```

Ports-and-adapters is load-bearing: `Element` is engine-agnostic, with `to_scene_dict()`
and `to_torch()` as adapters. That is what stops `ns_probe` and `ns_autodiff` from
re-deriving the same geometry twice.

Gate: `pin_geometry.py` green after every commit.

## Phase 2 — strangle the runners

Six QD runners collapse to six ~20-line configs behind one driver. Each old script
becomes a shim that forwards to the driver, so every existing command line keeps working.

Gate: for one fixed seed, the new driver must reproduce an existing `archive.json`
cell-for-cell before its shim lands.

## Phase 3 — analyze / render / export

Seven `analyze_*` and 20 `render_*` collapse onto shared primitives. De-duplicate the 8
byte-identical files between `codev_handoff/` and `ns_probe/`. Repoint hardcoded result
paths at `runs/`, then delete the compat symlinks.

## Phase 4 — delete the shims

Once configs cover every recorded experiment, shims go. Expected endpoint: **~35-45
source files**, from 265.

## Resolved

1. **Repo location** — done. `NS_lensdesign` is now the repository, continuing the
   history and `origin` remote of the former hand-synced copy at
   `~/Desktop/AI-for-NS-Lens-Design` (which is now redundant and can be deleted once a
   push succeeds). `ray-optics/.git` was renamed to `.git-upstream` so the parent tracks
   the vendored engine as real files rather than a gitlink — without that, the
   uncommitted engine mods including `runner_persistent.js` would not have been tracked.
2. **Remaining `optical_rl` legacy** — deleted in phase 0b.

## Outstanding

- **`optical_rl/pat.txt` holds a live GitHub PAT** (`ghp_`, classic, 40 chars), on disk
  since 2026-07-01 and previously caught by push protection. It is gitignored and was
  never committed, but plaintext-on-disk plus a near-push means it should be
  **revoked and rotated**.
- The `origin` remote has not been pushed to since the restructure. `git push` will be a
  large diff (139 deletions in phase 0 alone); force is not required since history is
  continuous, but review it before pushing.
