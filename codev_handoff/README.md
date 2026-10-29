# CODE V handoff package

> **Start at the `codev` skill, not here.** `.claude/skills/codev/SKILL.md` carries the
> conventions, the loop and the current target design in a form small enough to load every
> time; its `references/` hold the condensed syntax, geometry, failure and freeform notes.
> This package remains the home of the *executable* parts — `verify_v2.py`,
> `check_seq_v2.py`, the emitters, the `.seq` files and the Tier A benchmark — plus the
> unabridged history. Read the long docs below only when the skill's references send you.

Everything needed to get an off-axis freeform-prism design tracing in **CODE V 2026**,
assembled for a operator/agent that has **direct CODE V access and a command line** —
which the authors of this package did not.

Read this file, then `AGENT_RUNBOOK.md`. Everything else is reference.

---

## 0. The one-paragraph version

We have a discovered folded-prism optical design and we want it in CODE V so it can be
rendered, ray-traced, benchmarked and optimised. CODE V models this class **sequentially**
— the surface list simply names a physical face more than once. Getting the geometry right
was never the hard part; **getting CODE V's coordinate/tilt/reflection conventions right
was**, because we could not run CODE V to test against. Three live runs have now settled
most of those conventions. `benchmark_tierA_v2c.seq` is the current best file. Its geometry
is independently verified; its remaining unknowns are all CODE V behaviours that one run
will resolve.

---

## 1. Two objectives, in order

### Objective A — get `benchmark_tierA_v2c.seq` tracing (do this first)

A 3-face off-axis prism eyepiece: **refract in → mirror → TIR off the entry face → refract
out**, then a relay to a microdisplay. 5 fields (0, ±11, ±22°), 3 wavelengths, 8 mm pupil.

This is the *proving ground*, not the end goal. It is deliberately the simplest system that
still contains every hard convention: a face listed twice, a mirror **and** a TIR surface
that must not be interchanged, off-axis tilts, and a global-coordinate placement problem.
Its geometry is **verified satisfiable** — all 5 fields close over the full pupil at all 3
wavelengths (`verify_v2.py`). If this traces, the conventions are settled.

### Objective B — the real design (`target_fold_design/`)

The design we actually care about: a 4-TIR-bounce folded prism discovered by CMA-ME search
in this repo. Its geometry is exact and validated against the source engine to <0.001 mm,
but its `.seq` scatters in CODE V. **Do not start here.** Everything learned in Objective A
transfers directly; see `target_fold_design/INCREMENTAL_BUILD.md`.

---

## 2. Current state — read this before touching anything

| item | status |
|---|---|
| Prism geometry (vertices, tilts, radii, apertures) | **verified**, twice, independently |
| All 5 fields close over full pupil × 3 wavelengths | **verified** (`verify_v2.py`) |
| `.seq` matches that geometry to <0.05 mm / <0.05° | **verified** (`check_seq_v2.py`) |
| Reflections real in CODE V (`Reflect`, `Only TIR`) | **confirmed live**, run 3 |
| Private-catalog glasses + dispersion | **confirmed live**, run 3 |
| Pickups (`PIK`) and paraxial solve (`PIM`) | **confirmed live**, run 3 |
| Surfaces placed at correct **global** coordinates | **UNRESOLVED** — the live blocker |
| Chief ray completes at 5 fields | **UNRESOLVED** — expected to follow from the above |
| Relay group is physically realisable | **NO — known defect**, see §4 |

**`benchmark_tierA_v2c.seq` has never been run in CODE V.** Runs 1–3 were of its
predecessors (`benchmark_tierA_v2.seq`, `_v2b.seq`). v2c fixes the two defects run 3
exposed. Its static checks all pass. Treat the DAR placement in it as a **hypothesis to
test**, not a finished result.

---

## 3. What "success" looks like

In ascending order. Report which rung you reach.

1. File loads with **no errors** in the Error Log (warnings about image-surface apertures
   and decentred apertures on paraxial `GO` are expected and benign).
2. Lens Data Manager shows: surface 3 Refract Mode = **Reflect**, surface 4 = **Only TIR**,
   surface 4 radius flagged **`P`** (pickup), surface 10 thickness flagged **`S`** (solve).
3. Non-Centered Data reads **Decenter & Return** on surfaces 2–5 and **Basic Decenter** on
   surface 6.
4. **View Lens shows a prism** — one closed glass block, not scattered faces.
5. **No "Unable to find chief ray"** — this is the headline goal.
6. Real rays trace end to end at all 5 fields × 3 wavelengths.
7. Global coordinates of surfaces 2–5 match §2 of `reference/BENCHMARK_PROMPT_V2.md`,
   and the centres of curvature match the three listed in the `.seq` header.

Rungs 1–4 and 7 are checkable by inspection. Rung 5 is the one that has blocked this work
from the beginning.

---

## 4. Known defects — do not spend time "fixing" these

**The relay group is not physically realisable.** The prescription asks for a Ø26 mm
cemented doublet whose radii are R1 = +6.4 and R2 = −4.8 mm. A 13 mm semi-aperture on a
4.8 mm sphere does not exist; the group is capped at Ø9.6 mm. `BENCHMARK_PROMPT_V2.md` §4
concedes the relay is a stand-in "not a manufacturable f/#". Apertures in v2c are set just
under the radii so the warning clears. **It will vignette badly and any image-quality
metric (spot size, MTF, wavefront) is meaningless until the relay is redesigned.** It does
**not** obstruct the chief ray, so it does not block Objective A.

**`PIM` solved a negative airspace (−0.2463 mm) on run 3.** Expected while the prism was
scattered. If it is still negative once the layout is correct, the relay needs redesigning
— that is the first task of any metrics work.

**Apertures on the image surface are ignored by CODE V by design.** It says so. They are
kept in the file to document the 12.0 × 10.0 mm display. Add a dummy surface in contact
with SI if you want the display to actually clip.

---

## 5. File map

```
benchmark_tierA_v2c.seq     <- THE FILE. Current best. Run this.
verify_v2.py                <- geometry SOURCE OF TRUTH. Generates every number in the
                               .seq from 8 parameters + proves all 5 fields close.
check_seq_v2.py             <- static checker: parses a .seq back and verifies it against
                               verify_v2. Covers every rubric item not needing CODE V.

AGENT_RUNBOOK.md            <- step-by-step for the CODE V operator. START HERE after this.
CODEV_SYNTAX_FINDINGS.md    <- empirically confirmed CODE V syntax. Highest-value doc;
                               every line was paid for with a live run.
BUILD_METHOD.md             <- how the geometry is constructed and why: coordinate frames,
                               surface instantiation, ray-surface interaction, the three
                               construction strategies we tried.

reference/
  BENCHMARK_PROMPT_V2.md    <- the prescription, answer key (§2) and rubric (§3).
  trace3d.py closure.py fixed_key.py chief_table.py benchmark_answer_key.py
                            <- independent V1 checkers; these are what FOUND the V1 defect.

history/
  benchmark_tierA.seq       <- V1. Plane faces. Geometry DEFECTIVE (only 2 of 5 fields).
  benchmark_tierA_v2.seq    <- run 1: glass pair ate REFL; GLB rejected.
  benchmark_tierA_v2b.seq   <- run 3: reflections worked; GLB rejected again.
  BENCHMARK_PROMPT.md       <- V1 prompt, incl. the still-valid Tier B freeform task (§4).
  FAILURE_LOG.md            <- what each run proved. Read before re-trying anything.

target_fold_design/         <- OBJECTIVE B. Not yet working. See its NOTES.md.
```

## 6. Quick start

```bash
python3 verify_v2.py        # regenerate + prove the geometry      -> "ALL FIELDS CLOSE"
python3 check_seq_v2.py     # check the .seq against it            -> "ALL CHECKS PASS"
python3 check_seq_v2.py history/benchmark_tierA_v2b.seq   # see it catch a real defect
```

Both scripts need only `numpy` and exit non-zero on failure, so they are CI-safe.
Then open `AGENT_RUNBOOK.md`.
