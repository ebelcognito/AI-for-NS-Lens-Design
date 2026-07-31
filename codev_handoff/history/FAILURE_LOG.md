# Failure log — what each run actually proved

Read this before re-trying anything. Several of these look like obvious things to attempt.

---

## Run 0 — `benchmark_tierA.seq` (V1, plane faces)

**Never a syntax problem — a geometry problem.** The V1 prescription was **unsatisfiable**.
Re-tracing it independently (`reference/trace3d.py`, `closure.py`, `fixed_key.py`) showed:

- chief rays close only over roughly **−11° to +2°** of field, not the ±22° prescribed;
- at +11° and +22° the post-TIR ray hits the exit *plane* behind itself (negative segment);
- at −22° the entry-face incidence is 37.4°, **under** the 40.8° critical angle → no TIR;
- required apertures were ±28.6 / ±14.5 / ±18.0 mm against stated ±10 / ±8 / ±7;
- with the full 8 mm pupil enforced, **no** field closed at all.

The V1 answer key was central-ray-only, which is exactly how an unsatisfiable spec got
through. **Lesson: verify over the full field × pupil × wavelength grid, never one ray.**
The repair (V2) needed spherical rather than plane faces, honest apertures, and a re-solved
fold. `verify_v2.py` exists to make this class of defect impossible to repeat.

---

## Run 1 — `benchmark_tierA_v2.seq`

Two defects, one catastrophic and near-invisible.

**(a) The silent-drop trap.** `S <rdy> <thi> 1.53 56.0 REFL` →
`Warning: Extra data '56.0 REFL' ignored`. The `S` line takes one glass token; given a
pair, CODE V read `1.53` as a zero-dispersion fictitious glass and **discarded the rest of
the line, including `REFL` and `REFLTIRO`**. The lens had **no reflective surfaces at all**
— LDM Refract Mode read "Refract" on every surface — and all three wavelengths traced
identically. Emitted as a *warning*; the file loaded and drew.

**(b) `GLB S2 S1` → `Error: Too many surface qualifiers`.** Every global reference failed,
so the LDM fell back to "Basic Decenter".

Also surfaced: the relay doublet's Ø26 mm aperture is impossible against R1 = +6.4,
R2 = −4.8 (`Aperture value is larger than radius. Reduced.`) — a defect in the
prescription, not the file.

**What already worked, and still does:** all seven radii correct; `PIK` took (LDM `P` on
surface 4); `PIM` took (solved 38.1647 mm); View Lens showed five collimated bundles
crossing cleanly at the 8 mm stop. **The input/stop setup has been fine since run 1** —
which is worth knowing, because it blocked the earliest builds for a long time.

---

## Run 2/3 — `benchmark_tierA_v2b.seq`

**Fixed and confirmed:** private-catalog glasses work, dispersion is real, and the
reflections became genuine — LDM showed surface 3 **"Reflect"** and surface 4
**"Only TIR"**.

**Two new findings:**

**(a) `GLB S2 1` → `Error: A global surface qualifier is expected`.** Second failed guess.
`GLB` was abandoned at this point rather than guessed a third time.

**(b) `REFL 'PRISM'` → `Warning: Extra data 'PRISM' ignored`.** A reflection does not change
the medium, so `REFL` takes no glass token; CODE V inherits it correctly regardless. Note
the asymmetry — `TIRO 'PRISM'` *is* accepted.

**The decisive diagnostic.** With every `GLB` failing, placement fell back to chained local
decenters. That predicts something very specific, and the run confirmed it: **surface 2
landed correctly** — its base frame *is* surface 1's frame at the origin — so rays
refracted cleanly through it, and **every surface after it** was measured from an
already-tilted, already-displaced frame and landed nowhere near the prism. Hence "rays get
through the first surface and stop at the second."

**This also reframed the long-standing chief-ray error.** With the stop as surface 1 in air
and the object at infinity, the chief ray is just the field-angle ray through the stop
centre — there is nothing to solve. `Unable to find chief ray` was CODE V failing to
*complete* it through scattered surfaces. It is a **symptom of the placement bug**, not an
independent problem. Earlier sessions spent considerable effort on `CRA` and Ray Aiming on
the assumption it was a cause.

Also seen: `PIM` solved a **negative** airspace (−0.2463 mm), expected while the prism was
scattered.

---

## Current hypothesis (untested)

`benchmark_tierA_v2c.seq` replaces `GLB` with `DAR` + `THI 0`, which produces global-
coordinate behaviour using syntax already proven in this project, and drops the glass token
after `REFL`. **If the placement diagnosis above is right, the chief ray should follow.**

That is a hypothesis, not a prediction — two `GLB` guesses have already been wrong. The
decisive evidence is the **global coordinate listing** for surfaces 2–5 (Runbook Step 1d),
which settles it regardless of whether the trace succeeds.

---

## Standing lessons

1. **Warnings can be catastrophic.** The worst defect found was a warning on a file that
   loaded and drew correctly. Read the Error Log every time.
2. **Verify over the whole field × pupil × wavelength grid.** A central-ray check passed an
   unsatisfiable design.
3. **Don't guess syntax twice.** Two `GLB` guesses cost two full cycles. Read the reference.
4. **Change one thing per run**, and prefer shrinking the system over patching it.
5. **Distinguish symptom from cause.** "Unable to find chief ray" was chased as a cause for
   a long time; it was downstream of placement the whole time.
