# Runbook — for the operator/agent with CODE V access

You have something nobody who built this package had: **the ability to run CODE V.** That
makes you the only one who can close the remaining questions. Please work in the loop
below rather than rewriting the file — the geometry in it is verified and the failures so
far have all been CODE V *conventions*, not optics.

---

## Ground rules

1. **Do not change geometry numbers.** Vertices, tilts, radii and apertures are generated
   and cross-checked by `verify_v2.py` / `check_seq_v2.py`. If you believe a number is
   wrong, change it in `verify_v2.py`, re-run both scripts, and regenerate — never edit the
   `.seq` alone, or the file and its proof drift apart.
2. **Change one thing at a time.** Every previous cycle that changed two things wasted a
   run. See `history/FAILURE_LOG.md`.
3. **Read the Error Log, not just the viewport.** The most damaging defect found so far was
   a *warning*, and the file loaded and drew fine while being physically wrong.
4. **Re-run the static checker after any edit:** `python3 check_seq_v2.py <file>`. It exits
   non-zero on failure and catches several classes of defect without CODE V.

---

## Step 0 — sanity, before CODE V

```bash
python3 verify_v2.py         # expect: "ALL FIELDS CLOSE"
python3 check_seq_v2.py      # expect: "ALL CHECKS PASS"
```

If either fails, stop — the package is inconsistent and nothing downstream is meaningful.

---

## Step 1 — load `benchmark_tierA_v2c.seq`

This file has **never been run**. It differs from the last-run version (`history/
benchmark_tierA_v2b.seq`) in exactly two ways: `GLB` → `DAR`, and `REFL` no longer carries
a glass token.

Then capture, in this order:

**(a) Error Log** — full text. Distinguish `Error:` from `Warning:`. Expected-benign
warnings are listed at the end of `CODEV_SYNTAX_FINDINGS.md`.

**(b) Lens Data Manager** — specifically these columns:

| check | expected |
|---|---|
| surface 3 Refract Mode | **Reflect** |
| surface 4 Refract Mode | **Only TIR** |
| surface 4 Y Radius | flagged **`P`** (pickup from surface 2) |
| surface 10 Thickness | flagged **`S`** (PIM solve), and **positive** |
| Non-Centered Data, surfaces 2–5 | **Decenter & Return** |
| Non-Centered Data, surface 6 | **Basic Decenter** |
| Glass column | named `'PRISM'`/`'CROWN'`/`'FLINT'`, **never** `100000.*` |

**(c) View Lens** — does it draw a **closed prism block**, or scattered faces?

**(d) Global coordinates** — the decisive test. List surface global coordinates and compare
surfaces 2–5 against `reference/BENCHMARK_PROMPT_V2.md` §2:

```
face A (CODE V S2 and S4)  vertex (0,  0.000000, 18.000000)   tilt +12.8562
face B (CODE V S3)         vertex (0,  0.760776, 27.679225)   tilt -16.8798
face C (CODE V S5)         vertex (0, 27.495340, 17.878452)   tilt +52.9653
```

Tolerance <0.05 mm / <0.05°. **This single check decides whether `DAR` solved the placement
problem.** Report it even if everything else fails.

---

## Step 2 — branch on what you see

### If global coordinates are CORRECT and the chief ray traces
Objective A is done. Go to Step 3.

### If global coordinates are CORRECT but the chief ray still fails
The placement hypothesis was right but insufficient. In order, one at a time:
1. Turn on **Reference Ray Aiming** (System Data → Advanced). This previously produced real
   rays for the first time in an earlier build.
2. Check whether TIR is actually occurring at surface 4 — if CODE V reports a ray failing
   at a "Only TIR" surface, the incidence has fallen below critical, which means the
   geometry is not what the file says (go back to Step 1d).
3. Only then consider `CRA`.

### If global coordinates are WRONG
`DAR` does not behave as assumed on this build. **Do not patch — shrink the problem.**
Strip to a 3-surface system (stop + face A + face B), get *those* placed correctly, then
re-add one surface at a time. This is the protocol in
`target_fold_design/INCREMENTAL_BUILD.md` §4 and it is the only method that has reliably
made progress here.

While you are at it, this is the moment to settle the open syntax question:

```
Help -> Command Reference -> GLB
```

Capture the actual signature. `GLB` is the *native* way to do this and would be preferable
to the `DAR` workaround. Two guesses have already been spent (see
`CODEV_SYNTAX_FINDINGS.md` §3) — please read the real syntax rather than trying a third.

### If reflections are missing again (Refract Mode says "Refract")
The mode token is being dropped. Check the Glass column for `100000.*` and re-read
`CODEV_SYNTAX_FINDINGS.md` §1 — the fix is always "one glass token per `S` line".

---

## Step 3 — once it traces

Only now do metrics mean anything, and **only partially**, because the relay group is a
known non-physical stand-in (README §4). In order:

1. **Real-ray trace, all 5 fields × 3 wavelengths.** Confirm rays reach SI. This is rubric
   check 6, the only item that requires CODE V.
2. **Confirm the TIR margin.** At surface 4 the incidence must exceed the critical angle at
   every field and pupil point. `verify_v2.py` predicts a worst case of **+4.67°** at the
   +22° field (C line). If CODE V disagrees materially, the geometry in CODE V is not the
   geometry we verified — that is a much more important finding than any spot size.
3. **Check `PIM`'s solved airspace is positive.** It was −0.2463 mm when the prism was
   scattered.
4. **Redesign the relay before any image-quality metric.** R2 = −4.8 mm caps that doublet at
   Ø9.6 mm against a beam that is ~26 mm wide at surface 5. Spot size, MTF and wavefront
   error are all meaningless until this is fixed. This is the first real *design* task in
   the package as opposed to a transcription task.

---

## Step 4 — Objective B, the real design

Only after Objective A traces. `target_fold_design/` holds the 4-TIR-bounce design that
motivated all of this, its exact validated geometry, and the analysis of why its current
`.seq` scatters. Start with `target_fold_design/NOTES.md`.

---

## What to report back

Even a total failure is valuable if it is specific. Please include:

- Full Error Log text (errors **and** warnings, verbatim).
- The LDM column checks from Step 1b.
- **The global coordinate listing** (Step 1d) — the single most useful artifact.
- A View Lens screenshot.
- If you resolved `GLB`'s syntax: the exact signature.

The failure mode that has cost the most time here is a file that *loads cleanly and looks
plausible while being physically wrong*. Verbatim log text beats a summary judgement.
