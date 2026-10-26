# Replication: two-stage GA for a freeform-prism AR display (Su & Romanova, Appl. Opt. 65, 5158, 2026)

Paper: `agent_context/paper_pdfs/GA_paper.pdf` · spec notes: `agent_context/ns_systems/02_data_generation_methods_and_dataset_audit.md` §11.
The authors' MATLAB code is not public; everything here is re-implemented from the text and figures.

| File | What it is |
|---|---|
| `prism.py` | Batched 3D sequential tracer (pupil → A → B → A (TIR) → C → image), status score S (Eq. 7), wedge penalty (Eqs. 5/6), Stage-1 fitness (Eq. 8), Stage-2 objectives (Eqs. 9–12) |
| `test_prism.py` | Physics checks: mirror paraxial focus, Newton vs bisection on a tilted freeform, vector Snell invariants, TIR threshold, plate shift, ray grid |
| `probe.py` | Seed and Table-1-box feasibility check (how the coordinate frame was confirmed) |
| `run_ga.py` | Stage 1 (GA 150 × 60), Stage 2 (NSGA-II 200 × 300 or × 50), selection, Monte Carlo sensitivity; batch runner |
| `analyze.py` | Table 2, Figs. 7–10 → `RESULTS.json`, `fig*.png` |
| `runs/{full,two50,only50}/run_NNN.json` | One file per run: Stage-1 result, final design (all 26 parameters), metrics, timing, sensitivity |

```bash
PY=/playpen/hisham26/deeplens_env/bin/python     # the env with pymoo
$PY -I test_prism.py && $PY -I probe.py
$PY -I run_ga.py two50 only50 full --runs 50 --workers 18     # ~15 min on 18 cores
$PY -I analyze.py
```

## What follows the paper exactly

- Surfaces: Eq. (2) even-asphere base plus `a1 y + a2 z + a3 y² + a4 yz + a5 z²` on A and B; C is a sphere (no conic in the
  16 variables). Local-frame sag along local x, Newton root-finding of Eq. (4), Snell / mirror / TIR-by-failed-refraction.
- 81 rays: 3 × 3 eye-box points (5 × 5 mm) × 3 × 3 field directions; a 25° × 25° field has the stated 35.35° diagonal.
- 16 Stage-1 variables and bounds from Table 1, with the Table 1 seed in the initial population. Status score bands
  1000/3000/5000/10000, wedge penalty weight 10⁴, `F = S + P_wedge`, population 150, 60 generations.
- Stage 2: 26 variables, freeform bounds ±5×10⁻³, p₁ copied 50 times with copies 2–50 scaled by `1 + U[−0.01, 0.01]`,
  objectives mean RMS / worst RMS / |W − 12 mm|, penalties for bad status or nonzero wedge, a leakage penalty below 99.9%
  throughput, a C-to-image minimum separation constraint, population 200, 300 generations.
- Selection: Pareto set → zero wedge penalty and 100% throughput → smallest center-field RMS among designs near the target width.
- Table 2 protocol: Stage-2-only from a random population for 50 generations versus Stage 1 + 50 Stage-2 generations, 50 runs each,
  IQR filtering. Batch study: 50 full runs, IQR filter on center RMS, Monte Carlo most-sensitive-parameter counts.

## What the paper leaves open, and what we chose

| Gap | Choice | Basis |
|---|---|---|
| Coordinate frame (Fig. 1 and Table 1 disagree) | B vertex at the origin; pupil plane at x = 25 mm; rays travel −X; Z up to the display | Fig. 7 layout. Under this frame the Table 1 seed sends all 81 rays to the image with W = 14 mm (target 12); under the Fig. 1 reading it reaches C with 0 rays |
| Tilt sign | Right-hand rotation about +Y: local x = (cos t, 0, −sin t) | Makes B (−23°) fold rays up toward A, C (78°) a dome over the prism, and the display (−135°) face up-left, as drawn in Fig. 7 |
| Sag-axis orientation | A, C, image: as rotated. B: reversed, so R_B < 0 is concave toward the eye | B must be the concave power mirror (R_B = −56 mm gives f ≈ 28 mm, which a 12 mm image at ±12.5° needs) |
| Glass index | n = 1.53 (Zeonex E48R), one wavelength | Not stated |
| Eye-box grid | Centered on the axis, ±2.5 mm | Fig. 7 hints at 0…5 mm; centered is the conventional reading |
| Field grid | (θy, θz) ∈ {−12.5°, 0, +12.5°}² | Diagonal 35.35°; the Fig. 7 spot diagram is symmetric about the center field |
| "Horizontal axis u" of the image plane (Eq. 11) | Global Y (out of the fold plane) | Horizontal for the viewer |
| Wedge regions Δ_AC, Δ_BC | Δ_AC: how far any TIR hit on A lies outside the glass side of C. Δ_BC: how far any C hit lies outside B. Checked on all rays (the paper checks Δ_BC on one pupil row) | Text describes the intent, not the geometry |
| Clear apertures | Only A is clipped (radius `apA`, a variable); B and C are unclipped | Only A has an aperture variable |
| Stage-2 "large penalty" | `1e4 + S + P_wedge` on all three objectives (graded, so a random population can still climb) | Value not given |
| Leakage penalty | +1 mm on f1 and f2 below 99.9% throughput (with 81 rays: any lost ray) | Value not given |
| Minimum C–image separation | 1 mm along the chief ray | Value not given |
| GA operators | pymoo 0.6.1 defaults: GA (tournament, SBX, polynomial mutation, elitist survival); NSGA-II with constraint domination | Paper names MATLAB `ga`/`gamultiobj` but no operators |
| Other 150 Stage-2 individuals | Uniform in bounds (what MATLAB's creation function does when the initial matrix has fewer rows than the population) | Paper gives 50 seeded copies and a population of 200 |
| "Near the target width" in selection | f3 ≤ 1 mm (falls back to the whole feasible front if empty) | Not quantified |
| Monte Carlo tolerancing | 200 samples, every parameter scaled by `1 + U[−0.5%, +0.5%]`; most sensitive = largest abs. Pearson correlation between a parameter's error and the center-RMS change; sensitivity score = mean abs. center-RMS change | Method only named |
| Representative design (Fig. 7) | Pareto run of Fig. 10 nearest the utopia point after min-max scaling | "balanced solution" |

The tracer was not cross-checked in CODE V (the paper's validation, Figs. 4–5). The `codev` skill could export the selected design.

## Results (2026-10-26, 150 runs, `RESULTS.json`)

**Paper's headline claim reproduces: once Stage 1 finds a TIR path, Stage 2 converges to comparable imaging every run.**
All 50 full runs produced a design with 100% throughput and no wedge violation. Our numbers come out *better* than the
paper's, most likely because the unstated choices (n = 1.53, ±12.5° field, pymoo operators) are not theirs.

| Batch (50 full runs, IQR-filtered) | Paper | Ours |
|---|---:|---:|
| Runs kept / removed | 43 / 7 | 49 / 1 |
| Best center-field RMS | 6.0 µm | 6.8 µm |
| Median center RMS (IQR) | 56.3 µm (33.1–69.3) | 27.4 µm (20.0–42.8) |
| Best / median worst-field RMS | 162 / 235 µm | 106 / 166 µm |
| Representative design: center / worst | 40 / 190 µm | 22 / 153 µm (run 15, W = 12.35 mm) |
| Median run time | 32 s (MATLAB, 16-core Ryzen) | 142 s (numpy, 1 core per run, 18 runs in parallel) |

**Table 2 (the two-stage benefit) only half reproduces.**

| Same 50-generation Stage-2 budget | Paper center / worst | Ours center / worst |
|---|---:|---:|
| Stage-2 only (random start) | 180 / 367 µm | 46 / 223 µm |
| Two stages | 113 / 315 µm | 35 / 246 µm |

Two stages wins on the center field in both, but ours is not significant (raw medians 52 vs 37 µm, Mann-Whitney
p = 0.08) and the worst field does not improve (p = 0.37). Our Stage-2-only baseline is far stronger than the
paper's, probably because our infeasibility penalty is graded (`1e4 + S + P_wedge`), so a random population still
climbs toward feasibility. A constant penalty, which is one reading of "set to a large penalty value", would likely
reopen the paper's gap. That is the first thing to test.

**Fig. 9 (sensitivity ranking) does not reproduce.** The paper's top drivers are the freeform z² term of A and the
y²/z² terms and tilt of B. Ours are B's radius (15/49), image z (12) and image tilt (11). Under a uniform ±0.5%
relative perturbation, a 0.5% change in R_B or in a ~50 mm image coordinate is a large physical error. The paper
does not give its perturbation scheme, so this ranking measures the tolerance model, not the design.

Caveat: the paper's model, and so ours, never builds a closed solid. A and C need not meet (visible in Fig. 7);
the wedge penalty is the only manufacturability check.
