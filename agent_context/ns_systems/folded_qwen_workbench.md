# Folded freeform workbench: measured status, 2026-09-10

The earlier unfolded benchmark tested tool use and a training pipeline. It was too
narrow for the intended folded freeform engineering objective. Further unfolded
adapter evaluations are parked. This experiment brings actual folded geometry into
the workbench, but does **not** yet demonstrate an improving learned engineer.

## What the folded experiment actually does

Frozen Qwen3-8B (revision `b968826d9c46dd6066d109eabc6255188de91218`)
controls inspection, coefficient selection, bounded perturbations, numerical
coordinate optimization, verification, and submission. The numerical optimizer does
the local shape search; Qwen chooses how to use it. No folded training has occurred.

The starting geometry comes from the fourth highest-fitness cell in
`runs/ns_autodiff__results_search_prism_freeform/archive_prism_freeform.json`.
The original face representation was converted to endpoint-preserving radial
freeforms and a closed, periodic C2 spline with 128 anchors. This changes the archived
geometry: the converted design was re-traced and certified anew rather than accepting
an old archive score. The original cell is retained in
`runs/folded_workbench_probe_20260910_v1/seed.json`.

Eight adjustable coefficients cover second- and fourth-order departures on four
faces, with ±2 mm bounds. Base geometry and refractive index remain fixed. Three
display source points feed a physical prism; useful rays refract into its top face,
undergo total internal reflection on face A, then refract through face B toward the
eyebox. This is a **2-D, one-primary-fold freeform prism**. Faint multi-bounce ghost
branches do not establish useful multi-fold architectures.

The two tasks corrupt the same seed's A second-order coefficient by ±0.75 mm.
Both fail the initial optical audit. These are two repair cases from **one ancestor**,
not independent design families or a held-out engineering benchmark. The witness is
excluded from the public task schema.

## Fixed verifier and budgets

- Every field must deliver at least 30% launched energy through the pupil along paths
  with at least one actual TIR event; straight-through transmission cannot pass.
- Per-field angular standard deviation must be ≤1°, with ordered field centroids and
  at least 12° output centroid FOV. These are prototype gates, not production specs.
- Eyebox plane is x=120 mm, with y within ±9 mm. Sources are (49,50), (55,50), (61,50)
  mm, aimed downward with a ±12° launch fan.
- Geometry validity, energy closure ≤1e-9, and unresolved energy ≤1e-4 are mandatory.
- Each episode allows 48 candidate traces, 16 actions, and two optional dense checks.
  Coordinate-search probes are individually charged. Invalid requests are recorded.
- Candidate evaluation uses 51 rays. Optional verification uses a shifted 99-ray fan;
  final audit uses a separately shifted 183-ray fan. Both outputs were additionally
  checked on 273 rays. All use the same physical tracer: this is sampling validation,
  not independent-simulator validation or proof of convergence.

## Results

| Method | Submitted and passed | Optical audit passed | Comment |
| --- | ---: | ---: | --- |
| Fixed numerical coordinate plan | 2/2 | 2/2 | 48 candidates each |
| Frozen Qwen, 64k cumulative language budget | 1/2 | 2/2 | Case 0 ran out of tokens before submission |
| Frozen Qwen, 128k cumulative language budget | 2/2 | 2/2 | Same prescriptions as the 64k run |

Increasing the language budget enabled submission; it did not improve the optics or
train the model. The first Qwen case made two oversized optimizer requests and then
an extra verification request beyond its quota. These errors remain in the logs.
The fixed numerical plan succeeds too, so this experiment establishes usable folded
tools, not a Qwen advantage.

| 128k Qwen result | Case 0 | Case 1 |
| --- | ---: | ---: |
| Candidate evaluations | 37 | 48 |
| Optional dense checks | 2 | 1 |
| Invalid tool requests | 3 | 0 |
| Cumulative input + output tokens | 79,442 | 19,330 |
| Worst-field folded throughput, 183-ray audit | 40.80% | 45.34% |
| Worst-field angular standard deviation | 0.9952° | 0.9708° |
| Output centroid FOV | 16.66° | 15.15° |
| 273-ray sampling check | Pass | Pass |

Case 0 is close to the 1° threshold and should not be treated as a robust engineering
solution. All numerical and Qwen episodes replay exactly, including state hashes,
individual evaluations, and final outcomes. The 273-ray checks were first performed
on the 64k prescriptions; exact coefficient equality with the 128k prescriptions is
recorded in the latter run's `convergence.json`.

## Saved evidence and reproduction

- Final run: `runs/folded_workbench_qwen_128k_20260910_v1/`.
- Actual ray-layout PNGs: `folded_solution_0.png`, `folded_solution_1.png` in that run.
  Each is re-traced and checked against the stored final audit before rendering.
  Red diamonds mark actual TIR events; plotted rays are a subset, while metrics include
  Fresnel branches. Companion JSON files record the source and renderer hashes.
- Preserved 64k failure: `runs/folded_workbench_qwen_20260910_v1/`.
- Numerical control: `runs/folded_workbench_scripted_20260910_v1/`.
- Inputs and initial checks: `runs/folded_workbench_probe_20260910_v1/`.
- Implementation: `nslens/workbench/folded_domain.py`, `folded_tools.py`,
  `folded_baseline.py`, `render_folded.py`.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.workbench.folded_baseline \
  runs/folded_workbench_probe_20260910_v1/tasks.json \
  --scripted --out runs/folded_numerical_new

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nslens.workbench.render_folded \
  runs/folded_workbench_qwen_128k_20260910_v1/episode001.json \
  --out runs/folded_solution.png

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s tests -p '*workbench*.py'
```

For Qwen use the local runtime environment documented in `nslens/workbench/README.md`,
replace `--scripted` with `--weights <pinned-model-directory> --max-tokens 128000`,
and use a fresh output directory.

## What is missing for the intended improving loop

1. Multiple independently certified folded ancestors, including genuinely useful
   two-or-more-fold paths and reflective as well as TIR designs. Split descendants by
   ancestor/topology before any teacher selection or training.
2. Tasks requiring consequential engineering choices: fixed versus editable faces,
   envelope and clearance, pupil placement, multiple fields, material choices, and
   competing throughput/aberration constraints. Include failures and infeasible tasks;
   avoid a corpus universally solved by one fixed optimizer invocation.
3. Budget-matched frozen-Qwen, numerical, and trained-Qwen comparisons on unseen
   families. Train on verified trajectories and diagnoses, retain every failed probe,
   and promote a checkpoint only for measured optical gains without hidden extra
   simulation budget. Current evidence contains no such folded learning gain.
4. Manufacturing/tolerance checks, denser angular and pupil sampling, 3-D freeform
   surfaces, and an independent optical-engine cross-check before making realistic
   optical-engineering claims.

The next experiment should expand folded task diversity before another fine-tuning
run. More unfolded smoke tests would not address these gaps.

Follow-up: [constrained folded decision benchmark](folded_decision_benchmark.md).
The pilot is complete; its stronger fixed numerical control solves all four tasks.
