# Spherical inception and compaction run — 2026-09-04

## TL;DR

- `ns_probe/run_novel_compaction.py` now implements a reproducible spherical-only
  geometry-first inception → sequence-preserving compaction pipeline.
- The novelty reference is frozen before generation: 2,145 stored signature strings,
  reduced to 1,750 surface-name-invariant canonical topologies. Every contributing
  file and SHA-256 is recorded in the run manifest.
- The conservative novelty runs (`v6` and `v8`, at least 3 edits and 35% normalized
  distance) evaluated 50,000 proposals and promoted **zero** designs. Do not claim a
  strongly dissimilar discovery from those runs.
- A lower-bound run (`v7`, at least 2 edits and 25%) produced a hardened viable,
  two-element route and demonstrates compaction. An equally near known route is its
  five-event prefix, so it is explicitly classified as a boundary demonstrator, not a
  strongly novel result.
- Compaction preserved the seven-event route, retained 99.14% of inception fitness,
  reduced thinness by 4.26% and optic-envelope width by 11.12%, and improved 0.1%-range
  perturbation persistence from 16% to 88%.

## Entry points

- Generator and compactor: `ns_probe/run_novel_compaction.py`
- Surface-name-invariant novelty: `nslens/qd/novelty.py`
- Two separated spherical-body problem: `nslens/qd/multi_element.py`
- Boundary demonstration: `runs/novel_spherical_compaction_20260904_v7/manifest.json`
- Conservative negative result: `runs/novel_spherical_compaction_20260904_v8/manifest.json`

## What was inventoried first

The runner reads every pre-existing `runs/**/archive.json` plus
`catalogue/catalogue.json`, extracts all available interaction-sequence fields, and
canonicalizes physical surfaces by first use. The frozen `reference_snapshot.json`
contains counts, source paths, hashes, fields, and the spherical parameterization.
Output directories are excluded and cannot be overwritten.

The most relevant work from the preceding day was also used directly:

- the interaction-fingerprint/canonicalization work;
- the dense fixed-baseline audit and its 156 intentional spherical designs;
- fixed-environment specifications;
- learned/random/CMA fixed-call comparisons; and
- the prospective Inception → Compaction split in
  `agent_context/ns_systems/03_future_directions_interaction_sequence_discovery.md`.

The audited designs are lifted into the two-element parameterization as anchors. Donor
shapes from other audited folds are cross-combined into E2 so generation is not limited
to an unpowered slab.

## Boundary compaction demonstration (`v7`)

Observed route:

```text
T@E1A -> TIR@E1T -> TIR@E1B -> TIR@E1A -> T@E1B -> T@E2A -> T@E2B
```

This route is absent from the frozen archive, and its canonical distance to the nearest
prior topology is 2/7 = 28.57%. The last two interactions are an extension through the
second powered spherical body. An equally near old topology is the five-event prefix,
so `qualification.json` prevents this from being described as strongly novel.

Dense independent validation uses 61 rays and five fields (-2, -1, 0, 1, 2 degrees).
Both inception and compacted prescriptions retain 100% intended-route root support,
the target remains dominant at every field, every field passes `clean_pack` and exit
convergence, and worst TIR margin is 0.561 degrees before / 1.389 degrees after.

| Metric | Inception | Compacted | Change |
|---|---:|---:|---:|
| hardened fitness | 0.76883 | 0.76220 | -0.86% |
| thinness T | 105.5607 | 101.0618 | -4.26% |
| optic envelope width | 50.6126 | 44.9846 | -11.12% |
| optic bounding-box area | 6073.51 | 5398.15 | -11.12% |

At Gaussian geometry perturbations expressed as a fraction of each gene's global
range, compacted persistence is 100% at 0.05%, 88% at 0.1%, and 68% at 0.2% (25 trials
per scale). The inception seed scores 24%, 16%, and 12%, respectively. Compaction thus
creates a materially wider basin, but does not meet an 80% persistence bar at 0.2%.

## Conservative novelty result (`v8`)

The `v6` run used -2/0/+2 degree fields and the `v8` follow-up reduced the field set to
zero degrees to make the optical gate easier. Both required at least 3 canonical edits
and 35% normalized distance. Together they promoted zero of 50,000 proposals. This
negative result is the current honest boundary: the generator is operational, but it
has not yet produced a strongly dissimilar hardened path from the present two-body
proposal family.

## Reproduce

```bash
python3 ns_probe/run_novel_compaction.py \
  --out runs/novel_spherical_compaction_NEW \
  --design-space two_element --fields=-2,0,2 \
  --min-edits 3 --min-distance 0.35 \
  --evals 30000 --workers 12 --batch-size 384 \
  --stop-after-new 3 --compaction-iterations 10 \
  --compaction-batch 160 --fitness-retention 0.80
```

Do not lower the novelty thresholds and then present a prefix extension as a strongly
new topology. Lower thresholds are useful for debugging the compaction module only.
