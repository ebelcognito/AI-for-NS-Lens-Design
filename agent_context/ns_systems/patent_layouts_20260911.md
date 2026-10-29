# Patent-inspired receiver layouts — development runs

Two matched development datasets test layouts inspired by US9239453B2 and
US10739578B2.  These are scale/layout analogues, not reproductions of either patent's
freeform prescription.  No blocker is present.  Both retain the unchanged acceptance
requirement of at least two TIR events.

## Frozen scope

Each layout used the retained-candidate (`portfolio`) proposer, fresh seeds 306–311,
3/4/5-corner bodies, 400 physical search calls per corner/seed, population 10, and up
to eight reserved 125-ray audits.  Each layout therefore contains 18 jobs, 7,200 traced
candidates, and 194,400 search source rays.  Seeds are paired across layouts.

Common material and geometry settings: n=1.60; body centre x=51–59 mm and y=-4–4 mm;
radial factor 0.75–1.0; angular jitter ±15% of nominal sector width; maximum absolute
sag/half-chord 0.268; minimum radius 8 mm; minimum edge 4 mm.

| Brief | Source | Body half-width / half-height | Detector | Actual curved-boundary clearance |
|---|---|---|---|---|
| patent_923 | (55,18), downward; 4 mm beam | 6–10 / 10–16 mm | vertical x=78, centre y=0, full length 10 mm | source 3–10 mm; detector 16–22 mm |
| patent_107 | (73,0), leftward; 4.8 mm beam | 10–16 / 6–10 mm | horizontal y=-32, centre x=55, full length 12 mm | source 3–10 mm; detector 20–28 mm |

The second codec is the first one's in-plane anisotropy rotated by 90 degrees.  This
models the patents as two briefs in one oriented-detector implementation rather than as
unrelated optical engines.

## Results

| Layout | Successful jobs | Passing stored audits | Audited routes |
|---|---:|---:|---:|
| US923-inspired | 1/18 (5.6%) | 2 | 1 |
| US107-inspired | 0/18 | 0 | 0 |

The US923 success is five-corner seed 311.  Its best stored audit has 65.0% worst-field
folded throughput, 76% root support, 6.40 mm source clearance, and 18.78 mm detector
clearance.  Its dominant route is entry refraction, three TIR events on distinct faces,
and exit refraction through the entry face.  All three stored audits replay exactly.

The best US923 audited design did not pass the post-run 909-ray stress check: throughput
was 28.4%, but root support fell to 34.7%, below the fixed 60% requirement.  It therefore
remains a development discovery rather than a robust design.

No US107 candidate was coarse-feasible and therefore none entered the reserved audit.
As a post-run diagnostic, the best triangle passes a 125-ray re-evaluation (50.9%
throughput, 64% support), but fails the 909-ray check (32.9% throughput, 41.6% support).
This diagnostic is not counted as a run success and shows sensitivity to pupil sampling.

Zero worst-field qualifying folded throughput occurred for 1,766/7,200 (24.5%) measured
US923 candidates and 2,132/7,200 (29.6%) US107 candidates.  At least one-fold progress
was present in 85.5% and 82.9%, respectively.  The exact clearance bounds rejected
1,737 and 1,513 additional geometry proposals before tracing.  All 36 budget logs match.

These are proposer-only development datasets and do not compare search methods or
support statistical claims between the two patent-inspired environments.

## Artifacts

- [US923 protocol](../../runs/patent_923_layout_20260911_v2/protocol.json)
- [US923 selected examples](../../runs/patent_923_layout_20260911_v2/discovery_examples.png)
- [US107 protocol](../../runs/patent_107_layout_20260911_v2/protocol.json)
- [US107 selected examples](../../runs/patent_107_layout_20260911_v2/discovery_examples.png)

The figures label stored audit passes separately from post-run dense passes.
The earlier `v1` directories are exploratory dry runs without enforced maximum
source/receiver clearance and are superseded by these `v2` datasets.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.benchmark \
  --out runs/patent_923_layout_NEW --stage development --arms portfolio \
  --seeds 306 307 308 309 310 311 --corners 3 4 5 \
  --briefs patent_923 --search-calls 400 --workers 6

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.benchmark \
  --out runs/patent_107_layout_NEW --stage development --arms portfolio \
  --seeds 306 307 308 309 310 311 --corners 3 4 5 \
  --briefs patent_107 --search-calls 400 --workers 6
```

Patent geometry sources:

- https://patents.google.com/patent/US9239453B2/en
- https://patents.google.com/patent/US10739578B2/en
