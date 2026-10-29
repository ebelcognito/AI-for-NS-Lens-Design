# Diverse clock-position and patent-layout design stock

## Outcome

Two frozen development cohorts evaluated 115,200 physical candidates using 4,492,800
search source rays.  No blocker was used.  The first cohort surveyed six layouts with
the original 9-ray-per-field search grid.  Because its selected designs all failed a
909-ray stress check, a second cohort searched the four productive layouts with 25 rays
per field.

The dense-search cohort contains **40 stored audit-passing designs**, spanning four
environments and eight environment/route pairs.  Stress testing every design with
101 rays at each of nine field angles leaves **13 verified designs across six
environment/route pairs**.

| Dense-search environment | Audit-passing designs | Distinct audited routes | 909-ray passes | Robust routes |
|---|---:|---:|---:|---:|
| bottom_mid | 9 | 2 | 8 | 2 |
| bottom_right | 1 | 1 | 1 | 1 |
| patent_107 | 27 | 4 | 3 | 2 |
| patent_923 | 3 | 1 | 1 | 1 |
| **Total** | **40** | **8 environment/route pairs** | **13** | **6 environment/route pairs** |

Best 909-ray representatives:

| Environment/family | Worst-field folded throughput | Root support |
|---|---:|---:|
| bottom_mid, 4 corners | 73.1% | 82.2% |
| bottom_mid, 3 corners | 72.8% | 83.2% |
| bottom_right, 4 corners | 64.8% | 72.3% |
| patent_107, 5 corners | 71.0% | 79.2% |
| patent_923, 5 corners | 34.9% | 61.4% |

All 40 stored dense audits replayed exactly, all 48 dense-search budget logs matched,
and all 40 designs were stress-tested.  Stress verification uses the same physical
engine at higher sampling density, not an independent-engine certification.

## Cohort 1: broad clock-position sweep

`runs/diverse_clock_patent_20260911_v1` used fresh seeds 312–315, 3/4/5 corners,
the portfolio proposer, and 1,200 physical calls per job over six environments:

- right-side detector at the existing low position;
- right-side detector at the reference midpoint;
- bottom horizontal detector at the horizontal midpoint;
- bottom horizontal detector toward the right;
- the constrained US923-inspired layout;
- the constrained US107-inspired layout.

This is 72 jobs, 86,400 traced candidates, and 2,332,800 search rays.  It produced ten
audit passes: two bottom-mid, two bottom-right, four patent-923, and two patent-107.
The selected best candidate in each of the four productive families failed the 909-ray
stress gate.  Neither right-side layout produced a coarse-feasible candidate.

## Cohort 2: dense stock search

`runs/diverse_stock_dense_search_20260911_v1` used fresh seeds 316–319 and the four
productive layouts.  Each of 48 jobs used 600 physical calls and 25 rays per field:
28,800 candidates and 2,160,000 search rays.  Search therefore saw the same pupil
sampling density as the reserved audit, while the audit added intermediate field angles.

The large improvement from zero robust representatives in the sparse sweep to 13/40
stress-verified designs in the dense cohort shows that pupil sampling density materially
affects retention.  It is not a randomized method comparison: seeds and budgets changed
along with sampling density.

## Stock artifacts

- [Central design manifest](../../runs/diverse_stock_dense_search_20260911_v1/stock_manifest.json)
- [Stress-verified gallery](../../runs/diverse_stock_dense_search_20260911_v1/stress_verified_stock.png)
- [Dense frozen protocol](../../runs/diverse_stock_dense_search_20260911_v1/protocol.json)
- [All-design 909-ray results](../../runs/diverse_stock_dense_search_20260911_v1/stock_stress_all.json)
- [Exact audit replay and representative stress checks](../../runs/diverse_stock_dense_search_20260911_v1/replay_and_stress.json)
- [Broad-sweep frozen protocol](../../runs/diverse_clock_patent_20260911_v1/protocol.json)
- [Broad-sweep replay and stress checks](../../runs/diverse_clock_patent_20260911_v1/replay_and_stress.json)

The central manifest includes genes, codec, complete brief, source archive, route,
audit quality/support, clearances, and 909-ray quality/support for every design.

## Reproduce dense cohort

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.benchmark \
  --out runs/diverse_stock_dense_search_NEW --stage development \
  --candidate portfolio --arms portfolio --seeds 316 317 318 319 \
  --corners 3 4 5 --briefs bottom_mid bottom_right patent_923 patent_107 \
  --search-calls 600 --search-rays-per-field 25 --workers 10
```
