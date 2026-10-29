# 19 — The topology axis: the archive was deleting what it exists to keep

**Date:** 2026-08-13. Follows [18_rl_engine_reward_plan.md](18_rl_engine_reward_plan.md); supplies
the catalogue that the signature-conditioned inverse model in that doc trains against.

---

## 1. The finding

Reconstructing archive cells from all 58,980 scored trace rows:

| | |
|---|---:|
| distinct dominant signatures found | **804** |
| signatures that ever win an archive cell | **29** |
| distinct signatures competing per occupied cell (mean) | **41.7** (median 28, max 116) |
| occupied cells holding more than one topology | **94%** |
| catalogued topologies that never win a cell anywhere | **133 of 171** |

The cause is structural, not a bug. The first behaviour axis is `tir_bounces` — fold *count* —
which is a 9-bin projection of an 804-value quantity, and in the fold24 regime
[experiments.py](../../nslens/qd/experiments.py) sets `bc_bins=(3, 16)`: **three attainable
values**. Two designs that route light through completely different sequences of faces land in
the same cell, and the one scoring marginally lower is deleted.

`A->B->A` and `A->T->A` are both "2 folds". They are not the same optic.

## 2. Choosing the replacement

Measured over every scored row — cells, collapse rate, and how many good topologies each
encoding still discards:

| encoding | cells | sigs/cell | designs/cell | topologies >0.6 still lost |
|---|---:|---:|---:|---:|
| `tir_count` (current) | **4** | 201.0 | 14,745 | **13** |
| `tir_count` × `fold_locus` | 12 | 67.0 | 4,915 | 11 |
| **TIR-face multiset** | **79** | **10.2** | **747** | **7** |
| TIR-face sequence | 254 | 3.2 | 232 | 2 |
| face sequence (all events) | 478 | 1.7 | 123 | 1 |
| full signature | 804 | 1.0 | 73 | 0 |

The temptation is to take the finest encoding. That is wrong, because **the archive is doing two
incompatible jobs**:

- *drive the search* — CMA-ME's "improved" signal only exists if designs contend for a cell.
  Wants coarse. At 804 cells, 69% hold fewer than 5 designs and there is no selection pressure
  at all.
- *preserve the catalogue* — never delete a working novel ray path. Wants fine and unbounded.

**So we split them.** The catalogue (§3) is keyed on the full signature and discards nothing.
The archive (§4) takes the **TIR-face multiset** — which faces the light bounces off and how
many times each, order dropped — at 79 observed classes with 747 designs per cell.

Two facts settled by the data rather than by taste:

- **An unbounded catalogue is safe.** The 342 singleton signatures top out at fitness **0.248**,
  with none above 0.3. They are search noise, not rare gems, so a `seen >= 10` filter cleans the
  catalogue without special handling.
- **Do not canonicalise the T/D mirror.** Merging mirror-image paths cuts 804 → 681 cells, but 4
  topologies above 0.5 are then discarded as "worse mirror twins", which should not happen if the
  mirror were a true symmetry. Probably a search artifact (both twins found, one in a worse local
  optimum) rather than a symmetry violation — but unverified, so left alone. Five-minute test:
  mirror a good genotype (`ray_y→−ray_y`, `ang→−ang`, swap `tsag`/`dsag`, negate tilts), retrace,
  check the fitness is identical. Note that `tsag`/`dsag` are crossed relative to the docstring.

## 3. The catalogue

[ns_probe/build_catalogue.py](../../ns_probe/build_catalogue.py) → `catalogue/catalogue.json`,
viewable at `catalogue/catalogue.html`.

Best genotype per full signature across all 45 traces, then **re-scored under one hardened
config** — worst-of-nine-fields, `max_sag_ratio=0.268`, `det_clearance=12`, `clean_pack` required
at every field. This step is the point: the trace scores came from runs with different field sets
and different gates, so they were never comparable to each other, and several of the highest were
the known exploits.

The gate ladder, on the 38 candidates above trace-fitness 0.2:

| level | survive |
|---|---:|
| raw trace | 36 / 38 |
| + geometry gates (sag ratio, detector clearance) | **23** |
| + `clean_pack` at the central field | **13** |
| + every field (`gate_all_fields`) | **6** |

**13 of 38 were genuine geometric exploits** — bulb faces, or glass intersecting the detector
plane. Scoring at each design's native field set instead of the canonical ±8° changes nothing, so
this is not a yardstick artifact.

Two bugs found and fixed while building it, both worth remembering:

1. **Genome interpretation must follow the run that produced the genotype.** `fold24_clean` and
   `fold24_clean_v3` set `relative_sag=True`, so their `fsag`/`bsag` genes are *fractions of the
   face's half-chord*, not millimetres. Re-scoring them under the absolute interpretation
   silently flattens the element (a 0.21 gene becomes a 0.21 mm sag instead of 21% of the chord).
   17 of 171 entries were affected, including two of the apparent survivors. `RELATIVE_SAG_RUNS`
   in the builder now selects the genome per entry.
2. **`thinness` is reported as `thinness_T`.** The driver excludes `thinness` from the info block
   by name because it is already written under `thinness_T`; reading `info["thinness"]` returns
   None silently.

After both fixes: **13 of 171 pass the hardened config, 6 keep their exact topology under it**,
and four structurally distinct routes clear 0.74 worst-field:

| hardened | folds | topology | spot |
|---:|---:|---|---:|
| **0.860** | 3 | `T@A -> TIR@T -> TIR@B -> TIR@A -> T@B` | 1.08 mm |
| **0.826** | 3 | `T@A -> TIR@D -> TIR@B -> TIR@A -> T@B` | 2.55 mm |
| **0.814** | 3 | `T@A -> TIR@B -> TIR@T -> TIR@A -> T@B` | 0.44 mm |
| **0.742** | 4 | `T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B` | 5.72 mm |

All four are flat across the nine fields. 0.860 matches the best fitness in the entire program.

## 4. The axis, and the experiment

`tir_faces` in [descriptors.py](../../nslens/qd/descriptors.py): a fixed enumeration of every
multiset of TIR faces up to depth 5 → **126 cells**. Verified against all 58,980 scored rows:
**0 out-of-vocabulary, 79 cells occupied**. Depth 5 covers 100% of evaluations; depth 4 would drop
1.8%. `TIR_FACES` order is load-bearing — it defines the cell indices written into every archive.

Out-of-vocabulary returns an index past the top bound so `reject_out_of_bounds` **discards** it
rather than clamping it into the last cell. Clamping is the failure this axis exists to fix.

**`fold24_topo`** is `fold24_clean` with one change: `descriptors=("tir_faces", "thinness")`,
`bc_bins=(126, 4)`. Genome box, fitness, field set, every gate, `param_scaling` and eval budget
are copied verbatim, so the pair isolates the axis. Thinness drops to 4 bins because the grid is a
product — 126 × 16 = 2016 cells against the ~900 designs that survive the gate stack in 6k evals
would be a lookup table.

Warm start ([ns_probe/make_topo_seed.py](../../ns_probe/make_topo_seed.py)): the topology axis
cannot cold-start any more than `fold24_clean` could — the gate stack keeps ~15% of proposals and
CMA-ME emitters need elites to sample around. `resume_from` reads each cell's coordinates *by
name*, and no archive on disk carries a `tir_faces` key, so the seed is rebuilt from the existing
`fold24_clean_seed` plus the catalogue's relative-sag hardened survivors: **6 cells, 5 distinct
topologies, best 0.860**.

### What the comparison measures

Equal budget, equal seeds, both arms. Report:

1. **distinct interaction signatures retained in the final archive** — the headline. The baseline
   can retain at most 3 fold-count classes; this is the quantity the axis is supposed to move.
2. **best fitness** — did buying diversity cost quality?
3. **coverage and designs-per-cell** — did the finer axis destroy selection pressure? This is the
   real risk of the change, and the reason the multiset was chosen over the sequence.

### Prediction, recorded before the run

The topology arm retains substantially more distinct signatures (baseline is capped near 3
classes), and best fitness is **roughly unchanged or slightly worse** — the seed already contains
0.860 and 6k evals under a 15% gate is not much budget to beat it. If best fitness collapses, the
finer axis has starved the emitters and the multiset is still too fine; the fallback is
`tir_count × fold_locus` at 12 cells.

## 5. Results

`tests/run_all.py`: **9/9 PASS** with the new descriptor, including QD-equivalence across 13
cases. 3 seeds per arm, 6,000 evals each.

### 5.1 The first run was rigged, and every headline metric looked like a win

With `bc_bounds=((0.0, 126.0), ...)` the topology arm reported 13 vs 7 signatures, best **0.960**
vs 0.843, and 27 vs 3 elites above 0.7. All of it was an artifact. The top cells were:

```
0.960   TIR@T -> T@B      tir=1     one bounce
0.922   T@A -> T@B        tir=0     NO FOLD AT ALL
```

`fold24_clean` confines itself to 2–5 folds through `bc_bounds=(2.0, 5.0)` on `tir_count` plus
`reject_out_of_bounds`. The topology axis admitted the `unfolded` and single-bounce multisets
(vocabulary indices 0–4), so the arm filled its archive with straight transmission — which scores
0.92 precisely because it pays no fold losses — against a baseline structurally forbidden from
keeping those designs. Fixed to `bc_bins=(121, 4)` over `[5, 126]`: one vocabulary entry per bin,
k<2 rejected.

**The lesson is not "check bounds".** It is that the arm which beat expectations was the arm with
the bug, and the summary table could not show it — only reading the winning *signatures* could.
Any future arm of this comparison must print its top cells' signatures before its metrics.

### 5.2 The corrected result

| | baseline (`tir_bounces`) | topology (`tir_faces`) |
|---|---:|---:|
| grid | 3 × 16 | 121 × 4 |
| elites kept | 28 | 29 |
| **signatures per seed** | 5, 6, 4 → **mean 5.0** | 9, 7, 6 → **mean 7.3** |
| **face multisets per seed** | 4, 5, 3 → mean 4.0 | 9, 7, 6 → mean 7.3 |
| signatures, union over seeds | 7 | 9 |
| best fitness | 0.843 | 0.860 |
| **best SEARCH-FOUND fitness** | **0.843** | **0.838** |
| median fitness | 0.421 | 0.287 |
| cells / seed | 9.3 | 9.7 |

**Retention: confirmed, ~1.5×.** 7.3 vs 5.0 distinct ray paths per run, and the direction holds in
every seed (9>5, 7>6, 6>4). In the topology arm signatures and multisets are equal in every seed —
one signature per occupied cell, exactly as the axis intends — whereas the baseline keeps two
signatures per multiset only by accident of the thinness axis.

**Fitness: no gain.** The +0.017 in the raw table is entirely the seed. The topology arm's 0.860
*is* its seed genotype, unimproved; its best search-found design is **0.838, below the baseline's
0.843**. Both arms found ~47% of their elites by search, so search was working — it simply did not
beat what it was handed. The prediction recorded in §4 was right on fitness and overstated on
retention ("substantially" → 1.5×).

**Selection pressure: intact.** 9.7 occupied cells per seed on a 484-cell grid, so the multiset is
not too fine at this budget — no lookup-table degeneration, and the `tir_count × fold_locus`
fallback is not needed.

Three routes the topology arm kept that the baseline never did — all involving a repeated bounce
on one face, which is precisely what a fold-count axis cannot distinguish:

```
T@A -> TIR@D -> TIR@A -> TIR@A -> T@B
T@A -> TIR@D -> TIR@A -> TIR@A -> TIR@T -> T@B
T@A -> TIR@T -> TIR@B -> TIR@B -> TIR@D -> T@B
```

### 5.3 What is not established

- **The seed confound is real and I introduced it.** The arms warm-start from different archives
  (baseline best 0.838 / median 0.384; topology best 0.860 / median 0.742), because the topology
  seed was built from the catalogue's hardened survivors. Retention is fairly robust to this — it
  measures what the archive *keeps* — but no fitness comparison here should be quoted. Fix by
  giving both arms the same seed restricted to what each axis can represent.
- **6,000 evals is too small.** Under a ~15% gate that is ~900 valid designs across a 484-cell
  grid, and neither arm moved meaningfully off its seed. This run establishes direction, not
  magnitude.
- **One environment, one genome.** `FoldProblem` with relative sag only.

### 5.4 The clean run: common seed, 40k evals

Both confounds fixed. Both arms `resume_from` `runs/fold24_common_seed` — 7 cells, 7 distinct
signatures, best 0.860, carrying BOTH `tir_count` and `tir_faces` on every cell so each arm reads
the coordinate its own `cell_bc_keys` names. Deduped by full signature, not by either arm's grid.
3 seeds × 40,000 evals per arm ([ns_probe/run_topo_ab_40k.sh](../../ns_probe/run_topo_ab_40k.sh)).

| | baseline (`tir_bounces`) | topology (`tir_faces`) |
|---|---:|---:|
| elites kept | **53** | 42 |
| cells / seed | 17.7 | 14.0 |
| **distinct signatures per seed** | 6, 5, 4 → **5.0** | 13, 11, 8 → **10.7** |
| distinct signatures, union | 7 | **17** |
| distinct face multisets | 5 | 15 |
| **best fitness** | **0.878** | **0.878** |
| best SEARCH-FOUND | 0.878 | 0.878 |
| median fitness | 0.158 | 0.164 |

**Fitness is identical.** Both arms reach exactly 0.878, both search-found, both improving on the
shared 0.860 seed. The topology axis costs nothing in peak quality — which is the result that
matters, because the entire risk of a finer axis was that it starves the emitters.

**Retention is 2.1× per seed and 2.4× on the union**, consistent in every seed (13>6, 11>5, 8>4).

The mechanism is visible in how the elites distribute. The baseline keeps MORE elites (53 vs 42)
across FEWER ray paths, because its cells are one topology at many thicknesses:

```
baseline, 53 elites over 7 paths        topology, 42 elites over 17 paths
  18x  T@A -> TIR@T -> TIR@B -> TIR@A     6x  T@A -> TIR@T -> TIR@B -> TIR@A
  14x  T@A -> TIR@T -> TIR@B -> PR@A      6x  T@A -> TIR@D -> TIR@B -> TIR@A
   9x  ...                                4x  ...        (max 6, flat tail)
```

Two ray paths account for 60% of the baseline's archive. The baseline's top SIX cells are the
same signature at six thicknesses; the topology arm's top six span three distinct routes at equal
fitness. That is the collapse, measured directly.

Routes the topology arm kept that the baseline never did — repeated bounces on a single face,
which a fold-count axis cannot distinguish from any other path of the same depth:

```
T@A -> TIR@T -> TIR@A -> TIR@A -> TIR@A -> TIR@A -> T@B   four consecutive bounces on A
T@A -> TIR@T -> TIR@A -> TIR@A -> TIR@B -> TIR@A -> T@B
T@A -> TIR@T -> TIR@B -> TIR@A -> TIR@B -> TIR@B -> T@B
```

Selection pressure held: 14.0 occupied cells per seed on a 484-cell grid, and 35 of 42 elites are
search-found (83%), so the emitters were not starved.

**Verdict: adopt the topology axis.** Same peak fitness, 2.1× the distinct ray paths per run, no
loss of selection pressure. The remaining scope limits are real — one environment (`FoldProblem`,
relative sag), one fitness, and the seed itself came from prior `tir_bounces` runs, so the
comparison shows what each axis *retains* from a shared start rather than what each would find
from scratch.

### 5.5 Next

1. Rerun with a **common seed** and 40k evals, which is the budget every archive on disk was
   built at. Only then is the fitness column worth reading.
2. If retention holds, the catalogue and the axis together are the artifact doc 18 §3 needs: a
   topology-indexed archive is the natural training target for the signature-conditioned inverse
   model, because its cells *are* the conditioning labels.
