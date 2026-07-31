# Non-Sequential Extension — Full Context for Resumption

**Status**: paused. Planning complete; implementation not started.
**Last worked**: 2026-04-21
**Resuming from**: Stage 1, first action below.

---

## 0. Pickup point — read this first when resuming

When you come back, the next concrete action is:

> Create `optical_rl/qd/lensmega/problem/ns/` with three files: `surface.py`, `ray.py`, `propagator.py`. Port DeepLens's sag/normal/intersection math into `surface.py` for a single spherical refracting surface. Write a `RayBatch` tensor-struct in `ray.py`. Write a forward-only `trace()` in `propagator.py` that handles refract and reflect only (no partial mirrors, no autograd pressure). **First runnable target**: trace a known Mangin mirror (refracting front, reflecting rear) and recover its textbook focal length and on-axis spot size.

Everything below is the reasoning behind that decision and the downstream plan.

---

## 1. Research thesis — the emergence narrative

LensMEGA's QD framework is motivated by a specific research question: **can we discover structurally novel optical designs through archive-illumination, without prescribing the mechanism?**

The lineage: a prior RL project on a 4-ray + deformable-optical-element refocusing task showed **spontaneous emergence of total internal reflection (TIR)** as a learned strategy — no reward term mentioned TIR. A Google lead optical engineer validated this as a compelling result. The emergence framing resonates with expert optical designers because classical lens design requires human-chosen starting topologies; an optimizer that *discovers* topology on its own is qualitatively different.

The NS extension scales this same pattern:
- Richer search space (full optical geometry, not a single deformable element)
- Richer topology (non-sequential — rays can fold, retrace, branch via reflectivity)
- **Descriptive BCs, not prescriptive constraints** — the archive reveals which topologies can exist at which (thickness, path-ratio, etc.), rather than fitness-pushing toward a pre-decided design.

**The canary: rediscovering the pancake lens from scratch**, with zero mention of folding, reflection, or thinness in the fitness or constraint set. If the archive's thin-high-path-ratio corner becomes occupied by 3-pass folded designs, the emergence thesis carries through to NS.

**What this rules out in the design**:
- Constraining reflectivity to 0.5 (prescribes the beamsplitter)
- Forcing a rear mirror (prescribes the fold)
- Fitness terms for thinness or efficiency (prescribes the outcome)
- Any BC axis that encodes "foldedness" directly

Allowed: BCs that *measure* path-length-ratio, thickness, interaction count. Those are descriptive — the archive can fill them with any topology, including sequential ones.

---

## 2. Why QD is the right tool for NS — the discontinuity analysis

**NS optimization is piecewise-smooth**, which is the central pathology. Tiny parameter changes flip discrete decisions:
- Which surface a ray hits first (argmin over intersection distances)
- Whether a ray is vignetted at an aperture
- How many bounces until flux threshold
- Whether Snell's critical angle is crossed (TIR)

Each is a step in the design-to-performance map. The fitness landscape is locally smooth *within a topology basin*, discontinuous *across basin boundaries*. Classical gradient-based lens design fails here by settling into an early topology basin and polishing to a local optimum — it cannot hop to a qualitatively different topology. That's why human designers still pick starting points.

**QD reframes this structurally**:
- Archive cells correspond (imperfectly but usefully) to topology classes
- Discontinuities become **archive cell walls, not optimizer traps**
- Both sides of a cliff get populated independently by different θ trajectories
- Stranded good solutions (in narrow basins) are preserved the moment stumbled on
- CMA-MEGA's multi-emitter setup handles both: evolutionary emitters stumble into new basins, gradient emitters polish within them

**The honest caveat**: straight-through gradients at argmin give biased signals near surface-choice crossovers (they exploit within a basin that's about to end). Mitigation: we rely on emitter diversity — evolutionary emitters don't care about gradient bias. Accepted trade-off.

**Second caveat**: BC space must actually separate topologies. If descriptor axes are wrongly chosen (e.g., correlated quantities), discontinuities re-appear intra-cell. **First pancake run doubles as a BC-space audit.**

---

## 3. Scope — Tier A first

| Dimension | Tier A (v1, planned) | Tier B (v2, deferred) |
|---|---|---|
| Reflectivity | Scalar continuous `r ∈ [0,1]` | Polarization-dependent |
| Polarization | Ignored | Full Jones calculus, QWP/polarizer surface types |
| Coatings | None | Multilayer coating model |
| Efficiency realism | Geometric flux split only | Physical throughput matching real pancake numbers |

Tier A is enough for geometric pancake discovery (~25% efficient unpolarized version). Tier B is the realistic VR-grade model. If Tier A succeeds, Tier B is a follow-up paper.

**Fixed-N surfaces**: topology emerges from continuous parameters only. No transdimensional moves (unlike Côté et al. SIGGRAPH Asia 2025). This keeps CMA-MEGA's fixed-dim θ space intact. A 6-slot homogeneous template can represent (via continuous parameters):
- Classical triplet (3 effective surfaces, others inert via `c→0` and matched indices)
- Mangin (front refract + rear reflect on same element)
- Pancake-like 3-pass (two partial mirrors with retrace)

All in the same θ-space. CMA-MEGA gradients remain well-defined across topologies.

---

## 4. Literature landscape

### Differentiable NS ray tracing — direct prior art
- **van Gestel et al. 2024, Optimization and Engineering** — [arXiv:2302.12031](https://arxiv.org/abs/2302.12031). Algorithmic-differentiable NS tracer for freeform *illumination*. Closest methodological prior art. **Still need to read full PDF** — my fetch was corrupted; need their exact gradient-aggregation technique at ray splits.
- **DiffRayFlow (Wu et al., Photonics 2025)** — [MDPI](https://www.mdpi.com/2304-6732/12/12/1243). Discrete optimal transport + differentiable NS, 20s for 1M rays. Illumination-focused.
- **Wang et al. 2022, IEEE TCI — dO** — [IEEE Xplore](https://ieeexplore.ieee.org/document/9919421/). The DeepLens foundation paper. Sequential only.

### Topology search in lens design
- **Côté et al. 2025, SIGGRAPH Asia — Automated compound lens design with discrete-continuous optimization** — [project page](https://imaging.cs.cmu.edu/automated_lens_design/). Transdimensional MCMC for add/remove/fuse/split. **Reference for what we're NOT doing.** Fixed-N keeps us out of their regime.
- **Yang et al. 2024, Nature Communications — Curriculum learning for ab initio deep learned refractive optics** — [nature.com](https://www.nature.com/articles/s41467-024-50835-7). Same DeepLens group. Sequential ab initio via curriculum. Inspiration for Stage 4 if QD bootstrap is hard.
- **Volatier et al. 2021, Nature LS&A — Three-mirror topology search** — [nature.com](https://www.nature.com/articles/s41377-021-00510-z).
- **Gagné et al. — Genetic algorithms for lens design review** — [arXiv:1811.10483](https://arxiv.org/abs/1811.10483).

### QD foundations
- **Fontaine & Nikolaidis 2021, NeurIPS — CMA-MEGA (Differentiable QD)** — [NeurIPS](https://proceedings.neurips.cc/paper/2021/file/532923f11ac97d3e7cb0130315b067dc-Paper.pdf). Our current solver.
- **Mouret & Clune — MAP-Elites original**. Background.
- **Self-Referential Differential MAP-Elites** — [arXiv:2107.04964](https://arxiv.org/abs/2107.04964). Worth knowing, probably not needed.

### Pancake lens physics (target validation)
- **Tripling Efficiency of Pancake Optics (MDPI Crystals 2025)** — [MDPI](https://www.mdpi.com/2073-4352/15/1/49). Baseline 25% → 71%+ with Faraday rotator, 93% with polarizer+AR coatings.
- **Avantier pancake technical overview** — [avantierinc.com](https://avantierinc.com/solutions/custom-optics/pancake-lenses-for-vr-optical-systems/). Canonical 17-21mm thickness.

### Gap
No work found applying MAP-Elites / QD to optical or photonic design. **This is research whitespace.**

---

## 5. Library landscape — why own tracer

Survey of candidate libraries for differentiable NS geometric ray tracing:

| Library | Diff? | NS? | Polarization? | Verdict |
|---|---|---|---|---|
| **DeepLens (public)** | Yes | **No** — NS is collab-only | No (collab-only) | Our current sequential base |
| **Optiland** | PyTorch autograd | **Planned**, not shipped | Partial | Closest architecture, but no code yet |
| **pyOpTools** | No (NumPy) | Yes (native) | Some | **Validation oracle only** |
| **Torch Lens Maker** | Yes | **No** (sequential-only) | No | Hobby project, GPL |
| **TorchOptics** | Yes | N/A (Fourier wave, not geometric) | Yes (Jones) | Wrong domain |

**No public library does differentiable NS geometric ray tracing.** Optiland is the closest architectural match and has NS on the roadmap, but it's unshipped. We write ~1.5–2k LoC of PyTorch NS tracer, reusing DeepLens's sag/normal primitives where useful.

**Why DeepLens's private NS mode wouldn't help even if we got access**: it's described as "for **stray light analysis and optimization**." Stray-light analysis asks "given a working sequential design, what parasitic paths exist that should be suppressed?" — the inverse of our discovery problem. That propagator is very likely architected for r≈0.02 unintended residuals, not intentional 50/50 beamsplitters with reflectivity as a design variable. Propagator-level reuse wouldn't happen; only intersection math (which is identical to sequential) would transfer.

---

## 6. Decisions committed

| Decision | Choice | Rationale |
|---|---|---|
| Own tracer vs. reuse | **Write our own** | No public option; see §5 |
| Ray branching at partial mirrors | **Soft (trace both branches)** + bounce budget 6 | Exact gradients; worst case 2^6=64 rays per input, tractable |
| "Which surface next" | **Hard argmin + straight-through gradient** | Simpler; swap to softmin only if gradients misbehave |
| Surface ordering | **Fully unordered** | Fold topologies require out-of-z visitation |
| Signed gaps | **No** (all gaps positive) | Direction reversal happens via reflectivity, not negative space |
| Topology encoding | **Fixed-N homogeneous slots, continuous params only** | CMA-MEGA-native; no transdimensional ops |
| Polarization | **Not in v1 (Tier A only)** | Most pancake geometry discoverable geometrically |
| Bounce budget | **6** | Accommodates 3-pass pancake with headroom |
| Backend | **PyTorch, CPU** | Matches existing framework; NS is CPU-native historically |
| Archive structure | **Grid, not CVT** | 2D BC space; CVT is premature. Revisit if BC dim ≥ 3 |
| Paraxial solver for NS | **Bypassed** | EFL estimated numerically from chief-ray exit angle |
| Descriptors for NS mode | **New `topology.py`** | Seidel/Zernike don't apply cleanly to NS |
| First pancake experiment | **On-axis only, single wavelength** | Simplify; expand after proof |
| θ seeding | **Random** | The point of discovery |

---

## 7. Architecture — component inventory

```
                        ┌──────────────────────────┐
                        │ config/schema.py         │
                        │ tracing_mode: "ns"       │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │ NSLensTemplate           │  (template_ns.py)
                        │ N homogeneous slots      │
                        │ shared volume constraint │
                        └────────────┬─────────────┘
                                     │ θ (continuous, fixed-dim)
                                     │
                        ┌────────────▼─────────────┐
                        │ NS tracer (problem/ns/)  │
                        │ ├── surface.py           │
                        │ ├── ray.py               │
                        │ └── propagator.py        │
                        └────────────┬─────────────┘
                                     │ rays on sensor + stats
                                     │
                ┌────────────────────┴────────────────────┐
                │                                         │
┌───────────────▼────────────┐     ┌────────────────────▼──────────┐
│ LensProblem._evaluate_ns   │     │ TopologyDescriptors           │
│ fitness = −RMS spot        │     │ BC1: path_length_ratio        │
│                            │     │ BC2: thickness/EFL            │
└──────────────┬─────────────┘     └────────────────┬──────────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              │ (fitness, measures, grads)
                              │
                 ┌────────────▼─────────────┐
                 │ CPUWorkerPool (unchanged)│
                 │ → CMA-MEGA runner        │
                 └──────────────────────────┘
```

### New code
- `problem/ns/surface.py` — `Surface` dataclass, kinds `{REFRACT, REFLECT, PARTIAL_MIRROR, SENSOR, ABSORB}`, methods `sag`, `normal`, `intersect`. Port from DeepLens.
- `problem/ns/ray.py` — `RayBatch` tensor-struct: origin, direction, flux, wavelength, bounces, alive mask, hit_history. All batched tensors with matching lead dim.
- `problem/ns/propagator.py` — `trace(template, rays, max_bounces) → rays_out, stats`. Inner loop: intersect-all-surfaces → hard argmin + straight-through → apply interaction → handle PARTIAL_MIRROR by split+concat → terminate on sensor/escape/bounce-budget/flux-threshold.
- `problem/ns/validation.py` — three oracle tests: (a) reproduce sequential triplet when all r=0; (b) Mangin textbook match; (c) energy conservation.
- `problem/template_ns.py` — `NSLensTemplate` with homogeneous slots; `build_ns_lens(template, θ) → List[Surface]`.
- `descriptors/topology.py` — `path_length_ratio`, `thickness_efl_ratio`, `interaction_count`, `reflection_fraction`, `footprint_aspect`.

### Extended code
- `config/schema.py` — add `tracing_mode: Literal["sequential", "non_sequential"]`, plus NS-specific fields (`ns_num_slots`, `ns_volume_diameter_mm`, `ns_volume_length_mm`, `ns_max_bounces`, `ns_flux_threshold`, `ns_allow_partial_mirrors`).
- `problem/lens_problem.py` — add `_evaluate_ns` branch alongside existing `_evaluate`; dispatch on `cfg.tracing_mode`.
- `config/examples/ns_pancake_discovery.yaml` — new example config.
- Archive renderer — NS-aware ray drawing (multiple segments per ray, color by bounce, annotate partial mirrors).

### Unchanged
- `core/runner.py`, `core/cpu_pool.py` — worker protocol unchanged
- `archive/` — structure unchanged
- CMA-MEGA solver in `optical_rl/qd/solvers/singlet_cmamega.py`
- Existing sequential pipeline (NS is opt-in)
- Existing descriptors (Seidel, Zernike, random) — still available for sequential mode

### Bypassed for NS
- `problem/paraxial.py` — paraxial EFL closure doesn't generalize; use numerical chief-ray method instead

---

## 8. Implementation roadmap

### Stage 1 — tracer foundation (~1 week)
- [ ] `problem/ns/surface.py` with `Surface` dataclass + intersection
- [ ] `problem/ns/ray.py` with `RayBatch`
- [ ] `problem/ns/propagator.py` forward-only, refract + reflect only (no partial mirrors)
- [ ] Unit tests: single refraction, single reflection, doublet, Mangin
- [ ] **Checkpoint: trace a Mangin, recover textbook focal length**

### Stage 2 — branching (~1 week)
- [ ] `PARTIAL_MIRROR` handling: ray-split + concat
- [ ] Bounce budget + flux threshold
- [ ] Autograd sanity: `∂(spot)/∂r` is finite and non-zero
- [ ] `problem/ns/validation.py` with three oracle tests
- [ ] **Checkpoint: gradient test passes; validation oracle green**

### Stage 3 — integration (~1 week)
- [ ] `problem/template_ns.py` with homogeneous slots
- [ ] `config/schema.py` extensions
- [ ] `problem/lens_problem.py` `_evaluate_ns` branch
- [ ] `descriptors/topology.py`
- [ ] `config/examples/ns_pancake_discovery.yaml`
- [ ] **Checkpoint: single θ runs through full eval path without crashing**

### Stage 4 — first QD run (~1 week: 2 days runtime + 3 days analysis)
- [ ] Short QD run (10k evals) on 4-slot template — archive populates
- [ ] Visual archive inspection; BC axes audit
- [ ] Full run (100k+ evals) on 6-slot template → **pancake discovery attempt**
- [ ] pyOpTools validation on elite designs

**Total: ~4 weeks of focused work.**

---

## 9. Pancake-discovery milestone — success criteria

Before claiming "we discovered pancake":

1. **Fitness/constraints are clean**: no term mentions folding, thinness, reflection count, efficiency, or pancake. Only RMS spot + flux-on-sensor penalty.
2. **BCs are descriptive**: `(path_length_ratio, thickness/EFL)`, both measured, not targeted.
3. After ≥50k evals, archive cells with `path_length_ratio ≥ 2.5 AND thickness/EFL < 0.5` are occupied with fitness comparable to the best sequential elite.
4. Chief-ray inspection of such an elite shows: ray enters → hits a partial mirror (transmit) → hits a rear mirror (reflect) → hits same partial mirror (reflect) → exits forward to sensor.
5. pyOpTools reproduces the topology and RMS spot within tolerance.

**Falsification**: if after 200k evals those cells remain empty or occupied only by degenerate designs, the archive-as-revelation premise failed; reconsider descriptor choice or template expressiveness.

---

## 10. Open questions (flag-for-later, don't block implementation)

1. **Surface-count regularization**: include as soft fitness term, as BC axis, or ignore? Default: ignore, let archive reveal.
2. **Multi-field in first pancake run**: on-axis only vs. 3-point field. Default: on-axis only for first proof.
3. **θ seeding**: random vs. Mangin-biased. Default: random.
4. **Elite visualization detail**: chief ray only vs. chief + marginal vs. full fan. Default: chief + marginal, color-coded by bounce.
5. **Exactly which gradient-aggregation technique from van Gestel et al. to adopt** — need to read the PDF properly. Currently the `.pdf` fetch returned binary; download locally when resuming.

---

## 11. Anti-patterns — what NOT to do

Things that would defeat the research thesis:
- ❌ Add a fitness term rewarding thinness or compactness
- ❌ Constrain reflectivity to 0.5 (prescribes the beamsplitter)
- ❌ Force a rear mirror (prescribes the fold)
- ❌ Use a BC axis that directly encodes "foldedness" (e.g., "number of reflections" as a target)
- ❌ Use transdimensional MCMC (Côté et al. pattern) — defeats fixed-N discipline
- ❌ Lean on DeepLens's private NS mode — it's architected for stray-light suppression, not discovery
- ❌ Add CVT archive for the first 2D experiment — premature complexity
- ❌ Build polarization into v1 — Tier B scope
- ❌ Switch to GPU before profiling — CPU is the historical NS fit

Things that would be technically wrong:
- ❌ Use softmin-with-temperature for argmin from the start — premature; hard + straight-through is simpler to debug
- ❌ Make NS the default mode — it's opt-in via `tracing_mode`
- ❌ Modify `core/runner.py` or `core/cpu_pool.py` — they don't need to know about NS

---

## 12. Resumption checklist

When picking this back up:

1. Reread this doc top-to-bottom (20 minutes).
2. Download the van Gestel et al. PDF locally from [arXiv:2302.12031](https://arxiv.org/abs/2302.12031); read Sections 2–4 carefully, specifically the gradient-aggregation technique.
3. Verify the memory pointer still exists: `~/.claude/projects/-home-hisham26-Desktop-RL/memory/project_lensmega_thesis.md`.
4. Check that sequential framework state is clean (current branch, no broken evaluations).
5. Start Stage 1 per §8. First code file: `problem/ns/surface.py`.

---

## 13. Sequential framework state at pause

At the time of pausing, the sequential framework was under active refinement (not NS work):
- Multiple ablation configs added (untracked): `ablation_random_bc.yaml`, `ablation_seidel_s2s3.yaml`, `ablation_single_objective.yaml`, `ablation_sixlet_discovery_materials_*.yaml`, `cooke_triplet_materials.yaml`, `quadruplet_discovery_materials.yaml`, `sixlet_discovery_materials.yaml`
- Scripts added (untracked): `compare_ablations.py`, `make_ablation_configs.py`, `refine_unrefined_elites.py`, `render_family_grid.py`, `render_lensnet_samples.py`, `rerender_elites.py`, `analyze_ef50_original.py`
- `descriptors/random_proj.py` added (untracked)
- Modifications in-progress across core problem/descriptor/runner files

Resume sequential-result refinement before NS work if the ablation sweeps haven't been consolidated.
