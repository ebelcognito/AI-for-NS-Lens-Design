# Jewellery optics: prior work and first demo protocol

Research date: 2026-09-29. Status: literature/product scan and proposed experiment, not a simulation result. No gemstone search, human study, or fabrication was performed for this brief. Rankings are engineering judgments. This is not an exhaustive novelty or patent-claim analysis.

## Decision

Build toward a synchronized motion demo: **one material, one size envelope, three optical personalities**. Discover broad flashes, fine sparkle, and a directional sweep of highlights. Start with the first two; treat the sweep as a stretch target. Show a conventional reference, brightness-only optimization, and appearance-directed discovery under identical lighting and camera settings. Switch to held-out illumination live or in a second synchronized clip.

Pitch hypothesis: a shared optical discovery platform can design a product's light behaviour, including aesthetic behaviour. Jewellery is one demonstrator branch. The experiment must establish whether discovery offers useful control and diversity beyond conventional parameter adjustment.

Do not pitch gemstone ray tracing, automatic angle optimization, cut generation, a visual selection grid, or preference feedback alone as new. All have close precedents.

## Closest precedents

1. **GemRay** — automatic Nelder–Mead optimization of crown/pavilion scale factors; merit is weighted brightness across lighting models and tilt angles. It is the closest simple optimization baseline. A reimplementation of that objective/parameter restriction is only “GemRay-like,” not a benchmark against the actual product. [Developer manual](https://www.gemcad.com/downloads/gemrayman.pdf)

2. **Gem Cut Studio** — integrated ray-traced preview, tilt performance and a manual optimizer showing a grid of crown/pavilion tangent-ratio combinations. The manual leaves the final subjective choice to the user. Thus an appearance gallery by itself is not the differentiator. [Developer manual, Manual Optimizer section](https://www.gemcutstudio.com/app_download/UserManual_v110.pdf)

3. **GIA cut appearance research** — weighted light return (WLR) and dispersed coloured light return (DCLR), with different preferred proportions for the two objectives. Use as metric precedents, not an assertion that our proxies reproduce GIA grades. GIA combined simulation with extensive observer evaluation. [2001 paper](https://www.gia.edu/gems-gemology/fall-2001-modeling-appearance-round-brilliant-cut-diamond-reinitz), [research background](https://www.gia.edu/diamond-cut/cut-grade-research-background)

4. **OctoNus / DiamCalc / Cutwise** — existing cut modelling, realistic images, movies, appearance metrics and standardized visual comparisons. Cut Evolution renders proposed cutting solutions, including absorption-dependent colour. This is a close commercial ecosystem, not just generic jewellery CAD. [DiamCalc documentation](https://www.octonus.com/static/DwKX5URr32hFLSy.pdf), [Cutwise workshop](https://docs.cutwise.com/blog/workshop-full-program), [Cut Evolution](https://pricing.cutwise.com/cut-evolution)

5. **Generation of gem cuts, US20240391055A1** — published application lists GIA as original assignee. Describes symmetry-based cut generation, tunable 3D models, ray-traced evaluation, genetic optimization, optional learned metric surrogates, and collecting high-performing models. This materially narrows broad novelty language about evolutionary discovery of gemstone cuts. Disclosure is not proof of measured performance or a conclusion about claim scope. [Published application](https://patents.google.com/patent/US20240391055A1/en)

6. **Method for determining a cut for a gemstone, EP2713800B1** — further prior disclosure around cut selection and optical responses including scintillation, brilliance and fire. Relevant to a subsequent detailed related-work review. [Publication](https://patents.google.com/patent/EP2713800B1/en)

7. **Fantasy/concave gemstone cutting** — existing lapidary art deliberately creates bright lines, contrast, optical illusions and apparent internal patterns. This is the craft benchmark for later freeform experiments. [GIA article](https://www.gia.edu/gia-news-research/making-cut-art-science-fantasy-carving)

8. **Jeweler-in-the-loop (2026)** — preference-based Bayesian optimization of alloy colour with a physical forward model. The reported study uses an in-silico simulated user; it is not evidence of a real gemstone preference trial. Useful precedent for a later preference layer. [Publisher PDF](https://pubs.rsc.org/en/content/articlepdf/2026/dd/d6dd00009f), [code archive](https://doi.org/10.5281/zenodo.18164808)

9. **Using AI to Design Stone Jewelry (2018)** — optimization and machine learning for arrangements of stones. Adjacent composition-design work; distinct from optimizing internal light transport. [Paper](https://arxiv.org/abs/1811.08759)

10. **High-contrast Computational Caustic Design (2014)** — inverse surface design for target projected caustics, including fabricated examples. A motif-projecting pendant would inherit this research problem and add wearable geometry/lighting constraints. [EPFL publication](https://graphsearch.epfl.ch/en/publication/d13e8a01-c1e9-4e8e-ae52-f5cad3ada804)

11. **diamond-webgl** — public real-time gemstone rendering code useful for presentation references and exploration. It includes postprocessing; visual quality alone does not establish quantitative optical fidelity. Check source assumptions and licensing before reuse. [Author repository](https://github.com/piellardj/diamond-webgl)

## Demo choices

| Candidate | Immediate value | Main difficulty | Decision |
|---|---|---|---|
| Three controlled sparkle styles | Directly shows discovery, diversity and refinement | Reliable dynamic metrics and honest rendering | First demo |
| A sweeping internal highlight or motif | Memorable motion and design intent | Target may be fragile across environments | Stretch target |
| Projected logo/flower caustic | Easy to communicate desired output | Source extent, receiving plane and orientation sensitivity | Separate later branch |
| Rich coloured fantasy cut | Strong jewellery relevance | Absorption, anisotropy and fabrication modelling | Later |
| Whole ring/setting co-design | Closer to finished product | Metals, occlusion, mechanical constraints and more DOFs | After single-stone validation |

## Experiment J0: validate the 3D optical evaluator

The repository contains reusable CMA search and guarded refinement, a general 2D Fresnel/TIR discovery implementation, and a dedicated 3D reflective/grating model. The new module should be isolated, tentatively `nslens/jewellery/`, without changing existing domains. Extending Fresnel/TIR to 3D is feasible; it is a prerequisite to measure, not a reason to abandon the application.

Implement closed convex faceted solids first. Use planes/half-space clipping or robust mesh intersections; validate outward normals and inside/outside transitions. Do not use a prescribed facet sequence. Start with a monochromatic material at fixed n=1.60: explicitly a synthetic optical material, not diamond and not a validated real-glass prescription. No fire or colour claim in this pilot.

Track reflected and transmitted branches, TIR and unresolved/pruned energy. Correct 3D polarization transport requires a changing local incidence basis; if scalar unpolarized Fresnel is used initially, label the approximation and verify ranking sensitivity against a polarization-capable reference before strong physical claims.

Tests before optimization:
- Normal-incidence Fresnel against the analytic expression; Snell refraction and critical angle on both sides of the boundary.
- Oblique plane-parallel slab and repeated internal reflections, with analytic total transmission/reflection.
- Rigid rotation invariance and reciprocal ray paths.
- Geometry entry/exit at edges and grazing incidence; no duplicate facet intersections.
- Energy closure with explicitly bounded residual/pruned energy, across bounce limits and cutoffs.
- Constant-radiance environment sanity check for a lossless solid with camera and environment in air.
- Image and descriptor convergence under increased pixels, samples, motion frames and bounce limits.

Use radiometrically consistent camera-path evaluation. With endpoints in air, refractive radiance factors must be treated consistently through complete paths. Resolve finite sources and observer obstruction. An independent renderer or analytic special case must validate more than another setting of the same code.

J0 deliverables: tests, render of a simple reference cut, rocking animation, convergence table and documented modelling scope. It is a capability demo, not an optimized-jewel result.

## Experiment J1: can appearance be controlled?

### Fixed setup

- One colourless material; fixed refractive index; no absorption/dispersion in the first stage.
- 10 mm maximum girdle diameter and common depth limits. Report actual volume; do not call equal diameter equal carat. For tight comparisons match volume within a predeclared tolerance or report a volume/performance tradeoff.
- Explicit facet families with shared bounds across all optimizers. Begin with a symmetric multi-tier family and a bounded asymmetric variant. Separate family search from continuous CMA parameters.
- Geometry gates: closed convex body, positive volume, minimum facet/edge size, nonzero girdle thickness and bounded depth. Pilot limits are not certified manufacturing rules; lapidary review is a later gate.
- Fixed orthographic camera for the first controlled study, followed by perspective/view-distance audit. No auto-exposure, bloom, synthetic sparkle overlays or candidate-specific lighting.
- Three training environments: broad overhead source, several finite spot sources, and a window-like source plus fill. Use equal illumination conventions across candidates. Reserve altered source positions, sizes and a new motion axis for testing.
- Training motion: two orthogonal rocking sweeps from -15 to +15 degrees, initially 31 frames each. Audit at 121 frames and finer angular offsets; inspect angular rather than arbitrary frame-rate effects.

### Descriptors and objectives

Compute raw linear luminance and mask the stone. Define metrics in a documented viewing scale. Bright-flash thresholds must be fixed from the reference/calibration, not rescaled per candidate. Apply the same optical/image resolution convention before connected-component measurement.

- Brightness B: mean luminance over stone, poses and environments.
- Persistent darkness D: fraction of surface-image area consistently below a fixed relative-to-illumination threshold, with pose registration or clear per-frame aggregation.
- Flash scale A: energy-weighted connected bright-region area divided by projected stone area.
- Motion activity V: average absolute luminance change per degree of rotation, accompanied by event duration and coverage.
- Brightness lower tail: low-percentile frame-average brightness, to prevent a single good pose dominating.

Target broad flashes with high A and acceptable activity; target fine sparkle with lower A, multiple resolved events and acceptable coverage. Activity alone is not beauty and can reward aliasing. A directional sweep requires a separate registered spatial highlight trajectory objective; omit it from claims until demonstrated.

Use feasibility/brightness/darkness gates, then maintain appearance niches in A and V. Preserve internal path signatures as diagnostics. Refine selected elites while retaining their intended style region. Re-evaluate all branches after refinement. Start with derivative-free local refinement; use autodiff only after validating image/objective gradients around visibility transitions.

### Controls and budget

Use the same geometry space, lighting, samples and evaluation accounting:
1. Conventional reference(s), unchanged.
2. Two-parameter crown/pavilion local optimization: a GemRay-like conceptual baseline, explicitly not an actual GemRay product run.
3. Uniform random search in the full space.
4. Multistart CMA with the same style objectives as the discovery arm, retaining all evaluated candidates for a fair post-hoc gallery.
5. Our archive/retained-candidate discovery plus refinement.

Separately include brightness-only full-space CMA to show how objective choice changes appearance; this is an objective ablation, not proof of an optimizer advantage.

Initial proposed budget: five seeds per search arm, 1,000 coarse candidate evaluations per seed, 200 refinement evaluations and 100 reserved dense validation evaluations. Define cost in traced camera rays/segments as well as calls and wall time. Share the total budget across facet families. Profile J0 before launching this budget; these are proposed allocations, not runtime estimates.

Reserve held-out illumination for a final report rather than repeatedly choosing winners on it. If a final set fails, report failure and treat any subsequent development as a new experiment with a fresh holdout.

### Evidence and gates

Report feasible yield, appearance coverage at a common quality floor, target-style accuracy, held-out degradation, volume/depth, and wall time across seeds. Use fixed archive bin edges established before comparing algorithms. Save all candidates so baselines receive the same display selection opportunity.

Proceed to a pitch-ready simulation demo only if:
- Numerical checks pass and selected descriptors/rankings are stable under finer evaluation.
- At least two styles remain clearly distinct across held-out lighting and motion; allow failure of the third.
- The result is not explained only by volume, exposure, resolution or a different geometry budget.
- Discovery improves coverage or search efficiency relative to matched multistart CMA, or the pitch is explicitly limited to appearance control rather than algorithm superiority.

A small blinded pilot can test whether people correctly distinguish intended styles and which they prefer. Report participant count, protocol and uncertainty; do not call this universal beauty validation. Human recruitment and physical cutting are subsequent work, not performed here.

## Pitch artifact

A 30–45 second synchronized sequence:
1. Common material/envelope and conventional reference.
2. Broad-flash and fine-sparkle candidates under the same rocking motion.
3. Switch lighting and motion to held-out conditions.
4. Brief geometry/ray-path view explaining the physical mechanism.
5. Archive view and before/after refinement for a selected candidate.

Caption every current-stage output “optical simulation.” A render is not a fabricated product. Show exact geometry parameters, material model, evaluation settings and quantitative comparison alongside the video.

A defensible future claim, conditional on J1: “Our optical discovery framework generates and refines geometry for distinct light behaviours. Jewellery demonstrates control over appearance in motion.”

Unproven claims: first AI gemstone designer, universally more beautiful jewellery, new gemstone physics, manufactured performance, or superior commercial value.

## Next implementation milestone

Implement J0 and measure its render cost. J1 follows only when the evaluator is credible. This brief completes the research-first step and fixes the comparison before choosing attractive examples.
