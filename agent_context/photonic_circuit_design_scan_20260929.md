# Photonic circuit design: transfer assessment and research opportunities

Scan date: 29 September 2026. Scope: classical integrated photonic circuits, with adjacent nanophotonic device, programmable-mesh, and quantum-circuit work checked for overlap. This is a targeted technical scan, not an exhaustive literature review. No PIC implementation, simulation, or performance demonstration was performed.

## Assessment

The framework can plausibly transfer at the search-workflow level. The strongest candidate is fabrication- and tuning-constrained circuit architecture discovery with a portfolio of physically distinct alternatives. The existing geometrical ray engine cannot evaluate these circuits. A coherent circuit model and a new representation are required.

There is substantial prior work on topology discovery, local refinement, diversity, and robust photonic design. None of those alone establishes a new contribution. The proposed research question is narrower: does preserving physically meaningful architecture classes improve validated yield, constrained retargeting, or resilience to model mismatch compared with graph-diverse and Pareto search at equal computational cost?

## What our repository actually supplies

The inspected implementation uses continuous geometric genomes, non-sequential tracing, measured interaction descriptors, quality-diversity retention, exact-signature cataloguing, physics gates, and guarded local refinement. Read README.md, ns_autodiff/README.md, nslens/qd/descriptors.py, nslens/qd/problems.py, and REPORT_OF_INVENTION_DRAFT.md for the project evidence and its limitations. Reported ray-tracer validation is evidence about the current optical model only.

| Existing capability | PIC translation | Reuse boundary |
|---|---|---|
| CMA-ME search and archives | Maintain competitive circuit families | Continuous emitters can optimize parameters; graph edits need a new discrete search layer |
| Ray interaction signature | Canonical component/port graph plus measured response descriptors | A dominant ray path is not a sufficient description of interference |
| Geometry and energy gates | Port validity, passivity, connectivity, loss, design rules, model validity | New gates and observations are necessary |
| Frozen-topology refinement | Hold connectivity fixed; optimize lengths, couplings, phases | Reuse the workflow, replace the differentiable physics |
| Dense independent rescoring | Finer spectra, held-out process samples, layout-aware extracted models | Component electromagnetic checks and independent model checks are needed |
| CODE V handoff | Netlist, routed layout, extracted netlist, resimulation | New exporter; layout generation alone does not establish foundry readiness |

Our current Problem interface embeds lens-specific source/detector and geometry assumptions. A separate PIC problem/evaluation adapter is more appropriate than treating this as one additional lens problem.

## Closest prior work

1. **AESOP / Inverse Design of Photonic Systems (2024).** Discrete graph mutation, differentiable parameter optimization, sensitivity-based simplification, and a Hall of Fame enforcing graph-edit-distance diversity already form a closely related workflow. Its described propagation model is a directed acyclic graph and excludes circuit loops and backward reflections. This is a necessary baseline; a generic diversity claim would be weak. [Paper](https://onlinelibrary.wiley.com/doi/10.1002/lpor.202300500), [author code](https://github.com/benjimaclellan/aesop).

2. **ADEPT-Z (ASPDAC 2025).** Evolutionary topology search returns diverse Pareto-optimal photonic tensor cores under hardware constraints. This makes optical AI architecture search a competitive entry point. [Paper](https://arxiv.org/abs/2410.01313).

3. **AutoScatter (PRX 2025).** Searches discrete and continuous coupled-mode systems and identifies minimal/interpretable scattering realizations. Feedback or resonator graphs cannot be presented as unexplored simply because AESOP uses feedforward graphs. [Author code and publication](https://github.com/jlandgr/autoscatter), [paper](https://arxiv.org/abs/2404.14887).

4. **Grammar-based topology search for reconfigurable integrated photonic spectrometers (September 2026 preprint).** Uses canonicalization, physical pruning, coherent circuit evaluation, and joint topology/parameter search. Its stated exclusions include fabrication error and thermal crosstalk; global recirculating architectures also lie outside its grammar. This is a concrete benchmark with identifiable modeling limits, not evidence that these topics are absent elsewhere. [Preprint, methods sections 2.1–2.2](https://arxiv.org/html/2609.03196v1).

5. **Recursive inverse design for hyper-spectral PICs (2025 preprint; 2026 journal article).** Adds reflective nodes and co-designs circuit and geometry; even component count can emerge from optimization. The bridge from circuit synthesis to geometry is already being studied. [Preprint](https://arxiv.org/abs/2510.12104), [journal article](https://www.nature.com/articles/s41467-026-76328-3).

6. **Robustness is established prior work.** Stochastic circuit optimization has been demonstrated on coupled-ring filters. Fabrication-constrained device inverse design has experimental demonstrations. The opportunity must exceed simply adding tolerance sampling. [Circuit study](https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2016-0110/html), [fabrication-constrained devices](https://pmc.ncbi.nlm.nih.gov/articles/PMC5431915/).

Exact-term searches for photonic MAP-Elites and quality-diversity did not establish a direct match in this scan. That is weak evidence: AESOP and ADEPT-Z already implement closely related diversity objectives using different terminology.

## Physics change

A circuit must propagate complex modal amplitudes and coherently combine them. At an output, power depends on |sum a_p exp(i phi_p)|^2, not the sum of independent path powers. Resonators involve repeated circulation. Therefore, selecting a single high-energy route, or freezing its event list, is generally invalid for a PIC.

Use frequency-dependent component scattering matrices. For a linear circuit, internal amplitudes can be written schematically as x = M x + B u; solve (I-M)x = B u and recover external outputs. Do not impose an arbitrary bounce cutoff. Near resonances, check residuals, conditioning, and spectral resolution; large internal buildup need not violate external power conservation.

With consistently power-normalized ports, a passive external S matrix satisfies S†S <= I; reciprocal components additionally obey reciprocity under the appropriate port conventions. Our proposed first study should use passive linear components and avoid gain/nonlinearity. A circuit model is valid only when its component and interconnection models capture relevant coupling, reflections, and dispersion. Closely packed structures may require joint electromagnetic characterization.

SAX supplies differentiable frequency-domain S-parameter simulation. GDSFactory supplies layout/netlist tools, and its SAX tutorials include layout-aware variation modeling. These are suitable candidate backends, with model/PDK validity still to be established for a chosen process. [SAX](https://github.com/gdsfactory/sax), [variation tutorial](https://gdsfactory.github.io/gplugins/notebooks/sax_01_sax/), [GDSFactory paper](https://gdsfactory.github.io/photonics_paper/).

## Ranked opportunities — research hypotheses

**1. Robust architecture portfolios for compact spectral circuits: strongest first test.** Search coupler, delay, phase-shifter, and optionally ring arrangements for a specified filter/demultiplexer response. Preserve distinct interference and resonance structures while optimizing validated performance. Evaluate width/thickness bias, local variation, temperature, coupling error, and bounded post-fabrication tuning. The possible advantage is retaining a topology that is slightly worse nominally but better under a new process corner or tighter tuning budget. This advantage must be demonstrated against robust Pareto search and graph-diverse search, not only nominal single-best optimization.

**2. Fabrication-aware reconfigurable spectrometers: clearest recent comparison.** Extend a bounded architecture grammar with process and control models; score held-out spectrum reconstruction after finite calibration and tuning. The September preprint provides a particularly relevant starting comparison. This is a larger study because reconstruction, calibration, and optical design must be controlled together.

**3. Recirculating circuit portfolios: greater physics challenge.** Investigate mixed interferometer/resonator architectures with coherent feedback, broadband specifications, and realizable couplings. The closest comparisons include AutoScatter and recursive inverse design. The opportunity is in constrained robustness and physical implementation; graph search itself is established.

**4. Programmable mesh recovery: secondary.** Preserve several low-loss, low-power realizations of a function and rapidly switch after component failure or drift. Fault-tolerant routing and self-configuration already exist, so quantify recovery cost or quality beyond those baselines. [Auto-routing study](https://riunet.upv.es/entities/publication/46ca80e3-a68c-4687-b4ab-0911abd6f508), [self-configuration study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7733469/).

**Lower priority:** generic voxel-level splitter/coupler inverse design, photonic tensor-core search, and quantum circuit discovery. These have strong domain-specific competitors and require more replacement of our current machinery. Quantum objectives additionally need an appropriate multiphoton model. [Quantum discovery example](https://doi.org/10.1103/c4v4-wz65).

## Representation and archive experiment

Maintain two identities: an exact canonical structural graph and a separate measured-response descriptor. Preserve component type, port role, source/detector identities, and control connections during canonicalization. Graph isomorphism is not the same as functional equivalence. Remove disconnected or provably redundant components, but do not erase a weak connection solely because it is dark at one wavelength.

Candidate coarse descriptors: number of independent feedback loops, delay-spread class, resonant versus nonresonant participation, and normalized phase/temperature sensitivity. Graph descriptors are cheap but may preserve cosmetic differences; response descriptors depend on wavelength sampling and may be unstable near degeneracies. Test these choices rather than assuming our ray-signature result transfers.

Keep the archive small initially: one structural descriptor and one engineering descriptor, with an exact graph catalogue beside it. Use feasibility thresholds and a quality metric inside each class. Report useful diversity only among designs that satisfy the same specifications.

## Concrete pilot and decision criteria

Proposed scope: a passive two-channel wavelength interleaver/demultiplexer on one selected platform, using a bounded library of characterized couplers, delays, and phase shifters. Begin with feedforward graphs; add rings only after verification. Freeze band, loss/crosstalk/ripple targets, footprint and tuning limits, fabrication distributions, and the graph grammar before comparing algorithms. Numerical specifications must come from the chosen process/application rather than arbitrary universal values.

1. Verify known MZI and cascaded-filter responses, energy conservation, numerical gradients, and independent circuit evaluation.
2. Compare multi-start conventional architecture optimization, random graph search plus identical local refinement, evolutionary graph search with graph-diverse retention, robust multiobjective search, and the proposed QD archive.
3. Use identical component models, variation samples during training, parameter bounds, candidate grammar, and refinement budgets. Report wall time as well as all circuit/gradient evaluations; distinct graphs may incur compilation overhead.
4. Hold out process samples, spectral points, and changed constraints. Measure all-specification pass probability, uncertainty in estimated yield, feasible structural diversity, performance/cost tradeoffs, and evaluations required to adapt to a changed tuning/loss budget.
5. Generate layouts for finalists, extract lengths/bends/crossings, and rescore with appropriate variation correlation. Validate suspect or novel components using electromagnetic simulation. Shared compact models cannot serve as independent physical validation of themselves.
6. Ablate structural-only versus response-informed archives, diversity retention, and robust scoring. A denser archive without better validated decisions is not a positive result.

Go forward if multiple independent runs show better feasible yield/performance at matched resources, or faster recovery under held-out constraints, with finalists surviving layout/model checks. Stop or narrow the claim if gains vanish against graph-diverse/Pareto baselines or after extracted-layout rescoring.

Practical recommendation: authorize a small circuit-model pilot before committing to a broad photonics pivot. Engineering transfer is plausible; competitive advantage remains an experimental question.
