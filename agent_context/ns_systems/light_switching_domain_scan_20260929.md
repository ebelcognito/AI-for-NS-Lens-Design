# Optical switching: framework fit and research opportunities

Research scan: 2026-09-29. Literature and implementation assessment only; no switch
was implemented or simulated. This is a targeted scan, not an exhaustive novelty search.

## Findings

- Strongest near-term fit: discover compact free-space beam-routing layouts using
  a constrained moving mirror or prism and shared static freeform optics.
- Potential contribution: discover different multi-state interaction sequences,
  then refine and tolerance-test their shared geometry and state-dependent controls.
  This remains a hypothesis, not an established literature gap or performance result.
- Integrated photonic switches are an active inverse-design field. Our search and
  learning ideas transfer conceptually, but the ray-tracing evaluator does not.
- A static passive element with a fixed input is not a controllable switch. The
  design must include an explicit physical control and its allowed states.
- Begin with a certified-solvable 1x2 case, then a constrained 1x4 case. Compare
  against conventional layouts and numerical search before investing in learning.

## Existing implementation and missing physics

The recent [camera implementation](../../nslens/camera/README.md) supports disjoint
2D bodies in air, nearest-surface selection across bodies, revisits, refraction,
TIR, ideal mirrors, and an energy ledger retaining Fresnel branches. Geometry,
pose and freeform coefficients are searchable. Body/face counts remain declared
priors. The [learning report](optical_learning_complete_writeup_20260910.md)
documents a small verified pencil-beam repair-planning experiment, not general
autonomous optical synthesis.

Useful reuse: scene geometry, branch accounting, input bundles, routing objectives,
CMA/restarts, interaction descriptors, repair tasks and verification infrastructure.
The older differentiable twin supports local refinement in its supported geometries;
it is not automatically a differentiable multi-body switching solver.

Needed even for a geometric pilot: shared geometry across states; bounded actuator
coordinates; finite output apertures AND acceptance angles; all-port energy scoring;
collision checks over motion; and dense state/tolerance audits. Port geometry and
source must stay fixed across switch commands.

Missing for qualified single-mode fiber results: 3D propagation and complex-field
mode overlap, including wavefront and diffraction. Captured ray energy is only a
geometric throughput proxy. Separate s/p intensities do not provide Jones-vector
phase or coherent interference. Real coating losses, dispersion, actuator dynamics,
mechanical reliability and thermal behavior require additional models.

## Domain ranking

| Area | Fit | Assessment |
| --- | --- | --- |
| Small free-space/moving-prism or MEMS routing modules | High for a 2D pilot | Best first test of shared optics and topology discovery across states |
| Optical relays within large optical circuit switches | Moderate | Relevant optical packaging problem, but 3D, coupling and many simultaneous paths matter |
| Industrial laser distribution/coupling | Moderate | Freeform geometry is relevant; power handling, thermal effects and beam quality complicate validation |
| TIR/FTIR switching | Conditional | Ordinary TIR is implemented; frustrated TIR requires evanescent-gap physics |
| LC/polarization or wavelength-selective switching | Low immediate fit | Needs polarization phase and/or diffraction models |
| Integrated MZI, ring, PCM or nonlinear all-optical switches | Low immediate fit | Needs Maxwell/coupled-mode and material/actuation physics; substantial existing inverse design |

Optical circuit switching routes a lightpath; it is distinct from packet switching
and from ultrafast nonlinear all-optical logic. Geometry optimization alone does
not establish switching speed, voltage or energy improvements.

## Primary sources and what they establish

1. [Mission Apollo (2022)](https://arxiv.org/abs/2208.10041): deployed datacenter
   optical circuit switching and coupled port-count, cost, speed and optical-performance
   requirements. Establishes application relevance, not a freeform-design gap.
2. [Optical MEMS Design for Telecommunications Applications (2024)](https://arxiv.org/abs/2403.04078):
   actuator, optical surface, integration and residual-stress design issues. Mechanical
   constraints cannot be inferred from an optical ray layout alone.
3. [Coherent moving-prism switches](https://www.coherent.com/networking/wavelength-management/dynamic-modules/opto-mechanical-switches):
   commercial 1x1, 1x2 and 2x2 devices. Moving prisms are an established mechanism.
4. [Optical MEMS switch with imaging system, US6704476B2](https://patents.google.com/patent/US6704476B2/en):
   imaging/telecentric relays and folded compact layouts. Folding a relay is not by
   itself a new contribution. Patent used as technical prior art, not legal analysis.
5. [Reduced-insertion-loss optical switching, US6438284B1](https://patents.google.com/patent/US6438284B1/en):
   prism geometry and FTIR switching already studied. Ordinary geometric TIR cannot
   evaluate tunneling through a subwavelength gap.
6. [Lou et al., inverse design through bilevel optimization (2023)](https://www.jessearodriguez.com/wp-content/uploads/2023/08/LouMeta.pdf),
   DOI 10.1021/acsphotonics.3c00113: fixed structure and tunable regions optimized
   jointly for multiple routing states. Broad claims of inventing multi-state
   co-design or meta-learning-inspired switch design are unavailable.
7. [Reconfigurable inverse designed phase-change photonics (2024)](https://arxiv.org/abs/2403.05649):
   pixel-level PCM control plus inverse design. Integrated reconfigurable photonics
   is already occupied; changing optimizer alone is a weak motivation.
8. [XLuminA (2024)](https://www.nature.com/articles/s41467-024-54696-y): automated
   optical experimental-layout discovery using a wave-optics framework. General
   automated optical discovery is also prior art.
9. [Differentiable geometric waveguide displays (2026)](https://arxiv.org/abs/2601.04370):
   adjacent non-sequential differentiable design; must be compared when framing the method.
10. [Coherent optical circuit switches](https://www.coherent.com/networking/optical-circuit-switch):
    an established digital-liquid-crystal product family; commercial opportunity
    cannot be equated with lack of existing solutions.

Additional relevant primary paper located through an author-uploaded manuscript:
[High-efficiency coupling for adjustable ring-mode laser optical switches (2025)](https://doi.org/10.1364/OE.572604).
The manuscript describes prescribed mirror layouts with optimized freeform coupling
optics and thermal/tolerance work. Publisher full text was not accessible in this scan.
This further weakens a broad claim that freeforms have not been applied to switches.

## Candidate gap and falsifiable experiment

Candidate question: under fixed ports, one bounded actuator and a tight package,
can non-sequential architecture search discover shared optical layouts with better
worst-state throughput/leakage/robustness tradeoffs than optimized conventional layouts?

This is an inference from the scan. No close example was identified here combining
our interaction-family exploration with that exact benchmark; absence from this
search is not evidence that none exists.

Protocol:

1. Build a known-solvable 1x2 moving-mirror baseline, then a 1x4 benchmark. Fix
   source, output ports, aperture/angle acceptance, materials, package and actuator
   stroke. Use one mechanically meaningful actuation variable and shared static
   geometry. Do not optimize an independent device for each state.
2. Compare an optimized conventional relay, matched-budget multi-start CMA,
   and interaction-diverse search with local refinement. Match physical degrees of
   freedom and manufacturing constraints where feasible; disclose remaining differences.
3. Maximize the minimum desired-port efficiency across states and perturbations;
   constrain power into each wrong port, return power, collision, aperture clipping,
   surface slopes/curvatures and actuator travel. Report tradeoffs rather than hiding
   poor states in an average. Include a feasible finite beam, not only a chief ray.
4. Audit with denser input sampling and lower branch-pruning thresholds. Unresolved
   energy must be below the claimed leakage threshold; a zero detected ghost does
   not establish zero crosstalk. Inspect intermediate actuator positions as well as
   endpoints for collisions and transient routing.
5. Evaluate multiple seeds and held-out port arrangements/tolerances. Archive by
   multi-state interaction signatures and useful physical tradeoffs. Different paths
   alone are not a win: demonstrate a better constrained performance frontier or
   lower cost to find acceptable layouts.
6. Upgrade successful candidates to 3D and Gaussian/physical-optics coupling checks
   before claiming fiber insertion loss or application readiness. Dense same-engine
   audits are not independent physical validation.

Example definitions: geometric desired-port efficiency is accepted desired-port
power divided by launched power. Its -10 log10 is geometric loss, not certified
single-mode insertion loss. State crosstalk can be reported as 10 log10 of each
wrong-port power divided by desired-port power, with the denominator stated.
Use absolute leakage limits too, so a nearly-dark device cannot pass by ratios.

Proposed decision rule: continue if the discovered architecture improves a useful
constraint tradeoff reproducibly and survives improved optical fidelity. If
conventional designs match it, this may still be a useful verification/learning
environment, but not evidence of an architecture-design advantage.

For the learning track, perturb actuator zero, pose, surface shape and port location;
learn a repair or calibration strategy across held-out physical design families.
Compare against finite-difference/local optimization and a fixed repair curriculum.
The existing four-task learning pilot does not establish transfer to switches.
