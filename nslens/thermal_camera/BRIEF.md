# Fraunhofer flat thermal camera: computational reconstruction brief

Target confirmed by the user on 2026-09-30: the **2024 flat, eight-channel
automotive camera**, not the 2015 IG4 monolithic night-vision objective.

## Public evidence

| Source | Published facts |
| --- | --- |
| [Fraunhofer IOF, 10 October 2024](https://www.iof.fraunhofer.de/en/pressrelease/2024/new-infrared-camera-to-increase-safety-in-autonomous-driving.html) | 2 x 4 uncooled bolometer channels; 8–14 micrometres; approximately 530 x 210 stitched pixels; 34 x 13 degree combined field; F/1.1; 16 pixels/degree; 10 mm flat construction; mirrors and prisms; wafer fabrication with Fraunhofer ISIT. |
| [Hubold et al., SPIE 2025, DOI 10.1117/12.3069616](https://publica.fraunhofer.de/entities/publication/6f58d01e-92bc-4e7e-a626-efdb73948c2a) | Two-mirror Schmidt–Cassegrain optical architecture; 2 x 4 channels; 524 x 207 stitched pixels; 34 x 13 degrees; 16 pixels/degree; **effective F/1.4**. Only the abstract was accessible in the primary repository. |
| [Related Fraunhofer patent, WO2022200331A1](https://patents.google.com/patent/WO2022200331A1/en) | Schematic multichannel embodiments with refractive correction/steering elements, reflective optics, and encapsulated sensors. Silicon, germanium and chalcogenides are listed as possible correction materials. The patent's illustrated single-mirror embodiments do not establish the realized camera's two-mirror prescription. |

F/1.1 and effective F/1.4 are recorded separately. Obscuration could explain
part of this difference, but the accessible sources do not establish its cause.
No public numerical surface prescription, sensor pitch, channel overlap,
aperture dimensions, coating efficiencies, or complete mechanical envelope
was recovered. This task searches for a candidate in the published optical
family; it cannot certify an exact reproduction of an undisclosed prescription.

## Explicit pilot assumptions

- 12 micrometre detector pitch, giving a target paraxial focal length of
  `16 * 0.012 * 180/pi = 11.0007897 mm`. This pitch is our assumption.
- Nominal F/1.1 sets a 10.000718 mm entrance-pupil diameter at that focal length.
- Identical channels cover approximately 9 x 7 degrees each. Channel centers
  are horizontally -12.75, -4.25, 4.25, 12.75 degrees and vertically +/-3.25
  degrees, with nominal overlap. These tiling choices are not measured hardware.
- The initial channel uses two powered mirrors and a thin refractive corrector.
  The corrector material is an assumed constant-index silicon approximation.
  Repeating geometrical rays at 8, 11, and 14 micrometres does not validate
  dispersion, absorption, coatings, diffraction, or thermal performance.
- 10 mm is the target depth. An optical-surface envelope is not a packaged
  camera thickness: detector package, prism, mounts and electronics also matter.

## Experiment and interpretation

Reuse `nslens.search.cma_me.CMAES`, as in the spectrometer domain. Independent
seed streams optimize a bounded base representation and then release higher
surface terms. This is architecture-conditioned scalar CMA, not a claim that
CMA discovered the Schmidt–Cassegrain topology or recovered Fraunhofer's design.
No published optimized surface vector initializes the search.

Keep every launched entrance-pupil ray in the throughput denominator,
including blocked rays. Record per-field support, spot size, image scale,
central obscuration, primary-hole clearance, and sampled depth. Final dense
sampling is an audit using the same tracer, not independent optical validation.
Compare start and final candidates under the same audit sampling.

Minimum deliverables: executable model and physics tests; actual CMA runs;
saved genes and decoded surfaces; convergence and optical-layout plots;
eight-channel coverage assessment; explicit pass/fail gates and remaining gaps.
Useful partial results must not be labelled a completed physical camera.
