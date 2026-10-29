# Optical application opportunities — 29 September 2026

Research assessment based on project summaries and public literature/product documentation. Rankings are engineering judgments, not demonstrated commercial demand or findings of patent novelty. No new optical simulations or customer interviews were performed. Some publisher material was accessible only as abstracts; reported performance belongs to its original configuration and is not directly comparable across applications.

## Recommendation

Start with a **thin folded photodiode receiver for a mechanically constrained industrial optical sensor**. Run a compact fluorescence collector as the second candidate if a cartridge/instrument partner is available. A custom multipass gas cell is the strongest repeated-path showcase, but needs a different optical embodiment and stronger beam-propagation modeling.

The commercial hypothesis is that topology discovery can meet a customer's fixed aperture, PCB, obstruction, and housing constraints with better collection, fewer aligned parts, or less thickness. The deliverable should be a manufactured, measured optical module and a reproducible design workflow.

## What the project evidence supports

The repository describes non-sequential tracing, typed interaction records, topology-diverse search, physical gates, guarded refinement, and downstream export. That combination is relevant when the order and reuse of physical surfaces are part of the design problem.

The saved September inventory reports two concentration candidates passing spot-refinement checks at the reported 4-degree setting, with approximately 91% throughput and 0.252/0.336 mm stress RMS. Those figures are task-specific simulation results, not demonstrated 3D collection efficiency or hardware measurements. The capability study also shows throughput/blur tradeoffs and dense-grid failures. Use these to motivate finite-detector collection experiments rather than to claim camera-grade performance. Sources: [run inventory](RUN_HISTORY_20260911_20260918.md), [capability findings](runs/optical_capability_map_20260916/FINDINGS.md).

Do not assume that a 2D extrusion retains its collection efficiency under a full 3D angular source. Establish a 3D verification path before commissioning precision optics. Audit wavelength-dependent index, transmission, Fresnel losses, coatings, detector angular response, and surface tolerances for each embodiment.

## Application screen

| Application | Mechanism fit | First-demo suitability | Plausible customer-specific gap | Main obstacle |
|---|---|---|---|---|
| Folded photodiode receiver for industrial sensing or optical wireless | High | High | Side-facing aperture feeding a PCB-mounted detector within a thin obstructed housing | Existing TIR concentrators; acceptance/concentration and background tradeoffs |
| Fluorescence collector around a capillary or cartridge | High | Medium-high with partner | Collection and routing around fluidic ports, lower alignment burden, excitation rejection | Extended 3D emission, filtering, autofluorescence, established collectors |
| Multipass gas cell | Very high for repeated paths | Medium | Unusual package, fixed ports, low sample volume, multiple path lengths | Existing optimized cells; coating loss, beam growth, tolerance, gas exchange |
| Machine-vision illumination around a gripper/nozzle | High | High for a basic demo | Obstructed access and specified illumination angles | Mature commercial design tools; need actual defect-contrast benefit |
| Folded miniature spectrometer | High | Medium-low | Particular bandwidth/resolution/throughput and detector-placement combination | Mature compact products; diffraction and spectral calibration |
| Industrial side-view probe/borescope | High | Medium-low | Inspection access impossible for a simple mirror/prism tip | 3D MTF, field coverage, fabrication |
| Photonic chip/fiber coupling | High geometrically | Low without partner | Unusual port arrangement or alignment tolerance | Mode overlap and phase, fabrication, strong existing demonstrations |
| AR prism/display relay | High | Low for first product | Customer-specific relay/packaging problem | See-through path, eyebox, full-color MTF, ghosts, polarization |

## 1. Folded photodiode receiver

**Proposed use:** a narrow-angle industrial modulated-light receiver whose detector must remain flat on a PCB while light enters through a side/front window. Start with an amplitude sensor or optical link, not a complete LiDAR system.

**Existing benchmark:** a 2022 TIR receiver study demonstrated concentration gain of 44.7 at field center, acceptance of ±5 degrees, and a 1.289 Gbit/s experimental link. Thus neither TIR concentration nor its use in communication is an open application category. [Primary paper](https://www.mdpi.com/2304-6732/9/5/276).

**Gap hypothesis:** simultaneously satisfy a fixed input window, offset detector, thin envelope, and a mechanical keep-out region. Discover whether a shared collection/folding element improves the complete receiver assembly relative to an optimized conventional arrangement.

**Demo:** choose one visible or near-IR wavelength compatible with available source, detector, and material; declare aperture, detector area, angular acceptance in both axes, and housing envelope. Compare a bare detector, a conventional lens plus fold mirror/prism, an appropriate CPC/TIR concentrator, and the discovered design. The bare detector is a sanity baseline, not the competitive baseline.

**Measurements:** received optical power versus two-axis incidence angle; useful power integrated over the required angular distribution; out-of-field/background response; assembly thickness and volume; lateral/angular alignment sensitivity. Add SNR or BER only with identical electronics and bandwidth. Count active detector area, not power near a freely recentered spot.

**Proposed advance criterion:** at least 30% less assembly thickness at matched useful throughput and angular coverage, or at least 1.5x useful detected power within the same constrained envelope, without worse background response or assembly tolerance. These are suggested decision thresholds, not predictions.

**Physical constraint:** passive concentration trades acceptance against concentration through conservation of étendue. Do not market simultaneous arbitrary aperture gain and broad acceptance. Angle diversity is an existing alternative and should be considered if wide coverage matters. [Receiver tradeoff study](https://www.research.ed.ac.uk/en/publications/design-tradeoffs-of-non-imaging-angle-diversity-receivers-for-6g-/).

## 2. Fluorescence collection module

**Proposed use:** a reusable collection optic around a disposable capillary/cartridge, routing fluorescence toward a fixed photodiode, fiber, or PMT while leaving fluidic and excitation access clear. Begin with dye or reference beads, not diagnostic claims.

**Existing work:** integrated fiber collection remains a practical approach in a 2024 droplet-microfluidics demonstration. TIR-based fluorescence collection is also established; an EPO patent document explicitly covers light collection using TIR. This search therefore does not establish novelty of a generic TIR collector. [2024 experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933342/), [EPO document](https://data.epo.org/publication-server/rest/v1.2/publication-dates/20231004/patents/EP2989457NWB1/document.pdf).

**Gap hypothesis:** optimize collection for a particular cartridge geometry and detector interface while constraining excitation leakage, interchannel crosstalk, and repeat-insertion sensitivity. Multiple useful ray paths are natural here; requiring one dominant signature could unnecessarily discard valuable light.

**Demo:** identical fluorescent sample, excitation, emission filter, detector, and electronics; compare with an optimized small collection lens or existing fiber collection arrangement. Measure detected signal, blank/background, SNR at fixed integration time, repeat-insertion variation, and sample-position sensitivity. More raw collected light is insufficient if excitation leakage rises too.

**Advance criterion:** a repeatable signal/SNR gain or meaningful package/alignment simplification at matched performance. A suggested target is 1.5x SNR, or removal of an alignment operation without loss of sensitivity. Use a 3D extended-source model, actual capillary/fluid refractive interfaces, filter angle dependence, and material autofluorescence screening.

## 3. Custom multipass gas cell

**Proposed use:** an OEM gas-analysis cell constrained by an unusual enclosure or prescribed source/detector ports, with sample volume and response time included in the objective.

**Existing benchmark:** genetic optimization already produced a four-mirror cell with 24 m optical path in 80 cubic centimeters in 2019. A 2023 NSGA-II-designed dual-channel cell demonstrated 27.6 m and 2.1 m paths in 233 cubic centimeters. Automated optimization and multiple path lengths are therefore already occupied ideas. Commercial multipass cells are available from Thorlabs. [2019 experiment](https://opg.optica.org/abstract.cfm?uri=cleo_europe-2019-ch_p_25), [2023 experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984412/), [commercial cell](https://www.thorlabs.com/thorproduct.cfm?partnumber=MGC1C%2FM-P01).

**Gap hypothesis:** jointly choose surface visitation, port location, mirror count, tolerance, and dead-volume-compatible geometry for a package the conventional cell does not fit. Compare against optimized Herriott/White-type solutions under the same constraints.

**Critical distinction:** the useful absorption path must pass through the sample gas. A long path inside solid glass does not provide the same benefit. This requires hollow reflective geometry, or a separate evanescent-sensing model and application.

**Demo:** first measure routing, exit-beam size, and transmission with a suitable alignment source; then demonstrate absorption at the actual operating wavelength. Evaluate gas interaction length, transmission, package and sample volumes, mirror alignment sensitivity, fringes, and gas-exchange response. A visible routing model alone does not establish IR sensor performance.

Include beam divergence/diffraction and loss accumulation. For illustration, 50 reflections at reflectivity 0.98 transmit only 0.98^50 ≈ 36% before other losses. Maximize information or signal quality under noise assumptions, not bounce count alone.

## Other directions and reasons to defer

**Machine-vision illumination:** a custom lightguide feeding angled light around a robotic gripper or dispensing nozzle offers a clear photographable demo. Evaluate uniformity, useful angular distribution, glare, thermal load, and defect contrast at equal electrical power. Existing suppliers already provide tailored freeform design; the proposed gap is obstruction-constrained routing. [Fraunhofer design capabilities](https://www.ilt.fraunhofer.de/content/dam/ilt/en/documents/brochure/b-design-freeform-optics.pdf), [manufacturer lighting guide](https://www.advancedillumination.com/wp-content/uploads/2023/05/A-Practical-Guide-to-Machine-Vision-Lighting-Second-Edition.pdf).

**Spectrometer:** shared/revisited surfaces fit the method, but compactness alone is insufficient. A 2025 paper already uses one freeform mirror for both collimation and imaging with a planar grating. Hamamatsu lists miniature products with 20.1 × 12.5 × 10.1 mm packages. Pursue a customer-specified spectral band, resolution, throughput, and footprint; extend the tracer for grating physics and calibrate spectral line spread. [2025 paper](https://www.sciencedirect.com/science/article/abs/pii/S0143816625000545), [manufacturer selection table](https://www.hamamatsu.com/jp/ja/product/photometry-systems/spectrometer/spectrometer-selection.html).

**Side-view imaging:** a 2016 manufactured endoscope already combined a freeform entrance surface with a two-reflection prism, doublet, and field lens. A niche industrial inspection probe is plausible, but the current collection results do not establish its MTF or image quality. [Primary demonstration](https://www.degruyterbrill.com/document/doi/10.1515/aot-2016-0031/html).

**Photonic packaging:** 3D-printed reflective freeform couplers and TIR interfaces already exist, including plug-and-play packaging research. This could become a valuable partnership direction after adding coherent mode-overlap evaluation and access to suitable fabrication. Geometric detector interception is not equivalent to single-mode coupling efficiency. [Reflective micro-optic research](https://arxiv.org/abs/2112.14357), [2025 packaging demonstration](https://pmc.ncbi.nlm.nih.gov/articles/PMC12467054/), [Nanoscribe manufacturing capabilities](https://www.nanoscribe.com/en/applications-markets/industrial-photonic-packaging/).

**AR:** technically aligned but requires substantially more than efficient folding. A fabricated freeform-prism HMD was published in 2022; 2026 research jointly optimizes non-sequential polarized transport and coatings in geometric waveguides. Treat a narrow relay/packaging contract as a more credible entry than a complete glasses promise. [Prism demonstration](https://pubmed.ncbi.nlm.nih.gov/36258613/), [2026 waveguide research](https://arxiv.org/abs/2601.04370).

## What would establish the method's contribution

Use both product and algorithm comparisons:

1. Optimized conventional optical layout under identical source, detector, material, packaging, and fabrication constraints.
2. Equal-budget multistart numerical search in the same geometry space.
3. Topology-diverse discovery plus guarded refinement, with compute and human intervention recorded.
4. Independent 3D verification, tolerance analysis, and eventual fabricated measurement.

For a routing task, score energy on fixed physical detector coordinates. For an illumination task, allow the ray crossings needed to form the desired distribution; imaging-specific no-caustic gates should not carry over blindly. For collectors and gas cells, preserve useful ensembles/path distributions rather than assuming every application must have one dominant image-forming signature.

Non-sequential optimization is already available in commercial workflows. Ansys documents lightguide optimization with Speos/optiSLang; LDG-EA research generates diverse lens solutions. The startup must demonstrate the practical value of discovering different interaction routes under customer constraints. [Ansys workflow](https://optics.ansys.com/hc/en-us/articles/9098002328595-Interior-Lightguide-Optimization), [LDG-EA](https://arxiv.org/abs/2601.22075).

## Immediate execution sequence

First obtain one real receiver envelope and a conventional reference from an optical-sensor team, or freeze a transparent internal benchmark if no partner is available. Establish the baseline and 3D verification path. Run discovery and equal-budget controls; carry only robust candidates into a fabrication quote. Commission hardware only after the predicted advantage exceeds expected measurement and fabrication uncertainty.

A good first presentation shows the same input window and detector, the baseline assembly, the discovered assembly, measured angular-response curves, and assembly sensitivity. The archive supports adaptation when the enclosure changes; the measured module establishes customer value.

Commercial interest remains untested. Proposed partner categories are industrial photoelectric sensor OEMs, compact optical-link teams, cartridge-reader developers, and gas-analyzer OEMs. The companies cited above are incumbent or capability examples, not confirmed prospects or collaborators.
