# Related Literature: Automated Freeform Optical Design and Solution-Space Search

_Compiled from the Optics project discussion; updated 3 September 2026._

## Scope and evidence standard

This file records the prior work discussed in the project and, most importantly, distinguishes **surface design**, **starting-point generation**, **fold-family search**, and **interaction-topology discovery**. It is a literature map, not the separate data-generation audit requested for File 2.

Claims below are tied to a primary paper or publisher page whenever one was available. A dagger (†) marks an item for which only limited public metadata or an abstract was available during compilation; detailed methodological claims are intentionally omitted in those cases.

## Executive taxonomy

| Work | What is supplied or fixed | What the method discovers or improves | Best classification |
|---|---|---|---|
| Yang, Jin & Zhu (2017) | Initial planar layout, fold type, stop location, object–image mapping | Freeform surface shapes and nearby pose changes | Automated construction/refinement within a designer-selected fold |
| Bauer, Schiesser & Rolland (2018) | Three-mirror class and candidate first-order geometries | Which geometry benefits from freeforms, then which Zernike terms to add | Aberration-guided geometry selection and design |
| Papa, Howard & Rolland (2019)† | Freeform-imager class and aberration-theory model | Reduced-DOF starting designs across a solution space | Aberration-theory solution-space exploration |
| Wu et al. (2020)† | Planar starting system | Rapid construction plus aberration correction | Faster automated refinement of a fixed fold |
| Zhang, Jin & Zhu (2021) | Three sequential mirrors and design constraints | Optical-power distributions, folded chief-ray geometries, and freeform prescriptions | Coarse/fine enumeration of the three-mirror design space |
| Chen et al. (2021) | Demonstrated system family and parameterization | Neural-network starting points from design requirements | Supervised starting-point generation |
| Nie et al. (2023) | Differentiable parameterized topology | Surface/system parameters optimized through differentiable ray tracing | Physics-in-the-loop learning/optimization |
| Mao et al. (2023), FreeformNet | Learned system/fold family and its parameterization | Fast multiple candidate prescriptions conditioned on requirements | Multi-solution learned starting-point generation |
| Chen et al. (2025), ODMINN | Three-mirror system and eight encoded folding geometries | Candidates across those predefined fold classes | Physics-informed multi-fold generation |
| Chen et al. (2026), cGAN | System parameters, folding-geometry parameters, and model parameterization | Multiple candidates through conditional generation and random latent inputs | One-to-many generative starting-point generation |
| Su & Romanova (2026), GA | Freeform-prism AR system class and parameter bounds | A ray-traceable structure, then improved image quality | Two-stage evolutionary optimization, not topology-free learning |
| Chen et al. (2026), performance-aware DNN† | Parameterized freeform system/fold families | Candidate design plus predicted performance | Learned generation with performance screening |

The core distinction is that **none of these methods is demonstrated as discovering an unconstrained ray–surface interaction sequence**. Several search spatial folding geometries; that is not the same as letting the traced geometry determine an emergent sequence such as repeated or non-monotonic surface visits.

---

## 1. Yang, Jin & Zhu (2017): the original automated construction–iteration framework

**Paper:** Tong Yang, Guofan Jin, and Jun Zhu, “Automated design of freeform imaging systems,” _Light: Science & Applications_ 6, e17081 (2017). [Paper and DOI](https://doi.org/10.1038/lsa.2017.81)

### What it does

The designer supplies an **initial planar optical system**, desired object–image relationships, and structural constraints. Feature rays are sampled across field and pupil. The method constructs discrete surface points and normals so that the rays are redirected toward their target image points, fits polynomial freeform surfaces to those point/normal data, and iteratively corrects the resulting system.

The paper combines:

1. **Construction–Iteration (CI):** point-by-point construction and negative-feedback correction of freeform surfaces.
2. **Single-Freedom Search (SFS):** sequential perturbation of selected positions and tilts, with the iteration stage rerun for each perturbation.
3. **HPIQ mode:** later iterations prioritize convergence of rays associated with a field around a common image location, improving image quality while allowing distortion tradeoffs.

This is a genuine automated construction-and-refinement method rather than ordinary optimization of freeform coefficients. The demonstrated systems reach high image quality starting from simple planes. [Method and examples](https://www.nature.com/articles/lsa201781)

### What it does **not** discover

The initial planes must already lie approximately where the final mirrors will be and must redirect light in approximately the desired way. The paper explicitly says that the folding-geometry type is determined before design, is maintained by constraints, and cannot change into a different folding geometry. The aperture-stop location is also fixed. [Specified folding-geometry limitation](https://www.nature.com/articles/lsa201781)

Therefore the correct summary is:

\[
\text{given planar fold} \rightarrow \text{construct freeforms} \rightarrow \text{iterate} \rightarrow \text{local pose search}.
\]

It is not a general fold-discovery or interaction-sequence-discovery engine.

---

## 2. Bauer, Schiesser & Rolland (2018): geometry selection through aberration fields

**Paper:** Aaron Bauer, Eric M. Schiesser, and Jannick P. Rolland, “Starting geometry creation and design method for freeform optics,” _Nature Communications_ 9, 1756 (2018). [Paper and DOI](https://doi.org/10.1038/s41467-018-04186-9)

This is the **aberration/geometry paper** discussed in the project. It should not be confused with Zhang et al.’s later (P\rightarrow C\rightarrow F) grid search.

### What it does

The authors use freeform-surface aberration theory and Zernike-based full-field displays to evaluate whether a candidate unobscured three-mirror geometry produces aberration field dependencies that available freeform terms can actually correct. They first create spherical, first-order candidate layouts, then down-select based on:

- the correctability of the large tilt-induced rotationally variant aberrations;
- mechanical/package constraints; and
- the freeform departure and slope likely to be required.

After selecting a promising geometry, they add only the freeform terms needed to target the observed limiting aberrations, alternating controlled surface-form additions with optimization of radii, airspaces, and tilts. The paper’s three-mirror example compares two geometries and reports at least a 16× performance difference, demonstrating that starting geometry strongly determines how much benefit freeform surfaces can provide. [Geometry down-selection and aberration-guided design](https://www.nature.com/articles/s41467-018-04186-9)

### Why it matters

This work addresses a question that the 2021 coarse/fine paper largely leaves to computation:

> Before paying for full refinement, does the candidate geometry have an aberration structure that is compatible with correction by the available freeform modes?

However, it remains a theory-guided study of candidate three-mirror geometries, not an automatic search over arbitrary hit sequences. Its main value for the present project is the idea that **refinement potential depends on the structure of the aberration field, not merely on the candidate’s current RMS error**.

---

## 3. Papa, Howard & Rolland (2019): automatic solution-space exploration†

**Paper:** Jonathan C. Papa, Joseph M. Howard, and Jannick P. Rolland, “Automatic Solution Space Exploration for Freeform Optical Design,” _Optical Design and Fabrication 2019_, paper FM4B.1. [Publisher record](https://doi.org/10.1364/FREEFORM.2019.FM4B.1)

The accessible conference abstract states that first-, second-, and third-order aberration theory are used to reduce the degrees of freedom, generate starting designs, and support a complete search of the freeform-imager solution space. [Abstract](https://opg.optica.org/abstract.cfm?uri=Freeform-2019-FM4B.1)

Because only the short abstract was available during this compilation, this file does not attribute a more detailed sampling or optimization pipeline to it. It belongs to the Rolland-group line of **aberration-theory-guided solution-space mapping**, distinct from the Tsinghua (P,C,F) enumeration.

---

## 4. Wu et al. (2020): fast construction and correction†

**Paper:** Weichen Wu, Hui Wang, Guofan Jin, and Jun Zhu, “Fast automatic design method for freeform imaging systems through system construction and correction,” _Optics Letters_ 45, 5140–5143 (2020). [Paper and DOI](https://doi.org/10.1364/OL.398924)

The method takes a **plane system as input** and combines direct system construction with an aberration-based correction stage. Its purpose is to retain the automated nature of the 2017 framework while reducing design time. The publisher abstract supports the construction/correction split, but the full methodological details were not re-audited here; those belong in the source-specific data-generation file. [Publisher record](https://opg.optica.org/ol/upcoming_pdf.cfm?id=398924)

Its structural limitation is similar to the 2017 method: it automates conversion and correction within a supplied planar fold rather than inventing a fundamentally different ray path.

---

## 5. Zhang, Jin & Zhu (2021): coarse and fine search of the three-mirror solution space

**Paper:** Benqi Zhang, Guofan Jin, and Jun Zhu, “Towards automatic freeform optics design: coarse and fine search of the three-mirror solution space,” _Light: Science & Applications_ 10, 65 (2021). [Paper and DOI](https://doi.org/10.1038/s41377-021-00510-z)

This is the main **(P\rightarrow C\rightarrow F)** solution-space paper.

### Phase structure

The pipeline is:

\[
\text{specifications and constraints}
\rightarrow \{P_m\}
\rightarrow \{\widetilde C_{m,t}\}
\rightarrow \{\widetilde F_{m,t}\}
\rightarrow \text{point-by-point refinement}
\rightarrow \text{performance filtering}.
\]

1. **Optical-power distributions (P):** enumerate coaxial spherical three-mirror systems with different mirror radii/powers and separations while satisfying first-order requirements such as focal length.
2. **Folded structures (C):** for each (P_m), vary chief-ray deflection angles to produce many off-axis spatial foldings of the same ordered three-mirror path. Trace and reject obscured or constraint-violating layouts.
3. **Freeform systems (F):** construct freeform surfaces for each surviving off-axis candidate and correct the total system power.
4. **Refinement:** iteratively recalculate surface shapes and optimize image-plane tilt to improve ray-intercept behavior.
5. **Evaluation:** calculate image-quality metrics and output candidates meeting the threshold. CODE V is used for final imaging-quality evaluation, while the authors implement the main ray tracing and surface calculations independently. [Five-phase method](https://www.nature.com/articles/s41377-021-00510-z)

### Coarse versus fine search

The coarse stage uses lower-density grid sampling to obtain systems spread across optical-power and structural space. A selected high-performing region is then resampled at higher density as a localized fine search. This is **denser enumeration around a promising region**, not ordinary local coefficient optimization.

### How plausible candidates are found

The paper does not first learn or analytically predict “refinement feasibility.” It filters first-order and geometric invalidity, then actually constructs and refines the surviving candidates. The authors explicitly identify aberration-theory guidance for the optical-power distribution, structure, and stop location as future work, confirming that this guidance is not the main candidate-selection mechanism in the presented pipeline. [Discussion of future aberration guidance](https://www.nature.com/articles/s41377-021-00510-z)

### Structural boundary

The interaction order remains:

\[
\text{object}\rightarrow M_1\rightarrow M_2\rightarrow M_3\rightarrow\text{image}.
\]

The search changes the **power allocation and spatial embedding** of this known sequence. It does not allow geometry to produce an emergent sequence with repeated, skipped, or reordered mirror interactions.

---

## 6. Chen et al. (2021): deep-learning starting-point generation

**Paper:** Wenchen Chen, Tong Yang, Dewen Cheng, and Yongtian Wang, “Generating starting points for designing freeform imaging optical systems based on deep learning,” _Optics Express_ 29, 27845–27870 (2021). [Publisher record](https://opg.optica.org/oe/abstract.cfm?uri=oe-29-17-27845)

This work proposes a generalized supervised-learning framework that maps design requirements/system descriptors to freeform starting points. It demonstrates that automatically generated optical systems can be used as training targets, after which a network can rapidly propose prescriptions for subsequent optical optimization.

The key conceptual change from Zhang 2021 is:

\[
\text{expensive automated design repeated online}
\quad\longrightarrow\quad
\text{learn that mapping offline and infer starting points quickly}.
\]

The output is still best understood as a **starting point within an encoded system family**, not a finished design with unknown topology.

---

## 7. Nie et al. (2023): differentiable 3-D ray tracing and unsupervised learning

**Paper:** Yunfeng Nie, Jingang Zhang, Runmu Su, and Heidi Ottevaere, “Freeform optical system design with differentiable three-dimensional ray tracing and unsupervised learning,” _Optics Express_ 31, 7450–7465 (2023). [Paper and DOI](https://doi.org/10.1364/OE.484531)

The authors embed a generic differentiable freeform ray-tracing module in the learning loop for off-axis, multi-surface freeform/aspheric systems. This reduces dependence on large labeled design corpora because image formation and optical losses can train the network directly. The trained system can infer many optical designs after one training run. [Abstract and metadata](https://pubmed.ncbi.nlm.nih.gov/36859875/)

This work is important because optical physics participates directly in optimization. Nevertheless, differentiability acts over a chosen representation and parameterized interaction structure; it does not by itself discover a discrete, unknown sequence of ray–surface contacts.

---

## 8. Mao et al. (2023): FreeformNet

**Paper:** Boyu Mao, Tong Yang, Huiming Xu, Wenchen Chen, Dewen Cheng, and Yongtian Wang, “FreeformNet: fast and automatic generation of multiple-solution freeform imaging systems enabled by deep learning,” _Photonics Research_ 11, 1408–1422 (2023). [Publisher record](https://opg.optica.org/prj/abstract.cfm?uri=prj-11-8-1408)

### What it adds

FreeformNet is trained on automatically generated and optimized freeform systems. Given design requirements, it produces candidate freeform imaging systems extremely quickly; the paper reports less than 0.003 s per generated system. Its feedback process expands the training set with optimized network outputs, improving the learned generator over multiple rounds. [Article](https://www.researching.cn/articles/OJd1f426e45f8cddf4)

It is therefore a major shift in **throughput**:

\[
\text{requirements}\rightarrow\text{learned candidate prescription}\rightarrow\text{short conventional refinement}.
\]

### Important interpretation

“Multiple solution” does not mean arbitrary topology discovery. The demonstrated mapping operates within a selected freeform system/folding family and parameterization. The network learns how to populate that family with strong candidates; it does not decide an unconstrained number or order of optical interactions.

---

## 9. Chen et al. (2025): ODMINN across multiple predefined folds

**Paper:** Guogen Chen, Tong Yang, Dewen Cheng, and Yongtian Wang, “Generating multiple-folding-geometry freeform reflective imaging systems based on optical design model-informed neural network,” _Optics & Laser Technology_ 191, 113322 (2025). [Paper and DOI](https://doi.org/10.1016/j.optlastec.2025.113322)

The optical-design-model-informed neural network (ODMINN) combines data-driven training with physics-informed loss computed through differentiable ray tracing. An automatic dataset-generation method and a fast light-obstruction evaluation are used to cover freeform off-axis three-mirror systems. The demonstrations span all eight folding geometries defined by the authors. [Publisher/author abstract](https://pure.bit.edu.cn/en/publications/generating-multiple-folding-geometry-freeform-reflective-imaging-/)

ODMINN directly addresses a limitation of networks trained for only one fold. Its remaining boundary is that the eight folding geometries are an **encoded taxonomy** for a three-mirror system. It chooses/generates within those classes rather than discovering a new interaction graph.

---

## 10. Chen et al. (2026): conditional-GAN freeform design

**Paper:** Guogen Chen, Boyu Mao, Dewen Cheng, Yongtian Wang, and Tong Yang, “Freeform optical design with a generative AI model based on a conditional generative adversarial network,” _Advanced Photonics Nexus_ 5(4), 046023 (2026). [Paper and DOI](https://doi.org/10.1117/1.APN.5.4.046023)

### Core mechanism

The generator is conditioned on system parameters and folding-geometry parameters and also receives random noise. Generator and discriminator are trained adversarially, augmented with data-driven and optical-model physics-informed losses. After training, changing the noise input while holding the design conditions fixed can yield different candidate prescriptions:

\[
G(\Phi,\Psi,z_i)\rightarrow X_i,
\]

where (\Phi) denotes system requirements, (\Psi) the encoded folding geometry, and (z_i) a latent noise vector. The paper describes the results as good starting points for rapid subsequent optimization. [Publisher abstract](https://www.spiedigitallibrary.org/journals/advanced-photonics-nexus/volume-5/issue-04/046023/Freeform-optical-design-with-a-generative-AI-model-based-on/10.1117/1.APN.5.4.046023.short)

### What is genuinely new

Compared with deterministic supervised regression, the cGAN explicitly models a **one-to-many conditional design distribution**. That is useful when several materially different prescriptions satisfy the same requirements.

### What remains fixed

The conditioning already includes a folding-geometry representation, and the generated systems live inside the system class and parameterization represented in training. Random latent variation produces diversity within that learned support; it is not evidence of discovering a wholly new ray-interaction topology.

---

## 11. Su & Romanova (2026): two-stage genetic algorithm for a freeform-prism AR system

**Paper:** Guanqiao Su and Galina Romanova, “Two-stage genetic algorithm optimization of a freeform prism AR system design,” _Applied Optics_ 65, 5158–5166 (2026). [Paper and DOI](https://doi.org/10.1364/AO.599617)

The first GA stage searches for a ray-traceable structure and roughly optimizes it. The second stage uses a multi-objective GA to improve image quality. The workflow is implemented in MATLAB and tested on a freeform-prism augmented-reality system. [Publisher abstract](https://opg.optica.org/ao/abstract.cfm?uri=ao-65-15-5158)

This is an **optimizer**, not a learned data-driven generator. Its evolving populations could be logged as a dataset, but the paper’s primary objective is to solve the AR-prism design problem through staged evolutionary search. The freeform-prism architecture and its bounded parameterization are supplied; the GA is not shown inventing arbitrary optical interaction sequences.

---

## 12. Chen et al. (2026): performance-aware DNN†

**Paper:** Guogen Chen, Tong Yang, Dewen Cheng, and Yongtian Wang, “Design of freeform imaging systems using a performance-aware deep neural network,” _Applied Optics_ (2026), DOI 10.1364/AO.607567. [DOI](https://doi.org/10.1364/AO.607567)

This paper was discussed as a particularly relevant extension because it adds an estimate of candidate performance to learned system generation, potentially supporting screening before full optimization. Only limited public metadata was available during this compilation, so exact architecture, labels, training-set construction, and demonstrated fold coverage are deliberately deferred to File 2 unless the full primary paper is available.

---

## Relationship among the main lines of work

### Construction line

Yang 2017 and Wu 2020 answer:

> Given an approximate fold, can freeform surfaces be constructed and corrected automatically?

They substantially automate surface creation, but the designer still supplies the fold family.

### Aberration-guided geometry line

Bauer 2018 and Papa 2019 answer:

> Which geometry is compatible with correction by the available freeform modes, and how can aberration theory reduce the search?

This line provides physical insight into refinement potential but searches structured candidate geometries.

### Enumerative solution-space line

Zhang 2021 answers:

> If optical-power distributions and chief-ray folds are densely enumerated, which three-mirror systems survive construction and refinement?

It yields a catalogue across (P\times C), but the sequential three-mirror interaction order is fixed.

### Learned generation line

Chen 2021, FreeformNet, ODMINN, the 2026 cGAN, and the performance-aware DNN answer progressively broader versions of:

> Can a model learn the expensive mapping from requirements and encoded structure to one or more strong starting prescriptions, optionally using differentiable physics and performance prediction?

These methods improve speed, diversity, fold-family coverage, and screening. Their outputs remain tied to the topology and folding representation contained in their training/design model.

### Direct physics-optimization line

Nie 2023 and the GA paper answer:

> Can optical physics be placed directly inside a gradient-based or evolutionary search so that viable parameters are found with less dependence on hand-built starting designs?

They expand optimization capability, but their topology is still parameterized in advance.

## Bottom-line comparison to interaction-sequence discovery

The literature above spans increasingly automated design, but its structural assumptions remain strong:

- a fold may be supplied as planes;
- a finite set of fold classes may be encoded;
- chief-ray angles may spatially fold a fixed (M_1\rightarrow M_2\rightarrow M_3) sequence; or
- a neural/evolutionary optimizer may vary continuous parameters inside a fixed simulator graph.

The unfilled category discussed in this project is different: **bounded geometry is proposed, the actual ray trace determines the interaction sequence, and repeated/non-sequential surface contacts may emerge without being enumerated beforehand**. That gap—and the proposed inception, compaction, and refinement modules—belongs in File 3 rather than this literature-only file.

## References

1. Yang, T., Jin, G.-F. & Zhu, J. “Automated design of freeform imaging systems.” _Light: Science & Applications_ 6, e17081 (2017). https://doi.org/10.1038/lsa.2017.81
2. Bauer, A., Schiesser, E. M. & Rolland, J. P. “Starting geometry creation and design method for freeform optics.” _Nature Communications_ 9, 1756 (2018). https://doi.org/10.1038/s41467-018-04186-9
3. Papa, J. C., Howard, J. M. & Rolland, J. P. “Automatic Solution Space Exploration for Freeform Optical Design.” _Optical Design and Fabrication 2019_, FM4B.1. https://doi.org/10.1364/FREEFORM.2019.FM4B.1
4. Wu, W., Wang, H., Jin, G. & Zhu, J. “Fast automatic design method for freeform imaging systems through system construction and correction.” _Optics Letters_ 45, 5140–5143 (2020). https://doi.org/10.1364/OL.398924
5. Zhang, B., Jin, G. & Zhu, J. “Towards automatic freeform optics design: coarse and fine search of the three-mirror solution space.” _Light: Science & Applications_ 10, 65 (2021). https://doi.org/10.1038/s41377-021-00510-z
6. Chen, W., Yang, T., Cheng, D. & Wang, Y. “Generating starting points for designing freeform imaging optical systems based on deep learning.” _Optics Express_ 29, 27845–27870 (2021). https://opg.optica.org/oe/abstract.cfm?uri=oe-29-17-27845
7. Nie, Y., Zhang, J., Su, R. & Ottevaere, H. “Freeform optical system design with differentiable three-dimensional ray tracing and unsupervised learning.” _Optics Express_ 31, 7450–7465 (2023). https://doi.org/10.1364/OE.484531
8. Mao, B., Yang, T., Xu, H., Chen, W., Cheng, D. & Wang, Y. “FreeformNet: fast and automatic generation of multiple-solution freeform imaging systems enabled by deep learning.” _Photonics Research_ 11, 1408–1422 (2023). https://opg.optica.org/prj/abstract.cfm?uri=prj-11-8-1408
9. Chen, G., Yang, T., Cheng, D. & Wang, Y. “Generating multiple-folding-geometry freeform reflective imaging systems based on optical design model-informed neural network.” _Optics & Laser Technology_ 191, 113322 (2025). https://doi.org/10.1016/j.optlastec.2025.113322
10. Chen, G., Mao, B., Cheng, D., Wang, Y. & Yang, T. “Freeform optical design with a generative AI model based on a conditional generative adversarial network.” _Advanced Photonics Nexus_ 5(4), 046023 (2026). https://doi.org/10.1117/1.APN.5.4.046023
11. Su, G. & Romanova, G. “Two-stage genetic algorithm optimization of a freeform prism AR system design.” _Applied Optics_ 65, 5158–5166 (2026). https://doi.org/10.1364/AO.599617
12. Chen, G., Yang, T., Cheng, D. & Wang, Y. “Design of freeform imaging systems using a performance-aware deep neural network.” _Applied Optics_ (2026). https://doi.org/10.1364/AO.607567
