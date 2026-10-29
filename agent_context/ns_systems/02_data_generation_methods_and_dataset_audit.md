# Exact design- and dataset-generation processes in the discussed freeform-optics literature

## Scope and evidence rule

This document answers one question only:

> **How does each paper generate the optical systems that become its results or training data?**

It is **not** an audit of where datasets can be downloaded. For every described process, the relevant paper section, subsection, figure, equation, phase, or page is given. A literal `*` means that the exact process or locator could not be verified from an accessible primary full text and has therefore not been reconstructed from memory.

The word **generation** is used in two distinct senses:

- **Design generation:** an algorithm constructs/searches optical systems, even if no neural network is trained.
- **Dataset generation:** the generated systems are stored as input–target examples for ML.

## Fast locator table

| Paper | Kind of generation | Exact primary-paper locator |
|---|---|---|
| Yang, Jin & Zhu (2017) | Design generation from a supplied planar fold | **Materials and methods** → “Preparing the design inputs,” “Determining the shapes of the freeform surfaces,” “Determining the locations of the freeform surfaces,” and “Optional high priority for image quality iteration mode”; Figs. 2–4 |
| Bauer, Schiesser & Rolland (2018) | Aberration-guided starting-geometry generation/down-selection | **Results** → “Aberrations from freeform surfaces” and “Starting geometry creation and down-selection”; Fig. 1 and aberration FFDs |
| Papa, Howard & Rolland (2019) | Aberration-theory-reduced solution-space generation | **§§1–4**; Figs. 1–2 |
| Wu et al. (2020) | Fast design generation through construction and correction | Main text, p. 5141, paragraphs beginning “The idea that guides…” and “For the system construction…”; abstract |
| Zhang, Jin & Zhu (2021) | Enumerative design catalogue over power and fold | **Materials and methods**, “Phase 1” through “Phase 5”; Fig. 5; Eqs. (1)–(4) for first-order construction |
| Chen et al. (2021) | Supervised dataset generation | **§2.1.3**, “Generation of the primary dataset,” and **§2.1.4**, “Generation of the secondary dataset”; Figs. 3–4; Eqs. (3)–(4) |
| Nie et al. (2023) | Specification-set generation for unsupervised physics training | **§3.1**, “Four-mirror freeform optical system,” paragraphs “Dataset” and “Training,” Table 1; **§3.2**, “Aspheric eyepiece design,” paragraphs “Dataset” and “Training,” Table 3 |
| Mao et al. (2023), FreeformNet | Optimized supervised dataset plus feedback expansion | **§2.B**, “Fundamental Data Set Generation,” Fig. 2; **§3 design example**, pp. 1416–1417 |
| Chen et al. (2025), ODMINN | Dataset spanning predefined folding geometries | `*` beyond the primary abstract |
| Chen et al. (2026), cGAN | Conditional generative-model dataset | **§2.3**, “Dataset Generation,” pp. 046023-4–5, Fig. 3; **§3**, pp. 046023-8–9, Fig. 5 |
| Su & Romanova (2026), GA | Online GA population generation, not supervised-dataset generation | **§2**, pp. 5159–5161, Figs. 1–5; **§3.A–B**, pp. 5161–5164, Fig. 6, Table 1, Eqs. (6)–(12) |
| Chen et al. (2026), performance-aware DNN | Optimized supervised dataset with post-optimization performance labels | **§2.B**, pp. 8917–8918; **§3**, pp. 8919–8920, Fig. 3; discussion on p. 8924 |

---

## 1. Yang, Jin & Zhu (2017): original automated freeform construction

**Paper:** Tong Yang, Guo-Fan Jin, and Jun Zhu, “Automated design of freeform imaging systems.” [Primary full text](https://www.nature.com/articles/lsa201781) · [DOI](https://doi.org/10.1038/lsa.2017.81)

### Exact source locations

- **Materials and methods → “Preparing the design inputs”**: supplied planar system, fixed folding type, stop position, feature-ray sampling, target image points, and constraints.
- **Materials and methods → “Determining the shapes of the freeform surfaces”**: point-by-point Construction–Iteration (CI), nearest-ray ordering, Fermat/reflection calculation, fitting, and negative-feedback iteration.
- **Materials and methods → “Determining the locations of the freeform surfaces”**: Single-Freedom Search (SFS).
- **Materials and methods → “Optional high priority for image quality iteration mode”**: HPIQ target definition.
- **Figs. 2–3**: construction and iteration details; **Fig. 4**: complete design flowchart.

### Exact design-generation process

1. Supply an initial set of decentered/tilted planes, the intended object–image relationship, and structural constraints. The planes must already occupy approximately the desired final locations and reproduce the required fold type. The method does not generate a different folding geometry. The stop location is also fixed.  
   **Locator:** “Preparing the design inputs.”

2. Sample feature rays across field and pupil. For the polar grid example, the number of feature rays is
   \[
   K=N\,P\,M,
   \]
   with (N) pupil angles, (P) pupil coordinates per angle, and (M) sampled fields. Calculate an ideal target point on the image plane for every feature ray.  
   **Locator:** “Preparing the design inputs,” especially the paragraphs defining feature rays and (T_{i,\mathrm{ideal}}); Fig. 1.

3. Construct each unknown surface point-by-point. The first point is the first ray’s intersection with the initial plane. The nearest-ray algorithm determines the order in which subsequent feature rays are processed.  
   **Locator:** “Determining the shapes…,” preliminary construction stage; Fig. 2.

4. At each point, use the known incident ray and required target point to obtain the outgoing ray through Fermat’s principle; then use reflection/refraction law to solve the required surface normal. Repeat until all feature points and normals are known.  
   **Locator:** same subsection; Fig. 2(b–c).

5. Fit a freeform representation to **both** point coordinates and normals, replace the corresponding plane, and construct the remaining surfaces successively.  
   **Locator:** same subsection; Fig. 3(a–b).

6. Iterate. Ray trace through the current system, retain the actual ray–surface intersection coordinates, recompute the normals needed to reduce image-point error, refit the surfaces, and repeat. In negative-feedback mode,
   \[
   T_i=T_{i,\mathrm{ideal}}+\varepsilon
   \bigl(T_{i,\mathrm{ideal}}-T_i^*\bigr),\qquad \varepsilon>0.
   \]
   **Locator:** iteration-stage paragraphs in “Determining the shapes…”; Fig. 3(c–f).

7. Search geometry with SFS. Select one positional/tilt degree of freedom, assign discrete values around its current value, reject constraint-breaking perturbations, rerun CI/iteration for every survivor, and retain the system with minimum image-point RMS. Move to the next DOF and repeat the DOF cycle if necessary.  
   **Locator:** “Determining the locations…”; paragraphs defining BaseS#1, DOF#1, and the repeated SFS rounds.

8. Optionally switch to HPIQ late in the run: use the current chief-ray intercept of each field as the common target for that field’s rays, prioritizing spot concentration over exact ideal-image placement.  
   **Locator:** “Optional high priority…”

### What the method generates

It generates a refined optical prescription from a **designer-supplied fold**. It is not a training-data generator in the paper.

---

## 2. Bauer, Schiesser & Rolland (2018): generating and filtering geometries using aberration fields

**Paper:** Aaron Bauer, Eric M. Schiesser, and Jannick P. Rolland, “Starting geometry creation and design method for freeform optics.” [Primary full text](https://www.nature.com/articles/s41467-018-04186-9) · [DOI](https://doi.org/10.1038/s41467-018-04186-9)

### Exact source locations

- **Results → “Aberrations from freeform surfaces”**: predicted aberration fields for Zernike terms and full-field displays (FFDs).
- **Results → “Starting geometry creation and down-selection”**: candidate layout creation, filtering, and selection.
- **Fig. 1**: hierarchy/taxonomy of three-mirror geometries.
- Paragraphs immediately following the chosen geometry: flat-field conversion and controlled Zernike-term addition.

### Exact design-generation process

1. Choose a three-mirror power distribution; the worked case is positive–negative–positive, with the aperture stop at the primary.  
   **Locator:** Results, introductory paragraphs before Fig. 1.

2. Enumerate the distinct mirror-tilt-direction geometries consistent with the intended unobscured three-mirror imager; organize them into a geometry hierarchy.  
   **Locator:** “Starting geometry creation and down-selection”; Fig. 1.

3. For every potential geometry, create a **first-order unobscured layout using spherical surfaces**, keep the volume near the desired package, and minimize mirror tilts as far as practical.  
   **Locator:** “Starting geometry creation and down-selection,” paragraph beginning “For each potential geometry…”.

4. Calculate the aberration FFDs of each spherical geometry. The FFDs plot aberration magnitude and orientation across sampled field points.  
   **Locator:** “Aberrations from freeform surfaces” and “Starting geometry creation and down-selection.”

5. Compare the geometry’s rotationally variant aberration field dependences and magnitudes with the field patterns that the available Zernike freeform modes can produce at or away from the stop.  
   **Locator:** “Aberrations from freeform surfaces,” especially the explanation following Eq. (1).

6. Eliminate geometries whose aberrations are not readily correctable by the freeform modes, together with layouts that fail mechanical filters. Do **not** rank only by smallest total starting aberration; rank by correctability.  
   **Locator:** “Starting geometry creation and down-selection,” especially the paragraphs before and after the first-order-layout paragraph.

7. For the selected geometry, modify mirror radii to make a flat-field design while holding focal length. Then use the FFDs to choose the specific Zernike term and mirror to add next, alternating controlled term addition with optimization.  
   **Locator:** paragraph beginning “With a starting geometry chosen…” and the subsequent FFD-guided optimization sequence.

### What the method generates

It generates and down-selects **starting geometries**, using aberration correctability as the filter. It does not create an ML dataset in the paper.

---

## 3. Papa, Howard & Rolland (2019): automatic solution-space exploration

**Paper:** Jonathan C. Papa, Joseph M. Howard, and Jannick P. Rolland, “Automatic Solution Space Exploration for Freeform Optical Design.” [Local PDF](../paper_pdfs/papa_15_to_5_rolland.pdf) · [Publisher record](https://opg.optica.org/abstract.cfm?uri=Freeform-2019-FM4B.1) · [DOI](https://doi.org/10.1364/FREEFORM.2019.FM4B.1)

### Exact source locations

- **§1, “Introduction,” p. 1:** 15-to-5 reduction, Cartesian-reflector base surfaces, third-order-correcting aspheric caps, and subsequent XY-polynomial optimization.
- **§2, “Organization of Brute-Force Solution Space Survey,” p. 1:** fixed specifications, power/tilt constraints, configuration families, and searched variables.
- **§3, “Sorting and Visualization of Results,” pp. 1–2:** performance plots, obscuration test, and volume calculation; Figs. 1–2.
- **§4, “Summary,” p. 2:** scope of the three-mirror brute-force survey.

### Exact generation process

1. Restrict the survey to objects at infinity, set \(|\mathrm{EFL}|=100\ \mathrm{mm}\), and choose an entrance-pupil diameter and half field of view to set the etendue.
2. Represent every mirror initially as a **Cartesian reflector**: a confocal conic that is stigmatic for its base-ray object and image points.
3. Choose the three thicknesses after the mirrors. Solve the three mirror powers under the specified EFL and back focal length while balancing the powers for a flat field. The power solution has **two flat-field branches**.
4. Choose the first two mirror tilts and solve the last mirror tilt as a function of those tilts and the mirror powers so that second-order linear astigmatism is eliminated.
5. Enumerate the **12 first-order three-mirror configuration families**:
   \[
   2\ \text{EFL signs}
   \times 2\ \text{flat-field power solutions}
   \times 3\ \text{stop locations}=12.
   \]
6. Within each family, brute-force sample the resulting five-dimensional space: **three post-mirror thicknesses plus two mirror tilts**. The other ten of the original 15 starting-point parameters are fixed or solved by the preceding first-/second-order constraints.
7. Add aspheric sag caps centered on each mirror's base ray, not on the base conic's axis of rotational symmetry. Choose the cap combination to eliminate third-order aberrations.
8. Form an optimization starting point by varying only the **fourth- through sixth-order XY-polynomial coefficients**. Keep mirror powers, separations, and tilts fixed during this surface-shape optimization.
9. Evaluate and sort the surveyed systems. Plot optimized performance against metrics such as volume and unoptimized starting performance (Fig. 1). Mark a layout as obstructed when a surface lies inside a polygon filling the ray-occupied region between mirrors (Fig. 2(a)); calculate package volume from a 3D convex hull around the rays from the first surface onward (Fig. 2(b)).

### Reproduction details not stated in the supplied two-page paper

- Numeric bounds and grid intervals for the five search variables.
- The explicit equations used to solve the two flat-field power branches and the final mirror tilt.
- The coefficient values of the third-order-correcting caps and the detailed optimizer/merit function used for the fourth- through sixth-order terms.

The supplied PDF verifies the process above, but it is a two-page conference paper rather than a fully reproducible implementation report.

### What the method generates

It generates starting prescriptions and optimized surface-shape variants across a reduced three-mirror solution space. It is not an ML dataset-generation pipeline.

---

## 4. Wu et al. (2020): construction followed by alternating correction

**Paper:** Rengmao Wu et al., “Fast automatic design method for freeform imaging systems through system construction and correction.” [Primary publisher page/PDF](https://opg.optica.org/ol/upcoming_pdf.cfm?id=398924) · [DOI](https://doi.org/10.1364/OL.398924)

### Exact source locations

- **Abstract.**
- **Main text, p. 5141**, paragraphs beginning “The idea that guides the method…” and “For the system construction…”. The Letter does not use the detailed subsection structure of the longer papers.

### Exact design-generation process supported by those passages

1. Specify the optical requirements and supply an initial plane system whose planes occupy the desired structural locations and avoid obscuration when converted into freeforms.
2. Use Construction–Iteration as the direct construction method to convert the planes into an intermediate freeform system satisfying fundamental quantities such as focal length and field of view.
3. Enter system correction: alternate **image-plane correction** and **surface correction** iteratively until image quality reaches a high level.
4. Vary the object–image relationship or plane positions and repeat the end-to-end run to obtain systems with different structures.

### Unverified finer implementation detail

*

The paper explicitly says this method contains no exhaustive geometry search; multiple designs arise by repeating fast end-to-end runs with different inputs.

---

## 5. Zhang, Jin & Zhu (2021): generating systems over (P\times C)

**Paper:** Bin Zhang, Gang Jin, and Jun Zhu, “Towards automatic freeform optics design: coarse and fine search of the three-mirror solution space.” [Primary full text](https://www.nature.com/articles/s41377-021-00510-z) · [DOI](https://doi.org/10.1038/s41377-021-00510-z)

### Exact source location

The entire recipe is in **Materials and methods**, introduced by **Fig. 5**, and divided explicitly into **Phase 1–Phase 5**.

### Exact design-generation process

#### Phase 1 — generate coaxial optical-power distributions

1. Describe a coaxial spherical system by radii (r_i), separations (d_i), and refractive indices (n_i).
2. Use the reflection and translation matrices in **Eqs. (1)–(3)** and the focal-length relation in **Eq. (4)**.
3. Grid-sample selected radii and separations at intervals (\Delta r) and (\Delta d); solve/constrain the remaining parameters so that the focal-length requirement is met.
4. Retain the resulting coaxial spherical systems
   \[
   P_m=[r_1,r_2,r_3,d_1,d_2,d_3].
   \]

**Locator:** “Phase 1,” Eqs. (1)–(4), including the paragraphs on (\Delta r), (\Delta d), and EPD-scaled distance ranges.

#### Phase 2 — generate folding geometries for every (P_m)

1. For each coaxial (P_m), tilt and reposition the mirrors while preserving the central-field chief-ray incidence direction in object space.
2. Parameterize the fold by the chief-ray deflection sequence
   \[
   C=[\theta_1,\theta_2,\theta_3].
   \]
3. Grid-sample the deflection angles to enumerate noncoaxial systems.
4. Ray trace and remove obscured systems.
5. Group survivors into structural categories using the signs of the deviations (\Delta C) from the coaxial sequence.
6. Within each category, favor compact layouts by minimizing the magnitudes of the angle deviations; optionally reject systems exceeding a volume limit.

**Locator:** “Phase 2,” Fig. 6 and the paragraphs defining (C), (\Delta C), structural categories, compactness, and volume filtering.

#### Phase 3 — convert survivors into freeforms

1. For every surviving folded system (\tilde C_{m,t}), calculate feature data points from feature rays sampled over field and pupil.
2. Use the intended object–image mapping to determine target image coordinates.
3. Apply the nearest-ray algorithm; use Fermat’s principle to obtain the outgoing direction and reflection law to solve the surface normal.
4. Fit coordinates and normals with an XY polynomial up to sixth order.
5. Construct the mirrors in a fixed computational order, e.g. tertiary → secondary → primary, producing (\tilde F_{m,t}).

**Locator:** “Phase 3,” especially the paragraphs defining feature rays/data points and the point-by-point construction sequence.

#### Phase 4 — refine every constructed freeform system

1. Trace the feature rays to retain their current surface intersections.
2. Re-solve normals from the desired object–image relation and refit every mirror.
3. One iteration round recalculates all mirrors; repeat until the intercept-error metric (\sigma) falls below (\sigma_{\mathrm{itr}}) or the improvement rate
   \[
   \tau=|\sigma'-\sigma|/\sigma
   \]
   falls below (\tau_{\mathrm{itr}}).
4. Search image-plane tilt in one dimension around the current best value, refine each trial, and repeat around the new best until its own quality/stagnation threshold is met.

**Locator:** “Phase 4,” including the definitions of (\sigma), (\tau), (\beta), and the one-dimensional tilt-search intervals.

#### Phase 5 — label/filter final results

1. Calculate spot diameter, MTF, and field-dependent RMS wavefront error.
2. Present systems with average RMS WFE below (0.075\lambda) as successful outputs.

**Locator:** “Phase 5.”

### Coarse versus fine generation

- Low-density grids over the same solution-space variables produce the broad **coarse** catalogue.
- A selected result defines a localized region; higher-density grid sampling produces the **fine** catalogue.

**Locator:** **Discussion**, paragraphs defining low-density coarse search and high-density localized fine search; the reported localized example around L73-2.

### What the method generates

Each search point is a specific **optical-power distribution plus sequential fold**, followed by actual freeform construction and refinement. It can be recorded as a dataset, but the paper presents it as a solution-space map/catalogue.

---

## 6. Chen et al. (2021): primary dataset plus KNN-generated secondary dataset

**Paper:** Wenchen Chen, Tong Yang, Dewen Cheng, and Yongtian Wang, “Generating starting points for designing freeform imaging optical systems based on deep learning.” [Primary publisher page](https://opg.optica.org/oe/abstract.cfm?uri=oe-29-17-27845) · [DOI](https://doi.org/10.1364/OE.432745)

### Exact source locations

- **§2.1.3 — “Generation of the primary dataset”**; Fig. 3.
- **§2.1.4 — “Generation of the secondary dataset”**; Fig. 4; Eqs. (3)–(4).
- **Section 3 design examples** instantiate the method and give corpus sizes and constraints.

### Primary-dataset generation — §2.1.3

1. Define a system-parameter space (SPS) from representative system variables and design constraints.
2. Sample system-parameter vectors (SYSP_{p,i}) inside the SPS either uniformly or randomly.
3. Create and optimize one easy initial base system. Its initial configuration may come from a PDE, SMS, or CI construction, after which constraints control specifications, structure/obscuration, distortion, and other aberrations.
4. For every ungenerated parameter vector, calculate its weighted Euclidean distance to generated systems. Select the next target vector and the already-generated base system closest to it.
5. Evolve/optimize that nearest system to the new target specification.
6. If the resulting system has fatal ray-tracing errors or poor performance, retry from the next-nearest existing system or discard the bad sample.
7. Repeat until the required base systems have been created. Store system parameters as inputs and optimized surface parameters as outputs.

The weighted distance is described in §2.1.3 as

\[
D=\left\|\boldsymbol\upsilon\odot(SYSP^*-SYSP^{**})\right\|_2.
\]

### Secondary-dataset generation — §2.1.4

1. Randomly draw (N_s) additional system-parameter combinations (SYSP_{s,j}) inside the same SPS.
2. For each new vector, find the (K) primary systems with minimum weighted distances.  
   **Locator:** §2.1.4, Eq. (3), Fig. 4.
3. Compute an initial surface-parameter vector as the distance-weighted average of the (K) neighbors:
   \[
   SURP^{\mathrm{initial}}_{s,j}=
   \frac{\sum_{m=1}^{K}W_{j,m}SURP^{*,j}_{p,m}}
        {\sum_{m=1}^{K}W_{j,m}},\qquad W_{j,m}=1/D_{j,m}.
   \]
   **Locator:** §2.1.4, Eq. (4).
4. Optimize all secondary systems from those interpolated prescriptions. Unlike primary evolution, these optimizations can run in parallel.
5. Use the same constraint types as in primary generation; remove systems with fatal errors or very poor performance.
6. Concatenate primary and secondary sets into the final supervised dataset:
   \[
   \text{input}=SYSP,\qquad \text{target}=SURP.
   \]

### Verified worked-example counts

- Off-axis three-mirror example: (N_p=8{,}457) primary systems and (N_s=8{,}414) secondary systems.  
  **Locator:** Section 3, three-mirror design example, paragraph reporting the primary generation time and following secondary-generation paragraph.
- Four-mirror afocal example: (N_p=7{,}056), (N_s=7{,}314), 14,370 total.  
  **Locator:** Section 3, four-mirror example, paragraph after Eq. (13).
- Exact prism-example counts: `*`

---

## 7. Nie et al. (2023): generate specifications, not prescription labels

**Paper:** Yunfeng Nie et al., “Freeform optical system design with differentiable three-dimensional ray tracing and unsupervised learning.” [Primary full text](https://opg.optica.org/oe/viewmedia.cfm?compressed=1&seq=0&uri=oe-31-5-7450) · [DOI](https://doi.org/10.1364/OE.484531)

### Exact source locations

- **§3.1 — “Four-mirror freeform (FMF) optical system”**, paragraphs headed “Dataset” and “Training”; Table 1.
- **§3.2 — “Aspheric eyepiece design”**, paragraphs headed “Dataset” and “Training”; Table 3.
- **§2.1–§2.2** supplies differentiable ray tracing and unsupervised loss definitions; Eqs. (12)–(19).

### FMF specification-set generation — §3.1 “Dataset”

1. Fix effective focal length at 250 mm.
2. Sample EPD from 50 to 125 mm in 2.5 mm increments.
3. Sample maximum x-FOV from 4° to 8° in 0.5° increments.
4. Sample maximum y-FOV from 4° to 20° in 0.5° increments.
5. Take the Cartesian product, producing 2,635 specifications.
6. Randomly split them into 2,028 training and 607 validation specifications.  
   **Locator:** §3.1 “Training.”
7. Do **not** attach an optimized prescription label to every specification. The network outputs 36 mirror-position/freeform parameters; differentiable ray tracing scores them directly using spot, distortion, missing-ray, and overlap losses.
8. Use ten variants of one prior design only to establish practical output ranges/scales, not as 2,635 ground-truth targets.

### Eyepiece specification-set generation — §3.2 “Dataset”

1. Fix image height at 18 mm.
2. Sample EPD from 4 to 6 mm in 0.5 mm increments.
3. Sample FOV from 36° to 48° in 2° increments.
4. Combine those values with three plastics for the two aspheres, five crown-glass choices, and five flint-glass choices for the cemented doublet.
5. The prose reports 2,625 specifications; Table 3’s caption says 2,525. The prose split is 2,047 training and 578 validation. The paper therefore contains an internal count discrepancy, which is left explicit here.
6. Ten designs adapted from Zemax ZEBASE C_001 set coefficient ranges; again, they are guiding ranges rather than a label for every specification.

### What the method generates

The “dataset” is principally a grid of **design requirements**. Candidate prescriptions are created during network training and judged by the differentiable optical model.

---

## 8. Mao et al. (2023): FreeformNet’s fundamental set and feedback expansion

**Paper:** Boyu Mao et al., “FreeformNet: fast and automatic generation of multiple-solution freeform imaging systems enabled by deep learning.” [Primary full text](https://opg.optica.org/prj/viewmedia.cfm?seq=0&uri=prj-11-8-1408) · [DOI](https://doi.org/10.1364/PRJ.492938)

### Exact source locations

- **§2.B — “Fundamental Data Set Generation.”**
- **Fig. 2 — “Illustration of the fundamental data set generation and feedback strategy.”**
- **§3 design example, pp. 1416–1417**, paragraphs beginning “Then, the full fundamental data set…” and “Next, the DNN was pretrained…”.

### Fundamental-dataset generation — §2.B and Fig. 2

1. Define a system-parameter space (\Phi) and a structure-parameter space (\Psi).
2. Divide both spaces into subspaces and pair corresponding regions (SP^{(i)}_{\Phi}-SP^{(i)}_{\Psi}).
3. Place/obtain a reference system (RSYS(i)) at the center of each subspace pair.
4. From each reference, generate good systems that cover that pair using the paper’s combined **sequential and random freeform-system evolution** procedure.
5. Record system parameters, structure parameters, and surface parameters. These records form the fundamental supervised dataset.

### Exact instantiated generation in §3

1. Automatically generate the reference systems and thereby determine all subspace pairs; the reported reference-generation time is 0.83 h.
2. Apply the **random optimization generation method** with
   \[
   P_f=0.6,\qquad P_{f1}=0.5,\qquad P_{f2}=0.2.
   \]
3. Generate 12 systems in each of 2,598 subspace pairs:
   \[
   2{,}598\times12=31{,}176
   \]
   systems, taking 10.2 h.
4. Form every record from its system, structure, and surface parameters.

### Feedback generation — Fig. 2 and §3 pp. 1416–1417

1. Pretrain the DNN on the 31,176-system fundamental set.
2. In each feedback execution, select 1,000 random parameter combinations from different subspace pairs and input them to the DNN.
3. Optimize the output systems with (P_s=0.5).
4. Add the good optimized systems to the training set.
5. Continue supervised training and repeat the feedback cycle.
6. The dataset grows to 109,703 systems after the reported 147,000 supervised-training epochs; the paper reports 73.3 h for this stage.

### Separate unsupervised input generation

During combined supervised/unsupervised training, the unsupervised inputs are not fixed: every epoch randomly selects 500 subspace pairs and generates one random input-parameter combination within each selected pair.  
**Locator:** §3, paragraph beginning “Next, unsupervised learning was introduced…”.

### What exactly becomes a supervised record

\[
\boxed{(\text{system parameters},\ \text{structure parameters})
\longrightarrow \text{surface and output-structure parameters}}
\]

The fold family is still encoded by the paper’s structure parameterization; FreeformNet does not generate an unrestricted ray–surface interaction sequence.

---

## 9. Chen et al. (2025): ODMINN

**Paper:** Guogen Chen et al., “Generating multiple-folding-geometry freeform reflective imaging systems based on optical design model-informed neural network.” [Primary author record](https://pure.bit.edu.cn/en/publications/generating-multiple-folding-geometry-freeform-reflective-imaging-/) · [DOI](https://doi.org/10.1016/j.optlastec.2025.113322)

### What the primary abstract verifies

1. An automatic training-dataset-generation method is combined with a fast obstruction evaluator based on equivalent spherical systems.
2. The resulting dataset contains systems with varied parameters and folding geometries.
3. The demonstration covers eight predefined folding geometries.
4. Training uses both data-driven and differential-ray-tracing physics losses.

### Exact dataset-generation section and process

*

No additional procedural step is inferred from the abstract.

---

## 10. Chen et al. (2026): cGAN dataset generation

**Paper:** Guogen Chen et al., “Freeform optical design with a generative AI model based on a conditional generative adversarial network.” [Local PDF](../paper_pdfs/GAN_paper.pdf) · [Primary full-text page](https://www.spiedigitallibrary.org/journals/advanced-photonics-nexus/volume-5/issue-04/046023/Freeform-optical-design-with-a-generative-AI-model-based-on/10.1117/1.APN.5.4.046023.full) · [DOI](https://doi.org/10.1117/1.APN.5.4.046023)

### Exact source locations

- **§2.3, “Dataset Generation,” pp. 046023-4–5:** generic sampling, obstruction prescreen, optical-system construction, optimization, parallelization, and record schema; Fig. 3.
- **§3, “Example Demonstration,” pp. 046023-8–9:** the exact three-mirror instantiation, ranges, filtering thresholds, generated/retained counts, and wall time; Fig. 5.

### Generic dataset-generation algorithm

1. Select system parameters \(\Psi\) and the discrete folding-geometry parameter \(G\). Combine them into the conditional-label vector \(C\).
2. Select the outputs: structure parameters \(Y_{\mathrm{str}}\), comprising surface positions and tilts, and surface parameters \(Y_{\mathrm{surf}}\), comprising freeform coefficients. Together they form \(Y\).
3. Define the conditional-label space \(SP_c\). Separately combine the ranges of system parameters, structure parameters, and fold identifier into the full generation space \(SP_e\).
4. Randomly sample \(SP_e\) to obtain candidate parameter combinations \(EP\).
5. Prescreen every candidate for obstruction. Construct its equivalent coaxial spherical system, solve the spherical mirror curvatures, estimate every full-field aperture using Gaussian optics and geometry, and calculate element-edge distances to the other beam envelopes. Reject combinations that violate the prescribed separation threshold, yielding \(EP^*\). **Locator:** §2.3 and Fig. 3.
6. For each \(ep_v^*\), reuse its equivalent-coaxial curvatures, assign its requested system parameters and off-axis element coordinates, build the initial off-axis reflective system, convert the surfaces to the selected freeform representation, and optimize the system. Each case is independent, so cases can run in parallel.
7. After optimization, record
   \[
   c_k^*=(\text{system parameters},\text{fold ID}),\qquad
   y_k^*=(\text{structure parameters},\text{surface parameters}),
   \]
   producing supervised pairs \((c_k^*,y_k^*)\).

### Exact three-mirror dataset used in the paper

1. Use unobscured off-axis three-mirror systems with no intermediate image, symmetry about the YOZ plane, the stop on M2, and sixth-order XY-polynomial mirrors with no base conic and no odd powers of \(x\).
2. Fix focal length at 1 mm. Use the four-dimensional label
   \[
   C=[\mathrm{XFOV},\mathrm{YFOV},\mathrm{ENPD},\mathrm{FG}],
   \]
   with XFOV and YFOV each in \([2^\circ,15^\circ]\), ENPD in \([0.2,0.526]\ \mathrm{mm}\) (F/1.9–F/5), and FG in \(\{1,\ldots,8\}\).
3. Use **10 structure outputs**
   \[
   [M1_y,M1_z,M3_y,M3_z,IMG_y,IMG_z,
   M1_{tilt},M2_{tilt},M3_{tilt},IMG_{tilt}]
   \]
   plus **42 surface coefficients**. M2's vertex is fixed at the global origin. Thus each target vector has 52 values.
4. Define a separate structure-parameter space for each of the eight folds, randomly sample it, and apply the fold-specific obstruction tests in Fig. 5 with a 0.05 mm threshold. Retain approximately 20,000–30,000 candidate combinations per fold after this prescreen.
5. For each survivor, calculate an initial off-axis system, upgrade all mirrors to XY-polynomial surfaces, and perform multiparameter optimization in optical-design software. Remove systems with ray-tracing errors or poor image quality.
6. Parallel generation produces **213,450 optimized systems in 246 h**. A second plausibility screen removes systems whose optimized surface tilts or coefficients lie abnormally far outside the normal range. The **actual CGAN training set contains 211,236 systems**, not 213,450.
7. Store system parameters, FG, structure parameters, and surface parameters for every retained system.

### Reproduction details the full paper still does not report

- The random-sampling distribution, seed, and exact numerical bounds of each fold-specific structure-parameter space.
- The complete optical-design merit function, optimizer settings, parameter bounds, and numerical poor-image-quality cutoff used in the §3 multiparameter optimization.
- Exact retained counts for the individual folds; only the pre-generation range of approximately 20,000–30,000 combinations per fold and the aggregate counts are stated.

These are omissions in the supplied full paper, not missing-access placeholders.

### Do not confuse dataset generation with cGAN sampling

After training, the generator receives

\[
(\text{system conditions},\ \text{fold conditions},\ z)
\]

and uses different noise vectors (z) to produce multiple prescriptions for the same conditions. That is **model inference**, not the procedure used to produce the 211,236 retained real training systems (from 213,450 optimized systems generated before the final plausibility screen).

---

## 11. Su & Romanova (2026): two-stage genetic algorithm

**Paper:** Guanqiao Su and Galina Romanova, “Two-stage genetic algorithm optimization of a freeform prism AR system design.” [Local PDF](../paper_pdfs/GA_paper.pdf) · [Publisher page](https://opg.optica.org/ao/abstract.cfm?uri=ao-65-15-5158) · [DOI](https://doi.org/10.1364/AO.599617)

### Exact source locations

- **§2, “Numerical Model and Ray-Tracing Procedure,” pp. 5159–5161:** surface representation, 81-ray evaluator, TIR handling, RMS aggregation, boundary penalty, and throughput status; Figs. 1–5, Eqs. (1)–(5).
- **§3.A, “Stage 1: Optical Path Search under TIR and Geometric Constraints,” pp. 5161–5163:** variables/bounds, population, fitness, and handoff; Fig. 6, Table 1, Eqs. (6)–(8).
- **§3.B, “Stage 2: Multi-Objective Optimization of Image Quality and Image Size,” pp. 5163–5164:** initialization, added freeform variables, objectives, constraints, population, and final selection; Eqs. (9)–(12).
- **§4, “Experimental Design,” pp. 5163–5165:** exact demonstration settings, repeated-run protocol, and IQR filtering; Table 2, Figs. 7–10.

### Candidate representation and evaluator

1. Trace the display channel in reverse: pupil → refractive surface A → reflective surface B → second hit on A under TIR → refractive exit surface C → image plane.
2. Express A and B as an even-asphere base plus five low-order terms,
   \[
   s(y,z)=s_{base}(r)+a_1y+a_2z+a_3y^2+a_4yz+a_5z^2.
   \]
   Surface C is an even asphere without these freeform terms.
3. For each wavelength, trace 81 rays: nine pupil/eye-box locations times nine incident directions. Reject individual rays lost to aperture clipping, TIR violation, or a missed downstream surface/image plane. Group valid image intercepts into nine field sets and calculate each field's RMS spot radius.
4. Return the field RMS values, a boundary-overlap penalty
   \[
   P_{wedge}=10^4\max(0,\delta_{AC})+10^4\max(0,\delta_{BC}),
   \]
   and a piecewise throughput status \(S\). When over 90% of rays reach the image, \(S=\bar R\); otherwise it falls into large penalty bands based on the fractions reaching C, the second A hit, and B: 1000, 3000, 5000, or 10000 plus the stated fractional shortfall. **Locator:** §2 and §3.A, Eqs. (5)–(8).

### Stage 1 — feasible-path search

1. Encode each individual as a 16-parameter geometry vector. Keep all A/B freeform coefficients at zero, so Stage 1 searches tilted/decentered conventional aspheres.

   | Variable | Bound | Representative initial value |
   |---|---:|---:|
   | B radius | [−300, −30] mm | −56 mm |
   | A x-position | [5, 25] mm | 13 mm |
   | A aperture radius | [20, 100] mm | 40 mm |
   | B tilt | [−60°, 10°] | −23° |
   | A tilt | [−10°, 10°] | 0° |
   | A base radius | [50, 300] mm | 132 mm |
   | C x-position | [0, 10] mm | 5 mm |
   | C tilt | [60°, 110°] | 78° |
   | C z-position | [20, 60] mm | 35 mm |
   | C base radius | [30, 150] mm | 35 mm |
   | A conic constant | [−1, 1] | −0.5 |
   | B conic constant | [−1, 1] | −0.5 |
   | Image-plane x | [−30, 10] mm | −10 mm |
   | Image-plane y | [−30, 30] mm | 0 mm |
   | Image-plane z | [10, 60] mm | 35 mm |
   | Image-plane tilt | [−160°, −100°] | −135° |

   **Source note:** Table 1 gives A's aperture-radius bound as [20, 100] mm, while the adjacent prose says [20, 80] mm. The table value is reproduced above; the paper is internally inconsistent on this one bound.
2. Minimize
   \[
   F_{stage1}=S+P_{wedge}.
   \]
   Infeasible individuals receive the large throughput/boundary penalties; high-throughput, boundary-valid paths compete on preliminary RMS.
3. Use a population of **150**, at most **60 generations**, and parallel individual evaluation. Pass the best geometry vector \(p_1\) to Stage 2.

### Stage 2 — freeform, image-quality, and image-size refinement

1. Continue varying the 16 geometry parameters and add five polynomial coefficients on each of A and B, for **26 variables total**. Bound each added coefficient to \(\pm5\times10^{-3}\).
2. Copy \(p_1\) 50 times; perturb copies 2–50 independently as
   \[
   p=p_1\odot(1+\epsilon),\qquad \epsilon\sim U[-0.01,0.01].
   \]
3. Run a Pareto multi-objective GA with three unweighted objectives over the nine fields:
   \[
   f_1=\frac{1}{9}\sum_{k=1}^{9}R_k,\qquad
   f_2=\max_k R_k,\qquad
   f_3=|W-12\ \mathrm{mm}|,
   \]
   where \(W\) is the peak-to-peak image-plane ray extent.
4. Enforce a minimum axial separation between C and the image plane. Set all objectives to a large penalty for a bad status score or nonzero wedge penalty, and add penalties to \(f_1,f_2\) if fewer than 99.9% of rays reach the image plane.
5. Use a population of **200** and at most **300 generations**. From the Pareto set, retain only zero-penalty/no-leakage candidates, then select a representative with the smallest RMS spot radius while checking center/worst-field behavior and closeness to the target image size. The selected vector \(p^*\) is the final design.

### Remaining implementation omissions and inconsistency

- The paper does not identify the GA selection, crossover, mutation, or elitism operators, nor a stopping rule other than the maximum generation counts.
- It supplies representative Stage-1 initial values but does not fully specify the Stage-1 population's sampling distribution.
- It says 50 perturbed copies of \(p_1\) form the Stage-2 initial population, but later specifies a Stage-2 population size of 200; it does not explain how the other 150 initial individuals are created.

### Interpretation

The GA creates candidate populations online during optimization. The 50 independent runs and the IQR-filtered set of 43 runs in §4 are for robustness/sensitivity statistics, not a supervised training dataset.

---

## 12. Chen et al. (2026): performance-aware DNN

**Paper:** Guogen Chen, Tong Yang, Dewen Cheng, and Yongtian Wang, “Design of freeform imaging systems using a performance-aware deep neural network.” [Local PDF](../paper_pdfs/chen_DNN_paper.pdf) · [DOI](https://doi.org/10.1364/AO.607567)

### Exact source locations

- **§2.A, “Basic Framework,” pp. 8916–8917:** focal-length normalization, wavelength unification, and the distinction between a candidate's immediate and post-optimization performance.
- **§2.B, “Dataset Generation,” pp. 8917–8918:** generic record schema, random sampling, obstruction prescreen, system construction, optimization, invalid-sample rejection, and recording.
- **§3, “Design Examples,” pp. 8919–8920:** exact three-mirror variables/ranges, eight-fold sampling, optimizer, constraints, metrics, counts, and min–max preprocessing; Fig. 3, Eqs. (4)–(5).
- **Discussion, p. 8924:** 181 h generation time and the exact invalid-sample policy.

### Exact generation process

1. Define each supervised record as
   \[
   \boxed{(SYP,ISTP,FG)\longrightarrow(OSTP,SUP,POIP)}.
   \]
   Here \(SYP\) contains system specifications; \(ISTP\) contains requested element coordinates but not surface tilts; \(FG\) is the folding-geometry ID; \(OSTP\) contains the realized element coordinates and tilts; \(SUP\) contains freeform coefficients; and \(POIP\) contains the best image-performance values reached by the prescribed optimization.
2. Choose system, input-structure, freeform-surface, and performance variables. Define system/structure ranges separately for every supported fold.
3. Randomly sample the multidimensional **input** space. For every sample, construct an equivalent coaxial spherical system, estimate each mirror's full-field beam aperture, and compare mirror/image-plane edge points with marginal beam rays. Reject combinations expected to have difficult-to-remove obstruction; retain no-obstruction or slight-obstruction combinations.
4. For each survivor, solve the curvatures of its equivalent coaxial spherical system, use them as the initial curvatures of an off-axis spherical system, and set the sampled specifications and coordinates.
5. Replace every reflective element with the chosen freeform surface, make the surface parameters optimization variables, and run the common optical-design optimization/evaluation procedure.
6. Remove candidates with incomplete ray tracing, severe obstruction, failed construction, failed optimization, or unavailable performance labels. **Do not filter valid records merely because their optimized image quality is poor**; the network must learn the range of optimization potential.
7. Store the sampled inputs and realized prescription together with its post-optimization performance metrics.

### Exact three-mirror instantiation

1. Generate eight unobscured off-axis three-mirror folds with no intermediate image, YOZ symmetry, the stop on M2, and three sixth-order XY-polynomial mirrors having no base term and no odd powers of \(x\). Use 1 mm focal length and a 10,000 nm reference wavelength for every generated record.
2. Use a **10-value input**:
   - \(SYP=[XFOV,YFOV,ENPD]\), with both FOV components in \([2^\circ,15^\circ]\) and ENPD in \([0.167,0.667]\ \mathrm{mm}\) (F/1.5–F/6);
   - six \(ISTP\) values: the \(y,z\) vertex coordinates of M1, M3, and the image plane, with M2 fixed at the global origin;
   - one \(FG\) value in \(\{1,\ldots,8\}\).
3. Use a **55-value target**:
   - 10 \(OSTP\) values: those six coordinates plus the tilts of M1, M2, M3, and the image plane;
   - 42 sixth-order XY-polynomial coefficients, 14 per mirror;
   - three \(POIP\) values: maximum relative distortion, mean RMS spot diameter, and mean RMS wavefront error.
4. Calculate those performance labels over six fields:
   \[
   (0,0),(0,\theta_y/2),(0,-\theta_y/2),
   (\theta_x/2,0),(\theta_x/2,\theta_y/2),(\theta_x/2,-\theta_y/2).
   \]
5. Generate records separately by fold until **10,000 valid systems per fold** are available. Within the active fold's prescribed structure space, sample XFOV, YFOV, ENPD, and all six \(ISTP\) coordinates randomly and independently.
6. For each prescreened initial system, optimize a transverse-aberration merit function with damped least squares and equal field weights. Add the design constraints through Lagrange multipliers rather than merit-function weights:
   - EFL in both X and Y equals 1 mm;
   - maximum absolute relative distortion in X and Y is at most 20%;
   - the central-field chief ray's local Y coordinate is zero on every mirror;
   - every fold-specific element-edge/beam clearance exceeds 0.05 mm;
   - the central-field entrance-pupil X/Y dimension ratio is generally in [0.9, 1.1].
7. Vary the three mirrors' sixth-order XY-polynomial coefficients for at most **30 optimization cycles**. Retain successfully traced/evaluated systems and calculate the three POIP labels.
8. The final dataset has **80,000 valid records**, balanced as 10,000 for each of eight folds, and took approximately **181 h** to generate.

### Dataset normalization

All input and output dimensions are independently min–max scaled to \([-1,1]\):

\[
x_{norm}=2\frac{x-x_{min}}{x_{max}-x_{min}}-1.
\]

Because coordinate ranges differ by fold, structure dimensions use the bounds for the record's fold. The broader framework also normalizes physical scale to 1 mm EFL (length-like values scale as \(x'=x/\mathrm{EFL}\)) and converts wavefront-error labels to the common reference wavelength.

### Reproduction details not reported

- Numerical bounds for the six fold-specific \(ISTP\) coordinate spaces.
- The random seed/distribution beyond “randomly and independently” sampled within the prescribed bounds.
- A train/validation/test split ratio, although §2.C says performance is evaluated on a test set.

These are paper-level omissions; the supplied PDF otherwise specifies the dataset pipeline and example in substantial detail.

---

## The four genuinely different ML data-generation recipes

### A. Sequential evolution plus neighborhood interpolation — Chen 2021

\[
\text{sample SPS}
\rightarrow
\text{sequential nearest-system evolution}
\rightarrow
\text{primary set}
\rightarrow
K\text{NN interpolation}
\rightarrow
\text{parallel optimization}
\rightarrow
\text{secondary set}
\]

**Exact locators:** §§2.1.3–2.1.4, Figs. 3–4, Eqs. (3)–(4).

### B. Subspace evolution plus network feedback — FreeformNet

\[
(SP_\Phi,SP_\Psi)
\rightarrow
RSYS(i)
\rightarrow
\text{random/sequential evolution}
\rightarrow
31{,}176
\rightarrow
\text{DNN feedback + optimization}
\rightarrow
109{,}703
\]

**Exact locators:** §2.B, Fig. 2, and §3 pp. 1416–1417.

### C. High-dimensional conditional sampling — cGAN

\[
(\text{system/structure space},\text{fold})
\rightarrow
\text{random sampling + obstruction prescreen}
\rightarrow
\text{spherical seed + freeform optimization}
\rightarrow
213{,}450
\rightarrow
\text{range screen}
\rightarrow
211{,}236
\]

**Exact locators:** §2.3, pp. 046023-4–5, and §3, pp. 046023-8–9.

### D. Fold-balanced optimization with performance labels — performance-aware DNN

\[
(SYP,ISTP,FG)
\rightarrow
\text{random sampling + obstruction prescreen}
\rightarrow
\text{spherical seed + constrained freeform optimization}
\rightarrow
(OSTP,SUP,POIP)
\rightarrow
80{,}000
\]

**Exact locators:** §2.B, pp. 8917–8918; §3, pp. 8919–8920; discussion on p. 8924.

## Bottom line

- When this file says **how** a system or dataset is generated, it immediately names the paper section that supports the statement.
- Zhang 2021 and Yang 2017 generate optical systems but do not construct ML datasets in those papers.
- Chen 2021, FreeformNet, the cGAN paper, and the performance-aware DNN paper now all have section-level supervised-data recipes in this audit.
- Nie 2023 samples specifications and lets differentiable ray tracing create the training signal; it does not need a ground-truth prescription for every input.
- The cGAN run generated 213,450 optimized systems in 246 h, then retained 211,236 after the tilt/coefficient plausibility screen; 211,236 is the training-set count.
- The performance-aware DNN dataset adds post-optimization performance labels and deliberately retains valid low-performance systems; it contains 80,000 records balanced across eight folds and took approximately 181 h to generate.
- ODMINN remains abstract-only in its own entry, as requested. Later papers restate parts of its generator, but those restatements are not presented as a substitute for auditing the unavailable ODMINN paper itself.
