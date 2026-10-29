# 17 — Related-work search: adaptive sampling / QD for optical design

**Date of search:** 2026-08-08. Closes the open action items in
[13_new_direction_framing.md §5](13_new_direction_framing.md) ("read/cite XLumina", "[to verify —
literature search]") and the seven-item literature-check list in
[05_milestone_experiments.md §"Open questions for a literature check"](05_milestone_experiments.md).

---

## TL;DR — the five things not to get wrong

1. **The "QD-in-optics whitespace" claim in doc 13 is dead as written.** Kononova & van Stein published
   **quality-diversity driven *robust* evolutionary optimization of optical designs** at SPIE
   Computational Optics 2024. That is QD + optics + robustness — i.e. doc 13's Track 1 *and* Track 3,
   for sequential spherical lens systems. We must cite it and re-scope the claim.
2. **The closest method-level competitor is 6 months old: LDG-EA** (arXiv 2601.22075, Jan 2026,
   Leiden + ASML). It partitions lens design space by **behavior descriptors** (curvature-sign
   patterns + material indices), learns a probabilistic model to steer evaluations to promising
   descriptors, and runs **CMA-self-adaptive** search inside each. That is structurally our
   archive-plus-CMA loop, in the sequential refractive regime.
3. **The closest domain-level competitor is 7 months old:** *End-to-end differentiable design of
   geometric waveguide displays* (arXiv 2601.04370, Jan 2026, Yang/Heidrich et al.) — the first
   end-to-end differentiable **non-sequential** Monte Carlo polarization ray tracer, for AR
   geometric waveguides, with topology optimization by layer pruning.
4. **XLuminA is mis-cited in doc 13.** It is not a "NeurIPS optics" paper — it is
   **Nature Communications 15, 10658 (2024)**, "Automated discovery of experimental designs in
   super-resolution microscopy with XLuminA". Same group's **Urania** (PRX 2025) is the stronger
   rival framing: an algorithm that discovered ~50 gravitational-wave-detector designs and released
   them as a public **"Detector Zoo"** — an archive of diverse machine-discovered optical setups.
5. **The surviving daylight is a conjunction, not a single leg.** Every individual leg —
   QD-in-optics, descriptor-partitioned CMA search, differentiable NS tracing, diverse-folded-system
   archives, AI-discovers-optical-setups — is now occupied by 2024–2026 work. What nobody does is
   *all of them at once*: **archive-structured search over non-sequential TIR interaction topology
   under an imaging objective with an energy ledger.** Write the novelty claim as the conjunction.

---

## 1. Direct hits — works that overlap our method

| Work | What it does | Regime | Our daylight |
|---|---|---|---|
| **Kononova & van Stein, "Quality-diversity driven robust evolutionary optimization of optical designs"**, SPIE Computational Optics 13023, 130230B (2024). DOI 10.1117/12.3017498 | QD-driven evolutionary search for optical designs with **robustness**; modified **HillVallEA**; finds a diverse set of high-quality 4-lens designs in one run from random curvatures. Billed as the first result where an optimizer *not* encoding optical knowledge produces relevant systems. | Sequential, spherical, refractive; lithography/metrology context | Ours is **non-sequential**: the diversity axis is *interaction topology* (TIR fold count / locus), not curvature values. Their robustness is our Track 3 — **they got there first, so Track 3 must be framed as "does robustness change which *topologies* win", not "we add robustness"**. |
| **Antonov, Tukker, Botari, Bäck, Kononova, van Stein — "Lens-Descriptor Guided Evolutionary Algorithm (LDG-EA)"**, arXiv:2601.22075 (29 Jan 2026) | Two-stage multimodal lens optimization. Stage 1 partitions design space into **behavior descriptors** = curvature-sign patterns + material indices, learns a probabilistic model over descriptors to allocate evaluations. Stage 2 runs HillVallEA **with covariance-matrix self-adaptation** inside each descriptor, then optional gradient refinement. On a 6-element Double-Gauss (18 continuous + 6 integer vars): ~**14 500 candidate minima over 636 unique descriptors**, an **order of magnitude more than a CMA-ES baseline**, in ~1 h wall clock. Best design slightly worse than the fine-tuned reference. | Sequential refractive, glass choice included | **This is the paper our §2.2 experiment is arguing with.** They already show descriptor-guided allocation ≫ plain CMA-ES *at producing distinct minima* — which is close to our "does CMA-ME earn 'search'?" result. Daylight: their descriptor is a *static geometric signature of a fixed topology* (sign string of a known Double-Gauss); ours is an *emergent physical event sequence* that the search itself creates. They never change the light path's topology; we discover it. Also: they do **not** use MAP-Elites/QD vocabulary or an elite archive — so "QD" is still ours to name, but "descriptor-partitioned CMA search for lens diversity" is not. |
| **Kononova et al., "Addressing the Multiplicity of Solutions in Optical Lens Design"**, GECCO 2021 Companion (p. 1596) | The earlier Leiden/ASML statement of the multimodal-lens-design problem and benchmark. | Sequential | Prior link in the same chain; cite for lineage. |

**Consequence for the paper.** Doc 13 §5 has no row for the Leiden/ASML line at all. It is the single
biggest gap in our related work, and it is *actively being extended* — two papers in the last 24 months,
one 6 months old.

---

## 2. Direct hits — works that overlap our engine / domain

| Work | What it does | Our daylight |
|---|---|---|
| **Yang, Liu, Nie, Fan, Shi, Bonar, Heidrich — "End-to-end differentiable design of geometric waveguide displays"**, arXiv:2601.04370 (7 Jan 2026) | First end-to-end differentiable framework for geometric waveguides: **non-sequential Monte Carlo polarization ray tracing** (MC sampling used *specifically to dodge the exponential ray-split blowup*) + differentiable transfer-matrix thin-film solver. >1000 layer-thickness params, billions of NS ray-surface intersections, multi-GPU. Layer pruning from over-parameterized stacks = **topology optimization of the coating stack**. Efficiency 4.1% → 33.5%. | The nearest neighbour in *physics and application* (AR, NS, TIR-adjacent, energy efficiency). Daylight: their **geometry is fixed** — a known Lumus-style stacked-mirror waveguide — and "topology" means *which coating layers survive*, not which light path exists. It is a single gradient descent to one optimum: **no diversity, no archive, no global search**. Our claim ("discover the fold topology") is exactly what they assume as input. Also their MC-vs-deterministic-splitting note is a direct answer to our bounce-budget/branching cost problem — **read it before any DeepLens-NS build**. |
| **van Gestel, Filosa, ten Thije Boonkkamp, IJzerman et al. — "Gradient descent-based freeform optics design for illumination using algorithmic differentiable non-sequential ray tracing"**, arXiv:2302.12031, *Optimization and Engineering* (2023); follow-ups: SPIE 12798 (2023) "Unsupervised design of illumination optics"; "…extended sources with truncated hierarchical B-splines" | Differentiable NS ray tracing for freeform **illumination** (prescribed irradiance), incl. handling of gradients through the tracer and an MLP surrogate. | Flagged as *unread* in doc 05 item 2 — **still the reference for gradient aggregation at ray splits**. Non-imaging; single freeform; no topology search. |
| **"Evolutionary optimization algorithms for nonimaging optical design"**, SPIE 11120 (2019) | EAs on non-sequential simulations for non-centred nonimaging systems. | Establishes EA+NS as a known pairing — so "EA on a non-sequential tracer" is **not** novel by itself. Our novelty must live in the *descriptor + archive + imaging objective*, not in "we ran CMA on a NS tracer". |

---

## 3. Diverse-design-space works in classical optics (the "archive" precedent)

| Work | Note |
|---|---|
| **Yang, Zhu, Wu, Jin — "Automated design of freeform imaging systems"**, *Light: Sci. Appl.* 6, e17081 (2017) | Point-by-point construction-iteration; automated unobscured freeform starting points. |
| **"Towards automatic freeform optics design: coarse and fine search of the three-mirror solution space"**, *Light: Sci. Appl.* 10, 65 (2021) | **Coarse search of the solution space** yields "a wide variety of high-quality systems" with different structures and power distributions, from specs alone. This is the closest classical-optics analogue to our archive — and it is over *folded reflective* systems. |
| **Bauer & Rolland et al., "Roadmap for the unobscured three-mirror freeform design space"**, *Opt. Express* 29(17), 26736 (2021) | A systematic map of a folded-mirror design space. Same "here is the space, not one point" spirit. |
| **Bociort et al. — saddle-point construction & networks of local minima**: *Opt. Lett.* 29(2) 189 (2004); *Opt. Eng.* 48(6) 063001 (2009); *Opt. Express* 23(5) 6679 (2015) | The theory of lens-design landscape structure: minima form a **connected network**, links carry Morse-index-1 saddles; SPC deliberately manufactures new minima. Doc 05 item 4's target. **This is the right frame for our basin/refinability results** — cite it when we claim anything about landscape structure. |

**Daylight for all three:** these are sequential (mirror-order-prescribed) systems, searched by
deterministic construction or systematic enumeration over a *known* structural taxonomy. None has an
emergent interaction-sequence descriptor, none traces non-sequentially, none has an energy ledger.

---

## 4. AI-discovers-optical-setups (the framing rivals)

| Work | Note |
|---|---|
| **XLuminA** — Rodríguez, Krenn et al., *Nature Communications* 15, 10658 (2024): "Automated discovery of experimental designs in super-resolution microscopy with XLuminA" (arXiv:2310.08408; code: github.com/artificial-scientist-lab/XLuminA) | JAX auto-differentiable optics simulator; rediscovers the optical telescope, STED, and radial-polarization focusing **from scratch**, then finds a novel sub-diffraction blueprint. ~10⁴× speed-up vs established numerical optimization. **Correct the doc-13 citation from "NeurIPS optics" to Nat. Comms. 2024.** Daylight unchanged and still good: optical-table layout for microscopy, not imaging lens/relay topology; no branching-TIR ledger; no diversity archive (single-objective gradient discovery). |
| **Urania** — Krenn et al., "Digital Discovery of Interferometric Gravitational Wave Detectors", *Phys. Rev. X* (11 Apr 2025) | AI search over interferometer topologies; released **50 top designs as a public "Detector Zoo"**. This is the **strongest precedent for our exact deliverable shape** — a machine-produced *collection* of diverse setups plus the "humans must now understand what the machine did" framing. If we claim "prior art reports prescriptions, we report a discovery process", Urania is the counterexample we must name and differentiate from (different physics, different diversity notion: they curate top-50, not an archive indexed by a physical descriptor). |
| **Melvin / Theseus** (Krenn et al., 2016–2021) | Graph-based automated discovery of quantum-optics experiments; lineage for the above. |
| **"Automated Discovery of Coupled Mode Setups"**, arXiv:2404.14887 | Same genre, photonics/coupled-mode. |

---

## 5. Learned / agentic lens design (the corpus-based rivals)

| Work | Note |
|---|---|
| **Côté, Lalonde, Thibault — "Deep learning-enabled framework for automatic lens design starting point generation"**, *Opt. Express* 29(3) 3841 (2021); and "Extrapolating from lens design databases using deep learning", *Opt. Express* 27(20) 28279 (2019) | DNN infers starting points from specs; explicitly outputs **diverse configurations for the same first-order specs**. Doc 13's "LensGAN/Côté" row is roughly right but should cite these two properly. |
| **Yang, Fu, Heidrich — "Curriculum learning for ab initio deep learned refractive optics" (DeepLens)**, *Nature Communications* 15 (2024) | Ab-initio compound-lens design from flat surfaces via curriculum learning + differentiable ray tracing. The "no human starting point" precedent — cite it, because "we design from scratch" alone is not a novelty claim any more. |
| **OptiAgent**, arXiv:2602.23761 — "A Physics-Driven Agentic Framework for Automated Optical Design" | LLM agent + OptiDesignQA dataset + GRPO with an Optical Lexicographic Reward. Doc 01/13's OptiAgent row now has a citable arXiv id. |
| **"Artificial intelligence in optical lens design"**, *Artificial Intelligence Review* 57 (2024), doi 10.1007/s10462-024-10842-y | Survey. Use it as the one-stop citation for "the field is crowded on sequential refractive". |
| **Kramer, LensRL** (github.com/HarrisonKramer/LensRL); Fraunhofer ILT RL-for-optical-design (Annual Report 2021); "Reinforcement learning for guiding optimization processes in optical design", SPIE 12227-9 | The RL-in-lens-design line we chose *not* to take. Cite to justify the choice (doc 03's CMA-ME-vs-RL decision). |
| **"Constrained optimization of a zoom lens with CMA-ES"**, EPJ Web Conf. (EOSAM 2025); **"Human-competitive lens system design with evolution strategies"**, *Applied Soft Computing* (2008) | CMA-ES on lens design is established and human-competitive. Again: the optimizer is not the novelty. |

---

## 6. QD-algorithm side (methods we should position against)

- **Fontaine et al., CMA-ME** — "Covariance Matrix Adaptation for the Rapid Illumination of Behavior
  Space", arXiv:1912.02400 / GECCO 2020. Our optimizer.
- **Fontaine & Nikolaidis, CMA-MAE** — "Covariance Matrix Adaptation MAP-Annealing", arXiv:2205.10752,
  ACM TELO (2024). **Fixes three CMA-ME failure modes we plausibly hit**: quick abandonment of
  hard-to-optimize objectives, poor exploration of flat objectives, inefficiency on low-resolution
  archives. Our fold archives are low-resolution and our fitness has flat regions — a reviewer will
  ask why we did not use CMA-MAE. Have an answer, or run it as an arm.
- **Cully & Demiris, "Quality and Diversity Optimization: A Unifying Modular Framework"**,
  arXiv:1708.09251; **Chatzilygeroudis et al., "Quality-Diversity Optimization: a novel branch of
  stochastic optimization"**, arXiv:2012.04322 — standard framing citations.
- **Gaier, Asteroth, Mouret — SAIL** (surrogate-assisted illumination), *Evol. Comput.* 26(3) (2018),
  arXiv:1806.05865 — already known in doc 03; the precedent if evaluation cost forces a surrogate.
- **"Dominated Novelty Search"**, arXiv:2502.00593 (2025) — current alternative to grid archives.

---

## 7. What actually changed for us

| Doc-13 claim | Status after this search |
|---|---|
| "Is the QD-in-optics whitespace still open?" (doc 05 item 1) | **No.** SPIE 13023 (2024) is QD + optical design + robustness. Re-scope to *non-sequential* QD. |
| "XLumina — NeurIPS optics" | **Mis-cited.** Nature Communications 15, 10658 (2024). |
| "prior art reports prescriptions; we report a discovery process" (doc 13 action item 2) | **Weakened.** Urania (PRX 2025) reports a discovery process *and* ships a 50-design zoo. Still true narrowly for **AR freeform prisms specifically** — no paper found that walks the *process* for Cheng-style OST-HMD prisms; the AR-prism literature is prescriptions + patents. Keep the claim, but scope it to that family and name Urania. |
| "our diversity is over interaction topology" | **Holds, and is now the load-bearing claim.** LDG-EA owns descriptor-guided CMA over *static* lens descriptors; the geometric-waveguide paper owns differentiable NS; neither searches over *which light path exists*. |
| "van Gestel unread" (doc 05 item 2) | Still unread — now with the full citation chain (§2). |
| "Automated discovery of folded/catadioptric systems beyond Volatier" (doc 05 item 5) | Partly eroded: LSA 2021 coarse/fine three-mirror search and the OE 2021 unobscured-three-mirror roadmap both map folded-mirror design spaces. Sequential and mirror-count-fixed, so the erosion is partial. |
| CMA-ME as optimizer choice | Needs defending against **CMA-MAE** and against LDG-EA's descriptor-allocation model. |

## 8. Recommended next actions

1. Rewrite doc 13 §5 with the six rows above (Leiden/ASML line, geometric-waveguide paper, Urania,
   Bociort, the two three-mirror solution-space papers, CMA-MAE) and fix the XLuminA citation.
2. Read **arXiv:2601.04370** §on Monte-Carlo-vs-deterministic splitting before any further work on
   bounce budgets or a differentiable NS twin.
3. Read **arXiv:2601.22075** and decide whether their descriptor-allocation model is an arm we should
   add to the discovery-control ladder ([14_discovery_control_experiment_plan.md](14_discovery_control_experiment_plan.md)).
4. Get the SPIE 13023 full text (paywalled; abstract only so far) — it is the paper that decides how
   we word the QD claim.
5. State the novelty as the **conjunction** (TL;DR item 5), not as any single leg.

**Not verified:** SPIE 13023 and the GECCO 2021 companion paper were read from abstracts/search
snippets only — the full texts are paywalled or failed to extract. Confirm their descriptors and
benchmarks before citing specifics.
