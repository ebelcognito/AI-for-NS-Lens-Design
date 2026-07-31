# Can we implement a QWP/HWP in this engine? Pancake-lens components and the Jones gap

> Assessment 2026-06-12 (Opus 4.8), in answer to: *"can we implement a half-wave / quarter-wave plate in this
> env? Check what components a pancake lens needs."* Short answer: **not in the current engine** — its
> polarization is intensity-only (no phase), and a waveplate's entire function is *phase retardance*. Adding
> QWP/HWP requires a Jones-vector upgrade (Tier B in [06](06_reflectivity_tir_analysis.md)). And per
> [08](08_fold_qd_diverse_efficient_folds.md), the efficient fold we actually want (TIR) needs **none** of this.

## 1. What a pancake lens actually needs (web research)

A VR **pancake** folds the optical path with **polarization-managed reflection**, packing ~3× the optical
path into a thin module. Canonical components:

1. A **semi-reflective ("50/50") mirror** — usually on a **curved** surface (this provides the optical power
   and the partial reflection).
2. A **quarter-wave plate (QWP)** — converts linear ↔ circular polarization, and **rotates linear
   polarization by 90° on a double pass**. This is the load-bearing trick.
3. A **reflective polarizer (RP)** — transmits one linear polarization, reflects the orthogonal one.
4. **Linear (absorptive) polarizers**, and a display emitting **circularly/linearly polarized** light.

The light path (why the QWP matters): polarized display light passes the 50/50 mirror → QWP → hits the
reflective polarizer at the *wrong* polarization, so it **reflects back** → back through the QWP (polarization
now rotated) → reflects off the curved 50/50 mirror **forward** → QWP again → now the *right* polarization to
**pass** the reflective polarizer → to the eye. The QWP + RP make the **3-pass folded light exit while the
single-pass stray light is blocked** (ghost suppression) — and this is what recovers efficiency (~25% bare →
70–90% with the polarization management). **Remove the QWP and the mechanism collapses.**

Sources:
- [How VR Optics Work — Avantier](https://avantierinc.com/resources/knowledge-center/how-vr-optics-work/)
- [Pancake Lenses for VR Optical Systems — Avantier](https://avantierinc.com/solutions/custom-optics/pancake-lenses-for-vr-optical-systems/)
- [Pancake Lenses: Metrology of Folded VR Optics — Rotlex](https://rotlex.com/pancake-lenses/)
- [Exploring Pancake Lens Design for VR — AZoOptics](https://www.azooptics.com/Article.aspx?ArticleID=2784)

## 2. The engine's polarization model (grounded in code)

A ray carries **two real intensity scalars**, `brightness_s` and `brightness_p` (each initialized to
`0.5·brightness` = unpolarized). Surfaces split them by Fresnel (`R_s ≠ R_p`). **That is the entire
polarization state.** A repo-wide grep confirms **no phase, no complex amplitude, no Jones vector, no
retardance, no waveplate/polarizer object** anywhere in `ray-optics/src/core/`. (`BaseCustomSurface` has a
`p ∈ {0,1}` s/p *label*; `BaseFilter` is a *colour* filter — neither is a polarization-state element.)

**Why this blocks a QWP/HWP.** A waveplate does nothing to the *intensities* `s` and `p` — it changes the
**relative phase** between the s and p field components (λ/4 retardance for a QWP, λ/2 for an HWP). That phase
is what turns linear into circular and rotates polarization on reflection. The current ray state **has no
phase**, so a QWP/HWP has *nothing to act on*. This is not a missing surface type you can bolt on; the **ray
state itself lacks the quantity a waveplate manipulates**.

- A **reflective polarizer** is *partially* representable in the intensity model (reflect all `s`, transmit
  all `p`), because the engine already separates s/p intensities. But **alone it cannot make a pancake** — the
  QWP rotations between passes are what route the 3-pass light out and block the 1-pass; without phase you
  can't model them, so the whole device fails.

## 3. What it would take (the Tier-B Jones upgrade)

To support QWP/HWP/RP and the pancake mechanism honestly:

1. **Ray polarization state → a Jones vector:** replace `{brightness_s, brightness_p}` with a complex 2-vector
   `(E_s, E_p)` carrying **amplitude and relative phase** (intensity = |E_s|² + |E_p|²). Touches the ray
   struct, every light source's emission, and the detector's power read.
2. **Surface interactions → 2×2 Jones matrices**, applied in the local s/p frame: **QWP/HWP** (retarders with
   a fast-axis angle), **linear polarizer**, **reflective polarizer**, and **polarization-resolved Fresnel**
   (the engine already has the R_s/R_p magnitudes; Jones additionally needs the reflection/transmission
   *phase*). New scene-object types for QWP/HWP/RP.
3. **Frame bookkeeping:** track each ray's s/p basis through 2D reflections/refractions so the Jones matrices
   compose correctly across surfaces.

This is well-defined but a real engine extension (not a patch). It is exactly the Tier-B item flagged in
[06](06_reflectivity_tir_analysis.md) §4.

## 4. Recommendation

- **Don't build the Jones/QWP upgrade unless the experiment is specifically the polarization-managed pancake
  family** (the *lossy* ~25%-bare route whose only point is polarization ghost-suppression).
- The program's high-energy fold is the **TIR route**, which [08](08_fold_qd_diverse_efficient_folds.md) just
  showed is **QD-discoverable as a diverse, efficient (>90%), focusing family with zero polarization**. That is
  the more efficient and the further-along track.
- If you *do* want the pancake (e.g., to test whether polarization-selective routing/ghost-suppression
  *emerges* when the agent is given Jones DOFs — the headline in [01](01_extracted_research_context.md)'s NS
  prompt fork 3), the Jones upgrade above is the prerequisite, and it's a clean standalone engine task to hand
  to Fable. It does **not** block any TIR-route work.
