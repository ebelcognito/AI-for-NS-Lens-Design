# Architecture-conditioned spectrometer discovery pilot

Target: Reimers et al., *Freeform spectrometer enabling increased compactness*
(2017), local source `agent_context/paper_pdfs/lsa201726a.pdf`.

The search is supplied the Offner–Chrisp family: a primary reflector, a convex
reflective grating, a tertiary reflector, and an image plane. Discovering this
architecture is outside this experiment. No published surface prescription,
optimized coefficient vector, or published-layout archive is supplied to CMA.
The chief-ray scaffold, allowed geometry, surface basis and merit function are
explicit engineering priors; random initialization does not remove these priors.

Paper targets include a 10 mm slit, 500–1100 nm band, 100 nm/mm dispersion,
300 lines/mm compact-design grating, and approximately 100 cm³ package versus
530 cm³ baseline. The pilot model's actual sampling, constraints, approximations
and metric definitions must be read with `model.py`; these targets are not a claim
that every requirement is implemented or satisfied.

Search uses the existing full-covariance `nslens.search.cma_me.CMAES`, reflection
at normalized bounds, scalar-loss ranking, and occasional restarts from candidates
retained within the same seed. This is not CMA-ME. Stage one optimizes base pose,
spacing and quadratic curvature variables. Stage two releases the model's higher
surface terms, initializing those terms to zero at the retained base candidate.
The basis is a pilot parameterization, not a transcription of every FRINGE
Zernike term in the paper; some of the paper's piston/wedge terms are linked and
must not be counted as independent optical degrees of freedom.

Each seed owns its RNG and retained-candidate pool. There is no cross-seed sharing.
Budgets count every optical evaluator call during search, including failed rays
and invalid candidates. Budgets round down to complete CMA populations; dense
post-search audit evaluations are logged separately. The first sampled population
is retained as a descriptive starting baseline, not a matched-budget random-search
comparison. No statistical superiority over another optimizer is claimed.

The default training samples are 3 slit positions × 3 wavelengths × the model's
5-point pupil grid. Final and frozen base-stage incumbents are audited with
8 slit positions × 8 wavelengths × a 10-point pupil grid. The denser grid adds
new interior slit and wavelength samples but shares endpoints. It is not a wholly
held-out grid, and it is not independent-engine validation. Final selection uses
training loss only. Missing rays must be assessed using throughput together with
spot and wavefront metrics; an excellent received-ray spot alone is insufficient.

Reproducible invocation:

```bash
.venv/bin/python -m nslens.spectrometer.search \
  --out experiments/spectrometer_pilot_v1 --seeds 101 102 103 \
  --base-evals 1200 --freeform-evals 2400 --workers 3
```

Each output directory must be new. `configuration.json` records the invocation;
seed directories retain checkpoints, normalized/decoded prescriptions, training
histories, initial samples and dense audits. `summary.json` compares independent
seed outcomes. A failed pilot is evidence about this representation, merit and
budget; it does not establish that CMA cannot solve the optical design problem.

Before claiming reproduction, validate the published reference independently,
validate the 3D grating law and surface normals, verify wavefront-reference and
pupil conventions, inspect obscuration and mechanical clearance, enforce the
paper's package metric, check the full slit/wavelength region and tolerances,
and compare finalists with an independent optical model. A surrogate geometric
score or raw bounding box alone does not certify diffraction-limited performance
or the paper's total package volume.

A separately labeled hybrid refinement is implemented in `polish.py`. It logs
actual residual calls, including numerical-Jacobian probes, separately from
SciPy's `nfev`. The initial hybrid merit enforces a linear spectral image map
and unit slit magnification (the latter is a chosen requirement, not established
as a paper specification here). A subsequent quality diagnostic can relax that
map with `--mapping-weight 0.1` while retaining a strong endpoint spectral-span
constraint and use `--rim-points 32` to explicitly train on the pupil boundary.
Both variants include all rays and a soft 100 cm³ sampled optical AABB penalty;
neither variant is CMA-only nor a claim of paper-level reproduction.
