# Spectrometer control adapter research — 2026-09-29

Status: code inspection and literature-grounded proposal; no adapter trained, no search launched, no live agent changed.

## Finding

The present Qwen interface is text/JSON tool calling, but text generation is not yet established as the cause of poor optical results. In experiments/qwen_paper_autonomous/optical_tools.py, run_cma exposes seeds, base/freeform evaluation budgets and population size. It cannot resume an incumbent or full CMA state, select active parameter blocks, alter step scale, or invoke polish.py. inspect_run removes nested arrays/dictionaries from metrics, including centroid maps. The controller consequently has limited optical observability and authority.

The stronger reference is a sequence of CMA, least-squares refinement, curvature-constrained refinement, and an aperture-only correction (nslens/spectrometer/README.md). Treating that as a CMA-only comparator would confound algorithm access and control quality. The saved final artifact has 100% geometric ray survival, approximately 4.53 micrometres aggregate RMS spot and 0.441 waves worst-cell RMS WFE on its saved audit. It does not meet the 0.07-wave reproduction target recorded in Qwen's notes. Matching our numerical reference and reproducing the paper are separate endpoints.

The two completed optical_* search artifacts currently visible each contain only 48 evaluations; runner.py explicitly distinguishes earlier harness smoke tests from research runs. They cannot establish Qwen's research performance. A complete attribution of the reported failure would require reconciling historical transcripts, research runs, budgets and tool versions.

Another mismatch is the objective: model.evaluate optimizes mean geometric spot size, ray survival, empty cells, dispersion and magnification. It does not directly optimize the worst-cell WFE, distortion, volume and curvature requirements jointly. A more capable controller cannot guarantee success from this merit or this fixed surface basis.

## Proposed architecture

User specification -> Qwen task interpretation -> validated numeric target vector.
Numeric target + optical diagnostics + optimizer state -> compact numeric policy -> bounded control action -> CMA or local refinement -> ray tracer -> diagnostics.

Start with a separate numerical adapter service. This works with the existing text endpoint. A neural head attached directly to Qwen hidden states is a later alternative requiring model-runtime access and training/inference integration; changing the prompt alone does not add that head. LoRA on JSON responses still produces text and is a separate comparison.

Inputs: normalized genes; stage; remaining evaluation budget; CMA mean, sigma, covariance summary and progress; field-by-wavelength spot/WFE/centroid residuals; ray survival and obstruction masks; per-surface footprint and curvature margins. Keep missing rays explicit. Start with structured tensors and a small MLP or set encoder; rendered spot diagrams are optional and should earn their inclusion in an ablation.

Outputs, initially: continue/restart/refine, bounded log step multiplier, and active parameter block (layout, curvature, detector, freeform). Restart must specify a permitted archived parent. Refinement uses a fixed approved objective and gates shared with baselines. Persist full optimizer and random-generator state so continuation is real continuation. Start with block control before learning 36 independent geometry updates.

A second stage could predict a bounded update in a local optical-response basis. Estimate J = d(residuals)/d(normalized genes), form a truncated SVD, and predict coefficients along well-observed parameter directions. A damped least-squares step in exactly the same basis is a mandatory nonlearned comparator. Finite-difference probes count toward budget; the basis is local and must be refreshed or rejected when ray routing/support changes. Numerical sensitivity is a hypothesis for better conditioning, not a proven advantage.

## Minimal discriminating experiment

1. Freeze evaluator version, targets, physical masks, acceptance gates, train sampling and unseen audit sampling. Record separate reference-matching and paper-reproduction criteria.
2. Build a resumable bounded environment and replay a numerical control sequence through it. Require identical candidates/metrics under the same state and random stream before involving Qwen.
3. Compare fixed numerical scheduling, frozen Qwen with equivalent tools/diagnostics, a numerical-only learned policy, and the same policy conditioned on Qwen's task representation. Add text LoRA only as a separately costed arm. First run CMA-only; then give every arm the same hybrid refinement access.
4. Collect training checkpoints across independent numerical runs and target settings, retaining failed states. At each checkpoint evaluate a small predefined action menu with paired random streams and repeated seeds. Learn expected verified improvement per evaluation or rank actions; do not label every random CMA proposal as a good demonstration.
5. Split by parent run and target family before selecting checkpoints. Adjacent checkpoints from one run must never cross training/test boundaries. Existing folded-optics continuation data can inform interface design but are not spectrometer demonstrations. One design can support a development pilot, not a generalization claim.
6. Report audit pass rate, evaluations to a fixed quality threshold, wall-clock time including inference/probes, invalid action rate and distribution of worst-cell metrics. Count all attempted candidate/residual calls, numerical Jacobian probes, and training-data simulation costs. Report offline training cost separately and estimate the number of deployments needed to recover it.
7. Freeze candidate selection before final audit. Keep ray survival, curvature, distortion, volume and WFE as distinct gates; learned scores cannot certify physics. Use multiple independent starts and paired uncertainty estimates. Claim an advantage only if it survives equal access and equal budget comparisons.

Suggested first milestone: adapter environment + numerical replay + frozen Qwen parity experiment. If a numerical-only policy matches or beats Qwen conditioning, use it for control and retain Qwen for specification and explanations. If numerical methods still miss the target, investigate objective and representational feasibility before expanding the language model.

## Research grounding

- OpenVLA-OFT, https://openvla-oft.github.io/ : primary research demonstrating continuous action prediction with parallel decoding and action chunking in robotics. Supports studying numeric action heads; its gains do not establish gains for optical optimization, and multiple changes contribute to its reported speedup.
- Liu et al., Training deep neural networks for the inverse design of nanophotonic structures, https://arxiv.org/abs/1710.04724 : tandem forward/inverse training addresses nonunique inverse mappings. Motivates evaluating optical response rather than demanding one exact geometry. Different physical domain; not evidence that a spectrometer adapter will work.

The defensible research question is whether response-conditioned bounded control improves verified optical quality per simulator call on held-out spectrometer tasks, and whether language conditioning adds value beyond an equally informed small numeric policy.
