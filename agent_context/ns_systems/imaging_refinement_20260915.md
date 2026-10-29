# Imaging refinement pilot — September 15

User explicitly selected separate sharp spots for imaging, not a common small receiver. Launched bounded CPU CMA pilot at `/playpen/hisham26/concentration_pilot_20260915`, PID in launch.json. Existing Qwen adapter job was not interrupted. No Qwen invocation in this pilot.

Three saved designs A/B/C from `/playpen/hisham26/fov_review_20260915/sweeps.json`; original full angular widths 10/4/4 degrees. Preserve detector, material, beam, FOV, and both initial body dimensions as upper bounds. Initial engineering target: worst per-field RMS spot radius <=0.5 mm, with >=80% worst-field folded throughput and root support. User did not specify the numerical spot target. Three seeds per design, 360 candidates per seed, four CPU workers, 3,240 total proposals. Search uses minimax violation reduction followed by sharper RMS; temporary infeasible candidates remain eligible, final passing claims retain all thresholds.

Dense final evaluations: 17 fields x 101 rays; 33 x 151; doubled branch capacity agreement. Final success additionally requires strict monotonicity of centroid versus angle. Monotonicity is diagnostic/final gating, not in search ranking. This does not establish imaging resolution, MTF, aberration correction, chromatic performance, or 3D performance. Spot RMS is measured on accepted folded detector energy, coupled to the throughput floor.

Initial search-grid worst spot radii A/B/C: 12.547/3.221/0.978 mm. Numerical smoke and feasible-ranking checks passed before launch. A has an unresolved-energy violation on this search grid, despite previous saved certification; all numerical ledger constraints remain enforced.

Artifacts: protocol.json and source_snapshot, tasks.json, run.py, status.json, run.log, per-task status/history/result, incremental summary.json. Supervisor is run.py; exceptions set failed status. Check actual final artifacts before claiming success. No full results existed at launch.
