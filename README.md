# AI for NS Lens Design

Non-sequential lens design by search, and the export of discovered designs into **CODE V**.

A CMA-ME quality-diversity search discovers folded solid-glass optical elements (TIR
prisms) in a non-sequential ray-tracing engine. This repo holds that pipeline, the engine
it runs on, and — the current focus — the work to get those designs **tracing in CODE V**
as sequential freeform prisms so they can be rendered, benchmarked and optimised.

**Code only.** Images, renders and bulk search results are excluded by `.gitignore`; the
one exception is `ns_probe/results_fold_qd/`, which holds the design archive the CODE V
exporters read.

---

## Start here

| if you want to... | read |
|---|---|
| **Get a design tracing in CODE V** | [`codev_handoff/README.md`](codev_handoff/README.md) |
| Run CODE V hands-on as an operator/agent | [`codev_handoff/AGENT_RUNBOOK.md`](codev_handoff/AGENT_RUNBOOK.md) |
| Know what CODE V syntax actually works | [`codev_handoff/CODEV_SYNTAX_FINDINGS.md`](codev_handoff/CODEV_SYNTAX_FINDINGS.md) |
| Understand how the geometry is built | [`codev_handoff/BUILD_METHOD.md`](codev_handoff/BUILD_METHOD.md) |
| Run the search pipeline / engine | [`RUN_NOTES.md`](RUN_NOTES.md) |
| Understand the codebase as a whole | [`agent_context/`](agent_context/) |

---

## Layout — do not rearrange

```
ns_probe/         CMA-ME quality-diversity probe: folds, AR/VR prism, refine_lm. The project.
ns_autodiff/      differentiable PyTorch twin (imports from ns_probe/)
optical_rl/       only optical_rl/utils/ is used: the Python <-> Node bridge
ray-optics/       the Node.js non-sequential engine
codev_handoff/    self-contained CODE V package: .seq, verifiers, runbook, findings
agent_context/    codebase maps and research notes
ns_smoke_test.py  end-to-end bridge smoke test
```

**Critical:** `optical_rl/` and `ray-optics/` must stay **siblings at this root**. The
bridge resolves the engine by relative path
(`optical_rl/utils/ray_optics_bridge_persistent.py` → `parents[2]/ray-optics/integrations/
runner_persistent.js`). Move either and the engine will not be found.

---

## Setup

### CODE V work only — no setup needed beyond numpy

The `codev_handoff/` package is self-contained and does **not** need Node, the engine, or
the search pipeline:

```bash
cd codev_handoff
python3 verify_v2.py      # expect: ALL FIELDS CLOSE
python3 check_seq_v2.py   # expect: ALL CHECKS PASS
```

Both exit non-zero on failure, so they are safe to gate on in CI.

### Full pipeline — needs Node and Python

`ray-optics/node_modules` is **not** committed (it includes the native `canvas` module,
which must be built on the target machine):

```bash
cd ray-optics && npm install && cd ..
PYTHONPATH=. python3 ns_smoke_test.py       # verifies the Node bridge
```

Python needs `numpy`, plus `torch`/`matplotlib`/`pyyaml` for the search and autodiff
pipelines. See [`RUN_NOTES.md`](RUN_NOTES.md) for the exact `PYTHONPATH` invocations —
`ns_probe` and `ns_autodiff` use bare imports and need specific paths.

### Note on the vendored engine

`ray-optics/` is vendored from
[`ebelcognito/ray-optics`](https://github.com/ebelcognito/ray-optics) and **carries local
modifications that this project depends on** — chiefly `src/core/Simulator.js`,
`src/core/sceneObjs/BaseGlass.js`, and the persistent-runner integration
(`integrations/runner_persistent.js`, `simulation_core.js`). Do not replace it with a clean
upstream checkout; the changes are not upstream.

`ray-optics/integrations/rayOptics.js` is a **built webpack bundle** copied from
`dist-node/`. If you edit engine sources you must rebuild and re-copy it — see the "Engine
rebuild gotcha" section of [`RUN_NOTES.md`](RUN_NOTES.md).

---

## Current status

The CODE V export is **in progress and not yet complete**. `codev_handoff/
benchmark_tierA_v2c.seq` is the current best file: its geometry is independently verified
(all 5 fields close over the full pupil at 3 wavelengths) and every check that does not
require CODE V passes. It has not yet been run in CODE V. The remaining unknowns are CODE V
placement conventions, documented with their exact error text in
`codev_handoff/history/FAILURE_LOG.md`.

The real target — a 4-TIR-bounce discovered fold — is staged in
`codev_handoff/target_fold_design/` and is deliberately **not** started yet.
