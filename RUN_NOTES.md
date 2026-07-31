# NS non-sequential lens-design framework — self-contained copy

Copied from `~/Desktop/RL` on 2026-07-10. This folder is self-contained: the NS
code, the shared Python glue it needs, and the Node ray-optics engine.

## Layout (do NOT rearrange)

```
NS_lensdesign/
  ns_probe/         # QD/CMA-ME probe, folds, AR/VR prism, refine_lm  (the project)
  ns_autodiff/      # differentiable PyTorch twin  (imports from ns_probe/)
  ns_smoke_test.py  # end-to-end bridge smoke test
  optical_rl/       # ONLY optical_rl/utils/ is used (the Python<->Node bridge + config)
  ray-optics/       # Node.js engine (integrations/runner_persistent.js + built rayOptics.js bundle)
  agent_context/    # the maps: README, ray_optics_engine.md, ns_rl_research/, ns_differentiable_refinement/
```

**Critical:** `optical_rl/` and `ray-optics/` must stay **siblings at this root**.
The bridge resolves the engine with a relative path:
`optical_rl/utils/ray_optics_bridge_persistent.py` → `parents[2]/ray-optics/integrations/runner_persistent.js`.
Move either one and the engine won't be found.

## Environment (external — not copied)

- Python: `/playpen/hisham26/deeplens_env/bin/python`  (venv, py3.12; torch/numpy/matplotlib/pyyaml)
- Node: on PATH (`node v20.20.0`); `ray-optics/node_modules` is copied in (incl. `canvas`).

## Running

```bash
cd ~/Desktop/NS_lensdesign

# smoke test (verifies the Node bridge)
PYTHONPATH=. /playpen/hisham26/deeplens_env/bin/python ns_smoke_test.py

# ns_probe scripts
PYTHONPATH=.:ns_probe /playpen/hisham26/deeplens_env/bin/python ns_probe/run_fold_qd.py ...

# ns_autodiff needs BOTH this root and ns_probe/ on the path (bare imports: cma_me, fold_demo, sequence_tools)
PYTHONPATH=.:ns_probe /playpen/hisham26/deeplens_env/bin/python -m ns_autodiff.search_freeform ...
```

## Engine rebuild gotcha

`ray-optics/integrations/rayOptics.js` is a **built webpack bundle** (copied from `dist-node/`).
If you edit engine `src/`, rebuild then re-copy the bundle into `integrations/` — otherwise
your changes silently don't take effect. See `agent_context/ray_optics_engine.md`.

## Verified 2026-07-10

`ns_smoke_test.py` runs (Node bridge OK, detector power=1.0); `ns_probe` and `ns_autodiff`
imports resolve from this copy.
