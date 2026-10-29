# NS non-sequential lens-design framework — self-contained copy

### Diverse design stock — 2026-09-11

Completed a six-environment long sweep followed by dense-pupil searches in the four
productive layouts.  Combined: 115,200 traced candidates and 4,492,800 search rays,
with no blocker.  The final stock has 40 exact audit-passing designs across eight
environment/route pairs; exhaustive 909-ray checks retain **13 designs across six
environment/route pairs**.  Breakdown: bottom-mid 8 robust designs, bottom-right 1,
US107-inspired 3, US923-inspired 1.  All 40 audits replay exactly and all budget logs
match.  The central machine-readable stock is
`runs/diverse_stock_dense_search_20260911_v1/stock_manifest.json`; see the
[gallery](runs/diverse_stock_dense_search_20260911_v1/stress_verified_stock.png) and
[full report](agent_context/ns_systems/diverse_stock_20260911.md).

### Patent-inspired detector layouts — 2026-09-11

Completed two proposer-only development datasets inspired by US9239453B2 and
US10739578B2: 18 jobs per layout, seeds 306–311, 3/4/5 corners, 400 physical calls per
job, and 194,400 search rays per layout.  The shared evaluator now supports directed
vertical or horizontal detector planes while preserving old protocols.  Actual curved-
boundary clearance bounds are enforced; no blocker is used.  US923-inspired: 1/18 jobs,
two stored audit passes on one route.  US107-inspired: 0/18 primary successes; one
post-run dense triangle pass was not audit-selected.  Both selected examples fail the
909-ray root-support gate, so neither is a robust design.  All stored audits replay
exactly and all 36 budget logs match.  See
[full report](agent_context/ns_systems/patent_layouts_20260911.md),
`runs/patent_923_layout_20260911_v2/`, and
`runs/patent_107_layout_20260911_v2/`.

### Detector 2.5 × beam width — 2026-09-11

Completed another 144 runs using the same seeds/settings as the 1.5× comparison,
changing only detector full height to 10/15 mm (4/6 mm input beams). Centre stays
(120,-20) mm; n=1.60. Portfolio **3/36 (8.3%)** successful runs, five passing audits;
naive, legacy-route adaptation and guided CMA **0/36 each**. No confirmed gain:
95% seed-block interval [0,+19.4] points, p=0.25. All five audits replay exactly;
all budgets checked. Two selected examples pass 909-ray checks with converged
larger branch capacities (one failed at the default cap). See
[full report](agent_context/ns_systems/detector_2p5x_20260911.md) and
`runs/detector_2p5x_20260911_v1/folded_examples.png`.

### Small-detector experiment — 2026-09-11

Completed 144 paired runs: 6 seeds (300–305), 3/4/5 corners, 4/6 mm beams,
four CMA policies. Detector full height = 1.5 × beam width (6/9 mm), x=120 mm,
centre y=-20 mm: one-third up the explicitly fixed reference envelope [-55,50] mm.
Material n=1.60. Each run used 400 search calls with equal reserved dense audits.
Portfolio: **1/36** successful runs; naive, historical-ranking route adaptation,
and earlier guided: **0/36 each**. **No confirmed gain** (95% seed-block gain
interval [0,+8.3] percentage points; p=0.5). The sole triangle passed 125-ray audit
and 909-ray stress verification. All 144 budgets checked; audit replay exact;
18 tests passed. The historical control adapts archive ranking/restarts to the
common route archive; it is not the old grid driver unchanged.
See [protocol and results](agent_context/ns_systems/small_detector_20260911.md)
and `runs/small_detector_20260911_v1/RESULTS.md`. The previous broad-detector
baseline below remains valid only for its original environment.

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

## Local Qwen server (GPU 1)

Qwen3.8-27B Q8 and a CUDA-enabled llama.cpp build live under `/playpen/hisham26`.
Start the local chat UI and OpenAI-compatible API with:

```bash
./start_qwen.sh
```

Then open `http://127.0.0.1:8080`. The API base URL is
`http://127.0.0.1:8080/v1`, and the model name is `qwen3.8-27b`.
The launcher uses GPU 1 and a 32k-token context window. Thinking is enabled with
an unrestricted reasoning-token budget; API responses expose it separately as
`reasoning_content`. The chat UI expands the complete reasoning block by default.
If the server is already healthy, the launcher prints its URL and exits successfully;
use `./start_qwen.sh --restart` to reload changed launcher or UI settings.
The expanded-after-completion behavior is a local change in
`/playpen/hisham26/llama.cpp/tools/ui/src/lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte`;
rebuild the UI and `llama-server` after replacing that llama.cpp checkout.

AZR resumption assessment and staged experiment:
[`agent_context/ns_systems/RLVR_AZR.md`](agent_context/ns_systems/RLVR_AZR.md).
The existing learning pilot uses Qwen3-8B through Transformers; the new GGUF server
still needs a tested workbench API connection and a separate training strategy.

Current direction: use **only the new Qwen** for future LLM work. The 8B pilot is
historical. Prioritize numerical starting-point discovery, then Qwen-guided compaction
and refinement. See the [proposer-method assessment](agent_context/ns_systems/proposer_assessment_20260911.md)
for CMA diagnostics, MetaDistill/EPOM/ConfigX/L2E/LQD comparison, and the proposed experiment.

Confirmed direction (2026-09-11): **strengthened numerical discovery → new-Qwen-guided
compaction and refinement → physical verification**. Full AZR, large-dataset end-to-end
training, and learned optimizer replacements are deferred.

The numerical discovery baseline milestone is now complete (results below). Next,
replace arbitrary compaction tradeoffs with explicit optical/package requirements
and connect the new Qwen to
bounded optical tools and compare its decisions against the strengthened numerical
compactor on identical starting designs. Initial Qwen evaluation uses frozen weights;
fine-tuning is not a prerequisite. Success means better verified final designs or fewer
traces to reach the same requirements, with failures and language costs included.

Discovery scope clarified: prioritize better geometry and curvature bounds, **3/4/5
corners**, finite-width beams, and interesting folded paths. The active prototype is
[`nslens/discovery`](nslens/discovery/README.md): cyclic polygon construction with
chord-relative curvature and minimum radius, explicit multi-ray TIR requirements,
random/flat-CMA/feedback-CMA controls, and dense verification. Initial pilot assumptions
use a 4 mm beam and at least two TIR events; this is routing discovery before imaging
or compaction. It replaces neither the historical experiments nor their acceptance
claims. See the package README for reproducible commands and parameter bounds.

Executed pilot: 8,100 candidates over 3/4/5 corners, three policies, and three seeds.
Dense routing checks passed for examples in all three shape families. Feedback CMA
retained 6 passing route representatives across its runs, random search 5, flat-rejection
CMA 0. These are development results on one brief, not globally novel topology counts
or an established optimizer gain. [Results and limitations](agent_context/ns_systems/polygon_discovery_results_20260911.md).
Ray layouts: `runs/polygon_discovery_pilot_20260911_v2/discovery_examples.png`.

### Confirmed discovery baseline — 2026-09-11

Use **retained-candidate CMA** (`portfolio`) for new discovery. Frozen confirmation:
**41/72 successful runs (56.9%)**, versus naive CMA **12/72 (16.7%)** and the earlier
guided CMA **24/72 (33.3%)**. Gain over naive: **+40.3 percentage points**, 95% paired
seed-block interval **[+26.4, +55.6]**, exact one-sided seed-sign p=0.000488.
Twelve fresh seeds, 3/4/5 corners, 4/6 mm beams; equal search and reserved audit budgets.
This measures verified folded-routing discovery yield within the new environment.

The policy ranks all optical constraint violations, seeds with 40 Sobol proposals,
retains twelve diverse measured candidates including failures, and uses CMA sigma
0.15 in [-1,1], population 10, reflected bounds and five-generation restart patience.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nslens.discovery.search \
  --out runs/discovery_NEW --seeds 200 201 202 --corners 3 4 5 \
  --briefs base wide --search-calls 400 --workers 6
```

All 228 selected-policy audits replayed exactly; all 288 run budget logs matched.
Six selected examples also passed 909-ray checks after increasing the bundle-wide
branch capacity; 200k/400k capacities agreed exactly. The initial 20k-cap failures
are preserved and explained in the report. No imaging, compaction or independent
engine claim follows from these checks.

[Full results, costs, limitations and replay commands](agent_context/ns_systems/discovery_improvement_20260911.md).
[Folded-path examples](runs/discovery_improvement_confirm_20260911_v1/folded_examples.png).
Replayable starts: `runs/discovery_improvement_confirm_20260911_v1/example_verification.json`.

## Verified 2026-07-10

`ns_smoke_test.py` runs (Node bridge OK, detector power=1.0); `ns_probe` and `ns_autodiff`
imports resolve from this copy.

### Five-design agentic correction pilot — 2026-09-11

Completed local Qwen3.8-27B control of bounded CMA batches on two triangles and
three quadrilaterals: three physically failing starts and two verified starts.
Each design received two 40-candidate batches for each independent repair,
5% longest-body-side compaction, and 2-degree full-FOV probe. Source/detector,
aperture, material and original codec bounds stayed fixed; neither body dimension
could grow for acceptance. No spot/freeform optimization was included.

All 30 batches completed: 1,200 candidates, 270,000 search rays, 3,375 baseline
rays, and 54,540 final audit/replay/capacity-check rays. All numerical CMA
populations replayed exactly, episode budgets reconciled, and final 909-ray optical
replays and 200k/400k branch-capacity checks agreed. Eighteen relevant tests pass.

No final design met the complete 5% compaction requirement. The weak triangle's
repair probe failed, but its compaction branch recovered routing with 2.0% shrinkage
and dense support improving from 34.7% to 66.3%. The good quadrilateral retained
routing at 4.1% shrinkage. The good triangle's 4.6% shrinkage failed dense routing.
The near-failing quadrilateral passed repair; its search-grid field success failed
dense audit. The good triangle's 2-degree field pass was already present initially.

Capacity correction to the earlier stock count: all 40 retained designs were
screened at 200k branches; 39 passed, versus 13 at the original 20k cap. Only the
five selected pilot starts additionally received 400k label checks. Preserve the
original campaign; do not describe its 27 failed stress checks as all physical
optical failures.

This was a development pilot with inference amendments: the server's shared
context pool was increased to 131072 tokens (four slots, 32768 per slot), and
thinking was later capped at 1536 tokens after repeated 4096-token responses
returned no action. Completed batches were preserved. Failed/interrupted early
response usage is incomplete; no frozen-policy or clean language-cost comparison
is claimed. See the amendment JSON files in the run directory. The original
start_qwen.sh remains unchanged; the running server uses the amended settings.

- Report: `runs/agentic_stock_probe_20260911_v1/REPORT.md`
- Portable fresh tasks: `runs/agentic_stock_probe_20260911_v1/frontier_tasks/README.md`
- Model-neutral workbench: `nslens/workbench/stock_probe.py`
- Selection / audit / report scripts: `nslens/analysis/{prepare,audit,report}_stock_probe.py`

Frontier tasks use the same request.json/action JSON contract and bounded CMA
executor, through either CLI tools or a compatible HTTP transport. No frontier
API has been called. Run fresh copies under one declared inference setting for
future comparisons; this five-start pilot does not establish ultimate potential.
