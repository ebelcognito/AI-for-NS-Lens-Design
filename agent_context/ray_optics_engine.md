# Non-Sequential Ray Tracer + Node.js Infra — Agent Map

## TL;DR / key facts (do not get these wrong)

1. **"NS ray tracer" means two different things in this repo.** Only one exists as running code:
   - ✅ **WORKING:** the `ray-optics/` Node.js project — a 2D **non-sequential** geometric tracer (the open-source [Ray Optics Simulation](https://phydemo.app/ray-optics/) by ricktu288, vendored in). This is "the node js infra." It is **not differentiable**.
   - 🚧 **PLANNED, NOT BUILT:** a **differentiable PyTorch** non-sequential tracer for `lensmega` (`problem/ns/`). Design doc only; `problem/ns/` does **not exist yet**. See [optical_rl/qd/lensmega/docs/ns_extension_context.md](../optical_rl/qd/lensmega/docs/ns_extension_context.md). Status in that doc: *"paused. Planning complete; implementation not started."*
2. Python talks to the Node engine over **stdin/stdout JSON** via a subprocess bridge. There is no FFI / no shared memory.
3. The scene format is the **exact same JSON the web app saves** (`{"version": 5, "objs": [...]}`). You can paste it into the web app's JSON editor to debug.
4. For loops (RL/QD eval), use the **persistent** runner (`PersistentRayOpticsRunner`) — it keeps one Node process alive over NDJSON to avoid ~Node-startup cost per call. Use the one-shot runner only for scripts.
5. Non-sequential = rays hit the **nearest** surface, with internal reflections / beamsplitters; the simulator tracks an infinite-internal-reflection series truncated at a brightness threshold (`totalTruncation` stat).

---

## Component map

### A. The engine core — `ray-optics/src/core/`
The actual physics. Authored in ES modules; bundled for runtime use.

| File | Role |
|------|------|
| [Simulator.js](../ray-optics/src/core/Simulator.js) | The ray-propagation loop: emit rays → find nearest intersection → apply refraction/reflection → recurse internal reflections until truncation threshold. Emits `simulationComplete`. Exposes `totalTruncation`, `processedRayCount`, `brightnessScale`. |
| [Scene.js](../ray-optics/src/core/Scene.js) | Loads/holds a scene from JSON (`loadJSON`), owns `objs`, `scale`, `origin`, `theme`. |
| [geometry.js](../ray-optics/src/core/geometry.js) | Vector / intersection math primitives. |
| [sceneObjs/](../ray-optics/src/core/sceneObjs/) | The object types (see below). |
| [CanvasRenderer.js](../ray-optics/src/core/CanvasRenderer.js) / [FloatColorRenderer.js](../ray-optics/src/core/FloatColorRenderer.js) | Rendering backends (used for PNG export via node-canvas). |

**Scene object types** — [ray-optics/src/core/sceneObjs/](../ray-optics/src/core/sceneObjs/):
- `lightSource/` — `Beam`, `SingleRay`, `PointSource`, `AngleSource` (how you inject rays)
- `glass/` — `Glass` (spherical/arc), `PlaneGlass`, `ParamGlass`, GRIN variants (lenses; carry refractive index)
- `mirror/` — mirrors, parabolic/arc, beam splitters, diffraction gratings
- `blocker/` — `Aperture`, `Blocker`, `CircleBlocker` (vignetting / stops)
- `other/` — **`Detector`** (reads power/momentum + irradiance map), **`CropBox`** (defines a PNG/SVG export region)

**Runtime bundle:** [ray-optics/integrations/rayOptics.js](../ray-optics/integrations/rayOptics.js) — a ~1.2 MB webpack bundle of `src/core`. This is what the integration layer `require()`s; you do **not** edit it by hand (rebuild from `src/` — see `ray-optics/webpack.*.config.mjs`).

### B. The integration / bridge layer — `ray-optics/integrations/`
The JS side of the Python↔Node boundary.

| File | Role |
|------|------|
| [simulation_core.js](../ray-optics/integrations/simulation_core.js) | `runSingleSimulation(sceneJson)`: builds `Scene` → runs `Simulator` → extracts per-detector readings (`power`, `normal` = F⊥, `shear` = F∥, optional `irradianceMap`/`binPositions`) and optional PNGs (per `CropBox`, needs node-canvas) → returns `{detectors, images, error, warning, totalTruncation, processedRayCount, brightnessScale, timing}`. |
| [runner.js](../ray-optics/integrations/runner.js) | **One-shot**: read one scene JSON on stdin → print one result JSON on stdout. One Node startup per call. |
| [runner_persistent.js](../ray-optics/integrations/runner_persistent.js) | **Persistent (NDJSON)**: long-lived loop, one JSON request per line, `QUIT\n` to stop. Echoes back `episode`/`step` if present. Used for training/QD. |
| [README.md.template](../ray-optics/integrations/README.md.template) | Upstream integration docs: scene JSON format, detector fields, persistent protocol, image gen, stats. Authoritative for the **protocol**. |
| [example_python.py](../ray-optics/integrations/example_python.py) / `example_julia.jl` | Minimal usage examples. |

### C. The Python side — `optical_rl/utils/`
How `optical_rl` actually drives the engine.

| File | Role |
|------|------|
| [optical_rl/utils/ray_optics_bridge_persistent.py](../optical_rl/utils/ray_optics_bridge_persistent.py) | **`PersistentRayOpticsRunner`** — the workhorse. Spawns `node runner_persistent.js`, writes scene dicts as NDJSON, reads results, with `select`-based timeout + auto-restart on timeout, `atexit` cleanup, per-call timing. Use as context manager: `with PersistentRayOpticsRunner() as r: result, timing = r.simulate(scene_dict, episode, step)`. |
| [optical_rl/utils/ray_optics_bridge.py](../optical_rl/utils/ray_optics_bridge.py) | One-shot `run_simulation(scene_dict)` (spawns `node runner.js` per call). For scripts, not loops. |

**Consumers** (grep `runner_persistent\|ray_optics_bridge\|rayOptics`): the QD/RL `main_*` scripts, `render_scenes.py`, `visualize_policy_*.py`, and tests including [optical_rl/tests/test_persistent_runner.py](../optical_rl/tests/test_persistent_runner.py).

---

## Data flow (one evaluation)

```
Python (build scene dict)
   └─ json over stdin ──▶ runner_persistent.js  (NDJSON, one line)
                              └─ simulation_core.js: runSingleSimulation(sceneJson)
                                     └─ Scene.loadJSON ──▶ new Simulator(scene)
                                            └─ updateSimulation() traces rays
                                                   └─ 'simulationComplete'
                                     ◀── {detectors, images, stats, timing}
   ◀─ json over stdout ── runner_persistent.js
Python parses result, reads detector power / irradiance map
```

Scene dict shape:
```json
{"version": 5, "objs": [
  {"type": "Beam",     "p1": {"x":0,"y":-25}, "p2": {"x":0,"y":25}, "brightness": 0.5},
  {"type": "Glass",    "...": "spherical lens surfaces, refractive index"},
  {"type": "Detector", "p1": {"x":240,"y":-50}, "p2": {"x":240,"y":50}, "irradMap": true, "binSize": 20}
]}
```
Beam direction is **perpendicular** to (p1→p2). Detector returns `power` and (if `irradMap`) an `irradianceMap`/`binPositions` pair.

---

## Entry points (where to start)

- **Run it end-to-end / sanity check:** [ns_smoke_test.py](../ns_smoke_test.py) (repo root) — collimated beam → biconvex lens (R±50, d5, n1.5) → detector at the focal plane (~f=50). Exercises both one-shot and persistent runners and prints timing. Best first thing to run.
- **Drive from Python in a loop:** copy the `with PersistentRayOpticsRunner() as r:` pattern from [optical_rl/tests/test_persistent_runner.py](../optical_rl/tests/test_persistent_runner.py).
- **Protocol / detector field reference:** [ray-optics/integrations/README.md.template](../ray-optics/integrations/README.md.template).
- **Change the physics:** edit `ray-optics/src/core/` then rebuild the bundle (`ray-optics/webpack.node.config.mjs`); the bridge loads the rebuilt [rayOptics.js](../ray-optics/integrations/rayOptics.js).
- **Typed interaction sequences + energy ledger (NS instrumentation):** scene flag `recordInteractions: true` returns `result.raySequences` (per detector crossing, every event typed `refract|tir|partial_reflect|mirror|sensor|...` with per-event energies) and `result.energyLedger` (emitted = delivered + escaped + absorbed + truncated, exact closure). Tooling: [ns_probe/sequence_tools.py](../ns_probe/sequence_tools.py); validation: `python3 ns_probe/test_typed_interactions.py`; demonstrator: `python3 ns_probe/fold_demo.py`. Full account: [ns_rl_research/07_typed_interactions_energy_ledger.md](ns_rl_research/07_typed_interactions_energy_ledger.md). NOTE: `Glass` defaults to `partialReflect: true` (full Fresnel).

---

## Gotchas

- **Requires Node.js on PATH.** No Node → bridge raises. PNG export additionally needs `npm install canvas` inside `ray-optics/integrations/` (node-canvas).
- **Edit `src/core/`, not the bundle.** `integrations/rayOptics.js` is generated; hand-edits get blown away on rebuild.
- **Persistent runner serializes requests** (`queue` promise chain in `runner_persistent.js`) — it processes one line at a time. Parallelism = multiple Node processes / multiple `PersistentRayOpticsRunner` instances, not concurrency within one.
- **Timeouts auto-restart the Node process** (`PersistentRayOpticsRunner._read_response_line` → `restart()`), so a hung scene kills+respawns rather than wedging the run.
- **This engine is non-differentiable & 2D.** If a task needs gradients through the trace (e.g. CMA-MEGA for lensmega), this engine cannot provide them — that is exactly why the PyTorch NS tracer is planned separately.

---

## The planned PyTorch NS tracer (context, not yet code)

For the lensmega pancake-discovery thesis you need a **differentiable** non-sequential tracer (autograd through refract/reflect/branch for gradient-based QD emitters). Full design — surfaces/rays/propagator layout, decisions, roadmap, success criteria, anti-patterns — lives in:

**[optical_rl/qd/lensmega/docs/ns_extension_context.md](../optical_rl/qd/lensmega/docs/ns_extension_context.md)**

Planned files (do not assume they exist): `optical_rl/qd/lensmega/problem/ns/{surface,ray,propagator,validation}.py`, `problem/template_ns.py`, `descriptors/topology.py`. First runnable target in that doc: trace a Mangin mirror and recover textbook focal length. Until those are written, the Node engine above is the only ray tracer in the repo.
