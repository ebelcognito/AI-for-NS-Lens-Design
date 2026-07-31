# CODE V export — session context / handoff

Status as of 2026-07-24. Goal: export the non-sequential fold/prism designs (freeform
TIR elements) from this repo into CODE V so they can be viewed and optimised there.

## 1. What the source design is

- 2-D **meridional (x,y)** non-sequential model, traced by the ray-optics JS engine.
- Family the user cares about = the **collinear fold** ([`ns_probe/fold_demo.py`](../../fold_demo.py),
  archive [`ns_probe/results_fold_qd/archive.json`](../archive.json)). The user pointed at
  `results_fold_qd/renders/champ_tir3.png`.
- **One glass element, 4 faces**: A=left (curved arc, tilt_a), B=right (curved arc, tilt_b),
  T=bottom edge (flat, y=-30), D=top edge (flat, y=+30). Face labels come from
  `sequence_tools.make_single_element_face_classifier` (y<=-H→'T', y>=+H→'D').
- Genotype (9-DOF): `[ray_y, ang, cx, thick, fsag, bsag, tilt_a, tilt_b, n]`.
- Constant index `n` (monochromatic, NO dispersion). Input = collimated beam, 21 rays,
  width `beam_w=8`, entering at x=0 centred on y=ray_y, travelling at `ang` deg above +x.
  Beam TIR-folds and exits to a detector plane at x=`X_SENSOR`=120.
- Each cell stores `dom_signature`, e.g. `T@A -> TIR@D -> TIR@B -> TIR@B -> T@B`
  (T=refract, TIR=total internal reflection).

## 2. What was built

[`ns_probe/export_fold_for_codev.py`](../../export_fold_for_codev.py) — reads a fold_qd
archive, selects designs, and emits per design:
- `<id>.spec.json` / `.spec.txt` — exact geometry table (SOURCE OF TRUTH).
- `<id>.seq` — best-effort CODE V macro.

Selection: `--tir N` (champion at N bounces), `--all-champs`, `--top N`, `--cell I0 I1`.
Default is the TIR-3 champion (the design the user pointed at).

## 3. Coordinate mapping (verified correct)

```
CODE V  Z = engine x   (optical axis / propagation)
CODE V  Y = engine y   (meridional transverse)
CODE V  X = out of plane  → curved faces emitted as SPHERES (meridionally identical
                            to the true cylinders; a sphere & same-radius cylinder trace
                            identically in the meridional plane)
```
- Each curved face: `RDY` = arc radius via circumcircle of (corner, y=0 apex, corner).
- Global build: every surface placed by absolute `YDE/ZDE` vertex + `ADE` tilt, `THI 0`,
  **decenter-and-return (DAR)** → all surfaces in one frame. TIR hits → `TIRO` (TIR-only,
  stays in glass). Refract hits toggle air↔glass. Same physical face reused N times is
  emitted N times identically.
- Object = infinity; single field `YAN = ang`; beam via `EPD`; input offset via decentered
  aperture stop S1 at engine x=0.

**Geometry was cross-checked against the engine's own ray trace and matches** (surface
vertices land exactly on the engine's dominant-ray hit points — see §6).

## 4. CODE V syntax findings (from live traces — IMPORTANT)

CONFIRMED VALID (these work in CODE V 2026):
- `S <rdy> <thi> [glass | REFL | TIRO [glass]]` — append surface, **NO surface index**.
  (Bug fixed: earlier we passed the index as the radius.)
- `SO <rdy> <thi>` — object; infinity = `SO 0 1e18`. (`SO INFINITY` is INVALID.)
- `TIRO 'FOLDN'` — TIR-only surface staying in glass (the right tool for uncoated TIR folds).
- Constant-index model glass via private catalog: `PRV` / `PWL 587.56` / `'FOLDN' 1.500000` / `END`. WORKS.
- `EPD <d>`, `YAN <ang>`, `WL <nm>`, `DIM M`, `REF 1`.
- Per-surface: `YDE Sn`, `ZDE Sn`, `ADE Sn`, `DAR Sn`, `CIR Sn <semi-ap>`, `STO Sn`, `SI <rdy> <thi>`.
- `CRA [Sk] [Wk] [Zk] [Fk] x_int y_int [x_aim y_aim]` — chief-ray aiming. Requires a field
  qualifier (e.g. `CRA S1 W1 Z1 F1 0 0`).

CONVENTION FINDINGS:
- **`ADE` must be NEGATED** relative to the engine tilt. Derived analytically: CODE V's
  "+ADE tilts +Y toward +Z" is opposite this repo's plane mapping. (Confirmed direction;
  did not by itself fix the trace — see §5.)
- `YAN` sign undetermined (tried both).

## 5. THE BLOCKER (unresolved)

The lens **imports correctly** (all 8 surfaces, radii, FOLDN glass, "Only TIR" modes,
Decenter & Return all show right in the Lens Data Manager). But CODE V **cannot find /
complete the chief (reference) ray** through the fold:

    Error: Unable to find chief ray: F1,W1,Z1.  Use CRA command.

Diagnosis: the **global-decenter (DAR) construction with an object at infinity gives CODE
V's first-order/paraxial reference-ray solver no coherent optical axis to follow** (all
surfaces return to one base frame sitting 1e18 away → paraxial angles underflow). This is
NOT a sign bug — all 4 ADE/YAN sign combos + generous apertures + `CRA` were tried; all fail.

Progress made: turning on **System Data → Advanced → Reference Ray Aiming** ("Center Chief
Ray...") got CODE V to trace **real rays for the first time** — but they only reach the
first face; the 4-bounce reference ray still won't complete. Element geometry draws
correctly in View Lens.

Root conclusion: the global-DAR approach is likely a dead end for CODE V's chief-ray
machinery on a multi-bounce curved-TIR fold. The CODE-V-native fix is to thread the axis
THROUGH the fold (real thicknesses between hit points + folding mirror surfaces), not
global DAR. Complication: the curved faces are hit far off their vertices, so folded-axis
tilts must use the hit-point normals (from the curvature centre), not the vertex tilt.

## 6. Reference data — chief-ray fold trace for cell 4_11 (the test design)

Genotype: `ray_y=9.4592, ang=23.8848, cx=58.3601, thick=28.6574, fsag=-3.9085,
bsag=2.5657, tilt_a=25.0, tilt_b=15.6382, n=1.5`
dom_signature: `T@A -> TIR@D -> TIR@B -> TIR@A -> TIR@T -> T@B`

Chief (beam-center) ray path, engine (x,y) = CODE V (Z,Y):
```
P0 source        ( 0.000,  9.459)
P1 refract @A    (34.021, 24.524)   seg P0→P1: L=37.207  dir=  23.885  (air, into A)
P2 tir     @D    (44.524, 30.000)   seg P1→P2: L=11.845  dir=  27.535  (glass, into D top)
P3 tir     @B    (69.784, 16.831)   seg P2→P3: L=28.487  dir= -27.535  (into B)
P4 tir     @A    (55.407,-20.004)   seg P3→P4: L=39.541  dir=-111.321  (into A)
P5 tir     @T    (69.378,-30.000)   seg P4→P5: L=17.178  dir= -35.584  (into T bottom)
P6 refract @B    (80.056,-22.360)   seg P5→P6: L=13.130  dir=  35.584  (into B)
P7 sensor        (120.000, 27.545)  seg P6→P7: L=63.922  dir=  51.326  (air, to sensor)
```
D and T are horizontal-mirror reflections (flip the y-component of direction). B and A are
the curved reflections. Regenerate this via the trace block used in the session (n_rays=1,
beam_w=0, PersistentRayOpticsRunner, fold_demo.scene).

## 7. Open decision (was mid-question when session ended)

Three paths were offered:
1. **Start with a 1-bounce design** (simplest fold, ~3 surfaces) to get CODE V tracing
   end-to-end and validate the workflow, then scale up. (Recommended — fastest real win.)
2. **Rebuild the 4-bounce as folded-axis** (real thicknesses + folding mirrors threading
   the axis, not global DAR) — targets the root cause but intricate; will likely need
   iteration since CODE V can't be driven from the assistant side.
3. **Finish the current import interactively in CODE V** (assistant guides pupil/reference-
   ray setup click-by-click; user drives).

Note the deeper reality: this last mile needs someone AT the CODE V keyboard — the
assistant has no CODE V to test against, so blind `.seq` iteration is slow. The `.spec.txt`
files give exact, unambiguous geometry for hand/interactive building.

## 8. Key files
- Exporter: `ns_probe/export_fold_for_codev.py`
- Outputs: `ns_probe/results_fold_qd/codev_export/*.{seq,spec.txt,spec.json}`
- Source geometry: `ns_probe/fold_demo.py`, `ns_probe/sequence_tools.py`
- Archive: `ns_probe/results_fold_qd/archive.json`
- Engine bridge: `optical_rl/utils/ray_optics_bridge_persistent.py` (needs Node; see RUN_NOTES.md)
