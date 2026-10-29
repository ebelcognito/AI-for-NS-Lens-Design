"""Spherical-only novel-path inception followed by sequence-preserving compaction.

This is the first executable form of the Inception -> Compaction boundary described in
``agent_context/ns_systems/03_future_directions_interaction_sequence_discovery.md``.

The run freezes every pre-existing archive signature before proposing anything.  A
candidate is called new only when its surface-name-invariant interaction topology clears
the run's declared Levenshtein-edit and normalized-distance thresholds against that
frozen inventory.  It must then pass the project's hardened gates at every declared
field plus explicit intended-sequence support gates.  Compaction is allowed to change
spherical geometry but not the sequence.

Example::

    python ns_probe/run_novel_compaction.py \
      --out runs/novel_compaction_20260904_v1 --evals 12000 --workers 12
"""
from __future__ import annotations

import argparse
import atexit
import base64
import glob
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nslens.engine import Runner  # noqa: E402
from nslens.qd import experiments  # noqa: E402
from nslens.qd.driver import _evaluate  # noqa: E402
from nslens.qd.driver import ExperimentConfig  # noqa: E402
from nslens.qd.multi_element import TwoElementSphericalProblem  # noqa: E402
from nslens.qd.novelty import TopologyExclusionSet  # noqa: E402


SCHEMA = "nslens.novel_spherical_compaction.v1"
DEFAULT_ANCHORS = "runs/**/archive.json"

_RUNNER: Runner | None = None
_PROBLEM = None
_CFG = None
_EXCLUSION: TopologyExclusionSet | None = None
_MIN_FITNESS = 0.0
_MODE = "discovery"
_TARGET = ""
_MIN_COMPACTION_FITNESS = 0.0


FIELDS = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)


def _problem_and_cfg(design_space: str, fields: Sequence[float] = FIELDS):
    if design_space == "single_3curved":
        cfg = experiments.get("fold24_topo3_gated")
        problem = cfg.problem
    elif design_space == "two_element":
        problem = TwoElementSphericalProblem()
        cfg = ExperimentConfig(
            name="two_element_spherical_inception",
            problem=problem,
            fitness="focused_throughput",
            fields=tuple(fields),
            field_aggregate="worst",
            descriptors=("tir_bounces", "thinness"),
            bc_bins=(16, 16), bc_bounds=((0.0, 16.0), (0.0, 122.0)),
            require_monotone_convergence=True,
            convergence_gate="clean_pack", gate_all_fields=True,
            require_exit_convergence=True,
            info_descriptors=("field_accept", "beam_convergence"),
        )
    else:
        raise ValueError(f"unknown design space {design_space!r}")
    # The historical archive cutoff at five TIR events is never an inception gate.
    return problem, replace(
        cfg,
        fields=tuple(fields),
        descriptors=("tir_bounces", "thinness"),
        bc_bins=(16, 16), bc_bounds=((0.0, 16.0), (0.0, 122.0)),
        reject_out_of_bounds=False, cell_bc_keys=("tir_count", "thinness_T"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _archive_signatures(root: Path, excluded_root: Path) -> tuple[set[str], list[dict]]:
    signatures: set[str] = set()
    sources = []
    paths = sorted(root.glob("runs/**/archive.json"))
    catalogue = root / "catalogue" / "catalogue.json"
    if catalogue.exists():
        paths.append(catalogue)
    for path in paths:
        try:
            resolved = path.resolve()
            if resolved == excluded_root or excluded_root in resolved.parents:
                continue
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        before = len(signatures)
        rows: Iterable[Mapping[str, Any]] = (
            payload.get("entries", []) if path == catalogue else payload.get("cells", []))
        for row in rows:
            for key in ("signature", "hardened_signature", "dom_signature",
                        "intended_sequence"):
                value = row.get(key)
                if isinstance(value, str) and "@" in value:
                    signatures.add(value.strip())
        if len(signatures) > before:
            sources.append({
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "new_signatures": len(signatures) - before,
            })
    return signatures, sources


def _anchors(pattern: str, problem) -> tuple[list[np.ndarray], list[dict]]:
    if isinstance(problem, TwoElementSphericalProblem):
        return _two_element_anchors(problem)
    anchors: list[np.ndarray] = []
    provenance = []
    for filename in sorted(glob.glob(str(REPO / pattern), recursive=True)):
        path = Path(filename)
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("meta", {}).get("param_names") != list(problem.param_names):
            continue
        accepted = 0
        for cell in payload.get("cells", []):
            genotype = np.asarray(cell.get("genotype", []), float)
            if genotype.shape == np.asarray(problem.lo).shape:
                anchors.append(genotype)
                accepted += 1
        if accepted:
            provenance.append({"path": str(path.resolve()), "sha256": _sha256_file(path),
                               "anchors": accepted})
    if not anchors:
        raise RuntimeError(f"no compatible spherical anchors matched {pattern!r}")
    # Parameter-identical cells across seeds add no proposal diversity.
    unique = {tuple(np.round(anchor, 10)): anchor for anchor in anchors}
    return list(unique.values()), provenance


def _two_element_anchors(problem: TwoElementSphericalProblem
                         ) -> tuple[list[np.ndarray], list[dict]]:
    """Lift yesterday's densely audited intentional designs into a second-body space.

    The added body starts as an unpowered, untilted spherical-limit slab just beyond the
    first body.  Search—not the lift—must make it participate in a new fold.  The passive
    two-refraction extension is only two edits from the old path and cannot clear the
    default three-edit novelty gate.
    """
    audit = REPO / "runs/fixed_baseline_audit_20260903_v1/strict_intentional.jsonl"
    specs_path = REPO / "runs/environment_pilot_20260903_v2/environment_specs.json"
    specs_payload = json.loads(specs_path.read_text())
    specs = {row["environment_id"]: row for row in specs_payload["environments"]}
    audited = [json.loads(line) for line in audit.read_text().splitlines()]
    donors = audited[::max(1, len(audited) // 8)]
    anchors = []
    for row in audited:
        spec = specs[row["fixed_environment_id"]]
        fixed, design = spec["fixed"], row["design"]
        prefix = [
            fixed["ray_y"], fixed["ang"], fixed["n"],
            design["cx"], design["thick"], design["fsag"], design["bsag"], 0.0,
            design["tilt_a"], design["tilt_b"],
        ]
        probe = np.asarray(prefix + [80.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        first = problem.elements(problem.vec_to_params(probe))[0]
        left = max(float(problem.lo[10]), first.x_extent()[1] +
                   problem.inter_element_clearance + 2.1)
        for extra_gap in (0.0, 4.0, 10.0):
            cx2 = min(float(problem.hi[10]), left + extra_gap)
            genotype = np.asarray(prefix + [cx2, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            genotype = np.clip(genotype, problem.lo, problem.hi)
            if problem.is_valid(problem.vec_to_params(genotype)):
                anchors.append(genotype)
        # Also seed the added body with spherical powers/wedges from independently
        # audited folds.  A passive slab cannot invent an active second-body motif;
        # these cross-combinations put that motif within a topology-changing mutation.
        for donor in donors:
            shape = donor["design"]
            cx2 = min(float(problem.hi[10]), left + 4.0)
            suffix = [cx2, min(25.0, max(4.0, shape["thick"])),
                      shape["fsag"], shape["bsag"], 0.0,
                      shape["tilt_a"], shape["tilt_b"]]
            genotype = np.clip(np.asarray(prefix + suffix), problem.lo, problem.hi)
            if problem.is_valid(problem.vec_to_params(genotype)):
                anchors.append(genotype)
    unique = {tuple(np.round(anchor, 10)): anchor for anchor in anchors}
    provenance = [{
        "path": str(audit.resolve()), "sha256": _sha256_file(audit),
        "anchors": len(unique), "operation": "add separated passive spherical body",
    }, {"path": str(specs_path.resolve()), "sha256": _sha256_file(specs_path)}]
    if not unique:
        raise RuntimeError("no valid two-element anchors could be lifted")
    return list(unique.values()), provenance


def _worker_close() -> None:
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.__exit__(None, None, None)
        _RUNNER = None


def _worker_init(signatures: Sequence[str], min_edits: int, min_distance: float,
                 min_fitness: float, mode: str, target: str,
                 min_compaction_fitness: float, design_space: str,
                 fields: Sequence[float]) -> None:
    global _RUNNER, _PROBLEM, _CFG, _EXCLUSION, _MIN_FITNESS, _MODE, _TARGET
    global _MIN_COMPACTION_FITNESS
    _PROBLEM, cfg = _problem_and_cfg(design_space, fields)
    _CFG = cfg
    _EXCLUSION = TopologyExclusionSet(
        signatures, min_edits=min_edits, min_normalized_distance=min_distance)
    _MIN_FITNESS = float(min_fitness)
    _MODE, _TARGET = mode, target
    _MIN_COMPACTION_FITNESS = float(min_compaction_fitness)
    if mode == "discovery":
        # Inception may cross invalid intermediate basins.  Do not delete a genuinely
        # new route before its explicit intended-path validation; the second evaluation
        # below reapplies every hardened gate and is the only route to `new_viable`.
        _CFG = replace(_CFG, require_monotone_convergence=False,
                       gate_all_fields=False, require_exit_convergence=False)
    if mode == "target_refine":
        _CFG = replace(
            _CFG, name="novel_target_refinement",
            fitness="intended_focused_throughput", intended_sequence=target,
            min_intended_root_support=0.0,
            min_intended_delivered_energy_share=0.0,
            require_intended_dominant_all_fields=False,
            min_intended_critical_margin_deg=None,
            require_monotone_convergence=False, gate_all_fields=False,
            require_exit_convergence=False,
        )
    if mode == "compaction":
        _CFG = replace(
            cfg,
            name="sequence_preserving_compaction",
            fitness="intended_focused_throughput",
            intended_sequence=target,
            min_intended_root_support=0.80,
            min_intended_delivered_energy_share=0.05,
            require_intended_dominant_all_fields=True,
            min_intended_critical_margin_deg=0.50,
        )
    _RUNNER = Runner(timeout_s=60)
    _RUNNER.__enter__()
    atexit.register(_worker_close)


def _package_metrics(genotype: Sequence[float], info: Mapping[str, Any]) -> dict:
    params = _PROBLEM.vec_to_params(genotype)
    element = _PROBLEM.element(params)
    outline = element.path()
    xs = [float(vertex["x"]) for vertex in outline]
    ys = [float(vertex["y"]) for vertex in outline]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return {
        "optic_envelope_width": float(width),
        "optic_envelope_height": float(height),
        "optic_bbox_area": float(width * height),
        "thinness_T": float(info["thinness_T"]),
        "frontmost_used_x": float(_PROBLEM.x_sensor - info["thinness_T"]),
        "max_sag_ratio": float(element.max_sag_ratio()),
    }


def _validate_as_intended(genotype: np.ndarray, signature: str) -> dict | None:
    cfg = replace(
        _CFG,
        name="novel_intended_validation",
        fitness="intended_focused_throughput",
        intended_sequence=signature,
        min_intended_root_support=0.80,
        min_intended_delivered_energy_share=0.05,
        require_intended_dominant_all_fields=True,
        min_intended_critical_margin_deg=0.50,
    )
    return _evaluate(_RUNNER, _PROBLEM, cfg, genotype)


def _evaluate_task(task: tuple[int, Sequence[float]]) -> dict:
    index, values = task
    genotype = np.asarray(values, float)
    diagnostics: dict = {}
    result = _evaluate(_RUNNER, _PROBLEM, _CFG, genotype, diagnostics=diagnostics)
    base = {"proposal": int(index), "genotype": [float(v) for v in genotype]}
    if result is None:
        return {**base, "status": "rejected", "rejection": diagnostics}
    info = result["info"]
    package = _package_metrics(genotype, info)
    if _MODE == "target_refine":
        return {**base, "status": "target_scored", "fitness": float(result["fitness"]),
                "signature": info.get("dom_signature"), "info": info,
                "package": package}
    if _MODE == "compaction":
        if float(result["fitness"]) < _MIN_COMPACTION_FITNESS:
            return {**base, "status": "performance_floor", "fitness": result["fitness"],
                    "signature": info.get("dom_signature"), "package": package}
        return {**base, "status": "accepted", "fitness": float(result["fitness"]),
                "info": info, "package": package}

    signature = info["dom_signature"]
    if not signature:
        return {**base, "status": "no_dominant_path",
                "fitness": float(result["fitness"]), "package": package}
    novelty = _EXCLUSION.report(signature)
    if float(result["fitness"]) < _MIN_FITNESS:
        status = "fitness_floor"
    elif not novelty.passes:
        status = "prior_like"
    else:
        # The initial trace infers a sequence.  Re-evaluate with that route explicitly
        # requested so field switching, low bundle support, weak functional energy and
        # knife-edge TIR cannot masquerade as a viable new design.
        validated = _validate_as_intended(genotype, signature)
        if validated is None:
            status = "failed_intended_validation"
        else:
            vinfo = validated["info"]
            return {
                **base, "status": "new_viable", "fitness": float(validated["fitness"]),
                "discovery_fitness": float(result["fitness"]), "signature": signature,
                "novelty": novelty.to_dict(), "info": vinfo,
                "package": _package_metrics(genotype, vinfo),
            }
    return {**base, "status": status, "fitness": float(result["fitness"]),
            "signature": signature, "novelty": novelty.to_dict(), "package": package}


def _proposal_batch(rng: np.random.Generator, anchors: Sequence[np.ndarray],
                    lo: np.ndarray, hi: np.ndarray, start: int, count: int) -> list:
    scale = hi - lo
    tasks = []
    for offset in range(count):
        if rng.random() < 0.08:
            genotype = rng.uniform(lo, hi)
        else:
            anchor = anchors[int(rng.integers(len(anchors)))]
            # A log mixture searches topology-changing jumps and basin-preserving steps.
            sigma = math.exp(rng.uniform(math.log(0.012), math.log(0.18)))
            weights = np.ones(len(lo))
            if len(lo) == 17:
                # Preserve the densely audited first-body basin more often while making
                # the added body's pose and spherical powers explore aggressively.
                weights[:10] = 0.30
            genotype = np.clip(
                anchor + rng.normal(size=len(lo)) * sigma * scale * weights, lo, hi)
        tasks.append((start + offset, genotype.tolist()))
    return tasks


def _run_discovery(args, output: Path, exclusion: TopologyExclusionSet,
                   anchors: Sequence[np.ndarray], problem) -> tuple[list[dict], dict]:
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    rng = np.random.default_rng(args.seed)
    trace_path = output / "inception_trace.jsonl"
    counts: dict[str, int] = {}
    viable: list[dict] = []
    started = time.time()
    processed = 0
    initargs = (exclusion.signatures, exclusion.min_edits,
                exclusion.min_normalized_distance, args.min_fitness,
                "discovery", "", 0.0, args.design_space, args.fields_parsed)
    with trace_path.open("w") as trace, Pool(
            args.workers, initializer=_worker_init, initargs=initargs) as pool:
        while processed < args.evals:
            count = min(args.batch_size, args.evals - processed)
            tasks = _proposal_batch(rng, anchors, lo, hi, processed, count)
            for row in pool.imap_unordered(_evaluate_task, tasks, chunksize=1):
                trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                if row["status"] == "new_viable":
                    viable.append(row)
            trace.flush()
            processed += count
            print(
                f"[inception] {processed}/{args.evals} proposals; "
                f"strict={processed-counts.get('rejected', 0)} "
                f"new_viable={len(viable)} ({time.time()-started:.1f}s)", flush=True)
            checkpoint = {
                "schema_version": SCHEMA, "stage": "inception", "proposals": processed,
                "status_counts": counts, "new_viable": len(viable),
                "best_new_fitness": max((row["fitness"] for row in viable), default=None),
                "wall_seconds": time.time() - started,
            }
            _atomic_json(output / "inception_checkpoint.json", checkpoint)
            if args.stop_after_new > 0 and len(viable) >= args.stop_after_new:
                break
    viable.sort(key=lambda row: (-row["fitness"],
                                 -row["novelty"]["normalized_distance"]))
    return viable, {**checkpoint, "trace": str(trace_path.resolve())}


def _compaction_tasks(rng: np.random.Generator, incumbent: np.ndarray,
                      lo: np.ndarray, hi: np.ndarray, start: int, count: int,
                      iteration: int) -> list:
    scale = hi - lo
    sigma = max(0.004, 0.035 * (0.82 ** iteration))
    tasks = []
    for offset in range(count):
        step = rng.normal(size=len(lo)) * sigma * scale
        if len(lo) == 10:
            # ray_y, ang, cx, thick, fsag, bsag, dsag, tilt_a, tilt_b, n
            step[2] += abs(rng.normal()) * sigma * scale[2] * 0.75
            step[3] -= abs(rng.normal()) * sigma * scale[3] * 0.75
        else:
            # ray_y, ang, n, [cx1, thick1, shapes/tilts], [cx2, thick2, ...]
            # Bring the bodies together, move the front body detector-ward, and reduce
            # both body depths.  Shape/pose genes remain free to restore the exact path.
            step[3] += abs(rng.normal()) * sigma * scale[3] * 0.55
            step[10] -= abs(rng.normal()) * sigma * scale[10] * 0.75
            step[4] -= abs(rng.normal()) * sigma * scale[4] * 0.45
            step[11] -= abs(rng.normal()) * sigma * scale[11] * 0.45
        genotype = np.clip(incumbent + step, lo, hi)
        tasks.append((start + offset, genotype.tolist()))
    return tasks


def _compact(args, output: Path, candidate: dict,
             exclusion: TopologyExclusionSet, problem) -> tuple[dict, list[dict], dict]:
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    initial = np.asarray(candidate["genotype"], float)
    incumbent = initial.copy()
    baseline_package = candidate["package"]
    incumbent_package = baseline_package
    baseline_fitness = float(candidate["fitness"])
    floor = baseline_fitness * args.fitness_retention
    rng = np.random.default_rng(args.seed + 1000003)
    trace_path = output / "compaction_trace.jsonl"
    trajectory = [{"iteration": -1, "proposal": candidate["proposal"],
                   "fitness": baseline_fitness, "package": baseline_package,
                   "genotype": candidate["genotype"], "info": candidate["info"],
                   "operation": "inception seed"}]
    counts: dict[str, int] = {}
    started = time.time()
    proposal = 0

    def score(package: Mapping[str, float]) -> float:
        thin = package["thinness_T"] / baseline_package["thinness_T"]
        width = (package["optic_envelope_width"]
                 / baseline_package["optic_envelope_width"])
        return 0.65 * thin + 0.35 * width

    initargs = (exclusion.signatures, exclusion.min_edits,
                exclusion.min_normalized_distance, 0.0, "compaction",
                candidate["signature"], floor, args.design_space, args.fields_parsed)
    with trace_path.open("w") as trace, Pool(
            args.workers, initializer=_worker_init, initargs=initargs) as pool:
        for iteration in range(args.compaction_iterations):
            tasks = _compaction_tasks(rng, incumbent, lo, hi, proposal,
                                      args.compaction_batch, iteration)
            rows = list(pool.imap_unordered(_evaluate_task, tasks, chunksize=1))
            for row in rows:
                trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            trace.flush()
            accepted = [row for row in rows if row["status"] == "accepted"]
            # Do not trade one package dimension against a larger one.  Compaction must
            # Pareto-dominate the inception geometry in both registered size measures.
            accepted = [row for row in accepted
                        if row["package"]["thinness_T"] <= baseline_package["thinness_T"]
                        and row["package"]["optic_envelope_width"]
                        <= baseline_package["optic_envelope_width"]]
            best = min(accepted, key=lambda row: (score(row["package"]),
                                                  -row["fitness"]), default=None)
            if best is not None and score(best["package"]) + 1e-9 < score(incumbent_package):
                incumbent = np.asarray(best["genotype"], float)
                incumbent_package = best["package"]
                trajectory.append({
                    "iteration": iteration, "proposal": best["proposal"],
                    "fitness": best["fitness"], "package": best["package"],
                    "genotype": best["genotype"], "info": best["info"],
                    "operation": "translate/reshape spherical element; sequence fixed",
                })
            proposal += len(tasks)
            print(
                f"[compaction] iter={iteration+1}/{args.compaction_iterations} "
                f"accepted={counts.get('accepted', 0)} "
                f"T={incumbent_package['thinness_T']:.2f} "
                f"width={incumbent_package['optic_envelope_width']:.2f} "
                f"fit>={floor:.3f} ({time.time()-started:.1f}s)", flush=True)

    final_row = trajectory[-1]
    summary = {
        "stage": "compaction", "proposals": proposal, "status_counts": counts,
        "accepted_steps": len(trajectory) - 1, "fitness_floor": floor,
        "wall_seconds": time.time() - started, "trace": str(trace_path.resolve()),
    }
    return final_row, trajectory, summary


def _render_pair(output: Path, before: Mapping[str, Any], after: Mapping[str, Any],
                 problem, fields: Sequence[float]) -> dict:
    """Render -8/0/+8 degree panels with one common frame for honest comparison."""
    try:
        import io
        from dataclasses import replace as dc_replace
        from PIL import Image, ImageDraw, ImageFont
        from nslens.scene import CropBox
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    fields = tuple(fields)
    params = [problem.vec_to_params(row["genotype"]) for row in (before, after)]
    images = []
    files = {}
    with Runner(timeout_s=60) as runner:
        frames = []
        for p in params:
            xs, ys = [0.0, problem.x_sensor], []
            for field in fields:
                layout = problem.layout(p, field_deg=field)
                raw, _ = runner.simulate(layout.to_scene_dict())
                for record in raw.get("raySequences", []):
                    for vertex in record.get("seq", []):
                        xs.append(float(vertex["x"])); ys.append(float(vertex["y"]))
            if not ys:
                ys = [-problem.half_h, problem.half_h]
            frames.append(((min(xs) - 8, min(ys) - 8),
                           (max(xs) + 8, max(ys) + 8)))
        frame = ((min(f[0][0] for f in frames), min(f[0][1] for f in frames)),
                 (max(f[1][0] for f in frames), max(f[1][1] for f in frames)))
        for stage, p in zip(("inception", "compacted"), params):
            for field in fields:
                layout = problem.layout(p, field_deg=field)
                dense_source = dc_replace(layout.source, n_rays=61)
                crop = CropBox(p1=frame[0], p4=frame[1], width=1100)
                layout = dc_replace(layout, source=dense_source, record=False, crop=crop)
                raw, _ = runner.simulate(layout.to_scene_dict())
                rendered = raw.get("images") or []
                png = (base64.b64decode(rendered[0]["dataUrl"].split(",", 1)[1])
                       if rendered and rendered[0].get("dataUrl") else None)
                if png is None:
                    continue
                filename = output / f"{stage}_field_{field:+.0f}.png".replace("+", "p")
                filename.write_bytes(png)
                files[f"{stage}:{field:+g}"] = str(filename.resolve())
                images.append((stage, field, Image.open(io.BytesIO(png)).convert("RGB")))
    if not images:
        return {"status": "failed", "files": files}
    cell_w = min(480, max(image.width for _, _, image in images))
    ratio = cell_w / images[0][2].width
    cell_h = int(images[0][2].height * ratio)
    label_h, pad = 28, 8
    sheet = Image.new("RGB", (3 * cell_w + 4 * pad,
                              2 * (cell_h + label_h) + 3 * pad), "#090909")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for stage_i, stage in enumerate(("inception", "compacted")):
        for field_i, field in enumerate(fields):
            match = next((image for s, f, image in images if s == stage and f == field), None)
            if match is None:
                continue
            x = pad + field_i * (cell_w + pad)
            y = pad + stage_i * (cell_h + label_h + pad)
            sheet.paste(match.resize((cell_w, cell_h), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 3, y + cell_h + 5), f"{stage} | field {field:+g} deg",
                      fill="#eeeecc", font=font)
    comparison = output / "inception_vs_compacted.png"
    sheet.save(comparison)
    files["comparison"] = str(comparison.resolve())
    return {"status": "ok", "files": files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--design-space", choices=("two_element", "single_3curved"),
                        default="two_element")
    parser.add_argument("--fields", default="-2,0,2",
                        help="comma-separated inception field angles in degrees")
    parser.add_argument("--evals", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=9042026)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--min-edits", type=int, default=3)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--min-fitness", type=float, default=0.05)
    parser.add_argument("--stop-after-new", type=int, default=3,
                        help="0 searches the entire budget")
    parser.add_argument("--compaction-iterations", type=int, default=8)
    parser.add_argument("--compaction-batch", type=int, default=128)
    parser.add_argument("--fitness-retention", type=float, default=0.80)
    args = parser.parse_args(argv)
    args.fields_parsed = tuple(float(value) for value in args.fields.split(","))
    if 0.0 not in args.fields_parsed:
        parser.error("fields must include 0 degrees")
    if args.evals <= 0 or args.workers <= 0 or args.batch_size <= 0:
        parser.error("evals, workers and batch-size must be positive")

    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    started = time.time()

    signatures, reference_sources = _archive_signatures(REPO, output)
    exclusion = TopologyExclusionSet(signatures, min_edits=args.min_edits,
                                     min_normalized_distance=args.min_distance)
    problem, _ = _problem_and_cfg(args.design_space, args.fields_parsed)
    anchors, anchor_sources = _anchors(args.anchors, problem)
    reference = {
        "signature_strings": len(exclusion.signatures),
        "canonical_topologies": len(exclusion.canonical),
        "minimum_edit_distance": exclusion.min_edits,
        "minimum_normalized_distance": exclusion.min_normalized_distance,
        "archive_sources": reference_sources,
        "anchor_sources": anchor_sources,
        "spherical_parameterization": list(problem.param_names),
        "fields_deg": list(args.fields_parsed),
        "surface_shapes": ({
            "E1A/E1B/E1T": "spherical arc", "E1D": "plane",
            "E2A/E2B/E2T": "spherical arc", "E2D": "plane",
        } if args.design_space == "two_element" else {
            "A": "spherical arc", "B": "spherical arc",
            "T": "spherical arc", "D": "plane",
        }),
    }
    _atomic_json(output / "reference_snapshot.json", reference)
    print(f"[inventory] {len(exclusion.signatures)} signatures / "
          f"{len(exclusion.canonical)} canonical topologies; {len(anchors)} anchors",
          flush=True)

    viable, inception = _run_discovery(args, output, exclusion, anchors, problem)
    if not viable:
        manifest = {
            "schema_version": SCHEMA, "status": "no_new_viable_candidate",
            "reference": reference, "inception": inception,
            "configuration": vars(args), "wall_seconds": time.time() - started,
        }
        _atomic_json(output / "manifest.json", manifest)
        print("[result] no candidate cleared the declared novelty and viability gates")
        return 2

    _atomic_json(output / "new_viable_candidates.json", viable)
    selected = viable[0]
    compacted, trajectory, compaction = _compact(
        args, output, selected, exclusion, problem)
    _atomic_json(output / "compaction_trajectory.json", trajectory)
    visual = _render_pair(output, selected, compacted, problem, args.fields_parsed)

    initial_package, final_package = selected["package"], compacted["package"]
    reduction = {
        key: 1.0 - float(final_package[key]) / float(initial_package[key])
        for key in ("thinness_T", "optic_envelope_width", "optic_bbox_area")
    }
    manifest = {
        "schema_version": SCHEMA, "status": "complete",
        "reference": reference, "configuration": vars(args),
        "inception": inception,
        "selected_new_design": selected,
        "compacted_design": compacted,
        "compaction_reduction_fraction": reduction,
        "compaction": compaction, "trajectory": trajectory,
        "visualization": visual, "wall_seconds": time.time() - started,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(f"[result] {selected['signature']}")
    print(f"[result] novelty distance={selected['novelty']['edit_distance']} / "
          f"{selected['novelty']['normalized_distance']:.1%}")
    print(f"[result] compaction T {initial_package['thinness_T']:.2f} -> "
          f"{final_package['thinness_T']:.2f}; optic width "
          f"{initial_package['optic_envelope_width']:.2f} -> "
          f"{final_package['optic_envelope_width']:.2f}")
    print(f"[result] wrote {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
