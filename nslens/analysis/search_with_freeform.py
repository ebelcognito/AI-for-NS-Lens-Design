"""Fresh donor-seeded search with freeform refinement inside the incumbent loop."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nslens.analysis.mirror_residual_experiment import save
from nslens.analysis import mirror_residual_moves as moves
from nslens.analysis.hard_optical_environments import decorate, refine_budget, random_direction
from nslens.analysis.optical_refinement import audit, refine_design
from nslens.analysis.variable_mirror import trace


SOURCE = Path("experiments/mirror_count_geometry_20261005")
TASKS = {
    "field8": dict(focal=75.0, aperture=35.0, half_field_deg=8.0),
    "aperture42": dict(focal=75.0, aperture=42.0, half_field_deg=6.0),
}
ARMS = ("reuse_only", "optical_residual", "direction_random")
DONORS = tuple(range(8))
REPEAT = 4
ROUNDS = 8
DISCOVERY_BUDGET = 257
REFINEMENT_BUDGET = 261


def run(job):
    started = time.monotonic()
    task_name = job["task"]
    task = TASKS[task_name]
    arm = job["arm"]
    donor = int(job["seed"])
    repeat = int(job["repeat"])
    seed_offset = repeat * 10_000_000
    source_path = SOURCE / f"cross_count_{3100 + donor}.json"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if job.get("source_hash") not in (None, source_hash):
        raise ValueError(f"donor source changed after protocol creation: {source_path}")
    source = json.loads(source_path.read_text())
    pool = {}
    history = []
    mutations = []
    discovery_calls = 0
    refinement_calls = 0

    def evaluate(genes, n):
        nonlocal discovery_calls
        row = decorate(trace(genes, n, **task), genes, n, task, None)
        discovery_calls += 1
        return row

    for round_index in range(ROUNDS):
        n = 3 + round_index % 4
        rng = np.random.default_rng(810_000 + seed_offset + donor * 100 + round_index)
        round_start = discovery_calls
        if n in pool:
            # Pool rows carry freeform coefficients, but this discovery stage
            # deliberately restarts from their parabolic genes with coefficients
            # reset to zero by the parabolic tracer.
            x = np.asarray(pool[n]["genes"], dtype=float).copy()
        else:
            x = np.asarray(source["coarse"][str(n)]["genes"], dtype=float).copy()

        parent = evaluate(x, n)
        discovery_endpoint = parent
        if arm in ("optical_residual", "direction_random"):
            parity = parent["parity"]

            def observe(genes):
                nonlocal discovery_calls
                observation = moves.observe(genes, n, task, parity)
                discovery_calls += 1
                observation["raw"] = decorate(
                    observation["raw"], genes, n, task, None)
                scale = max(task["focal"] * np.tan(np.deg2rad(task["half_field_deg"])), 1.0)
                extra = 100 * np.maximum(
                    np.asarray(observation["raw"]["envelope_mm"]) / 1e12 - 1, 0) / scale
                observation["residual"] = np.r_[observation["residual"], extra]
                observation["mask"] = np.r_[observation["mask"], [True, True]]
                return observation

            observation = observe(x)
            jacobian, valid = moves.jacobian(x, observation, observe)
            # Preserve the old common-valid-ray rule for the local model.
            mask = np.all(valid, axis=1)
            jacobian[~mask] = 0.0
            residual = observation["residual"].copy()
            residual[~mask] = 0.0
            best = parent
            for group in (np.arange(2 * n, 3 * n), np.arange(3 * n)):
                for radius in (0.03, 0.1):
                    step = moves.propose(x, residual, jacobian, group, radius)
                    if arm == "direction_random":
                        step = random_direction(x, group, np.linalg.norm(step), rng)
                    candidate = evaluate(x + step, n)
                    mutations.append(dict(
                        round=round_index, n=n,
                        group="curves" if len(group) == n else "all",
                        radius=radius, step_norm=float(np.linalg.norm(step)),
                        parent_loss=parent["loss"], child_loss=candidate["loss"]))
                    if candidate["loss"] < best["loss"]:
                        best = candidate
            discovery_endpoint = best
            x = np.asarray(best["genes"], dtype=float)
        else:
            best = parent

        remaining = DISCOVERY_BUDGET - (discovery_calls - round_start)
        if remaining < 0:
            raise AssertionError("discovery probes exceeded their round budget")
        best = refine_budget(
            lambda genes: evaluate(genes, n), x, best, remaining,
            820_000 + seed_offset + donor * 100 + round_index,
            0.03)
        used = discovery_calls - round_start
        if used != DISCOVERY_BUDGET:
            raise AssertionError(f"round used {used} discovery calls, expected {DISCOVERY_BUDGET}")

        # Refine before the parabolic endpoint can enter or replace the pool.
        refined = refine_design(
            best["genes"], n, task,
            seed=130_000_000 + donor * 100 + round_index,
            budget=REFINEMENT_BUDGET)
        refined_best = refined["coarse"]
        refinement_calls += refined["coarse_calls"]
        if n not in pool or refined_best["loss"] < pool[n]["loss"]:
            pool[n] = refined_best

        history.append(dict(
            round=round_index, n=n,
            discovery_endpoint_loss=discovery_endpoint["loss"],
            discovery_best_loss=best["loss"],
            refined_loss=refined_best["loss"],
            incumbent_loss=pool[n]["loss"],
            discovery_calls=used,
            refinement_calls=refined["coarse_calls"],
            compatibility=refined["compatibility"]))

    chosen = min(pool, key=lambda count: pool[count]["loss"])
    selected = pool[chosen]
    final = audit(selected["genes"], chosen,
                  selected["freeform_coefficients_mm"], task)
    final["success"] = final["threshold_reached"]
    expected_discovery = ROUNDS * DISCOVERY_BUDGET
    expected_refinement = ROUNDS * REFINEMENT_BUDGET
    if discovery_calls != expected_discovery or refinement_calls != expected_refinement:
        raise AssertionError("run-level budget accounting mismatch")
    result = dict(
        job=job, task=task, source=str(source_path), source_hash=source_hash,
        source_coarse_calls=source["coarse_calls"],
        coarse=pool, selected_n=chosen, selected_coarse=selected,
        audit=final, history=history, mutations=mutations,
        discovery_calls=discovery_calls, refinement_calls=refinement_calls,
        coarse_calls=discovery_calls + refinement_calls, dense_calls=1,
        ray_evaluations=discovery_calls * 27 + refinement_calls * 105,
        seconds=time.monotonic() - started)
    suffix = f"{task_name}__{arm}__{donor}__r{repeat}.json"
    save(Path(job["out"]) / suffix, result)
    return dict(task=task_name, arm=arm, donor=donor, repeat=repeat,
                selected_n=chosen, success=final["success"],
                coarse_calls=result["coarse_calls"], seconds=result["seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(SOURCE / f"cross_count_{3100 + donor}.json"): hashlib.sha256(
            (SOURCE / f"cross_count_{3100 + donor}.json").read_bytes()).hexdigest()
        for donor in DONORS
    }
    jobs = [dict(task=task, arm=arm, seed=donor, repeat=REPEAT, out=str(out),
                 source_hash=source_hashes[str(SOURCE / f"cross_count_{3100 + donor}.json")])
            for task in TASKS for donor in DONORS for arm in ARMS]
    if args.smoke:
        jobs = [dict(task="field8", arm=arm, seed=1, repeat=REPEAT, out=str(out),
                     source_hash=source_hashes[str(SOURCE / "cross_count_3101.json")])
                for arm in ARMS]
    source_cost = sum(json.loads((SOURCE / f"cross_count_{3100 + donor}.json").read_text())[
                      "coarse_calls"] for donor in DONORS)
    save(out / "protocol.json", dict(
        jobs=jobs, tasks=TASKS, arms=ARMS, donors=DONORS, repeat=REPEAT,
        source=str(SOURCE), source_sha256=source_hashes,
        source_cost_per_donor={str(d): json.loads(
            (SOURCE / f"cross_count_{3100 + d}.json").read_text())["coarse_calls"]
            for d in DONORS}, source_cost_sum=source_cost,
        scope="Fresh optimizer trajectories with repeat index4, initialized from eight cached cross-count donor pools; conditional on these donated starting designs.",
        seeding="Old hard-optical formulas with repeat fixed at4 (+40,000,000) for discovery random directions and CMA. Refinement seed is 130,000,000+donor*100+round, shared across tasks and arms.",
        discovery="Eight rounds, count schedule3,4,5,6 twice. Each round spends exactly257 parabolic 3x9 (27-ray) traces including initial, residual observation/Jacobian probes, four proposals, and CMA. Residual arms use the existing common-valid-ray Jacobian and direction_random preserves the proposed step norm while sampling a bounded inward Gaussian direction.",
        refinement="After each round's parabolic CMA endpoint, run shared last_freeform refinement for261 5x21 (105-ray) traces before updating the incumbent pool. Compare only refined 5x21 losses against the same-count refined pool. Future discovery reads only pool genes and resets added coefficients to zero before refitting; the parabolic discovery model is not freeform-aware.",
        budget="2056 discovery +2088 refinement =4144 coarse calls per run; 2056*27+2088*105 ray evaluations; one selected-design 15x101 dense audit. No package cap on these tasks.",
        quality="Dense audit reports threshold_reached as success: worst-field mapping RMS<=1mm, >=99% support, and no sampled surface crossing. Audit is only performed after count selection and does not feed back.",
        limits="Ideal 2D mirrors and analytic polynomial last surface; no package envelope, 3D, fabrication, tolerance, or diffraction guarantee."))
    for path in [Path(__file__), Path("nslens/analysis/hard_optical_environments.py"),
                 Path("nslens/analysis/mirror_residual_moves.py"),
                 Path("nslens/analysis/variable_mirror.py"),
                 Path("nslens/analysis/optical_refinement.py"),
                 Path("nslens/analysis/final_freeform.py"),
                 Path("nslens/search/cma_me.py")]:
        shutil.copy(path, out / path.name)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(run, jobs):
            print(json.dumps(row), flush=True)
    save(out / "completion.json", dict(
        runs=len(jobs), discovery_calls=len(jobs) * 2056,
        refinement_calls=len(jobs) * 2088, coarse_calls=len(jobs) * 4144,
        dense_calls=len(jobs), source_cost_sum=source_cost))


if __name__ == "__main__":
    main()
