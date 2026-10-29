"""Matched, frozen-Qwen midpoint decisions. Development evidence, not training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil

import numpy as np

from nslens.workbench.interactive_cma_pilot import (
    MODEL, SYSTEM, Qwen, digest, finalize, initial_state, observation, read,
    run_batch, write,
)

SEEDS = [2026091501, 2026091502, 2026091503]
CONTROLS = {"fixed": 1.0, "half": 0.5, "double": 2.0}


def continuation(multiplier):
    return dict(operation="continue", sigma_multiplier=multiplier,
                rationale="Predeclared numerical control.")


def fixed_start(task):
    return dict(operation="restart", parameters=list(range(len(task["start"]))),
                scales=[0.025] * len(task["start"]), radius=0.5,
                rationale="Shared fixed all-parameter first batch.")


def evaluate_action(task, midpoint, action, seed):
    before = digest(midpoint)
    batch = run_batch(task, midpoint, action, seed + 1)
    assert digest(midpoint) == before, "An arm mutated the shared midpoint"
    return dict(action=action, midpoint_hash=before, batch=batch,
                outcome=finalize(task, batch["state"]))


def run_case(job):
    root, task, seed, qwen = job
    key = f'{task["task_id"]}_seed_{seed}'
    folder = root / "episodes" / key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    first_path = folder / "midpoint.json"
    if not first_path.exists():
        write(first_path, run_batch(task, initial_state(task), fixed_start(task), seed))
    first = read(first_path)
    obs = observation(task, first["state"])
    write(folder / "observation.json", obs)
    # Obtain the policy's single action before executing any counterfactuals.
    error = None
    try:
        action = qwen.decide(obs, 1, key)[0]
    except RuntimeError as exc:
        if not str(exc).startswith("model failed three valid-action attempts"):
            raise
        action, error = None, str(exc)
    actions = {name: continuation(value) for name, value in CONTROLS.items()}
    if action is not None:
        actions["qwen"] = action
    arms = {}
    for name, selected in actions.items():
        path = folder / f"{name}.json"
        if not path.exists():
            write(path, evaluate_action(task, first["state"], selected, seed))
        arm = read(path)
        assert arm["midpoint_hash"] == digest(first["state"])
        assert arm["action"] == selected
        arms[name] = dict(action=selected, outcome=arm["outcome"])
    if action is None:
        arms["qwen"] = dict(action=None, outcome=finalize(task, first["state"]),
                            status="model_failure", error=error)
    row = dict(key=key, task_id=task["task_id"], family_id=task["family_id"],
               seed=seed, midpoint_hash=digest(first["state"]), arms=arms)
    write(folder / "result.json", row)
    return row


def summarize(root, tasks):
    rows = [read(p) for p in sorted((root / "episodes").glob("*/result.json"))]
    expected = len(tasks) * len(SEEDS)
    totals = {name: sum(int(r["arms"][name]["outcome"]["success"]) for r in rows)
              for name in [*CONTROLS, "qwen"]}
    comparisons = {}
    for name in CONTROLS:
        wins = sum(r["arms"]["qwen"]["outcome"]["success"] and
                   not r["arms"][name]["outcome"]["success"] for r in rows)
        losses = sum(r["arms"][name]["outcome"]["success"] and
                     not r["arms"]["qwen"]["outcome"]["success"] for r in rows)
        families = sorted({r["family_id"] for r in rows})
        grouped = [[int(r["arms"]["qwen"]["outcome"]["success"]) -
                    int(r["arms"][name]["outcome"]["success"])
                    for r in rows if r["family_id"] == f] for f in families]
        # Resample entire families, retaining all their tasks and numerical seeds.
        rng = np.random.default_rng(1509)
        samples = []
        if grouped:
            for _ in range(4000):
                selected = [grouped[i] for i in rng.integers(len(grouped), size=len(grouped))]
                samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
        comparisons[name] = dict(qwen_only=wins, control_only=losses,
            success_difference=(wins-losses)/len(rows) if rows else None,
            family_bootstrap_95_percent=np.quantile(samples, [.025, .975]).tolist() if samples else None)
    pairs = []
    for row in rows:
        if row["arms"]["qwen"]["action"] is None:
            continue
        for name in CONTROLS:
            a, b = row["arms"]["qwen"], row["arms"][name]
            delta = a["outcome"]["reward"] - b["outcome"]["reward"]
            if abs(delta) < .05:
                continue
            chosen, rejected = (a, b) if delta > 0 else (b, a)
            pairs.append(dict(key=row["key"], family_id=row["family_id"], seed=row["seed"],
                observation_path=f'episodes/{row["key"]}/observation.json',
                midpoint_hash=row["midpoint_hash"], chosen=chosen["action"], rejected=rejected["action"],
                chosen_reward=chosen["outcome"]["reward"], rejected_reward=rejected["outcome"]["reward"],
                label_scope="One matched midpoint action; one numerical seed, not a general causal claim."))
    calls = [read(p) for p in (root / "model_calls").glob("*.json") if not p.stem.endswith("_attempts")]
    attempts = [a for call in calls for a in call["attempts"]]
    for p in (root / "model_calls").glob("*_attempts.json"):
        if not p.with_name(p.name.replace("_attempts.json", ".json")).exists():
            attempts.extend(read(p))
    # Count executed work once: shared first batch, four second batches, final audits.
    evaluations = rays = 0
    for row in rows:
        first = read(root / "episodes" / row["key"] / "midpoint.json")
        evaluations += 16
        rays += first["state"]["candidate_rays"]
        for arm in row["arms"].values():
            out = arm["outcome"]
            evaluations += out["candidate_evaluations"] - 16 + 3
            rays += out["source_rays"] - first["state"]["candidate_rays"]
    summary = dict(complete=len(rows) == expected, completed=len(rows), expected=expected,
        tasks=len({r["task_id"] for r in rows}), families=len({r["family_id"] for r in rows}),
        verified_successes=totals, comparisons=comparisons, preference_pairs=len(pairs),
        model_failures=sum(r["arms"]["qwen"]["action"] is None for r in rows),
        qwen_operations={op: sum(r["arms"]["qwen"]["action"] is not None and
            r["arms"]["qwen"]["action"]["operation"] == op for r in rows) for op in ["continue", "restart"]},
        actual_evaluations_including_audits=evaluations, actual_source_rays=rays,
        model_requests=len(attempts), rejected_responses=sum("error" in a for a in attempts),
        input_tokens=sum(a["response"].get("usage", {}).get("prompt_tokens", 0) for a in attempts),
        output_tokens=sum(a["response"].get("usage", {}).get("completion_tokens", 0) for a in attempts))
    write(root / "summary.json", summary)
    write(root / "decision_preferences.json", pairs)
    lines = ["# Frozen Qwen: matched CMA midpoint pilot", "",
        f'Completed: {len(rows)}/{expected}. Development tasks; no model training.', "",
        "| Controller | Verified repairs |", "|---|---:|"]
    lines += [f"| {name} | {count}/{len(rows)} |" for name, count in totals.items()]
    lines += ["", "## Paired comparisons", ""]
    for name, comp in comparisons.items():
        lines.append(f'- Versus {name}: {comp["qwen_only"]} Qwen-only wins, '
                     f'{comp["control_only"]} control-only wins; family bootstrap interval '
                     f'{comp["family_bootstrap_95_percent"]} (success-rate difference).')
    lines += ["", f'Single-action candidate preference pairs: {len(pairs)}.', "",
        "All arms share an identical first 16-candidate CMA state; each valid action gets 16 more candidates.",
        "Three numerical seeds per task; Qwen chooses one action separately for each midpoint.",
        "Continuation preserves RNG and covariance. Restart uses the recorded next seed.",
        "Half/double controls are reported separately; no post-hoc oracle selection is a policy result.",
        "The data reuse six development families. Seeds/sibling tasks are not independent families;",
        "bootstrap intervals are exploratory with so few families and do not establish held-out generalization.",
        "No context ablation was run: optical understanding or rationale correctness is not established.",
        "Final checks share one physics engine; verification is numerical, not independent physical validation.",
        "Failed model calls retain and audit the midpoint without silently applying a fallback.",
        "Preference rows share states and controls and are not independent training examples.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report", "replay"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/playpen/hisham26/azr_interactive_pilot_20260914"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.out
    if args.command == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        paths = [Path(__file__), Path("nslens/workbench/interactive_cma_pilot.py"),
                 Path("nslens/analysis/azr_repair_pilot.py"), Path("nslens/search/cma_me.py")]
        hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        tasks = read(args.source / "tasks.json")
        protocol = dict(model=MODEL, seeds=SEEDS, task_hash=digest(tasks), tasks=len(tasks),
            controls=CONTROLS, candidate_budget_per_arm=32, primary="Qwen versus unchanged continuation: audited success",
            scope="Frozen controller; existing development families; no training or direct edits.",
            model_prompt=SYSTEM, source_hashes=hashes)
        if (root / "protocol.json").exists():
            assert read(root / "protocol.json") == protocol, "Use a new output directory for changed protocol/source"
        write(root / "protocol.json", protocol)
        write(root / "tasks.json", tasks)
        for path in paths:
            target = root / "source_snapshot" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        print(f"Prepared {len(tasks)} tasks x {len(SEEDS)} seeds", flush=True)
        return
    tasks = read(root / "tasks.json")
    protocol = read(root / "protocol.json")
    assert digest(tasks) == protocol["task_hash"]
    if args.command in ["run", "replay"]:
        for path, expected in protocol["source_hashes"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"Changed source: {path}"
    if args.command == "run":
        qwen = Qwen(args.endpoint, root)
        jobs = [(root, task, seed, qwen) for task in tasks for seed in SEEDS
                if not (root / "episodes" / f'{task["task_id"]}_seed_{seed}' / "result.json").exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                summary = summarize(root, tasks)
                print(f'{summary["completed"]}/{summary["expected"]} {row["key"]}: '
                      f'{summary["verified_successes"]}', flush=True)
    elif args.command == "replay":
        folder = sorted((root / "episodes").glob("*"))[0]
        row = read(folder / "result.json")
        task = next(t for t in tasks if t["task_id"] == row["task_id"])
        original = read(folder / "midpoint.json")
        first = run_batch(task, initial_state(task), fixed_start(task), row["seed"])
        assert first["state"] == original["state"] and first["records"] == original["records"]
        for name in ["fixed", "qwen"]:
            saved = read(folder / f"{name}.json")
            replay = evaluate_action(task, first["state"], saved["action"], row["seed"])
            assert replay["batch"]["state"] == saved["batch"]["state"]
            assert replay["batch"]["records"] == saved["batch"]["records"]
            assert {k:v for k,v in replay["outcome"].items() if k != "audit_seconds"} == {
                k:v for k,v in saved["outcome"].items() if k != "audit_seconds"}
        write(root / "replay.json", dict(key=row["key"], exact=True, arms=["fixed", "qwen"]))
        print("Exact numerical replay passed", flush=True)
    else:
        print(summarize(root, tasks), flush=True)


if __name__ == "__main__":
    main()
