"""Create verified tool-use episodes, with scripted provenance and family splits.

These are numerical-teacher demonstrations, not LLM-generated reasoning traces.
Preference labels compare complete plans started from the identical public task.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from nslens.analysis.run_repair_curriculum import digest, make_specs, write_json
from nslens.learning.freeform_repair import ImagingSpec
from .tools import SCHEMA, TOOL_SCHEMAS, Workbench, canonical_hash

INSTRUCTIONS = """You are operating a bounded optical engineering workbench. The problem
is a 2-D transmitting element with separate entrance and exit freeform regions. Choose
surface freedoms and numerical tools to meet the fixed image requirements. Coefficients
are polynomial sag departures in mm: z=y/(0.4*radius_y), multiplied by (1-(y/radius_y)^2)^2.
The material, pupil, source fields and detector are fixed. You have 24 candidate traces,
one initial measurement, at most two dense verifications, and 16 tool actions. Inspect
diagnostics, enable selected terms, and spend the budget deliberately. Internal optimizer
probes count toward the same allowance. Never claim success from prose; submit using finish.
The final independent-grid audit is hidden until the episode ends. No known solution is
provided. Verify when useful; restore_best restores the best coarse design you visited.
"""


def certify(spec_dict):
    spec = ImagingSpec(**spec_dict)
    task = dict(task_id=spec.family_id+"/certification", family_id=spec.family_id,
                split=spec.split, spec=spec_dict, start=[0.]*6,
                objective="Correct image error at all fixed fields and pupil positions.")
    wb = Workbench(task, budget=72)
    wb.call("enable", dict(surface="both", powers=[2,3,4]))
    for _ in range(3):
        wb.call("optimize", dict(max_evaluations=min(24, 72-wb.used)))
    wb.call("restore_best", {})
    wb.call("finish", dict(summary="Numerical certification attempt."))
    result = wb.finalize()
    return dict(spec=spec_dict, certified=result["success"], witness=wb.x.tolist(),
                audit=result, evaluations=wb.records)


def tasks_for(family, seed):
    rng = np.random.default_rng(seed)
    spec = family["spec"]
    tasks = []
    for mode, description in enumerate((
            "Restore image quality across the pupil without changing material or detector.",
            "Correct asymmetric off-axis image errors while retaining primary-ray throughput.",
            "Correct pupil-dependent image errors while preserving the two-transmission path.")):
        x = np.asarray(family["witness"]).copy()
        # Perturb different low-/high-order combinations. Exact corruption and
        # witness remain private; tasks expose only ordinary geometry/specification.
        side = int(rng.integers(2))
        indices = [[0], [1], [0,2]][mode]
        for i in indices:
            x[3*side+i] += float(rng.choice([-1.,1.])*rng.uniform(.3,.8))
        x = np.clip(x, -2., 2.)
        tasks.append(dict(task_id=spec["family_id"]+f"/case{mode}", family_id=spec["family_id"],
                          split=spec["split"], spec=spec, start=x.tolist(), objective=description))
    return tasks


def scripted_episode(job):
    task, strategy = job
    wb = Workbench(task)
    initial = wb.state()
    messages = [dict(role="system", content=INSTRUCTIONS),
                dict(role="user", content=json.dumps(task, allow_nan=False))]

    def call(name, arguments):
        call_id = f"call_{len(wb.events):03d}"
        messages.append(dict(role="assistant", tool_calls=[dict(id=call_id, type="function",
                        function=dict(name=name, arguments=json.dumps(arguments))) ]))
        result = wb.call(name, arguments)
        messages.append(dict(role="tool", tool_call_id=call_id,
                             content=json.dumps(result, allow_nan=False)))
        return result

    call("inspect", {})
    if strategy == "all_terms":
        call("enable", dict(surface="both", powers=[2,3,4]))
        call("optimize", dict(max_evaluations=24))
    elif strategy in ("entrance_only", "exit_only"):
        call("enable", dict(surface=strategy.split("_")[0], powers=[2,3,4]))
        call("optimize", dict(max_evaluations=24))
    elif strategy == "staged_order":
        call("enable", dict(surface="both", powers=[2]))
        call("optimize", dict(max_evaluations=8))
        call("inspect", {})
        call("enable", dict(surface="both", powers=[2,3,4]))
        call("optimize", dict(max_evaluations=24-wb.used))
    else:
        raise ValueError("unknown scripted strategy")
    call("restore_best", {})
    verified = call("verify", {})
    call("finish", dict(summary="Scripted numerical plan completed; measured dense RMS "
                             f"{verified['verification']['rms']:.6g} mm."))
    outcome = wb.finalize()
    return dict(schema_version=SCHEMA, provenance="scripted_numerical_teacher", strategy=strategy,
                task=task, initial_state=initial, initial_state_hash=canonical_hash(initial),
                messages=messages, events=wb.events, evaluations=wb.records, outcome=outcome)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-families", type=int, default=6)
    parser.add_argument("--validation-families", type=int, default=2)
    parser.add_argument("--test-families", type=int, default=4)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("use a new output directory")
    args.out.mkdir(parents=True)
    specs = make_specs(args.seed, (args.train_families, args.validation_families, args.test_families))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        families = list(pool.map(certify, specs))
        write_json(args.out / "private_certification.json", families)
        tasks = [task for i,f in enumerate(families) if f["certified"]
                 for task in tasks_for(f, args.seed+i)]
        write_json(args.out / "tasks.json", tasks)
        if not tasks:
            raise RuntimeError("no certified families; inspect private_certification.json")
        strategies = ("all_terms", "entrance_only", "exit_only", "staged_order")
        episodes = list(pool.map(scripted_episode, [(t,s) for t in tasks for s in strategies]))
    for split in ("train", "validation", "test"):
        rows = [e for e in episodes if e["task"]["split"] == split]
        with (args.out / f"{split}_episodes.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
    # SFT targets come only from successful, submitted training episodes. Hidden
    # final audits and witnesses are never appended to the model conversation.
    sft = [dict(messages=e["messages"], tools=TOOL_SCHEMAS,
                metadata=dict(task_id=e["task"]["task_id"], family_id=e["task"]["family_id"],
                              provenance=e["provenance"], strategy=e["strategy"]))
           for e in episodes if e["task"]["split"] == "train" and e["outcome"]["success"]]
    with (args.out / "train_sft.jsonl").open("w") as stream:
        for row in sft:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    preferences = []
    for task in tasks:
        if task["split"] != "train":
            continue
        group = [e for e in episodes if e["task"]["task_id"] == task["task_id"]]
        assert len({e["initial_state_hash"] for e in group}) == 1
        def rank(e):
            o=e["outcome"]
            return (o["success"], o["final_audit"]["physical_valid"], -o["final_audit"]["rms"])
        ordered = sorted(group, key=rank, reverse=True)
        if rank(ordered[0]) != rank(ordered[-1]):
            preferences.append(dict(task_id=task["task_id"], family_id=task["family_id"],
                                    initial_state_hash=group[0]["initial_state_hash"],
                                    preferred_plan=ordered[0]["strategy"],
                                    rejected_plan=ordered[-1]["strategy"],
                                    preferred_outcome=ordered[0]["outcome"],
                                    rejected_outcome=ordered[-1]["outcome"],
                                    label_scope="complete_plan_under_equal_candidate_allowance"))
    write_json(args.out / "train_plan_preferences.json", preferences)
    source_root = Path(__file__).resolve().parents[1]
    sources = list(Path(__file__).parent.glob("*.py"))
    sources += [source_root / "learning/freeform_repair.py"]
    sources += list((source_root / "freeform_body").glob("*.py"))
    hashes = {}
    for source in sources:
        relative = source.relative_to(source_root.parent)
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,target)
        hashes[str(relative)] = digest(source)
    summary = dict(schema_version=SCHEMA, config={**vars(args), "out":str(args.out)},
                   sources=hashes, families=len(families), certified_families=sum(f["certified"] for f in families),
                   tasks=len(tasks), episodes=len(episodes), train_sft=len(sft), train_preferences=len(preferences),
                   baseline_status="scripted baselines complete; no frozen language model evaluated",
                   results={s:dict(episodes=sum(e["strategy"]==s for e in episodes),
                             success_rate=float(np.mean([e["outcome"]["success"] for e in episodes if e["strategy"]==s])),
                             mean_candidate_calls=float(np.mean([e["outcome"]["candidate_calls"] for e in episodes if e["strategy"]==s]))) for s in strategies})
    write_json(args.out / "manifest.json", summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
