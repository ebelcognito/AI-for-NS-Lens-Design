"""Audit model-facing data boundaries and replay the recorded physical episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from nslens.analysis.run_repair_curriculum import digest, write_json
from .tools import Workbench, canonical_hash


def replay(episode):
    wb = Workbench(episode["task"])
    assert canonical_hash(wb.state()) == episode["initial_state_hash"]
    for event in episode["events"]:
        assert canonical_hash(wb.state()) == event["before_hash"]
        output = wb.call(event["tool"],event["arguments"])
        assert canonical_hash(output) == canonical_hash(event["output"])
        assert canonical_hash(wb.state()) == event["after_hash"]
    actual = wb.finalize()
    assert actual == episode["outcome"]
    assert wb.records == episode["evaluations"]
    return dict(task_id=episode["task"]["task_id"],strategy=episode["strategy"],passed=True)


def audit(root, workers=8):
    manifest = json.loads((root/"manifest.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    for name, hash_value in manifest["sources"].items():
        assert digest(root/"source_snapshot"/name) == hash_value
        assert digest(repo/name) == hash_value, f"source changed: {name}"
    split_families, episodes = {}, []
    for split in ("train","validation","test"):
        rows = [json.loads(line) for line in (root/f"{split}_episodes.jsonl").open()]
        split_families[split] = {e["task"]["family_id"] for e in rows}
        for e in rows:
            assert e["task"]["split"] == split == e["outcome"]["split"]
            assert e["provenance"] == "scripted_numerical_teacher"
            assert "witness" not in e["task"] and "corruption" not in e["task"]
            o=e["outcome"]
            assert o["candidate_calls"] <= 24 and o["verification_calls"] <= 2 and o["actions"] <= 16
            coarse = [r for r in e["evaluations"] if r["fidelity"]=="coarse"]
            assert len(coarse) == 1+o["candidate_calls"]
            assert sum(r["fidelity"]=="dense" for r in e["evaluations"]) == o["verification_calls"]
            assert sum(r["fidelity"]=="audit" for r in e["evaluations"]) == 1
            assert o["success"] == bool(o["submitted"] and o["final_audit"]["success"])
            assert o["nominal_ray_evaluations"] == sum(r["verdict"]["rays"] for r in e["evaluations"])
            # A transcript contains precisely the tool interactions that occurred,
            # never a fabricated reasoning explanation or the post-episode audit.
            assert len(e["messages"]) == 2+2*len(e["events"])
            assert json.loads(e["messages"][1]["content"]) == e["task"]
            for i,event in enumerate(e["events"]):
                call,result=e["messages"][2+2*i:4+2*i]
                function=call["tool_calls"][0]["function"]
                assert function["name"]==event["tool"]
                assert json.loads(function["arguments"])==event["arguments"]
                assert json.loads(result["content"])==event["output"]
        episodes.extend(rows)
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        assert not split_families[a]&split_families[b]
    indexed={(e["task"]["task_id"],e["strategy"]):e for e in episodes}
    sft_rows=[json.loads(line) for line in (root/"train_sft.jsonl").open()]
    for row in sft_rows:
        source=indexed[(row["metadata"]["task_id"],row["metadata"]["strategy"])]
        assert source["task"]["split"]=="train" and source["outcome"]["success"]
        assert row["messages"]==source["messages"]
    prefs=json.loads((root/"train_plan_preferences.json").read_text())
    for p in prefs:
        left=indexed[(p["task_id"],p["preferred_plan"])]
        right=indexed[(p["task_id"],p["rejected_plan"])]
        assert left["task"]["split"]==right["task"]["split"]=="train"
        assert left["initial_state_hash"]==right["initial_state_hash"]==p["initial_state_hash"]
        assert p["preferred_outcome"]==left["outcome"] and p["rejected_outcome"]==right["outcome"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        replays=list(pool.map(replay,episodes))
    report=dict(passed=True,episodes=len(episodes),full_episode_replays=len(replays),
                split_families={k:len(v) for k,v in split_families.items()},
                train_sft=len(sft_rows),train_plan_preferences=len(prefs),
                checks=["source/runtime hashes", "family isolation", "hidden-answer exclusion",
                        "actual tool transcripts", "all optimizer probes charged",
                        "independent-grid acceptance", "successful train-only SFT",
                        "same-start plan preferences", "full physical episode replay"],
                frozen_llm_evaluated=False)
    write_json(root/"audit.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    print(json.dumps(audit(a.root,a.workers),indent=2))


if __name__=="__main__":
    main()
