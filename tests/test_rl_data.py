"""Versioned proposer and CODE V data contracts."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.rl.data.build_proposer import (certified_records, format_records,
                                           verify_certified_anchors)         # noqa: E402
from nslens.rl.data.schema import (example_codev_transition,
                                   validate_codev_transition,
                                   validate_hardened_action_sft,
                                   validate_hardened_design,
                                   validate_proposer,
                                   validate_search_group,
                                   validate_search_preference)      # noqa: E402
from nslens.rl.data.build_hardened_catalogue import yardstick_payload  # noqa: E402
from nslens.rl.data.build_search_actions import _group, _preferences  # noqa: E402
from nslens.rl.data.derive_codev import build as derive_codev           # noqa: E402
from nslens.rl.data.trace_sources import TraceSource                  # noqa: E402
from nslens.rl import env_config                                      # noqa: E402
from nslens.rl.tasks.certify import Certified                      # noqa: E402


def _cert(topology: int, suffix: int) -> Certified:
    # Known buildable anchor. Dataset tests cover serialization and leakage; optical
    # correctness is covered by test_rl_reward.py.
    x = [-2.6719776, 55.0, 35.0, 22.6610529, -0.1040211, 0.0826189,
         19.6025789, 2.8707127, 1.6246999]
    return Certified(
        env="fold", fov=4.0 + 2.0 * (suffix % 3),
        focus_w=4.0 + 2.0 * (suffix % 3),
        topology=topology, thinness_bin=3, achievable=0.75,
        difficulty=1 / 3, seeds_hit=2, seeds_run=3, from_seed=False,
        genotype=x, signature=f"T@A -> TIR@T -> TIR@B -> TIR@A -> T@B [{topology}]",
        crit_margin=0.9, spec_id=f"fold_t{topology:03d}_{suffix}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    ok = True
    rows = certified_records([_cert(t, i) for i, t in enumerate(
        (20, 20, 26, 26, 43, 43, 55, 55, 68, 68, 83, 83))], split_seed=4)
    all_rows = rows["directed"] + rows["unconditioned"]
    errors = [e for r in all_rows for e in validate_proposer(r)]
    ok &= check("certified proposer records validate", not errors,
                "; ".join(errors[:3]))

    directed = rows["directed"]
    train_topo = {r["labels"]["topology_index"] for r in directed
                  if r["split"] in ("train", "validation_seen")}
    test_topo = {r["labels"]["topology_index"] for r in directed
                 if r["split"] == "test_cell"}
    ok &= check("test_cell has no topology leakage", not (train_topo & test_topo),
                f"overlap={sorted(train_topo & test_topo)}")
    ok &= check("directed prompt names topology",
                all("requested_topology_index" in r["input"] for r in directed))
    ok &= check("unconditioned prompt hides topology",
                all("requested_topology_index" not in r["input"]
                    and "requested_tir_face_multiset" not in r["input"]
                    for r in rows["unconditioned"]))

    scaffold = format_records(30, seed=7)
    ok &= check("format scaffold validates and has no topology target",
                all(not validate_proposer(r)
                    and r["input"]["quality_or_topology_target"] is None
                    for r in scaffold))

    good_cert = _cert(20, 0)
    detector_bad = replace(
        good_cert, spec_id="detector_bad",
        genotype=[0, 0, 85, 45, 0, 0, -25, 25, 1.7])
    verified, anchor_verdicts, anchor_audit = verify_certified_anchors(
        [good_cert, detector_bad], workers=1)
    ok &= check("serialized certified anchors are re-traced through every hard gate",
                len(verified) == 1 and good_cert.spec_id in anchor_verdicts
                and anchor_audit["dropped"] == 1,
                "detector-intersecting synthetic anchor must be dropped")

    transition = example_codev_transition()
    ok &= check("CODE V transition example validates",
                not validate_codev_transition(transition))
    broken = copy.deepcopy(transition)
    broken["result"]["delta_merit"] = -99.0
    ok &= check("CODE V validator catches inconsistent merit delta",
                any("delta_merit" in e for e in validate_codev_transition(broken)))

    env = env_config.get("fold")
    source = TraceSource(
        trace_path=Path("runs/example/seed0/trace.jsonl"),
        archive_path=Path("runs/example/seed0/archive.json"), run="example", seed=0,
        experiment_key="fold24_clean", experiment_label="fold24_wide_packing_nocaustic",
        parameter_names=tuple(env.param_names), lo=tuple(env.problem.lo),
        hi=tuple(env.problem.hi),
        meta={"fields": [-8, 0, 8], "focus_w": 8.0,
              "field_aggregate": "worst", "param_scaling": "normalized"})
    parent = [float((a + b) / 2) for a, b in zip(source.lo, source.hi)]
    trace_rows = []
    for i in range(10):
        x = parent.copy()
        x[0] += i * 0.05
        trace_rows.append({
            "eval": i + 1, "emitter": 0, "gen": 0, "sigma": 0.12,
            "parent": parent, "genotype": x,
            "status": "new" if i == 9 else "invalid",
            **({"fitness": 0.4, "bc": [10, 80],
                "info": {"dom_signature": "T@A -> TIR@T -> TIR@B -> T@B"}}
               if i == 9 else {}),
        })
    action_group = _group(source, trace_rows, "a" * 64, "train")
    ok &= check("same-parent action group validates",
                action_group is not None and not validate_search_group(action_group))
    prefs = _preferences(action_group)
    ok &= check("derived action preferences validate",
                len(prefs) == 1 and not validate_search_preference(prefs[0]))
    action_group["candidates"][0]["normalized_delta"][0] = 99.0
    ok &= check("action validator catches inconsistent normalized move",
                any("normalized_delta" in e for e in validate_search_group(action_group)))

    bounds = env.bounds()
    design = {n: float((a + b) / 2) for n, (a, b) in bounds.items()}
    hardened = {
        "schema_version": "nslens.hardened_design.v1", "design_id": "hard:test",
        "split": "train", "design": design, "parameter_bounds": bounds,
        "yardstick": yardstick_payload(),
        "hardened_verdict": {
            "stage": "scored", "fitness": 0.3, "topology_index": 10,
            "full_signature": "T@A -> TIR@T -> TIR@B -> T@B"},
    }
    ok &= check("hardened quality record validates", not validate_hardened_design(hardened))

    strict_action = {
        "schema_version": "nslens.hardened_action_sft.v1",
        "example_id": "strict:test", "state_id": "state:test",
        "source_group_id": "group:test", "split": "train", "sample_weight": 1.0,
        "messages": [
            {"role": "system", "content": "Return an action."},
            {"role": "user", "content": "{}"},
            {"role": "assistant",
             "content": '<action>{"normalized_delta":[0,0,0,0,0,0,0,0,0]}</action>'},
        ],
        "labels": {"parent_hard_pass": True, "candidate_hard_pass": True,
                   "topology_preserved": True, "parent_fitness": 0.4,
                   "candidate_fitness": 0.5},
    }
    ok &= check("strict refinement SFT requires hard pass and improvement",
                not validate_hardened_action_sft(strict_action))
    strict_action["labels"]["topology_preserved"] = False
    ok &= check("strict refinement validator catches topology change",
                any("topology_preserved" in e
                    for e in validate_hardened_action_sft(strict_action)))

    second = copy.deepcopy(transition)
    second["transition_id"] = "design_042_run03_step0017_try0004"
    second["candidate_index"] = 4
    second["result"].update(merit_after=1.80, delta_merit=-0.057,
                            accepted=False)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "transitions.jsonl"
        raw.write_text("\n".join(json.dumps(r) for r in (transition, second)) + "\n")
        derived = derive_codev([raw], Path(td) / "derived")
    ok &= check("CODE V derivation creates SFT, preference, and surrogate views",
                sum(derived["counts"]["sft"].values()) == 1
                and sum(derived["counts"]["preferences"].values()) == 1
                and sum(derived["counts"]["surrogate"].values()) == 2)

    print("ALL RL DATA CHECKS PASS" if ok else "RL DATA CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
