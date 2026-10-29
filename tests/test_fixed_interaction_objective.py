"""Fixed-condition search must optimise the requested path, not an accidental one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.engine import Runner  # noqa: E402
from nslens.engine.signatures import parse_target  # noqa: E402
from nslens.qd.driver import _evaluate  # noqa: E402
from nslens.qd.experiments import (fixed_environment,  # noqa: E402
                                   load_fixed_environment_specs)
from nslens.qd.interaction_objective import (  # noqa: E402
    target_focused_throughput, target_path_evidence)
from nslens.qd.observation import measure  # noqa: E402


def _record(ray_id, parent_id, brightness, events, y):
    seq = [{"t": "source", "x": 0.0, "y": y, "face": "source"}]
    for index, (interaction, face) in enumerate(events, 1):
        seq.append({"t": interaction, "x": float(index * 10), "y": y,
                    "face": face})
    seq.append({"t": "sensor", "x": 100.0, "y": y, "face": "sensor"})
    return {
        "rayId": ray_id, "parentRayId": parent_id,
        "x": 100.0, "y": y, "dirx": 1.0, "diry": 0.0,
        "b": brightness, "seq": seq,
        "outcome": "delivered", "historyComplete": True, "aggregate": False,
    }


def check_branch_safe_evidence() -> bool:
    target = parse_target("T@A TIR@D T@B")
    intended = (("refract", "A"), ("tir", "D"), ("refract", "B"))
    wrong = (("refract", "A"), ("refract", "B"))
    records = [
        _record(0, None, 0.4, intended, 0.0),
        _record(2, 0, 0.3, intended, 0.2),  # descendant: energy counts, root does not
        _record(1, None, 0.2, intended, -0.2),
        _record(3, 0, 0.1, wrong, 5.0),
    ]
    result = {
        "raySequences": records,
        "rayTerminations": records,
        "energyLedger": {
            "emitted": 1.0, "delivered": 1.0, "escaped": 0.0,
            "absorbed": 0.0, "truncated": 0.0, "unfinished": 0.0,
        },
    }
    face_of = lambda vertex: vertex["face"]
    evidence = target_path_evidence(result, face_of, target, n_source_rays=2)
    obs = measure(result, face_of, x_sensor=100.0)
    ok = bool(
        evidence["root_support"] == 1.0
        and abs(evidence["delivered_energy_share"] - 0.9) < 1e-12
        and evidence["target_is_dominant"]
        and obs is not None
        and abs(target_focused_throughput(obs, target, focus_w=1.0) - 0.9) < 1e-12)
    print("  " + ("PASS" if ok else "FAIL")
          + " branch descendants cannot inflate intended-path root support")
    return ok


def _anchors() -> dict[str, dict]:
    path = REPO / "runs" / "codev_validation_cohort_20260903_v2" / "cohort.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["design_id"]: row for row in rows
            if row["role"] == "intentional-anchor"}


def check_fixed_configs() -> bool:
    rows = load_fixed_environment_specs()
    configs = [fixed_environment(row["environment_id"]) for row in rows]
    ok = bool(len(configs) == 8 and all(
        len(cfg.problem.param_names) == 6
        and cfg.fitness == "intended_focused_throughput"
        and cfg.require_intended_dominant_all_fields
        and cfg.min_intended_root_support == 0.8
        and cfg.min_intended_delivered_energy_share == 0.05
        and cfg.min_intended_critical_margin_deg == 0.5
        and len(cfg.fields) == 9
        and cfg.trace
        and np.all(np.asarray(cfg.initial_genotype) >= cfg.problem.lo)
        and np.all(np.asarray(cfg.initial_genotype) <= cfg.problem.hi)
        for cfg in configs))
    print("  " + ("PASS" if ok else "FAIL")
          + " all eight fixed environments expose the audited six-DOF contract")
    return ok


def check_real_anchor() -> bool:
    spec = load_fixed_environment_specs()[1]
    cfg = fixed_environment(spec["environment_id"])
    anchor = _anchors()[spec["seed_design_id"]]
    x = np.asarray([anchor["prescription"][name]
                    for name in cfg.problem.param_names], dtype=float)
    diagnostics = {}
    with Runner(timeout_s=30) as runner:
        result = _evaluate(runner, cfg.problem, cfg, x, diagnostics=diagnostics)
    ok = bool(
        result is not None
        and result["fitness"] > 0.8
        and result["info"]["intended_root_support_min"] >= 0.8
        and result["info"]["intended_energy_share_mean"] > 0.8
        and result["info"]["intended_dominant_all"]
        and result["info"]["intended_critical_margin_min"] >= 0.5
        and len(result["info"]["intended_per_field"]) == 9
        and diagnostics.get("engine_calls") == 9)
    print("  " + ("PASS" if ok else "FAIL")
          + " known intentional anchor passes every inner-loop field gate")
    return ok


def main() -> int:
    print("fixed interaction-aware objective")
    ok = all((check_branch_safe_evidence(), check_fixed_configs(), check_real_anchor()))
    print("ALL FIXED-OBJECTIVE CHECKS PASS" if ok else
          "FIXED-OBJECTIVE CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
