"""Fast contract checks for fixed-baseline catalogue and strict admission."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.analysis.audit_fixed_baselines import (
    _collect_candidates, _sha256, _strict_rejection_reasons)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"  PASS {name}")


def eligible_record() -> dict:
    return {
        "audit_label": "intentional",
        "historical_dense_dominance_fraction": 1.0,
        "global_topology_field_persistence": 1.0,
        "global_topology_perturbation_persistence": 0.8,
        "historical_full_delivered_energy_share_mean": 0.05,
        "contacts": [{
            "index": 0, "minimum_root_support": 0.8,
            "perturbation_persistence": 0.8,
            "tir_critical_margin_min_deg": 0.5,
            "tir_grazing_margin_min_deg": 0.5,
        }],
    }


def main() -> None:
    print("fixed baseline audit")
    check("strict boundary values are admitted",
          _strict_rejection_reasons(eligible_record()) == [])
    weak = eligible_record()
    weak["historical_dense_dominance_fraction"] = 16 / 17
    check("one non-dominant dense field is rejected",
          "intended route is not dominant at every dense field"
          in _strict_rejection_reasons(weak))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_path = root / "archive.json"
        archive = {
            "meta": {"param_names": [
                "cx", "thick", "fsag", "bsag", "tilt_a", "tilt_b"]},
            "cells": [{"i0": 1, "i1": 2, "fitness": 0.7,
                       "genotype": [40, 20, 0.1, -0.1, 5, -5]}],
        }
        archive_path.write_text(json.dumps(archive))
        run = {
            "environment_id": "fixed-test", "algorithm": "cma", "seed": 0,
            "evals": 10, "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
        }
        manifest = {"runs": [run, {**run, "algorithm": "random"}]}
        specs = {"fixed-test": {
            "intended_sequence": "T@A -> TIR@B -> T@D"}}
        rows = _collect_candidates(manifest, specs)["fixed-test"]
        check("identical elites are deduplicated within an environment", len(rows) == 1)
        check("deduplication preserves both provenance sources",
              len(rows[0]["provenance"]["sources"]) == 2)

    print("ALL FIXED-BASELINE AUDIT CHECKS PASS")


if __name__ == "__main__":
    main()
