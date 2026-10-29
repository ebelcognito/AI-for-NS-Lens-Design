"""Consolidate passing discovery audits into one portable design-stock manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from nslens.discovery.run import write


def build(root: Path) -> dict:
    summaries = json.loads((root / "summary.json").read_text())
    all_stress_path = root / "stock_stress_all.json"
    verification_path = root / "replay_and_stress.json"
    if all_stress_path.exists():
        verification = json.loads(all_stress_path.read_text())
        stress_rows = verification.get("results", [])
    else:
        verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
        stress_rows = verification.get("stress_examples", [])
    stress = {}
    for row in stress_rows:
        result = row.get("result") or {}
        stress[(row["unit"], json.dumps(result.get("genes"), separators=(",", ":")))] = row

    designs = []
    for summary in summaries:
        archive_path = Path(summary["config"]["out"]) / "archive.json"
        archive = json.loads(archive_path.read_text())
        for audit in archive["audits"]:
            result = audit["result"]
            if not result["feasible"]:
                continue
            key = (summary["unit"], json.dumps(result["genes"], separators=(",", ":")))
            stress_row = stress.get(key)
            stress_result = stress_row.get("result", {}) if stress_row else {}
            designs.append({
                "environment": summary["brief_id"],
                "unit": summary["unit"],
                "seed": summary["seed"],
                "corners": summary["corners"],
                "route": result["route"],
                "quality": result["quality"],
                "support": result["support"],
                "source_clearance": result.get("source_clearance"),
                "detector_clearance": result.get("detector_clearance"),
                "stress_tested": stress_row is not None,
                "stress_verified": bool(stress_row and stress_row["feasible"]),
                "stress_quality": stress_result.get("quality"),
                "stress_support": stress_result.get("support"),
                "stress_route": stress_result.get("route"),
                "coarse_evaluation": audit["coarse_evaluation"],
                "genes": result["genes"],
                "codec": summary["config"]["codec"],
                "brief": summary["config"]["brief"],
                "source_archive": str(archive_path),
            })
    designs.sort(key=lambda row: (row["environment"], row["corners"], -row["quality"], row["seed"]))
    by_environment = Counter(row["environment"] for row in designs)
    routes = {(row["environment"], row["route"]) for row in designs}
    result = {
        "source_run": str(root.resolve()),
        "selection": "All stored dense-audit passes; no post-run dense discoveries included.",
        "designs": designs,
        "counts": {
            "designs": len(designs),
            "environments": len(by_environment),
            "environment_route_pairs": len(routes),
            "stress_tested": sum(row["stress_tested"] for row in designs),
            "stress_verified": sum(row["stress_verified"] for row in designs),
            "by_environment": dict(sorted(by_environment.items())),
        },
    }
    write(root / "stock_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = build(args.run)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
